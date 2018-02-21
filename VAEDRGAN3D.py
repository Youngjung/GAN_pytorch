import utils, torch, time, os, pickle, imageio, math
from scipy.misc import imsave
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import pdb

class Encoder( nn.Module ):
	def __init__( self, Nfx, Nid, Npcode ):
		super(Encoder, self).__init__()
		self.input_height = 100
		self.input_width = 100
		self.input_dim = 3
		self.Nfx = Nfx
		self.Nid = Nid
		self.Npcode = Npcode

		self.conv = nn.Sequential(
			nn.Conv2d(self.input_dim, 64, 11, 4, 1,bias=True),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2),
			nn.Conv2d(64, 128, 5, 2, 1,bias=True),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2),
			nn.Conv2d(128, 256, 5, 2, 1,bias=True),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2),
			nn.Conv2d(256, 512, 5, 2, 1,bias=True),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2),
			nn.Conv2d(512, Nfx, 8 , 1, 1, bias=True),
			nn.Sigmoid(),
		)

		self.fc = nn.Linear( Nfx, Nfx*2 )
		self.softplus = nn.Softplus()

		utils.initialize_weights(self)

	def forward(self, input):
		x = self.conv( input )
		x = x.view(-1,self.Nfx)

		gaussian_params = self.fc(x)

		mean = gaussian_params[:,:self.Nfx]

		# softplus -> stddev should be always positive
		# epsilon for numerical stability
		stddev = self.softplus(gaussian_params[:,self.Nfx:]) + 1e-6
		return mean, stddev

class Decoder( nn.Module ):
	def __init__(self, Nfx, Npcode, Nz, nOutputCh=4):
		super(Decoder, self).__init__()
		self.nOutputCh = nOutputCh

		self.fc = nn.Sequential(
			nn.Linear( Nfx+Npcode+Nz, Nfx )
		)

		self.fconv = nn.Sequential(
			nn.ConvTranspose3d(Nfx, 512, 4, bias=False),
			nn.BatchNorm3d(512),
			nn.ReLU(),
			nn.ConvTranspose3d(512, 256, 4, 2, 1, bias=False),
			nn.BatchNorm3d(256),
			nn.ReLU(),
			nn.ConvTranspose3d(256, 128, 4, 2, 1, bias=False),
			nn.BatchNorm3d(128),
			nn.ReLU(),
			nn.ConvTranspose3d(128, 64, 4, 2, 1, bias=False),
			nn.BatchNorm3d(64),
			nn.ReLU(),
			nn.ConvTranspose3d(64, 32, 4, 2, 1, bias=False),
			nn.BatchNorm3d(32),
			nn.ReLU(),
			nn.ConvTranspose3d(32, nOutputCh, 4, 2, 1, bias=False),
			nn.Sigmoid(),
		)
	def forward(self, fx, y_pcode_onehot, z):
		feature = torch.cat((fx, y_pcode_onehot, z),1)
		x = self.fc( feature )
		x = self.fconv( x.unsqueeze(2).unsqueeze(3).unsqueeze(4) )
		return x


class generator(nn.Module):
	def __init__(self, Nfx, Nid, Npcode, Nz):
		super(generator, self).__init__()

		self.Nfx = Nfx

		self.Genc = Encoder(Nfx, Nid, Npcode)
		self.Gdec = Decoder(Nfx, Npcode, Nz)

		utils.initialize_weights(self)

	def forward(self, x_, y_pcode_onehot_, z_, VAEz_):
		mu, sigma = self.Genc( x_ )
		fx = mu + VAEz_*sigma # reparam trick
		x_hat = self.Gdec(fx, y_pcode_onehot_, z_)

		return x_hat, mu, sigma

class discriminator(nn.Module):
	# Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
	# Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
	def __init__(self, Nid=105, Npcode=48, nInputCh=4, norm=nn.BatchNorm3d):
		super(discriminator, self).__init__()
		self.nInputCh = nInputCh

		self.conv = nn.Sequential(
			nn.Conv3d(nInputCh, 32, 4, 2, 1, bias=False),
			norm(32),
			nn.LeakyReLU(0.2),
			nn.Conv3d(32, 64, 4, 2, 1, bias=False),
			norm(64),
			nn.LeakyReLU(0.2),
			nn.Conv3d(64, 128, 4, 2, 1, bias=False),
			norm(128),
			nn.LeakyReLU(0.2),
			nn.Conv3d(128, 256, 4, 2, 1, bias=False),
			norm(256),
			nn.LeakyReLU(0.2),
			nn.Conv3d(256, 512, 4, 2, 1, bias=False),
			norm(512),
			nn.LeakyReLU(0.2)
		)

		self.convGAN = nn.Sequential(
			nn.Conv3d(512, 1, 4, bias=False),
			nn.Sigmoid()
		)

		self.convID = nn.Sequential(
			nn.Conv3d(512, Nid, 4, bias=False),
		)

		self.convPCode = nn.Sequential(
			nn.Conv3d(512, Npcode, 4, bias=False),
		)
		utils.initialize_weights(self)

	def forward(self, input):
		feature = self.conv(input)

		fGAN = self.convGAN( feature ).squeeze(4).squeeze(3).squeeze(2)
		fid = self.convID( feature ).squeeze(4).squeeze(3).squeeze(2)
		fcode = self.convPCode( feature ).squeeze(4).squeeze(3).squeeze(2)

		return fGAN, fid, fcode

class VAEDRGAN3D(object):
	def __init__(self, args):
		# parameters
		self.epoch = args.epoch
		self.sample_num = 49 
		self.batch_size = args.batch_size
		self.save_dir = args.save_dir
		self.result_dir = args.result_dir
		self.dataset = args.dataset
		self.dataroot_dir = args.dataroot_dir
		self.log_dir = args.log_dir
		self.gpu_mode = args.gpu_mode
		self.num_workers = args.num_workers
		self.model_name = args.gan_type
		self.centerBosphorus = args.centerBosphorus
		self.loss_option = args.loss_option
		if len(args.loss_option) > 0:
			self.model_name = self.model_name + '_' + args.loss_option
			self.loss_option = args.loss_option.split(',')
		if len(args.comment) > 0:
			self.model_name = self.model_name + '_' + args.comment
		self.lambda_ = 0.25
		self.n_critic = args.n_critic
		self.n_gen = args.n_gen
		self.c = 0.01 # for wgan
		if 'wass' in self.loss_option:
			self.n_critic = 5

		if self.dataset == 'MultiPie' or self.dataset == 'miniPie':
			self.Nd = 337 # 200
			self.Np = 9
			self.Ni = 20
			self.Nz = 50
		elif self.dataset == 'Bosphorus':
			self.Nz = 50
			self.Nfx = 320
		elif self.dataset == 'CASIA-WebFace':
			self.Nd = 10885 
			self.Np = 13
			self.Ni = 20
			self.Nz = 50

		# makedirs
		temp_save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)
		if not os.path.exists(temp_save_dir):
			os.makedirs(temp_save_dir)
		else:
			print('[warning] path exists: '+temp_save_dir)
		temp_result_dir = os.path.join(self.result_dir, self.dataset, self.model_name)
		if not os.path.exists(temp_result_dir):
			os.makedirs(temp_result_dir)
		else:
			print('[warning] path exists: '+temp_result_dir)

		# save args
		timestamp = time.strftime('%b_%d_%Y_%H;%M')
		with open(os.path.join(temp_save_dir, self.model_name + '_' + timestamp + '_args.pkl'), 'wb') as fhandle:
			pickle.dump(args, fhandle)


		# load dataset
		data_dir = os.path.join( self.dataroot_dir, self.dataset )
		if self.dataset == 'mnist':
			self.data_loader = DataLoader(datasets.MNIST(data_dir, train=True, download=True,
																		  transform=transforms.Compose(
																			  [transforms.ToTensor()])),
														   batch_size=self.batch_size, shuffle=True)
		elif self.dataset == 'fashion-mnist':
			self.data_loader = DataLoader(
				datasets.FashionMNIST(data_dir, train=True, download=True, transform=transforms.Compose(
					[transforms.ToTensor()])),
				batch_size=self.batch_size, shuffle=True)
		elif self.dataset == 'celebA':
			self.data_loader = utils.CustomDataLoader(data_dir, transform=transforms.Compose(
				[transforms.CenterCrop(160), transforms.Scale(64), transforms.ToTensor()]), batch_size=self.batch_size,
												 shuffle=True)
		elif self.dataset == 'MultiPie' or self.dataset == 'miniPie':
			self.data_loader = DataLoader( utils.MultiPie(data_dir,
					transform=transforms.Compose(
					[transforms.Scale(100), transforms.RandomCrop(96), transforms.ToTensor()])),
				batch_size=self.batch_size, shuffle=True) 
		elif self.dataset == 'CASIA-WebFace':
			self.data_loader = utils.CustomDataLoader(data_dir, transform=transforms.Compose(
				[transforms.Scale(100), transforms.RandomCrop(96), transforms.ToTensor()]), batch_size=self.batch_size,
												 shuffle=True)
		elif self.dataset == 'Bosphorus':
			self.data_loader = DataLoader( utils.Bosphorus(data_dir, use_image=True, skipCodes=['YR','PR','CR'],
											transform=transforms.ToTensor(),
											shape=128, image_shape=256, center=self.centerBosphorus),
											batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
			self.Nid = 105
			self.Npcode = len(self.data_loader.dataset.posecodemap)

		# fixed samples for reconstruction visualization
		path_sample = os.path.join( self.result_dir, self.dataset, self.model_name, 'fixed_sample' )
		if not os.path.exists( path_sample ):
			print( 'Generating fixed sample for visualization...' )
			os.makedirs( path_sample )
			nSamples = self.sample_num-self.Npcode
			nPcodes = self.Npcode
			sample_x2D_s = []
			sample_x3D_s = []
			for iB, (sample_x3D_,sample_y_,sample_x2D_) in enumerate(self.data_loader):
				sample_x2D_s.append( sample_x2D_ )
				sample_x3D_s.append( sample_x3D_ )
				if iB > nSamples // self.batch_size:
					break
			sample_x2D_s = torch.cat( sample_x2D_s )[:nSamples,:,:,:]
			sample_x3D_s = torch.cat( sample_x3D_s )[:nSamples,:,:,:]
			sample_x2D_s = torch.split( sample_x2D_s, 1 )
			sample_x3D_s = torch.split( sample_x3D_s, 1 )
			sample_x2D_s += (sample_x2D_s[0],)*nPcodes
			sample_x3D_s += (sample_x3D_s[0],)*nPcodes
	#		sample_x2D_s = [ [x]*nPcodes for x in sample_x2D_s ]
	#		sample_x3D_s = [ [x]*nPcodes for x in sample_x3D_s ]
	#		flatten = lambda l: [item for sublist in l for item in sublist]
			self.sample_x2D_ = torch.cat( sample_x2D_s )
			self.sample_x3D_ = torch.cat( sample_x3D_s )
	#		sample_x2D_s = [sample_x2D_s[0][0].unsqueeze(0)]*nSamples
			self.sample_pcode_ = torch.zeros( nSamples+nPcodes, self.Npcode )
			self.sample_pcode_[:nSamples,0]=1
			for iS in range( nPcodes ):
				ii = iS%self.Npcode
				self.sample_pcode_[iS+nSamples,ii] = 1
			self.sample_z_ = torch.rand( nSamples+nPcodes, self.Nz )
			reparamZ_ = torch.normal( torch.zeros(nSamples+nPcodes, self.Nfx), torch.ones(nSamples+nPcodes,self.Nfx) )
	
			nSpS = int(math.ceil( math.sqrt( nSamples+nPcodes ) )) # num samples per side
			fname = os.path.join( path_sample, 'sampleGT.png')
			utils.save_images(self.sample_x2D_[:nSpS*nSpS,:,:,:].numpy().transpose(0,2,3,1), [nSpS,nSpS],fname)
	
			fname = os.path.join( path_sample, 'sampleGT_2D.npy')
			self.sample_x2D_.numpy().dump( fname )
			fname = os.path.join( path_sample, 'sampleGT_3D.npy')
			self.sample_x3D_.numpy().dump( fname )
			fname = os.path.join( path_sample, 'sampleGT_z.npy')
			self.sample_z_.numpy().dump( fname )
			fname = os.path.join( path_sample, 'sampleGT_pcode.npy')
			self.sample_pcode_.numpy().dump( fname )
			fname = os.path.join( path_sample, 'sample_reparamZ.npy')
			reparamZ_.numpy().dump( fname )
		else:
			print( 'Loading fixed sample for visualization...' )
			fname = os.path.join( path_sample, 'sampleGT_2D.npy')
			with open( fname ) as fhandle:
				self.sample_x2D_ = torch.Tensor(pickle.load( fhandle ))
			fname = os.path.join( path_sample, 'sampleGT_3D.npy')
			with open( fname ) as fhandle:
				self.sample_x3D_ = torch.Tensor(pickle.load( fhandle ))
			fname = os.path.join( path_sample, 'sampleGT_z.npy')
			with open( fname ) as fhandle:
				self.sample_z_ = torch.Tensor( pickle.load( fhandle ))
			fname = os.path.join( path_sample, 'sampleGT_pcode.npy')
			with open( fname ) as fhandle:
				self.sample_pcode_ = torch.Tensor( pickle.load( fhandle ))
			fname = os.path.join( path_sample, 'sample_reparamZ.npy')
			with open( fname ) as fhandle:
				reparamZ_ = torch.Tensor( pickle.load( fhandle ))

		if self.gpu_mode:
			self.sample_x2D_ = Variable(self.sample_x2D_.cuda(), volatile=True)
			self.sample_z_ = Variable(self.sample_z_.cuda(), volatile=True)
			self.sample_pcode_ = Variable(self.sample_pcode_.cuda(), volatile=True)
			self.reparamZ_ = Variable(reparamZ_.cuda())
		else:
			self.sample_x2D_ = Variable(self.sample_x2D_, volatile=True)
			self.sample_z_ = Variable(self.sample_z_, volatile=True)
			self.sample_pcode_ = Variable(self.sample_pcode_, volatile=True)
			self.reparamZ_ = Variable(reparamZ_)

		# networks init
		self.Genc = Encoder(self.Nfx, self.Nid, self.Npcode)
		self.Gdec = Decoder(self.Nfx, self.Npcode, self.Nz)
		self.D = discriminator(self.Nid, self.Npcode)
		self.Genc_optimizer = optim.Adam(self.Genc.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
		self.Gdec_optimizer = optim.Adam(self.Gdec.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
		self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

		if self.gpu_mode:
			self.Genc.cuda()
			self.Gdec.cuda()
			self.D.cuda()
			self.CE_loss = nn.CrossEntropyLoss().cuda()
			self.BCE_loss = nn.BCELoss().cuda()
			self.MSE_loss = nn.MSELoss().cuda()
			self.L1_loss = nn.L1Loss().cuda()
		else:
			self.CE_loss = nn.CrossEntropyLoss()
			self.BCE_loss = nn.BCELoss()
			self.MSE_loss = nn.MSELoss()
			self.L1_loss = nn.L1Loss()

#		print('---------- Networks architecture -------------')
#		utils.print_network(self.G)
#		utils.print_network(self.D)
#		print('-----------------------------------------------')


	def train(self):
		train_hist_keys = ['D_loss',
                           'D_loss_GAN_real',
                           'D_loss_id',
                           'D_loss_pcode',
                           'D_loss_GAN_fake',
                           'D_acc',
                           'Genc_loss',
                           'Gdec_loss',
                           'Gdec_loss_GAN_fake',
                           'Gdec_loss_id',
                           'Gdec_loss_pcode',
                           'Gdec_loss_recon',
                           'Gdec_loss_dist',
                           'per_epoch_time',
                           'total_time']

		if not hasattr(self, 'epoch_start'):
			self.epoch_start = 0
		if not hasattr(self, 'train_hist') :
			self.train_hist = {}
			for key in train_hist_keys:
				self.train_hist[key] = []
		else:
			existing_keys = self.train_hist.keys()
			num_hist = [len(self.train_hist[key]) for key in existing_keys]
			num_hist = max(num_hist)
			for key in train_hist_keys:
				if key not in existing_keys:
					self.train_hist[key] = [0]*num_hist
					print('new key added: {}'.format(key))

		if self.gpu_mode:
			self.y_real_ = Variable((torch.ones(self.batch_size,1)).cuda())
			self.y_fake_ = Variable((torch.zeros(self.batch_size,1)).cuda())
		else:
			self.y_real_ = Variable((torch.ones(self.batch_size,1)))
			self.y_fake_ = Variable((torch.zeros(self.batch_size,1)))

		nPairs = self.batch_size*(self.batch_size-1)
		normalizerA = self.data_loader.dataset.muA/self.data_loader.dataset.stddevA # normalization
		normalizerB = self.data_loader.dataset.muB/self.data_loader.dataset.stddevB # normalization
		eps = 1e-16

		self.D.train()
		start_time = time.time()
		print('training start from epoch {}!!'.format(self.epoch_start+1))
		for epoch in range(self.epoch_start, self.epoch):
			self.Genc.train()
			self.Gdec.train()
			epoch_start_time = time.time()
			start_time_epoch = time.time()

			for iB, (x3D_, y_, x2D_ ) in enumerate(self.data_loader):
				if iB == self.data_loader.dataset.__len__() // self.batch_size:
					break

				z_ = torch.rand((self.batch_size, self.Nz))
				y_random_pcode_ = torch.floor(torch.rand(self.batch_size)*self.Npcode).long()
				y_random_pcode_onehot_ = torch.zeros( self.batch_size, self.Npcode )
				y_random_pcode_onehot_.scatter_(1, y_random_pcode_.view(-1,1), 1)
				y_id_ = y_['id']
				y_pcode_ = y_['pcode']
				y_pcode_onehot_ = torch.zeros( self.batch_size, self.Npcode )
				y_pcode_onehot_.scatter_(1, y_pcode_.view(-1,1), 1)

				if self.gpu_mode:
					x2D_, z_ = Variable(x2D_.cuda()), Variable(z_.cuda())
					x3D_ = Variable(x3D_.cuda())
					y_id_ = Variable( y_id_.cuda() )
					y_pcode_ = Variable(y_pcode_.cuda())
					y_pcode_onehot_ = Variable( y_pcode_onehot_.cuda() )
					y_random_pcode_ = Variable(y_random_pcode_.cuda())
					y_random_pcode_onehot_ = Variable( y_random_pcode_onehot_.cuda() )
				else:
					x2D_, z_ = Variable(x2D_), Variable(z_)
					x3D_ = Variable(x3D_)
					y_id_ = Variable(y_id_)
					y_pcode_ = Variable(y_pcode_)
					y_pcode_onehot_ = Variable( y_pcode_onehot_ )
					y_random_pcode_ = Variable(y_random_pcode_)
					y_random_pcode_onehot_ = Variable( y_random_pcode_onehot_ )

				# update D network
				for iD in range(self.n_critic) :
					self.D_optimizer.zero_grad()
	
					D_GAN_real, D_id, D_pcode = self.D(x3D_)
					if 'wass' in self.loss_option:
						D_loss_GANreal = -torch.mean(D_GAN_real)
					else:
						D_loss_GANreal = self.BCE_loss(D_GAN_real, self.y_real_)
					D_loss_real_id = self.CE_loss(D_id, y_id_)
					D_loss_real_pcode = self.CE_loss(D_pcode, y_pcode_)
	
					reparamZ_ = torch.normal( torch.zeros(self.batch_size, self.Nfx), torch.ones(self.batch_size,self.Nfx) )
					if self.gpu_mode:
					    reparamZ_ = Variable(reparamZ_.cuda())
					else:
					    reparamZ_ = Variable(reparamZ_)

					mu, sigma = self.Genc(x2D_)
	
					fx = mu + reparamZ_*sigma # reparam trick
	
					x3D_hat = self.Gdec(fx, y_random_pcode_onehot_, z_)
	
					D_GAN_fake, _, _ = self.D(x3D_hat)
					if 'wass' in self.loss_option:
						D_loss_GANfake = torch.mean(D_GAN_fake)
					else:
						D_loss_GANfake = self.BCE_loss(D_GAN_fake, self.y_fake_)
	
					num_correct_real = torch.sum(D_GAN_real>0.5)
					num_correct_fake = torch.sum(D_GAN_fake<0.5)
					D_acc = float(num_correct_real.data[0] + num_correct_fake.data[0]) / (self.batch_size*2)
	
					if 'GP' in self.loss_option:
						if 'wass' in self.loss_option:
							# gradient penalty from WGAN_GP.py
							if self.gpu_mode:
								alpha = torch.rand(x3D_.size()).cuda()
							else:
								alpha = torch.rand(x3D_.size())
			
							x_hat = Variable(alpha * x3D_.data + (1 - alpha) * x3D_hat.data, requires_grad=True)
			
							pred_hat, _, _ = self.D(x_hat)
							if self.gpu_mode:
								gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
											 create_graph=True, retain_graph=True, only_inputs=True)[0]
							else:
								gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()),
												 create_graph=True, retain_graph=True, only_inputs=True)[0]
			
							gradient_penalty = self.lambda_ * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()


						else:
							# DRAGAN Loss (Gradient penalty)
							if self.gpu_mode:
								alpha = torch.rand(x2D_.size()).cuda()
								x2D_hat = Variable(alpha*x2D_.data +
													(1-alpha)*(x2D_.data+0.5*x2D_.data.std()*torch.rand(x2D_.size()).cuda()),
													requires_grad=True)
							else:
								alpha = torch.rand(x2D_.size())
								x2D_hat = Variable(alpha*x2D_.data +
													(1-alpha)*(x2D_.data+0.5*x2D_.data.std()*torch.rand(x2D_.size())),
													requires_grad=True)
							pred_hat,_,_,_ = self.D(x2D_hat)
							if self.gpu_mode:
								gradients = grad(outputs=pred_hat, inputs=x2D_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
													create_graph=True, retain_graph=True, only_inputs=True)[0]
							else:
								gradients = grad(outputs=pred_hat, inputs=x2D_hat, grad_outputs=torch.ones(pred_hat.size()),
													create_graph=True, retain_graph=True, only_inputs=True)[0]
			
							gradient_penalty = self.lambda_ * ((gradients.view(gradients.size(0),-1).norm(2,1)-1)**2).mean()
		
						D_loss = D_loss_GANreal + D_loss_real_id + D_loss_real_pcode + D_loss_GANfake + gradient_penalty
					else:
						D_loss = D_loss_GANreal + D_loss_real_id + D_loss_real_pcode + D_loss_GANfake

					if iD == 0:	
						self.train_hist['D_loss'].append(D_loss.data[0])
						self.train_hist['D_loss_GAN_real'].append(D_loss_GANreal.data[0])
						self.train_hist['D_loss_id'].append(D_loss_real_id.data[0])
						self.train_hist['D_loss_pcode'].append(D_loss_real_pcode.data[0])
						self.train_hist['D_loss_GAN_fake'].append(D_loss_GANfake.data[0])
						self.train_hist['D_acc'].append(D_acc)
	
					D_loss.backward()
					if D_acc < 0.8:
						self.D_optimizer.step()

					if 'wass' in self.loss_option and 'GP' not in self.loss_option:
						for p in self.D.parameters():
							p.data.clamp_(-self.c, self.c)

				# update Genc network
				self.Genc_optimizer.zero_grad()

				reparamZ_ = torch.normal( torch.zeros(self.batch_size, self.Nfx), torch.ones(self.batch_size,self.Nfx) )
				if self.gpu_mode:
				    reparamZ_ = Variable(reparamZ_.cuda())
				else:
				    reparamZ_ = Variable(reparamZ_)

				mu, sigma = self.Genc(x2D_)

				fx = mu + reparamZ_*sigma # reparam trick

				x_3D_hat = self.Gdec(fx, y_random_pcode_onehot_, z_)
				D_fake_GAN, D_fake_id, D_fake_pcode = self.D(x3D_hat)

				Genc_loss_GANfake = self.BCE_loss(D_fake_GAN, self.y_real_)
				Genc_loss_id = self.CE_loss(D_fake_id, y_id_)
				Genc_loss_pcode = self.CE_loss(D_fake_pcode, y_random_pcode_)
				KL_loss = 0.5 * torch.sum(mu**2 + sigma**2 - torch.log(1e-8 + sigma**2)-1) / self.batch_size

				Genc_loss = KL_loss + Genc_loss_GANfake + Genc_loss_id + Genc_loss_pcode

				Genc_loss.backward()
				self.train_hist['Genc_loss'].append(Genc_loss.data[0])
				self.Genc_optimizer.step()

				# update Gdec network
				for iG in range( self.n_gen ):
					self.Gdec_optimizer.zero_grad()
		
					reparamZ_ = torch.normal( torch.zeros(self.batch_size, self.Nfx), torch.ones(self.batch_size,self.Nfx) )
					if self.gpu_mode:
					    reparamZ_ = Variable(reparamZ_.cuda())
					else:
					    reparamZ_ = Variable(reparamZ_)
					mu, sigma = self.Genc(x2D_)

					fx = mu + reparamZ_*sigma # reparam trick

					x_3D_hat = self.Gdec(fx, y_random_pcode_onehot_, z_)

					D_fake_GAN, D_fake_id, D_fake_pcode = self.D(x3D_hat)

					G_loss_GANfake = self.BCE_loss(D_fake_GAN, self.y_real_)
					G_loss_id = self.CE_loss(D_fake_id, y_id_)
					G_loss_pcode = self.CE_loss(D_fake_pcode, y_random_pcode_)
	
					G_loss = G_loss_GANfake + G_loss_id + G_loss_pcode

					if 'recon' in self.loss_option:
						G_loss_recon = self.MSE_loss(x3D_hat, x3D_)
						G_loss += G_loss_recon
					elif 'reconL1' in self.loss_option:
						G_loss_recon = self.L1_loss(x3D_hat, x3D_)
						G_loss += G_loss_recon
	
					if 'dist' in self.loss_option:
						sumA = 0
						sumB = 0
						for iA in range(self.batch_size):
							dist_2D = x2D_[iA]-x2D_ + eps
							dist_3D = x3D_hat[iA]-x3D_hat + eps
							normdist_2D = torch.norm(dist_2D.view(self.batch_size,-1),1, dim=1)
							normdist_3D = torch.norm(dist_3D.view(self.batch_size,-1),1, dim=1)
							sumA += normdist_2D
							sumB += normdist_3D
		
						sumA /= self.data_loader.dataset.stddevA # normalization
						sumB /= self.data_loader.dataset.stddevB # normalization
						sumA /= nPairs # expectation
						sumB /= nPairs # expectation
		
						G_loss_distance = torch.abs( torch.sum( sumA - sumB - normalizerA + normalizerB )) / self.batch_size
						G_loss += G_loss_distance
	
	
					if iG == 0:
						self.train_hist['G_loss'].append(G_loss.data[0])
						self.train_hist['G_loss_GAN_fake'].append(G_loss_GANfake.data[0])
						self.train_hist['G_loss_id'].append(G_loss_id.data[0])
						self.train_hist['G_loss_pcode'].append(G_loss_pcode.data[0])
						if 'recon' in self.loss_option or 'reconL1' in self.loss_option:
							self.train_hist['G_loss_recon'].append(G_loss_recon.data[0])
						if 'dist' in self.loss_option:
							self.train_hist['G_loss_dist'].append(G_loss_distance.data[0])
		
					G_loss.backward()
					self.G_optimizer.step()
					
					if 'recon' in self.loss_option and 'dist' in self.loss_option:
						G_loss = G_loss_GANfake + G_loss_id + G_loss_pcode + G_loss_recon + G_loss_distance
					elif 'recon' in self.loss_option :
						G_loss = G_loss_GANfake + G_loss_id + G_loss_pcode + G_loss_recon
					elif 'dist' in self.loss_option:
						G_loss = G_loss_GANfake + G_loss_id + G_loss_pcode + G_loss_distance
					else:
						G_loss = G_loss_GANfake + G_loss_id + G_loss_pcode
						

	
				if ((iB + 1) % 10) == 0:
					secs = time.time()-start_time_epoch
					hours = secs//3600
					mins = secs/60%60
					#print("%2dh%2dm E:[%2d] B:[%4d/%4d] D: %.4f=%.4f+%.4f+%.4f+%.4f,\n\t\t\t G: %.4f=%.4f+%.4f+%.4f" %
					print("%2dh%2dm E[%2d] B[%d/%d] D: %.4f,G: %.4f, D_acc:%.4f" %
						  (hours,mins, (epoch + 1), (iB + 1), self.data_loader.dataset.__len__() // self.batch_size, 
						  D_loss.data[0], G_loss.data[0], D_acc) )
#						  D_loss.data[0], D_loss_GANreal.data[0], D_loss_real_id.data[0],
#						  D_loss_real_pcode.data[0], D_loss_GANfake.data[0],
#						  G_loss.data[0], G_loss_GANfake.data[0], G_loss_id.data[0],
#						  G_loss_pcode.data[0]) )

			self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
			self.save()
			utils.loss_plot(self.train_hist,
							os.path.join(self.save_dir, self.dataset, self.model_name),
							self.model_name, use_subplot=True)
			self.visualize_results((epoch+1))

		self.train_hist['total_time'].append(time.time() - start_time)
		print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
			  self.epoch, self.train_hist['total_time'][0]))
		print("Training finish!... save training results")

		self.save()
		utils.loss_plot(self.train_hist,
						os.path.join(self.save_dir, self.dataset, self.model_name),
						self.model_name, use_subplot=True)


	def visualize_results(self, epoch, fix=True):
		self.Genc.eval()
		self.Gdec.eval()

		if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
			os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

		if fix:
			""" fixed noise """
			mu, sigma = self.Genc(self.sample_x2D_)

			fx = mu + self.reparamZ_*sigma # reparam trick

			x3D_hat = self.Gdec(fx, iself.sample_pcode_, self.sample_z_)
	
		else:
			""" random noise """
			if self.gpu_mode:
				sample_z_ = Variable(torch.rand((self.batch_size, self.Nz)).cuda(), volatile=True)
			else:
				sample_z_ = Variable(torch.rand((self.batch_size, self.Nz)), volatile=True)

			samples = self.G(sample_z_)

		if self.gpu_mode:
			samples = samples.cpu().data.numpy().squeeze()
		else:
			samples = samples.data.numpy().squeeze()

		fname = self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.npy'
		samples.dump(fname)


	def save(self):
		save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
		torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

		with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
			pickle.dump(self.train_hist, f)

	def load(self):
		save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

		self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
		self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))

		try:
			with open(os.path.join(save_dir, self.model_name + '_history.pkl')) as fhandle:
				self.train_hist = pickle.load(fhandle)
			
			self.epoch_start = len(self.train_hist['per_epoch_time'])
			print( 'loaded epoch {}'.format(self.epoch_start) )
			print( 'history has following keys:' )
			print( self.train_hist.keys() )
		except:
			print('history is not found and ignored')

