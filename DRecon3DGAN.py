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
	def __init__( self, name, Nid, Npcode ):
		super(Encoder, self).__init__()
		self.input_height = 100
		self.input_width = 100
		self.input_dim = 3
		self.name = name
		self.Nid = Nid
		self.Npcode = Npcode

		self.conv = nn.Sequential(
			nn.Conv2d(self.input_dim, 64, 11, 4, 1,bias=True),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 128, 5, 2, 1,bias=True),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 256, 5, 2, 1,bias=True),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256, 512, 5, 2, 1,bias=True),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d(512, 320, 8 , 1, 1, bias=True),
			nn.Sigmoid(),
		)

		utils.initialize_weights(self)

	def forward(self, input):
		x = self.conv( input )
		return x

class Decoder3d( nn.Module ):
	def __init__(self, Npcode, nOutputCh=4):
		super(Decoder3d, self).__init__()
		self.nOutputCh = nOutputCh
		self.Npcode = Npcode

		self.deconv = nn.Sequential(
			nn.ConvTranspose3d(320+Npcode, 512, 4, bias=False),
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
	def forward(self, fx, y_pcode_onehot):
		feature = torch.cat((fx, y_pcode_onehot),1)
		x = self.deconv( feature.unsqueeze(2).unsqueeze(3).unsqueeze(4) )
		return x

class Decoder2d( nn.Module ):
	def __init__(self, Npcode, nOutputCh=4):
		super(Decoder2d, self).__init__()
		self.nOutputCh = nOutputCh

		self.fc = nn.Sequential(
			nn.Linear( 320+Npcode, 320 )
		)

		self.fconv = nn.Sequential(
			nn.ConvTranspose2d(320, 512, 4, bias=False),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.ConvTranspose2d(32, nOutputCh, 4, 2, 1, bias=False),
			nn.Sigmoid(),
		)
	def forward(self, fx, y_pcode_onehot):
		feature = torch.cat((fx, y_pcode_onehot),1)
		x = self.fc( feature )
		x = self.fconv( x.unsqueeze(2).unsqueeze(3) )
		return x

class generator2d3d(nn.Module):
	def __init__(self, Nid, Npcode, nOutputCh=4):
		super(generator2d3d, self).__init__()

		self.Genc = Encoder('Genc', Nid, Npcode)
		self.Gdec2d = Decoder2d(Npcode, nOutputCh['2d'])
		self.Gdec3d = Decoder3d(Npcode, nOutputCh['3d'])

		utils.initialize_weights(self)

	def forward(self, x_, y_pcode_onehot_):
		fx = self.Genc( x_ )
		fx = fx.view(-1,320)
		xhat2d = self.Gdec2d(fx, y_pcode_onehot_)
		xhat3d = self.Gdec3d(fx, y_pcode_onehot_)

		return xhat2d, xhat3d

class discriminator2d(nn.Module):
	# Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
	# Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
	def __init__(self, Nid=105, Npcode=48, nInputCh=4, norm=nn.BatchNorm2d):
		super(discriminator2d, self).__init__()
		self.nInputCh = nInputCh

		self.conv = nn.Sequential(
			nn.Conv2d(nInputCh, 32, 4, 2, 1, bias=False), # 128 -> 64
			norm(32),
			nn.LeakyReLU(0.2),
			nn.Conv2d(32, 64, 4, 2, 1, bias=False), # 64 -> 32
			norm(64),
			nn.LeakyReLU(0.2),
			nn.Conv2d(64, 128, 4, 2, 1, bias=False), # 32 -> 16
			norm(128),
			nn.LeakyReLU(0.2),
			nn.Conv2d(128, 256, 4, 2, 1, bias=False), # 16 -> 8
			norm(256),
			nn.LeakyReLU(0.2),
			nn.Conv2d(256, 512, 4, 2, 1, bias=False), # 8 -> 4
			norm(512),
			nn.LeakyReLU(0.2)
		)

		self.convGAN = nn.Sequential(
			nn.Conv2d(512, 1, 4, bias=False),
			nn.Sigmoid()
		)

		self.convID = nn.Sequential(
			nn.Conv2d(512, Nid, 4, bias=False),
		)

		self.convPCode = nn.Sequential(
			nn.Conv2d(512, Npcode, 4, bias=False),
		)
		utils.initialize_weights(self)

	def forward(self, input):
		feature = self.conv(input)

		fGAN = self.convGAN( feature ).squeeze(3).squeeze(2)
		fid = self.convID( feature ).squeeze(3).squeeze(2)
		fcode = self.convPCode( feature ).squeeze(3).squeeze(2)

		return fGAN, fid, fcode

class discriminator3d(nn.Module):
	# Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
	# Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
	def __init__(self, Nid=105, Npcode=48, nInputCh=4, norm=nn.BatchNorm3d):
		super(discriminator3d, self).__init__()
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

class DRecon3DGAN(object):
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
		self.multi_gpu = args.multi_gpu
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
		self.nDaccAvg = args.nDaccAvg
		if 'wass' in self.loss_option:
			self.n_critic = 5

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
			self.Nd = 337 # 200
			self.Np = 9
			self.Ni = 20
			self.Nz = 50
		elif self.dataset == 'CASIA-WebFace':
			self.data_loader = utils.CustomDataLoader(data_dir, transform=transforms.Compose(
				[transforms.Scale(100), transforms.RandomCrop(96), transforms.ToTensor()]), batch_size=self.batch_size,
												 shuffle=True)
			self.Nd = 10885 
			self.Np = 13
			self.Ni = 20
			self.Nz = 50
		elif self.dataset == 'Bosphorus':
			self.data_loader = DataLoader( utils.Bosphorus(data_dir, use_image=True, fname_cache=args.fname_cache,
											transform=transforms.ToTensor(),
											shape=128, image_shape=256, center=self.centerBosphorus,
											use_colorPCL=True),
											batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
			self.Nid = 105
			self.Npcode = len(self.data_loader.dataset.posecodemap)
			self.Nz = 50

		# networks init
		self.G = generator2d3d(self.Nid, self.Npcode, nOutputCh={'2d':3,'3d':1})
		self.D2d = discriminator2d(self.Nid, self.Npcode, nInputCh=3)
		self.D3d = discriminator3d(self.Nid, self.Npcode, nInputCh=1)
		self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
		self.D2d_optimizer = optim.Adam(self.D2d.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
		self.D3d_optimizer = optim.Adam(self.D3d.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

		if self.gpu_mode:
			self.G.cuda()
			self.D2d.cuda()
			self.D3d.cuda()
			self.CE_loss = nn.CrossEntropyLoss().cuda()
			self.BCE_loss = nn.BCELoss().cuda()
			self.MSE_loss = nn.MSELoss().cuda()
			self.L1_loss = nn.L1Loss().cuda()

			if self.multi_gpu:
				gpus = [0,1]
				self.G = torch.nn.DataParallel(self.G, device_ids=gpus).cuda() 
				self.D2d = torch.nn.DataParallel(self.D2d, device_ids=gpus).cuda() 
				self.D3d = torch.nn.DataParallel(self.D3d, device_ids=gpus).cuda() 
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
		train_hist_keys = ['D2d_loss',
                           'D3d_loss',
                           'D2d_acc',
                           'D3d_acc',
                           'G_loss',
                           'per_epoch_time',
                           'total_time']
		if 'recon' in self.loss_option:
			train_hist_keys.append('G_loss_recon')
		if 'dist' in self.loss_option:
			train_hist_keys.append('G_loss_dist')

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

		self.D2d.train()
		self.D3d.train()
		start_time = time.time()

		nBatchesPerEpoch = self.data_loader.dataset.__len__() // self.batch_size
		print('training start from epoch {}!!'.format(self.epoch_start+1))
		for epoch in range(self.epoch_start, self.epoch):
			self.G.train()
			epoch_start_time = time.time()
			start_time_epoch = time.time()

			for iB, (x3D_, y_, x2D_ ) in enumerate(self.data_loader):
				if iB == nBatchesPerEpoch:
					break

				projected, _ = torch.max( x3D_[:,1:,:,:,:], 4, keepdim=False)
				x3D_ = x3D_[:,0:1,:,:,:]
				y_random_pcode_ = torch.floor(torch.rand(self.batch_size)*self.Npcode).long()
				y_random_pcode_onehot_ = torch.zeros( self.batch_size, self.Npcode )
				y_random_pcode_onehot_.scatter_(1, y_random_pcode_.view(-1,1), 1)
				y_id_ = y_['id']
				y_pcode_ = y_['pcode']
				y_pcode_onehot_ = torch.zeros( self.batch_size, self.Npcode )
				y_pcode_onehot_.scatter_(1, y_pcode_.view(-1,1), 1)

				if self.gpu_mode:
					x2D_= Variable(x2D_.cuda())
					x3D_ = Variable(x3D_.cuda())
					projected = Variable(projected.cuda())
					y_id_ = Variable( y_id_.cuda() )
					y_pcode_ = Variable(y_pcode_.cuda())
					y_pcode_onehot_ = Variable( y_pcode_onehot_.cuda() )
					y_random_pcode_ = Variable(y_random_pcode_.cuda())
					y_random_pcode_onehot_ = Variable( y_random_pcode_onehot_.cuda() )
				else:
					x2D_= Variable(x2D_)
					x3D_ = Variable(x3D_)
					projected = Variable(projected)
					y_id_ = Variable(y_id_)
					y_pcode_ = Variable(y_pcode_)
					y_pcode_onehot_ = Variable( y_pcode_onehot_ )
					y_random_pcode_ = Variable(y_random_pcode_)
					y_random_pcode_onehot_ = Variable( y_random_pcode_onehot_ )

				# update D network
				for iD in range(self.n_critic) :
					self.D2d_optimizer.zero_grad()
					self.D3d_optimizer.zero_grad()
	
					d_gan2d, d_id2d, d_expr2d = self.D2d(projected)
					loss_d_real_gan2d = self.BCE_loss(d_gan2d, self.y_real_)
					loss_d_real_id2d = self.CE_loss(d_id2d, y_id_)
					loss_d_real_expr2d = self.CE_loss(d_expr2d, y_pcode_)

					d_gan3d, d_id3d, d_expr3d = self.D3d(x3D_)
					loss_d_real_gan3d = self.BCE_loss(d_gan3d, self.y_real_)
					loss_d_real_id3d = self.CE_loss(d_id3d, y_id_)
					loss_d_real_expr3d = self.CE_loss(d_expr3d, y_pcode_)
	
					xhat2d, xhat3d = self.G(x2D_, y_random_pcode_onehot_)
					d_fake_gan2d, _, _ = self.D2d( xhat2d )
					d_fake_gan3d, _, _ = self.D3d( xhat3d )
					loss_d_fake_gan2d = self.BCE_loss(d_fake_gan2d, self.y_fake_)
					loss_d_fake_gan3d = self.BCE_loss(d_fake_gan3d, self.y_fake_)
	
					num_correct_real2d = torch.sum(d_gan2d>0.5)
					num_correct_fake2d = torch.sum(d_fake_gan2d<0.5)
					D2d_acc = float(num_correct_real2d.data[0] + num_correct_fake2d.data[0]) / (self.batch_size*2)
					num_correct_real3d = torch.sum(d_gan3d>0.5)
					num_correct_fake3d = torch.sum(d_fake_gan3d<0.5)
					D3d_acc = float(num_correct_real3d.data[0] + num_correct_fake3d.data[0]) / (self.batch_size*2)
	
					D2d_loss = loss_d_real_gan2d + loss_d_real_id2d + loss_d_real_expr2d + loss_d_fake_gan2d
					D3d_loss = loss_d_real_gan3d + loss_d_real_id3d + loss_d_real_expr3d + loss_d_fake_gan3d

					if iD == 0:	
						self.train_hist['D2d_loss'].append(D2d_loss.data[0])
						self.train_hist['D3d_loss'].append(D3d_loss.data[0])
						self.train_hist['D2d_acc'].append(D2d_acc)
						self.train_hist['D3d_acc'].append(D3d_acc)
	
					D2d_loss.backward(retain_graph=True)
					D3d_loss.backward()
					if D2d_acc < 0.8:
						self.D2d_optimizer.step()
					if D3d_acc < 0.8:
						self.D3d_optimizer.step()

				# update G network
				for iG in range( self.n_gen ):
					self.G_optimizer.zero_grad()
		
					xhat2d, xhat3d = self.G(x2D_, y_pcode_onehot_)

					d_gan2d, d_id2d, d_expr2d = self.D2d(xhat2d)
					loss_g_gan2d = self.BCE_loss(d_gan2d, self.y_real_)
					loss_g_id2d = self.CE_loss(d_id2d, y_id_)
					loss_g_expr2d = self.CE_loss(d_expr2d, y_pcode_)

					d_gan3d, d_id3d, d_expr3d = self.D3d(xhat3d)
					loss_g_gan3d = self.BCE_loss(d_gan3d, self.y_real_)
					loss_g_id3d = self.CE_loss(d_id3d, y_id_)
					loss_g_expr3d = self.CE_loss(d_expr3d, y_pcode_)

					G_loss = loss_g_gan2d + loss_g_id2d + loss_g_expr2d + \
								loss_g_gan3d + loss_g_id3d + loss_g_expr3d
	
					if iG == 0:
						self.train_hist['G_loss'].append(G_loss.data[0])
		
					G_loss.backward()
					self.G_optimizer.step()
					
				if ((iB + 1) % 10) == 0 or (iB+1)==nBatchesPerEpoch:
					secs = time.time()-start_time_epoch
					hours = secs//3600
					mins = secs/60%60
					#print("%2dh%2dm E[%2d] B[%d/%d] D: %.4f, G: %.4f, D_acc:%.4f"% 
					print("%2dh%2dm E[%2d] B[%d/%d] D: %.4f/%.4f, G: %.4f, D_acc:%.4f/%.4f" %
						  (hours,mins, (epoch + 1), (iB + 1), nBatchesPerEpoch,
						  D2d_loss.data[0],
						  D3d_loss.data[0],
						  G_loss.data[0],
						  D3d_acc,
						  D2d_acc) )
				
			self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
			if epoch==0 or (epoch+1)%5 == 0:
				self.dump_x_hat(xhat2d, xhat3d, epoch+1)
			self.save()
			utils.loss_plot(self.train_hist,
							os.path.join(self.save_dir, self.dataset, self.model_name),
							self.model_name, use_subplot=True)

		self.train_hist['total_time'].append(time.time() - start_time)
		print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
			  self.epoch, self.train_hist['total_time'][0]))
		print("Training finish!... save training results")

		self.save()
		utils.loss_plot(self.train_hist,
						os.path.join(self.save_dir, self.dataset, self.model_name),
						self.model_name, use_subplot=True)


	def dump_x_hat(self, xhat2d, xhat3d, epoch):
		print( 'dump x_hat...' )

		save_dir = os.path.join( self.result_dir, self.dataset, self.model_name )
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		if self.gpu_mode:
			xhat2d = xhat2d.cpu().data.numpy().squeeze()
			xhat3d = xhat3d.cpu().data.numpy().squeeze()
		else:
			xhat2d = xhat2d.data.numpy().squeeze()
			xhat3d = xhat3d.data.numpy().squeeze()

		fname = os.path.join( save_dir , self.model_name + '_xhat2d_epoch%03d' % epoch + '.npy' )
		xhat2d.dump(fname)
		fname = os.path.join( save_dir , self.model_name + '_xhat3d_epoch%03d' % epoch + '.npy' )
		xhat3d.dump(fname)

	def get_image_batch(self):
		dataIter = iter(self.data_loader)
		return next(dataIter)

	def visualize_results(self,a=None,b=None):
		print( 'visualizing result...' )
		save_dir = os.path.join(self.result_dir, self.dataset, self.model_name, 'generate') 
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		self.G.eval()

		# reconstruction (inference 2D-to-3D )
		x3D, y_, x2D = self.get_image_batch()
		y_ = y_['pcode']
		y_onehot_ = torch.zeros( self.batch_size, self.Npcode )
		y_onehot_.scatter_(1, y_.view(-1,1), 1)
	
		x2D_= Variable(x2D.cuda(),volatile=True)
		y_ = Variable( y_.cuda(), volatile=True )
		y_onehot_ = Variable( y_onehot_.cuda(), volatile=True )

		xhat2d, xhat3d = self.G(x2D_, y_onehot_)
	
		if self.gpu_mode:
			xhat2d = xhat2d.cpu().data.numpy().squeeze()
			xhat3d = xhat3d.cpu().data.numpy().squeeze()
			x3D = x3D.numpy()
		else:
			xhat2d = xhat2d.data.numpy().squeeze()
			xhat3d = xhat3d.data.numpy().squeeze()
			x3D = x3D.numpy()

		print( 'saving...')
		for i in range( self.batch_size ):
			fname = os.path.join(save_dir, self.model_name + '_%02d_expr%02d.png'%(i,y_[i].data[0]))
			imageio.imwrite(fname, x2D[i].numpy().transpose(1,2,0))

			filename = os.path.join( save_dir,self.model_name+'_recon%02d_expr%02d_xhat2d.npy'%(i,y_[i].data[0]))
			xhat2d[i:i+1].dump( filename )
	
			filename = os.path.join( save_dir,self.model_name+'_recon%02d_expr%02d_xhat3d.npy'%(i,y_[i].data[0]))
			xhat3d[i:i+1].dump( filename )
	
			filename = os.path.join( save_dir,self.model_name+'_recon%02d_GT_expr%02d.npy'%(i,y_[i].data[0]))
			x3D[i:i+1].dump( filename )

		print( 'fixed input with different expr...')
		# fixed input with different expr
		nPcodes = self.Npcode
		x2ds = x2D_[0:3,:,:,:]
		temp_x2ds = torch.split(x2ds,1)
		x2ds = []
		for i in range(3):
			x2ds += temp_x2ds[i:i+1]*nPcodes
		x2ds = torch.cat(x2ds,0)
		sample_pcode_onehot = torch.zeros( nPcodes*3, nPcodes )
		for iS in range( nPcodes*3 ):
			ii = iS%nPcodes
			sample_pcode_onehot[iS,ii] = 1
		sample_pcode_onehot = Variable( sample_pcode_onehot.cuda(), volatile=True )

		xhat2d, xhat3d = self.G( x2ds, sample_pcode_onehot )
		xhat2d = xhat2d.cpu().data.numpy().squeeze()
		xhat3d = xhat3d.cpu().data.numpy().squeeze()
		for i in range( 3 ):
			print( 'saving...{}'.format(i))
			fname = os.path.join(save_dir,self.model_name + '_%02d_varyingexpr.png'%(i))
			imageio.imwrite(fname, x2D_[i].cpu().data.numpy().transpose(1,2,0))

			fname = os.path.join(save_dir,self.model_name + 'xhat2d_%02d.npy'%(i))
			xhat2d[nPcodes*i:nPcodes*(i+1)].dump(fname)
			
			fname = os.path.join(save_dir,self.model_name + 'xhat3d_%02d.npy'%(i))
			xhat3d[nPcodes*i:nPcodes*(i+1)].dump(fname)

#			for j in range( nPcodes ):
#				filename = os.path.join( self.result_dir, self.dataset, self.model_name, 'generate',
#											self.model_name+'_sample%03d_expr%02d.npy'%(i,j))
#				np.expand_dims(samples_numpy[j],0).dump( filename )



	def save(self):
		save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		if self.multi_gpu:
			torch.save(self.G.module.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
			torch.save(self.D2d.module.state_dict(), os.path.join(save_dir, self.model_name + '_D2d.pkl'))
			torch.save(self.D3d.module.state_dict(), os.path.join(save_dir, self.model_name + '_D3d.pkl'))
		else:
			torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
			torch.save(self.D2d.state_dict(), os.path.join(save_dir, self.model_name + '_D2d.pkl'))
			torch.save(self.D3d.state_dict(), os.path.join(save_dir, self.model_name + '_D3d.pkl'))

		with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
			pickle.dump(self.train_hist, f)

	def load(self):
		save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

		if self.multi_gpu:
			self.G.module.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
			self.D2d.module.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D2d.pkl')))
			self.D3d.module.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D3d.pkl')))
		else:
			self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
			self.D2d.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D2d.pkl')))
			self.D3d.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D3d.pkl')))

		try:
			with open(os.path.join(save_dir, self.model_name + '_history.pkl')) as fhandle:
				self.train_hist = pickle.load(fhandle)
			
			self.epoch_start = len(self.train_hist['per_epoch_time'])
			print( 'loaded epoch {}'.format(self.epoch_start) )
			print( 'history has following keys:' )
			print( self.train_hist.keys() )
		except:
			print('history is not found and ignored')

	def interpolate_id(self, opts):
		print( 'interpolate_id()...' )
		save_dir = os.path.join(self.result_dir, self.dataset, self.model_name, 'interp_id') 
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		
		self.G.eval()

		n_interp = opts.n_interp

		x3D, y, x2D = self.get_image_batch()

		fname = os.path.join(save_dir, self.model_name + '_A.png')
		imageio.imwrite(fname, x2D[0].numpy().transpose(1,2,0))
		fname = os.path.join(save_dir, self.model_name + '_B.png')
		imageio.imwrite(fname, x2D[1].numpy().transpose(1,2,0))
		
		y = y['pcode']
		y_onehot = torch.zeros( self.batch_size, self.Npcode )
		y_onehot.scatter_(1, y.view(-1,1), 1)

		if self.gpu_mode:
			self.G = self.G.cuda()
			x2D = Variable(x2D.cuda(),volatile=True)
			y = Variable( y.cuda(), volatile=True )
			y_onehot = Variable( y_onehot.cuda(), volatile=True )

		#make interpolation 3D
		print( 'interpolating...' )

		Genc = self.G.Genc
		Gdec2d = self.G.Gdec2d
		Gdec3d = self.G.Gdec3d

		fx = Genc( x2D )
		fx = fx.view(-1,320)

		dfx = (fx[1].unsqueeze(0)-fx[0].unsqueeze(0))/n_interp

		single_y = y_onehot[0].unsqueeze(0)
		for i in range(1, n_interp):
			save_dir_i = os.path.join( save_dir, str(i) )
			if not os.path.exists(save_dir_i):
				os.makedirs(save_dir_i)

			fx_interp = fx[0].unsqueeze(0) + i*dfx

			xhat2d = Gdec2d(fx_interp , single_y)
			xhat3d = Gdec3d(fx_interp , single_y)
			if self.gpu_mode:
				xhat2d = xhat2d.cpu().data.numpy()
				xhat3d = xhat3d.cpu().data.numpy()
			else:
				xhat2d = xhat2d.data.numpy()
				xhat3d = xhat3d.data.numpy()
			fname = os.path.join(save_dir_i, '2d.npy' )
			xhat2d.dump(fname)
			fname = os.path.join(save_dir_i, '3d.npy' )
			xhat3d.dump(fname)

	def interpolate_expr(self, opts):
		print( 'interpolate_expr()...' )
		save_dir = os.path.join(self.result_dir, self.dataset, self.model_name, 'interp_expr') 
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		
		self.G.eval()

		n_interp = opts.n_interp

		x3D, y, x2D = self.get_image_batch()

		fname = os.path.join(save_dir, self.model_name + '_A.png')
		imageio.imwrite(fname, x2D[0].numpy().transpose(1,2,0))
		fname = os.path.join(save_dir, self.model_name + '_B.png')
		imageio.imwrite(fname, x2D[1].numpy().transpose(1,2,0))
		
		y = y['pcode']
		y_onehot = torch.zeros( self.batch_size, self.Npcode )
		y_onehot.scatter_(1, y.view(-1,1), 1)

		if self.gpu_mode:
			self.G = self.G.cuda()
			x2D = Variable(x2D.cuda(),volatile=True)
			y = Variable( y.cuda(), volatile=True )
			y_onehot = Variable( y_onehot.cuda(), volatile=True )

		dy = (y_onehot[1].unsqueeze(0)-y_onehot[0].unsqueeze(0))/n_interp

		#make interpolation 3D
		print( 'interpolating...' )
		print( y_onehot[0] )
		print( y_onehot[1] )

		singleX2D = x2D[0].unsqueeze(0)
		for i in range(1, n_interp):
			save_dir_i = os.path.join( save_dir, str(i) )
			if not os.path.exists(save_dir_i):
				os.makedirs(save_dir_i)

			y_interp = y_onehot[0].unsqueeze(0) + i*dy
			xhat2d, xhat3d = self.G(singleX2D, y_interp)
			if self.gpu_mode:
				xhat2d = xhat2d.cpu().data.numpy()
				xhat3d = xhat3d.cpu().data.numpy()
			else:
				xhat2d = xhat2d.data.numpy()
				xhat3d = xhat3d.data.numpy()
			fname = os.path.join(save_dir_i, '2d.npy' )
			xhat2d.dump(fname)
			fname = os.path.join(save_dir_i, '3d.npy' )
			xhat3d.dump(fname)
	
	def control_expr(self):
		print( 'control_expr()...' )
		save_dir = os.path.join(self.result_dir, self.dataset, self.model_name, 'control_expr') 
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		
		self.G.eval()

		nSubj = 3

		x3D, y, x2D = self.get_image_batch()

		for i in range( nSubj ):
			fname = os.path.join(save_dir, 'sub_{}.png'.format(i))
			imageio.imwrite(fname, x2D[i].numpy().transpose(1,2,0))

		# fixed input with different expr
		nPcodes = self.Npcode
		x2ds = x2D[0:nSubj,:,:,:]
		temp_x2ds = torch.split(x2ds,1)
		x2ds = []
		for i in range(nSubj):
			x2ds += temp_x2ds[i:i+1]*nPcodes
		x2ds = torch.cat(x2ds,0)

		# different expr
		y_onehot = torch.zeros( nPcodes*nSubj, nPcodes )
		for iS in range( nPcodes*nSubj ):
			ii = iS%nPcodes
			y_onehot[iS,ii] = 1

		if self.gpu_mode:
			self.G = self.G.cuda()
			x2ds = Variable(x2ds.cuda(),volatile=True)
			y_onehot = Variable( y_onehot.cuda(), volatile=True )

		# forward and save
		xhat2d, xhat3d = self.G( x2ds, y_onehot )
		xhat2d = xhat2d.cpu().data.numpy()
		xhat3d = xhat3d.cpu().data.numpy()
		#for i in range( nSubj ):
		for i in range( nSubj*nPcodes ):
			print( 'saving...{}'.format(i))
			save_dir_i = os.path.join( save_dir, str(i) )
			if not os.path.exists(save_dir_i):
				os.makedirs(save_dir_i)

			fname = os.path.join(save_dir_i,'2d.npy')
			xhat2d[i:(i+1)].dump(fname)
			#xhat2d[nPcodes*i:nPcodes*(i+1)].dump(fname)
			
			fname = os.path.join(save_dir_i,'3d.npy')
			xhat3d[i:(i+1)].dump(fname)
			#xhat3d[nPcodes*i:nPcodes*(i+1)].dump(fname)

	def reconstruct(self):
		print( 'reconstruct()...' )
		save_dir = os.path.join(self.result_dir, self.dataset, self.model_name, 'reconstruct') 
		GT_dir = os.path.join( save_dir, 'GTs' )
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		if not os.path.exists(GT_dir):
			os.makedirs(GT_dir)
		
		self.G.eval()

		x3D, y, x2D = self.get_image_batch()

		y = y['pcode']
		y_onehot = torch.zeros( self.batch_size, self.Npcode )
		y_onehot.scatter_(1, y.view(-1,1), 1)

		print( 'saving input...')
		for i in range( self.batch_size ):
			indicator = 'recon%02d_expr%02d'%(i,y[i])
			fname = os.path.join(save_dir, indicator + '.png')
			imageio.imwrite(fname, x2D[i].numpy().transpose(1,2,0))


		if self.gpu_mode:
			self.G = self.G.cuda()
			x2D = Variable(x2D.cuda(),volatile=True)
			y = Variable( y.cuda(), volatile=True )
			y_onehot = Variable( y_onehot.cuda(), volatile=True )


		xhat2d, xhat3d = self.G(x2D, y_onehot )
	
		xhat2d = xhat2d.cpu().data.numpy()
		xhat3d = xhat3d.cpu().data.numpy()
		x3D = x3D.numpy()
		print( 'saving recon...')
		for i in range( self.batch_size ):
			indicator = 'recon%02d_expr%02d'%(i,y[i].data[0])
			
			save_dir_i = os.path.join( save_dir, indicator )
			if not os.path.exists(save_dir_i):
				os.makedirs(save_dir_i)

			filename = os.path.join( save_dir_i, '2d.npy')
			np.expand_dims(xhat2d[i],0).dump( filename )
			filename = os.path.join( save_dir_i, '3d.npy')
			np.expand_dims(xhat3d[i],0).dump( filename )

			filename = os.path.join( GT_dir, indicator+'.npy' )
			np.expand_dims(x3D[i],0).dump( filename )
		

	def compare(self, x2D, y_, y_onehot, dir_dest='' ):
		print( 'comparing result...' )
		if len(dir_dest) > 0:
			save_dir = dir_dest
		else:
			save_dir = os.path.join(self.result_dir, self.dataset, 'compare' )
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		# reconstruction (inference 2D-to-3D )
		xhat2d, xhat3d = self.G(x2D, y_onehot)
	
		xhat2d = xhat2d.cpu().data.numpy()
		xhat3d = xhat3d.cpu().data.numpy()

		print( 'saving compare...')
		for i in range( self.batch_size ):
			indicator = '%02d_expr%02d'%(i,y[i].data[0])
			
			save_dir_i = os.path.join( save_dir, indicator )
			if not os.path.exists(save_dir_i):
				os.makedirs(save_dir_i)

			filename = os.path.join( save_dir_i, '2d.npy')
			np.expand_dims(xhat2d[i],0).dump( filename )
			filename = os.path.join( save_dir_i, '3d.npy')
			np.expand_dims(xhat3d[i],0).dump( filename )

			filename = os.path.join( GT_dir, indicator+'.npy' )
			np.expand_dims(x3D[i],0).dump( filename )
		
