import utils, torch, time, os, pickle
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
			# Conv11, Conv12
			nn.Conv2d( self.input_dim, 32, 3, 1, 1 ),
			nn.BatchNorm2d( 32 ),
			nn.ELU(),
			nn.Conv2d( 32, 64, 3, 1, 1 ),
			nn.BatchNorm2d( 64 ),
			nn.ELU(),

			# Conv21, Conv22, Conv23
			nn.Conv2d( 64, 64, 3, 2, 1 ),
			nn.BatchNorm2d( 64 ),
			nn.ELU(),
			nn.Conv2d( 64, 64, 3, 1, 1 ),
			nn.BatchNorm2d( 64 ),
			nn.ELU(),
			nn.Conv2d( 64, 128, 3, 1, 1 ),
			nn.BatchNorm2d( 128 ),
			nn.ELU(),

			# Conv31, Conv32, Conv33
			nn.Conv2d( 128, 128, 3, 2, 1 ),
			nn.BatchNorm2d( 128 ),
			nn.ELU(),
			nn.Conv2d( 128, 96, 3, 1, 1 ),
			nn.BatchNorm2d( 96 ),
			nn.ELU(),
			nn.Conv2d( 96, 192, 3, 1, 1 ),
			nn.BatchNorm2d( 192 ),
			nn.ELU(),

			# Conv41, Conv42, Conv43
			nn.Conv2d( 192, 192, 3, 2, 1 ),
			nn.BatchNorm2d( 192 ),
			nn.ELU(),
			nn.Conv2d( 192, 128, 3, 1, 1 ),
			nn.BatchNorm2d( 128 ),
			nn.ELU(),
			nn.Conv2d( 128, 256, 3, 1, 1 ),
			nn.BatchNorm2d( 256 ),
			nn.ELU(),

			# Conv51, Conv52, Conv53
			nn.Conv2d( 256, 256, 3, 2, 1 ),
			nn.BatchNorm2d( 256 ),
			nn.ELU(),
			nn.Conv2d( 256, 160, 3, 1, 1 ),
			nn.BatchNorm2d( 160 ),
			nn.ELU(),
			nn.Conv2d( 160, 320, 3, 1, 1 ),
			nn.BatchNorm2d( 320 ),
			nn.ELU(),

			# AvgPool
			nn.AvgPool2d( 6 )
		)

		if self.name == 'D':
			self.fc_GAN = nn.Sequential(
				nn.Linear( 320, 1 ),
				nn.Sigmoid()
			)
			self.fc_id = nn.Linear( 320, Nid )
			self.fc_pcode = nn.Linear( 320, Npcode )

		utils.initialize_weights(self)

	def forward(self, input):
		x = self.conv( input )
		if self.name == 'D':
			x_flat = x.view(x.size(0),-1)
			fGAN = self.fc_GAN( x_flat )
			fid = self.fc_id( x_flat )
			fpcode = self.fc_pcode( x_flat )
			x = ( fGAN, fid, fpcode )
		return x

class Decoder( nn.Module ):
	def __init__(self, Npcode, Nz):
		super(Decoder, self).__init__()
		self.output_dim = 3

		self.fc = nn.Sequential(
			nn.Linear( 320+Npcode+Nz, 6*6*320 )
		)

		self.fconv = nn.Sequential(
			# FConv52, FConv51
			nn.ConvTranspose2d( 320, 160, 3, 1, 1 ), 
			nn.BatchNorm2d( 160 ),
			nn.ELU(),
			nn.ConvTranspose2d( 160, 256, 3, 1, 1 ), 
			nn.BatchNorm2d( 256 ),
			nn.ELU(),

			# FConv43, FConv42, FConv41
			nn.ConvTranspose2d( 256, 256, 3, 2, 1, output_padding=1 ), 
			nn.BatchNorm2d( 256 ),
			nn.ELU(),
			nn.ConvTranspose2d( 256, 128, 3, 1, 1 ), 
			nn.BatchNorm2d( 128 ),
			nn.ELU(),
			nn.ConvTranspose2d( 128, 192, 3, 1, 1 ), 
			nn.BatchNorm2d( 192 ),
			nn.ELU(),

			# FConv33, FConv32, FConv31
			nn.ConvTranspose2d( 192, 192, 3, 2, 1, output_padding=1 ), 
			nn.BatchNorm2d( 192 ),
			nn.ELU(),
			nn.ConvTranspose2d( 192, 96, 3, 1, 1 ), 
			nn.BatchNorm2d( 96 ),
			nn.ELU(),
			nn.ConvTranspose2d( 96, 128, 3, 1, 1 ), 
			nn.BatchNorm2d( 128 ),
			nn.ELU(),

			# FConv23, FConv22, FConv21
			nn.ConvTranspose2d( 128, 128, 3, 2, 1, output_padding=1 ), 
			nn.BatchNorm2d( 128 ),
			nn.ELU(),
			nn.ConvTranspose2d( 128, 64, 3, 1, 1 ), 
			nn.BatchNorm2d( 64 ),
			nn.ELU(),
			nn.ConvTranspose2d( 64, 64, 3, 1, 1 ), 
			nn.BatchNorm2d( 64 ),
			nn.ELU(),

			# FConv13, FConv12, FConv11
			nn.ConvTranspose2d( 64, 64, 3, 2, 1, output_padding=1 ), 
			nn.BatchNorm2d( 64 ),
			nn.ELU(),
			nn.ConvTranspose2d( 64, 32, 3, 1, 1 ), 
			nn.BatchNorm2d( 32 ),
			nn.ELU(),
			nn.ConvTranspose2d( 32, self.output_dim, 3, 1, 1 ),
			nn.Sigmoid(),
		)
	def forward(self, fx, y_pcode_onehot, z):
		feature = torch.cat((fx, y_pcode_onehot, z),1)
		x = self.fc( feature )
		x = x.view(-1,320,6,6)
		x = self.fconv( x )
		return x


class generator(nn.Module):
	def __init__(self, Nid, Npcode, Nz):
		super(generator, self).__init__()

		self.Genc = Encoder('Genc', Nid, Npcode)
		self.Gdec = Decoder(Npcode, Nz)

		utils.initialize_weights(self)

	def forward(self, x_, y_pcode_onehot_, z_):
		fx = self.Genc( x_ )
		fx = fx.view(-1, 320)
		x_hat = self.Gdec(fx, y_pcode_onehot_, z_)

		return x_hat

class DRGAN3D(object):
	def __init__(self, args):
		# parameters
		self.epoch = args.epoch
		self.sample_num = 16
		self.batch_size = args.batch_size
		self.save_dir = args.save_dir
		self.result_dir = args.result_dir
		self.dataset = args.dataset
		self.dataroot_dir = args.dataroot_dir
		self.log_dir = args.log_dir
		self.gpu_mode = args.gpu_mode
		self.num_workers = args.num_workers
		self.model_name = args.gan_type
		self.use_GP = args.use_GP
		if self.use_GP:
			self.model_name = self.model_name + '_GP'
		if len(args.comment) > 0:
			self.model_name = self.model_name + '_' + args.comment
		self.lambda_ = 0.25
		self.nSamples2visualize = 10

		if self.dataset == 'MultiPie' or self.dataset == 'miniPie':
			self.Nd = 337 # 200
			self.Np = 9
			self.Ni = 20
			self.Nz = 50
		elif self.dataset == 'Bosphorus':
			self.Nz = 50
		elif self.dataset == 'CASIA-WebFace':
			self.Nd = 10885 
			self.Np = 13
			self.Ni = 20
			self.Nz = 50


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
											shape=(128,128,128), image_shape=96),
											batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
			self.Nid = 105
			self.Npcode = len(self.data_loader.dataset.posecodemap)

		# fixed samples for reconstruction visualization
		nSamples = self.nSamples2visualize
		sample_x2D_s = []
		for iB, (sample_x3D_,sample_y_,sample_x2D_) in enumerate(self.data_loader):
			sample_x2D_s.append( sample_x2D_ )
			break
			if iB > nSamples // self.batch_size:
				break
		sample_x2D_s = torch.cat( sample_x2D_s )[:nSamples,:,:,:]
		sample_x2D_s = torch.split( sample_x2D_s, 1 )
		sample_x2D_s = [ [x]*self.Npcode for x in sample_x2D_s ]
		flatten = lambda l: [item for sublist in l for item in sublist]
		self.sample_x2D_ = torch.cat( flatten(sample_x2D_s) )
#		sample_x2D_s = [sample_x2D_s[0][0].unsqueeze(0)]*nSamples
		self.sample_pcode_ = torch.zeros( nSamples*self.Npcode, self.Npcode )
		for iS in range( self.Npcode*nSamples ):
			ii = iS%self.Npcode
			self.sample_pcode_[iS,ii] = 1
		self.sample_z_ = torch.rand( nSamples*self.Npcode, self.Nz )

		if self.gpu_mode:
			self.sample_x2D_ = Variable(self.sample_x2D_.cuda(), volatile=True)
			self.sample_z_ = Variable(self.sample_z_.cuda(), volatile=True)
			self.sample_pcode_ = Variable(self.sample_pcode_.cuda(), volatile=True)
		else:
			self.sample_x2D_ = Variable(self.sample_x2D_, volatile=True)
			self.sample_z_ = Variable(self.sample_z_, volatile=True)
			self.sample_pcode_ = Variable(self.sample_pcode_, volatile=True)

		# networks init
		self.G = generator(self.Nid, self.Npcode, self.Nz)
		self.D = Encoder('D', self.Nid, self.Npcode)
		self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
		self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

		if self.gpu_mode:
			self.G.cuda()
			self.D.cuda()
			self.CE_loss = nn.CrossEntropyLoss().cuda()
			self.BCE_loss = nn.BCELoss().cuda()
			self.MSE_loss = nn.MSELoss().cuda()
		else:
			self.CE_loss = nn.CrossEntropyLoss()
			self.BCE_loss = nn.BCELoss()
			self.MSE_loss = nn.MSELoss()

#		print('---------- Networks architecture -------------')
#		utils.print_network(self.G)
#		utils.print_network(self.D)
#		print('-----------------------------------------------')


	def train(self):
		self.train_hist = {}
		self.train_hist['D_loss'] = []
		self.train_hist['D_loss_GAN_real'] = []
		self.train_hist['D_loss_id'] = []
		self.train_hist['D_loss_pcode'] = []
		self.train_hist['D_loss_GAN_fake'] = []
		self.train_hist['G_loss'] = []
		self.train_hist['G_loss'] = []
		self.train_hist['G_loss_GAN_fake'] = []
		self.train_hist['G_loss_id'] = []
		self.train_hist['G_loss_pcode'] = []
		self.train_hist['per_epoch_time'] = []
		self.train_hist['total_time'] = []

		if self.gpu_mode:
			self.y_real_ = Variable((torch.ones(self.batch_size,1)).cuda())
			self.y_fake_ = Variable((torch.zeros(self.batch_size,1)).cuda())
		else:
			self.y_real_ = Variable((torch.ones(self.batch_size,1)))
			self.y_fake_ = Variable((torch.zeros(self.batch_size,1)))

		#self.D.train()
		print('training start!!')
		start_time = time.time()
		for epoch in range(self.epoch):
			self.G.train()
			epoch_start_time = time.time()
			start_time_epoch = time.time()

			for iB, (sample_x3D_, sample_y_, sample_x2D_ ) in enumerate(self.data_loader):
				if iB == self.data_loader.dataset.__len__() // self.batch_size:
					break

				z_ = torch.rand((self.batch_size, self.Nz))
				x2D_ = sample_x2D_
				y_id_ = sample_y_['id']
				y_pcode_ = sample_y_['pcode']
				y_pcode_onehot_ = torch.zeros( self.batch_size, self.Npcode )
				y_pcode_onehot_.scatter_(1, y_pcode_.view(-1,1), 1)

				if self.gpu_mode:
					x2D_, z_ = Variable(x2D_.cuda()), Variable(z_.cuda())
					y_id_ = Variable( y_id_.cuda() )
					y_pcode_ = Variable(y_pcode_.cuda())
					y_pcode_onehot_ = Variable( y_pcode_onehot_.cuda() )
				else:
					x2D_, z_ = Variable(x2D_), Variable(z_)
					y_id_ = Variable(y_id_)
					y_pcode_ = Variable(y_pcode_)
					y_pcode_onehot_ = Variable( y_pcode_onehot_ )

				# update D network
				self.D_optimizer.zero_grad()

				D_GAN_real, D_id, D_pcode = self.D(x2D_)
				D_loss_GANreal = self.BCE_loss(D_GAN_real, self.y_real_)
				D_loss_real_id = self.CE_loss(D_id, y_id_)
				D_loss_real_pcode = self.CE_loss(D_pcode, y_pcode_)

				x2D_hat = self.G(x2D_, y_pcode_onehot_, z_)
				D_GAN_fake, _, _ = self.D(x2D_hat)
				D_loss_GANfake = self.BCE_loss(D_GAN_fake, self.y_fake_)

				# DRAGAN Loss (Gradient penalty)
				if self.use_GP:
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
				else:
					gradient_penalty = 0

				D_loss = D_loss_GANreal + D_loss_real_id + D_loss_real_pcode + D_loss_GANfake + gradient_penalty
				self.train_hist['D_loss'].append(D_loss.data[0])
				self.train_hist['D_loss_GAN_real'].append(D_loss_GANreal.data[0])
				self.train_hist['D_loss_id'].append(D_loss_real_id.data[0])
				self.train_hist['D_loss_pcode'].append(D_loss_real_pcode.data[0])
				self.train_hist['D_loss_GAN_fake'].append(D_loss_GANfake.data[0])

				D_loss.backward()
				self.D_optimizer.step()

				# update G network
				for iG in range(4):
					self.G_optimizer.zero_grad()
	
					x2D_hat = self.G(x2D_, y_pcode_onehot_, z_)
					D_fake_GAN, D_fake_id, D_fake_pcode = self.D(x2D_hat)
					G_loss_GANfake = self.BCE_loss(D_fake_GAN, self.y_real_)
					G_loss_id = self.CE_loss(D_fake_id, y_id_)
					G_loss_pcode = self.CE_loss(D_fake_pcode, y_pcode_)
					G_loss = G_loss_GANfake + G_loss_id + G_loss_pcode 
					if iG == 0:
						self.train_hist['G_loss'].append(G_loss.data[0])
						self.train_hist['G_loss_GAN_fake'].append(G_loss_GANfake.data[0])
						self.train_hist['G_loss_id'].append(G_loss_id.data[0])
						self.train_hist['G_loss_pcode'].append(G_loss_pcode.data[0])
	
					G_loss.backward()
					self.G_optimizer.step()
	
				if ((iB + 1) % 10) == 0:
					secs = time.time()-start_time_epoch
					hours = secs//3600
					mins = secs/60%60
					print("%2dh%2dm E:[%2d] B:[%4d/%4d] D: %.4f=%.4f+%.4f+%.4f,%.4f,\n\t\t\t G: %.4f=%.4f+%.4f+%.4f" %
						  (hours,mins, (epoch + 1), (iB + 1), self.data_loader.dataset.__len__() // self.batch_size, 
						  D_loss.data[0], D_loss_GANreal.data[0], D_loss_real_id.data[0],
						  D_loss_real_pcode.data[0], D_loss_GANfake.data[0],
						  G_loss.data[0], G_loss_GANfake.data[0], G_loss_id.data[0],
						  G_loss_pcode.data[0]) )

			self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
			self.save()
			utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)
			self.visualize_results((epoch+1))

		self.train_hist['total_time'].append(time.time() - start_time)
		print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
			  self.epoch, self.train_hist['total_time'][0]))
		print("Training finish!... save training results")

		self.save()
		utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
								 self.epoch)
		utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

	def visualize_results(self, epoch, fix=True):
		self.G.eval()

		if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
			os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

		nRows = self.nSamples2visualize
		nCols = self.Npcode

		if fix:
			""" fixed noise """
			samples = self.G(self.sample_x2D_, self.sample_pcode_, self.sample_z_ )
		else:
			""" random noise """
			if self.gpu_mode:
				sample_z_ = Variable(torch.rand((self.batch_size, self.Nz)).cuda(), volatile=True)
			else:
				sample_z_ = Variable(torch.rand((self.batch_size, self.Nz)), volatile=True)

			samples = self.G(sample_z_)

		if self.gpu_mode:
			samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
			sample_x2D_ = self.sample_x2D_.cpu().data.numpy().transpose(0, 2, 3, 1)
		else:
			samples = samples.data.numpy().transpose(0, 2, 3, 1)
			sample_x2D_ = self.sample_x2D_.data.numpy().transpose(0, 2, 3, 1)

		if epoch == 1:
			utils.save_images(sample_x2D_[:nRows*nCols, :, :, :], [nRows, nCols],
						  self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d_x' % epoch + '.png')
		utils.save_images(samples[:nRows*nCols, :, :, :], [nRows, nCols],
						  self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

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
