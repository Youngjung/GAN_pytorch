import utils, torch, time, os, pickle
from scipy.misc import imsave
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import pdb

class Encoder( nn.Module ):
	def __init__( self, name, Nd, Np, Ni=0 ):
		super(Encoder, self).__init__()
		self.input_height = 100
		self.input_width = 100
		self.input_dim = 1
		self.name = name
		self.Nd = Nd
		self.Np = Np
		self.Ni = Ni

# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
# torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True)
# torch.nn.ELU(alpha=1.0, inplace=False)
# torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
# torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
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

		self.fc_id = nn.Linear( 320, Nd+1 )
		#self.fc_pose = nn.Linear( 320, Np )
		#self.fc_illum = nn.Linear( 320, Ni )

		utils.initialize_weights(self)

	def forward(self, input):
		x = self.conv( input )
		if self.name == 'D':
			x_flat = x.view(x.size(0),-1)
			fid = self.fc_id( x_flat )
			fpose = None
			fillum = None
			#fpose = self.fc_pose( x_flat )
			#fillum = self.fc_illum( x_flat )
			x = ( fid, fpose, fillum )
		return x

class Decoder( nn.Module ):
	def __init__(self, Nz, Np, Ni):
		super(Decoder, self).__init__()

		self.fc = nn.Sequential(
			nn.Linear( 320+Np+Ni+Nz, 6*6*320 )
		)

		self.fconv = nn.Sequential(
			# FConv52, FConv51
			nn.ConvTranspose2d( 320, 160, 3, 1, 1 ), 
			nn.ConvTranspose2d( 160, 256, 3, 1, 1 ), 

			# FConv43, FConv42, FConv41
			nn.ConvTranspose2d( 256, 256, 3, 2, 1, output_padding=1 ), 
			nn.ConvTranspose2d( 256, 128, 3, 1, 1 ), 
			nn.ConvTranspose2d( 128, 192, 3, 1, 1 ), 

			# FConv33, FConv32, FConv31
			nn.ConvTranspose2d( 192, 192, 3, 2, 1, output_padding=1 ), 
			nn.ConvTranspose2d( 192, 96, 3, 1, 1 ), 
			nn.ConvTranspose2d( 96, 128, 3, 1, 1 ), 

			# FConv23, FConv22, FConv21
			nn.ConvTranspose2d( 128, 128, 3, 2, 1, output_padding=1 ), 
			nn.ConvTranspose2d( 128, 64, 3, 1, 1 ), 
			nn.ConvTranspose2d( 64, 64, 3, 1, 1 ), 

			# FConv13, FConv12, FConv11
			nn.ConvTranspose2d( 64, 64, 3, 2, 1, output_padding=1 ), 
			nn.ConvTranspose2d( 64, 32, 3, 1, 1 ), 
			nn.ConvTranspose2d( 32, 1, 3, 1, 1 )
		)
	def forward(self, fid, fpose, fillum, z):
		# feature = torch.cat((fx, y_pose_onehot_, y_illum_onehot_, z_),1)
		feature = torch.cat((fid, z),1)
		x = self.fc( feature )
		x = x.view(-1,320,6,6)
		x = self.fconv( x )
		return x


class generator(nn.Module):
	# Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
	# Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
	def __init__(self, Nz, Nd, Np, Ni=0):
		super(generator, self).__init__()

		self.Genc = Encoder('Genc', Nd, Np, Ni)
		self.Gdec = Decoder(Nz, Np, Ni)

		utils.initialize_weights(self)

	def forward(self, x_, y_pose_onehot_, y_illum_onehot_, z_):
		fx = self.Genc( x_ )
		fx = fx.view(-1, 320)
		x_hat = self.Gdec(fx, y_pose_onehot_, y_illum_onehot_, z_)

		return x_hat

class DRGAN(object):
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
		self.model_name = args.gan_type

		if self.dataset == 'MultiPie' or self.dataset == 'miniPie':
			self.Nd = 346 # 200
			self.Np = 9
			self.Ni = 20
			self.Nz = 50
		elif self.dataset == 'CASIA-WebFace':
			self.Nd = 10885 
			self.Np = 13
			self.Ni = 20
			self.Nz = 50


		# networks init
		# self.G = generator(self.Nz, self.Nd, self.Np, self.Ni)
		self.G = generator(self.Nz,self.Nd,0,0)
		self.D = Encoder('D', self.Nd, self.Np, self.Ni)
		self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
		self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

		if self.gpu_mode:
			self.G.cuda()
			self.D.cuda()
			self.CE_loss = nn.CrossEntropyLoss().cuda()
			self.MSE_loss = nn.MSELoss().cuda()
		else:
			self.CE_loss = nn.CrossEntropyLoss()
			self.MSE_loss = nn.MSELoss()

#		print('---------- Networks architecture -------------')
#		utils.print_network(self.G)
#		utils.print_network(self.D)
#		print('-----------------------------------------------')

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
				[transforms.Scale(100), transforms.RandomCrop(96), transforms.ToTensor()]),
				cam_ids=[51]),
				batch_size=self.batch_size, shuffle=True) 
		elif self.dataset == 'CASIA-WebFace':
			self.data_loader = utils.CustomDataLoader(data_dir, transform=transforms.Compose(
				[transforms.Scale(100), transforms.RandomCrop(96), transforms.ToTensor()]), batch_size=self.batch_size,
												 shuffle=True)

		for iB, (sample_x_,sample_y_) in enumerate(self.data_loader):
			self.sample_x_ = sample_x_
			break
		self.sample_z_ = torch.rand( self.batch_size, self.Nz )
		if self.gpu_mode:
			self.sample_x_ = Variable(self.sample_x_.cuda(), volatile=True)
			self.sample_z_ = Variable(self.sample_z_.cuda(), volatile=True)
		else:
			self.sample_x_ = Variable(self.sample_x_, volatile=True)
			self.sample_z_ = Variable(self.sample_z_, volatile=True)


		# fixed noise
		#sample_x_ = self.data_loader.dataset[0][0].unsqueeze(0)
		#self.sample_x_pose_ = sample_x_.repeat( self.sample_num*self.Np,1,1,1 )
		#self.sample_x_illum_ = sample_x_.repeat( self.sample_num*self.Ni,1,1,1 )
		#sample_z_source_ = torch.rand( self.sample_num, self.Nz )
		#self.sample_z_pose_ = torch.zeros( self.sample_num*self.Np, self.Nz )
		#self.sample_z_illum_ = torch.zeros( self.sample_num*self.Ni, self.Nz )

		#for i in range(self.sample_num):
		#	self.sample_z_pose_[i*self.Np:(i+1)*self.Np] = sample_z_source_[i].repeat(self.Np,1)
		#	self.sample_z_illum_[i*self.Ni:(i+1)*self.Ni] = sample_z_source_[i].repeat(self.Ni,1)
		#sample_pose = torch.LongTensor( range(self.Np) ).repeat(self.sample_num)
		#sample_illum = torch.LongTensor( range(self.Ni) ).repeat(self.sample_num)
		#sample_pose_fixed_ = (torch.ones( self.sample_num*self.Ni )*4).long()
		#sample_illum_fixed_ = (torch.ones( self.sample_num*self.Np )*10).long()
		#self.sample_pose_ = torch.zeros( self.sample_num*self.Np, self.Np )
		#self.sample_illum_ = torch.zeros( self.sample_num*self.Ni, self.Ni )
		#self.sample_pose_fixed_ = torch.zeros( self.sample_num*self.Ni, self.Np )
		#self.sample_illum_fixed_ = torch.zeros( self.sample_num*self.Np, self.Ni )
		#self.sample_pose_.scatter_(1, sample_pose.view(-1,1), 1)
		#self.sample_illum_.scatter_(1, sample_illum.view(-1,1), 1)
		#self.sample_pose_fixed_.scatter_(1, sample_pose_fixed_.view(-1,1), 1)
		#self.sample_illum_fixed_.scatter_(1, sample_illum_fixed_.view(-1,1), 1)
		#if self.gpu_mode:
		#	self.sample_x_pose_ = Variable( self.sample_x_pose_.cuda(), volatile=True )
		#	self.sample_x_illum_ = Variable( self.sample_x_illum_.cuda(), volatile=True )
		#	self.sample_pose_ = Variable( self.sample_pose_.cuda(), volatile=True )
		#	self.sample_pose_fixed_ = Variable( self.sample_pose_fixed_.cuda(), volatile=True )
		#	self.sample_illum_ = Variable( self.sample_illum_.cuda(), volatile=True )
		#	self.sample_illum_fixed_ = Variable( self.sample_illum_fixed_.cuda(), volatile=True )
		#	self.sample_z_pose_ = Variable(self.sample_z_pose_.cuda(), volatile=True)
		#	self.sample_z_illum_ = Variable(self.sample_z_illum_.cuda(), volatile=True)
		#else:
		#	self.sample_x_pose_ = Variable( self.sample_x_pose_, volatile=True )
		#	self.sample_x_illum_ = Variable( self.sample_x_illum_, volatile=True )
		#	self.sample_pose_ = Variable( self.sample_pose_, volatile=True )
		#	self.sample_pose_fixed_ = Variable( self.sample_pose_fixed_, volatile=True )
		#	self.sample_illum_ = Variable( self.sample_illum_, volatile=True )
		#	self.sample_illum_fixed_ = Variable( self.sample_illum_fixed_, volatile=True )
		#	self.sample_z_pose_ = Variable(self.sample_z_pose_, volatile=True)
		#	self.sample_z_illum_ = Variable(self.sample_z_illum_, volatile=True)

	def train(self):
		self.train_hist = {}
		self.train_hist['D_loss'] = []
		self.train_hist['G_loss'] = []
		self.train_hist['per_epoch_time'] = []
		self.train_hist['total_time'] = []

		if self.gpu_mode:
			self.y_fake_ = Variable((torch.ones(self.batch_size)*self.Nd).long().cuda())
		else:
			self.y_fake_ = Variable((torch.ones(self.batch_size)*self.Nd).long())

		#self.D.train()
		print('training start!!')
		start_time = time.time()
		for epoch in range(self.epoch):
			self.G.train()
			epoch_start_time = time.time()
			for iB, (sample_x_,sample_y_) in enumerate(self.data_loader):
				if iB == self.data_loader.dataset.__len__() // self.batch_size:
					break

				z_ = torch.rand((self.batch_size, self.Nz))
				#x_ = (sample_x_*2-255)/255
				x_ = sample_x_
				y_id_ = sample_y_['id']
				y_pose_ = sample_y_['pose']
				y_illum_ = sample_y_['illum']
				y_pose_onehot_ = torch.zeros( self.batch_size, self.Np )
				y_pose_onehot_.scatter_(1, y_pose_.view(-1,1), 1)
				y_illum_onehot_ = torch.zeros( self.batch_size, self.Ni )
				y_illum_onehot_.scatter_(1, y_illum_.view(-1,1), 1)

				if self.gpu_mode:
					x_, z_ = Variable(x_.cuda()), Variable(z_.cuda())
					y_id_ = Variable( y_id_.cuda() )
					y_pose_, y_illum_ = Variable(y_pose_.cuda()), Variable(y_illum_.cuda())
					y_pose_onehot_ = Variable( y_pose_onehot_.cuda() )
					y_illum_onehot_ = Variable( y_illum_onehot_.cuda() )
				else:
					x_, z_ = Variable(x_), Variable(z_)
					y_id_ = Variable(y_id_)
					y_pose_, y_illum_ = Variable(y_pose_), Variable(y_illum_)
					y_pose_onehot_ = Variable( y_pose_onehot_ )
					y_illum_onehot_ = Variable( y_illum_onehot_ )

				# update D network
				self.D_optimizer.zero_grad()

				D_id, D_pose, D_illum = self.D(x_)
				D_real_loss_id = self.CE_loss(D_id, y_id_)
				#D_real_loss_pose = self.CE_loss(D_pose, y_pose_)
				#D_real_loss_illum = self.CE_loss(D_illum, y_illum_)

				x_hat = self.G(x_, y_pose_onehot_, y_illum_onehot_, z_)
				D_fake, _, _ = self.D(x_hat)
				D_fake_loss = self.CE_loss(D_fake, self.y_fake_)

				#D_loss = D_real_loss_id + D_real_loss_pose + D_real_loss_illum + D_fake_loss
				D_loss = D_real_loss_id + D_fake_loss
				self.train_hist['D_loss'].append(D_loss.data[0])

				D_loss.backward()
				self.D_optimizer.step()

				# update G network
				for iG in range(4):
					self.G_optimizer.zero_grad()
	
					x_hat = self.G(x_, y_pose_onehot_, y_illum_onehot_, z_)
					#G_loss = self.MSE_loss( x_hat, x_ )
					D_fake_id, D_fake_pose, D_fake_illum = self.D(x_hat)
					D_fake_loss_id = self.CE_loss(D_fake_id, y_id_)
					#D_fake_loss_pose = self.CE_loss(D_fake_pose, y_pose_)
					#D_fake_loss_illum = self.CE_loss(D_fake_illum, y_illum_)
					#G_loss = D_fake_loss_id + D_fake_loss_pose + D_fake_loss_illum
					G_loss = D_fake_loss_id
					if iG == 0:
						self.train_hist['G_loss'].append(G_loss.data[0])
	
					G_loss.backward()
					self.G_optimizer.step()
	
				if ((iB + 1) % 10) == 0:
					print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
						  ((epoch + 1), (iB + 1), self.data_loader.dataset.__len__() // self.batch_size, 
						  D_loss.data[0], G_loss.data[0]))

			self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
			self.visualize_results((epoch+1))
			utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

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

		image_frame_width = self.sample_num
		image_frame_height = self.Np

		if fix:
			""" fixed noise """
			samples = self.G(self.sample_x_, None, None, self.sample_z_ )
			#samples = self.G(self.sample_x_pose_, self.sample_pose_, self.sample_illum_fixed_, self.sample_z_pose_)
		else:
			""" random noise """
			if self.gpu_mode:
				sample_z_ = Variable(torch.rand((self.batch_size, self.Nz)).cuda(), volatile=True)
			else:
				sample_z_ = Variable(torch.rand((self.batch_size, self.Nz)), volatile=True)

			samples = self.G(sample_z_)

		if self.gpu_mode:
			samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
		else:
			samples = samples.data.numpy().transpose(0, 2, 3, 1)

		utils.save_images(samples[:image_frame_width* image_frame_height, :, :, :], [image_frame_width, image_frame_height],
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
