import numpy as np
import torch.nn as nn
import torch.optim as optim
from scipy.misc import imsave
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import utils, torch, time, os, pickle, imageio, math
from utils import Flatten, Inflate
import pdb

class Encoder2D( nn.Module ):
	def __init__( self ):
		super(Encoder2D, self).__init__()
		self.input_dim = 3

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
			Flatten(),
		)

		utils.initialize_weights(self)

	def forward(self, input):
		x = self.conv( input )
		return x


class Encoder3D( nn.Module ):
	def __init__( self, nInputCh=4 ):
		super(Encoder3D, self).__init__()
		self.nInputCh = nInputCh

		self.conv = nn.Sequential(
			nn.Conv3d(nInputCh, 32, 4, 2, 1, bias=False), # 128 -> 64
			nn.BatchNorm3d(32),
			nn.LeakyReLU(0.2),
			nn.Conv3d(32, 64, 4, 2, 1, bias=False), # 64 -> 32
			nn.BatchNorm3d(64),
			nn.LeakyReLU(0.2),
			nn.Conv3d(64, 128, 4, 2, 1, bias=False), # 32 -> 16
			nn.BatchNorm3d(128),
			nn.LeakyReLU(0.2),
			nn.Conv3d(128, 256, 4, 2, 1, bias=False), # 16 -> 8
			nn.BatchNorm3d(256),
			nn.LeakyReLU(0.2),
			nn.Conv3d(256, 512, 4, 2, 1, bias=False), # 8 -> 4
			nn.BatchNorm3d(512),
			nn.LeakyReLU(0.2),
			nn.Conv3d(512, 320, 4, bias=False), # 4 -> 1
			nn.Sigmoid(),
			Flatten(),
		)

		utils.initialize_weights(self)

	def forward(self, input):
		x = self.conv( input )
		return x


class Decoder2D( nn.Module ):
	def __init__(self, Npcode, Nz, nOutputCh=3):
		super(Decoder2D, self).__init__()
		self.nOutputCh = nOutputCh

		self.fc = nn.Sequential(
			nn.Linear( 320+Npcode+Nz, 320 )
		)

		# from DiscoGAN
		self.fconv = nn.Sequential(
			Inflate(2),
			nn.ConvTranspose2d(320, 64*32, 4, bias=False), # 1 -> 4
			nn.BatchNorm2d(64*32),
			nn.ReLU(True),
			nn.ConvTranspose2d(64*32, 64*16, 4, 2, 1, bias=False), # 4 -> 8
			nn.BatchNorm2d(64*16),
			nn.ReLU(True),
			nn.ConvTranspose2d(64*16, 64*8, 4, 2, 1, bias=False), # 8 -> 16
			nn.BatchNorm2d(64*8),
			nn.ReLU(True),
			nn.ConvTranspose2d(64*8, 64*4, 4, 2, 1, bias=False), # 16 -> 32
			nn.BatchNorm2d(64*4),
			nn.ReLU(True),
			nn.ConvTranspose2d(64*4, 64*2, 4, 2, 1, bias=False), # 32 -> 64
			nn.BatchNorm2d(64*2),
			nn.ReLU(True),
			nn.ConvTranspose2d(64*2, 64, 4, 2, 1, bias=False), # 64 -> 128
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.ConvTranspose2d(64, nOutputCh, 4, 2, 1, bias=False), # 128 -> 256
			nn.Sigmoid()
		)
	def forward(self, fx, y_pcode_onehot, z):
		feature = torch.cat((fx, y_pcode_onehot, z),1)
		x = self.fc( feature )
		x = self.fconv( x )
		return x


class Decoder3D( nn.Module ):
	def __init__(self, Npcode, Nz, nOutputCh=4):
		super(Decoder3D, self).__init__()
		self.nOutputCh = nOutputCh

		self.fc = nn.Sequential(
			nn.Linear( 320+Npcode+Nz, 320 )
		)

		self.fconv = nn.Sequential(
			Inflate(3),
			nn.ConvTranspose3d(320, 512, 4, bias=False),
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
		x = self.fconv( x )
		return x


class generator2Dto3D(nn.Module):
	def __init__(self, Nid, Npcode, Nz):
		super(generator2Dto3D, self).__init__()

		self.Genc = Encoder2D()
		self.Gdec = Decoder3D(Npcode, Nz)

		utils.initialize_weights(self)

	def forward(self, x_, y_pcode_onehot_, z_):
		fx = self.Genc( x_ )
		x_hat = self.Gdec(fx, y_pcode_onehot_, z_)

		return x_hat


class generator3Dto2D(nn.Module):
	def __init__(self, Nid, Npcode, Nz):
		super(generator3Dto2D, self).__init__()

		self.Genc = Encoder3D()
		self.Gdec = Decoder2D(Npcode, Nz)

		utils.initialize_weights(self)

	def forward(self, x_, y_pcode_onehot_, z_):
		fx = self.Genc( x_ )
		x_hat = self.Gdec(fx, y_pcode_onehot_, z_)

		return x_hat


class discriminator2D(nn.Module):
	# Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
	# Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
	def __init__(self, Nid=105, Npcode=48, nInputCh=3):
		super(discriminator2D, self).__init__()
		self.nInputCh = nInputCh

		self.conv = nn.Sequential(
			nn.Conv2d(nInputCh, 64, 11, 4, 1,bias=True), # 256 -> 64
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 128, 5, 2, 1,bias=True), # 64 -> 32
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 256, 5, 2, 1,bias=True), # 32 -> 16
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256, 512, 5, 2, 1,bias=True), # 16 -> 8
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d(512, 320, 8 , 1, 1, bias=True),
			nn.Sigmoid(),
			Flatten(),
		)

		self.fcGAN = nn.Sequential(
			nn.Linear(320, 1),
			nn.Sigmoid(),
			Flatten()
		)

		self.fcID = nn.Sequential(
			nn.Linear(320, Nid),
			Flatten()
		)

		self.fcPCode = nn.Sequential(
			nn.Linear(320, Npcode),
			Flatten()
		)
		utils.initialize_weights(self)

	def forward(self, input):
		feature = self.conv(input)

		fGAN = self.fcGAN( feature )
		fid = self.fcID( feature )
		fcode = self.fcPCode( feature )

		return fGAN, fid, fcode


class discriminator3D(nn.Module):
	# Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
	# Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
	def __init__(self, Nid=105, Npcode=48, nInputCh=4):
		super(discriminator3D, self).__init__()
		self.nInputCh = nInputCh

		self.conv = nn.Sequential(
			nn.Conv3d(nInputCh, 32, 4, 2, 1, bias=False), # 128 -> 64
			nn.BatchNorm3d(32),
			nn.LeakyReLU(0.2),
			nn.Conv3d(32, 64, 4, 2, 1, bias=False), # 64 -> 32
			nn.BatchNorm3d(64),
			nn.LeakyReLU(0.2),
			nn.Conv3d(64, 128, 4, 2, 1, bias=False), # 32 -> 16
			nn.BatchNorm3d(128),
			nn.LeakyReLU(0.2),
			nn.Conv3d(128, 256, 4, 2, 1, bias=False), # 16 -> 8
			nn.BatchNorm3d(256),
			nn.LeakyReLU(0.2),
			nn.Conv3d(256, 512, 4, 2, 1, bias=False), # 8 -> 4
			nn.BatchNorm3d(512),
			nn.LeakyReLU(0.2)
		)

		self.convGAN = nn.Sequential(
			nn.Conv3d(512, 1, 4, bias=False),
			nn.Sigmoid(),
			Flatten()
		)

		self.convID = nn.Sequential(
			nn.Conv3d(512, Nid, 4, bias=False),
			Flatten()
		)

		self.convPCode = nn.Sequential(
			nn.Conv3d(512, Npcode, 4, bias=False),
			Flatten()
		)
		utils.initialize_weights(self)

	def forward(self, input):
		feature = self.conv(input)

		fGAN = self.convGAN( feature )
		fid = self.convID( feature )
		fcode = self.convPCode( feature )

		return fGAN, fid, fcode

class DRcycleGAN3D(object):
	def __init__(self, args):
		# parameters
		self.epoch = args.epoch
		self.sample_num = 19
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

		if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
			os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)
		if not os.path.exists(os.path.join(self.save_dir, self.dataset, self.model_name)):
			os.makedirs(os.path.join(self.save_dir, self.dataset, self.model_name))

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
											shape=128, image_shape=256),
											batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
			self.Nid = 105
			self.Npcode = len(self.data_loader.dataset.posecodemap)

		# fixed samples for reconstruction visualization
		print( 'Generating fixed sample for visualization...' )
		nPcodes = self.Npcode//4
		nSamples = self.sample_num-nPcodes
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
		self.sample_x2D_ = torch.cat( sample_x2D_s )
		self.sample_x3D_ = torch.cat( sample_x3D_s )
		self.sample_pcode_ = torch.zeros( nSamples+nPcodes, self.Npcode )
		self.sample_pcode_[:nSamples,0]=1
		for iS in range( nPcodes ):
			ii = iS%self.Npcode
			self.sample_pcode_[iS+nSamples,ii] = 1
		self.sample_z_ = torch.rand( nSamples+nPcodes, self.Nz )

		fname = os.path.join( self.result_dir, self.dataset, self.model_name, 'samples.png' )
		nSpS = int(math.ceil( math.sqrt( nSamples+nPcodes ) )) # num samples per side
		utils.save_images(self.sample_x2D_[:nSpS*nSpS,:,:,:].numpy().transpose(0,2,3,1), [nSpS,nSpS],fname)

		fname = os.path.join( self.result_dir, self.dataset, self.model_name, 'sampleGT.npy')
		self.sample_x3D_.numpy().squeeze().dump( fname )

		if self.gpu_mode:
			self.sample_x2D_ = Variable(self.sample_x2D_.cuda(), volatile=True)
			self.sample_x3D_ = Variable(self.sample_x3D_.cuda(), volatile=True)
			self.sample_z_ = Variable(self.sample_z_.cuda(), volatile=True)
			self.sample_pcode_ = Variable(self.sample_pcode_.cuda(), volatile=True)
		else:
			self.sample_x2D_ = Variable(self.sample_x2D_, volatile=True)
			self.sample_x3D_ = Variable(self.sample_x3D_, volatile=True)
			self.sample_z_ = Variable(self.sample_z_, volatile=True)
			self.sample_pcode_ = Variable(self.sample_pcode_, volatile=True)

		# networks init
		self.G_2Dto3D = generator2Dto3D(self.Nid, self.Npcode, self.Nz)
		self.D_3D = discriminator3D(self.Nid, self.Npcode)
		
		self.G_2Dto3D_optimizer = optim.Adam(self.G_2Dto3D.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
		self.D_3D_optimizer = optim.Adam(self.D_3D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

		self.G_3Dto2D = generator3Dto2D(self.Nid, self.Npcode, self.Nz)
		self.D_2D = discriminator2D(self.Nid, self.Npcode)
		
		self.G_3Dto2D_optimizer = optim.Adam(self.G_3Dto2D.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
		self.D_2D_optimizer = optim.Adam(self.D_2D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

		if self.gpu_mode:
			self.G_2Dto3D.cuda()
			self.G_3Dto2D.cuda()
			self.D_3D.cuda()
			self.D_2D.cuda()
			self.CE_loss = nn.CrossEntropyLoss().cuda()
			self.BCE_loss = nn.BCELoss().cuda()
			self.MSE_loss = nn.MSELoss().cuda()
			self.L1_loss = nn.L1Loss().cuda()
		else:
			self.CE_loss = nn.CrossEntropyLoss()
			self.BCE_loss = nn.BCELoss()
			self.MSE_loss = nn.MSELoss()
			self.L1_loss = nn.L1Loss()

		print('init done')

#		print('---------- Networks architecture -------------')
#		utils.print_network(self.G)
#		utils.print_network(self.D)
#		print('-----------------------------------------------')


	def train(self):
		self.train_hist = {}
		self.train_hist['D_3D_loss'] = []
		self.train_hist['D_3D_loss_GAN_real'] = []
		self.train_hist['D_3D_loss_id'] = []
		self.train_hist['D_3D_loss_pcode'] = []
		self.train_hist['D_3D_loss_GAN_fake'] = []
		self.train_hist['D_3D_acc'] = []

		self.train_hist['D_2D_loss'] = []
		self.train_hist['D_2D_loss_GAN_real'] = []
		self.train_hist['D_2D_loss_id'] = []
		self.train_hist['D_2D_loss_pcode'] = []
		self.train_hist['D_2D_loss_GAN_fake'] = []
		self.train_hist['D_2D_acc'] = []

		self.train_hist['G_3D_loss'] = []
		self.train_hist['G_3D_loss'] = []
		self.train_hist['G_3D_loss_GAN_fake'] = []
		self.train_hist['G_3D_loss_id'] = []
		self.train_hist['G_3D_loss_pcode'] = []

		self.train_hist['G_2D_loss'] = []
		self.train_hist['G_2D_loss'] = []
		self.train_hist['G_2D_loss_GAN_fake'] = []
		self.train_hist['G_2D_loss_id'] = []
		self.train_hist['G_2D_loss_pcode'] = []

		self.train_hist['per_epoch_time'] = []
		self.train_hist['total_time'] = []

		if self.gpu_mode:
			self.y_real_ = Variable((torch.ones(self.batch_size,1)).cuda())
			self.y_fake_ = Variable((torch.zeros(self.batch_size,1)).cuda())
		else:
			self.y_real_ = Variable((torch.ones(self.batch_size,1)))
			self.y_fake_ = Variable((torch.zeros(self.batch_size,1)))

		self.D_2D.train()
		self.D_3D.train()

		print('training start!!')
		start_time = time.time()
		for epoch in range(self.epoch):
			self.G_2Dto3D.train()
			self.G_3Dto2D.train()
			epoch_start_time = time.time()
			start_time_epoch = time.time()

			for iB, (x3D_, y_, x2D_ ) in enumerate(self.data_loader):
				if iB == self.data_loader.dataset.__len__() // self.batch_size:
					break

				z_ = torch.rand((self.batch_size, self.Nz))
				y_id_ = y_['id']
				y_pcode_ = y_['pcode']
				y_pcode_onehot_ = torch.zeros( self.batch_size, self.Npcode )
				y_pcode_onehot_.scatter_(1, y_pcode_.view(-1,1), 1)
				y_random_pcode_ = torch.floor(torch.rand(self.batch_size)*self.Npcode).long()
				y_random_pcode_onehot_ = torch.zeros( self.batch_size, self.Npcode )
				y_random_pcode_onehot_.scatter_(1, y_random_pcode_.view(-1,1), 1)

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

				# update D_3D network
				self.D_3D_optimizer.zero_grad()

				D_3D_GAN_real, D_3D_id, D_3D_pcode = self.D_3D(x3D_)
				D_3D_loss_GANreal = self.BCE_loss(D_3D_GAN_real, self.y_real_)
				D_3D_loss_real_id = self.CE_loss(D_3D_id, y_id_)
				D_3D_loss_real_pcode = self.CE_loss(D_3D_pcode, y_pcode_)

				x3D_hat = self.G_2Dto3D(x2D_, y_random_pcode_onehot_, z_)
				D_3D_GAN_fake, _, _ = self.D_3D(x3D_hat)
				D_3D_loss_GANfake = self.BCE_loss(D_3D_GAN_fake, self.y_fake_)

				num_correct_real = torch.sum(D_3D_GAN_real>0.5)
				num_correct_fake = torch.sum(D_3D_GAN_fake<0.5)
				D_3D_acc = float(num_correct_real.data[0] + num_correct_fake.data[0]) / (self.batch_size*2)

				D_3D_loss = D_3D_loss_GANreal + D_3D_loss_real_id + D_3D_loss_real_pcode + D_3D_loss_GANfake

				D_3D_loss.backward()
				if D_3D_acc < 0.8:
					self.D_3D_optimizer.step()

				self.train_hist['D_3D_loss'].append(D_3D_loss.data[0])
				self.train_hist['D_3D_loss_GAN_real'].append(D_3D_loss_GANreal.data[0])
				self.train_hist['D_3D_loss_id'].append(D_3D_loss_real_id.data[0])
				self.train_hist['D_3D_loss_pcode'].append(D_3D_loss_real_pcode.data[0])
				self.train_hist['D_3D_loss_GAN_fake'].append(D_3D_loss_GANfake.data[0])
				self.train_hist['D_3D_acc'].append(D_3D_acc)


				# update D_2D network
				self.D_2D_optimizer.zero_grad()

				D_2D_GAN_real, D_2D_id, D_2D_pcode = self.D_2D(x2D_)
				D_2D_loss_GANreal = self.BCE_loss(D_2D_GAN_real, self.y_real_)
				D_2D_loss_real_id = self.CE_loss(D_2D_id, y_id_)
				D_2D_loss_real_pcode = self.CE_loss(D_2D_pcode, y_pcode_)

				x2D_hat = self.G_3Dto2D(x3D_, y_random_pcode_onehot_, z_)
				D_2D_GAN_fake, _, _ = self.D_2D(x2D_hat)
				D_2D_loss_GANfake = self.BCE_loss(D_2D_GAN_fake, self.y_fake_)

				num_correct_real = torch.sum(D_2D_GAN_real>0.5)
				num_correct_fake = torch.sum(D_2D_GAN_fake<0.5)
				D_2D_acc = float(num_correct_real.data[0] + num_correct_fake.data[0]) / (self.batch_size*2)

				D_2D_loss = D_2D_loss_GANreal + D_2D_loss_real_id + D_2D_loss_real_pcode + D_2D_loss_GANfake

				D_2D_loss.backward()
				if D_2D_acc < 0.8:
					self.D_2D_optimizer.step()

				self.train_hist['D_2D_loss'].append(D_2D_loss.data[0])
				self.train_hist['D_2D_loss_GAN_real'].append(D_2D_loss_GANreal.data[0])
				self.train_hist['D_2D_loss_id'].append(D_2D_loss_real_id.data[0])
				self.train_hist['D_2D_loss_pcode'].append(D_2D_loss_real_pcode.data[0])
				self.train_hist['D_2D_loss_GAN_fake'].append(D_2D_loss_GANfake.data[0])
				self.train_hist['D_2D_acc'].append(D_2D_acc)


				# update G_2Dto3D and G_3Dto2D network
				for iG in range(4):
					self.G_2Dto3D_optimizer.zero_grad()
					self.G_3Dto2D_optimizer.zero_grad()
	
					# simple GAN loss
					x3D_hat = self.G_2Dto3D(x2D_, y_random_pcode_onehot_, z_)
					D_3D_fake_GAN, D_3D_fake_id, D_3D_fake_pcode = self.D_3D(x3D_hat)
					G_3D_loss_GANfake = self.BCE_loss(D_3D_fake_GAN, self.y_real_)
					G_3D_loss_id = self.CE_loss(D_3D_fake_id, y_id_)
					G_3D_loss_pcode = self.CE_loss(D_3D_fake_pcode, y_random_pcode_)
					G_3D_loss = G_3D_loss_GANfake + G_3D_loss_id + G_3D_loss_pcode 

					x2D_hat = self.G_3Dto2D(x3D_, y_random_pcode_onehot_, z_)
					D_2D_fake_GAN, D_2D_fake_id, D_2D_fake_pcode = self.D_2D(x2D_hat)
					G_2D_loss_GANfake = self.BCE_loss(D_2D_fake_GAN, self.y_real_)
					G_2D_loss_id = self.CE_loss(D_2D_fake_id, y_id_)
					G_2D_loss_pcode = self.CE_loss(D_2D_fake_pcode, y_random_pcode_)
					G_2D_loss = G_2D_loss_GANfake + G_2D_loss_id + G_2D_loss_pcode

					x2D_recon = self.G_3Dto2D(x3D_hat, y_pcode_onehot_, z_)
#					D_cycle_2D_GAN, D_cycle_2D_id, D_cycle_2D_pcode = self.D_2D( x2D_recon )
					loss_recon2D = self.L1_loss(x2D_recon, x2D_)
#					loss_id = self.CE_loss( D_cycle_2D_id, y_id_ )
#					loss_pose = self.CE_loss( D_cycle_2D_pcode, y_pcode_ )
#					G_3Dto2D_cycle_loss = loss_recon + loss_id + loss_pose

					x3D_recon = self.G_2Dto3D(x2D_hat, y_pcode_onehot_, z_)
#					D_cycle_3D_GAN, D_cycle_3D_id, D_cycle_3D_pcode = self.D_3D( x3D_recon )
					loss_recon3D = self.MSE_loss(x3D_recon, x3D_)
#					loss_id = self.CE_loss( D_cycle_3D_id, y_id_ )
#					loss_pose = self.CE_loss( D_cycle_3D_pcode, y_pcode_ )
#					G_2Dto3D_cycle_loss = loss_recon + loss_id + loss_pose

					if iG == 0:
						self.train_hist['G_3D_loss'].append(G_3D_loss.data[0])
						self.train_hist['G_3D_loss_GAN_fake'].append(G_3D_loss_GANfake.data[0])
						self.train_hist['G_3D_loss_id'].append(G_3D_loss_id.data[0])
						self.train_hist['G_3D_loss_pcode'].append(G_3D_loss_pcode.data[0])
						self.train_hist['G_2D_loss'].append(G_2D_loss.data[0])
						self.train_hist['G_2D_loss_GAN_fake'].append(G_2D_loss_GANfake.data[0])
						self.train_hist['G_2D_loss_id'].append(G_2D_loss_id.data[0])
						self.train_hist['G_2D_loss_pcode'].append(G_2D_loss_pcode.data[0])
	
					G_loss = G_2D_loss + G_3D_loss + 10*loss_recon2D + 10*loss_recon3D
					G_loss.backward()

					self.G_2Dto3D_optimizer.step()
					self.G_3Dto2D_optimizer.step()
	
				if ((iB + 1) % 10) == 0:
					secs = time.time()-start_time_epoch
					hours = secs//3600
					mins = secs/60%60
					print("%2dh%2dm E[%2d] B[%d/%d] D2: %.4f, D3: %.4f, G: %.4f, D_acc:%.4f, %.4f" %
						  (hours,mins, (epoch + 1), (iB + 1), self.data_loader.dataset.__len__() // self.batch_size, 
						  D_2D_loss.data[0], D_3D_loss.data[0], G_loss.data[0], D_2D_acc, D_3D_acc) )
					utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

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
		self.G_2Dto3D.eval()
		self.G_3Dto2D.eval()

		if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
			os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

		nRows = int( math.ceil( math.sqrt( self.sample_num) ) )
		nCols = nRows

		""" fixed noise """
		samples_3D = self.G_2Dto3D(self.sample_x2D_, self.sample_pcode_, self.sample_z_ )
		samples_2D = self.G_3Dto2D(self.sample_x3D_, self.sample_pcode_, self.sample_z_ )

		recon_3D = self.G_2Dto3D(samples_2D, self.sample_pcode_, self.sample_z_ )
		recon_2D = self.G_3Dto2D(samples_3D, self.sample_pcode_, self.sample_z_ )

		if self.gpu_mode:
			samples_3D = samples_3D.cpu().data.numpy().squeeze()
			samples_2D = samples_2D.cpu().data.numpy().transpose(0,2,3,1)
			recon_3D = recon_3D.cpu().data.numpy().squeeze()
			recon_2D = recon_2D.cpu().data.numpy().transpose(0,2,3,1)
		else:
			samples_3D = samples_3D.data.numpy().squeeze()
			samples_2D = samples_2D.data.numpy().transpose(0,2,3,1)
			recon_3D = recon_3D.data.numpy().squeeze()
			recon_2D = recon_2D.data.numpy().transpose(0,2,3,1)

		fname_prefix = self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch

		fname = fname_prefix + '.npy'
		samples_3D.dump(fname)

		fname = fname_prefix + '_recon.npy'
		recon_3D.dump(fname)

		fname = fname_prefix + '.png'
		utils.save_images(samples_2D[:nRows*nCols,:,:,:], [nRows, nCols],fname)

		fname = fname_prefix + '_recon.png'
		utils.save_images(recon_2D[:nRows*nCols,:,:,:], [nRows, nCols],fname)

	def save(self):
		save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		torch.save(self.G_2Dto3D.state_dict(), os.path.join(save_dir, self.model_name + '_G_2Dto3D.pkl'))
		torch.save(self.G_3Dto2D.state_dict(), os.path.join(save_dir, self.model_name + '_G_3Dto2D.pkl'))
		torch.save(self.D_2D.state_dict(), os.path.join(save_dir, self.model_name + '_D_2D.pkl'))
		torch.save(self.D_3D.state_dict(), os.path.join(save_dir, self.model_name + '_D_3D.pkl'))

		with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
			pickle.dump(self.train_hist, f)

	def load(self):
		save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)
		print( 'loading from {}...'.format(save_dir) )

		self.G_2Dto3D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G_2Dto3D.pkl')))
		self.G_3Dto2D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G_3Dto2D.pkl')))
		self.D_2D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D_2D.pkl')))
		self.D_3D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D_3D.pkl')))

		try:
			fhandle = open(os.path.join(save_dir, self.model_name + '_history.pkl'))
			self.train_hist = pickle.load(fhandle)
			fhandle.close()
			
			self.epoch_start = len(self.train_hist['per_epoch_time'])
			print( 'loaded epoch {}'.format(self.epoch_start) )
		except:
			print('history is not found and ignored')

