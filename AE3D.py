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

class Encoder3D( nn.Module ):
	def __init__( self, nInputCh=4, zdim=320 ):
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
			nn.Conv3d(512, zdim, 4, bias=False), # 4 -> 1
			nn.Sigmoid(),
			Flatten(),
		)

		utils.initialize_weights(self)

	def forward(self, input):
		x = self.conv( input )
		return x

class Decoder3D( nn.Module ):
	def __init__(self, nOutputCh=4, zdim=320):
		super(Decoder3D, self).__init__()
		self.nOutputCh = nOutputCh

		self.fc = nn.Sequential(
			nn.Linear( zdim, 320 )
		)

		self.fconv = nn.Sequential(
			Inflate(3),
			nn.ConvTranspose3d(320, 512, 4, bias=False), # 1 -> 4
			nn.BatchNorm3d(512),
			nn.ReLU(),
			nn.ConvTranspose3d(512, 256, 4, 2, 1, bias=False), # 4 -> 8
			nn.BatchNorm3d(256),
			nn.ReLU(),
			nn.ConvTranspose3d(256, 128, 4, 2, 1, bias=False), # 8 -> 16
			nn.BatchNorm3d(128),
			nn.ReLU(),
			nn.ConvTranspose3d(128, 64, 4, 2, 1, bias=False), # 16 -> 32
			nn.BatchNorm3d(64),
			nn.ReLU(),
			nn.ConvTranspose3d(64, 32, 4, 2, 1, bias=False), # 32 -> 64
			nn.BatchNorm3d(32),
			nn.ReLU(),
			nn.ConvTranspose3d(32, nOutputCh, 4, 2, 1, bias=False), # 64 -> 128
			nn.Sigmoid(),
		)
	def forward(self, fx):
#		x = self.fc( fx )
		x = self.fconv( fx )
		return x

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

		utils.initialize_weights(self)

	def forward(self, input):
		feature = self.conv(input)

		fGAN = self.convGAN( feature )

		return fGAN

class AE3D(nn.Module):
	def __init__(self):
		super(AE3D,self).__init__()

		self.Enc = Encoder3D()
		self.Dec = Decoder3D()

	def forward(self, input):
		fx = self.Enc( input )
		recon = self.Dec( fx )
		return recon

class AutoEncoder3D(object):
	def __init__(self, args):
		# parameters
		self.epoch = args.epoch
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
		self.sample_num = 16

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
			self.data_loader = DataLoader( utils.Bosphorus(data_dir, skipCodes=['YR','PR','CR'],
											transform=transforms.ToTensor(),
											shape=128, image_shape=256),
											batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
			self.Nid = 105
			self.Npcode = len(self.data_loader.dataset.posecodemap)

		# fixed samples for reconstruction visualization
		print( 'Generating fixed sample for visualization...' )
		nSamples = self.sample_num
		sample_x3D_s = []
		for iB, (sample_x3D_,  _) in enumerate(self.data_loader):
			sample_x3D_s.append( sample_x3D_ )
			if iB > nSamples // self.batch_size:
				break
		self.sample_x3D_ = torch.cat( sample_x3D_s )[:nSamples,:,:,:]

		fname = os.path.join( self.result_dir, self.dataset, self.model_name, 'sampleGT.npy')
		self.sample_x3D_.numpy().squeeze().dump( fname )

		if self.gpu_mode:
			self.sample_x3D_ = Variable(self.sample_x3D_.cuda(), volatile=True)
		else:
			self.sample_x3D_ = Variable(self.sample_x3D_, volatile=True)

		# networks init
		self.AE = AE3D()
		
		self.optimizer = optim.Adam(self.AE.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))

		if self.gpu_mode:
			self.AE.cuda()
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
		if not hasattr(self, 'train_hist') :
			self.train_hist = {}
			self.train_hist['recon_loss'] = []

			self.train_hist['per_epoch_time'] = []
			self.train_hist['total_time'] = []


		print('training start!!')
		start_time = time.time()
		if not hasattr(self, 'epoch_start'):
			self.epoch_start = 0
		for epoch in range(self.epoch_start, self.epoch):
			self.AE.train()
			epoch_start_time = time.time()
			start_time_epoch = time.time()

			for iB, (x3D_, _ ) in enumerate(self.data_loader):
				if iB == self.data_loader.dataset.__len__() // self.batch_size:
					break

				z_ = torch.rand((self.batch_size, self.Nz))

				if self.gpu_mode:
					z_ = Variable(z_.cuda())
					x3D_ = Variable(x3D_.cuda())
				else:
					z_ = Variable(z_)
					x3D_ = Variable(x3D_)

				# update network
				self.optimizer.zero_grad()

				recon = self.AE( x3D_ )
				loss_recon = self.MSE_loss( recon, x3D_)

				loss_recon.backward()
				self.optimizer.step()

				self.train_hist['recon_loss'].append(loss_recon.data[0])

				if ((iB + 1) % 10) == 0:
					secs = time.time()-start_time_epoch
					hours = secs//3600
					mins = secs/60%60
					print("%2dh%2dm E[%2d] B[%d/%d] recon losos: %.4f" %
						  (hours,mins, (epoch + 1), (iB + 1), self.data_loader.dataset.__len__() // self.batch_size, 
							loss_recon.data[0]) )
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
		self.AE.eval()

		if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
			os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

		""" fixed noise """
		recon = self.AE(self.sample_x3D_)

		if self.gpu_mode:
			recon = recon.cpu().data.numpy().squeeze()
		else:
			recon = recon.data.numpy().squeeze()

		fname_prefix = self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch

		fname = fname_prefix + '_recon.npy'
		recon.dump(fname)

	def save(self):
		save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		torch.save(self.AE.state_dict(), os.path.join(save_dir, self.model_name + '_AE.pkl'))

		with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
			pickle.dump(self.train_hist, f)

	def load(self):
		save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)
		print( 'loading from {}...'.format(save_dir) )

		self.AE.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_AE.pkl')))

		try:
			fhandle = open(os.path.join(save_dir, self.model_name + '_history.pkl'))
			self.train_hist = pickle.load(fhandle)
			fhandle.close()
			
			self.epoch_start = len(self.train_hist['per_epoch_time'])
			print( 'loaded epoch {}'.format(self.epoch_start) )
		except:
			print('history is not found and ignored')

