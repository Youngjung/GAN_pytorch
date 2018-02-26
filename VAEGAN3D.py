import utils, torch, time, os, pickle, imageio, math
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils3D.visualize import plot_voxel
import pdb

def Gaussian_distribution(vector):
	# input : Variable (batch_size x 400 x 1 x 1)
	# output : two Variables (batch_size x 200)
	zdim = vector.size(1)/2
	mean = vector[:, :zdim, 0,0]
	variance = vector[:, zdim:, 0,0]
	return mean, variance

class Encoder(nn.Module):
	def __init__(self, Nz=200, nInputChannels=3):
		super(Encoder,self).__init__()
		self.nInputChannels = nInputChannels
		self.Nz = Nz

		self.conv = nn.Sequential(
			nn.Conv2d(self.nInputChannels, 64, 11, 4, 1,bias=True),
			nn.BatchNorm3d(64),
			nn.ReLU(),
			nn.Conv2d(64, 128, 5, 2, 1,bias=True),
			nn.BatchNorm3d(128),
			nn.ReLU(),
			nn.Conv2d(128, 256, 5, 2, 1,bias=True),
			nn.BatchNorm3d(256),
			nn.ReLU(),
			nn.Conv2d(256, 512, 5, 2, 1,bias=True),
			nn.BatchNorm3d(512),
			nn.ReLU(),
			nn.Conv2d(512, self.Nz*2, 8 , 1, 1, bias=True),
			nn.Sigmoid(),
		)
		utils.initialize_weights(self)

	def forward(self,input):
		x = self.conv(input)
		return x


class generator(nn.Module):
	# Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
	# Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
	def __init__(self, Nz=200, nOutputChannels=4 ):
		super(generator, self).__init__()
		self.Nz = Nz
		self.nOutputChannels = nOutputChannels

		self.deconv = nn.Sequential(
			nn.ConvTranspose3d(Nz, 512, 4, bias=False),
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
			nn.ConvTranspose3d(32, self.nOutputChannels, 4, 2, 1, bias=False),
			nn.Sigmoid(),
		)
		utils.initialize_weights(self)

	def forward(self, x):
		x = x.view(-1, self.Nz, 1,1,1 )
		x = self.deconv(x)

		return x

class discriminator(nn.Module):
	# Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
	# Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
	def __init__(self, nOutputChannels=4):
		super(discriminator, self).__init__()
		self.nOutputChannels = nOutputChannels

		self.conv = nn.Sequential(
			nn.Conv3d(self.nOutputChannels, 32, 4, 2, 1, bias=False),
			nn.BatchNorm3d(32),
			nn.LeakyReLU(0.2),
			nn.Conv3d(32, 64, 4, 2, 1, bias=False),
			nn.BatchNorm3d(64),
			nn.LeakyReLU(0.2),
			nn.Conv3d(64, 128, 4, 2, 1, bias=False),
			nn.BatchNorm3d(128),
			nn.LeakyReLU(0.2),
			nn.Conv3d(128, 256, 4, 2, 1, bias=False),
			nn.BatchNorm3d(256),
			nn.LeakyReLU(0.2),
			nn.Conv3d(256, 512, 4, 2, 1, bias=False),
			nn.BatchNorm3d(512),
			nn.LeakyReLU(0.2),
			nn.Conv3d(512, 1, 4, bias=False),
			nn.Sigmoid(),
		)
		utils.initialize_weights(self)

	def forward(self, input):
		x = self.conv(input)
		x = x.squeeze(4).squeeze(3).squeeze(2)

		return x


class VAEGAN3D(object):
	def __init__(self, args):
		# parameters
		self.epoch = args.epoch
		self.batch_size = args.batch_size
		self.sample_num = 49 
		self.test_sample_size = min(args.test_sample_size, args.batch_size)
		self.save_dir = args.save_dir
		self.result_dir = args.result_dir
		self.dataset = args.dataset
		self.dataroot_dir = args.dataroot_dir
		self.log_dir = args.log_dir
		self.gpu_mode = args.gpu_mode
		self.model_name = args.gan_type
		self.num_workers = args.num_workers
		self.centerBosphorus = args.centerBosphorus
		if len(args.comment) > 0:
			self.model_name = self.model_name + '_' + args.comment
		self.lambda_ = 0.25
		self.D_threshold = 0.8

		self.alpha1 = 5
		self.alpha2 = 0.0001

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
		if self.dataset == 'ShapeNet':
			self.data_loader = DataLoader( utils.ShapeNet(data_dir,synsetId=args.synsetId),
											batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
		elif self.dataset == 'Bosphorus':
			self.data_loader = DataLoader( utils.Bosphorus(data_dir, use_image=True, skipCodes=['YR','PR','CR'],
											transform=transforms.ToTensor(),
											shape=128, image_shape=256, center=self.centerBosphorus),
											batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
			self.Nid = 105
			self.Npcode = len(self.data_loader.dataset.posecodemap)
			self.Nz = 50
			self.nInputChannels = 3
			self.nOutputChannels = 4
		elif self.dataset == 'IKEA':
			self.transform = transforms.Compose([transforms.Scale((256, 256)), transforms.ToTensor()])
			self.data_loader =DataLoader(utils.IKEA(IKEA_data_dir, transform=self.transform), 
										 batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
		else:
			exit("unknown dataset: " + self.dataset)


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
			self.sample_x2D_ = torch.cat( sample_x2D_s )
			self.sample_x3D_ = torch.cat( sample_x3D_s )
			self.sample_pcode_ = torch.zeros( nSamples+nPcodes, self.Npcode )
			self.sample_pcode_[:nSamples,0]=1
			for iS in range( nPcodes ):
				ii = iS%self.Npcode
				self.sample_pcode_[iS+nSamples,ii] = 1
			self.sample_z_ = torch.rand( nSamples+nPcodes, self.Nz )
	
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
		elif args.interpolate or args.generate:
			print( 'skipping fixed sample for visualization...: interpolate/generate' )
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

		if not args.interpolate and not args.generate:
			if self.gpu_mode:
				self.sample_x2D_ = Variable(self.sample_x2D_.cuda(), volatile=True)
				self.sample_z_ = Variable(self.sample_z_.cuda(), volatile=True)
				self.sample_pcode_ = Variable(self.sample_pcode_.cuda(), volatile=True)
			else:
				self.sample_x2D_ = Variable(self.sample_x2D_, volatile=True)
				self.sample_z_ = Variable(self.sample_z_, volatile=True)
				self.sample_pcode_ = Variable(self.sample_pcode_, volatile=True)

		# networks init
		self.G = generator( self.Nz, self.nOutputChannels )
		self.D = discriminator( )
		self.Enc = Encoder( self.Nz, self.nInputChannels )
		self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
		self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
		self.Enc_optimizer = optim.Adam(self.Enc.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

		if self.gpu_mode:
			self.G.cuda()
			self.D.cuda()
			self.Enc.cuda()
			self.BCE_loss = nn.BCELoss().cuda()
			self.KL_loss = nn.KLDivLoss().cuda()
			self.MSE_loss = nn.MSELoss().cuda()
		else:
			self.BCE_loss = nn.BCELoss()
			self.KL_loss = nn.KLDivLoss()
			self.MSE_loss = nn.MSELoss()

#		print('---------- Networks architecture -------------')
#		utils.print_network(self.G)
#		utils.print_network(self.D)
#		print('-----------------------------------------------')



	def train(self):
		if not hasattr(self, 'train_hist') :
			self.train_hist = {}
			self.train_hist['D_loss'] = []
			self.train_hist['E_loss'] = []
			self.train_hist['G_loss'] = []
			self.train_hist['D_acc'] = []
			self.train_hist['per_epoch_time'] = []
			self.train_hist['total_time'] = 0

		if self.gpu_mode:
			self.y_real_ = Variable(torch.ones(self.batch_size, 1).cuda())
			self.y_fake_ = Variable(torch.zeros(self.batch_size, 1).cuda())
		else:
			self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(torch.zeros(self.batch_size, 1))

		self.D.train()
		print('training start!!')
		start_time = time.time()
		if not hasattr(self, 'epoch_start'):
			self.epoch_start = 0
		for epoch in range(self.epoch_start, self.epoch):
			self.G.train()
			epoch_start_time = time.time()
			start_time_epoch = time.time()

			for iB, (x_, _, y_) in enumerate(self.data_loader):
				if iB == self.data_loader.dataset.__len__() // self.batch_size:
					break

				z_ = torch.normal( torch.zeros(self.batch_size, self.Nz), torch.ones(self.batch_size,self.Nz) )
				if self.gpu_mode:
					x_ = Variable(x_.cuda())
					y_ = Variable(y_.cuda())
					z_ = Variable(z_.cuda())
				else:
					x_ = Variable(x_)
					y_ = Variable(y_)
					z_ = Variable(z_)

				
				# update D network
				self.D_optimizer.zero_grad()

				D_real = self.D(x_)
				D_real_loss = self.BCE_loss(D_real, self.y_real_)
				num_correct_real = torch.sum(D_real>0.5)

				G_ = self.G(z_)
				D_fake = self.D(G_)
				D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)
				num_correct_fake = torch.sum(D_fake<0.5)

				D_loss = D_real_loss + D_fake_loss
				D_loss.backward()
				self.train_hist['D_loss'].append(D_loss.data[0])

				# D gets updated only if its accuracy is below 80%
				D_acc = float(num_correct_real.data[0] + num_correct_fake.data[0]) / (self.batch_size*2)
				self.train_hist['D_acc'].append(D_acc)
				if D_acc < self.D_threshold:
					self.D_optimizer.step()

				# update Enc network
				self.Enc_optimizer.zero_grad()

				temp = self.Enc(y_)
				mu, sigma= Gaussian_distribution(temp)
				reparamZ_ = torch.normal( torch.zeros(self.batch_size, self.Nz), torch.ones(self.batch_size,self.Nz) )
				if self.gpu_mode:
					reparamZ_ = Variable(reparamZ_.cuda())
				else:
					reparamZ_ = Variable(reparamZ_)

				zey_ = mu + reparamZ_*sigma
				Gey_ = self.G(zey_)

				KL_div = 0.5 * torch.sum(mu**2 + sigma**2 - torch.log(1e-8 + sigma**2)-1) / self.batch_size
				E_loss_MSE = self.MSE_loss( Gey_, x_ )
				E_loss = KL_div*self.alpha1 + E_loss_MSE*self.alpha2
				E_loss.backward()
				self.train_hist['E_loss'].append(E_loss.data[0])
				self.Enc_optimizer.step()

				# update G network
				self.G_optimizer.zero_grad()

				G_ = self.G(z_)
				D_fake = self.D(G_)

				temp = self.Enc(y_)
				mu, sigma= Gaussian_distribution(temp)
				reparamZ_ = torch.normal( torch.zeros(self.batch_size, self.Nz), torch.ones(self.batch_size,self.Nz) )
				if self.gpu_mode:
					reparamZ_ = Variable(reparamZ_.cuda())
				else:
					reparamZ_ = Variable(reparamZ_)

				zey_ = mu + reparamZ_*sigma
				Gey_ = self.G(zey_)

				G_loss_GAN = self.BCE_loss(D_fake, self.y_real_)
				G_loss_MSE = self.MSE_loss(Gey_, x_)
				G_loss = G_loss_GAN + G_loss_MSE*self.alpha2
				self.train_hist['G_loss'].append(G_loss.data[0])

				G_loss.backward()
				self.G_optimizer.step()

				if ((iB + 1) % 10) == 0:
					secs = time.time()-start_time_epoch
					hours = secs//3600
					mins = secs/60%60
					print("%2dh%2dm E[%2d] B[%d/%d] D_loss: %.4f, E_loss: %.4f, G_loss: %.4f, D_acc:%.4f" %
						  (hours,mins, (epoch + 1), (iB + 1), self.data_loader.dataset.__len__() // self.batch_size,
						  D_loss.data[0], E_loss.data[0], G_loss.data[0], D_acc))

			self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
			print("dumping x_hat from epoch {}".format(epoch+1))
			self.save()
			utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name, y_max=15)
			self.dump_x_hat((epoch+1))

		self.train_hist['total_time'] = time.time() - start_time + self.train_hist['total_time']
		print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
			  self.epoch, self.train_hist['total_time']))
		print("Training finish!... save training results")

		self.save()
		utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

	def dump_x_hat(self, epoch, fix=True):
		self.G.eval()


		""" fixed image """
		temp = self.Enc(self.sample_x2D_)
		mu, sigma= Gaussian_distribution(temp)
		reparamZ_ = torch.normal( torch.zeros(self.sample_num, self.Nz), torch.ones(self.sample_num,self.Nz) )
		if self.gpu_mode:
			reparamZ_ = Variable(reparamZ_.cuda())
		else:
			reparamZ_ = Variable(reparamZ_)
		zey_ = mu + reparamZ_*sigma
		samples = self.G(zey_)

		if self.gpu_mode:
			samples = samples.cpu().data.numpy().squeeze()
		else:
			samples = samples.data.numpy().squeeze()

		fname = os.path.join( self.result_dir, self.dataset, self.model_name,
										self.model_name+'_E%03d.npy'%(epoch))
		samples.dump(fname)


	def visualize_results(self, epoch, fix=True):
		print( 'visualizing result...' )
		save_dir = os.path.join(self.result_dir, self.dataset, self.model_name, 'generate') 
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		self.G.eval()
		self.Enc.eval()

#		if fix:
#			""" fixed noise """
#			samples = self.G(self.sample_z_)
#		else:
#			""" random noise """
#			sample_z_ = torch.normal( torch.zeros(self.batch_size, self.Nz), torch.ones(self.batch_size,self.Nz) )
#			if self.gpu_mode:
#				sample_z_ = Variable(sample_z_.cuda(), volatile=True)
#			else:
#				sample_z_ = Variable(sample_z_, volatile=True)
#
#			samples = self.G(sample_z_)
#
#		if self.gpu_mode:
#			samples = samples.cpu().data.numpy().squeeze()
#		else:
#			samples = samples.data.numpy().squeeze()
#
#		for i in range( self.batch_size ):
#			filename = os.path.join( self.result_dir, self.dataset, self.model_name, 'generate',
#										self.model_name+'_e%03d_random_sample%03d.npy'%(epoch,i))
#			np.expand_dims(samples[i],0).dump( filename )

		# reconstruction (inference 2D-to-3D )
		x2d_ = self.get_image_batch()

		x_ = Variable(x2d_.cuda(), volatile=True)
	
		dis = self.Enc(x_)
		mu, sigma = Gaussian_distribution(dis)

		z = torch.FloatTensor(self.batch_size, self.Nz).normal_(0.0, 1.0)
		z = Variable(z.cuda())
		
		z_enc = mu + z*sigma

		samples = self.G(z_enc)
		samples = samples.cpu().data.numpy()
		print( 'saving...')
		for i in range( self.batch_size ):
			fname = os.path.join(self.result_dir, self.dataset, self.model_name, 'generate', self.model_name + '_%03d.png'%i)
			imageio.imwrite(fname, x2d_[i].numpy().transpose(1,2,0))
			filename = os.path.join( self.result_dir, self.dataset, self.model_name, 'generate',
										self.model_name+'_e%03d_sample%03d.npy'%(epoch,i))
			np.expand_dims(samples[i],0).dump( filename )


	def save(self):
		save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
		torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))
		torch.save(self.Enc.state_dict(), os.path.join(save_dir, self.model_name + '_Enc.pkl'))

		with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
			pickle.dump(self.train_hist, f)

	def load(self):
		save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

		print( 'loading from {}...'.format(save_dir) )
		self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
		self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))
		self.Enc.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_Enc.pkl')))

		try:
			fhandle = open(os.path.join(save_dir, self.model_name + '_history.pkl'))
			self.train_hist = pickle.load(fhandle)
			fhandle.close()
			
			self.epoch_start = len(self.train_hist['per_epoch_time'])
			print( 'loaded epoch {}'.format(self.epoch_start) )
		except:
			print('history is not found and ignored')

	def get_image_batch(self):
		dataIter = iter(self.data_loader)
		return next(dataIter)[2]

	def interpolate(self, opts):
		save_dir = os.path.join(self.result_dir, self.dataset, self.model_name, 'interp') 
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		
		self.G.eval()
		self.Enc.eval()

		if self.gpu_mode:
			self.Enc = self.Enc.cuda()

		n_interp = opts.n_interp
		nz = 50
		is_enc = opts.is_enc
		tran = transforms.ToTensor()

		# interpolate between twe noise(z1, z2).
		f_type = ""	
		if is_enc:	
			x2d_ = self.get_image_batch()

			fname = os.path.join(self.result_dir, self.dataset, self.model_name, 'interp', self.model_name + f_type+'_A.png')
			imageio.imwrite(fname, x2d_[0].numpy().transpose(1,2,0))
			fname = os.path.join(self.result_dir, self.dataset, self.model_name, 'interp', self.model_name + f_type+'_B.png')
			imageio.imwrite(fname, x2d_[1].numpy().transpose(1,2,0))
			
			f_type = "enc"

			x_ = Variable(x2d_.cuda(), volatile=True)
		
			dis = self.Enc(x_)
			mu, sigma = Gaussian_distribution(dis)

			z = torch.FloatTensor(self.batch_size, nz).normal_(0.0, 1.0)
			z = Variable(z.cuda(), volatile=True)
			
			z_enc = mu + z*sigma
			z1 = (z_enc[0].unsqueeze(0))
			z2 = (z_enc[1].unsqueeze(0))
			
		else:
			z1 = torch.FloatTensor(1, nz).normal_(0.0, 1.0)
			z2 = torch.FloatTensor(1, nz).normal_(0.0, 1.0)
			z1, z2 = Variable(z1, volatile=True), Variable(z2, volatile=True)

		
		
		z_interp = torch.FloatTensor(1, nz)

		if self.gpu_mode:
			z_interp = z_interp.cuda()
			z1 = z1.cuda()
			z2 = z2.cuda()
			self.G = self.G.cuda()

		samples_a = self.G(z1)
		samples_a = samples_a.cpu().data.numpy()
		fname = os.path.join(self.result_dir, self.dataset, self.model_name, 'interp', self.model_name+f_type+'_A.npy')
		samples_a.dump(fname)

		samples_b = self.G(z2)
		samples_b = samples_b.cpu().data.numpy()
		fname = os.path.join(self.result_dir, self.dataset, self.model_name, 'interp', self.model_name + f_type+'_B.npy')
		samples_b.dump(fname)


		dz = (z2-z1)/n_interp

		#make interpolation 3D
		for i in range(1, n_interp + 1):
			z_interp = z1 + i*dz
			samples = self.G(z_interp)
			if self.gpu_mode:
				samples = samples.cpu().data.numpy()
			else:
				samples = samples.data.numpy()
			fname = os.path.join(self.result_dir, self.dataset, self.model_name, 'interp', self.model_name +f_type +'%03d.npy' % (i))
			samples.dump(fname)

