import utils, torch, time, os, pickle, imageio
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
	mean = vector[:, :200, 0,0]
	variance = vector[:, 200:, 0,0]
	return mean, variance

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder,self).__init__()
		self.input_dim = 3

		self.conv = nn.Sequential(
			nn.Conv2d(self.input_dim, 64, 11, 4, 1,bias=True),
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
			nn.Conv2d(512, 400, 8 , 1, 1, bias=True),
			nn.Sigmoid(),
		)
		utils.initialize_weights(self)

	def forward(self,input):
		fx = self.conv(input)
		return fx


class generator(nn.Module):
	# Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
	# Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
	def __init__(self, input_dim=200, Npcode=48 ):

		super(generator, self).__init__()
		self.input_dim = input_dim
		self.Npcode = Npcode

		self.deconv = nn.Sequential(
			nn.ConvTranspose3d(input_dim+Npcode, 512, 4, bias=False),
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
			nn.ConvTranspose3d(32, 1, 4, 2, 1, bias=False),
			nn.Sigmoid(),
		)
		utils.initialize_weights(self)

	def forward(self, fx, y_pcode_onehot):
		fx = fx.view(-1, self.input_dim, 1,1,1 )
#		y_pcode_onehot = y_pcode_onehot.view(-1, self.Npcode, 1,1,1 )
#		x = torch.cat( (fx, y_pcode_onehot), 1 )
		x = self.deconv(fx)

		return x

class discriminator(nn.Module):
	# Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
	# Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
	def __init__(self, Nid=105, Ncode=48):
	
		super(discriminator, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv3d(1, 32, 4, 2, 1, bias=False),
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
			nn.Conv3d(512, Ncode, 4, bias=False),
		)
		utils.initialize_weights(self)

	def forward(self, input):
		feature = self.conv(input)

		fGAN = self.convGAN( feature ).squeeze(4).squeeze(3).squeeze(2)
		fid = self.convID( feature ).squeeze(4).squeeze(3).squeeze(2)
		#fcode = self.convPCode( feature ).squeeze(4).squeeze(3).squeeze(2)

		return fGAN, fid#, fcode


class VAEDRGAN3D(object):
	def __init__(self, args):
		# parameters
		self.epoch = args.epoch
		self.batch_size = args.batch_size
		self.test_sample_size = min(args.test_sample_size, args.batch_size)
		self.save_dir = args.save_dir
		self.result_dir = args.result_dir
		self.dataset = args.dataset
		self.dataroot_dir = args.dataroot_dir
		self.log_dir = args.log_dir
		self.gpu_mode = args.gpu_mode
		self.use_GP = args.use_GP
		self.model_name = args.gan_type
		self.num_workers = args.num_workers
		if self.use_GP:
			self.model_name = self.model_name + '_GP'
		if len(args.comment) > 0:
			self.model_name = self.model_name + '_' + args.comment
		self.lambda_ = 0.25
		self.D_threshold = 0.8

		self.alpha1 = 5
		self.alpha2 = 0.0001
		self.c = 0.01

		# load dataset
		data_dir = os.path.join( self.dataroot_dir, self.dataset )
		if self.dataset == 'ShapeNet':
			self.data_loader = DataLoader( utils.ShapeNet(data_dir,synsetId=args.synsetId),
											batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
		elif self.dataset == 'Bosphorus':
			self.data_loader = DataLoader( utils.Bosphorus(data_dir, use_image=True, skipCodes=['YR','PR','CR'],
											transform=transforms.ToTensor(),
											shape=(128,128,128)),
											batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
			self.Nid = 105
			self.Npcode = len(self.data_loader.dataset.posecodemap)
		elif self.dataset == 'IKEA':
			self.transform = transforms.Compose([transforms.Scale((256, 256)), transforms.ToTensor()])
			self.data_loader =DataLoader(utils.IKEA(IKEA_data_dir, transform=self.transform), 
										 batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
		else:
			exit("unknown dataset: " + self.dataset)

		self.z_dim = 200

		for iB, (sample_x_, sample_y_, sample_image_) in enumerate(self.data_loader):
			self.sample_x_ = sample_x_
			self.sample_image_ = sample_image_
			self.sample_y_id_ = sample_y_['id']
			self.sample_y_pcode_ = sample_y_['pcode']
			break

		self.sample_x_ = self.sample_x_[:self.test_sample_size,:,:,:,:]
		self.sample_image_ = self.sample_image_[:self.test_sample_size,:,:,:]
		self.sample_y_id_ = self.sample_y_id_[:self.test_sample_size]
		self.sample_y_pcode_ = self.sample_y_pcode_[:self.test_sample_size]
		self.sample_y_pcode_onehot_ = torch.zeros( self.test_sample_size, self.Npcode )
		self.sample_y_pcode_onehot_.scatter_(1, self.sample_y_pcode_.view(-1,1), 1)
#		for iS in range(self.test_sample_size):
#			fname = os.path.join( self.result_dir, self.dataset, self.model_name, 'sample_%03d.png'%(iS))
#			imageio.imwrite(fname, self.sample_image_[iS].numpy().transpose(1,2,0))
#
#		fname = os.path.join( self.result_dir, self.dataset, self.model_name, 'sampleGT.npy')
#		self.sample_x_.numpy().squeeze().dump( fname )
			
		if self.gpu_mode:
			self.sample_x_ = Variable( self.sample_x_.cuda(), volatile=True )
			self.sample_image_ = Variable( self.sample_image_.cuda(), volatile=True )
			self.sample_y_id_ = Variable( self.sample_y_id_.cuda(), volatile=True )
			self.sample_y_pcode_onehot_ = Variable( self.sample_y_pcode_onehot_.cuda(), volatile=True )
			self.sample_z_ = Variable( 
						torch.normal(torch.zeros(self.test_sample_size, self.z_dim),
									 torch.ones(self.test_sample_size,self.z_dim)*0.33).cuda(),
						volatile=True)
		else:
			self.sample_x_ = Variable( self.sample_x_, volatile=True )
			self.sample_image_ = Variable( self.sample_image_, volatile=True )
			self.sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim)), volatile=True)
			self.sample_y_id_ = Variable( self.sample_y_id_, volatile=True )
			self.sample_y_pcode_onehot_ = Variable( self.sample_y_pcode_onehot_, volatile=True )

		# networks init
		self.G = generator(self.z_dim, 0) # self.Npcode)
		self.D = discriminator(self.Nid, self.Npcode)
		self.Enc = Encoder()
		self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
		self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
		self.Enc_optimizer = optim.Adam(self.Enc.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

		if self.gpu_mode:
			self.G.cuda()
			self.D.cuda()
			self.Enc.cuda()
			self.BCE_loss = nn.BCELoss().cuda()
			self.CE_loss = nn.CrossEntropyLoss().cuda()
			self.MSE_loss = nn.MSELoss().cuda()
		else:
			self.BCE_loss = nn.BCELoss()
			self.CE_loss = nn.CrossEntropyLoss()
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
			self.Enc.train()
			self.G.train()
			epoch_start_time = time.time()
			start_time_epoch = time.time()

			for iB, (x_, y_, image_) in enumerate(self.data_loader):
				if iB == self.data_loader.dataset.__len__() // self.batch_size:
					break

				y_id_ = y_['id']
				y_pcode_ = y_['pcode']


				#y_id_onehot_ = torch.zeros( self.batch_size, self.Nid )
				#y_id_onehot_.scatter_(1, y_id_.view(-1,1), 1)
				y_pcode_onehot_ = torch.zeros( self.batch_size, self.Npcode )
				y_pcode_onehot_.scatter_(1, y_pcode_.view(-1,1), 1)


				z_ = torch.normal( torch.zeros(self.batch_size, self.z_dim), torch.ones(self.batch_size,self.z_dim) )
				if self.gpu_mode:
					x_ = Variable(x_.cuda())
					image_ = Variable(image_.cuda())
					z_ = Variable(z_.cuda())
					y_id_ = Variable(y_id_.cuda())
					y_pcode_ = Variable(y_pcode_.cuda())
					y_pcode_onehot_ = Variable(y_pcode_onehot_.cuda())
				else:
					x_ = Variable(x_)
					image_ = Variable(image_)
					z_ = Variable(z_)
					y_id_ = Variable(y_id_)
					y_pcode_ = Variable(y_pcode_)
					y_pcode_onehot_ = Variable(y_pcode_onehot_)

				
				# update D network
				self.D_optimizer.zero_grad()

				D_real, D_id= self.D(x_)
#				D_loss_real = -torch.mean(D_real)
				D_loss_real = self.BCE_loss(D_real, self.y_real_)
				D_loss_id = self.CE_loss(D_id, y_id_)
#				D_loss_pcode = self.CE_loss(D_pcode, y_pcode_)
				num_correct_real = torch.sum(D_real>0.5)

				G_ = self.G(z_,None)#, y_pcode_onehot_)
				D_fake, _ = self.D(G_)
#				D_loss_fake = torch.mean(D_fake)
				D_loss_fake = self.BCE_loss(D_fake, self.y_fake_)
				num_correct_fake = torch.sum(D_fake<0.5)

				""" DRAGAN Loss (Gradient penalty) """
				if self.use_GP:
					# This is borrowed from https://github.com/jfsantos/dragan-pytorch/blob/master/dragan.py
					if self.gpu_mode:
						alpha = torch.rand(x_.size()).cuda()
						x_hat = Variable(alpha* x_.data + (1-alpha)* (x_.data + 0.5 * x_.data.std() * torch.rand(x_.size()).cuda()),
									 requires_grad=True)
					else:
						alpha = torch.rand(x_.size())
						x_hat = Variable(alpha * x_.data + (1 - alpha) * (x_.data + 0.5 * x_.data.std() * torch.rand(x_.size())),
							requires_grad=True)
					pred_hat, _ = self.D(x_hat)
					if self.gpu_mode:
						gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
									 create_graph=True, retain_graph=True, only_inputs=True)[0]
					else:
						gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()),
										 create_graph=True, retain_graph=True, only_inputs=True)[0]
	
					gradient_penalty = self.lambda_ * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()
	
					D_loss = D_loss_real + D_loss_id + D_loss_fake + gradient_penalty
					#D_loss = D_loss_real + D_loss_id + D_loss_pcode + D_loss_fake + gradient_penalty
				else:
					D_loss = D_loss_real + D_loss_id + D_loss_fake
					#D_loss = D_loss_real + D_loss_id + D_loss_pcode + D_loss_fake
				D_loss.backward()
				self.train_hist['D_loss'].append(D_loss.data[0])

				# D gets updated only if its accuracy is below 80%
				D_acc = float(num_correct_real.data[0] + num_correct_fake.data[0]) / (self.batch_size*2)
				self.train_hist['D_acc'].append(D_acc)
				if D_acc < self.D_threshold:
					self.D_optimizer.step()
#				for p in self.D.parameters():
#					p.data.clamp_(-self.c, self.c)

				# update Enc network
				self.Enc_optimizer.zero_grad()

				temp = self.Enc(image_)
				mu, sigma= Gaussian_distribution(temp)
				reparamZ_ = torch.normal( torch.zeros(self.batch_size, self.z_dim), torch.ones(self.batch_size,self.z_dim) )
				if self.gpu_mode:
					reparamZ_ = Variable(reparamZ_.cuda())
				else:
					reparamZ_ = Variable(reparamZ_)

				zey_ = mu + reparamZ_*sigma
				Gey_ = self.G(zey_,None) #, y_pcode_onehot_)
				D_fake, D_fake_id = self.D(Gey_) #, y_pcode_onehot_)

				KL_div = 0.5 * torch.sum(mu**2 + sigma**2 - torch.log(1e-8 + sigma**2)-1) / self.batch_size
				E_loss_MSE = self.MSE_loss( Gey_, x_ )
#				E_loss_GAN = -torch.mean( D_fake )
#				E_loss_GAN = self.BCE_loss( D_fake, self.y_real_ )
				E_loss_id = self.CE_loss( D_fake_id, y_id_ )
				E_loss = KL_div*self.alpha1 + E_loss_MSE*self.alpha2 + E_loss_id # + E_loss_GAN 
				E_loss.backward()
				self.train_hist['E_loss'].append(E_loss.data[0])
				self.Enc_optimizer.step()

				# update G network
				self.G_optimizer.zero_grad()

				temp = self.Enc(image_)
				mu, sigma= Gaussian_distribution(temp)
				reparamZ_ = torch.normal( torch.zeros(self.batch_size, self.z_dim), torch.ones(self.batch_size,self.z_dim) )
				if self.gpu_mode:
					reparamZ_ = Variable(reparamZ_.cuda())
				else:
					reparamZ_ = Variable(reparamZ_)

				zey_ = mu + reparamZ_*sigma
				Gey_ = self.G(zey_,None) #, y_pcode_onehot_)

				D_fake, D_id = self.D(Gey_)

#				G_loss_GAN = torch.mean(D_fake)
				G_loss_GAN = self.BCE_loss(D_fake, self.y_real_)
				G_loss_MSE = self.MSE_loss(Gey_, x_)
				G_loss_id = self.CE_loss(D_id, y_id_)
#				G_loss_pcode = self.CE_loss(D_pcode, y_pcode_)
				G_loss = G_loss_GAN + G_loss_MSE*self.alpha2 + G_loss_id #+ G_loss_pcode
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
			utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name, y_max=10)
			self.dump_x_hat((epoch+1))
#			self.visualize_results((epoch+1))

		self.train_hist['total_time'] = time.time() - start_time + self.train_hist['total_time']
		print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
			  self.epoch, self.train_hist['total_time']))
		print("Training finish!... save training results")

		self.save()
#		utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name, self.epoch)
		utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

	def dump_x_hat(self, epoch, fix=True):
		self.Enc.eval()
		self.G.eval()

		result_dir = os.path.join(self.result_dir, self.dataset, self.model_name)
		if not os.path.exists(result_dir):
			os.makedirs(result_dir)

		""" fixed image """
		temp = self.Enc(self.sample_image_)
		mu, sigma= Gaussian_distribution(temp)
		reparamZ_ = torch.normal( torch.zeros(self.test_sample_size, self.z_dim), torch.ones(self.test_sample_size,self.z_dim) )
		if self.gpu_mode:
			reparamZ_ = Variable(reparamZ_.cuda())
		else:
			reparamZ_ = Variable(reparamZ_)
		zey_ = mu + reparamZ_*sigma
		samples = self.G( zey_, None ) #self.sample_y_pcode_onehot_ )

		if self.gpu_mode:
			samples = samples.cpu().data.numpy().squeeze()
		else:
			samples = samples.data.numpy().squeeze()

		fname = os.path.join( self.result_dir, self.dataset, self.model_name,
										self.model_name+'_E%03d.npy'%(epoch))
		samples.dump(fname)


	def visualize_results(self, epoch):
		self.dump_x_hat(epoch)

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
