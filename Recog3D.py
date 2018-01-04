import utils, torch, time, os, pickle, imageio
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils3D.visualize import plot_voxel
import pdb

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

		self.convID = nn.Sequential(
			nn.Conv3d(512, Nid, 4, bias=False),
		)

		self.convPCode = nn.Sequential(
			nn.Conv3d(512, Ncode, 4, bias=False),
		)
		utils.initialize_weights(self)

	def forward(self, input):
		feature = self.conv(input)

		#fid = self.convID( feature ).squeeze(4).squeeze(3).squeeze(2)
		fcode = self.convPCode( feature ).squeeze(4).squeeze(3).squeeze(2)

		return fcode#fid, fcode


class Recog3D(object):
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
		if len(args.comment) > 0:
			self.model_name = self.model_name + '_' + args.comment
		self.lambda_ = 0.25
		self.D_threshold = 0.8

		self.alpha1 = 5
		self.alpha2 = 0.0001

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


		result_dir = os.path.join(self.result_dir, self.dataset, self.model_name)
		if not os.path.exists(result_dir):
			os.makedirs(result_dir)
		for iS in range(self.test_sample_size):
			fname = os.path.join( self.result_dir, self.dataset, self.model_name, 'sample_%03d.png'%(iS))
			imageio.imwrite(fname, self.sample_image_[iS].numpy().transpose(1,2,0))

		fname = os.path.join( self.result_dir, self.dataset, self.model_name, 'sampleGT.npy')
		self.sample_x_.numpy().squeeze().dump( fname )
			
		if self.gpu_mode:
			self.sample_x_ = Variable( self.sample_x_.cuda(), volatile=True )
			self.sample_image_ = Variable( self.sample_image_.cuda(), volatile=True )
			self.sample_y_id_ = Variable( self.sample_y_id_.cuda(), volatile=True )
			self.sample_y_pcode_ = Variable( self.sample_y_pcode_.cuda(), volatile=True )
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
			self.sample_y_pcode_ = Variable( self.sample_y_pcode_, volatile=True )
			self.sample_y_pcode_onehot_ = Variable( self.sample_y_pcode_onehot_, volatile=True )

		# networks init
		self.D = discriminator(self.Nid, self.Npcode)
		self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

		if self.gpu_mode:
			self.D.cuda()
			self.CE_loss = nn.CrossEntropyLoss().cuda()
		else:
			self.CE_loss = nn.CrossEntropyLoss()

#		print('---------- Networks architecture -------------')
#		utils.print_network(self.G)
#		utils.print_network(self.D)
#		print('-----------------------------------------------')


	def train(self):
		if not hasattr(self, 'train_hist') :
			self.train_hist = {}
			self.train_hist['D_loss'] = []
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

				D_pcode = self.D(x_)
#				D_loss_id = self.CE_loss(D_id, y_id_)
				D_loss_pcode = self.CE_loss(D_pcode, y_pcode_)

				D_loss = D_loss_pcode
				D_loss.backward()
				self.train_hist['D_loss'].append(D_loss.data[0])

				if self.gpu_mode:
					_, predicted = torch.max(D_pcode.cpu().data, 1)
				else:
					_, predicted = torch.max(D_pcode.data, 1)
				total = len(y_pcode_)
				correct = (predicted == y_['pcode']).sum()
				D_acc = float(correct) / total
				# D gets updated only if its accuracy is below 80%
				self.train_hist['D_acc'].append(D_acc)
				self.D_optimizer.step()

				if ((iB + 1) % 10) == 0:
					secs = time.time()-start_time_epoch
					hours = secs//3600
					mins = secs/60%60
					print("%2dh%2dm E[%2d] B[%d/%d] D_loss: %.4f, D_acc:%.4f" %
						  (hours,mins, (epoch + 1), (iB + 1), self.data_loader.dataset.__len__() // self.batch_size,
						  D_loss.data[0], D_acc))

			self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
			print("dumping x_hat from epoch {}".format(epoch+1))
			self.save()
			utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name, y_max=10)
			self.write_validation_result((epoch+1))

		self.train_hist['total_time'] = time.time() - start_time + self.train_hist['total_time']
		print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
			  self.epoch, self.train_hist['total_time']))
		print("Training finish!... save training results")

		self.save()
		utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

	def write_validation_result(self, epoch, fix=True):
		self.D.eval()

		result_dir = os.path.join(self.result_dir, self.dataset, self.model_name)
		if not os.path.exists(result_dir):
			os.makedirs(result_dir)

		""" fixed image """
		D_pcode = self.D( self.sample_x_ )
		if self.gpu_mode:
			_, predicted = torch.max(D_pcode.cpu().data, 1)
			labels = self.sample_y_pcode_.cpu()
		else:
			_, predicted = torch.max(D_pcode.data, 1)
			labels = self.sample_y_pcode_.data

		fname = os.path.join( self.result_dir, self.dataset, self.model_name,
										self.model_name+'_E%03d.txt'%(epoch))
		with open(fname,'w') as f:
			for (pred, label) in zip(predicted, labels):
				f.write( "{} - {}\n".format(pred, label))


	def visualize_results(self, epoch):
		self.write_validation_result(epoch)

	def save(self):
		save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

		with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
			pickle.dump(self.train_hist, f)

	def load(self):
		save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

		print( 'loading from {}...'.format(save_dir) )
		self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))

		try:
			fhandle = open(os.path.join(save_dir, self.model_name + '_history.pkl'))
			self.train_hist = pickle.load(fhandle)
			fhandle.close()
			
			self.epoch_start = len(self.train_hist['per_epoch_time'])
			print( 'loaded epoch {}'.format(self.epoch_start) )
		except:
			print('history is not found and ignored')
