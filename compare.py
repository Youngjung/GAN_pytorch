import argparse, os, pickle
from GAN import GAN
from CGAN import CGAN
from LSGAN import LSGAN
from DRAGAN import DRAGAN
from ACGAN import ACGAN
from WGAN import WGAN
from WGAN_GP import WGAN_GP
from infoGAN import infoGAN
from EBGAN import EBGAN
from BEGAN import BEGAN
from DRGAN import DRGAN
from AE import AutoEncoder
from GAN3D import GAN3D
from VAEGAN3D import VAEGAN3D
from DRGAN3D import DRGAN3D
from Recog3D import Recog3D
from VAEDRGAN3D import VAEDRGAN3D
from DRcycleGAN3D import DRcycleGAN3D
from CycleGAN3D import CycleGAN3D
from AE3D import AutoEncoder3D
from DRGAN2D import DRGAN2D 
from DRecon3DGAN import DRecon3DGAN

import pdb
import torch, imageio
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import Bosphorus

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

"""parsing and configuration"""
def parse_args():
	desc = "Pytorch implementation of GAN collections"
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--dataset', type=str, default='Bosphorus', 
						choices=['mnist', 'fashion-mnist', 'celebA', 'MultiPie', 'miniPie', 'CASIA-WebFace','ShapeNet', 'Bosphorus'],
						help='The name of dataset')
	parser.add_argument('--synsetId', type=str, default='chair', help='synsetId of ShapeNet')
	parser.add_argument('--dataroot_dir', type=str, default='data', help='root path of data')
	parser.add_argument('--epoch', type=int, default=25, help='The number of epochs to run')
	parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
	parser.add_argument('--test_sample_size', type=int, default=64, help='The number of samples to test')
	parser.add_argument('--save_dir', type=str, default='models',
						help='Directory name to save the model')
	parser.add_argument('--result_dir', type=str, default='results',
						help='Directory name to save the generated images')
	parser.add_argument('--log_dir', type=str, default='logs',
						help='Directory name to save training logs')
	parser.add_argument('--lrG', type=float, default=0.0002)
	parser.add_argument('--lrD', type=float, default=0.0002)
	parser.add_argument('--beta1', type=float, default=0.5)
	parser.add_argument('--beta2', type=float, default=0.999)
	parser.add_argument('--gpu_mode', type=str2bool, default=True)
	parser.add_argument('--num_workers', type=int, default='1', help='number of threads for DataLoader')
	parser.add_argument('--comment', type=str, default='', help='not used')
	parser.add_argument('--comment1', type=str, default='', help='comment1 to put on model_name')
	parser.add_argument('--comment2', type=str, default='', help='comment2 to put on model_name')
	parser.add_argument('--resume', type=str2bool, default=False, help='resume training from saved model')
	parser.add_argument('--centerBosphorus', type=str2bool, default=True, help='center Bosphorus PCL in voxel space')
	parser.add_argument('--loss_option', type=str, default='', help='recon,dist,GP')
	parser.add_argument('--n_critic', type=int, default=1, help='n_critic')
	parser.add_argument('--n_gen', type=int, default=1, help='n_gen')
	parser.add_argument('--nDaccAvg', type=int, default=5, help='number of batches for moving averaging D_acc')
	parser.add_argument('--fname_cache', type=str, default='', help='filename of cached datalist, ex)cache_Bosphorus.txt')
	parser.add_argument('--multi_gpu', type=str2bool, default=False)

	return check_args(parser.parse_args())

"""checking arguments"""
def check_args(opts):
	# --save_dir
	if not os.path.exists(opts.save_dir):
		os.makedirs(opts.save_dir)

	# --result_dir
	if not os.path.exists(opts.result_dir):
		os.makedirs(opts.result_dir)

	# --epoch
	try:
		assert opts.epoch >= 1
	except:
		print('number of epochs must be larger than or equal to one')

	# --batch_size
	try:
		assert opts.batch_size >= 1
	except:
		print('batch size must be larger than or equal to one')

	print( opts )

	return opts

"""main"""
def main():
	# parse arguments
	opts = parse_args()
	if opts is None:
		exit()


	opts.comment = '2d3d_fixed'
	opts.gan_type = 'DRecon3DGAN'
	gan2 = DRecon3DGAN(opts)

	opts.comment = 'expr11_DaccAvg'
	opts.gan_type = 'DRGAN3D'
	opts.fname_cache = 'cache_Bosphorus_12.txt'
	gan1 = DRGAN3D(opts)

	print(" [*] Loading saved model...")
	gan1.load()
	gan2.load()
	gan1.G = gan1.G.cuda()
	gan2.G = gan2.G.cuda()
	print(" [*] Loading finished!")

	data_dir = os.path.join( opts.dataroot_dir, opts.dataset )
	data_loader = DataLoader( Bosphorus(data_dir, use_image=True,
										skipCodes=['YR','PR','CR'],
										transform=transforms.ToTensor(),
										shape=128, image_shape=256, center=True),
								batch_size=opts.batch_size, shuffle=True,
								num_workers=opts.num_workers)
	Nid = 105
	Npcode = len(data_loader.dataset.posecodemap)

	# load batch
	x3D, y, x2D = get_image_batch( data_loader )
	y = y['pcode']
	y_onehot = torch.zeros( opts.batch_size, Npcode )
	y_onehot.scatter_(1, y.view(-1,1), 1)

	y_onehot1 = torch.zeros( opts.batch_size, Npcode+1 )
	y_onehot1.scatter_(1, (y+1).view(-1,1), 1)

	# save input image
	for i in range(opts.batch_size):
		fname = os.path.join(opts.result_dir, opts.dataset, 'compare','input_%03d.png'%i)
		imageio.imwrite(fname, x2D[i].numpy().transpose(1,2,0))

	# to CUDA
	x2D = Variable( x2D.cuda(), volatile=True )
	y_onehot = Variable( y_onehot.cuda(), volatile=True )

	# visualize learned generator
	dir_compare = 'compare'

	dir_dest = os.path.join( dir_compare, 'DRGAN3D' )
	gan1.compare( x2D, y, y_onehot1, dir_dest )

	dir_dest = os.path.join( dir_compare, 'DRecon3DGAN' )
	gan2.compare( x2D, y, y_onehot, dir_dest )

	print(" [*] Testing finished!")


def get_image_batch(data_loader):
	dataIter = iter(data_loader)
	return next(dataIter)
if __name__ == '__main__':
	main()
