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

	parser.add_argument('--gan_type1', type=str, default='EBGAN',
						choices=['GAN', 'CGAN', 'infoGAN', 'ACGAN', 'EBGAN', 'BEGAN', 'WGAN', 'WGAN_GP', 'DRAGAN', 'LSGAN',
									'DRGAN', 'AE',
									'GAN3D', 'VAEGAN3D', 'DRGAN3D', 'DRGAN2D',
									'Recog3D', 'VAEDRGAN3D', 'DRcycleGAN3D', 'CycleGAN3D',
									'AE3D'],
						help='The type of GAN')#, required=True)
	parser.add_argument('--gan_type2', type=str, default='EBGAN',
						choices=['GAN', 'CGAN', 'infoGAN', 'ACGAN', 'EBGAN', 'BEGAN', 'WGAN', 'WGAN_GP', 'DRAGAN', 'LSGAN',
									'DRGAN', 'AE',
									'GAN3D', 'VAEGAN3D', 'DRGAN3D', 'DRGAN2D',
									'Recog3D', 'VAEDRGAN3D', 'DRcycleGAN3D', 'CycleGAN3D',
									'AE3D'],
						help='The type of GAN')#, required=True)
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

	# below arguments are for interpolation (eval mode)
	parser.add_argument('--interpolate', type=str2bool, default=False, help='generate samples with interpolation from saved model')
	parser.add_argument('--is_enc', type=str2bool, default=False, help='make latent variable from input images')
	parser.add_argument('--n_interp', type=int, default=20, help='number of interpolation points')

	# below arguments are for generation (eval mode)
	parser.add_argument('--generate', type=str2bool, default=False, help='generate samples from saved model')
	parser.add_argument('--fix_z', type=str2bool, default=False, help='fix z')

	return check_args(parser.parse_args())

"""checking arguments"""
def check_args(opts):
	# --save_dir
	if not os.path.exists(opts.save_dir):
		os.makedirs(opts.save_dir)

	# --result_dir
	if not os.path.exists(opts.result_dir):
		os.makedirs(opts.result_dir)

	# --result_dir
	if not os.path.exists(opts.log_dir):
		os.makedirs(opts.log_dir)

	# --loss_option
	if len(opts.loss_option)>0:
		option_part = '_'+opts.loss_option
	else:
		option_part = ''

	if len(opts.comment)>0:
		print( "comment: " + opts.comment )
		comment_part = '_'+opts.comment
	else:
		comment_part = ''
	tempconcat = opts.gan_type1+'_'+opts.gan_type2+option_part+comment_part
	print( 'models and loss plot -> ' + os.path.join( opts.save_dir, opts.dataset, tempconcat ) )
	print( 'results -> ' + os.path.join( opts.result_dir, opts.dataset, tempconcat ) )

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

	opts.comment = opts.comment1
	opts.gan_type = opts.gan_type1
	# declare instance for GAN
	if opts.gan_type1 == 'GAN':
		gan1 = GAN(opts)
	elif opts.gan_type1 == 'CGAN':
		gan1 = CGAN(opts)
	elif opts.gan_type1 == 'ACGAN':
		gan1 = ACGAN(opts)
	elif opts.gan_type1 == 'infoGAN':
		gan1 = infoGAN(opts, SUPERVISED = True)
	elif opts.gan_type1 == 'EBGAN':
		gan1 = EBGAN(opts)
	elif opts.gan_type1 == 'WGAN':
		gan1 = WGAN(opts)
	elif opts.gan_type1 == 'WGAN_GP':
		gan1 = WGAN_GP(opts)
	elif opts.gan_type1 == 'DRAGAN':
		gan1 = DRAGAN(opts)
	elif opts.gan_type1 == 'LSGAN':
		gan1 = LSGAN(opts)
	elif opts.gan_type1 == 'BEGAN':
		gan1 = BEGAN(opts)
	elif opts.gan_type1 == 'DRGAN':
		gan1 = DRGAN(opts)
	elif opts.gan_type1 == 'AE':
		gan1 = AutoEncoder(opts)
	elif opts.gan_type1 == 'GAN3D':
		gan1 = GAN3D(opts)
	elif opts.gan_type1 == 'VAEGAN3D':
		gan1 = VAEGAN3D(opts)
	elif opts.gan_type1 == 'DRGAN3D':
		gan1 = DRGAN3D(opts)
	elif opts.gan_type1 == 'Recog3D':
		gan1 = Recog3D(opts)
	elif opts.gan_type1 == 'VAEDRGAN3D':
		gan1 = VAEDRGAN3D(opts)
	elif opts.gan_type1 == 'DRcycleGAN3D':
		gan1 = DRcycleGAN3D(opts)
	elif opts.gan_type1 == 'CycleGAN3D':
		gan1 = CycleGAN3D(opts)
	elif opts.gan_type1 == 'AE3D':
		gan1 = AutoEncoder3D(opts)
	elif opts.gan_type1 == 'DRGAN2D':
		gan1 = DRGAN2D(opts)
	else:
		raise Exception("[!] There is no option for " + opts.gan_type1)

	opts.comment = opts.comment2
	opts.gan_type = opts.gan_type2
	# declare instance for GAN
	if opts.gan_type2 == 'GAN':
		gan2 = GAN(opts)
	elif opts.gan_type2 == 'CGAN':
		gan2 = CGAN(opts)
	elif opts.gan_type2 == 'ACGAN':
		gan2 = ACGAN(opts)
	elif opts.gan_type2 == 'infoGAN':
		gan2 = infoGAN(opts, SUPERVISED = True)
	elif opts.gan_type2 == 'EBGAN':
		gan2 = EBGAN(opts)
	elif opts.gan_type2 == 'WGAN':
		gan2 = WGAN(opts)
	elif opts.gan_type2 == 'WGAN_GP':
		gan2 = WGAN_GP(opts)
	elif opts.gan_type2 == 'DRAGAN':
		gan2 = DRAGAN(opts)
	elif opts.gan_type2 == 'LSGAN':
		gan2 = LSGAN(opts)
	elif opts.gan_type2 == 'BEGAN':
		gan2 = BEGAN(opts)
	elif opts.gan_type2 == 'DRGAN':
		gan2 = DRGAN(opts)
	elif opts.gan_type2 == 'AE':
		gan2 = AutoEncoder(opts)
	elif opts.gan_type2 == 'GAN3D':
		gan2 = GAN3D(opts)
	elif opts.gan_type2 == 'VAEGAN3D':
		gan2 = VAEGAN3D(opts)
	elif opts.gan_type2 == 'DRGAN3D':
		gan2 = DRGAN3D(opts)
	elif opts.gan_type2 == 'Recog3D':
		gan2 = Recog3D(opts)
	elif opts.gan_type2 == 'VAEDRGAN3D':
		gan2 = VAEDRGAN3D(opts)
	elif opts.gan_type2 == 'DRcycleGAN3D':
		gan2 = DRcycleGAN3D(opts)
	elif opts.gan_type2 == 'CycleGAN3D':
		gan2 = CycleGAN3D(opts)
	elif opts.gan_type2 == 'AE3D':
		gan2 = AutoEncoder3D(opts)
	elif opts.gan_type2 == 'DRGAN2D':
		gan2 = DRGAN2D(opts)
	else:
		raise Exception("[!] There is no option for " + opts.gan_type2)

	print(" [*] Loading saved model...")
	gan1.load()
	gan2.load()
	gan1.G = gan1.G.cuda()
	gan2.G = gan2.G.cuda()
	gan2.Enc = gan2.Enc.cuda()
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
	Nz = 50

	# load batch
	x3D, y, x2D = get_image_batch( data_loader )
	y = y['pcode']
	y_onehot = torch.zeros( opts.batch_size, Npcode )
	y_onehot.scatter_(1, y.view(-1,1), 1)

	# save input image
	for i in range(opts.batch_size):
		fname = os.path.join(opts.result_dir, opts.dataset, 'compare','input_%03d.png'%i)
		imageio.imwrite(fname, x2D[i].numpy().transpose(1,2,0))

	# to CUDA
	x2D = Variable( x2D.cuda(), volatile=True )
	y_onehot = Variable( y_onehot.cuda(), volatile=True )

	# visualize learned generator
	gan1.compare( x2D, y, y_onehot )
	gan2.compare( x2D )
	print(" [*] Testing finished!")


def get_image_batch(data_loader):
	dataIter = iter(data_loader)
	return next(dataIter)
if __name__ == '__main__':
	main()
