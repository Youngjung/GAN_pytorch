import argparse, os
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

"""parsing and configuration"""
def parse_args():
	desc = "Pytorch implementation of GAN collections"
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--gan_type', type=str, default='EBGAN',
						choices=['GAN', 'CGAN', 'infoGAN', 'ACGAN', 'EBGAN', 'BEGAN', 'WGAN', 'WGAN_GP', 'DRAGAN', 'LSGAN', 'DRGAN', 'AE', 'GAN3D'],
						help='The type of GAN')#, required=True)
	parser.add_argument('--dataset', type=str, default='mnist', 
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
	parser.add_argument('--gpu_mode', type=bool, default=True)
	parser.add_argument('--use_GP', type=bool, default=False, help='use Gradient Penalty')
	parser.add_argument('--num_workers', type=int, default='1', help='number of threads for DataLoader')
	parser.add_argument('--comment', type=str, default='', help='comment to put on model_name')

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

	if len(opts.comment)>0:
		print( "comment: " + opts.comment )
		tempconcat = opts.gan_type + '_' + opts.comment

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

	try:
		assert not ( (opts.gan_type=='GAN3D') ^
					 (opts.dataset in ('ShapeNet','Bosphorus'))
				   )
	except:
		exit( '--gan_type=GAN3D and --dataset=ShapeNet or Bosphorus should be used together' )

	return opts

"""main"""
def main():
	# parse arguments
	opts = parse_args()
	if opts is None:
		exit()

		# declare instance for GAN
	if opts.gan_type == 'GAN':
		gan = GAN(opts)
	elif opts.gan_type == 'CGAN':
		gan = CGAN(opts)
	elif opts.gan_type == 'ACGAN':
		gan = ACGAN(opts)
	elif opts.gan_type == 'infoGAN':
		gan = infoGAN(opts, SUPERVISED = True)
	elif opts.gan_type == 'EBGAN':
		gan = EBGAN(opts)
	elif opts.gan_type == 'WGAN':
		gan = WGAN(opts)
	elif opts.gan_type == 'WGAN_GP':
		gan = WGAN_GP(opts)
	elif opts.gan_type == 'DRAGAN':
		gan = DRAGAN(opts)
	elif opts.gan_type == 'LSGAN':
		gan = LSGAN(opts)
	elif opts.gan_type == 'BEGAN':
		gan = BEGAN(opts)
	elif opts.gan_type == 'DRGAN':
		gan = DRGAN(opts)
	elif opts.gan_type == 'AE':
		gan = AutoEncoder(opts)
	elif opts.gan_type == 'GAN3D':
		gan = GAN3D(opts)
	else:
		raise Exception("[!] There is no option for " + opts.gan_type)

		# launch the graph in a session
	gan.train()
	print(" [*] Training finished!")

	# visualize learned generator
	gan.visualize_results(opts.epoch)
	print(" [*] Testing finished!")

if __name__ == '__main__':
	main()
