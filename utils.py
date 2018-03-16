from __future__ import print_function
import os, csv, sys, gzip, torch, time, pickle, argparse
import torch.nn as nn
import numpy as np
import scipy.misc
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from utils3D.data_io import read_binvox, read_bnt, bnt2voxel, bnt2voxel_wColor
from utils3D.visualize import plot_voxel

import pdb

def load_mnist(dataset, dataroot_dir="./data"):
	data_dir = os.path.join(dataroot_dir, dataset)

	def extract_data(filename, num_data, head_size, data_size):
		with gzip.open(filename) as bytestream:
			bytestream.read(head_size)
			buf = bytestream.read(data_size * num_data)
			data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
		return data

	data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
	trX = data.reshape((60000, 28, 28, 1))

	data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
	trY = data.reshape((60000))

	data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
	teX = data.reshape((10000, 28, 28, 1))

	data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
	teY = data.reshape((10000))

	trY = np.asarray(trY).astype(np.int)
	teY = np.asarray(teY)

	X = np.concatenate((trX, teX), axis=0)
	y = np.concatenate((trY, teY), axis=0).astype(np.int)

	seed = 547
	np.random.seed(seed)
	np.random.shuffle(X)
	np.random.seed(seed)
	np.random.shuffle(y)

	y_vec = np.zeros((len(y), 10), dtype=np.float)
	for i, label in enumerate(y):
		y_vec[i, y[i]] = 1

	X = X.transpose(0, 3, 1, 2) / 255.
	# y_vec = y_vec.transpose(0, 3, 1, 2)

	X = torch.from_numpy(X).type(torch.FloatTensor)
	y_vec = torch.from_numpy(y_vec).type(torch.FloatTensor)
	return X, y_vec

def CustomDataLoader(path, transform, batch_size, shuffle):
	# transform = transforms.Compose([
	#	 transforms.CenterCrop(160),
	#	 transform.Scale(64)
	#	 transforms.ToTensor(),
	#	 transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
	# ])

	# data_dir = 'data/celebA'  # this path depends on your computer
	dset = datasets.ImageFolder(path, transform)
	data_loader = torch.utils.data.DataLoader(dset, batch_size, shuffle)

	return data_loader


class IKEA(Dataset):
	def __init__(self,root_dir="data", transform=None):
		self.filenames = []
		self.root_dir = root_dir
		self.transform = transform

		path = os.path.join(root_dir,"3d_toolbox_notfinal","data","img")
		#print("path :" , path)
		self.filenames = [os.path.join(dirpath,f) for dirpath, dirnames, files in os.walk(path)
							for f in files if f.endswith('.jpg') ]
		#print(self.filenames)
	def __len__(self):
		return len(self.filenames)

	def __getitem__(self, idx):
		basename = os.path.basename(self.filenames[idx])
		image = Image.open(self.filenames[idx]).convert('RGB')
		if self.transform:
			image = self.transform(image)
		labels = {'id': "chair"}
		return image , labels


class Bosphorus( Dataset ):
	def __init__( self, root_dir, transform=None, use_image=False, use_colorPCL=True, fname_cache='',
					inclCodes=[], skipCodes=[], shape=64, image_shape=256, center=True):
		self.root_dir = root_dir
		self.filenames = {}
		self.transform = transform
		self.suffix = '_trim.bnt'
		self.use_image = use_image
		self.use_colorPCL = use_colorPCL
		self.shape = shape
		self.image_shape = image_shape
		self.center = center

		if len(skipCodes) > 0:
			self.skipCodes = skipCodes
		else:
			self.skipCodes = ['O_GLASSES',
								'O_HAIR',
								'YR_',
								'CR_',
								'PR_',
								'CAU_',
								'E_',
								'LFAU_10',
								'LFAU_12LW',
								'LFAU_14',
								'LFAU_15',
								'LFAU_16',
								'LFAU_17',
								'LFAU_18',
								'LFAU_20',
								'LFAU_23',
								'LFAU_24',
								'LFAU_25',
								'LFAU_26',
								'LFAU_28',
								'UFAU_44',
								'UFAU_1',
								'IGN',
								]
		if len(inclCodes) > 0:
			self.inclCodes = inclCodes
		else:
			self.inclCodes = ['LFAU_9',
								'LFAU_12',
								'LFAU_12L',
								'LFAU_12R',
								'LFAU_22',
								'LFAU_27',
								'LFAU_34',
								'N_N',
								'UFAU_2',
								'UFAU_4',
								'UFAU_43',
								]

		print('Loading Bosphorus metadata...', end='')
		print('\t(center={})\t'.format(center), end='')
		sys.stdout.flush()
		time_start = time.time()

		if len(fname_cache) == 0:
			fname_cache = 'cache_Bosphorus.txt'
		if os.path.exists(fname_cache):
			self.filenames = open(fname_cache).read().splitlines()
			print( '{} samples restored from {}'.format(len(self.filenames),fname_cache) )
		else:
			def checkCode(fname, codes, skipCodes = []):
				identity, poseclass, posecode, samplenum =  fname[:-len(self.suffix)].split('_')
#				return poseclass in codes and poseclass not in skipCodes
				return poseclass+"_"+posecode in codes
				
			self.filenames = [os.path.join(dirpath,f) for dirpath, dirnames, files in os.walk(root_dir)
							for f in files if f.endswith(self.suffix) and checkCode(f,self.inclCodes,self.skipCodes) ]
	
			print('{:.0f}sec, {} files found.'.format( time.time()-time_start, len(self.filenames)))
	
			with open(fname_cache, 'w') as f:
				for fname in self.filenames:
					f.write(fname+'\n')
			print( 'cached in {}'.format(fname_cache) )
	
		self.poseclasses = sorted( set( [ os.path.basename(f).split('_')[1]
											for f in self.filenames ] ) )
		self.poseclassmap = {}
		for i, poseclass in enumerate( self.poseclasses ):
			self.poseclassmap[poseclass] = i

		self.posecodes = sorted( set( [ os.path.basename(f).split('_')[2]
											for f in self.filenames ] ) )
		self.posecodemap = {}
		for i, posecode in enumerate( self.posecodes ):
			self.posecodemap[posecode] = i


		fname_stats = os.path.join( root_dir, 'stats.pkl' )
		with open( fname_stats ) as fhandle:
			stats = pickle.load( fhandle )
		self.muA = stats['muA']
		self.muB = stats['muB']
		self.stddevA = stats['stddevA']
		self.stddevB = stats['stddevB']

		print( 'Loading Bosphorus done' )

	def __len__( self ):
		return len( self.filenames )
	
	def __getitem__( self, idx ):
		# load image
		if self.use_image or self.use_colorPCL:
			image = Image.open( self.filenames[idx][:-len(self.suffix)]+'.png' )
			image_original = image
		if self.use_image:
			ratio = float(self.image_shape)/image.size[1] # size returns (w,h), we need h (longer side)
			image = scipy.misc.imresize( image, (self.image_shape,int(ratio*image.size[0])) )
			width = image.shape[1]
			w = (self.image_shape-width)//2
			image = np.pad( image, ((0,0),(w,w+width%2),(0,0)),'constant',constant_values=0 )
			if self.transform:
				image = self.transform(image)
		if self.use_colorPCL and self.transform:
			image_original = self.transform(image_original)

		# load point cloud and fill voxel
		bnt_data, nrows, ncols, imfile = read_bnt( self.filenames[idx] )
		if self.use_colorPCL:
			voxel = bnt2voxel_wColor( bnt_data, image_original, self.shape, self.center )
		else:
			voxel = bnt2voxel( bnt_data, self.shape, self.center )
		voxel = torch.Tensor( voxel )

		# parsing
		basename = os.path.basename( self.filenames[idx] )
		assert( imfile == (basename[:-len(self.suffix)]+'.png') )
		identity, poseclass, posecode, samplenum =  basename[:-len(self.suffix)].split('_')
		try:
			identity = int(identity[2:])
			poseclass = self.poseclassmap[poseclass]
			posecode = self.posecodemap[posecode]
		except:
			print( identity, poseclass, posecode )
			exit("parsing failed")
		labels = { 'id': identity,
					'pclass': poseclass,
					'pcode': posecode }

		# return
		if self.use_image:
			return voxel, labels, image
		else:
			return voxel, labels

class MultiPie( Dataset ):
	def __init__( self, root_dir, transform=None, cam_ids=None):
		self.filenames = []
		self.root_dir = root_dir
		self.transform = transform

		#cam_ids = [10, 41, 50, 51, 80, 81, 90, 110, 120, 130, 140, 190, 191, 200, 240]
		#cam_ids = [41, 50, 51, 80, 90, 130, 140, 190, 200]
		if cam_ids is None:
			cam_ids = [200, 190, 41, 50, 51, 140, 130, 80, 90]
		self.cam_map = {}
		for i, cam in enumerate(cam_ids):
			self.cam_map[cam] = i

		print('Loading MultiPie metadata...', end='')
		sys.stdout.flush()
		time_start = time.time()

		fname_cache = 'cache_multipie.txt'
		if os.path.exists(fname_cache):
			self.filenames = open(fname_cache).read().splitlines()
			print( 'restored from {} : {} samples'.format(fname_cache, len(self.filenames)) )
		else:
			path = os.path.join( root_dir, 'Multi-Pie', 'data' )
			self.filenames = [os.path.join(dirpath,f) for dirpath, dirnames, files in os.walk(path)
							for f in files if f.endswith('.png') ]
	
			print('{:.0f}sec, {} images found.'.format( time.time()-time_start, len(self.filenames)))
	
			with open(fname_cache, 'w') as f:
				for fname in self.filenames:
					f.write(fname+'\n')
			print( 'cached in {}'.format(fname_cache) )

		# filtering : 9 cams and 200 subjects
		self.filenames = [ f for f in self.filenames 
							if int(os.path.basename(f)[10:13]) in cam_ids ]
							#if int(os.path.basename(f)[10:13]) in cam_ids and int(os.path.basename(f)[:3]) < 201 ]
		self.subj_ids = sorted( set( [ int(os.path.basename(f)[:3]) for f in self.filenames ] ) )
		self.subj_map = {}
		for i, subj in enumerate( self.subj_ids ):
			self.subj_map[subj] = i
		print( '{} samples remain after filtering'.format( len(self.filenames) ) )
	

	def __len__( self ):
		return len( self.filenames )
	
	def __getitem__( self, idx ):
		basename = os.path.basename( self.filenames[idx] )
		identity, sessionNum, recordingNum, pose, illum =  basename[:-4].split('_')
		image = Image.open( self.filenames[idx] ).convert('L')
		pose = self.cam_map[int(pose)]
		if self.transform:
			image = self.transform(image)
		labels = { 'id': self.subj_map[int(identity)],
					'pose': pose,
					'illum': int(illum)}
		return image, labels 

class ShapeNet( Dataset ):
	def __init__( self, root_dir, transform=None, synsetId='chair'):
		self.dict_list = []
		self.root_dir = root_dir
		self.transform = transform

		fname_cache = 'cache_ShapeNet_'+synsetId+'.csv'

		# convert word to synsetID
		if not synsetId.isdigit():
			# read wordnet
			wordnet = {}
			with open('words.txt') as f_wordnet:
				for line in f_wordnet:
					tokens = line[:-1].split('\t')
					if not tokens[1] in wordnet:
						wordnet[ tokens[1] ] = tokens[0][1:]
			try:
				synsetId = wordnet[synsetId]
			except:
				exit( 'synsetId {} is not found in wordnet'.format(synsetId) )
			print( 'synsetId = {}'.format(synsetId) )

		# read shapenet split
		if os.path.exists(fname_cache):
			with open(fname_cache) as f:
				reader = csv.DictReader(f)
				for line in reader:
					self.dict_list.append(line)
			print( 'restored ShapeNet from {}'.format(fname_cache) )
		else:
			path_csv = os.path.join(root_dir,'all.csv')
			print( 'loading all.csv with synsetId {}...'.format(synsetId) )
			with open(path_csv) as f:
				reader = csv.DictReader(f)
				for line in reader:
					if int(line['synsetId']) == int(synsetId):
						self.dict_list.append(line)
			nRawSamples = len(self.dict_list)
			print( 'checking existence of actual models from all.csv...')
			self.dict_list = [ sample for sample in self.dict_list
							if os.path.exists( os.path.join( self.root_dir, 'ShapeNetCore.v2', sample['synsetId'], sample['modelId'],
												'models','model_normalized.solid.binvox' ) ) ]
			nExistingSamples = len(self.dict_list)
			print( '{} samples exist among {} samples from all.csv'.format(nExistingSamples,nRawSamples) )

			fieldnames = ['id', 'synsetId', 'subSynsetId', 'modelId', 'split']
			with open(fname_cache, 'w') as f:
				writer = csv.DictWriter(f, fieldnames=fieldnames)
				writer.writeheader()
				for sample in self.dict_list:
					writer.writerow(sample)
			print( 'cached in {}'.format(fname_cache) )

	def __len__( self ):
		return len( self.dict_list )
	
	def __getitem__( self, idx ):
		data = self.dict_list[idx]
		path_sample = os.path.join( self.root_dir, 'ShapeNetCore.v2', data['synsetId'], data['modelId'] )
		path_binvox = os.path.join( path_sample, 'models', 'model_normalized.solid.binvox' )
		voxel_data = read_binvox( path_binvox )
		voxel_data = np.expand_dims(voxel_data,0)
#		plot_voxel( voxel_data, save_file='sample_{}.png'.format(idx) )

		if self.transform:
			voxel_data = self.transform(voxel_data)
		voxel_data = torch.Tensor( voxel_data )
		labels = { 'id': data['id'],
					'synsetId': data['synsetId'] }
		return voxel_data, labels 


def sample_z( nSamples, nDims, isCuda=False ):
	z = torch.rand( nSamples, nDims )
	if isCuda:
		z = Variable(z.cuda())
	else:
		z = Variable(z)
	return z

def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)

def save_images(images, size, image_path):
	return imsave(images, size, image_path)

def imsave(images, size, path):
	image = np.squeeze(merge(images, size))
	return scipy.misc.imsave(path, image)

def merge(images, size):
	h, w = images.shape[1], images.shape[2]
	if (images.shape[3] in (3,4)):
		c = images.shape[3]
		img = np.zeros((h * size[0], w * size[1], c))
		for idx, image in enumerate(images):
			i = idx % size[1]
			j = idx // size[1]
			img[j * h:j * h + h, i * w:i * w + w, :] = image
		return img
	elif images.shape[3]==1:
		img = np.zeros((h * size[0], w * size[1]))
		for idx, image in enumerate(images):
			i = idx % size[1]
			j = idx // size[1]
			img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
		return img
	else:
		raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def generate_animation(path, num):
	images = []
	for e in range(num):
		img_name = path + '_epoch%03d' % (e+1) + '.png'
		images.append(imageio.imread(img_name))
	imageio.mimsave(path + '_generate_animation.gif', images, fps=5)

def loss_plot(hist, path='.', model_name='model', y_max=None, use_subplot=False, keys_to_show=[] ):
	try:
		x = range(len(hist['D_loss']))
	except:
		keys = hist.keys()
		losskey = [ k for k in keys if 'loss' in k ]
		losskey = losskey[0]
		try:
			x = range(len(hist[losskey]))
		except:
			print( 'loss plot failed, continuing...' )
			return

	if use_subplot:
		f, axarr = plt.subplots(2, sharex=True)
		
	plt.xlabel('Iter')
	plt.ylabel('Loss')
	plt.tight_layout()

	if len(keys_to_show) == 0:
		keys_to_show = hist.keys()
	for key,value in hist.iteritems():
		if 'time' in key or key not in keys_to_show:
			continue
		y = value
		if len(x) != len(y):
			print('[warning] loss_plot() found mismatching dimensions: {}'.format(key))
			continue
		if use_subplot and 'acc' in key:
			axarr[1].plot(x, y, label=key)
		elif use_subplot:
			axarr[0].plot(x, y, label=key)
		else:
			plt.plot(x, y, label=key)

	if use_subplot:
		axarr[0].legend(loc=1)
		axarr[0].grid(True)
		axarr[1].legend(loc=1)
		axarr[1].grid(True)
	else:
		plt.legend(loc=1)
		plt.grid(True)


	if y_max is not None:
		if use_subplot:
			x_min, x_max, y_min, _ = axarr[0].axis()
			axarr[0].axis( (x_min, x_max, -y_max/20, y_max) )
		else:
			x_min, x_max, y_min, _ = plt.axis()
			plt.axis( (x_min, x_max, -y_max/20, y_max) )

	path = os.path.join(path, model_name + '_loss.png')

	plt.savefig(path)

	plt.close()

def initialize_weights(net):
	for m in net.modules():
		if isinstance(m, nn.Conv2d):
			m.weight.data.normal_(0, 0.02)
			m.bias.data.zero_()
		elif isinstance(m, nn.ConvTranspose2d):
			m.weight.data.normal_(0, 0.02)
#			m.bias.data.zero_()
		elif isinstance(m, nn.Conv3d):
			nn.init.xavier_uniform(m.weight)
		elif isinstance(m, nn.ConvTranspose3d):
			nn.init.xavier_uniform(m.weight)
		elif isinstance(m, nn.Linear):
			m.weight.data.normal_(0, 0.02)
			m.bias.data.zero_()


class Flatten(nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()

	def forward(self, x):
		return x.view(x.size(0), -1)


class Inflate(nn.Module):
	def __init__(self, nDims2add):
		super(Inflate, self).__init__()
		self.nDims2add = nDims2add

	def forward(self, x):
		shape = x.size() + (1,)*self.nDims2add
		return x.view(shape)


def parse_args():
	desc = "plot loss"
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--fname_hist', type=str, default='', help='history path', required=True)
	parser.add_argument('--fname_dest', type=str, default='.', help='filename of png')
	return parser.parse_args()

if __name__ == '__main__':
	opts = parse_args()
	with open( opts.fname_hist ) as fhandle:
		history = pickle.load(fhandle)
		loss_plot( history, opts.fname_dest )
