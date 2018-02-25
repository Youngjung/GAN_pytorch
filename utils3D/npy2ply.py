import argparse, os
import numpy as np
import visdom
from visualize import plot_voxel
from multiprocessing import Process
from plyfile import PlyData, PlyElement

import pdb

def parse_opts():
	desc = "visualize npy voxels of prob to visdom"
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--dir_npy', type=str, default='', help='directory that contains npy files')
	parser.add_argument('--dir_dest', type=str, default='', help='directory to put result png files')
	parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
	parser.add_argument('--epoch_from', type=int, default=0, help='epoch to start plotting from')
	parser.add_argument('--epoch_to', type=int, default=-1, help='epoch to end plotting')
	parser.add_argument('--epoch_every', type=int, default=1, help='plot every N epochs')
	parser.add_argument('--fname', type=str, default='', help='single file to visualize')
	parser.add_argument('--thresh', type=float, default=0.5, help='single file to visualize')
 
	return check_opts(parser.parse_args())

"""checking arguments"""
def check_opts(opts):
	return opts

def main():

	opts = parse_opts()
	if opts is None:
		exit()

	if len(opts.fname)==0 and len(opts.dir_npy)==0:
		print( 'Target is not provided' )
		exit()

	if len(opts.dir_npy) == 0:
		opts.dir_npy = os.path.dirname( opts.fname )

	if len(opts.dir_dest) == 0:
		opts.dir_dest = os.path.join( opts.dir_npy, 'ply' )

	if not os.path.exists( opts.dir_dest ):
		os.makedirs( opts.dir_dest )
	
	if len(opts.fname)>0:
		fnames = [opts.fname]
		try:
			opts.epoch_to = int(opts.fname[-12:-9])
		except:
			print('ignoring epoch...'+opts.fname)
	else:
		fnames = [os.path.join(dirpath,f) \
					for dirpath, dirnames, files in os.walk(opts.dir_npy)
							for f in files if f.endswith('.npy') ]
		fnames.sort()
	for iF in range(len(fnames)//opts.epoch_every):
		fname = fnames[iF*opts.epoch_every]
		print( 'loading from {}'.format(fname) )
		try:
			epoch = int(fname[-12:-9])
			if epoch < opts.epoch_from or opts.epoch_to < epoch:
				continue
		except:
			print('ignoring epoch...'+fname)
			epoch=os.path.basename(fname)[:-4]

		g_objects = np.load(fname)
	
		# non-colored results have 4 dims due to squeeze()
		if g_objects.ndim == 4:
			for i in range(g_objects.shape[0]):
				if g_objects[i].max() > opts.thresh:
					print( 'plotting epoch {} sample {}...'.format(epoch, i) )
					binarized = g_objects[i]>opts.thresh
					ind = np.nonzero( binarized )
					indnp = np.array(ind)
					list_of_tuples = list(map(tuple,indnp.transpose()))
					vertex = np.array( list_of_tuples, dtype=[('x','f4'),('y','f4'),('z','f4')])
	
					el = PlyElement.describe( vertex, 'vertex' )
					ply_filename = os.path.join(opts.dir_dest,'_'.join(map(str,[epoch,i]))) + '.ply'
					PlyData([el]).write( ply_filename )
				else:
					print( 'max={}'.format(g_objects[i].max()) )
		elif g_objects.ndim == 5:
			for i in range(g_objects.shape[0]):
				if g_objects[i,0].max() > opts.thresh:
					print( 'plotting epoch {} sample {}...'.format(epoch, i) )
					binarized = g_objects[i,0]>opts.thresh
					ind = np.nonzero( binarized )
					r = tuple(g_objects[i,1,x,y,z]*255 for (x,y,z) in zip(ind[0],ind[1],ind[2]))
					g = tuple(g_objects[i,2,x,y,z]*255 for (x,y,z) in zip(ind[0],ind[1],ind[2]))
					b = tuple(g_objects[i,3,x,y,z]*255 for (x,y,z) in zip(ind[0],ind[1],ind[2]))
					ind = ind + (r,) + (g,) + (b,)
					indnp = np.array(ind)
					list_of_tuples = list(map(tuple,indnp.transpose()))
					vertex = np.array( list_of_tuples,
						dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])

					el = PlyElement.describe( vertex, 'vertex' )
					ply_filename = os.path.join(opts.dir_dest,'_'.join(map(str,[epoch,i]))) + '.ply'
					PlyData([el],text=True).write( ply_filename )
				else:
					print( 'max={}'.format(g_objects[i].max()) )
		else:
			error("not supported dims : {}".format(g_objects.ndim))
	

if __name__ == '__main__':
	main()
