import argparse, os
import numpy as np
import visdom
import dataIO

import pdb

def parse_opts():
	desc = "visualize npy voxels of prob to visdom"
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--dir_npy', type=str, default='./', help='directory that contains npy files')
	parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
	parser.add_argument('--epoch_from', type=int, default=0, help='epoch to start plotting from')
	parser.add_argument('--epoch_to', type=int, default=-1, help='epoch to end plotting')
	parser.add_argument('--epoch_every', type=int, default=1, help='plot every N epochs')
 
	return check_opts(parser.parse_args())

"""checking arguments"""
def check_opts(opts):
	return opts

def main():

	opts = parse_opts()
	if opts is None:
		exit()
	vis = visdom.Visdom()
	
	fnames = [os.path.join(dirpath,f) for dirpath, dirnames, files in os.walk(opts.dir_npy)
							for f in files if f.endswith('.npy') ]
	fnames.sort()
	if opts.epoch_to < 0:
		opts.epoch_to = len(fnames)
	for iF in range(len(fnames)//opts.epoch_every):
		fname = fnames[iF*opts.epoch_every]
		print( 'loading from {}'.format(fname) )
		epoch = int(fname[-12:-9])
		if epoch < opts.epoch_from or opts.epoch_to < epoch:
			continue
		g_objects = np.load(fname)
	
#	id_ch = np.random.randint(0, batch_size, 4)
		id_ch = [0,1,2,3]
		for i in range(4):
			if g_objects[id_ch[i]].max() > 0.5:
				print( 'plotting epoch {} sample {}...'.format(fname, i) )
				dataIO.plotVoxelVisdom(np.squeeze(g_objects[id_ch[i]]>0.5), vis, '_'.join(map(str,[epoch,i])))
			else:
				print( 'max={}'.format(g_objects[id_ch[i]].max()) )
	

if __name__ == '__main__':
	main()
