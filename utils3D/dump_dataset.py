from utils import Bosphorus
from torch.utils.data import DataLoader

dataloader = DataLoader(Bosphorus('data/Bosphorus'),batch_size=64,shuffle=False)
for iB, (x_,_) in enumerate(dataloader):
	if iB == len(dataloader.dataset)//64:
		break
	fname = 'Bosphorus_temp/sample_{}.npy'.format(iB)
	x_.numpy().dump(fname)

