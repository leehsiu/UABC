import torch
import torch.optim
import torch.nn.functional as F

import cv2
import os.path
import time
import os
import glob
import numpy as np
from datetime import datetime
from scipy.signal import convolve2d
from nuocnet.models.usrnet import USRNet as net
import matplotlib.pyplot as plt
import nuocnet.utils.utils_image as util
import nuocnet.utils.utils_deblur as util_deblur
from torch.utils.tensorboard import SummaryWriter

def load_kernels(kernel_path):
	kernels = []
	kernel_files = glob.glob(os.path.join(kernel_path,'*.npz'))
	kernel_files.sort()
	for kf in kernel_files:
		PSF_grid = np.load(kf)['PSF']
		PSF_grid = PSF_grid.astype(np.float32)
		kw,kh = PSF_grid.shape[:2]
		for w_ in range(kw):
			for h_ in range(kh):
				PSF_grid[w_,h_] = PSF_grid[w_,h_]/np.sum(PSF_grid[w_,h_],axis=(0,1))
		kernels.append(PSF_grid)
	return kernels

def get_kernels(kernels,n_patch):
	PSF0_idx = np.random.randint(len(kernels))
	loc = np.random.randint(4)
	w,h = kernels[PSF0_idx].shape[:2]
	#l,b,r,t
	t = np.random.randint(0,w-n_patch+1)
	l = np.random.randint(0,h-n_patch+1)
	if loc==0:
		PSF0 = kernels[PSF0_idx][:n_patch,l:l+n_patch]
	elif loc==1:
		PSF0 = kernels[PSF0_idx][t:t+n_patch,-n_patch:]
	elif loc==2:
		PSF0 = kernels[PSF0_idx][-n_patch:,l:l+n_patch]
	elif loc==3:
		PSF0 = kernels[PSF0_idx][t:t+n_patch,-n_patch:]
	print(loc,t,l,PSF0_idx)
	return PSF0

def rand_kernels(n_patch):
	PSF1 = np.zeros((n_patch,n_patch,25,25,3))
	for w_ in range(n_patch):
		for h_ in range(n_patch):
			PSF1[w_,h_,...,0] = util_deblur.gen_kernel()
			PSF1[w_,h_,...,1] = util_deblur.gen_kernel()
			PSF1[w_,h_,...,2] = util_deblur.gen_kernel()
	return PSF1


def main():
	root  = '/home/xiu/databag/deblur/pretrain/patchwise'
	logger = SummaryWriter(root)
	# ----------------------------------------
	# load kernels
	# ----------------------------------------
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	PSF_grid = np.load('./data/ZEMAX-AC254-075-A-new.npz')['PSF']
	PSF_grid = PSF_grid.astype(np.float32)
	kw,kh = PSF_grid.shape[:2]
	for w_ in range(kw):
		for h_ in range(kh):
			PSF_grid[w_,h_] = PSF_grid[w_,h_]/np.sum(PSF_grid[w_,h_],axis=(0,1))
	# ----------------------------------------
	# build
	# ----------------------------------------
	model = net(n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512],
					nb=3, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
	model.train()
	for _, v in model.named_parameters():
		v.requires_grad = True
	model = model.to(device)

	train_imgs = glob.glob('/home/xiu/databag/deblur/images/*/**.png',recursive=True)
	train_imgs.sort()

	#global model. 3x4
	n_epoch = 200
	n_stage = 5
	n_batch = 3
	w_patch = 128
	n_patch = 2
	#also 3x4,but different strategy
	
	ab_buffer = np.ones((1,1,2*n_stage+1,3),dtype=np.float32)*0.1
	ab_param = torch.tensor(ab_buffer,device=device,requires_grad=True)

	params = []
	params += [{"params":[ab_param],"lr":1e-4}]
	for key,value in model.named_parameters():
		params += [{"params":[value],"lr":1e-4}]
	optimizer = torch.optim.Adam(params,lr=1e-4)

	img_index = np.arange(len(train_imgs))

	global_iter = 0
	expand = PSF_grid.shape[2]//2*2
	
	#using this small PSF only.
	PSF = PSF_grid[1,2]
	
	for epoch in range(n_epoch):
		np.random.shuffle(img_index)
		for iteration in range(len(train_imgs)//n_batch):
			imgs = []
			for ii in range(n_batch):
				imgs.append(cv2.imread(train_imgs[img_index[iteration*n_batch+ii]]))
			global_iter += 1

			vis_H = []
			vis_L = []
			vis_E = []

			x = []
			y = []
			for img in imgs:
				w,h = img.shape[:2]
				for _ in range(n_patch*n_patch):
					x_start = np.random.randint(0,w-w_patch-expand*2+1)
					y_start = np.random.randint(0,h-w_patch-expand*2+1)
					patch_H = img[x_start:x_start+w_patch+expand*2,y_start:y_start+w_patch+expand*2]
					patch_L = util_deblur.uniformConv2d(patch_H,PSF)

					vis_H.append(patch_H[expand:-expand,expand:-expand])
					vis_L.append(patch_L[expand//2:-expand//2,expand//2:-expand//2])
					x_L = util.uint2single(patch_L)
					x_L = util.single2tensor4(x_L)
					x_gt = util.uint2single(patch_H[expand:-expand,expand:-expand])
					x_gt = util.single2tensor4(x_gt)
					y.append(x_L)
					x.append(x_gt)
			
			ab = F.softplus(ab_param)
			loss = 0
			k = util.single2tensor4(PSF)
			k = k.to(device)
			for i in range(n_batch*n_patch*n_patch):
				yy = y[i].to(device)
				xx = x[i].to(device)
				#xE = model.forward_patchdeconv(yy,kk,ab[0],[1,1],w_patch)
				xE = model.forward_globaldeconv(yy,k,ab[0],w_patch)
				loss += F.l1_loss(xE[-2],xx)
				patch_E = xE[-2].clone().detach().cpu()
				vis_E.append(util.tensor2uint(patch_E))
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if global_iter % 10 ==0 :
				print('iter {}: loss{}.'.format(global_iter,loss.item()))
				logger.add_scalar('train-loss',loss.item(),global_iter)
				for i in range(n_batch):
					show1 = np.hstack((vis_H[i],vis_L[i],vis_E[i]))
					logger.add_image('show-{}'.format(i),util.uint2tensor3(show1[:,:,::-1]))
					logger.flush()	
		ab_numpy = ab.detach().cpu().numpy()[:,:,0,0]
		ab_numpy = ab_numpy.flatten()
		torch.save(model.state_dict(),os.path.join(root,'usrnet_patchwise1x2_epoch{}.pth'.format(epoch)))
		np.savetxt(os.path.join(root,'patchwise_ours.txt'),ab_numpy)


	#	torch.save(model.state_dict(),'usrnet_bench.pth')
	#	np.savetxt('ab_bench.txt',ab_numpy)



if __name__ == '__main__':

	main()
