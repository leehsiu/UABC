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
	# ----------------------------------------
	# load kernels
	# ----------------------------------------
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	all_PSFs = load_kernels('./data')
	# ----------------------------------------
	# build
	# ----------------------------------------
	model = net(n_iter=8, h_nc=64, in_nc=3, out_nc=3, nc=[64, 128, 256, 512],
					nb=3, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
	model.train()
	for _, v in model.named_parameters():
		v.requires_grad = True
	model = model.to(device)

	train_imgs = glob.glob('/home/xiu/databag/deblur/images/*/**.png',recursive=True)
	train_imgs.sort()

	n_stage = 5
	n_batch = 9
	n_epoch = 200

	w_patch = 128
	n_patch = 1

	ab_buffer = np.ones((n_batch,n_patch*n_patch,2*n_stage+1,3),dtype=np.float32)*0.1
	ab_param = torch.tensor(ab_buffer,device=device,requires_grad=False)

	params = []
	#params += [{"params":[ab_param],"lr":1e-4}]
	for key,value in model.named_parameters():
		params += [{"params":[value],"lr":1e-4}]
	optimizer = torch.optim.Adam(params,lr=1e-4)

	img_index = np.arange(len(train_imgs))

	global_iter = 0

	PSFs = []
	for i in range(n_batch):
	#	if i % 2==0:
		PSFs.append(all_PSFs[0][0:1,0:1])
	#	else:
	#		PSFs.append(rand_kernels(n_patch))
	expands = []
	for i in range(n_batch):
		expands.append(PSFs[i].shape[2]//2)
	
	for epoch in range(n_epoch):
		np.random.shuffle(img_index)
		for iteration in range(len(train_imgs)//n_batch):
			imgs = []
			for ii in range(n_batch):
				imgs.append(cv2.imread(train_imgs[img_index[iteration*n_batch+ii]]))

			global_iter += 1
			if global_iter % 100==0:
				PSFs = []
				for i in range(n_batch):
					PSFs.append(all_PSFs[0][0:1,0:1])
				expands = []
				for i in range(n_batch):
					expands.append(PSFs[i].shape[2]//2)

				#get new kernel.
			x = []
			y = []
			for img,expand,PSF in zip(imgs,expands,PSFs):
				w,h = img.shape[:2]
				x_start = np.random.randint(0,w-w_patch*n_patch-expand*2+1)
				y_start = np.random.randint(0,h-w_patch*n_patch-expand*2+1)
				patch_H = img[x_start:x_start+w_patch*n_patch+expand*2,y_start:y_start+w_patch*n_patch+expand*2]
				patch_L = util_deblur.uniformConv2d(patch_H,PSF)

				x_L = util.uint2single(patch_L)
				x_L = util.single2tensor4(x_L)
				x_gt = util.uint2single(patch_H[expand:-expand,expand:-expand])
				x_gt = util.single2tensor4(x_gt)
				y.append(x_L)
				x.append(x_gt)

			ab = F.softplus(ab_param)
			loss = 0
			x_E = []
			for i in range(n_batch):
				yy = y[i].to(device)
				xx = x[i].to(device)
				xE = model.forward_patchtranslate(yy,ab[i])
				loss += F.l1_loss(xE,xx)
				x_E.append(util.tensor2uint(xE))
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			print('loss {}'.format(loss.item()))

			gt = util.tensor2uint(x[-1])
			# patch_E_all = [util.tensor2uint(pp) for pp in x_E]
			# patch_E_z = np.hstack((patch_E_all[::2]))
			# patch_E_x = np.hstack((patch_E_all[1::2]))
			# patch_E_show = np.vstack((patch_E_z,patch_E_x))
			cv2.imshow('res',np.hstack((gt,x_E[-1])))
			cv2.waitKey(1)

		ab_numpy = ab.detach().cpu().numpy()[:,:,0,0]
		torch.save(model.state_dict(),'usrnet_bench.pth')
		np.savetxt('ab_bench.txt',ab_numpy)



if __name__ == '__main__':
	main()
