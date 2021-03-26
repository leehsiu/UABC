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

def make_size_divisible(img,stride):
	w,h,_ = img.shape

	w_new = w//stride*stride
	h_new = h//stride*stride

	return img[:w_new,:h_new,:]

def main():
	# ----------------------------------------
	# load kernels
	# ----------------------------------------
	PSF_grid = np.load('./data/Schuler_PSF01.npz')['PSF']
	#PSF_grid = np.load('./data/Schuler_PSF02.npz')['PSF']
	#PSF_grid = np.load('./data/Schuler_PSF03.npz')['PSF']
	#PSF_grid = np.load('./data/PSF.npz')['PSF']
	#print(PSF_grid.shape)
	
	PSF_grid = PSF_grid.astype(np.float32)

	gx,gy = PSF_grid.shape[:2]
	for xx in range(gx):
		for yy in range(gy):
			PSF_grid[xx,yy] = PSF_grid[xx,yy]/np.sum(PSF_grid[xx,yy],axis=(0,1))

	# ----------------------------------------
	# load model
	# ----------------------------------------
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = net(n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512],
					nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
	model.load_state_dict(torch.load('usrnet_bench.pth'), strict=True)
	model.eval()
	#model.train()
	for _, v in model.named_parameters():
		v.requires_grad = False
	#	v.requires_grad = False
	model = model.to(device)



	img_L = cv2.imread('/home/xiu/workspace/dwgan/MPI_data/bench/blurry.jpg')
	img_H = cv2.imread('/home/xiu/workspace/dwgan/MPI_data/bench/schuler.jpg')
	print(img_L.shape)
	#10x6
	#1097x730
	#109.7,12


	patch_size = 2*256
	num_patch = 2
	expand = PSF_grid.shape[2]//2
	b_size = patch_size//num_patch

	ab_numpy = np.loadtxt('ab_bench.txt').astype(np.float32)
	ab_numpy = ab_numpy[...,None,None]
	#ab_numpy[:,0] = 0.01
	#ab_numpy[:,1] = 0.01
	#ab_numpy[:,2] = 0.01

	ab = torch.tensor(ab_numpy,device=device,requires_grad=False)
	
	running = True


	while running:
		#alpha.beta
		#px_start = np.random.randint(0,PSF_grid.shape[0]//2+1)
		#py_start = np.random.randint(0,PSF_grid.shape[1]//2+1)
		px_start = 0
		py_start = 0

		PSF_patch = PSF_grid[px_start:px_start+num_patch,py_start:py_start+num_patch]

		# x = util.uint2single(patch_L)
		block_size = patch_size//num_patch
		patch_L = img_L[px_start*b_size:(px_start+num_patch)*b_size,py_start*b_size:py_start*b_size+patch_size,:]
		patch_H = img_H[px_start*b_size:(px_start+num_patch)*b_size,py_start*b_size:py_start*b_size+patch_size,:]
		#block_expand = expand*2
		block_expand = expand
		#block_expand = 1
		if block_expand > 0:
			patch_L_wrap = util_deblur.wrap_boundary_liu(patch_L,(patch_size+block_expand*2,patch_size+block_expand*2))
			#centralize
			patch_L_wrap = np.hstack((patch_L_wrap[:,-block_expand:,:],patch_L_wrap[:,:patch_size+block_expand,:]))
			patch_L_wrap = np.vstack((patch_L_wrap[-block_expand:,:,:],patch_L_wrap[:patch_size+block_expand,:,:]))
		else:
			patch_L_wrap = patch_L
		if block_expand>0:
			x = util.uint2single(patch_L_wrap)
		else:
			x = util.uint2single(patch_L)
		x_blocky = []
		for h_ in range(num_patch):
			for w_ in range(num_patch):
				x_blocky.append(x[w_*block_size:w_*block_size+block_size+block_expand*2,\
					h_*block_size:h_*block_size+block_size+block_expand*2:])	
		x_blocky = [util.single2tensor4(el) for el in x_blocky]
		x_blocky = torch.cat(x_blocky,dim=0)

		# x = util.single2tensor4(x)

		# x_blocky = torch.cat(torch.chunk(x,num_patch,dim=2),dim=0)
		# x_blocky = torch.cat(torch.chunk(x_blocky,num_patch,dim=3),dim=0)

		k_all = []
		for w_ in range(num_patch):
			for h_ in range(num_patch):
				k_all.append(util.single2tensor4(PSF_patch[h_,w_]))
		k = torch.cat(k_all,dim=0)

		[x_blocky,k] = [el.to(device) for el in [x_blocky,k]]

		x_E = model.forward_patchdeconv(x_blocky,k,ab,[num_patch,num_patch],patch_sz=patch_size//num_patch)
		x_E = x_E[:-1]

		patch_L = patch_L_wrap.astype(np.uint8)

		patch_E = util.tensor2uint(x_E[-1])
		patch_E_all = [util.tensor2uint(pp) for pp in x_E]
		patch_E_z = np.hstack((patch_E_all[::2]))
		patch_E_x = np.hstack((patch_E_all[1::2]))

		patch_E_show = np.vstack((patch_E_z,patch_E_x))
		if block_expand>0:
			show = np.hstack((patch_H,patch_L[block_expand:-block_expand,block_expand:-block_expand],patch_E))
		else:
			show = np.hstack((patch_H,patch_L,patch_E))

		#get kernel
		cv2.imshow('stage',patch_E_show)
		cv2.imshow('HL',show)
		key = cv2.waitKey(-1)
		if key==ord('n'):
			break




	


if __name__ == '__main__':

	main()
