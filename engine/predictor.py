import torch
import torch.optim
import torch.nn.functional as F
import cv2
import os.path
import time
import os
import glob
import numpy as np
from collections import OrderedDict
from datetime import datetime
from scipy.signal import convolve2d
from models.uabcnet import UABCNet as net
import matplotlib.pyplot as plt
import utils.utils_image as util
import utils.utils_deblur as util_deblur
import time


def main():
	# ----------------------------------------
	# load kernels
	# ----------------------------------------
	PSF_grid = np.load('./data/AC254-075-A-ML-Zemax(ZMX).npz')['PSF']
	
	PSF_grid = PSF_grid.astype(np.float32)

	gx,gy = PSF_grid.shape[:2]
	for xx in range(gx):
		for yy in range(gy):
			PSF_grid[xx,yy] = PSF_grid[xx,yy]/np.sum(PSF_grid[xx,yy],axis=(0,1))

	# ----------------------------------------
	# load model
	# ----------------------------------------
	stage = 8
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = net(n_iter=stage, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512],
					nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")

	model_code = 'iter17000'
	loaded_state = torch.load('/home/xiu/databag/deblur/models/ZEMAX/uabcnet_{}.pth'.format(model_code))
	model.load_state_dict(loaded_state, strict=True)

	model.eval()
	for _, v in model.named_parameters():
		v.requires_grad = False
	model = model.to(device)

	img_names = glob.glob('/home/xiu/databag/deblur/ICCV2021/suo_image/*/AC254-075-A-ML-Zemax(ZMX).bmp')
	img_names.sort()
	for img_id,img_name in enumerate(img_names):
		img_L = cv2.imread(img_name)
		img_L = img_L.astype(np.float32)
		W, H = img_L.shape[:2]
		num_patch = [6,8]
		#positional alpha-beta parameters for HQS
		ab_numpy = np.loadtxt('/home/xiu/databag/deblur/models/ZEMAX/ab_{}.txt'.format(model_code)).astype(np.float32).reshape(gx,gy,stage*2,3)
		ab = torch.tensor(ab_numpy,device=device,requires_grad=False)

		#save img_L

		t0 = time.time()

		px_start = 0
		py_start = 0

		PSF_patch = PSF_grid[px_start:px_start+num_patch[0],py_start:py_start+num_patch[1]]
		#block_expand = 1
		patch_L = img_L[px_start*W//gx:(px_start+num_patch[0])*W//gx,py_start*H//gy:(py_start+num_patch[1])*H//gy,:]

		p_W,p_H= patch_L.shape[:2]
		expand = max(PSF_grid.shape[2]//2,p_W//16)
		block_expand = expand
		patch_L_wrap = util_deblur.wrap_boundary_liu(patch_L,(p_W+block_expand*2,p_H+block_expand*2))
		#centralize
		patch_L_wrap = np.hstack((patch_L_wrap[:,-block_expand:,:],patch_L_wrap[:,:p_H+block_expand,:]))
		patch_L_wrap = np.vstack((patch_L_wrap[-block_expand:,:,:],patch_L_wrap[:p_W+block_expand,:,:]))
		x = util.uint2single(patch_L_wrap)
		x = util.single2tensor4(x)

		k_all = []
		for h_ in range(num_patch[1]):
			for w_ in range(num_patch[0]):
				k_all.append(util.single2tensor4(PSF_patch[w_,h_]))
		k = torch.cat(k_all,dim=0)

		[x,k] = [el.to(device) for el in [x,k]]

		ab_patch = F.softplus(ab[px_start:px_start+num_patch[0],py_start:py_start+num_patch[1]])
		cd = []
		for h_ in range(num_patch[1]):
			for w_ in range(num_patch[0]):
				cd.append(ab_patch[w_:w_+1,h_])
		cd = torch.cat(cd,dim=0)

		x_E = model.forward_patchwise(x,k,cd,num_patch,[W//gx,H//gy])
		x_E = x_E[...,block_expand:block_expand+p_W,block_expand:block_expand+p_H]

		patch_L = patch_L_wrap.astype(np.uint8)

		patch_E = util.tensor2uint(x_E)

		t1 = time.time()

		print('[{}/{}]: {} s per frame'.format(img_id,len(img_names),t1-t0))

		xk = patch_E
		xk = xk.astype(np.uint8)

		cv2.imshow('res',xk)
		cv2.imshow('input',patch_L.astype(np.uint8))

		key = cv2.waitKey(-1)
		if key==ord('q'):
			break



if __name__=='__main__':
	main()




