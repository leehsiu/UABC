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

def strip_prefix_if_present(state_dict, prefix):
	keys = sorted(state_dict.keys())
	#if not all(key.startswith(prefix) for key in keys):
	#    return state_dict
	stripped_state_dict = OrderedDict()
	for key, value in state_dict.items():
		if key.startswith(prefix):
			stripped_state_dict[key.replace(prefix, "")] = value
	return stripped_state_dict

def make_size_divisible(img,stride):
	w,h,_ = img.shape

	w_new = w//stride*stride
	h_new = h//stride*stride

	return img[:w_new,:h_new,:]

def main():
	# ----------------------------------------
	# load kernels
	# ----------------------------------------
	#PSF_grid = np.load('./data/AC254-075-A-ML-Zemax(ZMX).npz')['PSF']
	PSF_grid = np.load('./data/Schuler_PSF_bench.npz')['PSF']
	
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

	model_code = 'iter4700'
	loaded_state = torch.load('/home/xiu/databag/deblur/models/bench/uabcnet_{}.pth'.format(model_code))
	#strip_state = strip_prefix_if_present(loaded_state,prefix="p.")
	model.load_state_dict(loaded_state, strict=True)

	model.eval()
	for _, v in model.named_parameters():
		v.requires_grad = False
	model = model.to(device)
	for img_id in range(1,237):
	#for img_id in range(1,12):
		#img_L = cv2.imread('/home/xiu/workspace/UABC/ICCV2021/video1-3/res/2_{:03d}.bmp'.format(img_id))
		#img_L = cv2.imread('/home/xiu/workspace/UABC/ICCV2021/video/{:08d}.bmp'.format(img_id))
		#img_L = cv2.imread('/home/xiu/databag/deblur/ICCV2021/suo_image/{}/AC254-075-A-ML-Zemax(ZMX).bmp'.format(img_id))
		#img_L = cv2.imread('/home/xiu/workspace/UABC/ICCV2021/ResolutionChart/Reso.bmp')
		img_L = cv2.imread('/home/xiu/databag/deblur/ICCV2021/MPI_data/bench/blurry.jpg')
		img_L = img_L.astype(np.float32)
		img_L = np.pad(img_L,((1,1),(61,62),(0,0)),mode='edge')

		W, H = img_L.shape[:2]
		print(gx,gy)
		num_patch = [6,10]
		#positional alpha-beta parameters for HQS
		#ab_numpy = np.ones((num_patch*num_patch,17,1,1),dtype=np.float32)*0.1
		#ab_numpy[:,0,:,:] = 0.01
		ab_numpy = np.loadtxt('/home/xiu/databag/deblur/models/bench/ab_{}.txt'.format(model_code)).astype(np.float32).reshape(gx,gy,stage*2,3)
		#ab_numpy = ab_numpy[:,1:-1,:,:]

		#ab_numpy = ab_numpy[...,None,None]
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

		#patch_E_z = np.hstack((patch_E_all[::2]))
		#patch_E_x = np.hstack((patch_E_all[1::2]))

		#patch_E_show = np.vstack((patch_E_z,patch_E_x))
		#if block_expand>0:
		#	show = np.hstack((patch_L[block_expand:-block_expand,block_expand:-block_expand],patch_E))
		#else:
		#	show = np.hstack((patch_L,patch_E))


		#cv2.imshow('stage',patch_E_show)
		#cv2.imshow('HL',show)
		#cv2.imshow('RGB',rgb)
		#key = cv2.waitKey(-1)
		#if key==ord('n'):
		#	break

		t1 = time.time()

		print(t1-t0)

		# print(i)
		xk = patch_E
		# #zk = zk.astype(np.uint8)
		xk = xk.astype(np.uint8)
		#cv2.imwrite('/home/xiu/workspace/UABC/ICCV2021/new_image/image/ours-{}.png'.format(img_id),xk)
		#cv2.imwrite('/home/xiu/workspace/UABC/ICCV2021/video_deblur/{:08d}.png'.format(img_id),xk)
		#cv2.imwrite('/home/xiu/workspace/UABC/ICCV2021/cap_result/1_{:03d}.png'.format(img_id),xk)
		cv2.imshow('xx',xk)
		cv2.imshow('img_L',patch_L.astype(np.uint8))
		key = cv2.waitKey(-1)
		if key==ord('q'):
			break



if __name__=='__main__':
	main()




