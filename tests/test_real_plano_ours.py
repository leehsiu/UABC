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
import time

def make_size_divisible(img,stride):
	w,h,_ = img.shape

	w_new = w//stride*stride
	h_new = h//stride*stride

	return img[:w_new,:h_new,:]

def main():
	# ----------------------------------------
	# load kernels
	# ----------------------------------------
	#PSF_grid = np.load('./data/Schuler_PSF01.npz')['PSF']
	#PSF_grid = np.load('./data/Schuler_PSF_facade.npz')['PSF']
	#PSF_grid = np.load('./data/ZEMAX-AC254-075-A-new.npz')['PSF']
	PSF_grid = np.load('./data/Heidel_PSF_plano_small.npz')['PSF']
	#PSF_grid = np.load('./data/PSF.npz')['PSF']
	#print(PSF_grid.shape)

	PSF_grid = PSF_grid.astype(np.float32)

	gx,gy = PSF_grid.shape[:2]
	for xx in range(gx):
		for yy in range(gy):
			PSF_grid[xx,yy] = PSF_grid[xx,yy]/np.sum(PSF_grid[xx,yy],axis=(0,1))

	#PSF_grid = PSF_grid[:,1:-1,...]
	# ----------------------------------------
	# load model
	# ----------------------------------------
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = net(n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512],
					nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
	#model.load_state_dict(torch.load('/home/xiu/databag/deblur/usrnet_ZEMAX_finetune.pth'), strict=True)
	model.load_state_dict(torch.load('usrnet_plano_small.pth'), strict=True)
	model.eval()
	#model.train()
	for _, v in model.named_parameters():
		v.requires_grad = False
	#	v.requires_grad = False
	model = model.to(device)


	#img_L = cv2.imread('/home/xiu/workspace/dwgan/new_image/image/{}/AC254-075-A-ML-Zemax(ZMX).bmp'.format(img_id))
	#img_L = cv2.imread('/home/xiu/workspace/dwgan/Supplement_SimpleLensImaging_Heide2013/supplemental_material/images/single_lens/plano_convex_lens/office_clock/cap.jpg')
	img_L = cv2.imread('/home/xiu/workspace/dwgan/Supplement_SimpleLensImaging_Heide2013/supplemental_material/images/single_lens/plano_convex_lens/drain/cap.jpg')
	img_L = img_L.astype(np.float32)

	img_L = img_L[38:-39,74:-74]
	img_L = cv2.resize(img_L,dsize=None,fx=0.5,fy=0.5)
	#img_L = img_L[:,13:-14,:]
	#img_L = img_L.

	#img_L = util_deblur.blockConv2d(img_H,PSF_grid)
	img_E = np.zeros_like(img_L)

	img_E_deconv = []
	img_E_denoise = []
	for i in range(5):
		img_E_deconv.append(np.zeros_like(img_L))
		img_E_denoise.append(np.zeros_like(img_L))


	weight_E = np.zeros_like(img_L)

	patch_size = 7*152
	num_patch = 7
	p_size = patch_size//num_patch
	expand = PSF_grid.shape[2]//2

	#315*11 = 3492
	#         3465
	
	# remove 148x77 pixels




	#positional alpha-beta parameters for HQS
	#ab_numpy = np.ones((num_patch*num_patch,17,1,1),dtype=np.float32)*0.1
	#ab_numpy[:,0,:,:] = 0.01
	ab_numpy = np.loadtxt('ab_plano_small.txt').astype(np.float32).reshape(7,11,11,3)
	#ab_numpy = ab_numpy[:,1:-1,:,:]

	#ab_numpy = ab_numpy[...,None,None]
	ab = torch.tensor(ab_numpy,device=device,requires_grad=False)

	#save img_L

	t0 = time.time()
	#while running:
	for px_start in range(0,7-7+1,1):
		for py_start in range(0,11-7+1,1):

			#px_start = np.random.randint(0,PSF_grid.shape[0]+1-num_patch)
			#py_start = np.random.randint(0,PSF_grid.shape[1]+1-num_patch)

			PSF_patch = PSF_grid[px_start:px_start+num_patch,py_start:py_start+num_patch]

			# x = util.uint2single(patch_L)
			block_size = patch_size//num_patch
			patch_L = img_L[px_start*p_size:(px_start+num_patch)*p_size,py_start*p_size:py_start*p_size+num_patch*p_size,:]
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

			cd = F.softplus(ab[px_start:px_start+num_patch,py_start:py_start+num_patch])
			cd = cd.view(num_patch**2,2*5+1,3)

			x_E = model.forward_patchdeconv(x_blocky,k,cd,[num_patch,num_patch],patch_sz=patch_size//num_patch)
			x_E = x_E[:-1]

			patch_L = patch_L_wrap.astype(np.uint8)

			patch_E = util.tensor2uint(x_E[-1])
			patch_E_all = [util.tensor2uint(pp) for pp in x_E]

			#patch_E_z = np.hstack((patch_E_all[::2]))
			#patch_E_x = np.hstack((patch_E_all[1::2]))

			#patch_E_show = np.vstack((patch_E_z,patch_E_x))
			#if block_expand>0:
			#	show = np.hstack((patch_L[block_expand:-block_expand,block_expand:-block_expand],patch_E))
			#else:
			#	show = np.hstack((patch_L,patch_E))

	
			#get kernel
			for i in range(5):
				img_E_deconv[i][px_start*p_size:(px_start+num_patch)*p_size,py_start*p_size:py_start*p_size+num_patch*p_size,:] += patch_E_all[2*i]
				img_E_denoise[i][px_start*p_size:(px_start+num_patch)*p_size,py_start*p_size:py_start*p_size+num_patch*p_size,:] += patch_E_all[2*i+1]
			weight_E[px_start*p_size:(px_start+num_patch)*p_size,py_start*p_size:py_start*p_size+num_patch*p_size,:] += 1.0

			#cv2.imshow('stage',patch_E_show)
			#cv2.imshow('HL',show)
			#cv2.imshow('RGB',rgb)
			#key = cv2.waitKey(-1)
			#if key==ord('n'):
		#	break

	t1 = time.time()

	print(t1-t0)
	img_E = img_E/weight_E
	img_E_deconv = [pp/weight_E for pp in img_E_deconv]
	img_E_denoise = [pp/weight_E for pp in img_E_denoise]

	# img_L = img_L.astype(np.uint8)
	# img_E = img_E.astype(np.uint8)
	# img_E_deconv = img_E_deconv.astype(np.uint8)
	# img_E_denoise = img_E_denoise.astype(np.uint8)
	# cv2.imshow('imE',img_E)
	# cv2.imshow('imE_deconv',img_E_deconv)
	# cv2.imshow('imE_denoise',img_E_denoise)
	# cv2.imshow('imL',img_L)
	#for i in range(5):
		#zk = img_E_deconv[i]
	# print(i)
	xk = img_E_denoise[-1]
	# #zk = zk.astype(np.uint8)
	xk = xk.astype(np.uint8)
	cv2.imshow('xk',xk)
	cv2.waitKey(-1)
		# cv2.imwrite('/home/xiu/workspace/dwgan/new_image/image/ours-{}.png'.format(img_id),xk)






	


if __name__ == '__main__':

	main()
