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
	#PSF_grid = np.load('./data/Schuler_PSF01.npz')['PSF']
	#PSF_grid = np.load('./data/Schuler_PSF_facade.npz')['PSF']
	PSF_grid = np.load('./data/ZEMAX-AC254-075-A-new.npz')['PSF']
	#PSF_grid = np.load('./data/Schuler_PSF03.npz')['PSF']
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

	loaded_state = torch.load('./usrnet_ZEMAX.pth')
	#strip_state = strip_prefix_if_present(loaded_state,prefix="p.")
	model.load_state_dict(loaded_state, strict=True)

	model.eval()
	#model.train()
	for _, v in model.named_parameters():
		v.requires_grad = False
	#	v.requires_grad = False
	model = model.to(device)


	for img_id in range(100):
		img_L = cv2.imread('/home/xiu/workspace/UABC/ICCV2021/video/{:08d}.bmp'.format(img_id))
		img_L = img_L.astype(np.float32)

		img_E = np.zeros_like(img_L)

		weight_E = np.zeros_like(img_L)

		patch_size = 2*128
		num_patch = 2
		p_size = patch_size//num_patch
		expand = PSF_grid.shape[2]

		ab_numpy = np.loadtxt('ab_ZEMAX.txt').astype(np.float32).reshape(6,8,16,3)
		ab = torch.tensor(ab_numpy,device=device,requires_grad=False)

		#save img_L

		t0 = time.time()
		#while running:
		for px_start in range(0,6-2+1,2):
			for py_start in range(0,8-2+1,2):

				PSF_patch = PSF_grid[px_start:px_start+num_patch,py_start:py_start+num_patch]

				patch_L = img_L[px_start*p_size:(px_start+num_patch)*p_size,py_start*p_size:py_start*p_size+num_patch*p_size,:]
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

				x = util.single2tensor4(x)

				# x_blocky = torch.cat(torch.chunk(x,num_patch,dim=2),dim=0)
				# x_blocky = torch.cat(torch.chunk(x_blocky,num_patch,dim=3),dim=0)

				k_all = []
				for w_ in range(num_patch):
					for h_ in range(num_patch):
						k_all.append(util.single2tensor4(PSF_patch[h_,w_]))
				k = torch.cat(k_all,dim=0)

				[x,k] = [el.to(device) for el in [x,k]]

				cd = F.softplus(ab[px_start:px_start+num_patch,py_start:py_start+num_patch])
				cd = cd.view(num_patch**2,2*8,3)

				x_E = model.forward_patchwise(x,k,cd,[num_patch,num_patch],[patch_size//num_patch,patch_size//num_patch])

				patch_L = patch_L_wrap.astype(np.uint8)

				patch_E = util.tensor2uint(x_E)
				patch_E_all = [util.tensor2uint(pp) for pp in x_E]

				#patch_E_z = np.hstack((patch_E_all[::2]))
				#patch_E_x = np.hstack((patch_E_all[1::2]))

				#patch_E_show = np.vstack((patch_E_z,patch_E_x))
				#if block_expand>0:
				#	show = np.hstack((patch_L[block_expand:-block_expand,block_expand:-block_expand],patch_E))
				#else:
				#	show = np.hstack((patch_L,patch_E))

		
				#get kernel
				for i in range(8):
					img_E_deconv[i][px_start*p_size:(px_start+num_patch)*p_size,py_start*p_size:py_start*p_size+num_patch*p_size,:] += patch_E_all[-2][expand:-expand,expand:-expand]
					img_E_denoise[i][px_start*p_size:(px_start+num_patch)*p_size,py_start*p_size:py_start*p_size+num_patch*p_size,:] += patch_E_all[-1][expand:-expand,expand:-expand]
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

		# print(i)
		xk = img_E_denoise[-1]
		# #zk = zk.astype(np.uint8)
		xk = xk.astype(np.uint8)
		#cv2.imwrite('/home/xiu/workspace/UABC/ICCV2021/video_deblur/{:08d}.png'.format(img_id),xk)
		cv2.imshow('xx',xk)
		cv2.imshow('img_L',img_L.astype(np.uint8))
		cv2.waitKey(-1)






	


if __name__ == '__main__':

	main()
