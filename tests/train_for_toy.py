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
	#PSF_grid = np.load('./data/Schuler_PSF01.npz')['PSF']
	#PSF_grid = np.load('./data/Schuler_PSF_bench.npz')['PSF']
	PSF_grid = np.load('./data/Heidel_PSF_toy.npz')['PSF']
	#PSF_grid = np.load('./data/ZEMAX-LA1608.npz')['PSF']
	#PSF_grid = np.load('./data/Schuler_PSF03.npz')['PSF']
	#PSF_grid = np.load('./data/PSF.npz')['PSF']
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
	#model.train()
	#model.load_state_dict(torch.load('./data/usrgan.pth'), strict=True)
	model.load_state_dict(torch.load('./data/usrnet.pth'), strict=True)
	#model.load_state_dict(torch.load('/home/xiu/databag/deblur/usrnet_ours_epoch10.pth'), strict=True)
	for _, v in model.named_parameters():
		v.requires_grad = True
	model = model.to(device)

	imgs = glob.glob('/home/xiu/databag/deblur/images/*/**.png',recursive=True)
	imgs.sort()

	patch_size = 2*118
	num_patch = 2
	expand = PSF_grid.shape[2]//2

	#positional alpha-beta parameters for HQS
	stage = 8
	ab_buffer = np.ones((gx,gy,2*stage+1,3),dtype=np.float32)*0.1
	ab_buffer[:,:,0,:] = 0.01
	ab = torch.tensor(ab_buffer,device=device,requires_grad=True)

	params = []
	params += [{"params":[ab],"lr":1e-4}]
	for key,value in model.named_parameters():
		params += [{"params":[value],"lr":1e-4}]

	optimizer = torch.optim.Adam(params,lr=1e-4)


	running = True

	while running:
		#alpha.beta
		img_idx = np.random.randint(len(imgs))
		img = imgs[img_idx]
		img_H = cv2.imread(img)
		w,h = img_H.shape[:2]

		mode = np.random.randint(5)
		px_start = np.random.randint(0,gx-num_patch+1)
		py_start = np.random.randint(0,gy-num_patch+1)
		if mode==0:
			px_start = 0
		if mode==1:
			px_start = gx-num_patch
		if mode==2:
			py_start = 0
		if mode==3:
			py_start = gy-num_patch


		x_start = np.random.randint(0,w-patch_size-expand*2+1)
		y_start = np.random.randint(0,h-patch_size-expand*2+1)
		PSF_patch = PSF_grid[px_start:px_start+num_patch,py_start:py_start+num_patch]

		patch_H = img_H[x_start:x_start+patch_size+expand*2,y_start:y_start+patch_size+expand*2]
		patch_L = util_deblur.blockConv2d(patch_H,PSF_patch,expand)

		block_size = patch_size//num_patch

		block_expand = max(patch_size//16,expand)
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


		k_all = []
		for w_ in range(num_patch):
			for h_ in range(num_patch):
				k_all.append(util.single2tensor4(PSF_patch[h_,w_]))
		k = torch.cat(k_all,dim=0)
		x_gt = util.uint2single(patch_H[expand:-expand,expand:-expand])
		x_gt = util.single2tensor4(x_gt)

		[x_blocky,x_gt,k] = [el.to(device) for el in [x_blocky,x_gt,k]]
		
		#cd = F.softplus(ab[px_start:px_start+num_patch,py_start:py_start+num_patch].reshape(num_patch**2,2*stage+1,1,1))
		#for n_iter in range(optim_iter):
		cd = F.softplus(ab[px_start:px_start+num_patch,py_start:py_start+num_patch])
		cd = cd.view(num_patch**2,2*stage+1,3)
		x_E = model.forward_patchdeconv(x_blocky,k,cd,[num_patch,num_patch],patch_sz=patch_size//num_patch)
		loss = 0
		#for xx in x_E[::2]:
		loss = F.l1_loss(x_E[-2],x_gt)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print('loss {}'.format(loss.item()))

		x_E = x_E[:-1]

		patch_L = patch_L_wrap.astype(np.uint8)
		patch_E = util.tensor2uint(x_E[-1])
		patch_E_all = [util.tensor2uint(pp) for pp in x_E]
		patch_E_z = np.hstack((patch_E_all[::2]))
		patch_E_x = np.hstack((patch_E_all[1::2]))

		patch_E_show = np.vstack((patch_E_z,patch_E_x))
		if block_expand>0:
			show = np.hstack((patch_H[expand:-expand,expand:-expand],patch_L[block_expand:-block_expand,block_expand:-block_expand],patch_E))
		else:
			show = np.hstack((patch_H[expand:-expand,expand:-expand],patch_L,patch_E))

		#get kernel
		#cv2.imshow('stage',patch_E_show)
		#rgb = np.hstack((patch_E[:,:,0],patch_E[:,:,1],patch_E[:,:,2]))
		cv2.imshow('HL',show)
		#cv2.imshow('RGB',rgb)
		key = cv2.waitKey(1)

		if key==ord('q'):
			running = False
			break

	ab_numpy = ab.detach().cpu().numpy().flatten()#.reshape(-1,2*stage+1)
	torch.save(model.state_dict(),'usrnet_toy_ours.pth')
	np.savetxt('ab_toy_ours.txt',ab_numpy)



if __name__ == '__main__':

	main()
