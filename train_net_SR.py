import torch
import torch.optim
import torch.nn.functional as F

import cv2
import os.path
import time
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import utils.utils_image as util
import utils.utils_deblur as util_deblur
import utils.utils_psf as util_psf
from models.uabcnet import UABCNet as net

def draw_training_pair(image_H,psf,sf,patch_num,patch_size,image_L=None):
	#if no image_L is provided
	w,h = image_H.shape[:2]
	gx,gy = psf.shape[:2]

	mode = np.random.randint(5)
	px_start = np.random.randint(0,gx-patch_num[0]+1)
	py_start = np.random.randint(0,gy-patch_num[1]+1)
	if mode==0:
		px_start = 0
	if mode==1:
		px_start = gx-patch_num[0]
	if mode==2:
		py_start = 0
	if mode==3:
		py_start = gy-patch_num[1]

	psf_patch = psf[px_start:px_start+patch_num[0],py_start:py_start+patch_num[1]]
	patch_size_H = [patch_size[0]*sf,patch_size[1]*sf]

	if image_L is None:
		expand = psf.shape[2]//2
		x_start = np.random.randint(0,w-patch_size_H[0]*patch_num[0]-expand*2+1)
		y_start = np.random.randint(0,h-patch_size_H[1]*patch_num[1]-expand*2+1)
		patch_H = img_H[x_start:x_start+patch_size_H[0]*patch_num[0]+expand*2,\
		y_start:y_start+patch_size_H[1]*patch_num[1]+expand*2]
		patch_L = util_deblur.blockConv2d(patch_H,PSF_patch,expand)
		block_expand = max(patch_size_H[0]//8,expand)
		block_expand = np.ceil(block_expand/sf)*sf
		patch_L_wrap = util_deblur.wrap_boundary_liu(patch_L,(patch_size[0]*patch_num[0]+block_expand*2,patch_size[1]*patch_num[1]+block_expand*2))
		patch_L_wrap = np.hstack((patch_L_wrap[:,-block_expand:,:],patch_L_wrap[:,:patch_size[1]*patch_num[1]+block_expand,:]))
		patch_L_wrap = np.vstack((patch_L_wrap[-block_expand:,:,:],patch_L_wrap[:patch_size[0]*patch_num[0]+block_expand,:,:]))
		patch_L = patch_L_wrap[::sf,::sf,:]
		patch_H = patch_H[expand:-expand,expand:-expand]
		block_expand = block_expand // sf
	else:
		x_start = px_start * patch_size_H[0]
		y_start = py_start * patch_size_H[1]
		patch_H = image_H[x_start:x_start+patch_size_H[0]*patch_num[0],\
			y_start:y_start+patch_size_H[1]*patch_num[1]]
		x_start = px_start * patch_size[0]
		y_start = py_start * patch_size[1]
		patch_L = image_L[x_start:x_start+patch_size[0]*patch_num[0],\
			y_start:y_start+patch_size[1]*patch_num[1]]
		block_expand = patch_size[0]//8
		block_expand = np.ceil(block_expand/sf)*sf
		patch_L_wrap = util_deblur.wrap_boundary_liu(patch_L,(patch_size[0]*patch_num[0]+block_expand*2,patch_size[1]*patch_num[1]+block_expand*2))
		patch_L_wrap = np.hstack((patch_L_wrap[:,-block_expand:,:],patch_L_wrap[:,:patch_size[1]*patch_num[1]+block_expand,:]))
		patch_L_wrap = np.vstack((patch_L_wrap[-block_expand:,:,:],patch_L_wrap[:patch_size[0]*patch_num[0]+block_expand,:,:]))
		patch_L = patch_L_wrap[::sf,::sf,:]

	return patch_L,patch_H,patch_psf,px_start,py_start,block_expand

def main():

	#0. global config
	#scale factor
	sf = 4	
	stage = 8
	patch_size = [128,128]
	patch_num = [2,2]

	#1. local PSF
	#shape: grid_x, grid_y,kw_x,kw_y,c
	PSF_grid = np.load('./data/AC254-075-A-ML-Zemax(ZMX).npz')['PSF']
	PSF_grid = util_psf.normalize_PSF(PSF_grid)
	gx,gy = PSF_grid.shape[:2]

	#2. local model
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = net(n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512],
					nb=2,sf=sf, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
	model.proj.load_state_dict(torch.load('./data/usrnet_pretrain.pth'),strict=True)
	model.train()
	for _, v in model.named_parameters():
		v.requires_grad = True
	model = model.to(device)

	#positional lambda, mu for HQS, set as free trainable parameters here.
	ab_buffer = np.ones((gx,gy,2*stage,3),dtype=np.float32)*0.1
	ab = torch.tensor(ab_buffer,device=device,requires_grad=True)

	params = []
	params += [{"params":[ab],"lr":0.0005}]
	for key,value in model.named_parameters():
		params += [{"params":[value],"lr":0.0001}]
	optimizer = torch.optim.Adam(params,lr=0.0001,betas=(0.9,0.999))
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1000,gamma=0.9)


	#3.load training data
	imgs = glob.glob('/home/xiu/databag/deblur/images/*/**.png',recursive=True)
	imgs.sort()

	


	global_iter = 0
	N_maxiter = 200000


	#def get_train_pairs()

	for i in range(N_maxiter)

		#draw random image.
		img_idx = np.random.randint(len(imgs))
		img = imgs[img_idx]
		img_H = cv2.imread(img)
		#img_L = cv2.imread(img_L)


		#draw random patch from image
		#a. without img_L
		patch_L,patch_H,px_start,py_start,block_expand = draw_training_pair(img_H,PSF_grid,sf,patch_num,patch_size)
		#b.	with img_L
		#patch_L,patch_H,px_start,py_start = draw_training_pair(img_H,PSF_grid,sf,patch_num,patch_size,img_L)

		x = util.uint2single(patch_L)
		x = util.single2tensor4(x)
		x_gt = util.uint2single(patch_H)
		x_gt = util.single2tensor4(x_gt)


		#inv_weight_patch = torch.ones_like(x_gt)
		k_local = []
		for h_ in range(patch_num[1]):
			for w_ in range(patch_num[0]):
				k_all.append(util.single2tensor4(PSF_patch[w_,h_]))
		k = torch.cat(k_local,dim=0)

		[x,x_gt,k] = [el.to(device) for el in [x,x_gt,k]]
		ab_patch = F.softplus(ab[px_start:px_start+patch_num[0],py_start:py_start+patch_num[1]])
		ab_patch_v = []
		for h_ in range(patch_num[1]):
			for w_ in range(patch_num[0]):
				ab_patch_v.append(ab_patch[w_:w_+1,h_])
		ab_patch_v = torch.cat(cd,dim=0)

		x_E = model.forward_patchwise_SR(x,k,cd,patch_num,[patch_size[0],patch_size[1]],sf)

		predict = x_E[...,block_expand*sf:sf*block_expand+sf*patch_size[0]*patch_num[0],\
			block_expand*sf:sf*block_expand+sf*patch_size[1]*patch_num[1]]
		loss = F.l1_loss(predict,x_gt)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step()

		print('iter:{},loss {}'.format(global_iter+1,loss.item()))

		patch_L = cv2.resize(patch_L,dsize=None,fx=sf,fy=sf,interpolation=cv2.INTER_NEAREST)
		patch_L = patch_L[block_expand*sf:-block_expand*sf,block_expand*sf:-block_expand*sf]
		patch_E = util.tensor2uint(x_E)[block_expand*sf:-block_expand*sf,block_expand*sf:-block_expand*sf]
		show = np.hstack((patch_H,patch_L,patch_E))

		cv2.imshow('H,L,E',show)
		key = cv2.waitKey(1)
		global_iter+= 1

		# for logging model weight.
		# if global_iter % 100 ==0:
		# 	ab_numpy = ab.detach().cpu().numpy().flatten()
		# 	torch.save(model.state_dict(),'./logs/uabcnet_{}.pth'.format(global_iter))
		# 	np.savetxt('./logs/ab_{}.txt'.format(global_iter),ab_numpy)

		if key==ord('q'):
			running = False
			break
		if key==ord('s'):
			ab_numpy = ab.detach().cpu().numpy().flatten()
			torch.save(model.state_dict(),'./logs/uabcnet.pth')
			np.savetxt('./logs/ab.txt',ab_numpy)

	ab_numpy = ab.detach().cpu().numpy().flatten()
	torch.save(model.state_dict(),'./logs/uabcnet.pth')
	np.savetxt('./logs/ab.txt',ab_numpy)




if __name__ == '__main__':

	main()
