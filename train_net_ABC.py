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
from models.uabcnet import UABCNet as net



def main():
	# ----------------------------------------
	# load kernels
	# ----------------------------------------
	PSF_grid = np.load('./data/AC254-075-A-ML-Zemax(ZMX).npz')['PSF']
	PSF_grid = PSF_grid.astype(np.float32)
	gx,gy = PSF_grid.shape[:2]

	k_tensor = []
	for yy in range(gy):
		for xx in range(gx):
			PSF_grid[xx,yy] = PSF_grid[xx,yy]/np.sum(PSF_grid[xx,yy],axis=(0,1))
			k_tensor.append(util.single2tensor4(PSF_grid[xx,yy]))

	k_tensor = torch.cat(k_tensor,dim=0)
	inv_weight = util_deblur.get_inv_spatial_weight(k_tensor)

	# ----------------------------------------
	# load model
	# ----------------------------------------
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = net(n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512],
					nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
	model.proj.load_state_dict(torch.load('./data/usrnet_pretrain.pth'),strict=True)
	model.train()
	for _, v in model.named_parameters():
		v.requires_grad = True
	model = model.to(device)

	# ----------------------------------------
	# load training data
	# ----------------------------------------
	imgs = glob.glob('/home/xiu/databag/deblur/images/*/**.png',recursive=True)
	imgs.sort()


	# ----------------------------------------
	# positional lambda\mu for HQS
	# ----------------------------------------
	stage = 8
	ab_buffer = np.ones((gx,gy,2*stage,3),dtype=np.float32)*0.1
	#ab_buffer[:,:,0,:] = 0.01
	ab = torch.tensor(ab_buffer,device=device,requires_grad=True)

	# ----------------------------------------
	# build optimizer
	# ----------------------------------------
	params = []
	params += [{"params":[ab],"lr":0.0005}]
	for key,value in model.named_parameters():
		params += [{"params":[value],"lr":0.0001}]
	optimizer = torch.optim.Adam(params,lr=0.0001,betas=(0.9,0.999))
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1000,gamma=0.9)

	patch_size = [128,128]
	expand = PSF_grid.shape[2]//2
	patch_num = [2,2]

	global_iter = 0

	running = True

	while running:
		#alpha.beta
		img_idx = np.random.randint(len(imgs))
		img = imgs[img_idx]
		img_H = cv2.imread(img)
		w,h = img_H.shape[:2]

		#focus on the edges

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

		x_start = np.random.randint(0,w-patch_size[0]*patch_num[0]-expand*2+1)
		y_start = np.random.randint(0,h-patch_size[1]*patch_num[1]-expand*2+1)
		PSF_patch = PSF_grid[px_start:px_start+patch_num[0],py_start:py_start+patch_num[1]]

		patch_H = img_H[x_start:x_start+patch_size[0]*patch_num[0]+expand*2,\
			y_start:y_start+patch_size[1]*patch_num[1]+expand*2]
		patch_L = util_deblur.blockConv2d(patch_H,PSF_patch,expand)

		block_expand = max(patch_size[0]//8,expand)

		patch_L_wrap = util_deblur.wrap_boundary_liu(patch_L,(patch_size[0]*patch_num[0]+block_expand*2,patch_size[1]*patch_num[1]+block_expand*2))
		patch_L_wrap = np.hstack((patch_L_wrap[:,-block_expand:,:],patch_L_wrap[:,:patch_size[1]*patch_num[1]+block_expand,:]))
		patch_L_wrap = np.vstack((patch_L_wrap[-block_expand:,:,:],patch_L_wrap[:patch_size[0]*patch_num[0]+block_expand,:,:]))
		x = util.uint2single(patch_L_wrap)
		x = util.single2tensor4(x)


		x_gt = util.uint2single(patch_H[expand:-expand,expand:-expand])
		x_gt = util.single2tensor4(x_gt)
		inv_weight_patch = torch.ones_like(x_gt)

		k_local = []

		for h_ in range(patch_num[1]):
			for w_ in range(patch_num[0]):
				inv_weight_patch[0,0,w_*patch_size[0]:(w_+1)*patch_size[0],h_*patch_size[1]:(h_+1)*patch_size[1]] = inv_weight[w_+h_*patch_num[0],0]
				inv_weight_patch[0,1,w_*patch_size[0]:(w_+1)*patch_size[0],h_*patch_size[1]:(h_+1)*patch_size[1]] = inv_weight[w_+h_*patch_num[0],1]
				inv_weight_patch[0,2,w_*patch_size[0]:(w_+1)*patch_size[0],h_*patch_size[1]:(h_+1)*patch_size[1]] = inv_weight[w_+h_*patch_num[0],2]
				k_local.append(k_tensor[w_+h_*patch_num[0]:w_+h_*patch_num[0]+1])

		k = torch.cat(k_local,dim=0)
		[x,x_gt,k,inv_weight_patch] = [el.to(device) for el in [x,x_gt,k,inv_weight_patch]]
		ab_patch = F.softplus(ab[px_start:px_start+patch_num[0],py_start:py_start+patch_num[1]])
		cd = []
		for h_ in range(patch_num[1]):
			for w_ in range(patch_num[0]):
				cd.append(ab_patch[w_:w_+1,h_])
		cd = torch.cat(cd,dim=0)
		x_E = model.forward_patchwise(x,k,cd,patch_num,patch_size)

		predict = x_E[...,block_expand:block_expand+patch_size[0]*patch_num[0],\
			block_expand:block_expand+patch_size[1]*patch_num[1]]
		loss = F.l1_loss(predict.div(inv_weight_patch),x_gt.div(inv_weight_patch))
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step()
		print('iter:{},loss {}'.format(global_iter+1,loss.item()))

		patch_L = patch_L_wrap.astype(np.uint8)
		patch_E = util.tensor2uint(x_E)[block_expand:-block_expand,block_expand:-block_expand]

		show = np.hstack((patch_H[expand:-expand,expand:-expand],patch_L[block_expand:-block_expand,block_expand:-block_expand],patch_E))

		cv2.imshow('HL',show)
		key = cv2.waitKey(1)

		global_iter+= 1

		#change the save period
		if global_iter % 100 ==0:
			ab_numpy = ab.detach().cpu().numpy().flatten()
			torch.save(model.state_dict(),'./ZEMAX_model/usrnet_ZEMAX_iter{}.pth'.format(global_iter))
			np.savetxt('./ZEMAX_model/ab_ZEMAX_iter{}.txt'.format(global_iter),ab_numpy)
		if key==ord('q'):
			running = False
			break
	ab_numpy = ab.detach().cpu().numpy().flatten()
	torch.save(model.state_dict(),'./ZEMAX_model/usrnet_ZEMAX.pth')
	np.savetxt('./ZEMAX_model/ab_ZEMAX.txt',ab_numpy)



if __name__ == '__main__':

	main()
