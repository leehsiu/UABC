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

np.random.seed(0)

def load_kernels(kernel_path):
	kernels = []
	kernel_files = glob.glob(os.path.join(kernel_path,'*.npz'))
	kernel_files.sort()
	for kf in kernel_files:
		PSF_grid = np.load(kf)['PSF']
		PSF_grid = util_psf.normalize_PSF(PSF_grid)
		kernels.append(PSF_grid)
	return kernels

def draw_random_kernel(kernels,patch_num):
	psf = kernels[0]
	psf = psf[:2,:2]
	#if i<0:
	#	psf = kernels[i]
	#else:
	#	psf = gaussian_kernel_map(patch_num)
	return psf

def gaussian_kernel_map(patch_num):
	PSF = np.zeros((patch_num[0],patch_num[1],25,25,3))
	for w_ in range(patch_num[0]):
		for h_ in range(patch_num[1]):
			PSF[w_,h_,...,0] = util_deblur.gen_kernel()
			PSF[w_,h_,...,1] = util_deblur.gen_kernel()
			PSF[w_,h_,...,2] = util_deblur.gen_kernel()
	return PSF


def draw_training_pair(image_H,psf,sf,patch_num,patch_size,image_L=None):
	w,h = image_H.shape[:2]
	gx,gy = psf.shape[:2]
	px_start = np.random.randint(0,gx-patch_num[0]+1)
	py_start = np.random.randint(0,gy-patch_num[1]+1)
	#wether or not to focus on edges.
	# mode = np.random.randint(5)
	# if mode==0:
	# 	px_start = 0
	# if mode==1:
	# 	px_start = gx-patch_num[0]
	# if mode==2:
	# 	py_start = 0
	# if mode==3:
	# 	py_start = gy-patch_num[1]

	psf_patch = psf[px_start:px_start+patch_num[0],py_start:py_start+patch_num[1]]
	patch_size_H = [patch_size[0]*sf,patch_size[1]*sf]

	if image_L is None:
		#generate image_L on-the-fly
		conv_expand = psf.shape[2]//2
		x_start = np.random.randint(0,w-patch_size_H[0]*patch_num[0]-conv_expand*2+1)
		y_start = np.random.randint(0,h-patch_size_H[1]*patch_num[1]-conv_expand*2+1)
		patch_H = image_H[x_start:x_start+patch_size_H[0]*patch_num[0]+conv_expand*2,\
		y_start:y_start+patch_size_H[1]*patch_num[1]+conv_expand*2]
		patch_L = util_deblur.blockConv2d(patch_H,psf_patch,conv_expand)

		patch_H = patch_H[conv_expand:-conv_expand,conv_expand:-conv_expand]
		patch_L = patch_L[::sf,::sf]

		#wrap_edges around patch_L to avoid FFT boundary effect.
		#wrap_expand = patch_size[0]//8
		# patch_L_wrap = util_deblur.wrap_boundary_liu(patch_L,(patch_size[0]*patch_num[0]+wrap_expand*2,\
		# patch_size[1]*patch_num[1]+wrap_expand*2))
		# patch_L_wrap = np.hstack((patch_L_wrap[:,-wrap_expand:,:],patch_L_wrap[:,:patch_size[1]*patch_num[1]+wrap_expand,:]))
		# patch_L_wrap = np.vstack((patch_L_wrap[-wrap_expand:,:,:],patch_L_wrap[:patch_size[0]*patch_num[0]+wrap_expand,:,:]))
		# patch_L = patch_L_wrap

	else:
		x_start = px_start * patch_size_H[0]
		y_start = py_start * patch_size_H[1]
		patch_H = image_H[x_start:x_start+patch_size_H[0]*patch_num[0],\
			y_start:y_start+patch_size_H[1]*patch_num[1]]
		x_start = px_start * patch_size[0]
		y_start = py_start * patch_size[1]
		patch_L = image_L[x_start:x_start+patch_size[0]*patch_num[0],\
			y_start:y_start+patch_size[1]*patch_num[1]]

	return patch_L,patch_H,psf_patch

def main():
	#0. global config
	#scale factor
	sf = 4	
	stage = 8
	patch_size = [32,32]
	patch_num = [2,2]

	#1. local PSF
	#shape: gx,gy,kw,kw,3
	all_PSFs = load_kernels('./data')


	#2. local model
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = net(n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512],
					nb=2,sf=sf, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
	model.load_state_dict(torch.load('./data/uabcnet_final.pth'),strict=True)
	model.train()
	for _, v in model.named_parameters():
		v.requires_grad = True
	model = model.to(device)

	#positional lambda, mu for HQS, set as free trainable parameters here.

	ab_buffer = np.loadtxt('./data/ab.txt').reshape((patch_num[0],patch_num[1],2*stage,3)).astype(np.float32)
	#ab_buffer = np.ones((patch_num[0],patch_num[1],2*stage,3),dtype=np.float32)*0.1
	ab = torch.tensor(ab_buffer,device=device,requires_grad=True)
	params = []
	params += [{"params":[ab],"lr":0.0005}]
	for key,value in model.named_parameters():
		params += [{"params":[value],"lr":1e-6}]

	optimizer = torch.optim.Adam(params,lr=0.0001,betas=(0.9,0.999))
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1000,gamma=0.9)

	#3.load training data
	imgs_H = glob.glob('/home/xiu/databag/deblur/images/DIV2K_train/*.png',recursive=True)
	imgs_H.sort()

	global_iter = 0

	all_PSNR = []
	N_maxiter = 2000

	PSF_grid = draw_random_kernel(all_PSFs,patch_num)
	#def get_train_pairs()

	for i in range(N_maxiter):

		t0 = time.time()
		#draw random image.
		img_idx = np.random.randint(len(imgs_H))

		img_H = cv2.imread(imgs_H[img_idx])

		#img2 = imgs_L[img_idx]
		#img_L = cv2.imread(img2)
		#draw random patch from image
		#a. without img_L

		#draw random kernel


		patch_L,patch_H,patch_psf = draw_training_pair(img_H,PSF_grid,sf,patch_num,patch_size)
		#b.	with img_L
		#patch_L, patch_H, patch_psf,px_start, py_start,block_expand = draw_training_pair(img_H, PSF_grid, sf, patch_num, patch_size, img_L)
		t_data = time.time()-t0

		x = util.uint2single(patch_L)
		x = util.single2tensor4(x)
		x_gt = util.uint2single(patch_H)
		x_gt = util.single2tensor4(x_gt)

		k_local = []
		for h_ in range(patch_num[1]):
			for w_ in range(patch_num[0]):
				k_local.append(util.single2tensor4(patch_psf[w_,h_]))
		k = torch.cat(k_local,dim=0)
		[x,x_gt,k] = [el.to(device) for el in [x,x_gt,k]]
		
		ab_patch = F.softplus(ab)
		ab_patch_v = []
		for h_ in range(patch_num[1]):
			for w_ in range(patch_num[0]):
				ab_patch_v.append(ab_patch[w_:w_+1,h_])
		ab_patch_v = torch.cat(ab_patch_v,dim=0)

		x_E = model.forward_patchwise_SR(x,k,ab_patch_v,patch_num,[patch_size[0],patch_size[1]],sf)

		loss = F.l1_loss(x_E,x_gt)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step()

		t_iter = time.time() - t0 - t_data

		print('[iter:{}] loss:{:.4f}, data_time:{:.2f}s, net_time:{:.2f}s'.format(global_iter+1,loss.item(),t_data,t_iter))

		patch_L = cv2.resize(patch_L,dsize=None,fx=sf,fy=sf,interpolation=cv2.INTER_NEAREST)
		#patch_L = patch_L[block_expand*sf:-block_expand*sf,block_expand*sf:-block_expand*sf]
		patch_E = util.tensor2uint((x_E))
		show = np.hstack((patch_H,patch_L,patch_E))
		cv2.imshow('H,L,E',show)
		key = cv2.waitKey(1)
		global_iter+= 1

		if key==ord('q'):
			break
		if key==ord('s'):
			ab_numpy = ab.detach().cpu().numpy().flatten()
			np.savetxt('./data/ab.txt',ab_numpy)


	ab_numpy = ab.detach().cpu().numpy().flatten()
	torch.save(model.state_dict(),'./data/uabcnet_finetune.pth')
	np.savetxt('./data/ab_finetune.txt',ab_numpy)
if __name__ == '__main__':

	main()
