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

np.random.seed(15)

def load_kernels(kernel_path):
	kernels = []
	kernel_files = glob.glob(os.path.join(kernel_path,'*.npz'))
	kernel_files.sort()
	for kf in kernel_files:
		PSF_grid = np.load(kf)['PSF']
		PSF_grid = util_psf.normalize_PSF(PSF_grid)
		kernels.append(PSF_grid)
	return kernels


def using_AC254_lens(kernels,patch_num):
	psf = kernels[0]
	psf = psf[:patch_num[0],:patch_num[1]]
	return psf



def draw_random_kernel(kernels,patch_num):
	n = len(kernels)
	i = np.random.randint(2*n)
	if i<0:
		psf = kernels[i]
	else:
		psf = gaussian_kernel_map(patch_num)
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
	#loaded_state_dict=  torch.load('./data/uabcnet_final.pth')
	loaded_state_dict=  torch.load('./data/uabcnet_finetune.pth')
	model.load_state_dict(loaded_state_dict,strict=True)
	model.eval()
	for _, v in model.named_parameters():
		v.requires_grad = False
	model = model.to(device)

	#positional lambda, mu for HQS, set as free trainable parameters here.
	ab_buffer = np.loadtxt('./data/ab.txt').reshape((patch_num[0],patch_num[1],2*stage,3)).astype(np.float32)
	#ab[2x2,2*stage,3]
	#ab_buffer = np.ones((patch_num[0],patch_num[1],2*stage,3),dtype=np.float32)*0.1
	ab = torch.tensor(ab_buffer,device=device,requires_grad=False)
	#ab = F.softplus(ab)



	
	#3.load training data
	imgs_H = glob.glob('/home/xiu/databag/deblur/images/DIV2K_train/*.png',recursive=True)
	imgs_H.sort()

	global_iter = 0
	N_maxiter = 1000


	PSF_grid = using_AC254_lens(all_PSFs,patch_num)

	all_PSNR = []


	for i in range(N_maxiter):

		#draw random image.
		img_idx = np.random.randint(len(imgs_H))

		img_H = cv2.imread(imgs_H[img_idx])


		patch_L,patch_H,patch_psf = draw_training_pair(img_H,PSF_grid,sf,patch_num,patch_size)

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


		patch_L = cv2.resize(patch_L,dsize=None,fx=sf,fy=sf,interpolation=cv2.INTER_NEAREST)
		patch_E = util.tensor2uint((x_E))
		
		psnr = cv2.PSNR(patch_E,patch_H)
		all_PSNR.append(psnr)

		show = np.hstack((patch_H,patch_L,patch_E))
		#cv2.imwrite(os.path.join('./result',out_folder,'result-{:04d}.png'.format(i+1)),show)
	#np.savetxt(os.path.join('./result',out_folder,'psnr.txt'),all_PSNR)

if __name__ == '__main__':

	main()
