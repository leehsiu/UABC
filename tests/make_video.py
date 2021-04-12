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

def make_size_divisible(img,stride):
	w,h,_ = img.shape

	w_new = w//stride*stride
	h_new = h//stride*stride

	return img[:w_new,:h_new,:]

def main():
	# ----------------------------------------
	# load kernels
	# ----------------------------------------

	img_L = glob.glob('/home/xiu/databag/deblur/ICCV2021/video_ZEMAX/blurry/*.bmp')
	img_H = glob.glob('/home/xiu/databag/deblur/ICCV2021/video_ZEMAX/result/*.bmp')

	img_L.sort()
	img_H.sort()

	mask = np.tri(1024,1024,-2)
	mask_top = mask.T
	
	mask = cv2.resize(mask,dsize=(1024,768),interpolation=cv2.INTER_NEAREST)
	mask_top = cv2.resize(mask_top,dsize=(1024,768),interpolation=cv2.INTER_NEAREST)

	mask = np.dstack((mask,mask,mask))
	mask_top = np.dstack((mask_top,mask_top,mask_top))


	idx  = 0 
	for l,h in zip(img_L,img_H):
		L = cv2.imread(l)
		H = cv2.imread(h)

		L = L.astype(np.float32)
		H = H.astype(np.float32)

		fuse = L*mask_top + H*mask
		fuse = fuse.astype(np.uint8)
		print(idx)
		idx += 1
		cv2.imshow("fuse",fuse)
		cv2.waitKey(-1)
		#out.write(fuse)























	


if __name__ == '__main__':

	main()
