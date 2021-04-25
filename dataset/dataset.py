import glob
import os.path
import numpy as np
import cv2
import torch.utils.data
import torch.nn.functional as F

#not used currently
class DeblurData(torch.utils.data.Dataset):
	def __init__(self,root_path):
		super(DeblurData,self).__init__()
		#json?
		#file_meta as 
		#file_path, height, width
		self.root_path = root_path
		self.patch_size = patch_size
		#images
		#load kernels
		img_files = glob.glob(os.path.join(root_path,'images/**/*.png'))
		img_files.sort()
		self.images = img_files
		self.n_image = len(self.images)

		#kernels
		kernel_files = glob.glob(os.path.join(root_path,'kernels/*.npz'))
		kernel_files.sort()
		self.kernels = []
		for k_file in kernel_files:
			k = np.load(k_file)['PSF']
			k = k.astype(np.float32)
			#normalize
			for w_ in range(k.shape[0]):
				for h_ in range(k.shape[1]):
					k[w_,h_] /= np.sum(k[w_,h_],axis=(0,1))
			self.kernels.append(k)

	def __len__(self):
		return self.n_image

	def __getitem__(self,index):
		#load image
		img_path = self.images[index]
		img = cv2.imread(img_path)
		img = img.astype(np.float32)/255.0
		img = torch.as_tensor(img.transpose(2,0,1))
		#return.
		return img
