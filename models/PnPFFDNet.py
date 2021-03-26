import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nuocnet.models.layers.usrnet_block as B
import nuocnet.models.basicblock as basicblock
import nuocnet.utils.utils_deblur as util_deblur
import nuocnet.utils.utils_image as util

class FFDNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R'):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        # ------------------------------------
        """
        super(FFDNet, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True
        sf = 2
        self.m_down = basicblock.PixelUnShuffle(upscale_factor=sf)
        m_head = basicblock.conv(in_nc*sf*sf+1, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [basicblock.conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = basicblock.conv(nc, out_nc*sf*sf, mode='C', bias=bias)

        self.model = basicblock.sequential(m_head, *m_body, m_tail)

        self.m_up = nn.PixelShuffle(upscale_factor=sf)

    def forward(self, x, sigma):

        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/2)*2-h)
        paddingRight = int(np.ceil(w/2)*2-w)
        x = torch.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x = self.m_down(x)
        # m = torch.ones(sigma.size()[0], sigma.size()[1], x.size()[-2], x.size()[-1]).type_as(x).mul(sigma)
        m = sigma.repeat(1, 1, x.size()[-2], x.size()[-1])
        x = torch.cat((x, m), 1)
        x = self.model(x)
        x = self.m_up(x)
        
        x = x[..., :h, :w]
        return x




"""
Deep unfolding network for image super-resolution
https://arxiv.org/abs/2003.10428v1
"""

def splits(a, sf):
	'''split a into sfxsf distinct blocks

	Args:
		a: NxCxWxHx2
		sf: split factor

	Returns:
		b: NxCx(W/sf)x(H/sf)x2x(sf^2)
	'''
	b = torch.stack(torch.chunk(a, sf, dim=2), dim=5)
	b = torch.cat(torch.chunk(b, sf, dim=3), dim=5)
	return b


def c2c(x):
	return torch.from_numpy(np.stack([np.float32(x.real), np.float32(x.imag)], axis=-1))


def r2c(x):
	# convert real to complex
	return torch.stack([x, torch.zeros_like(x)], -1)


def cdiv(x, y):
	# complex division
	a, b = x[..., 0], x[..., 1]
	c, d = y[..., 0], y[..., 1]
	cd2 = c**2 + d**2
	return torch.stack([(a*c+b*d)/cd2, (b*c-a*d)/cd2], -1)


def crdiv(x, y):
	# complex/real division
	a, b = x[..., 0], x[..., 1]
	return torch.stack([a/y, b/y], -1)


def csum(x, y):
	# complex + real
	return torch.stack([x[..., 0] + y, x[..., 1]], -1)


def cabs(x):
	# modulus of a complex number
	return torch.pow(x[..., 0]**2+x[..., 1]**2, 0.5)


def cabs2(x):
	return x[..., 0]**2+x[..., 1]**2


def cmul(t1, t2):
	'''complex multiplication

	Args:
		t1: NxCxHxWx2, complex tensor
		t2: NxCxHxWx2

	Returns:
		output: NxCxHxWx2
	'''
	real1, imag1 = t1[..., 0], t1[..., 1]
	real2, imag2 = t2[..., 0], t2[..., 1]
	return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1)


def cconj(t, inplace=False):
	'''complex's conjugation

	Args:
		t: NxCxHxWx2

	Returns:
		output: NxCxHxWx2
	'''
	c = t.clone() if not inplace else t
	c[..., 1] *= -1
	return c


def rfft(t):
	# Real-to-complex Discrete Fourier Transform
	return torch.rfft(t, 2, onesided=False)


def irfft(t):
	# Complex-to-real Inverse Discrete Fourier Transform
	return torch.irfft(t, 2, onesided=False)


def fft(t):
	# Complex-to-complex Discrete Fourier Transform
	return torch.fft(t, 2)


def ifft(t):
	# Complex-to-complex Inverse Discrete Fourier Transform
	return torch.ifft(t, 2)


def p2o(psf, shape):
	'''
	Convert point-spread function to optical transfer function.
	otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
	point-spread function (PSF) array and creates the optical transfer
	function (OTF) array that is not influenced by the PSF off-centering.

	Args:
		psf: NxCxhxw
		shape: [H, W]

	Returns:
		otf: NxCxHxWx2
	'''
	otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
	otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
	for axis, axis_size in enumerate(psf.shape[2:]):
		otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
	otf = torch.rfft(otf, 2, onesided=False)
	n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
	otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
	return otf


def upsample(x, sf=3):
	'''s-fold upsampler

	Upsampling the spatial size by filling the new entries with zeros

	x: tensor image, NxCxWxH
	'''
	st = 0
	z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*sf, x.shape[3]*sf)).type_as(x)
	z[..., st::sf, st::sf].copy_(x)
	return z


def downsample(x, sf=3):
	'''s-fold downsampler

	Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

	x: tensor image, NxCxWxH
	'''
	st = 0
	return x[..., st::sf, st::sf]


def downsample_np(x, sf=3):
	st = 0
	return x[st::sf, st::sf, ...]


class RefDeconv(nn.Module):
	def __init__(self):
		super(RefDeconv,self).__init__()

	#reference-based deconvolution.
	def forward(self,z,FkCFy,F2k,alpha):
		#rfft is in-place operation,thus
		#clone is necessary.
		zt = z.clone()
		Fz = torch.rfft(zt,2,onesided=False)
		Ff = Fz*alpha[...,None,None,None]
		FX = cdiv(FkCFy+Ff,csum(F2k,alpha[...,None,None]))
		x = torch.irfft(FX,2,onesided=False)
		return x


class UABCNetPnP(nn.Module):
	def __init__(self):
		super(UABCNetPnP, self).__init__()

		self.deconv = RefDeconv()
		self.p = FFDNet(in_nc=3,out_nc=3,nc=96,nb=12)

	def forward_patchdeconv(self,y,k,ab,patch_num=[2,2],patch_sz=64):
		#input:
		W,H = y.shape[-2:]
		F2Ks = []
		FkCFys = []
		extend = (W - patch_sz)//2
		n_iter = (ab.shape[1] - 1)//2
		#for h_ in range(patch_num[0]):
		#	for w_ in range(patch_num[1]):
			#idx = w_*patch_num[1] + h_
		for idx in range(patch_num[0]*patch_num[1]):
			Fk = p2o(k[idx].unsqueeze(0),(W,H))
			FkC = cconj(Fk,inplace=False)
			F2k = r2c(cabs2(Fk))
			#y_local = y[idx:idx+1].clone()
			FkCFy = cmul(FkC,torch.rfft(y[idx:idx+1],2,onesided=False))
			FkCFys.append(FkCFy)
			F2Ks.append(F2k)

		x = []
		#1. weiner filter but with estimated per-patch alpha
		for idx in range(patch_num[0]*patch_num[1]):
		#    x_patches.append(self.d.weiner_deconv(FkCFys[idx],F2Ks[idx],ab[idx,0]))
			#y[idx] = self.d.weiner_deconv(FkCFys[idx],F2Ks[idx],ab[idx,0])
			FX = cdiv(FkCFys[idx],csum(F2Ks[idx],ab[idx,0].unsqueeze(0)[...,None,None]))
			x.append(torch.irfft(FX,2,onesided=False))
		x = torch.cat(x,dim=0)
		if extend>0:
			x = torch.cat(torch.chunk(x[:,:,extend:extend+patch_sz,extend:extend+patch_sz],patch_num[0],dim=0),dim=3)
			x = torch.cat(torch.chunk(x,patch_num[0],dim=0),dim=2)
		else:
			x = torch.cat(torch.chunk(x,patch_num[0],dim=0),dim=3)
			x = torch.cat(torch.chunk(x,patch_num[0],dim=0),dim=2)

		output = []
		output.append(x.clone())

		for i in range(n_iter):

			x = self.p(x,torch.tensor(0.02).cuda())

			output.append(x.clone())

			y = torch.cat(torch.chunk(x,patch_num[0],dim=2),dim=0)
			y = torch.cat(torch.chunk(y,patch_num[0],dim=3),dim=0)

			#y = nn.ReplicationPad2d(extend)(y)
			y = nn.ReflectionPad2d(extend)(y)


			for idx in range(patch_num[0]*patch_num[1]):
				y[idx] = self.deconv(y[idx:idx+1],FkCFys[idx],F2Ks[idx],ab[idx,i+1:i+2])

			if extend>0:
				z = torch.cat(torch.chunk(y[:,:,extend:extend+patch_sz,extend:extend+patch_sz],patch_num[0],dim=0),dim=3)
				z = torch.cat(torch.chunk(z,patch_num[0],dim=0),dim=2)
			else:
				z = torch.cat(torch.chunk(y,patch_num[0],dim=0),dim=3)
				z = torch.cat(torch.chunk(z,patch_num[0],dim=0),dim=2)

			#x = (1-merge_mask)*x + merge_mask*z
			x = z

			output.append(x.clone())

			
		return output






	
	#orignal forward model as in paper:USRNet
	def forward(self, x, k, sigma):
		'''
		x: tensor, NxCxWxH
		k: tensor, Nx(1,3)xwxh
		sf: integer, 1
		sigma: tensor, Nx1x1x1
		'''
		# initialization & pre-calculation
		w, h = x.shape[-2:]
		FB = p2o(k, (w,h))
		FBC = cconj(FB, inplace=False)
		F2B = r2c(cabs2(FB))
		FBFy = cmul(FBC, torch.rfft(x, 2, onesided=False))

		# hyper-parameter, alpha & beta
		ab = self.h(torch.cat((sigma,torch.tensor(1.0).type_as(sigma).expand_as(sigma)), dim=1))
		output = []

		for i in range(self.n):
			x = self.d(x,FBC,FBFy,F2B,ab[:, i:i+1, ...])
			output.append(x.clone().detach())
			x = self.p(torch.cat((x, ab[:, i+self.n:i+self.n+1, ...].repeat(1, 1, x.size(2), x.size(3))), dim=1))
			output.append(x.clone().detach())
		return x,output
