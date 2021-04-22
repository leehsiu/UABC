import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models.layers.usrnet_block as B
import utils.utils_deblur as util_deblur
import utils.utils_image as util


def upsample(x, sf=3):
    '''s-fold upsampler

    Upsampling the spatial size by filling the new entries with zeros

    x: tensor image, NxCxWxH
    '''
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]
                     * sf, x.shape[3]*sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf=3):
    '''s-fold downsampler

    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]


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
    # complex multiplication

    real1, imag1 = t1[..., 0], t1[..., 1]
    real2, imag2 = t2[..., 0], t2[..., 1]
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1)


def cconj(t, inplace=False):
    # complex's conjugation
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


def psf2otf(psf, shape):

    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[..., :psf.shape[2], :psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.rfft(otf, 2, onesided=False)
    n_ops = torch.sum(torch.tensor(psf.shape).type_as(
        psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    otf[..., 1][torch.abs(otf[..., 1]) < n_ops *
                2.22e-16] = torch.tensor(0).type_as(psf)
    return otf


"""
# (1) Prior module; ResUNet: act as a non-blind denoiser
# x_k = P(z_k, lambda_k)
"""


class ResUNet(nn.Module):
    def __init__(self, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(ResUNet, self).__init__()

        #self.m_head = B.conv(in_nc, nc[0],kernel_size=7,padding=3, bias=False, mode='C')
        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError(
                'downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C')
                                      for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C')
                                      for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C')
                                      for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body = B.sequential(
            *[B.ResBlock(nc[3], nc[3], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError(
                'upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[
                                  B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[
                                  B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[
                                  B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        #self.m_tail = B.conv(nc[0], out_nc,kernel_size=7,padding=3, bias=False, mode='C')
        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x):

        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/8)*8-h)
        paddingRight = int(np.ceil(w/8)*8-w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)

        x = x[..., :h, :w]

        return x


"""
# (2) Data module, closed-form solution
# It is a trainable-parameter-free module  ^_^
# z_k = D(x_{k-1}, s, k, y, alpha_k)
# some can be pre-calculated
"""


class RefDeconv(nn.Module):
    def __init__(self):
        super(RefDeconv, self).__init__()

    # reference-based deconvolution.
    def forward(self, z, FkCFy, F2k, alpha):
        # rfft is in-place operation,thus
        # clone is necessary.
        zt = z.clone()
        Fz = torch.rfft(zt, 2, onesided=False)
        Ff = Fz*alpha[..., None]
        FX = cdiv(FkCFy+Ff, csum(F2k, alpha))  # [...,None,None]))
        x = torch.irfft(FX, 2, onesided=False)
        return x

# RefSR


class RefSR(nn.Module):
    def __init__(self):
        super(RefSR, self).__init__()

    def forward(self, z, Fk, FkC, F2k, FkCFy, alpha, sf):
        zt = z.clone()
        FR = FkCFy + torch.rfft(alpha*zt, 2, onesided=False)
        x1 = cmul(Fk, FR)
        FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
        invW = torch.mean(splits(F2k, sf), dim=-1, keepdim=False)
        invWBR = cdiv(FBR, csum(invW, alpha))
        FCBinvWBR = cmul(FkC, invWBR.repeat(1, 1, sf, sf, 1))
        FX = (FR-FCBinvWBR)/alpha.unsqueeze(-1)
        x = torch.irfft(FX, 2, onesided=False)
        return x


# UABCNet:Universial ABerration Correction Network
class UABCNet(nn.Module):
    def __init__(self, n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, sf=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(UABCNet, self).__init__()
        if sf == 1:
            self.deconv = RefDeconv()
        else:
            self.deconv = RefSR()
        self.proj = ResUNet(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb, act_mode=act_mode,
                            downsample_mode=downsample_mode, upsample_mode=upsample_mode)
        self.proj.in_channels = in_nc
        #self.h = HyPaNet(in_nc=2, out_nc=n_iter*2, channel=h_nc)
        self.n = n_iter

    def assemble_patches(self, z, patch_num, patch_size):

        w, h = z.shape[-2:]
        w_pad = (w - patch_size[0])//2
        h_pad = (h - patch_size[1])//2

        x_00 = z[0:1][..., :w_pad, :h_pad]
        x_01 = torch.cat(torch.split(
            z[::patch_num[0]][..., 0:w_pad, h_pad:h_pad+patch_size[1]], 1), dim=3)
        x_02 = z[(patch_num[1]-1)*patch_num[0]:(patch_num[1]-1) *
                 patch_num[0]+1][..., :w_pad, h_pad+patch_size[1]:]
        x_0 = torch.cat([x_00, x_01, x_02], dim=3)

        x_10 = torch.cat(torch.split(
            z[:patch_num[0]][..., w_pad:w_pad+patch_size[0], :h_pad], 1), dim=2)
        x_11 = torch.cat(torch.chunk(
            z[..., w_pad:w_pad+patch_size[0], h_pad:h_pad+patch_size[1]], patch_num[1], dim=0), dim=3)
        x_11 = torch.cat(torch.chunk(x_11, patch_num[0], dim=0), dim=2)
        x_12 = torch.cat(torch.split(z[patch_num[0]*(patch_num[1]-1):patch_num[0]*patch_num[1]]
                                     [..., w_pad:w_pad+patch_size[0], h_pad+patch_size[1]:], 1), dim=2)
        x_1 = torch.cat([x_10, x_11, x_12], dim=3)

        x_20 = z[patch_num[0]-1:patch_num[0]
                 ][..., w_pad+patch_size[0]:, :h_pad]
        x_21 = torch.cat(torch.split(
            z[patch_num[0]-1::patch_num[0]][..., :w_pad, h_pad:h_pad+patch_size[1]], 1), dim=3)
        x_22 = z[patch_num[0]*patch_num[1]-1:patch_num[0]*patch_num[1]
                 ][..., w_pad+patch_size[0]:, h_pad+patch_size[1]:]
        x_2 = torch.cat([x_20, x_21, x_22], dim=3)
        x = torch.cat([x_0, x_1, x_2], dim=2)

        return x

    def chop_to_patches(self, x, patch_num, patch_size):
        W, H = x.shape[-2:]
        w_pad = (W-patch_num[0]*patch_size[0])//2
        h_pad = (H-patch_num[1]*patch_size[1])//2
        x_patches = []
        for h_ in range(patch_num[1]):
            for w_ in range(patch_num[0]):
                x_p = x[..., w_*patch_size[0]:(w_+1)*patch_size[0]+w_pad*2,
                        h_*patch_size[1]:(h_+1)*patch_size[1]+h_pad*2]
                x_patches.append(x_p)

        x_patches = torch.cat(x_patches, dim=0)
        return x_patches

    def forward_patchwise_SR(self, y, k, ab, patch_num=[2, 2], patch_size=[128, 128], sf=4):
        # only batch-size=1 is supported currently.
        W, H = y.shape[-2:]
        w_pad = (W - patch_num[0]*patch_size[0])//2
        h_pad = (H - patch_num[1]*patch_size[1])//2
        # 1. chop y to patch_num patch_num
        #y_patches = []
        Fk_all = []
        FkC_all = []
        F2k_all = []
        FkCFy_all = []
        y_init_all = []

        for h_ in range(patch_num[1]):
            for w_ in range(patch_num[0]):
                y_p = y[..., w_*patch_size[0]:(w_+1)*patch_size[0]+w_pad*2,
                        h_*patch_size[1]:(h_+1)*patch_size[1]+h_pad*2]
                # y_patches.append(y_p)
                Fk = psf2otf(k[w_+h_*patch_num[0]].unsqueeze(0), (sf *
                                                                  patch_size[0]+sf*w_pad*2, sf*patch_size[1]+sf*h_pad*2))
                FkC = cconj(Fk, inplace=False)
                F2k = r2c(cabs(FkC))
                y_upsample = upsample(y_p, sf)

                y_init_all.append(y_upsample)
                FkCFy = cmul(FkC, torch.rfft(y_upsample, 2, onesided=False))
                FkCFy_all.append(FkCFy)
                F2k_all.append(F2k)
                Fk_all.append(Fk)
                FkC_all.append(FkC)

        x = torch.cat(y_init_all, dim=0)
        x = self.assemble_patches(
            x, patch_num, [patch_size[0]*sf, patch_size[1]*sf])
        
        for i in range(self.n):
            # chop
            # (1)ref-deconv
            z = self.chop_to_patches(
                x, patch_num, [patch_size[0]*sf, patch_size[1]*sf])
            z_temp = []
            for idx in range(patch_num[0]*patch_num[1]):
                mu_p = ab[idx, 2*i].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                z_p = self.deconv(
                    z[idx:idx+1], Fk_all[idx], FkC_all[idx], F2k_all[idx], FkCFy_all[idx], mu_p, sf)
                z_temp.append(z_p)
            z_temp = torch.cat(z_temp, dim=0)

            # (2)proj
            z = self.assemble_patches(z_temp, patch_num, [patch_size[0]*sf,patch_size[1]*sf])
            beta = torch.zeros((1, 3, z.size(2), z.size(3)), device=z.device)
            for h_ in range(patch_num[1]):
                for w_ in range(patch_num[0]):
                    idx = w_ + h_*patch_num[0]
                    beta[0, 0, w_*patch_size[0]*sf:(w_+1)*patch_size[0]*sf+w_pad*2*sf,
                         h_*patch_size[1]*sf:(h_+1)*patch_size[1]*sf+h_pad*2*sf] = ab[idx, 1, 0]
                    beta[0, 1, w_*patch_size[0]*sf:(w_+1)*patch_size[0]*sf+w_pad*2*sf,
                         h_*patch_size[1]*sf:(h_+1)*patch_size[1]*sf+h_pad*2*sf] = ab[idx, 1, 1]
                    beta[0, 2, w_*patch_size[0]*sf:(w_+1)*patch_size[0]*sf+w_pad*2*sf,
                         h_*patch_size[1]*sf:(h_+1)*patch_size[1]*sf+h_pad*2*sf] = ab[idx, 1, 2]

            if self.proj.in_channels == 4:
                beta = torch.mean(beta, dim=1, keepdim=True)
            x = self.proj(torch.cat([z, beta[0:1]], dim=1))
        return x

    # ab,corresponds to the \mu,\lambda in the paper.
    def forward_patchwise(self, y, k, ab, patch_num=[2, 2], patch_size=[128, 128]):
        # only batch-size=1 is supported currently.
        W, H = y.shape[-2:]
        w_pad = (W - patch_num[0]*patch_size[0])//2
        h_pad = (H - patch_num[1]*patch_size[1])//2

        # 1. chop y to patch_num patch_num
        #y_patches = []
        FkCFy_all = []
        FkCFk_all = []
        Fy_all = []

        for h_ in range(patch_num[1]):
            for w_ in range(patch_num[0]):
                y_p = y[..., w_*patch_size[0]:(w_+1)*patch_size[0]+w_pad*2,
                        h_*patch_size[1]:(h_+1)*patch_size[1]+h_pad*2]
                # y_patches.append(y_p)
                Fk = psf2otf(k[w_+h_*patch_num[0]].unsqueeze(0),
                             (patch_size[0]+w_pad*2, patch_size[1]+h_pad*2))
                FkC = cconj(Fk, inplace=False)
                FkCFk = r2c(cabs(FkC))
                Fy = torch.rfft(y_p, 2, onesided=False)
                FkCFy = cmul(FkC, Fy)
                FkCFy_all.append(FkCFy)
                FkCFk_all.append(FkCFk)
                Fy_all.append(Fy)

        # 2-(1).Init deconv. Weiner filter.
        z = []
        for idx in range(patch_num[0]*patch_num[1]):
            mu_p = ab[idx, 0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            Fy = Fy_all[idx]
            Fy_real = Fy[..., 0] * mu_p
            Fy_image = Fy[..., 1] * mu_p
            Fy = torch.stack([Fy_real, Fy_image], dim=-1)
            FZ = cdiv(FkCFy_all[idx]+Fy, csum(FkCFk_all[idx], mu_p))
            z.append(torch.irfft(FZ, 2, onesided=False))
        z = torch.cat(z, dim=0)

        # 2-(2).Init proj.
        z = self.assemble_patches(z, patch_num, patch_size)
        beta = torch.zeros((1, 3, z.size(2), z.size(3)), device=z.device)
        for h_ in range(patch_num[1]):
            for w_ in range(patch_num[0]):
                idx = w_ + h_*patch_num[0]
                beta[0, 0, w_*patch_size[0]:(w_+1)*patch_size[0]+w_pad*2,
                     h_*patch_size[1]:(h_+1)*patch_size[1]+h_pad*2] = ab[idx, 1, 0]
                beta[0, 1, w_*patch_size[0]:(w_+1)*patch_size[0]+w_pad*2,
                     h_*patch_size[1]:(h_+1)*patch_size[1]+h_pad*2] = ab[idx, 1, 1]
                beta[0, 2, w_*patch_size[0]:(w_+1)*patch_size[0]+w_pad*2,
                     h_*patch_size[1]:(h_+1)*patch_size[1]+h_pad*2] = ab[idx, 1, 2]

                # set out part as large value

        if self.proj.in_channels == 4:
            beta = torch.mean(beta, dim=1, keepdim=True)
        x = self.proj(torch.cat([z, beta], dim=1))

        # output.append(x.clone())
        for i in range(self.n-1):
            # chop
            # (1)ref-proj
            z = self.chop_to_patches(x, patch_num, patch_size)
            z_temp = []
            for idx in range(patch_num[0]*patch_num[1]):
                mu_p = ab[idx, 2*i].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                z_p = self.deconv(
                    z[idx:idx+1], FkCFy_all[idx], FkCFk_all[idx], mu_p)
                z_temp.append(z_p)
            z_temp = torch.cat(z_temp, dim=0)

            # (2)proj
            z = self.assemble_patches(z_temp, patch_num, patch_size)
            beta = torch.zeros((1, 3, z.size(2), z.size(3)), device=z.device)
            for h_ in range(patch_num[1]):
                for w_ in range(patch_num[0]):
                    idx = w_ + h_*patch_num[0]
                    beta[0, 0, w_*patch_size[0]:(w_+1)*patch_size[0]+w_pad*2,
                         h_*patch_size[1]:(h_+1)*patch_size[1]+h_pad*2] = ab[idx, 1, 0]
                    beta[0, 1, w_*patch_size[0]:(w_+1)*patch_size[0]+w_pad*2,
                         h_*patch_size[1]:(h_+1)*patch_size[1]+h_pad*2] = ab[idx, 1, 1]
                    beta[0, 2, w_*patch_size[0]:(w_+1)*patch_size[0]+w_pad*2,
                         h_*patch_size[1]:(h_+1)*patch_size[1]+h_pad*2] = ab[idx, 1, 2]
                    # beta[0,0,w_*patch_size[0]:(w_+1)*patch_size[0]+w_pad*2,\
                    #	h_*patch_size[1]:(h_+1)*patch_size[1]+h_pad*2] = ab[idx,1,0]
            if self.proj.in_channels == 4:
                beta = torch.mean(beta, dim=1, keepdim=True)
            x = self.proj(torch.cat([z, beta[0:1]], dim=1))
        return x
