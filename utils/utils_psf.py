# Xiu Li
# modified from
# 08-May-2015, Behzad Tabibian
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def mv2pm(mv, K, mm2pixel):
  RY1 = np.array([[np.cos(mv[0]), 0, np.sin(mv[0])],
                  [0, 1, 0],
                  [-np.sin(mv[0]),0, np.cos(mv[0])]])
  RX2 = np.array([[1.0,0.0,0.0],
                  [0,np.cos(mv[1]), -np.sin(mv[1])],
                  [0, np.sin(mv[1]), np.cos(mv[1])]])
  RZ3 = np.array([[np.cos(mv[2]), -np.sin(mv[2]), 0],
                  [np.sin(mv[2]), np.cos(mv[2]), 0],
                  [0, 0, 1]])
  R = RX2.dot(RY1).dot(RZ3)
  t = mv[3:]*mm2pixel
  P = K.dot(np.vstack((R.T, t)).T)
  return P

def PSFprojection(walk,params):
  cols, rows = params['RowColPixels']
  chip_x, chip_y = params['SensorSize']
  f = params['FocalLength']
  D = params['Distance']
  crop = params['Crop']
  mm2pixel = rows/chip_y
  f_px = f*cols/chip_x
  Kext = np.array([[f_px,0,crop/2],[0,f_px,crop/2],[0,0,1.0]])
  x = np.arange(cols/2-350,cols/2+351,50)-cols/2
  y = np.arange(rows/2-350, rows/2+351,50)-rows/2
  X, Y = np.meshgrid(x,y)
  #50x50 kernel
  M = np.vstack((X.T.flatten()/f*D,Y.T.flatten()/f*D,
                 np.ones((1, X.size))*D*mm2pixel,np.ones((1, X.size))))
  ker = np.zeros((walk.shape[2],15,15,225,225))
  for iImg in range(walk.shape[2]):
    motion = iImg
    kernel = np.zeros((crop,crop))
    mv = np.zeros(6)
    for i in range(walk.shape[0]):
      walkCur = walk[i,:,motion]
      mv[0] =  walkCur[2] * np.pi / 180.0
      mv[1] =- walkCur[0] * np.pi / 180.0
      mv[2] =- walkCur[1] * np.pi / 180.0
      mv[3] =- walkCur[3]
      mv[4] =  walkCur[5]
      mv[5] =- walkCur[4]
      P = mv2pm(mv, Kext,mm2pixel)
      m = P.dot(M)
      x = m[0, :] / m[2, :]
      y = m[1, :] / m[2, :]
      L =  (np.round(x)<1) | (np.round(x)>crop) | (np.round(y)<1) | (np.round(y)>crop)
      I = np.logical_not(L)
      kernel[np.round(x[I]).astype(int)-1,np.round(y[I]).astype(int)-1] = kernel[np.round(x[I]).astype(int)-1,np.round(y[I]).astype(int)-1] + 1
    ker[iImg,:,:] = kernel/walk.shape[0]
  return ker


def camera_motion_to_psf():
    param = {'RowColPixels':(1872.0,2808.0),
            'SensorSize': (24.0,36.0),
            'FocalLength': 50.0,
            'Distance': 620.0,
            'Crop': 800
            }
    walk = scipy.io.loadmat("/home/xiu/databag/deblur/ICCV2021/motion_blur/500Frames/walkAll.mat")
    ker = PSFprojection(walk['walk'],param)
    return ker

def lens_param_to_psf():
    import numpy as __np__
    from numpy import sqrt as __sqrt__
    from numpy import cos as __cos__
    from numpy import sin as __sin__
    import matplotlib.pyplot as __plt__
    from matplotlib import cm as __cm__
    from matplotlib.ticker import LinearLocator as __LinearLocator__
    from matplotlib.ticker import FormatStrFormatter as __FormatStrFormatter__
    from mpl_toolkits.mplot3d import Axes3D
    from numpy.fft import fftshift as __fftshift__
    from numpy.fft import ifftshift as __ifftshift__
    from numpy.fft import fft2 as __fft2__
    def __apershow__(obj):
        obj = -abs(obj)
        __plt__.imshow(obj)
        __plt__.set_cmap('Greys')
        __plt__.show()

    l1 = 100
    #Generate test surface matrix from a detector
    x = __np__.linspace(-1, 1, l1)
    y = __np__.linspace(-1, 1, l1)
    [X,Y] = __np__.meshgrid(x,y)
    r = __sqrt__(X**2+Y**2)
    Z = __sqrt__(14)*(8*X**4-8*X**2*r**2+r**4)*(6*r**2-5)
    for i in range(len(Z)):
        for j in range(len(Z)):
            if x[i]**2+y[j]**2>1:
                Z[i][j]=0

    fig = __plt__.figure(1)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=__cm__.RdYlGn,
        linewidth=0, antialiased=False, alpha = 0.6)

    v = max(abs(Z.max()),abs(Z.min()))
    ax.set_zlim(-v*5, v*5)
    cset = ax.contourf(X, Y, Z, zdir='z', offset=-v*5, cmap=__cm__.RdYlGn)
    ax.zaxis.set_major_locator(__LinearLocator__(10))
    ax.zaxis.set_major_formatter(__FormatStrFormatter__('%.02f'))
    fig.colorbar(surf, shrink=1, aspect=30)
    __plt__.show()

    d = 800
    A = __np__.zeros([d,d])
    A[d//2-49:d//2+51,d//2-49:d//2+51] = Z
    __plt__.imshow(A)
    __plt__.show()

    abbe = __np__.exp(1j*2*__np__.pi*A)
    for i in range(len(abbe)):
        for j in range(len(abbe)):
            if abbe[i][j]==1:
                abbe[i][j]=0
    fig = __plt__.figure(2)
    AP = abs(__fftshift__(__fft2__(__fftshift__(abbe))))**2
    AP = AP/AP.max()

    __plt__.imshow(AP)
    __plt__.show()


def bbox(arr):
    rows = np.any(arr, axis=1)
    cols = np.any(arr, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def Schuler_PSF():
    #821x493 -> 820x492  (10x6)  -> 82x82
    psf_img = cv2.imread('//home/xiu/workspace/dwgan/MPI_data/bench/blind_psf.png')
    PSF = np.zeros((6,10,81,81,3),psf_img.dtype)

    for h_ in range(6):
        for w_ in range(10):
            PSF[h_,w_] = psf_img[1+82*h_:82*(h_+1),1+82*w_:82*(w_+1)]

    np.savez('../data/Schuler_PSF_bench.npz',PSF=PSF)

def Schuler_PSF_facade():
    #361x241 -> 360x240  (12x8)  -> 30x30
    grid_h = 8
    grid_w = 12
    k_sz = 29

    psf_img = cv2.imread('/home/xiu/databag/deblur/kernels/facade_psf.png')
    PSF = np.zeros((grid_h,grid_w,k_sz,k_sz,3),psf_img.dtype)

    for h_ in range(grid_h):
        for w_ in range(grid_w):
            PSF[h_,w_] = psf_img[1+(k_sz+1)*h_:(k_sz+1)*(h_+1),1+(k_sz+1)*w_:(k_sz+1)*(w_+1)]

    np.savez('../data/Schuler_PSF_facade.npz',PSF=PSF)

def Schuler_PSF_bridge():
    #361x241 -> 360x240  (12x8)  -> 30x30a
    grid_h = 8
    grid_w = 10
    k_sz = 19

    psf_img = cv2.imread('/home/xiu/databag/deblur/kernels/bridge_psf.png')
    PSF = np.zeros((grid_h,grid_w,k_sz,k_sz,3),psf_img.dtype)

    for h_ in range(grid_h):
        for w_ in range(grid_w):
            PSF[h_,w_] = psf_img[1+(k_sz+1)*h_:(k_sz+1)*(h_+1),1+(k_sz+1)*w_:(k_sz+1)*(w_+1)]

    np.savez('../data/Schuler_PSF_bridge.npz',PSF=PSF)

def PSF_ZEMAX_1024x768_aligned(file_path):
    img = cv2.imread(file_path)
    grid_h = 6
    grid_w = 8
    k_sz = 25
    PSF = np.zeros((grid_h,grid_w,k_sz,k_sz,3),img.dtype)
    for h_ in range(grid_w):
        for w_ in range(grid_h):
            local_patch = img[64+128*w_-12:64+128*w_+13,\
                64+128*h_-12:64+128*h_+13]
            PSF[w_,h_] = local_patch

    file_name = os.path.basename(file_path)
    file_name = os.path.splitext(file_name)[0]
    np.savez('../data/{}.npz'.format(file_name),PSF=PSF)
def PSF_toy():
     #361x241 -> 360x240  (12x8)  -> 30x30a
    grid_h = 8
    grid_w = 12
    k_sz = 51
    psf_img_r = cv2.imread('/home/xiu/workspace/dwgan/MPI_data/toy/kernels_channels_1_nm_regularized.jpg')
    psf_img_g = cv2.imread('/home/xiu/workspace/dwgan/MPI_data/toy/kernels_channels_3_nm_regularized.jpg')
    psf_img_b = cv2.imread('/home/xiu/workspace/dwgan/MPI_data/toy/kernels_channels_4_nm_regularized.jpg')

    PSF = np.zeros((grid_h,grid_w,k_sz,k_sz,3),psf_img_r.dtype)

    for h_ in range(grid_h):
        for w_ in range(grid_w):
            PSF[h_,w_,:,:,0] = psf_img_r[(k_sz+2)*h_+1:(k_sz+2)*(h_+1)-1,1+(k_sz+2)*w_:(k_sz+2)*(w_+1)-1,0]
            PSF[h_,w_,:,:,1] = psf_img_g[(k_sz+2)*h_+1:(k_sz+2)*(h_+1)-1,1+(k_sz+2)*w_:(k_sz+2)*(w_+1)-1,1]
            PSF[h_,w_,:,:,2] = psf_img_b[(k_sz+2)*h_+1:(k_sz+2)*(h_+1)-1,1+(k_sz+2)*w_:(k_sz+2)*(w_+1)-1,2]
    
    PSF_all = PSF.astype(np.float32)
    PSF_all = np.sum(PSF_all,axis=(-1)).reshape(-1,k_sz,k_sz)
    all_bbox = [bbox(arr) for arr in PSF_all]
    all_bbox = np.array(all_bbox)
    cv2.waitKey(-1)

    np.savez('../data/Heidel_PSF_toy.npz',PSF=PSF)

def PSF_plano_resize():
    grid_h = 7
    grid_w = 11
    k_sz = 49
    psf_img_r = cv2.imread('/home/xiu/workspace/dwgan/Supplement_SimpleLensImaging_Heide2013/supplemental_material/images/single_lens/plano_convex_lens/psfs/kernels_channels_1_nm_regularized.jpg')
    psf_img_g = cv2.imread('/home/xiu/workspace/dwgan/Supplement_SimpleLensImaging_Heide2013/supplemental_material/images/single_lens/plano_convex_lens/psfs/kernels_channels_3_nm_regularized.jpg')
    psf_img_b = cv2.imread('/home/xiu/workspace/dwgan/Supplement_SimpleLensImaging_Heide2013/supplemental_material/images/single_lens/plano_convex_lens/psfs/kernels_channels_4_nm_regularized.jpg')

    PSF = np.zeros((grid_h,grid_w,k_sz,k_sz,3),psf_img_r.dtype)

    k_sz_small = 25
    PSF_small = np.zeros((grid_h,grid_w,k_sz_small,k_sz_small,3),psf_img_r.dtype)

    for h_ in range(grid_h):
        for w_ in range(grid_w):
            PSF[h_,w_,:,:,0] = psf_img_r[(k_sz+2)*h_+1:(k_sz+2)*(h_+1)-1,1+(k_sz+2)*w_:(k_sz+2)*(w_+1)-1,0]
            PSF[h_,w_,:,:,1] = psf_img_g[(k_sz+2)*h_+1:(k_sz+2)*(h_+1)-1,1+(k_sz+2)*w_:(k_sz+2)*(w_+1)-1,1]
            PSF[h_,w_,:,:,2] = psf_img_b[(k_sz+2)*h_+1:(k_sz+2)*(h_+1)-1,1+(k_sz+2)*w_:(k_sz+2)*(w_+1)-1,2]

            PSF_small[h_,w_] = cv2.resize(PSF[h_,w_],dsize=(25,25))
    
    PSF_all = PSF.astype(np.float32)
    PSF_all = np.sum(PSF_all,axis=(-1)).reshape(-1,k_sz,k_sz)
    all_bbox = [bbox(arr) for arr in PSF_all]
    all_bbox = np.array(all_bbox)

    np.savez('../data/Heidel_PSF_plano.npz',PSF=PSF)
    np.savez('../data/Heidel_PSF_plano_small.npz',PSF=PSF_small)

def PSF_Motion(img_id=0):
    grid_h = 15
    grid_w = 15
    k_sz = 49

    img_f = '/home/xiu/databag/deblur/ICCV2021/motion_blur/KernelsAsMat/Non_Stationary_kernel_trajectory_{}.mat'.format(img_id)
    psf_img = sio.loadmat(img_f)['kernel']

    PSF = np.zeros((grid_h,grid_w,k_sz,k_sz,3),np.float32)

    for h_ in range(grid_h):
        for w_ in range(grid_w):
            PSF[h_,w_,:,:,0] = psf_img[50*(h_+1)-24:50*(h_+1)+25,50*(w_+1)-24:50*(w_+1)+25]
            PSF[h_,w_,:,:,1] = psf_img[50*(h_+1)-24:50*(h_+1)+25,50*(w_+1)-24:50*(w_+1)+25]
            PSF[h_,w_,:,:,2] = psf_img[50*(h_+1)-24:50*(h_+1)+25,50*(w_+1)-24:50*(w_+1)+25]
    
    np.savez('../data/Motion_PSF_{}.npz'.format(img_id),PSF=PSF)
