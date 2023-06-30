import numpy as np
import torch
import matplotlib.pyplot as plt

import scipy.io as scio
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import os
import ctlib
from model import generate_mask

    
# load original image
image = scio.loadmat('data/image/data_0001.mat')['data']

# load sparse view sinogram with 128 views
sinogram = scio.loadmat('data/sinogram/128views/data_0001.mat')['data']
views = sinogram.shape[0]
n_bins = sinogram.shape[1]
sinogram = torch.FloatTensor(sinogram).contiguous().cuda()

# options: views, dets, width, height, dImg, dDet, Ang0, dAng, s2r, d2r, binshift, scan_type
views = 128
dets = n_bins
width = 256
height = 256
dImg = 0.006641
dDet = 0.0072
Ang0 = 0
dAng = 0.006134 * (1024//views)
s2r = 2.5
d2r = 2.5
binshift = 0
scan_type = 0
options = torch.FloatTensor([views, dets, width, height, dImg, dDet, Ang0, dAng, s2r, d2r, binshift, scan_type])
options = options.cuda()

# reconstruct image by using fbp as an example using CTLIB
img_recon = ctlib.fbp(sinogram.view(1,1,views,n_bins) / 3.84, options)
mask = generate_mask(dImg, dDet)
img_recon = img_recon.detach().cpu().numpy().squeeze().clip(0,1) * mask

fig, ax = plt.subplots(1,2)
ax[0].imshow(image, cmap='gray')
ax[0].axis('off')
ax[0].set_title('ground truth')

ax[1].imshow(img_recon, cmap='gray')
ax[1].axis('off')
ax[1].set_title('fbp 128 views')
plt.show()

print("Result for fbp using 128 views: ")
print('PSNR (128 views): {:.4f}'.format(psnr(img_recon.astype(image.dtype), image)))
print('SSIM (128 views): {:.4f}'.format(ssim(img_recon.astype(image.dtype), image)))