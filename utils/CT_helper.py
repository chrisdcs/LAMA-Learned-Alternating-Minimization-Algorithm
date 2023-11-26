import os
import torch
import ctlib
import yaml
import numpy as np
import scipy.io as scio
from pathlib import Path

from utils.general import LOGGER

ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # relative path to current working directory

def load_CT_config(config_file):
    if isinstance(config_file, str) or isinstance(config_file, Path):
        with open(config_file, errors='ignore') as f:
            hyp = yaml.safe_load(f)
    views, dets, width, height, dImg, dDet, Ang0, dAng, s2r, d2r, binshift, scan_type = \
    hyp['views'], hyp['dets'], hyp['width'], hyp['height'], hyp['dImg'], hyp['dDet'], hyp['Ang0'], hyp['dAng'], hyp['s2r'], hyp['d2r'], hyp['binshift'], hyp['scan_type']  
    options = torch.FloatTensor([views, dets, width, height, dImg, dDet, Ang0, dAng, s2r, d2r, binshift, scan_type])
    options = options.cuda()
    
    return options

def load_LAMA_config(config_file):
    if isinstance(config_file, str) or isinstance(config_file, Path):
        with open(config_file, errors='ignore') as f:
            hyp = yaml.safe_load(f)
    return hyp


def generate_mask(dImg=0.006641, dDet=0.0072):
    """
        This function generates a mask for area of interest
        Parameters
        ----------
        dImg : float
            pixel size of image
        dDet : float
            pixel size of detector
    """
    imgN = 256
    m = np.arange(-imgN/2+1/2,imgN/2-1/2+1,1)
    m = m**2
    mask = np.zeros((imgN,imgN))
    for i in range(imgN):
        mask[:,i] = np.sqrt(m + m[i]) * dImg
    
    detL = dDet * 512
    dedge = detL / 2 -dDet / 2
    scanR = 500 / 100 / 2
    detR = 500 / 100 / 2
    dd = dedge * scanR / np.sqrt(dedge**2 + (scanR+detR)**2)
    
    return mask <= dd

def down_sample(dataset, n_views, fullview_dir, train=True):
    """
    This function downsamples full-view sinogram data into sparse-view sinogram data
    
    Parameters
    ----------
    dataset : str
        'mayo' or 'NBIA'
    n_views : int
        number of views for sparse-view CT. must be divisible by 1024.
    fullview_dir : str
        ex: 'FullViewNoiseless'
    train : bool, optional
        generate training or test set. The default is True.

    """
    
    # get files from full view data folder
    if train:
        data_dir = ROOT / 'dataset' / dataset / 'train'
    else:
        data_dir = ROOT / 'dataset' / dataset / 'test'
    F_list = sorted(list((data_dir / fullview_dir).glob('data*.mat')))
    
    # get full view number and number of partitions
    sample = scio.loadmat(F_list[0])['data']
    full_view = sample.shape[0]
    n_partition = full_view // n_views
    mode = 'train' if train else 'test'
    
    # generate save path
    save_dir = data_dir / f'{n_views}views'
    if save_dir.exists():
        LOGGER.info(f'Downsample ({dataset}, {mode}) to {n_views} views already exists!')
        return
    save_dir.mkdir(parents=True)
    
    for file in F_list:
        data = scio.loadmat(file)['data']
        sparse_data = np.zeros((n_views, data.shape[1]))
        for j in range(n_views):
            sparse_data[j,:] = data[j * n_partition, :]
        scio.savemat(save_dir / file.name, {'data': sparse_data})
    # print(f'Downsample ({dataset}, {mode}) to {n_views} views finished!')
    LOGGER.info(f'Downsample ({dataset}, {mode}) to {n_views} views finished!')
    
def fbp_data(dataset, n_views, train=True):
    """
    This function applies fbp to projection data to obtain initial images: range (0,1)

    Parameters
    ----------
    dataset : str
        'mayo' or 'NBIA'
    n_views : int
        number of views for sparse-view CT. must be divisible by 1024.
    train : bool, optional
        apply on training/test set. The default is True.

    """
    # get files from full view data folder
    if train:
        data_dir = ROOT / 'dataset' / dataset / 'train'
    else:
        data_dir = ROOT / 'dataset' / dataset / 'test'
    F_list = sorted(list((data_dir / f'{n_views}views').glob('data*.mat')))
    mode = 'train' if train else 'test'
    
    # generate save path
    save_path = data_dir / f'FBP_{n_views}views'
    if save_path.exists():
        LOGGER.info(f'FBP on sinogram ({dataset}, {mode}) {n_views} views already exists!')
        return
    save_path.mkdir(parents=True)

    cfg_file = ROOT / 'config' / f'{n_views}views.yaml'
    ct_cfg = load_CT_config(cfg_file)
    mask = generate_mask()
    for file in F_list:
        data = scio.loadmat(file)['data']
        data = torch.FloatTensor(data.reshape(1,1,n_views,512)).cuda().contiguous()
        recon_data = ctlib.fbp(data,ct_cfg)
        recon_data = recon_data.squeeze().detach().cpu().numpy()
        recon_data = recon_data * mask
        recon_data = recon_data / recon_data.max()
        recon_data = recon_data.clip(0,1)
        scio.savemat(save_path / file.name, {'data': recon_data})
    
    # print(f'FBP on sinogram ({dataset}, {mode}) {n_views} views finished!')
    LOGGER.info(f'FBP on sinogram ({dataset}, {mode}) {n_views} views finished!')