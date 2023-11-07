import os
import sys
import argparse
import scipy.io as scio
import utils.CT_helper as CT
from utils.model import InitNet
from utils.dataloaders import CNN_loader
from utils.general import print_args, LOGGER

import torch
import ctlib
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # relative path to current working directory

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
datasets = ['mayo', 'NBIA']
views = [64, 128]

def parse_opt():
    parser = argparse.ArgumentParser(description='Init-Net')
    parser.add_argument('--dataset', type=str, default='mayo', help='name of dataset')
    parser.add_argument('--train', type=str, default='True', help='training data or test data')
    parser.add_argument('--n_views', type=int, default=64, help='number of views for sparse-view CT')
    parser.add_argument('--network', type=str, default='CNN', help='initialization network name')
    
    opt = parser.parse_args()
    opt.train = True if opt.train.upper() == 'TRUE' else False
    print_args(vars(opt))
    return opt

def generate_sparse_CT(dataset, n_views, train=True):
    LOGGER.info("Sparse-View CT data generation start!")
    CT.down_sample(dataset, n_views, 'FullViewNoiseless', train=train)
    CT.fbp_data(dataset, n_views, train=train)
    LOGGER.info("Sparse-View CT data generation complete!\n")

def initNet_data(dataset, n_views, network, train=True):
    # will add code for different networks later
    model = InitNet()
    model = nn.DataParallel(model)
    model = model.to(device)
    weights = ROOT / 'models' / 'initNet' / f'{dataset}{n_views}-{network}.pkl'
    model.load_state_dict(torch.load(weights))
    mode = 'train' if train else 'test'
    img_save_dir = ROOT / 'dataset' / dataset / mode / f'Img_{network}_{n_views}views'
    sin_save_dir = ROOT / 'dataset' / dataset / mode / f'Sin_{network}_{n_views}views'
    if img_save_dir.exists() and sin_save_dir.exists():
        LOGGER.info(f"Init-Net data generation ({dataset} {mode} {n_views}views {network}) already exists!")
        LOGGER.info(f"Init-Net data generation ({dataset} {mode} {n_views}views {network}) complete!\n")
        return
    img_save_dir.mkdir(parents=True)
    sin_save_dir.mkdir(parents=True)
    mask = CT.generate_mask()
    ct_cfg = CT.load_CT_config(ROOT / 'config' / '512views.yaml')
    ct_cfg_full = CT.load_CT_config(ROOT / 'config' / 'full-view.yaml')
    data_loader = DataLoader(CNN_loader(ROOT, dataset, n_views, train=train), batch_size=1, shuffle=False, num_workers=8)
    
    LOGGER.info(f"Init-Net data generation ({dataset} {mode} {n_views}views {network}) start!")
    with torch.no_grad():
        avg_p = 0
        avg_s = 0
        for i, data in enumerate(data_loader):
            x, label, file_name = data
            x = x.to(device)
            label = label.to(device)
            output = model(x)
            
            img = ctlib.fbp(output / 3.84, ct_cfg)
            img = img.detach().cpu().squeeze().numpy()
            img = img.clip(0,1) * mask
            sinogram = output.squeeze().detach().cpu().numpy()
            
            scio.savemat(img_save_dir / file_name[0], {'data': img})
            scio.savemat(sin_save_dir / file_name[0], {'data': sinogram})
            
            img_label = ctlib.fbp(label / 3.84, ct_cfg_full)
            img_label = img_label.squeeze().cpu().numpy()
            img_label = img_label.clip(0,1) * mask
            
            p = psnr(img, img_label)
            s = ssim(img, img_label)
            
            avg_p += p
            avg_s += s
            
            # LOGGER.info(f'{file_name[0]}\tpsnr {p:.3f}\tssim {s:.3f}')
        LOGGER.info(f"avg_psnr {avg_p / (i+1):.3f}\tavg_ssim {avg_s / (i+1):.3f}")
        LOGGER.info(f"Init-Net data generation ({dataset} {mode} {n_views}views {network}) complete!\n")

def run(**kwargs):
    dataset = kwargs['dataset']
    n_views = kwargs['n_views']
    train = kwargs['train']
    network = kwargs['network']
    
    generate_sparse_CT(dataset, n_views, train=train)
    initNet_data(dataset, n_views, network, train=train)
    
    

def main():
    opt = parse_opt()
    run(**vars(opt))

if __name__ == '__main__':
    main()