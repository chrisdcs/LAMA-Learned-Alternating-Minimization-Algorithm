import os
import sys
import argparse
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

datasets = ['mayo', 'NBIA']
views = [64, 128]


def generate_sparse_CT(datasets = ['mayo', 'NBIA'], n_views = [64, 128]):
    for dataset in datasets:
        for view in views:
            CT.down_sample(dataset, view, 'FullViewNoiseless', train=True)
            CT.down_sample(dataset, view, 'FullViewNoiseless', train=False)
            CT.fbp_data(dataset, view, train=True)
            CT.fbp_data(dataset, view, train=False)
    
    # print("Sparse-View CT data generation Done!")
    LOGGER.info("Sparse-View CT data generation Done!\n")
    
def run(
        device='',
        network='',
        n_views=64,
        dataset='mayo'
        ):
    model = InitNet()
    model = nn.DataParallel(model)
    model = model.to(device)
    weights = ROOT / 'models' / 'initNet' / f'{dataset}{n_views}-{network}.pkl'
    model.load_state_dict(torch.load(weights))
    
    mask = CT.generate_mask()
    ct_cfg = CT.load_CT_config(ROOT / 'config' / '512views.yaml')
    ct_cfg_full = CT.load_CT_config(ROOT / 'config' / 'full-view.yaml')
    dataset = DataLoader(CNN_loader(ROOT, dataset, n_views, train=True), batch_size=1, shuffle=False, num_workers=8)
    
    with torch.no_grad():
        avg_p = 0
        avg_s = 0
        for i, data in enumerate(dataset):
            x, label, file_name = data
            x = x.to(device)
            label = label.to(device)
            output = model(x)
            # output = output.squeeze().cpu().numpy()
            # scio.savemat(ROOT / 'dataset' / dataset / 'train' / f'{n_views}views' / file_name, {'data': output})
            
            img = ctlib.fbp(output / 3.84, ct_cfg)
            img = img.detach().cpu().squeeze().numpy()
            img = img.clip(0,1) * mask
            
            img_label = ctlib.fbp(label / 3.84, ct_cfg_full)
            img_label = img_label.squeeze().cpu().numpy()
            img_label = img_label.clip(0,1) * mask
            
            p = psnr(img, img_label)
            s = ssim(img, img_label)
            
            avg_p += p
            avg_s += s
            
            LOGGER.info(f'{file_name[0]}\tpsnr {p:.3f}\tssim {s:.3f}')
        LOGGER.info(f'avg_psnr {avg_p / (i+1):.3f}\tavg_ssim {avg_s / (i+1):.3f}')

def parse_opt():
    parser = argparse.ArgumentParser(description='Init-Net')
    parser.add_argument('--dataset', type=str, default='mayo', help='dataset name')
    parser.add_argument('--n_views', type=int, default=64, help='number of views')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--network', type=str, default='CNN', help='initialization network name')
    
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt
    
    

def main():
    opt = parse_opt()
    generate_sparse_CT()
    run(**vars(opt))

if __name__ == '__main__':
    main()