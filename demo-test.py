# the demo code for testing the pretrained model, note the pretrained model was not trained with seed, 
# if you want to reproduce the results, please train the model with seed, it may not be exactly the 
# same using the demo-train.py I provided, but should be close enough.

import os
import sys
import argparse
from pathlib import Path
from utils.general import print_args
from utils.model import LAMA
from utils.dataloaders import LAMA_loader
import utils.CT_helper as CT
from torch.utils.data import DataLoader

import torch
import ctlib
import numpy as np
import torch.nn as nn

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # relative path to current working directory
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_opt():
    parser = argparse.ArgumentParser(description='LAMA')
    parser.add_argument('--dataset', type=str, default='mayo', help='dataset name')
    parser.add_argument('--n_views', type=int, default=64, help='number of views')
    parser.add_argument('--n_iter', type=int, default=15, help='number of iterations')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--cfg', type=str, default='baseline', help='model.yaml type')
    
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt

def run(**kwargs):
    dataset = kwargs['dataset']
    model_type = kwargs['cfg']
    n_views = kwargs['n_views']
    n_iter = kwargs['n_iter']
    start_iter = n_iter
    
    # LAMA
    model = LAMA(n_iter = n_iter, start_iter = start_iter, n_views = n_views, type=model_type)
    model = nn.DataParallel(model)
    model = model.to(device)
    filename = ROOT / 'models' / f'{dataset}{n_views}-LAMA.pth.tar'
    checkpoint = torch.load(filename)
    if checkpoint:
        print("=> loaded checkpoint '{}' (iter {}, epoch {})"
                    .format(filename, checkpoint['iter'], checkpoint['epoch']))
    else:
        print("=> no checkpoint found")
    model.load_state_dict(checkpoint['state_dict'])
    
    dataroot = ROOT / "dataset" / "mayo"
    test_loader = DataLoader(dataset=LAMA_loader(
                                dataroot, "Img_CNN_64views", "Sin_CNN_64views", False), 
                                batch_size=1, num_workers=8,shuffle=False)
    mask = CT.generate_mask()
    mask = torch.FloatTensor(mask[None,None,:,:]).to(device)
    options = CT.load_CT_config(ROOT / 'config' / 'full-view.yaml')

    if dataset[:4] == 'mayo':
        n_samples = 100
    elif dataset[:4] == 'NBIA':
        n_samples = 40
    PSNR_All = np.zeros((n_samples), dtype=np.float32)
    SSIM_All = np.zeros((n_samples), dtype=np.float32)

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x, x_label, z, file_name = data
            # prepare data
            x = x.to(device)
            z = z.to(device)

            x_list, z_list = model(x,z)
            x_out = x_list[-1].clip(0,1) * mask
            x_out = x_out.squeeze().detach().cpu().numpy()
            x_label = x_label.squeeze().detach().cpu().numpy()
            
            p = psnr(x_out, x_label, data_range=1)
            s = ssim(x_out, x_label, data_range=1)
            PSNR_All[i] = p
            SSIM_All[i] = s
            
            print(file_name[0], "\t psnr: {:.4f}".format(p), "\t ssim: {:.4f}".format(s))

    avg_psnr = np.mean(PSNR_All)
    psnr_var = np.var(PSNR_All)
    avg_ssim = np.mean(SSIM_All)
    ssim_var = np.var(SSIM_All)
    print()
    print("avg PSNR:", avg_psnr, "\t std:", psnr_var)
    print("avg SSIM:", avg_ssim, "\t std:", ssim_var)
    print()

def main():
    opt = parse_opt()
    run(**vars(opt))
    
if __name__ == '__main__':
    main()