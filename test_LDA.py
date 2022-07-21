# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 15:37:49 2022

@author: Chi Ding
"""

import torch
import torch.nn as nn
from utils import *
from torch.utils.data import DataLoader
import platform
import scipy.io as scio

import numpy as np
import os

from argparse import ArgumentParser
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

parser = ArgumentParser(description='Learnable Descent Algorithms (LDA)')

parser.add_argument('--epoch_num', type=int, default=300, help='epoch number of start training')
parser.add_argument('--phase_num', type=int, default=9, help='phase number of start training')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--layer_num', type=int, default=15, help='phase number of LDA-Net')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha parameter')
parser.add_argument('--beta', type=float, default=0.2, help='beta parameter')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--root_dir', type=str, default='mayo_data_low_dose_256', help='root directory')
parser.add_argument('--file_dir', type=str, default='input_64views', help='parent files directory')
parser.add_argument('--file_prj_dir', type=str, default='z_64views', help='projection files directory')
parser.add_argument('--sparse_view_num', type=int, default=64, help='number of sparse views')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for loading data')

parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--test_name', type=str, default='mayo', help='name of test set')

args = parser.parse_args()

#%% experiement setup
epoch_num = args.epoch_num
phase_num = args.phase_num
layer_num = args.layer_num
group_num = args.group_num
learning_rate = args.learning_rate
alpha = args.alpha
beta = args.beta
gpu_list = args.gpu_list
test_name = args.test_name
root = args.root_dir
file_dir = args.file_dir
file_prj_dir = args.file_prj_dir
sparse_view_num = args.sparse_view_num
batch_size = args.batch_size

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%% Load Data
print('Load Data...')

if (platform.system() == 'Windows'):
    test_loader = DataLoader(dataset=Random_loader(root, file_dir, file_prj_dir, sparse_view_num, False), 
                             batch_size=batch_size, num_workers=0,shuffle=False)
else:
    test_loader = DataLoader(dataset=Random_loader(root, file_dir, file_prj_dir, sparse_view_num, False), 
                             batch_size=batch_size, num_workers=8,shuffle=False)

result_dir = os.path.join(args.result_dir, test_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    
PSNR_All = np.zeros((100), dtype=np.float32)
SSIM_All = np.zeros((100), dtype=np.float32)
#%% test
model = LDA_weighted(layer_num, phase_num, sparse_view_num, alpha, beta)
model = nn.DataParallel(model)
model.to(device)

model_dir = "./%s/LDA_layer_%d_group_%d_lr_%.4f" % (
                    args.model_dir, layer_num, group_num, learning_rate)# +'_input_prj'
# Load pre-trained model with epoch number
model.load_state_dict(torch.load("./%s/net_params_epoch%d_phase%d.pkl" % \
           (model_dir, epoch_num, phase_num)))
print(model.module.reg_ratios)
print('\n')
print("Reconstruction Start")
mask = generate_mask(0.006641,0.0072)
f_mask = torch.zeros((1,1,sparse_view_num*8,sparse_view_num*8))
for i in range(sparse_view_num):
    f_mask[:,:,i*8,:] = 1
    
with torch.no_grad():
    for i, data in enumerate(test_loader):
        input_data, label_data, prj_data, name = data
        input_data = input_data.to(device)
        label_data = label_data.squeeze().numpy()
        prj_data = prj_data.to(device)
        
        x_list, _ = model(input_data,prj_data,f_mask)
        x_output = x_list[-1]
        x_output = x_output.squeeze().detach().cpu().numpy() * mask
        
        # x_output = x_output / x_output.max()
        x_output = x_output.clip(0,1)
        p = psnr(x_output.astype(label_data.dtype), label_data, data_range=1)
        s = ssim(x_output.astype(label_data.dtype), label_data, data_range=1)
        PSNR_All[i] = p
        SSIM_All[i] = s
        
        print(name[0],'psnr:', p, 'ssim:', s)
        scio.savemat(os.path.join(result_dir, name[0]), {'data': x_output})
        
print('average psnr', np.mean(PSNR_All))
print('average ssim', np.mean(SSIM_All))
