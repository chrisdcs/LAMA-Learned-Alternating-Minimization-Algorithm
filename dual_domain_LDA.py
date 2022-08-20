# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 11:00:58 2022

@author: Chi Ding
"""

import torch
import torch.nn as nn
from pytorch_msssim import ssim
from torch.utils.data import DataLoader

from utils import *
import numpy as np
import ctlib

# import numpy as np
import os
import platform

from argparse import ArgumentParser
from skimage.metrics import peak_signal_noise_ratio as psnr

parser = ArgumentParser(description='Learnable Descent Algorithms (LDA)')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=200, help='epoch number of end training')
parser.add_argument('--start_phase', type=int, default=3, help='phase number of start training')
parser.add_argument('--end_phase', type=int, default=15, help='phase number of end training')
parser.add_argument('--layer_num', type=int, default=15, help='phase number of LDA-Net')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decaying learning rate by it every 100 epochs')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for loading data')
parser.add_argument('--alpha', type=float, default=1e-12, help='alpha parameter')
parser.add_argument('--beta', type=float, default=1e-12, help='beta parameter')
parser.add_argument('--mu', type=float, default=1e-12, help='beta parameter')
parser.add_argument('--nu', type=float, default=1e-12, help='beta parameter')
parser.add_argument('--lam', type=float, default=10., help='beta parameter')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--root_dir', type=str, default='mayo_data_low_dose_256', help='root directory')
parser.add_argument('--file_dir', type=str, default='input_64views', help='input files directory')
parser.add_argument('--file_prj_dir', type=str, default='z_64views', help='projection files directory')
parser.add_argument('--sparse_view_num', type=int, default=64, help='number of sparse views')

parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')

args = parser.parse_args()

#%% experiement setup
start_epoch = args.start_epoch
end_epoch = args.end_epoch
start_phase = args.start_phase
end_phase = args.end_phase
learning_rate = args.learning_rate
decay_rate = args.decay_rate
layer_num = args.layer_num
alpha = args.alpha
beta = args.beta
mu = args.mu
nu = args.nu
lam = args.lam
group_num = args.group_num
gpu_list = args.gpu_list
batch_size = args.batch_size
root = args.root_dir
file_dir = args.file_dir
file_prj_dir = args.file_prj_dir
sparse_view_num = args.sparse_view_num

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_dir = "./%s/DDLDA_layer_%d_group_%d_lr_%.4f" % \
    (args.model_dir, layer_num, group_num, learning_rate)
log_file_name = "./%s/DDLDA_layer_%d_group_%d_lr_%.4f.txt" % \
    (args.log_dir, layer_num, group_num, learning_rate)
#%% Load Data
print('Load Data...')
if (platform.system() == 'Windows'):
    rand_loader = DataLoader(dataset=Random_loader(root, file_dir, file_prj_dir, sparse_view_num, True), 
                             batch_size=batch_size, num_workers=0,shuffle=True)
else:
    rand_loader = DataLoader(dataset=Random_loader(root, file_dir, file_prj_dir, sparse_view_num, True), 
                             batch_size=batch_size, num_workers=8,shuffle=True)
#%% initialize model
model = Dual_Domain_LDA(layer_num, start_phase, sparse_view_num, alpha, beta, mu, nu, lam)
model = nn.DataParallel(model)
model.to(device)

# if not start from beginning load pretrained models
if start_epoch > 0:
    # start from checkpoint
    model.load_state_dict(torch.load('%s/net_params_epoch%d_phase%d.pkl' % \
                                     (model_dir, start_epoch, start_phase), 
                                     map_location=device))
    if start_phase == 3:
        power = start_phase - np.ceil((500-start_epoch)/100)
    else:
        power = start_phase + 2 - np.ceil((end_epoch-start_epoch)/100)
    learning_rate = learning_rate * decay_rate**power

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=decay_rate)

print_flag = 1   # print parameter number

if print_flag:
    num_count = 0
    for para in model.parameters():
        num_count += 1
        print('Layer %d' % num_count)
        print(para.size())

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
#%% training
mask = generate_mask(0.006641,0.0072)
mask = torch.FloatTensor(mask[None,None,:,:]).to(device)
for PhaseNo in range(start_phase, end_phase+1, 2):
    # add new phases
    model.module.set_PhaseNo(PhaseNo)
    if PhaseNo == 3:
        end_epoch = 500
    else:
        end_epoch = args.end_epoch
        
        
    for epoch_i in range(start_epoch+1, end_epoch+1):
        progress = 0
        PSNR_list = []
        loss_list = []
        
        for _, data in enumerate(rand_loader):
            input_data, label_data, prj_data = data
            progress += 1
            
            # prepare data
            input_data = input_data.to(device)
            label_data = label_data.to(device)
            prj_data = prj_data.to(device)
            
            x_list, prj_list = model(input_data,prj_data)
            x_output = x_list[-1].clip(0,1) * mask
            prj_output = prj_list[-1].clip(0)
            prj_label = projection.apply(label_data, model.module.options_sparse_view)
            
            # compute and print loss
            rec_loss = 0
            ssim_loss = 0
            
            rec_loss = torch.mean(torch.pow(x_output-label_data,2))
            sinogram_loss = torch.mean(torch.pow(prj_output-prj_label,2))
            ssim_loss = 1-ssim(x_output,label_data,data_range=1)
            
            loss_all = rec_loss + sinogram_loss + 0.01 * ssim_loss
            loss_list.append(loss_all.item())
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            
            if PhaseNo > 3 or epoch_i > 10:
                assert loss_all.item() < 0.5
            
            
            p = 0
            for i in range(batch_size):
                output = x_output[i].detach().cpu().numpy().squeeze()
                label = label_data[i].detach().cpu().numpy().squeeze()
                p += psnr(output.astype(label.dtype), label)
                
            p /= batch_size
            PSNR_list.append(p)
            # print progress
            if progress % 20 == 0:
                output_data = "[Phase %02d] [Epoch %02d/%02d] Total Loss: %.4f" % \
                    (PhaseNo, epoch_i, end_epoch, loss_all.item()) \
                    + "\t progress: %02f" % (progress * batch_size / 400 * 100) \
                    + '%\t psnr: ' + str(p) + "\n"
                print(output_data)
        
        avg_p = np.mean(PSNR_list)
        avg_l = np.mean(loss_list)
        epoch_data = '[Phase %02d] [Epoch %02d/%02d] Avg Loss: %.4f \t Avg PSNR: %.4f\n\n' % \
            (PhaseNo, epoch_i, end_epoch, avg_l, avg_p)
        print(epoch_data)
        output_file = open(log_file_name, 'a')
        output_file.write(epoch_data)
        output_file.close()
        
        
        # save the parameters
        if epoch_i % 10 == 0:
            # save the parameters
            torch.save(model.state_dict(), "./%s/net_params_epoch%d_phase%d.pkl" % \
                       (model_dir, epoch_i, PhaseNo))
        
        # reduce learning rate every 100 epochs
        if epoch_i % 100 == 0:
            scheduler.step()
                
    
    # after finish training current phases, introduce new phases and start from epoch 0
    start_epoch = 0
