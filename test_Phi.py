# -*- coding: utf-8 -*-
"""
Created on Fri May 27 13:47:26 2022

@author: Chi Ding
"""

import torch
import torch.nn as nn
from utils_DDAD import *
import scipy.io as scio
import os
import platform
import numpy as np
from torch.utils.data import DataLoader

from argparse import ArgumentParser

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

parser = ArgumentParser(description='CT-Trial1-Test')

parser.add_argument('--epoch_num', type=int, default=500, help='epoch number of start training')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--phase_number', type=int, default=3, help='phase number for ISTA Net')
parser.add_argument('--sparse_view_num', type=int, default=128, help='number of sparse views')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for test dataloader')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')

parser.add_argument('--root_dir', type=str, default='NBIA', help='root directory')
parser.add_argument('--file_dir', type=str, default='projection_512views', help='parent files directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')
#%% define model and prepare for data
args = parser.parse_args()

epoch_num = args.epoch_num
learning_rate = args.learning_rate
phase_number = args.phase_number
batch_size = args.batch_size
group_num = args.group_num
gpu_list = args.gpu_list
root = os.path.join("../dataset",args.root_dir)
file_dir = args.file_dir
sparse_view_num = args.sparse_view_num

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_forward = ResNet(phase_number)
model_dir = "./%s/" % args.model_dir + args.root_dir[:4] + '_' +  "forward_ResNet_layer_%d_%dviews" % \
    (phase_number, sparse_view_num)

model_forward = nn.DataParallel(model_forward)
model_forward = model_forward.to(device)

# Load pre-trained model with epoch number
model_forward.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, epoch_num)))


model_backward = ResNet(phase_number)
model_dir = "./%s/" % args.model_dir + args.root_dir[:4] + '_' +  "backward_ResNet_layer_%d_%dviews" % \
    (phase_number, sparse_view_num)

model_backward = nn.DataParallel(model_backward)
model_backward = model_backward.to(device)

# Load pre-trained model with epoch number
model_backward.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, epoch_num)))

print('Load Data...')
if (platform.system() == 'Windows'):
    test_loader = DataLoader(dataset = Phi_loader(root, file_dir, sparse_view_num, False), 
                             batch_size=batch_size, num_workers=0,shuffle=False)
else:
    test_loader = DataLoader(dataset = Phi_loader(root, file_dir, sparse_view_num, False), 
                             batch_size=batch_size, num_workers=8, shuffle=False)
    
if (platform.system() == 'Windows'):
    train_loader = DataLoader(dataset = Phi_loader(root, file_dir, sparse_view_num, True), 
                             batch_size=batch_size, num_workers=0,shuffle=False)
else:
    train_loader = DataLoader(dataset = Phi_loader(root, file_dir, sparse_view_num, True), 
                             batch_size=batch_size, num_workers=8, shuffle=False)
    
#%% test
data_dir = os.path.join(root, 'test', file_dir)
first_file = os.listdir(data_dir)[0]
sample = scio.loadmat(os.path.join(data_dir, first_file))['data']

full_view_num = sample.shape[0]
n_partitions = full_view_num // sparse_view_num

views = 1024
dets = 512
width = 256
height = 256
dImg = 0.006641
dDet = 0.0072
Ang0 = 0
dAng = 0.006134
s2r = 2.5
d2r = 2.5
binshift = 0
scan_type = 0
mask = generate_mask(dImg, dDet)
options = torch.FloatTensor([full_view_num, dets, width, height, dImg, dDet, Ang0, 
                             dAng * (1024 // full_view_num), 
                             s2r, d2r, binshift, scan_type]).to(device)

folder_test = 'input' + '_' + str(sparse_view_num) + 'views'
folder_train = 'input' + '_' + str(sparse_view_num) + 'views'
folder_test_z = 'z' + '_' + str(sparse_view_num) + 'views'
folder_train_z = 'z' + '_' + str(sparse_view_num) + 'views'
save_dir_test = os.path.join(root, 'test', folder_test)
save_dir_train = os.path.join(root, 'train', folder_train)
save_dir_test_z = save_dir_test.replace(folder_test, folder_test_z)
save_dir_train_z = save_dir_train.replace(folder_train, folder_train_z)

if not os.path.exists(save_dir_test):
    os.mkdir(save_dir_test)
if not os.path.exists(save_dir_train):
    os.mkdir(save_dir_train)
    
if not os.path.exists(save_dir_test_z):
    os.mkdir(save_dir_test_z)
if not os.path.exists(save_dir_train_z):
    os.mkdir(save_dir_train_z)
    
    
#%% 
def generate_train_initial():
    print('Generating training initial input...')
    PSNR_list = []
    SSIM_list = []
    RMSE_list = []
    for _, everything in enumerate(train_loader):
        data, data_list, img, name = everything
        f_list_forward = []
        f_list_backward = []
        data = data.to(device)
        x = data_list[0]
        x = x.to(device)
        f_list_forward.append(x.squeeze())
        x0 = x
        for i in range(n_partitions//2):
            x0 = model_forward(x0)
            f_list_forward.append(x0.squeeze())
            
            
        x0 = x
        for i in range(n_partitions//2):
            x0 = model_backward(x0)
            if i == 0:
                x0 = torch.roll(x0, -1, 2)
            f_list_backward.append(x0.squeeze())
        
        f_list = []
        for i in range(n_partitions//2+1):
            f_list.append(f_list_forward.pop(0))
        f_list[-1] = (f_list[-1] + f_list_backward.pop())/2
        for i in range(n_partitions - n_partitions//2 - 1):
            f_list.append(f_list_backward.pop())
            
        assert len(f_list_backward) == 0
        assert len(f_list) == n_partitions
        
        f_all_pred = torch.zeros((full_view_num,512)).to(device)
        for i in range(n_partitions):
            for j in range(sparse_view_num):
                f_all_pred[i + n_partitions * j] = f_list[i][j,:]
        
        f_all_pred = f_all_pred.clip(0,data.max().item()).unsqueeze_(0).unsqueeze_(0)
        RMSE = torch.mean(torch.pow(f_all_pred-data,2))/torch.mean(torch.pow(data,2))
        img_recon = ctlib.fbp(f_all_pred.contiguous(), options)
        img_recon = img_recon.squeeze().detach().cpu().numpy()
        img = img.squeeze().detach().cpu().numpy()
        # print(img_recon)
        img_recon = img_recon / 3.84
        img_recon = img_recon.clip(0,1)
        img_recon = img_recon * mask
        
        scio.savemat(os.path.join(save_dir_train, name[0]), {'data': img_recon})
        scio.savemat(os.path.join(save_dir_train_z, name[0]), {'data': f_all_pred.squeeze().detach().cpu().numpy()})
        S = ssim(img_recon, img)
        P = psnr(img_recon, img)
        SSIM_list.append(S)
        PSNR_list.append(P)
        RMSE_list.append(RMSE.item())
        print(name[0] +' PSNR:', P, 'SSIM:', S, 
              'RMSE:', RMSE.item())
        
    print('average PSNR:', np.mean(np.array(PSNR_list)))
    print('average SSIM:', np.mean(np.array(SSIM_list)))
    print('average RMSE:', np.mean(np.array(RMSE_list)))
    print('\n\n\n\n\n')

def generate_test_initial():
    print('Generate test initial input...')
    PSNR_list = []
    SSIM_list = []
    RMSE_list = []
    for _, everything in enumerate(test_loader):
        data, data_list, img, name = everything
        f_list_forward = []
        f_list_backward = []
        data = data.to(device)
        x = data_list[0]
        x = x.to(device)
        f_list_forward.append(x.squeeze())
        x0 = x
        for i in range(n_partitions//2):
            x0 = model_forward(x0)
            f_list_forward.append(x0.squeeze())
            
            
        x0 = x
        for i in range(n_partitions//2):
            x0 = model_backward(x0)
            if i == 0:
                x0 = torch.roll(x0, -1, 2)
            f_list_backward.append(x0.squeeze())
        
        f_list = []
        for i in range(n_partitions//2+1):
            f_list.append(f_list_forward.pop(0))
        f_list[-1] = (f_list[-1] + f_list_backward.pop())/2
        for i in range(n_partitions - n_partitions//2 - 1):
            f_list.append(f_list_backward.pop())
            
        assert len(f_list_backward) == 0
        assert len(f_list) == n_partitions
        
        f_all_pred = torch.zeros((full_view_num,512)).to(device)
        for i in range(n_partitions):
            for j in range(sparse_view_num):
                f_all_pred[i + n_partitions*j] = f_list[i][j,:]
        
        f_all_pred = f_all_pred.clip(0,data.max().item()).unsqueeze_(0).unsqueeze_(0)
        RMSE = torch.mean(torch.pow(f_all_pred-data,2))/torch.mean(torch.pow(data,2))
        img_recon = ctlib.fbp(f_all_pred.contiguous(), options)
        img_recon = img_recon.squeeze().detach().cpu().numpy()
        img = img.squeeze().detach().cpu().numpy()
        # print(img_recon)
        img_recon = img_recon / 3.84
        img_recon = img_recon.clip(0,1)
        img_recon = img_recon * mask
        
        scio.savemat(os.path.join(save_dir_test, name[0]), {'data': img_recon})
        scio.savemat(os.path.join(save_dir_test_z, name[0]), {'data': f_all_pred.squeeze().detach().cpu().numpy()})
        S = ssim(img_recon, img)
        P = psnr(img_recon, img)
        SSIM_list.append(S)
        PSNR_list.append(P)
        RMSE_list.append(RMSE.item())
        print(name[0] +' PSNR:', P, 'SSIM:', S, 
              'RMSE:', RMSE.item())
        
    print('average PSNR:', np.mean(np.array(PSNR_list)))
    print('average SSIM:', np.mean(np.array(SSIM_list)))
    print('average RMSE:', np.mean(np.array(RMSE_list)))
        
generate_train_initial()
generate_test_initial()
