# -*- coding: utf-8 -*-
"""
Created on Wed May 25 13:12:58 2022

@author: Chi Ding
"""

from utils import *
import torch.nn as nn
import torch
import os
import platform
# from dataset import trainset_loader
from torch.utils.data import DataLoader

from argparse import ArgumentParser
import scipy.io as scio

parser = ArgumentParser(description='CT-Trial1')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=500, help='epoch number of end training')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--phase_number', type=int, default=3, help='phase number of ResNet archi.')
parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
parser.add_argument('--sparse_view_num', type=int, default=64, help='number of sparse views')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')

parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--root_dir', type=str, default='mayo_data_low_dose_256', help='root directory')
parser.add_argument('--file_dir', type=str, default='projection_512views', help='parent files directory')

args = parser.parse_args()

start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
phase_number = args.phase_number
batch_size = args.batch_size
group_num = args.group_num
sparse_view_num = args.sparse_view_num
gpu_list = args.gpu_list

root = args.root_dir
file_dir = args.file_dir

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Training
# get parameters
def train(direction):
    model = ResNet(phase_number)
    model = nn.DataParallel(model)
    model = model.to(device)

    print_flag = 1   # print parameter number

    if print_flag:
        num_count = 0
        for para in model.parameters():
            num_count += 1
            print('Layer %d' % num_count)
            print(para.size())

    print('Load Data...')    
    if (platform.system() == 'Windows'):
        rand_loader = DataLoader(dataset = Phi_loader(root, file_dir, sparse_view_num, True), 
                                 batch_size=batch_size, num_workers=0,shuffle=True)
    else:
        rand_loader = DataLoader(dataset = Phi_loader(root, file_dir, sparse_view_num, True), 
                                 batch_size=batch_size, num_workers=8,shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model_dir = "./%s/" % args.model_dir + direction + "_ResNet_layer_%d_group_%d_lr_%.4f" % \
        (phase_number, group_num, learning_rate)
    log_file_name = "./%s/" % args.log_dir + direction + "_ResNet_layer_%d_group_%d_lr_%.4f.txt" % \
        (phase_number, group_num, learning_rate)

    # generate model saving directory
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # generate log saving directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        
    if start_epoch > 0:
        # continue training if start epoch is not 0
        pre_model_dir = model_dir
        model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (pre_model_dir, start_epoch), 
                                         map_location=device))
    # start training
    sample = scio.loadmat(os.path.join(root, 'train', file_dir, 'data_0001.mat'))['data']
    full_view_num = sample.shape[0]
    n_partitions = full_view_num // sparse_view_num
    
    for epoch_i in range(start_epoch+1, end_epoch+1):
        for _, everything in enumerate(rand_loader):
            loss_discrepancy_total = 0
            data, data_list, img, name = everything
            
            if direction == 'forward':
                for i in range(n_partitions-1):
                    x = data_list[i]
                    x = x.to(device)
                    x_next = data_list[i+1]
                    x_next = x_next.to(device)
                    x_output = model(x)
                
                    # compute and print loss
                    loss_discrepancy = torch.mean(torch.pow(x_output - x_next, 2))
                    loss_discrepancy_total += loss_discrepancy
                
                x = data_list[-1].to(device)
                x_output = model(x)
                x_next = data_list[0].to(device)
                x_next = torch.roll(x_next, -1, 2)
                
                loss_discrepancy_total += torch.mean(torch.pow(x_output - x_next, 2))
            
            elif direction == 'backward':
                x = data_list[0]
                x = x.to(device)
                x_output = model(x)
                x_prev = data_list[-1].to(device)
                x_prev = torch.roll(x_prev, 1, 2)
                
                loss_discrepancy_total += torch.mean(torch.pow(x_output - x_prev, 2))
                
                for i in range(n_partitions-1,0,-1):
                    x = data_list[i]
                    x = x.to(device)
                    x_prev = data_list[i-1]
                    x_prev = x_prev.to(device)
                    x_output = model(x)
                
                    # compute and print loss
                    loss_discrepancy = torch.mean(torch.pow(x_output - x_prev, 2))
                    loss_discrepancy_total += loss_discrepancy
                    
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss_discrepancy_total.backward()
            optimizer.step()
    
            output_data = "[%02d/%02d] Total Loss: %.4f\n" % \
                (epoch_i, end_epoch, loss_discrepancy_total.item())
            print(output_data)
            
        output_file = open(log_file_name, 'a')
        output_file.write(output_data)
        output_file.close()
    
        if epoch_i % 5 == 0:
            torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters
    
train('forward')
train('backward')