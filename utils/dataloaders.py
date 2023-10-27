import os
import glob
import torch
import scipy.io as scio
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize

# Simple CNN Init-Net Data Loader
class CNN_loader(Dataset):
    def __init__(self, root, file_path, num_sparse_view, train):
        self.train = train
        if train:
            folder = 'train'
        else:
            folder = 'test'
        
        self.file_path = file_path
        self.num_sparse_view = num_sparse_view
        self.files = sorted(glob.glob(os.path.join(root, folder, file_path, 'data')+'*.mat'))
        self.index = [(i) for i in range(0,512,512/ num_sparse_view)]
    
    def __getitem__(self, index):
        file = self.files[index]
        data = scio.loadmat(file)['data']
        sparse = data[self.index]
        
        data = torch.FloatTensor(data).unsqueeze_(0)
        sparse = torch.FloatTensor(sparse).unsqueeze_(0)
        sparse = resize(sparse, [512,512])
        
        return sparse, data, file[-13:]
    
    def __len__(self):
        return len(self.files)

# Init-Net Phi Data Loader
class Phi_loader(Dataset):
    def __init__(self, root, file_path, num_sparse_view, train):
        self.train = train
        if train == True:
            folder = 'train'
        else:
            folder = 'test'
        
        self.file_path = file_path
        self.num_sparse_view = num_sparse_view
        self.files = sorted(glob.glob(os.path.join(root, folder, file_path, 'data')+'*.mat'))
        self.full_view_num = scio.loadmat(self.files[0])['data'].shape[0]
        self.n_partition = self.full_view_num // self.num_sparse_view
        
    
    def __getitem__(self, index):
        file = self.files[index]
        data_list = []
        data = scio.loadmat(file)['data']
        for i in range(self.n_partition):
            f_i = np.zeros((self.num_sparse_view, data.shape[1]))
            for j in range(self.num_sparse_view):
                f_i[j,:] = data[i + j*self.n_partition, :]
            data_list.append(torch.FloatTensor(f_i).unsqueeze_(0))
        
        img_file = file.replace(self.file_path, 'label_single')
        data = torch.FloatTensor(data).unsqueeze_(0)
        img = scio.loadmat(img_file)['data']
        img = torch.FloatTensor(img).unsqueeze_(0)
        
        return data, data_list, img, file[-13:]
    
    def __len__(self):
        return len(self.files)

    
# Data Loader for LAMA
class LAMA_loader(Dataset):
    # need projection data, ground truth and input images
    def __init__(self, root, file_path, prj_file_path, num_sparse_view, train):
        self.train = train
        if train == True:
            folder = 'train'
        else:
            folder = 'test'
        
        self.file_path = file_path
        self.prj_file_path = prj_file_path
        self.num_sparse_view = num_sparse_view
        self.files = sorted(glob.glob(os.path.join(root, folder, self.file_path, 'data')+'*.mat'))
        
        
    
    def __getitem__(self, index):
        file = self.files[index]
        file_prj = file.replace(self.file_path, self.prj_file_path)
        file_label = file.replace(self.file_path, 'label_single')
        
        input_data = scio.loadmat(file)['data']
        prj_data = scio.loadmat(file_prj)['data'] / 3.84
        label_data = scio.loadmat(file_label)['data']
        
        input_data = torch.FloatTensor(input_data).unsqueeze_(0)
        prj_data = torch.FloatTensor(prj_data).unsqueeze_(0)
        label_data = torch.FloatTensor(label_data).unsqueeze_(0)
        
        if prj_data.shape[1] < 512:
            prj_data = resize(prj_data, [512,512])
        if self.train:
            return input_data, label_data, prj_data
        else:
            return input_data, label_data, prj_data, file[-13:]
    
    def __len__(self):
        return len(self.files)