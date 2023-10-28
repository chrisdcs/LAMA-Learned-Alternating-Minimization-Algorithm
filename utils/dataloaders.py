import os
import glob
from pathlib import Path
import torch
import scipy.io as scio
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize

# Simple CNN Init-Net Data Loader
class CNN_loader(Dataset):
    def __init__(self, root, dataset, n_views, train):
        self.train = train
        if train:
            mode = 'train'
        else:
            mode = 'test'
        
        self.n_views = n_views
        path = root / 'dataset' / dataset / mode / f'{n_views}views'
        self.F_list = sorted(list(path.glob('data*.mat')))
    
    def __getitem__(self, index):
        file = self.F_list[index]
        label = Path(str(file).replace(f'{self.n_views}views', 'FullViewNoiseless'))
        data = scio.loadmat(file)['data']
        ground_truth = scio.loadmat(label)['data']
        ground_truth = torch.FloatTensor(ground_truth).unsqueeze_(0)
        
        x = np.zeros((512, 512))
        x[::512 // self.n_views, :] = data
        x = torch.FloatTensor(x).unsqueeze_(0)
        
        return x, ground_truth, file.name
    
    def __len__(self):
        return len(self.F_list)

# Init-Net Phi Data Loader
class Phi_loader(Dataset):
    def __init__(self, root, file_path, n_views, train):
        self.train = train
        if train == True:
            folder = 'train'
        else:
            folder = 'test'
        
        self.file_path = file_path
        self.n_views = n_views
        self.F_list = sorted(glob.glob(os.path.join(root, folder, file_path, 'data')+'*.mat'))
        self.full_view_num = scio.loadmat(self.F_list[0])['data'].shape[0]
        self.n_partition = self.full_view_num // self.n_views
        
    
    def __getitem__(self, index):
        file = self.F_list[index]
        data_list = []
        data = scio.loadmat(file)['data']
        for i in range(self.n_partition):
            f_i = np.zeros((self.n_views, data.shape[1]))
            for j in range(self.n_views):
                f_i[j,:] = data[i + j*self.n_partition, :]
            data_list.append(torch.FloatTensor(f_i).unsqueeze_(0))
        
        img_file = file.replace(self.file_path, 'label_single')
        data = torch.FloatTensor(data).unsqueeze_(0)
        img = scio.loadmat(img_file)['data']
        img = torch.FloatTensor(img).unsqueeze_(0)
        
        return data, data_list, img, file[-13:]
    
    def __len__(self):
        return len(self.F_list)

    
# Data Loader for LAMA
class LAMA_loader(Dataset):
    # need projection data, ground truth and input images
    def __init__(self, root, file_path, prj_file_path, n_views, train):
        self.train = train
        if train == True:
            folder = 'train'
        else:
            folder = 'test'
        
        self.file_path = file_path
        self.prj_file_path = prj_file_path
        self.n_views = n_views
        self.F_list = sorted(glob.glob(os.path.join(root, folder, self.file_path, 'data')+'*.mat'))
        
        
    
    def __getitem__(self, index):
        file = self.F_list[index]
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
        return len(self.F_list)