import numpy as np
import os
import scipy.io as scio
import glob


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

def down_sample(dataset_name, num_sparse_view, full_view_folder, save_folder_name, train=True):
    """
    This function downsamples full-view sinogram data into sparse-view sinogram data
    
    Parameters
    ----------
    dataset_name : str
        'mayo' or 'NBIA'
    num_sparse_view : int
        number of views for sparse-view CT. must be divisible by 1024.
    full_view_folder : str
        ex: 'FullViewNoiseless'
    save_folder_name : str
        ex: 'Sparse'
    train : bool, optional
        generate training or test set. The default is True.

    """
    
    # get files from full view data folder
    if train:
        file_path = dataset_name + r'/train'
    else:
        file_path = dataset_name + r'/test'
    files = sorted(glob.glob(os.path.join(file_path, full_view_folder, 'data')+'*.mat'))
    
    # get full view number and number of partitions
    l = len(files)
    sample = scio.loadmat(files[0])['data']
    full_view_num = sample.shape[0]
    n_partition = full_view_num//num_sparse_view
    
    # generate save path
    save_path = os.path.join(file_path, save_folder_name + '_' + str(num_sparse_view) + 'views')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    for i in range(l):
        print('downsampling', files[i][-13:])
        data = scio.loadmat(files[i])['data']
        sparse_data = np.zeros((num_sparse_view, data.shape[1]))
        for j in range(num_sparse_view):
            sparse_data[j,:] = data[j * n_partition, :]
        name = os.path.join(save_path, files[i][-13:])
        scio.savemat(name, {'data': sparse_data})
        
    print('Done!')
    
def fbp_data(dataset_name, num_sparse_view, sparse_sinogram_folder, save_folder_name, train=True):
    """
    This function applies fbp to projection data to obtain initial images: range (0,1)

    Parameters
    ----------
    dataset_name : str
        'mayo' or 'NBIA'
    num_sparse_view : int
        number of views for sparse-view CT. must be divisible by 1024.
    sparse_sinogram_folder : str
        the folder that stores sparse-view sinogram data.
    save_folder_name : str
        the name you want to call the folder for storing fbp images.
    train : bool, optional
        apply on training/test set. The default is True.

    """
    # get files from full view data folder
    if train:
        file_path = dataset_name + r'/train'
    else:
        file_path = dataset_name + r'/test'
    files = sorted(glob.glob(os.path.join(file_path, sparse_sinogram_folder, 'data')+'*.mat'))
    
    # get full view number and number of partitions
    l = len(files)
    
    # generate save path
    save_path = os.path.join(file_path, save_folder_name + '_' + str(num_sparse_view) + 'views')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    ratio = 1024//num_sparse_view
    options = torch.Tensor([num_sparse_view,512,256,256,0.006641,0.0072,0,0.006134*(ratio),2.5,2.5,0,0])
    options = options.cuda()
    mask = generate_mask()
    for i in range(l):
        
        data = scio.loadmat(files[i])['data']
        data = torch.FloatTensor(data.reshape(1,1,num_sparse_view,512)).cuda().contiguous()
        recon_data = ctlib.fbp(data,options)
        recon_data = recon_data.squeeze().detach().cpu().numpy()
        recon_data = recon_data * mask
        recon_data = recon_data / recon_data.max()
        recon_data = recon_data.clip(0,1)
        name = os.path.join(save_path, files[i][-13:])
        scio.savemat(name, {'data': recon_data})
        
    print('Done!')


