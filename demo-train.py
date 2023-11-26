import os
import sys
import shutil
import argparse
from pathlib import Path
from utils.general import print_args, init_seeds
from utils.model import *
from utils.dataloaders import *
import utils.CT_helper as CT
from torch.utils.data import DataLoader

import torch
import ctlib
import numpy as np
import torch.nn as nn

from pytorch_msssim import ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # relative path to current working directory
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class pipeline:
        
    def __init__(self, **kwargs):
        self.args = kwargs
        self.model = LAMA(n_iter = self.args['n_iter'], 
                          start_iter = self.args['start_iter'],
                          n_views = self.args['n_views'], 
                          type=self.args['cfg'])
        self.model = nn.DataParallel(self.model)
        self.initialize_weights()
        self.model.to(device)
        
        # image net and sinogram net needs to be trained separately
        self.optim = torch.optim.Adam([
                                    {'params':self.model.module.ImgNet.parameters(), 'lr': self.args['lr_I']},
                                    {'params':self.model.module.SNet.parameters(), 'lr': self.args['lr_S']},
                                    {'params':[param for param in self.model.module.hyper_params], 'lr': self.args['lr_p']}
                                    ])
        
        # scheduler to reduce learning rate
        self.sched = torch.optim.lr_scheduler.StepLR(self.optim, step_size=1, gamma=self.args['decay_rate'])
        self.avg_loss = {'s-loss': 0, 'i-loss': 0, 't-loss': 0}
        mask = CT.generate_mask()
        self.mask = torch.FloatTensor(mask[None,None,:,:]).to(device)
        
        self.best_psnr = 0
        self.start_epoch = 0
        self.steps = 0
        self.done = False
        self.is_scheduler = self.args['is_scheduler'] 
        self.is_continue = self.args['is_continue']
        self.start_iter = self.args['start_iter']
        
        self.save_dir = ROOT / 'runs' / f'{self.args["dataset"]}{self.args["n_views"]}-{self.args["cfg"]}-LAMA'
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)
        self.train_data = DataLoader(dataset=LAMA_loader(ROOT / 'dataset' / self.args['dataset'], 
                                                         self.args['img_dir'], self.args['prj_dir'], train=True), 
                                    batch_size=self.args['batch_size'], num_workers=self.args['n_cpu'], shuffle=True)
        
    def initialize_weights(self):
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
                
    def save_checkpoint(self, state, is_best, 
                        filename='checkpoint.pth.tar'):
        torch.save(state, self.save_dir / filename)
        if is_best:
            shutil.copyfile(self.save_dir / filename, self.save_dir / 'model_best.pth.tar')
    
    def load_checkpoint(self, filename='checkpoint.pth.tar'):
        if (self.save_dir / filename).exists():
            checkpoint = torch.load(filename, map_location=device)
            self.start_iter = checkpoint['iter']
            self.start_epoch = checkpoint['epoch']
            self.best_psnr = checkpoint['best_psnr']
            self.avg_loss = checkpoint['avg_loss']
            self.steps = checkpoint['steps']

            self.model.load_state_dict(checkpoint['state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer'])
            self.sched.load_state_dict(checkpoint['scheduler'])

            print("=> loaded checkpoint '{}' (iter {}, epoch {})"
                  .format(filename, checkpoint['iter'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(filename))
            
    def train(self):
        if self.is_continue:
            self.load_checkpoint()
        
        for itr in range(self.start_iter, self.args['n_iter']+1, 2):
            # increment 2 phases (iterations) every time
            self.model.module.set_iteration(itr)
            self.start_iter=itr
            n_ep = self.args['n_epoch_warmup'] if self.start_iter==self.args['start_iter'] else self.args['n_epoch']
            for ep in range(self.start_epoch+1, n_ep+1):
                progress = 0
                i_loss_list,s_loss_list,loss_list,psnr_list,ssim_list = [],[],[],[],[]
                
                for _, data in enumerate(self.train_data):
                    x, x_label, z = data
                    progress += 1
                    
                    # prepare data
                    x = x.to(device)
                    x_label = x_label.to(device)
                    z = z.to(device)

                    x_list, z_list = self.model(x,z)
                    x_out = x_list[-1].clip(0,1) * self.mask
                    z_out = z_list[-1].clip(0)
                    z_label = projection.apply(x_label, self.model.module.options_sparse_view)

                    # compute loss: sum of squared loss and ssim loss
                    i_loss = torch.sum(torch.pow(x_out-x_label,2))
                    s_loss = torch.sum(torch.pow(z_out-z_label,2))
                    ssim_loss = 1-ssim(x_out,x_label,data_range=1)
                    t_loss = i_loss + s_loss + 0.1 * ssim_loss
                    
                    self.optim.zero_grad()
                    t_loss.backward()
                    self.optim.step()
                    if self.is_scheduler:
                        self.sched.step()
                        self.is_scheduler = False
                    
                    p = 0
                    for i in range(self.args['batch_size']):
                        output = x_out[i].detach().cpu().numpy().squeeze()
                        label = x_label[i].detach().cpu().numpy().squeeze()
                        p += psnr(output.astype(label.dtype), label)
                    
                    # record loss
                    i_loss_list.append(i_loss.item())
                    s_loss_list.append(s_loss.item())
                    loss_list.append(t_loss.item())
                    psnr_list.append(p/self.args['batch_size'])
                    ssim_list.append(1-ssim_loss.item())
                    
                    if progress % 20 == 0:
                        output_data = \
                        "[iter {:2d}][epoch {:2d}/{:2d}] s-loss: {:2.4f}\t i-loss: {:2.4f}\t t-loss: {:2.4f}\t psnr: {:2.4f}\t ssim: {:2.4f}\t progress: {:2.2f}%\n".format(
                            itr, ep, n_ep, 
                            s_loss_list[-1], i_loss_list[-1], loss_list[-1], psnr_list[-1], ssim_list[-1], 
                            self.args['batch_size'] * progress * 100 / len(self.train_data))
                        print(output_data)
                
                # job after each epoch
                avg_sloss = np.mean(s_loss_list)
                avg_iloss = np.mean(i_loss_list)
                avg_tloss = np.mean(loss_list)
                avg_psnr = np.mean(psnr_list)
                avg_ssim = np.mean(ssim_list)
                
                # exponential weighted average loss
                losses = [avg_sloss, avg_iloss, avg_tloss]
                for loss_idx, loss_key in enumerate(self.avg_loss):
                    if self.steps >= 1:
                        if losses[loss_idx] > 2 * (self.avg_loss[loss_key] / (1-0.9**(self.steps+1))):
                            # detect instability in backpropagation and reduce learning rate
                            self.is_continue = True
                            self.is_scheduler = True
                            print('reduce lr and continue...')
                            return
                    self.avg_loss[loss_key] = 0.9 * self.avg_loss[loss_key] + 0.1 * losses[loss_idx].item()
                self.steps += 1
                is_best = avg_psnr > self.best_psnr
                self.best_psnr = max(self.best_psnr, avg_psnr)
                checkpoint = {
                                'iter': itr,
                                'epoch': ep, 
                                'best_psnr': self.best_psnr,
                                'avg_loss': self.avg_loss,
                                'steps': self.steps,
                                'state_dict': self.model.state_dict(),
                                'optimizer': self.optim.state_dict(),
                                'scheduler': self.sched.state_dict()
                             }
                self.save_checkpoint(checkpoint, is_best)
                epoch_data="[iter {:2d}][epoch {:2d}/{:2d}] (average) s-loss: {:2.4f}\t i-loss: {:2.4f}\t t-loss: {:2.4f}\t psnr: {:2.4f}\t ssim: {:2.4f}\n\n".format(
                    itr, ep, n_ep, 
                    avg_sloss, avg_iloss, avg_tloss, avg_psnr, avg_ssim)
                print(epoch_data)
                log_file = open(self.save_dir / 'log.txt', 'a')
                log_file.write(epoch_data)
                log_file.close()
                
            self.start_epoch = 0
        self.done=True

def parse_opt():
    parser = argparse.ArgumentParser(description='LAMA')
    parser.add_argument('--dataset', type=str, default='mayo', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--img_dir', type=str, default='Img_CNN_64views', help='image directory')
    parser.add_argument('--prj_dir', type=str, default='Sin_CNN_64views', help='projection directory')
    parser.add_argument('--cfg', type=str, default='baseline', help='model.yaml type')
    parser.add_argument('--n_views', type=int, default=64, help='number of views')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    
    parser.add_argument('--n_iter', type=int, default=15, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=3, help='start iteration')
    parser.add_argument('--n_epoch_warmup', type=int, default=300, help='number of epoch for warmup')
    parser.add_argument('--n_epoch', type=int, default=200, help='number of epochs for training after warmup')
    parser.add_argument('--is_continue', type=bool, default=False, help='continue training')
    parser.add_argument('--is_scheduler', type=bool, default=False, help='use scheduler')
    parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate')
    parser.add_argument('--lr_I', type=float, default=1e-4, help='learning rate for ImgNet')
    parser.add_argument('--lr_S', type=float, default=6e-5, help='learning rate for SNet')
    parser.add_argument('--lr_p', type=float, default=1e-4, help='hyper parameters learning rate')
    
    
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt

def run(**kwargs):
    init_seeds()
    P = pipeline(**kwargs)
    while not P.done:
        P.train()

def main():
    opt = parse_opt()
    run(**vars(opt))
    
if __name__ == '__main__':
    main()