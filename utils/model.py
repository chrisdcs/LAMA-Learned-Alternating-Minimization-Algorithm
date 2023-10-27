import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Function

import numpy as np
import ctlib

# Helper Functions
def generate_mask(dImg, dDet):
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


# Init-Net Blocks
class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        
        # size: out channels  x in channels x filter size x filter size
        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv4_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))
        
    def forward(self, x_input):
        
        x = F.conv2d(x_input, self.conv1_forward, padding=1)
        x = F.relu(x)
        x = F.conv2d(x, self.conv2_forward, padding=1)
        x = F.relu(x)
        x = F.conv2d(x, self.conv3_forward, padding=1)
        x = F.relu(x)
        x = F.conv2d(x, self.conv4_forward, padding=1)
        # resnet structure
        x = F.relu(x + x_input)
        
        return x

class LongBlock(nn.Module):
    def __init__(self):
        super(LongBlock, self).__init__()
        
        # size: out channels  x in channels x filter size x filter size
        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 15)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 15)))
        self.conv3_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 15)))
        self.conv4_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 15)))

        
    def forward(self, x_input):
        
        x = F.conv2d(x_input, self.conv1_forward, padding=(1,7))
        x = F.relu(x)
        x = F.conv2d(x, self.conv2_forward, padding=(1,7))
        x = F.relu(x)
        x = F.conv2d(x, self.conv3_forward, padding=(1,7))
        x = F.relu(x)
        x = F.conv2d(x, self.conv4_forward, padding=(1,7))
        # resnet structure
        x = x + x_input
        
        return x

# Init-Net
class InitNet(torch.nn.Module):
    def __init__(self, LayerNo):
        super(InitNet, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        
        for i in range(LayerNo):
            onelayer.append(LongBlock())
        
        self.fcs = nn.ModuleList(onelayer)
        
    def forward(self, x):
        for i in range(self.LayerNo):
            # resnet architecture
            x = self.fcs[i](x)
        return x

# CT library functions
class projection(Function):
    @staticmethod
    def forward(self, input_data, options):
        out = ctlib.projection(input_data, options)
        self.save_for_backward(options, input_data)
        return out

    @staticmethod
    def backward(self, grad_output):
        options, input_data = self.saved_tensors
        grad_input = ctlib.projection_t(grad_output, options)
        return grad_input, None
    
class projection_t(Function):
    @staticmethod
    def forward(self, input_data, options):
        out = ctlib.projection_t(input_data, options)
        self.save_for_backward(options, input_data)
        return out

    @staticmethod
    def backward(self, grad_output):
        options, input_data = self.saved_tensors
        grad_input = ctlib.projection(grad_output, options)
        return grad_input, None

# smoothed activation functions
class sigma_activation(nn.Module):
    def __init__(self, ddelta):
        super(sigma_activation, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.ddelta = ddelta
        self.coeff = 1.0 / (4.0 * self.ddelta)
    
    def forward(self, x_i):
        x_i_relu = self.relu(x_i)
        x_square = torch.mul(x_i, x_i) * self.coeff
        return torch.where(torch.abs(x_i) > self.ddelta, x_i_relu, x_square + 0.5*x_i + 0.25 * self.ddelta)
    
class sigma_derivative(nn.Module):
    def __init__(self, ddelta):
        super(sigma_derivative, self).__init__()
        self.ddelta = ddelta
        self.coeff2 = 1.0 / (2.0 * self.ddelta)

    def forward(self, x_i):
        x_i_relu_deri = torch.where(x_i > 0, torch.ones_like(x_i), torch.zeros_like(x_i))
        return torch.where(torch.abs(x_i) > self.ddelta, x_i_relu_deri, self.coeff2 *x_i + 0.5)

# Exact Learnable Block
class Learnable_Block(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Learnable_Block, self).__init__()
        
        n_feats = kwargs['n_feats']
        n_convs = kwargs['n_convs']
        k_size = kwargs['k_size']
        padding = kwargs['padding']
        self.padding = padding
        
        self.soft_thr = nn.Parameter(torch.Tensor([0.002]),requires_grad=True)
        convs = [nn.Conv2d(1, n_feats, kernel_size=k_size, padding=padding)] + \
                [nn.Conv2d(n_feats, n_feats, kernel_size=k_size, padding=padding) for i in range(n_convs-1)]
        self.convs = nn.ModuleList(convs)
        
        self.act = sigma_activation(0.001)
        self.act_der = sigma_derivative(0.001)
    
    def gradient(self, forward_cache, gamma):
        soft_thr = torch.abs(self.soft_thr) * gamma
        g = forward_cache[-1]
        
        # compute gradient of smoothed regularization
        norm_g = torch.norm(g, dim = 1, keepdim=True)
        denominator = torch.where(norm_g > soft_thr, norm_g, soft_thr)
        out = torch.div(g, denominator)

        for i in range(len(forward_cache)-1, 0, -1):
            out = F.conv_transpose2d(out, self.convs[i].weight, padding=self.padding) * self.act_der(forward_cache[i-1])
        out = F.conv_transpose2d(out, self.convs[0].weight, padding=self.padding)
        return out
    
    def smoothed_reg(self, forward_cache, gamma):
        soft_thr = torch.abs(self.soft_thr) * gamma
        g = forward_cache[-1]
        
        norm_g = torch.norm(g, dim = 1, keepdim=True)
        reg = torch.where(norm_g > soft_thr, norm_g - torch.div(soft_thr,2), 
                          torch.square(norm_g)/(torch.mul(soft_thr,2)))
        reg = torch.flatten(reg, start_dim=1)
        reg = torch.sum(reg, -1, keepdim=True)
        
        return reg
        
    def forward(self, var):
        cache = []
        for i, conv in enumerate(self.convs):
                if i == 0:
                    var = conv(var)
                else:
                    var = conv(self.act(var))
                cache.append(var)
        return cache

# Exact LAMA
class LAMA(torch.nn.Module):
    def __init__(self, **kwargs):
        super(LAMA, self).__init__()
        self.n_u = 0
        self.n_v = 0
        self.phi_diff_list = []
        
        # number of iterations
        n_iter = kwargs['n_iter']
        # current iteration
        cur_iter = kwargs['start_iter']
        # number of sparse views
        n_views = kwargs['n_views']
        
        # I: image net, S: sinogram net, feats: number of features, convs: number of convolutions (layers)
        n_Ifeats = kwargs['n_Ifeats']
        n_Sfeats = kwargs['n_Sfeats']
        n_Iconvs = kwargs['n_Iconvs']
        n_Sconvs = kwargs['n_Sconvs']
        Iksize = kwargs['Iksize']
        Sksize = kwargs['Sksize']
        Ipadding = kwargs['Ipadding']
        Spadding = kwargs['Spadding']
        
        # Radon Transform paramters
        views = kwargs['views']
        dets = kwargs['dets']
        width = kwargs['width']
        height = kwargs['height']
        dImg = kwargs['dImg']
        dDet = kwargs['dDet']
        Ang0 = kwargs['Ang0']
        dAng = kwargs['dAng']
        s2r = kwargs['s2r']
        d2r = kwargs['d2r']
        binshift = kwargs['binshift']
        scan_type = kwargs['scan_type']
        
        # options = torch.tensor([512, 512, 256, 256, 0.006641, 0.0072, 0, 0.006134 * 2, 2.5, 2.5, 0, 0])
        options = torch.tensor([views, dets, width, height, dImg, dDet, Ang0, dAng, s2r, d2r, binshift, scan_type])
        self.options_sparse_view = nn.Parameter(options, requires_grad=False)
        
        
        # LAMA parameters
        alpha = kwargs['alpha']
        beta = kwargs['beta']
        mu = kwargs['mu']
        nu = kwargs['nu']
        lam = kwargs['lam']
        eta = kwargs['eta']
        
        self.eta = eta
        self.sigma = 10**4
        self.cur_iter = cur_iter
        self.lam = nn.Parameter(torch.tensor(lam), requires_grad=True)
        self.alphas = nn.Parameter(torch.tensor([alpha] * n_iter), requires_grad=True)
        self.betas = nn.Parameter(torch.tensor([beta] * n_iter), requires_grad=True)
        self.mus = nn.Parameter(torch.tensor([mu] * n_iter), requires_grad=True)
        self.nus = nn.Parameter(torch.tensor([nu] * n_iter), requires_grad=True)
        self.hyper_params = nn.ParameterList([self.lam, self.alphas, self.betas, self.mus, self.nus])
        
        self.ImgNet = Learnable_Block(
            n_feats=n_Ifeats, 
            n_convs=n_Iconvs,
            k_size=Iksize,
            padding=Ipadding)
        self.SNet = Learnable_Block(
            n_feats=n_Sfeats, 
            n_convs=n_Sconvs,
            k_size=Sksize,
            padding=Spadding)
        
        # Down-sample matrix D
        self.index = nn.Parameter(torch.tensor([i*(512//n_views) for i in range(n_views)],dtype=torch.int32),
                                  requires_grad=False)
        D = torch.zeros([n_views,512])
        n_p = 512//n_views
        for i in range(n_views):
            D[i,i*n_p] = 1
        DT = D.T
        self.DT = nn.Parameter(DT.reshape(1,1,512,n_views), requires_grad=False)
    
    def set_iteration(self, cur_iter):
        # set iteration number
        self.cur_iter = cur_iter
    
    def phi(self, x, s, z, gamma):
        # compute the loss function of minimization problem
        x = x.detach()
        z = z.detach()
        s = s.detach()
        lam = self.lam.detach()
        
        cache_x = self.ImgNet(x)
        cache_z = self.SNet(z)
        
        loss = 1/2 * torch.norm(
                (ctlib.projection(x, self.options_sparse_view) - z).flatten(start_dim=1),
                dim=-1, keepdim=True)**2 + \
                lam / 2 * torch.norm((torch.index_select(z,2,self.index)-s).flatten(start_dim=1),
                dim=-1, keepdim=True)**2 + \
                self.ImgNet.smoothed_reg(cache_x, gamma) + self.SNet.smoothed_reg(cache_z, gamma)
                
        return loss
    
    def norm_grad_phi(self, x, z, s, gamma):
        # compute the norm of the gradient of phi
        x = x.detach()
        z = z.detach()
        s = s.detach()
        lam = self.lam.detach()
        
        Ax = ctlib.projection(x, self.options_sparse_view)
        
        grad_phi_x = \
                        (projection_t.apply(
                                Ax-z, self.options_sparse_view)
                         + self.ImgNet.gradient(self.ImgNet(x), gamma)).flatten(start_dim=1)
        
        grad_phi_z = \
                        (z - Ax + lam * self.DT @ \
                            (torch.index_select(z,2,self.index)-s)
                        + self.SNet.gradient(self.SNet(z), gamma)).flatten(start_dim=1)
        
        grad_phi = torch.concat([grad_phi_x, grad_phi_z],dim=-1)
        output = torch.norm(grad_phi, dim=-1, keepdim=True)
        
        return output
    
    def norm_diff(self, x, y):
        
        x = x.detach()
        y = y.detach()
        diff = (x-y).reshape(-1,int(x.shape[-1]*x.shape[-1]))
        return torch.norm(diff, dim=-1, keepdim=True)
    
    def phase(self, x, s, z, phase, gamma):
        '''
            computation for each phase
        '''
        alpha = torch.abs(self.alphas[phase])
        beta = torch.abs(self.betas[phase])
        mu = torch.abs(self.mus[phase])
        nu = torch.abs(self.nus[phase])
        lam = self.lam
        eta = self.eta
        
        # update z
        Ax = projection.apply(x, self.options_sparse_view)
        residual_I = z - Ax 
        residual_S = torch.index_select(z,2,self.index)-s
        grad_fz = residual_I + lam * self.DT @ residual_S
        b = z - mu * grad_fz
        cache_z = self.SNet(b)
        uz = b - nu * self.SNet.gradient(cache_z, gamma)
        
        # update x
        residual_S_new = Ax - uz
        grad_fx = projection_t.apply(residual_S_new, self.options_sparse_view)
        c = x - alpha * grad_fx
        cache_x = self.ImgNet(c)
        ux = c - beta * self.ImgNet.gradient(cache_x, gamma)
        
        z_next = uz
        x_next = ux
        
        return x_next, z_next
    
    def forward(self, x, z, loss=False):
        gamma = nn.Parameter(torch.Tensor([1.0]), requires_grad=False).cuda()
        x_list = []
        z_list = []
        f = torch.index_select(z, 2, self.index)
        for phase in range(self.cur_iter):
            x, z = self.phase(x, f, z, phase, gamma)
            x_list.append(x)
            z_list.append(z)
            
            # update soft threshold
            norm_grad_phi_next = self.norm_grad_phi(x, z, f, gamma)
            sig_gam_eps = self.sigma * gamma * (torch.abs(self.SNet.soft_thr)+torch.abs(self.ImgNet.soft_thr)) / 2
            sig_gam_eps = sig_gam_eps.detach()
            gamma = torch.where(torch.mean(norm_grad_phi_next) < sig_gam_eps, torch.mul(gamma, 0.9), gamma)
            
        if loss:
            loss_val = self.phi(x, f, z, gamma)
            return loss_val
        
        return x_list, z_list