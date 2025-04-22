import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import numbers
import math

from utils import compute_normal
#from Functions import generate_grid

def generate_grid(imgshape):
    x = np.arange(imgshape[0])/(imgshape[0]-1)*2-1
    y = np.arange(imgshape[1])/(imgshape[1]-1)*2-1
    z = np.arange(imgshape[2])/(imgshape[2]-1)*2-1
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid,0,2)
    grid = np.swapaxes(grid,1,2)
    return grid.astype(np.float32)


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, padding='same', dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

        self.padding = padding

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=self.padding)


"""
Deformation Block
nc: number of channels
K: kernal size for local conv operation
n_scale: num of layers of image pyramid
"""
class DiffDeformBlock(nn.Module):
    def __init__(self, n_in_channels, n_start_filters):
        super(DiffDeformBlock, self).__init__()
        self.n_in_channels = n_in_channels
        self.n_start_filters = n_start_filters
        
        bias_opt = True
       
        # pool of square window of size=3, stride=2
        self.c0 = torch.nn.AvgPool3d(3, stride=2)
        self.c1 = self.encoder(self.n_in_channels, self.n_start_filters, stride=2, bias=bias_opt)
        self.c2 = self.encoder(self.n_start_filters, self.n_start_filters * 2, stride=2, bias=bias_opt)
        self.c3 = self.encoder(self.n_start_filters * 2, self.n_start_filters * 4, stride=2, bias=bias_opt)
        self.c4 = self.encoder(self.n_start_filters * 4, self.n_start_filters * 8, stride=2, bias=bias_opt)

        self.d1 = self.decoder(self.n_start_filters * 8, self.n_start_filters * 4, stride=2, bias=bias_opt)
        self.d2 = self.decoder(self.n_start_filters * 8, self.n_start_filters * 2, stride=2, bias=bias_opt)
        self.d3 = self.decoder(self.n_start_filters * 4, self.n_start_filters, stride=2, bias=bias_opt)
        self.d4 = self.decoder(self.n_start_filters * 2, self.n_start_filters, stride=2, bias=bias_opt)

        self.c5_m = self.outputs(self.n_start_filters, 3, kernel_size=3, stride=1, bias=bias_opt)
        self.c5_g = self.outputs(self.n_start_filters, 3, kernel_size=3, stride=1, bias=bias_opt)
        self.c5_w = self.outputs(self.n_start_filters, 3, kernel_size=3, stride=1, bias=bias_opt)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, volume):
        #x_in = self.c0(volume)
        #e1 = self.c1(x_in)
        
        e1 = self.c1(volume)
        #print(e1.size())
        e2 = self.c2(e1)
        #print(e2.size())
        e3 = self.c3(e2)
        #print(e3.size())
        e4 = self.c4(e3)
        #print(e4.size())
        d1 = self.d1(e4)
        #print(d1.size())
        d2 = torch.cat((d1, e3), 1)
        d2 = self.d2(d2)
        #print(d2.size())
        d3 = torch.cat((d2, e2), 1)
        d3 = self.d3(d3)
        #print(d3.size())
        d4 = torch.cat((d3, e1), 1)
        d4 = self.d4(d4)
        #print(d4.size())
        
        vf_m = self.c5_m(d4)
        vf_g = self.c5_g(d4)
        vf_w = self.c5_w(d4)
        return vf_m, vf_g, vf_w

        
    def initialize(self, L, W, H, device):
        """initialize necessary constants"""
        # will add code to generate grid for sampling in diffmorphic
        self.L = L
        self.W = W
        self.H = H
        
        # for storage of sample grid
        imgshape = [L, W, H]
        self.sample_grid = torch.from_numpy(generate_grid(imgshape)).to(device)


class SpatialTransform(nn.Module):
    def __init__(self):
        super(SpatialTransform, self).__init__()
    def forward(self, x,flow,sample_grid):
        sample_grid = sample_grid+flow
        #size_tensor = sample_grid.size()
        #sample_grid[0, :, :, :, 0] = (sample_grid[0, :, :, :, 0] - ((size_tensor[3] - 1) / 2)) / (
        #            size_tensor[3] - 1) * 2
        #sample_grid[0, :, :, :, 1] = (sample_grid[0, :, :, :, 1] - ((size_tensor[2] - 1) / 2)) / (
        #            size_tensor[2] - 1) * 2
        #sample_grid[0, :, :, :, 2] = (sample_grid[0, :, :, :, 2] - ((size_tensor[1] - 1) / 2)) / (
        #            size_tensor[1] - 1) * 2
        # make sure sample grid with values in [-1, 1]
        sample_grid = sample_grid.clamp(min=-1, max=1)
        #flow = F.grid_sample(x, sample_grid, mode = 'bilinear', align_corners=True)
        flow = F.grid_sample(x, sample_grid, mode = 'bilinear')

        return flow


class DiffeomorphicTransform(nn.Module):
    def __init__(self, time_step=7):
        super(DiffeomorphicTransform, self).__init__()
        self.time_step = time_step

    def forward(self, velocity, sample_grid, range_flow):
        # sample_grid can be generated with torch.nn.functional.affine_grid(theta, size, align_corners=None)
        flow = velocity/(2.0**self.time_step)     
        for _ in range(self.time_step):
            print('def', sample_grid.shape, (flow.permute(0,2,3,4,1) * range_flow).shape)
            grid = sample_grid + (flow.permute(0,2,3,4,1) * range_flow)
            print('grid', grid.shape)
            #grid[0, :, :, :, 0] = (grid[0, :, :, :, 0] - ((size_tensor[3]-1) / 2)) / (size_tensor[3]-1) * 2
            #grid[0, :, :, :, 1] = (grid[0, :, :, :, 1] - ((size_tensor[2]-1) / 2)) / (size_tensor[2]-1) * 2
            #grid[0, :, :, :, 2] = (grid[0, :, :, :, 2] - ((size_tensor[1]-1) / 2)) / (size_tensor[1]-1) * 2
            # make sure grid with values in [-1, 1]
            grid = grid.clamp(min=-1, max=1)
            #flow = flow + F.grid_sample(flow, grid, mode='bilinear', align_corners=True)
            flow = flow + F.grid_sample(flow, grid, mode='bilinear')
        return flow


def smoothloss(y_pred):
    dy = torch.abs(y_pred[:,:,1:, :, :] - y_pred[:,:, :-1, :, :])
    dx = torch.abs(y_pred[:,:,:, 1:, :] - y_pred[:,:, :, :-1, :])
    dz = torch.abs(y_pred[:,:,:, :, 1:] - y_pred[:,:, :, :, :-1])
    return (torch.mean(dx * dx)+torch.mean(dy*dy)+torch.mean(dz*dz))/3.0



def JacboianDet(y_pred, sample_grid):
    J = y_pred + sample_grid
    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:,:,:,:,0] * (dy[:,:,:,:,1] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,1])
    Jdet1 = dx[:,:,:,:,1] * (dy[:,:,:,:,0] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,0])
    Jdet2 = dx[:,:,:,:,2] * (dy[:,:,:,:,0] * dz[:,:,:,:,1] - dy[:,:,:,:,1] * dz[:,:,:,:,0])

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet


def neg_Jdet_loss(y_pred, sample_grid):
    neg_Jdet = -1.0 * JacboianDet(y_pred, sample_grid)
    selected_neg_Jdet = F.relu(neg_Jdet)

    return torch.mean(selected_neg_Jdet)

"""
SurfNN with deformation + 1 Laplacian smoothing layer
"""

class SurfNN(nn.Module):
    def __init__(self, n_in_channels=1, n_start_filters=16):
        super(SurfNN, self).__init__()
        self.block = DiffDeformBlock(n_in_channels, n_start_filters)

        self.smooth = LaplacianSmooth(3, 3, aggr='mean')

        #self.smooth = GaussianSmoothing(channels=3, kernel_size=5, sigma=1, dim=3)

        self.time_step = 10 

        self.time_step_mid = 1

    def forward(self, v, f, volume, n_smooth=1, lambd=1.0):

        x, x_g, x_w = self.block(volume) 
        
        # compute the mid layer 
        flow1 = x*(1.0/self.time_step_mid)  
        trj_mid = []   
        new_v = v.clone()
        for _ in range(self.time_step_mid):  
            samples = new_v[:,:,[2,1,0]].unsqueeze(2).unsqueeze(2)
            delta_s = F.grid_sample(flow1, samples, mode='bilinear', align_corners=True) #* range_flow
            delta_s = delta_s.squeeze(4).squeeze(3).permute(0,2,1)
            trj_mid.append(delta_s)
            new_v = new_v + delta_s
            new_v = new_v.clamp(min=-1, max=1)
        
        #edge_list = torch.cat([f[0,:,[0,1]], f[0,:,[1,2]], f[0,:,[2,0]]], dim=0).transpose(1,0)
        #for i in range(n_smooth):
        #    new_v = self.smooth(new_v, edge_list, lambd=lambd)
        #v_mid = new_v.clamp(min=-1, max=1) 

        v_mid = new_v

        # WM 
        flow2 = x_w*(1.0/self.time_step) 
        trj_wm = [] 
        v_wm = v_mid 
        for _ in range(self.time_step):  
            samples = v_wm[:,:,[2,1,0]].unsqueeze(2).unsqueeze(2)
            delta_s = F.grid_sample(flow2, samples, mode='bilinear', align_corners=True) #* range_flow
            delta_s = delta_s.squeeze(4).squeeze(3).permute(0,2,1)
            trj_wm.append(delta_s)
            v_wm = v_wm + delta_s
            v_wm = v_wm.clamp(min=-1, max=1)


        # GM 
        flow3 = x_g*(1.0/self.time_step) 
        trj_gm = [] 
        v_gm = v_mid 
        for _ in range(self.time_step):  
            samples = v_gm[:,:,[2,1,0]].unsqueeze(2).unsqueeze(2)
            delta_s = F.grid_sample(flow3, samples, mode='bilinear', align_corners=True) #* range_flow
            delta_s = delta_s.squeeze(4).squeeze(3).permute(0,2,1)
            trj_gm.append(delta_s)
            v_gm = v_gm + delta_s
            v_gm = v_gm.clamp(min=-1, max=1)


        # cycle 1: from GM_pred to WM 
        trj_gm2mid = []
        v_gm2mid = v_gm 
        for _ in range(self.time_step):  
            samples = v_gm2mid[:,:,[2,1,0]].unsqueeze(2).unsqueeze(2)
            delta_s = F.grid_sample(flow2, samples, mode='bilinear', align_corners=True) #* range_flow
            delta_s = delta_s.squeeze(4).squeeze(3).permute(0,2,1)
            trj_gm2mid.append(delta_s)
            v_gm2mid = v_gm2mid + delta_s
            v_gm2mid = v_gm2mid.clamp(min=-1, max=1)

        # cycle 2: from WM_pred to GM 
        trj_wm2mid = []
        v_wm2mid = v_wm 
        for _ in range(self.time_step):  
            samples = v_wm2mid[:,:,[2,1,0]].unsqueeze(2).unsqueeze(2)
            delta_s = F.grid_sample(flow3, samples, mode='bilinear', align_corners=True) #* range_flow
            delta_s = delta_s.squeeze(4).squeeze(3).permute(0,2,1)
            trj_wm2mid.append(delta_s)
            v_wm2mid = v_wm2mid + delta_s
            v_wm2mid = v_wm2mid.clamp(min=-1, max=1)


        return v_gm, v_wm, v_mid, v_gm2mid, v_wm2mid, \
               trj_mid, trj_gm, trj_wm, trj_gm2mid, trj_wm2mid #, l_sm_all, l_sm_all2 
        

    def initialize(self, L, W, H, device=None):
        self.block.initialize(L, W, H, device)


"""
LaplacianSmooth() is a differentiable Laplacian smoothing layer.
The code is implemented based on the torch_geometric.nn.conv.GraphConv.
For original GraphConv implementation, please see
https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/conv/graph_conv.py


x: the coordinates of the vertices, (|V|, 3).
edge_index: the list of edges, (2, |E|), e.g. [[0,1],[1,3],...]. 
lambd: weight for Laplacian smoothing, between [0,1].
out: the smoothed vertices, (|V|, 3).
"""

from typing import Union, Tuple
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing


class LaplacianSmooth(MessagePassing):

    def __init__(self, in_channels: Union[int, Tuple[int,
                                                     int]], out_channels: int,
                 aggr: str = 'add', bias: bool = True, **kwargs):
        super(LaplacianSmooth, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, size: Size = None, lambd=0.5) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size)
        out = lambd * out 
        x_r = x[1]
        if x_r is not None:
            out += (1-lambd) * x_r

        return out


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

