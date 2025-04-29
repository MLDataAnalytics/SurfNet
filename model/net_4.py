import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os 
import skimage.io as io


# segmentation U-Net
class Unet(nn.Module):
    def __init__(self, c_in=1, c_out=2):
        super(Unet, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=c_in, out_channels=16, kernel_size=3,
                               stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3,
                               stride=2, padding=1)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3,
                               stride=2, padding=1)
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3,
                               stride=2, padding=1)
        self.conv5 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3,
                               stride=2, padding=1)

        self.deconv4 = nn.Conv3d(in_channels=256, out_channels=64, kernel_size=3,
                               stride=1, padding=1)
        self.deconv3 = nn.Conv3d(in_channels=128, out_channels=32, kernel_size=3,
                               stride=1, padding=1)
        self.deconv2 = nn.Conv3d(in_channels=64, out_channels=16, kernel_size=3,
                               stride=1, padding=1)
        self.deconv1 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=3,
                               stride=1, padding=1)
        
        self.lastconv1 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3,
                                   stride=1, padding=1)
        self.lastconv2 = nn.Conv3d(in_channels=16, out_channels=c_out, kernel_size=3,
                                   stride=1, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear')
        
    def forward(self, x):

        x1 = F.leaky_relu(self.conv1(x), 0.2)
        x2 = F.leaky_relu(self.conv2(x1), 0.2)
        x3 = F.leaky_relu(self.conv3(x2), 0.2)
        x4 = F.leaky_relu(self.conv4(x3), 0.2)
        x  = F.leaky_relu(self.conv5(x4), 0.2)
        x  = self.up(x)
        
        x = torch.cat([x, x4], dim=1)
        x = F.leaky_relu(self.deconv4(x), 0.2)
        x = self.up(x)
        
        x = torch.cat([x, x3], dim=1)
        x = F.leaky_relu(self.deconv3(x), 0.2)
        x = self.up(x)
        
        x = torch.cat([x, x2], dim=1)
        x = F.leaky_relu(self.deconv2(x), 0.2)
        x = self.up(x)
        
        x = torch.cat([x, x1], dim=1)
        x = F.leaky_relu(self.deconv1(x), 0.2)

        x = F.leaky_relu(self.lastconv1(x), 0.2)
        x = self.lastconv2(x)

        return x


class CortexODE(nn.Module):
    """
    The deformation network of CortexODE model.

    dim_in: input dimension
    dim_h (C): hidden dimension
    kernel_size (K): size of convolutional kernels
    n_scale (Q): number of scales of the multi-scale input
    """
    
    def __init__(self, dim_in=3,
                       dim_h=128,
                       kernel_size=5,
                       n_scale=3):
        
        super(CortexODE, self).__init__()


        C = dim_h        # hidden dimension
        K = kernel_size  # kernel size
        Q = n_scale      # number of scales
        
        self.C = C
        self.K = K
        self.Q = Q

        # FC layers
        self.fc1 = nn.Linear(dim_in, C)
        self.fc2 = nn.Linear(C*2, C*4)
        self.fc3 = nn.Linear(C*4, C*2)
        self.fc4 = nn.Linear(C*2, dim_in)
        
        # local convolution
        self.localconv = nn.Conv3d(Q, C, (K, K, K))
        self.localfc = nn.Linear(C, C)
        
        # for cube sampling
        self.initialized = False
        grid = np.linspace(-K//2, K//2, K)
        grid_3d = np.stack(np.meshgrid(grid, grid, grid), axis=0).transpose(2,1,3,0)
        # print('grid_3d shape', grid_3d.shape)   # (5, 5, 5, 3)
        self.x_shift = torch.Tensor(grid_3d.copy()).view(-1,3)
        # print('x_shift shape', self.x_shift.shape)  # torch.Size([125, 3])
        self.cubes = torch.zeros([1, self.Q, self.K, self.K, self.K])

    def _initialize(self, V):
        # initialize coordinates shift and cubes
        self.x_shift = self.x_shift.to(V.device)
        self.cubes = self.cubes.to(V.device)
        self.initialized == True
        
    def set_data(self, x, V):
        # x: coordinats
        # V: input brain MRI volume
        if not self.initialized:
            self._initialize(V)
            
        # set the shape of the volume
        D1,D2,D3 = V[0,0].shape
        D = max([D1,D2,D3])
        # rescale for grid sampling
        self.rescale = torch.Tensor([D3/D, D2/D, D1/D]).to(V.device)
        self.D = D

        self.m = x.shape[1]    # number of points
        self.neighbors = self.cubes.repeat(self.m,1,1,1,1)    # repeat m cubes
        
        # set multi-scale volume
        self.Vq = [V]
        for q in range(1, self.Q):
            # iteratively downsampling
            self.Vq.append(F.avg_pool3d(self.Vq[-1], 2))

    def forward(self, t, x):
        
        # local feature
        #print('{} sampling: {}'.format(t, x[0,80000,:])) 
        # print('sampling: {}'.format(x[0,80000,:])) 
        z_local = self.cube_sampling(x)
        print(x.shape, z_local.shape)
        z_local = self.localconv(z_local)
        z_local = z_local.view(-1, self.m, self.C)
        z_local = self.localfc(z_local)
        
        # point feature
        z_point = F.leaky_relu(self.fc1(x), 0.2)
        
        # feature fusion
        z = torch.cat([z_point, z_local], 2)
        z = F.leaky_relu(self.fc2(z), 0.2)
        z = F.leaky_relu(self.fc3(z), 0.2)
        dx = self.fc4(z)
        # print(t,dx[0,80000,:])
        
        return dx
    
    def cube_sampling(self, x):
        # x: coordinates
        with torch.no_grad():
            for q in range(self.Q):
                # make sure the cubes have the same size
                delta = self.x_shift / self.D * 2 * (2**q)
                xq = x.unsqueeze(-2) + delta
                xq = xq.contiguous().view(1,-1,3).unsqueeze(-2).unsqueeze(-2)
                xq = xq / self.rescale  # rescale the coordinates 
                # sample the q-th cube
                vq = F.grid_sample(self.Vq[q], xq, mode='bilinear', padding_mode='border', align_corners=True)
                # update the cubes
                self.neighbors[:,q] = vq[0,0].view(self.m, self.K, self.K, self.K)
        
        return self.neighbors.clone()


def norm(dim):
    return nn.GroupNorm(min(16,dim),dim)


class BackboneNet2(nn.Module):
    def __init__(self, n_in_channels, n_start_filters):
        super(BackboneNet2, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=n_in_channels, out_channels=n_in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=n_in_channels, out_channels=n_in_channels, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(in_channels=n_in_channels, out_channels=n_in_channels, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(in_channels=n_in_channels, out_channels=n_in_channels, kernel_size=3, stride=2, padding=1)

        
    def forward(self, x):

        # x1 = F.avg_pool3d(x,2)
        # x2 = F.avg_pool3d(x1,2)
        # x3 = F.avg_pool3d(x2,2)
        # x4 = F.avg_pool3d(x3,2)

        x1 = F.leaky_relu(self.conv1(x), 0.2)
        x2 = F.leaky_relu(self.conv2(x1), 0.2)
        x3 = F.leaky_relu(self.conv3(x2), 0.2)
        x4 = F.leaky_relu(self.conv4(x3), 0.2)

        return [x, x1, x2, x3, x4] 


class BackboneNet(nn.Module):
    def __init__(self, n_in_channels, n_start_filters):
        super(BackboneNet, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=n_in_channels, out_channels=n_start_filters*2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=n_start_filters*2, out_channels=n_start_filters*2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(in_channels=n_start_filters*2, out_channels=n_start_filters*2, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(in_channels=n_start_filters*2, out_channels=n_start_filters*2, kernel_size=3, stride=2, padding=1)

        '''
        self.norm1 = norm(n_start_filters)
        self.norm2 = norm(n_start_filters*2)
        self.norm3 = norm(n_start_filters*4)
        self.norm4 = norm(n_start_filters*8)
        '''
        self.norm1 = nn.BatchNorm3d(n_start_filters*2)
        self.norm2 = nn.BatchNorm3d(n_start_filters*2)
        self.norm3 = nn.BatchNorm3d(n_start_filters*2)
        self.norm4 = nn.BatchNorm3d(n_start_filters*2)
        
    def forward(self, x):
        
        x1 = F.leaky_relu(self.norm1(self.conv1(x)), 0.2)
        x2 = F.leaky_relu(self.norm2(self.conv2(x1)), 0.2)
        x3 = F.leaky_relu(self.norm3(self.conv3(x2)), 0.2)
        x4 = F.leaky_relu(self.norm4(self.conv4(x3)), 0.2)
        '''
        x1 = F.leaky_relu(self.conv1(x), 0.2)
        x2 = F.leaky_relu(self.conv2(x1), 0.2)
        x3 = F.leaky_relu(self.conv3(x2), 0.2)
        x4 = F.leaky_relu(self.conv4(x3), 0.2)
        '''
        return [x1, x2, x3, x4] 


class SurfNNODE_0(nn.Module):
    def __init__(self, n_in_channels=3, n_out_channels=16, n_start_filters=16):
        super(SurfNNODE_0, self).__init__()
        self.block = BackboneNet(n_in_channels, n_start_filters)
        
    def forward(self, volume):
        return self.block(volume) 


class SurfNNODE(nn.Module):
    def __init__(self, n_in_channels=1, n_start_filters=16):
        super(SurfNNODE, self).__init__()
        self.n_start_filters = n_start_filters

        #self.block = BackboneNet(n_in_channels, n_start_filters) 
        #self.block = SurfNNODE_0(n_in_channels, n_start_filters) 

        # self.smooth = LaplacianSmooth(3, 3, aggr='mean')

        self.K = 3 
        self.Q = 4 

        self.localconv = nn.Conv3d(n_start_filters*2, n_start_filters*4, (self.K, self.K, self.K))
        self.localfc = nn.Linear(n_start_filters*4, n_start_filters*8)

        self.fc1 = nn.Linear(3, n_start_filters*8)

        self.fc2 = nn.Linear(n_start_filters*16, n_start_filters*32)
        self.fc3 = nn.Linear(n_start_filters*32, n_start_filters*16)
        self.fc4 = nn.Linear(n_start_filters*16, 3)



        #self.initialized = False
        grid = np.linspace(-self.K//2, self.K//2, self.K)
        grid_3d = np.stack(np.meshgrid(grid, grid, grid), axis=0).transpose(2,1,3,0)
        self.x_shift = torch.Tensor(grid_3d.copy()).view(-1,3)
        # self.cubes = torch.zeros([1, self.Q, self.K, self.K, self.K])
        # self.cubes = torch.zeros([1, self.Q+1, self.K, self.K, self.K])
        self.mscubes = torch.zeros([1, n_start_filters*2, self.K, self.K, self.K])
  

    #def _initialize(self, V):
    #    # initialize coordinates shift and cubes
    #    self.x_shift = self.x_shift.to(V.device)
    #    # self.cubes = self.cubes.to(V.device)
        
    #    #self.mscubes = self.mscubes.to(V.device)
    #    self.initialized == True

    def set_data(self, v, f, volume, n_smooth=1, lambd=1.0):
        # if not self.initialized:
        #     self._initialize(volume[0])

        self.x_shift = self.x_shift.to(volume[0].device)
        #self.x_shift = self.x_shift.to(volume.device)

        self.v = v 
        self.f = f
        self.volume = volume 
        self.n_smooth = n_smooth
        self.lambd = lambd

        # set the shape of the volume
        D1,D2,D3 = self.volume[0][0,0].shape
        #D1,D2,D3 = self.volume[0,0].shape
        D = max([D1,D2,D3])
        # rescale for grid sampling
        self.rescale = torch.Tensor([D3/D, D2/D, D1/D]).to(self.volume[0].device)
        #self.rescale = torch.Tensor([D3/D, D2/D, D1/D]).to(self.volume.device)

        self.D = D

        self.m = self.v.shape[1]    # number of points
        # self.neighbors = self.mscubes.repeat(self.m,1,1,1,1)    # repeat m cubes
        # self.neighbors = self.mscubes.repeat(self.m,self.Q,1,1,1)    # repeat m cubes 
        
        #self.neighbors = self.mscubes.repeat(self.m,1,1,1,1)    # repeat m cubes 
        

    def forward(self, t, x):

        #volume_feat = self.block(self.volume) 
        # print(volume_feat.shape) # [1, 2, 192, 224, 192]

        B = []
        with torch.no_grad():
            for q in range(self.Q): #len(Vf)
                # make sure the cubes have the same size
                xq = x.unsqueeze(-2) + self.x_shift / self.D * 2 * (2**q) #A[q]
                xq = xq.contiguous().view(1,-1,3).unsqueeze(-2).unsqueeze(-2)
                xq = xq / self.rescale  # rescale the coordinates
                # sample the q-th cube
                vq = F.grid_sample(self.volume[q], xq, mode='bilinear', padding_mode='border', align_corners=True)
                # print(q, vq.shape) # torch.Size([1, Nc, 4135023, 1, 1])
                # update the cubes

                vq = vq[0].view(self.n_start_filters*2, self.m, self.K, self.K, self.K).permute(1, 0, 2, 3, 4).contiguous().view(self.m, self.n_start_filters*2, self.K, self.K, self.K)
                # print(q, vq.shape) # torch.Size([153149, Nc, 3, 3, 3])
                B.append(vq) 

        z_local = B[0] + B[1] + B[2] + B[3]

        # z_local = self.cube_sampling(x,self.volume)

        z_local = self.localconv(z_local)
        z_local = z_local.view(-1, self.m, self.n_start_filters*4)
        z_local = self.localfc(z_local)

        z_point = F.leaky_relu(self.fc1(x), 0.2)

        z = torch.cat([z_point, z_local], 2)
        z = F.leaky_relu(self.fc2(z), 0.2)
        z = F.leaky_relu(self.fc3(z), 0.2)
        dx = self.fc4(z)

        return dx

    '''
    def cube_sampling(self, x, Vf):
        # x: coordinates
        # A=[0,0,1,2,3]
        B = []
        with torch.no_grad():
            for q in range(self.Q): #len(Vf)
                # make sure the cubes have the same size
                xq = x.unsqueeze(-2) + self.x_shift / self.D * 2 * (2**q) #A[q]
                xq = xq.contiguous().view(1,-1,3).unsqueeze(-2).unsqueeze(-2)
                xq = xq / self.rescale  # rescale the coordinates
                # sample the q-th cube
                vq = F.grid_sample(Vf[q], xq, mode='bilinear', padding_mode='border', align_corners=True)
                # print(q, vq.shape) # torch.Size([1, Nc, 4135023, 1, 1])
                # update the cubes

                vq = vq[0].view(self.n_start_filters*2, self.m, self.K, self.K, self.K).permute(1, 0, 2, 3, 4).contiguous().view(self.m, self.n_start_filters*2, self.K, self.K, self.K)
                # print(q, vq.shape) # torch.Size([153149, Nc, 3, 3, 3])
                B.append(vq) 

        # self.neighbors = torch.cat([B[0], B[1], B[2], B[3]], 1)
        self.neighbors = B[0] + B[1] + B[2] + B[3] # torch.cat([B[0], B[1], B[2], B[3]], 1)

        return self.neighbors#.clone()
    '''
