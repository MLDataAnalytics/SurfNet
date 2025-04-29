'''
code partially adapted from cortexODE 
https://github.com/m-qiang/CortexODE
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import nibabel as nib
import numpy as np
from tqdm import tqdm
from data.dataloader import load_seg_data, load_surf_data
from model.net_4 import CortexODE, Unet, SurfNNODE, BackboneNet 
from util.mesh import compute_dice

import trimesh
from trimesh import Trimesh
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_normal_consistency
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

import os 
import logging
from torchdiffeq import odeint_adjoint as odeint
from config_node import load_config

import skimage.io as io
import time 

def train_seg(config):
    """training WM segmentation"""
    
    # --------------------------
    # load configuration
    # --------------------------
    model_dir = config.model_dir   # the directory to save the checkpoints
    data_name = config.data_name
    device = config.device
    tag = config.tag
    n_epochs = config.n_epochs
    lr = config.lr

    # start training logging
    logging.basicConfig(filename=model_dir+'model_seg_'+data_name+'_'+tag+'.log',
                        level=logging.INFO, format='%(asctime)s %(message)s')
    
    # --------------------------
    # load dataset
    # --------------------------
    logging.info("load dataset ...")
    trainset = load_seg_data(config, data_usage='train')
    validset = load_seg_data(config, data_usage='valid')

    trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
    validloader = DataLoader(validset, batch_size=1, shuffle=False)
    
    # --------------------------
    # initialize model
    # --------------------------
    logging.info("initalize model ...")
    segnet = Unet(c_in=1, c_out=3).to(device)
    optimizer = optim.Adam(segnet.parameters(), lr=lr)
    # in case you need to load a checkpoint
    # segnet.load_state_dict(torch.load(model_dir+'model_seg_'+data_name+'_'+tag+'_XXepochs.pt'))
    # segnet.load_state_dict(torch.load('./ckpts/pretrained/adni/model_seg_adni_pretrained.pt'))

    # --------------------------
    # training model
    # --------------------------
    logging.info("start training ...")
    for epoch in tqdm(range(n_epochs+1)):
        avg_loss = []
        for idx, data in enumerate(trainloader):
            volume_in, seg_gt = data

            optimizer.zero_grad()
            volume_in = volume_in.to(device)
            seg_gt = seg_gt.long().to(device)

            seg_out = segnet(volume_in)
            loss = nn.CrossEntropyLoss()(seg_out, seg_gt)
            avg_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        logging.info("epoch:{}, loss:{}".format(epoch,np.mean(avg_loss)))

        if epoch % 10 == 0:
            logging.info('-------------validation--------------')
            with torch.no_grad():
                avg_error = []
                avg_dice = []
                for idx, data in enumerate(validloader):
                    volume_in, seg_gt = data
                    volume_in = volume_in.to(device)
                    seg_gt = seg_gt.long().to(device)
                    seg_out = segnet(volume_in)
                    avg_error.append(nn.CrossEntropyLoss()(seg_out, seg_gt).item())
                    
                    # compute dice score
                    seg_out = torch.argmax(seg_out, dim=1)
                    seg_out = F.one_hot(seg_out, num_classes=3).permute(0,4,1,2,3)[:,1:]
                    seg_gt = F.one_hot(seg_gt, num_classes=3).permute(0,4,1,2,3)[:,1:]
                    dice = compute_dice(seg_out, seg_gt, '3d')
                    avg_dice.append(dice)
                logging.info("epoch:{}, validation error:{}".format(epoch, np.mean(avg_error)))
                logging.info("Dice score:{}".format(np.mean(avg_dice)))
                logging.info('-------------------------------------')
        # save model checkpoints
        if epoch % 20 == 0:
            torch.save(segnet.state_dict(),
                       model_dir+'model_seg_'+data_name+'_'+tag+'_'+str(epoch)+'epochs.pt')
    # save final model
    torch.save(segnet.state_dict(),
               model_dir+'model_seg_'+data_name+'_'+tag+'.pt')


def train_surf(config):
    # --------------------------
    # load configuration
    # --------------------------
    model_dir = config.model_dir
    data_name = config.data_name
    surf_type = config.surf_type
    surf_hemi = config.surf_hemi
    device = config.device
    '''
    if torch.cuda.is_available():
        device_name = "cuda:3"
    else:
        device_name = "cpu"
    device = torch.device(device_name)
    #device = torch.device("cpu")
    print(device)
    '''

    tag = config.tag
    
    n_epochs = config.n_epochs
    n_samples = config.n_samples
    lr = config.lr
    
    C = config.dim_h     # hidden dimension of features
    K = config.kernel_size    # kernel / cube size
    Q = config.n_scale    # multi-scale input
    
    step_size = config.step_size    # step size of integration
    solver = config.solver    # ODE solver
    
    # create log file
    logging.basicConfig(filename=model_dir+'/model_'+surf_type+'_'+data_name+'_'+surf_hemi+'_'+tag+'.log',
                        level=logging.INFO, format='%(asctime)s %(message)s')
    
    # --------------------------
    # load dataset
    # --------------------------
    logging.info("load dataset ...")
    trainset = load_surf_data(config, 'train')
    validset = load_surf_data(config, 'valid')

    trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
    validloader = DataLoader(validset, batch_size=1, shuffle=False)

    # --------------------------
    # initialize models
    # --------------------------
    logging.info("initalize model ...")

    model_ft = BackboneNet(3,16).to(device)
    model_mid = SurfNNODE().to(device)
    model_wm = SurfNNODE().to(device)
    model_gm = SurfNNODE().to(device)

    model1_parameters = model_ft.parameters()
    model2_parameters = model_mid.parameters()
    model3_parameters = model_wm.parameters()
    model4_parameters = model_gm.parameters()

    mesh_smooth = LaplacianSmooth(3, 3, aggr='mean')

    all_parameters = list(model1_parameters) + list(model3_parameters) + list(model4_parameters) + list(model2_parameters) # 
    optimizer = optim.Adam(all_parameters, lr=lr) 

    T = torch.Tensor([0,1]).to(device)    # integration time interval for ODE 

    # --------------------------
    # training
    # --------------------------
    logging.info("start training ...")
    for epoch in tqdm(range(n_epochs+1)):
        avg_loss = []
        for idx, data in enumerate(trainloader):
            # volume_in,wmdm,gmdm, v_in, v_mid, v_gt, f_in, f_mid, f_gt = data
            volume_in,wmdm,gmdm,seg_vol, v_in, v_mid, v_gt, f_in, f_mid, f_gt, subid, sub_surf_hemi = data
            

            start_time = time.time() 

            # print(volume_in.shape)

            optimizer.zero_grad()

            volume_in = volume_in.to(device).float()
            wmdm = wmdm.to(device).float()
            gmdm = gmdm.to(device).float()
            #seg_vol = seg_vol.to(device).float()
            combined_in = torch.cat((volume_in,wmdm,gmdm),1)
            # print(volume_in.shape, volume_in.max(), volume_in.min())   

            v_in = v_in.to(device)
            f_in = f_in.to(device)
            v_mid = v_mid.to(device)
            f_mid = f_mid.to(device)
            v_gt = v_gt.to(device)
            f_gt = f_gt.to(device)
            
            
            # print(' ======= extract feature ======= ')
            ftr = model_ft(combined_in) 

            # print(' ======= set mid model ======= ')
            model_mid.set_data(v_mid, f_mid, ftr)
            v_out_mid = odeint(model_mid, v_mid, t=T, method=solver, options=dict(step_size=step_size))[-1]

            edge_list = torch.cat([f_mid[0,:,[0,1]], f_mid[0,:,[1,2]], f_mid[0,:,[2,0]]], dim=0).transpose(1,0)
            for i in range(1):
                v_out_mid = mesh_smooth(v_out_mid, edge_list, lambd=0.5)

            # print(' ======= set wm model ======= ')
            model_wm.set_data(v_out_mid, f_mid, ftr)
            v_out_wm = odeint(model_wm, v_out_mid, t=T, method=solver, options=dict(step_size=step_size))[-1]

            # edge_list = torch.cat([f_mid[0,:,[0,1]], f_mid[0,:,[1,2]], f_mid[0,:,[2,0]]], dim=0).transpose(1,0)
            for i in range(1):
                v_out_wm = mesh_smooth(v_out_wm, edge_list, lambd=0.5)
            
            # print(' ======= set gm model ======= ')
            model_gm.set_data(v_out_mid, f_mid, ftr) 
            v_out_gm = odeint(model_gm, v_out_mid, t=T, method=solver, options=dict(step_size=step_size))[-1]

            # edge_list = torch.cat([f_mid[0,:,[0,1]], f_mid[0,:,[1,2]], f_mid[0,:,[2,0]]], dim=0).transpose(1,0)
            for i in range(1):
                v_out_gm = mesh_smooth(v_out_gm, edge_list, lambd=0.5)

            # print(' ======= compute loss ======= ')
            wm_pred = Meshes(verts=v_out_wm, faces=f_in)
            wm_gt = Meshes(verts=v_in, faces=f_in)
            wm_v_pred = sample_points_from_meshes(wm_pred, n_samples)
            wm_v_gt = sample_points_from_meshes(wm_gt, n_samples)

            gm_pred = Meshes(verts=v_out_gm, faces=f_gt)
            gm_gt = Meshes(verts=v_gt, faces=f_gt)
            gm_v_pred = sample_points_from_meshes(gm_pred, n_samples)
            gm_v_gt = sample_points_from_meshes(gm_gt, n_samples)
                
            # scale by 1e3 since the coordinates are rescaled to [-1,1]
            l_wm = 1e3 * chamfer_distance(wm_v_pred, wm_v_gt)[0]    # chamfer loss
            l_gm = 1e3 * chamfer_distance(gm_v_pred, gm_v_gt)[0]    # chamfer loss

            # normal consistency 
            l_nc = (mesh_normal_consistency(gm_pred) + mesh_normal_consistency(wm_pred)) * 1e-3 

            # cycle loss 
            # print('cyc pred')
            v_out_gm2mid = odeint(model_wm, v_out_gm, t=T, method=solver, options=dict(step_size=step_size))[-1]
            v_out_wm2mid = odeint(model_gm, v_out_wm, t=T, method=solver, options=dict(step_size=step_size))[-1]
            l_cyc = (nn.MSELoss()(v_mid,v_out_gm2mid) + nn.MSELoss()(v_mid, v_out_wm2mid)) * 1e+3 

            loss = l_wm + l_gm + l_nc + l_cyc

            print('epoch {}-{} {}_{} loss term {:.6f} = l_gm {:.6f} + l_wm {:.6f} + l_nc {:.6f} + l_cyc {:.6f} '.format(\
                      epoch, idx, subid[0], sub_surf_hemi[0], loss.item(), l_gm.item(), l_wm.item(), l_nc.item(), l_cyc.item() ))
            
                
            avg_loss.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=1.0)
            optimizer.step()
            torch.cuda.empty_cache()

            end_time = time.time() 

            if idx % 50 == 0: 
                with open('train_err.txt', 'a') as f:
                    f.writelines('Epoch {}-{} {}_{} loss term {:.6f} = l_gm {:.6f} + l_wm {:.6f} + l_nc {:.6f} + l_cyc {:.6f} '.format(\
                        epoch+1, idx, subid[0], sub_surf_hemi[0], loss.item(), l_gm.item(), l_wm.item(), l_nc.item(), l_cyc.item() ) \
                        + ' running time: {} sec'.format(np.round(end_time - start_time, 4)) + '\n')
            

        logging.info('epoch:{}, loss:{}'.format(epoch, np.mean(avg_loss)))
        
        if epoch % 20 == 0:
            logging.info('-------------validation--------------')
            with torch.no_grad():
                valid_error, gm_val_err, wm_val_err  = [],[],[]
                for idx, data in enumerate(validloader):
                    # volume_in,wmdm,gmdm, v_in, v_mid, v_gt, f_in, f_mid, f_gt = data
                    volume_in,wmdm,gmdm,seg_vol, v_in, v_mid, v_gt, f_in, f_mid, f_gt, subid, sub_surf_hemi = data
            

                    optimizer.zero_grad()

                    volume_in = volume_in.to(device).float()
                    wmdm = wmdm.to(device).float()
                    gmdm = gmdm.to(device).float()
                    #seg_vol = seg_vol.to(device).float()
                    combined_in = torch.cat((volume_in,wmdm,gmdm),1)
                    v_in = v_in.to(device)
                    f_in = f_in.to(device)
                    v_mid = v_mid.to(device)
                    f_mid = f_mid.to(device)
                    v_gt = v_gt.to(device)
                    f_gt = f_gt.to(device)


                    ftr = model_ft(combined_in) 

                    model_mid.set_data(v_mid, f_mid, ftr)
                    v_out_mid = odeint(model_mid, v_mid, t=T, method=solver, options=dict(step_size=step_size))[-1]
                    edge_list = torch.cat([f_mid[0,:,[0,1]], f_mid[0,:,[1,2]], f_mid[0,:,[2,0]]], dim=0).transpose(1,0)
                    for i in range(1):
                        v_out_mid = mesh_smooth(v_out_mid, edge_list, lambd=0.5)

                    model_wm.set_data(v_out_mid, f_mid, ftr)
                    v_out_wm = odeint(model_wm, v_out_mid, t=T, method=solver, options=dict(step_size=step_size))[-1]
                    for i in range(1):
                        v_out_wm = mesh_smooth(v_out_wm, edge_list, lambd=0.5)            

                    model_gm.set_data(v_out_mid, f_mid, ftr) 
                    v_out_gm = odeint(model_gm, v_out_mid, t=T, method=solver, options=dict(step_size=step_size))[-1]
                    for i in range(1):
                        v_out_gm = mesh_smooth(v_out_gm, edge_list, lambd=0.5)

                    g_e = 1e3 * chamfer_distance(v_out_gm, v_gt)[0].item()
                    w_e = 1e3 * chamfer_distance(v_out_wm, v_in)[0].item()
                    gm_val_err.append(g_e)
                    wm_val_err.append(w_e)
                    valid_error.append(g_e+w_e) 

                logging.info('epoch:{}, validation error:{}'.format(epoch, np.mean(valid_error)))
                logging.info('-------------------------------------')

                with open('val_err.txt', 'a') as f:
                    f.writelines('Epoch_{}_{} gm:{} wm:{}'.format(epoch,np.mean(valid_error),np.mean(gm_val_err),np.mean(wm_val_err)) + '\n')

            print('Save surface mesh ... ')
            path_save_mesh = "./ckpts/experiment_1/mesh/"+ surf_hemi +"_mid_"+str(epoch)+"epochs.obj"
            mesh_pred = trimesh.Trimesh(v_out_mid[0].cpu().numpy(), f_in[0].cpu().numpy())
            mesh_pred.export(path_save_mesh)

            path_save_mesh = "./ckpts/experiment_1/mesh/"+ surf_hemi +"_wm_"+str(epoch)+"epochs.obj"
            mesh_pred = trimesh.Trimesh(v_out_wm[0].cpu().numpy(), f_in[0].cpu().numpy())
            mesh_pred.export(path_save_mesh)


            path_save_mesh = "./ckpts/experiment_1/mesh/"+ surf_hemi +"_gm_"+str(epoch)+"epochs.obj"
            mesh_pred = trimesh.Trimesh(v_out_gm[0].cpu().numpy(), f_gt[0].cpu().numpy())
            mesh_pred.export(path_save_mesh)

        # save model checkpoints 
        if epoch % 50 == 0:
            torch.save(model_ft.state_dict(), model_dir+'/model_'+ data_name+'_'+surf_hemi+'_'+tag+'_ftr_' +str(epoch)+'epochs.pt')
            torch.save(model_mid.state_dict(), model_dir+'/model_'+ data_name+'_'+surf_hemi+'_'+tag+'_mid_' +str(epoch)+'epochs.pt')
            torch.save(model_wm.state_dict(), model_dir+'/model_'+ data_name+'_'+surf_hemi+'_'+tag+'_wm_' +str(epoch)+'epochs.pt')
            torch.save(model_gm.state_dict(), model_dir+'/model_'+ data_name+'_'+surf_hemi+'_'+tag+'_gm_' +str(epoch)+'epochs.pt') 



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


if __name__ == '__main__':
    config = load_config()
    if config.train_type == 'surf':
        train_surf(config)
    elif config.train_type == 'seg':
        train_seg(config)


