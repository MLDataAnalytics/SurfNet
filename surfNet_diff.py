import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from config import load_config
from dataload import ADNIDataset2 
import nibabel as nib
import os
from surfnn import SurfNN 
from utils import compute_normal, save_mesh_obj

import matplotlib.pyplot as plt
import seaborn as sns

from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_normal_consistency
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes


import time

if __name__ == '__main__':
        
    gpu=torch.cuda.current_device()
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    print(device)

    """load configuration"""
    config = load_config()
    
    """load data"""
    print("----------------------------")
    print("Start loading dataset ...")

    # version 2: custom dataloader 
    # load training and validation datasets
    train_set = ADNIDataset2(root_dir=config.data_path+'train/',hemisphere=config.hemisphere)
    valid_set = ADNIDataset2(root_dir=config.data_path+'valid/',hemisphere=config.hemisphere)

    # batch size can only be 1
    trainloader = DataLoader(train_set, batch_size=1, shuffle=True)
    validloader = DataLoader(valid_set, batch_size=1, shuffle=False)
    
    print("Finish loading dataset. There are {} training subjects, {} validate subjects.".format(len(trainloader.dataset), len(validloader.dataset)))
    print("----------------------------")
    
    # shape of MRI
    sample_list = sorted(os.listdir(config.data_path+'train/')) 
    sample_brain = nib.load(config.data_path+'train/' + sample_list[0] + '/mri.nii.gz')
    sample_brain_arr = sample_brain.get_fdata()
    sample_brain_arr = sample_brain_arr / 255. 
    # pad from [182, 218, 182] to [192,224,192] by 10/2,6/2,10/2
    sample_brain_arr = F.pad(torch.Tensor(sample_brain_arr),(5,5,3,3,5,5))
    
    L,W,H = sample_brain_arr.shape    
    LWHmax = max([L,W,H])

    
    """load model"""
    print("Start loading model ...")
    model = SurfNN(4, 32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    model.initialize(L,W,H,device)
    print("Finish loading model")
    print("----------------------------")
    
    """training"""
    print("Start training {} epochs ...".format(config.n_epoch))    
    for epoch in tqdm(range(config.n_epoch+1)):
        avg_loss = []
        for idx, data in enumerate(trainloader):
            volume_in, seg_vol, gmdm, wmdm, v_gt, f_gt, v_in, f_in, v_mid, f_mid, subid, sub_surf_hemi = data 

            volume_in = volume_in.to(device)
            seg_vol = seg_vol.to(device)
            gmdm = gmdm.to(device)
            wmdm = wmdm.to(device)
            combined_in = torch.cat((volume_in,wmdm,gmdm,seg_vol),1)
            v_gt = v_gt.to(device)
            f_gt = f_gt.to(device)
            v_in = v_in.to(device)
            f_in = f_in.to(device)
            v_mid = v_mid.to(device)
            f_mid = f_mid.to(device)

            optimizer.zero_grad()

            start_time = time.time() 

            v_out, v_out2, v_mid, v_gm2, v_wm2, Dmid, Dgm, Dwm, Dg2m, Dw2m = model(v=v_mid, f=f_mid, volume=combined_in, n_smooth=config.n_smooth, lambd=config.lambd)
            
            # part 1: chamfer distance 
            gm_pred = Meshes(verts=v_out, faces=f_gt)
            gm_gt = Meshes(verts=v_gt, faces=f_gt)
            gm_v_pred = sample_points_from_meshes(gm_pred, v_out.shape[1])
            gm_v_gt = sample_points_from_meshes(gm_gt, v_out.shape[1])
            l_gm = 1e3 * chamfer_distance(gm_v_pred, gm_v_gt)[0]    # scale by 1e3 since the coordinates are rescaled to [-1,1]

            wm_pred = Meshes(verts=v_out2, faces=f_in)
            wm_gt = Meshes(verts=v_in, faces=f_in)
            wm_v_pred = sample_points_from_meshes(wm_pred, v_out2.shape[1])
            wm_v_gt = sample_points_from_meshes(wm_gt, v_out2.shape[1])
            l_wm = 1e3 * chamfer_distance(wm_v_pred, wm_v_gt)[0]    # scale by 1e3 since the coordinates are rescaled to [-1,1]


            # part 2: regularization terms 
            # 10 steps 
            wm_d1, wm_d2, wm_d3, wm_d4, wm_d5, wm_d6, wm_d7, wm_d8, wm_d9, wm_d10 = Dwm[0], Dwm[1], Dwm[2], Dwm[3], Dwm[4], Dwm[5], Dwm[6], Dwm[7], Dwm[8], Dwm[9]
            gm_d1, gm_d2, gm_d3, gm_d4, gm_d5, gm_d6, gm_d7, gm_d8, gm_d9, gm_d10 = Dgm[0], Dgm[1], Dgm[2], Dgm[3], Dgm[4], Dgm[5], Dgm[6], Dgm[7], Dgm[8], Dgm[9]
            
            wm_d1_L = torch.norm(wm_d1,dim=2)
            wm_d2_L = torch.norm(wm_d2,dim=2)
            wm_d3_L = torch.norm(wm_d3,dim=2)
            wm_d4_L = torch.norm(wm_d4,dim=2)
            wm_d5_L = torch.norm(wm_d5,dim=2)
            wm_d6_L = torch.norm(wm_d6,dim=2)
            wm_d7_L = torch.norm(wm_d7,dim=2)
            wm_d8_L = torch.norm(wm_d8,dim=2)
            wm_d9_L = torch.norm(wm_d9,dim=2)
            wm_d10_L = torch.norm(wm_d10,dim=2)

            gm_d1_L = torch.norm(gm_d1,dim=2)
            gm_d2_L = torch.norm(gm_d2,dim=2)
            gm_d3_L = torch.norm(gm_d3,dim=2)
            gm_d4_L = torch.norm(gm_d4,dim=2)
            gm_d5_L = torch.norm(gm_d5,dim=2)
            gm_d6_L = torch.norm(gm_d6,dim=2)
            gm_d7_L = torch.norm(gm_d7,dim=2)
            gm_d8_L = torch.norm(gm_d8,dim=2)
            gm_d9_L = torch.norm(gm_d9,dim=2)
            gm_d10_L = torch.norm(gm_d10,dim=2)

            L_wm_and_gm = nn.MSELoss()(wm_d1_L+wm_d2_L+wm_d3_L+wm_d4_L+wm_d5_L+wm_d6_L+wm_d7_L+wm_d8_L+wm_d9_L+wm_d10_L, gm_d1_L+gm_d2_L+gm_d3_L+gm_d4_L+gm_d5_L+gm_d6_L+gm_d7_L+gm_d8_L+gm_d9_L+gm_d10_L) * 1e+3


            # Part 3: vf smoothness term 
            # l_sm_all, l_sm_all2 

            # Part 4: 
            # normal consistency for each pair of neighboring faces  
            l_nc = (mesh_normal_consistency(gm_pred) + mesh_normal_consistency(wm_pred)) * 0.001

            # Part 5: cycle loss 
            l_cyc = (nn.MSELoss()(v_mid,v_gm2) + nn.MSELoss()(v_mid, v_wm2)) * 1e+3 

            # total loss 
            loss  = l_gm + l_wm + l_cyc + L_wm_and_gm + l_nc 
            print('epoch {}-{} {}_{} loss term {:.6f} = l_gm {:.6f} + l_wm {:.6f} + l_cyc {:.6f} + l_dist {:.6f} + l_nc {:.6f} '.format(\
                      epoch,idx, subid[0],sub_surf_hemi[0], loss.item(), l_gm.item(), l_wm.item(), l_cyc.item(), L_wm_and_gm.item(), l_nc.item() ))
            
            avg_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            end_time = time.time() 

            if idx % 50 == 0: 
                with open('train_err_50.txt', 'a') as f:
                    f.writelines('Epoch {}-{} {}_{} loss term {:.6f} = l_gm {:.6f} + l_wm {:.6f} + l_cyc {:.6f} + l_dist {:.6f} + l_nc {:.6f} '.format(\
                        epoch+1, idx, subid[0], sub_surf_hemi[0], loss.item(), l_gm.item(), l_wm.item(), l_cyc.item(), L_wm_and_gm.item(), l_nc.item() ) \
                        + ' running time: {} sec'.format(np.round(end_time - start_time, 4)) + '\n') 

        if config.report_training_loss:
            print("Epoch:{}, training loss:{}".format(epoch+1, np.mean(avg_loss))) 
            with open('train_loss.txt', 'a') as f:
                f.writelines('Epoch_{}_{}'.format(epoch+1, np.mean(avg_loss)) + '\n') 

        if ((epoch+1) % config.ckpts_interval == 0) or ((epoch+1)<50 and (epoch+1)%10 == 0):
            print("----------------------------")
            print("Start validation ...")
            with torch.no_grad():
                error, err_gm, err_wm = [],[],[]
                for idx, data in enumerate(validloader):
                    volume_in, seg_vol, gmdm, wmdm, v_gt, f_gt, v_in, f_in, v_mid, f_mid, subid, sub_surf_hemi = data 
                    volume_in = volume_in.to(device)
                    seg_vol = seg_vol.to(device)
                    v_gt = v_gt.to(device)
                    f_gt = f_gt.to(device)
                    v_in = v_in.to(device)
                    f_in = f_in.to(device)
                    v_mid = v_mid.to(device)
                    f_mid = f_mid.to(device)
                    gmdm = gmdm.to(device)
                    wmdm = wmdm.to(device)
                    combined_in = torch.cat((volume_in,wmdm,gmdm,seg_vol),1)

                    v_out, v_out2, v_mid, v_gm2, v_wm2, Dmid, Dgm, Dwm, Dg2m, Dw2m = model(v=v_mid, f=f_mid, volume=combined_in, n_smooth=config.n_smooth, lambd=config.lambd) 

                    gm_pred = Meshes(verts=v_out, faces=f_gt)
                    gm_gt = Meshes(verts=v_gt, faces=f_gt)
                    gm_v_pred = sample_points_from_meshes(gm_pred, v_out.shape[1])
                    gm_v_gt = sample_points_from_meshes(gm_gt, v_out.shape[1])
                    e_gm = 1e3 * chamfer_distance(gm_v_pred, gm_v_gt)[0]
                    wm_pred = Meshes(verts=v_out2, faces=f_in)
                    wm_gt = Meshes(verts=v_in, faces=f_in)
                    wm_v_pred = sample_points_from_meshes(wm_pred, v_out2.shape[1])
                    wm_v_gt = sample_points_from_meshes(wm_gt, v_out2.shape[1])
                    e_wm = 1e3 * chamfer_distance(wm_v_pred, wm_v_gt)[0]    # chamfer loss
                    error.append(e_gm.item()+e_wm.item())
                    err_gm.append(e_gm.item())
                    err_wm.append(e_wm.item())


            print("Validation error:{}".format(np.mean(error)))
            with open('val_err.txt', 'a') as f:
                f.writelines('Epoch_{}_{} gm:{} wm:{}'.format(epoch+1,np.mean(error),np.mean(err_gm),np.mean(err_wm) ) + '\n') 



            if config.save_model:
                print('Save model checkpoints ... ')
                path_save_model = "./ckpts/model/surfnn_model_"+config.hemisphere+"_"+str(epoch)+"epochs.pt"
                torch.save(model.state_dict(), path_save_model)

            if config.save_mesh_train:
                print('Save pial surface mesh ... ')
                path_save_mesh = "./ckpts/mesh/surfnn_mesh_"+config.hemisphere+"_"+str(epoch)+"epochs_gm.obj"

                normal = compute_normal(v_out, f_in)
                v_gm = v_out[0].cpu().numpy() 
                f_gm = f_in[0].cpu().numpy()
                n_gm = normal[0].cpu().numpy()

                save_mesh_obj(v_gm, f_gm, n_gm, path_save_mesh)

                print('Save wm surface mesh ... ')
                path_save_mesh = "./ckpts/mesh/surfnn_mesh_"+config.hemisphere+"_"+str(epoch)+"epochs_wm.obj"
                normal = compute_normal(v_out2, f_in)
                v_wm = v_out2[0].cpu().numpy() 
                f_wm = f_in[0].cpu().numpy()
                n_wm = normal[0].cpu().numpy()
                save_mesh_obj(v_wm, f_wm, n_wm, path_save_mesh)


            print("Finish validation.")
            print("----------------------------")

    print("Finish training.")
    print("----------------------------")
