import os
import nibabel as nib
import trimesh
import numpy as np
from tqdm import tqdm
from scipy.ndimage import distance_transform_cdt as cdt
from skimage.measure import marching_cubes
from skimage.measure import label as compute_cc
from skimage.filters import gaussian

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

from data.preprocess import process_volume, process_surface, process_surface_inverse
from util.mesh import laplacian_smooth, compute_normal, compute_mesh_distance, check_self_intersect, compute_distance
#from util.tca import topology
from model.net_4 import CortexODE, Unet, SurfNNODE, BackboneNet 
from config_node import load_config

# initialize topology correction
#topo_correct = topology()

if __name__ == '__main__':
    
    # ------ load configuration ------
    config = load_config()
    test_type = config.test_type  # initial surface / prediction / evaluation
    data_dir = config.data_dir  # directory of datasets
    model_dir = config.model_dir  # directory of pretrained models
    init_dir = config.init_dir  # directory for saving the initial surfaces
    result_dir = config.result_dir  # directory for saving the predicted surfaces
    data_name = config.data_name  # hcp, adni, dhcp
    surf_hemi = config.surf_hemi  # lh, rh
    device = config.device
    tag = config.tag  # identity of the experiment

    C = config.dim_h     # hidden dimension of features
    K = config.kernel_size    # kernel / cube size
    Q = config.n_scale    # multi-scale input
    
    step_size = config.step_size    # step size of integration
    solver = config.solver    # ODE solver
    
    n_inflate = config.n_inflate  # inflation iterations
    rho = config.rho # inflation scale

    eval_epoch_idx = 400 
    # ------ load models ------
    if test_type == 'pred' or test_type == 'eval':
        T = torch.Tensor(np.linspace(0,1,11)).to(device) # 10 steps 

        model_ft = BackboneNet(4,16).to(device)
        model_mid = SurfNNODE().to(device)
        model_wm = SurfNNODE().to(device)
        model_gm = SurfNNODE().to(device)

        model_ft.load_state_dict(torch.load(model_dir+'/model_'+ data_name+'_ths_'+tag+'_ftr_{}epochs.pt'.format(eval_epoch_idx)))
        model_mid.load_state_dict(torch.load(model_dir+'/model_'+ data_name+'_ths_'+tag+'_mid_{}epochs.pt'.format(eval_epoch_idx)))
        model_wm.load_state_dict(torch.load(model_dir+'/model_'+ data_name+'_ths_'+tag+'_wm_{}epochs.pt'.format(eval_epoch_idx)))
        model_gm.load_state_dict(torch.load(model_dir+'/model_'+ data_name+'_ths_'+tag+'_gm_{}epochs.pt'.format(eval_epoch_idx)))

        model_ft.eval()
        model_mid.eval()
        model_wm.eval()
        model_gm.eval()


    # ------ start testing ------
    subject_list = sorted(os.listdir(data_dir))

    if test_type == 'eval':
        CD_gm, AD_gm, HD_gm, SIF_gm = [], [], [], []
        CD_wm, AD_wm, HD_wm, SIF_wm = [], [], [], []

    for i in tqdm(range(len(subject_list))):
        subid = subject_list[i]

        # ------- load brain MRI ------- 
        if data_name == 'hcp' or data_name == 'adni':
            brain = nib.load(data_dir+subid+'/mri/orig.mgz')
            brain_arr = brain.get_fdata()
            brain_arr = (brain_arr / 255.).astype(np.float32)
        elif data_name == 'dhcp':
            brain = nib.load(data_dir+subid+'/'+subid+'_T2w.nii.gz')
            brain_arr = brain.get_fdata()
            brain_arr = (brain_arr / 20).astype(np.float16)
        brain_arr = process_volume(brain_arr, data_name)
        volume_in = torch.Tensor(brain_arr).unsqueeze(0).to(device)

        # ------- load distance map (WM & GM) ------- 
        if data_name == 'adni':
            wm_vol_dist = nib.load(data_dir+subid + '/mri/' + surf_hemi+'_wm_dist.mgz')
            wm_vol_dist = wm_vol_dist.get_fdata() / 1.
            temp_min = wm_vol_dist.min()
            temp_max = wm_vol_dist.max()
            wm_vol_dist = (wm_vol_dist-temp_min)/(temp_max-temp_min)

            gm_vol_dist = nib.load(data_dir+subid + '/mri/' + surf_hemi+'_gm_dist.mgz')
            gm_vol_dist = gm_vol_dist.get_fdata() / 1. 
            temp_min = gm_vol_dist.min()
            temp_max = gm_vol_dist.max()
            gm_vol_dist = (gm_vol_dist-temp_min)/(temp_max-temp_min)
        wm_vol_dist = process_volume(wm_vol_dist, data_name)
        gm_vol_dist = process_volume(gm_vol_dist, data_name)
        wmdm = torch.Tensor(wm_vol_dist).unsqueeze(0).to(device).float()
        gmdm = torch.Tensor(gm_vol_dist).unsqueeze(0).to(device).float()

        # ------- load ribbon seg map ------- 
        if data_name == 'adni':
            seg_vol = nib.load(data_dir + subid + '/mri/ribbon.mgz') 
            seg_vol = seg_vol.get_fdata() / 1.
            seg_vol_map = np.zeros_like(seg_vol)
            if surf_hemi == 'lh':
                seg_vol_map[seg_vol==2] = 0.5
                seg_vol_map[seg_vol==3] = 1
            else:
                seg_vol_map[seg_vol==41] = 0.5
                seg_vol_map[seg_vol==42] = 1
        seg_vol_map = process_volume(seg_vol_map, data_name) 
        segmap = torch.Tensor(seg_vol_map).unsqueeze(0).to(device).float()

        # ------- load input surface ------- 
        if data_name == 'adni':
            # WM 
            # print(data_dir+subid+'/surf/'+surf_hemi+'.white')
            v_in, f_in = nib.freesurfer.io.read_geometry(data_dir+subid+'/surf/'+surf_hemi+'.white')
            v_in, f_in = process_surface(v_in, f_in, data_name)

            # mid 
            v_mid, f_mid = nib.freesurfer.io.read_geometry(data_dir+subid+'/surf/'+surf_hemi+'.mid') 
            v_mid, f_mid = process_surface(v_mid, f_mid, data_name)

            # GM 
            v_gt, f_gt = nib.freesurfer.io.read_geometry(data_dir+subid+'/surf/'+surf_hemi+'.pial')
            #v_gt, f_gt = process_surface(v_gt, f_gt, data_name)
        
        v_mid = torch.Tensor(v_mid).unsqueeze(0).to(device)
        f_mid = torch.LongTensor(f_mid).unsqueeze(0).to(device)

        v_in = v_in*96

        # ------- predict cortical surfaces ------- 
        if test_type == 'pred' or test_type == 'eval':
            with torch.no_grad():
                
                combined_in = torch.cat((volume_in,wmdm,gmdm,segmap),1)

                ftr = model_ft(combined_in) 

                # mid surface 
                model_mid.set_data(v_mid, f_mid, ftr)
                v_out_mid = odeint(model_mid, v_mid, t=T, method=solver, options=dict(step_size=step_size))[-1]
                # wm surface
                model_wm.set_data(v_out_mid, f_mid, ftr)
                v_out_wm = odeint(model_wm, v_out_mid, t=T, method=solver, options=dict(step_size=step_size))[-1]
                # wm surface
                model_gm.set_data(v_out_mid, f_mid, ftr) 
                v_out_gm = odeint(model_gm, v_out_mid, t=T, method=solver, options=dict(step_size=step_size))[-1]         

            v_wm_pred = v_out_wm[0].cpu().numpy()
            f_wm_pred = f_in#[0]#.cpu().numpy()
            v_gm_pred = v_out_gm[0].cpu().numpy()
            f_gm_pred = f_in#[0]#.cpu().numpy()
            
            # inverse 
            v_wm_pred = v_wm_pred*96 
            v_gm_pred = v_gm_pred*96 

        # ------- save predictde surfaces ------- 
        if test_type == 'pred' or test_type == 'eval':
            # save the surfaces in FreeSurfer format
            nib.freesurfer.io.write_geometry(result_dir+data_name+'_'+surf_hemi+'_'+subid+'.white', v_wm_pred, f_wm_pred)
            nib.freesurfer.io.write_geometry(result_dir+data_name+'_'+surf_hemi+'_'+subid+'.pial', v_gm_pred, f_gm_pred)

        # ------- evaluation -------
        if test_type == 'eval':
            
            n_test_pts = v_wm_pred.shape[0]
            #print(v_wm_pred.shape,v_in.shape, f_wm_pred.shape, f_in.shape)
            cd_wm, assd_wm, hd_wm = compute_distance(v_wm_pred, v_in, f_wm_pred, f_in, n_test_pts) 
            cd_gm, assd_gm, hd_gm = compute_distance(v_gm_pred, v_gt, f_gm_pred, f_gt, n_test_pts) 

            CD_wm.append(cd_wm)
            AD_wm.append(assd_wm)
            HD_wm.append(hd_wm)

            CD_gm.append(cd_gm)
            AD_gm.append(assd_gm)
            HD_gm.append(hd_gm)

            ### compute percentage of self-intersecting faces
            ### uncomment below if you have installed torch-mesh-isect
            ### https://github.com/vchoutas/torch-mesh-isect
            # SIF_wm.append(check_self_intersect(v_wm_pred, f_wm_pred, collisions=20))
            # SIF_gm.append(check_self_intersect(v_gm_pred, f_gm_pred, collisions=20)) 
            sif_wm = 0 # check_self_intersect(v_wm_pred, f_wm_pred, collisions=20) 
            sif_gm = 0 # check_self_intersect(v_gm_pred, f_gm_pred, collisions=20) 
            SIF_wm.append(sif_wm)
            SIF_gm.append(sif_gm)

            """
            print('======== wm ========')
            print('cd:', np.mean(CD_wm), np.std(CD_wm))
            print('assd:', np.mean(AD_wm), np.std(AD_wm))
            print('hd:', np.mean(HD_wm), np.std(HD_wm))
            print('sif:', np.mean(SIF_wm), np.std(SIF_wm))
            print('======== gm ========')
            print('cd:', np.mean(CD_gm), np.std(CD_gm))
            print('assd:', np.mean(AD_gm), np.std(AD_gm))
            print('hd:', np.mean(HD_gm), np.std(HD_gm))
            print('sif:', np.mean(SIF_gm), np.std(SIF_gm))
            """

            with open('eval.txt', 'a') as f:
                f.writelines('{} | WM cd {:.6f} assd {:.6f} hd {:.6f} sif {:.6f} | GM cd {:.6f} assd {:.6f} hd {:.6f} sif {:.6f} \n'.format(\
                        subid, cd_wm,assd_wm,hd_wm,sif_wm,  cd_gm,assd_gm, hd_gm, sif_gm ))


    # ------- report the final results ------- 
    if test_type == 'eval':
        print("------------- Grey Matter ---------------")
        print("CD: {} ({})".format(np.mean(CD_gm), np.std(CD_gm)))
        print("AD: {} ({})".format(np.mean(AD_gm), np.std(AD_gm)))
        print("HD: {} ({})".format(np.mean(HD_gm), np.std(HD_gm)))
        print("SIF: {} ({})".format(np.mean(SIF_gm), np.std(SIF_gm)))

        print("------------- White Matter ---------------")
        print("CD: {} ({})".format(np.mean(CD_wm), np.std(CD_wm)))
        print("AD: {} ({})".format(np.mean(AD_wm), np.std(AD_wm)))
        print("HD: {} ({})".format(np.mean(HD_wm), np.std(HD_wm)))
        print("SIF: {} ({})".format(np.mean(SIF_wm), np.std(SIF_wm)))
        print("Finish evaluation.")
        print("----------------------------")
