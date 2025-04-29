import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import load_config
from dataload import ADNIDataset2_test, transform_mesh_affine
from surfnn import SurfNN
from utils import compute_normal, save_mesh_obj, compute_distance, check_self_intersect
from pytorch3d.structures import Meshes
import trimesh
import os
import nibabel as nib
import time


def inv_process_surface(v, data_name='adni'):
    if data_name == 'adni':
        # clip/pad the surface according to the volume & normalize to [-1, 1]
        v[:,0] = (v[:,0]+1)/2. * 192 - 5
        v[:,1] = (v[:,1]+1)/2. * 224 - 3
        v[:,2] = (v[:,2]+1)/2. * 192 - 5
    else:
        raise ValueError("data_name should be in ['hcp','adni','dhcp']")
    return v

if __name__ == '__main__':
    
    """set device"""
    gpu=torch.cuda.current_device()
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    print(device)

    """load configuration"""
    config = load_config()


    """load dataset"""
    print("----------------------------")
    print("Start loading dataset ...")
    test_set = ADNIDataset2_test(root_dir=config.data_path+'test/',hemisphere=config.hemisphere)
    testloader = DataLoader(test_set, batch_size=1, shuffle=False)
    print("Finish loading dataset. There are {} testing subjects.".format(len(testloader.dataset)))
    print("----------------------------")

    # shape of MRI
    sample_list = sorted(os.listdir(config.data_path+'test/')) 
    sample_brain = nib.load(config.data_path+'test/' + sample_list[0] + '/mri.nii.gz')
    sample_brain_arr = sample_brain.get_fdata()
    sample_brain_arr = sample_brain_arr / 255. 
    # pad from [182, 218, 182] to [192,224,192] by 10/2,6/2,10/2
    sample_brain_arr = F.pad(torch.Tensor(sample_brain_arr),(5,5,3,3,5,5))
    
    L,W,H = sample_brain_arr.shape    
    LWHmax = max([L,W,H])

    eval_epoch_idx = 399 
    """load model"""
    print("Start loading model ...")
    model = SurfNN(4, 32).to(device)
    model.load_state_dict(torch.load("./ckpts/model/surfnn_model_ths_{}epochs.pt".format(eval_epoch_idx),
                                     map_location=device))
    model.initialize(L, W, H, device)
    print("Finish loading model")
    print("----------------------------")
    

    """evaluation"""
    print("Start evaluation ...")
    with torch.no_grad():
        CD_gm, AD_gm, HD_gm, SIF_gm = [], [], [], []
        CD_wm, AD_wm, HD_wm, SIF_wm = [], [], [], []
        inf_time = []
        for idx, data in tqdm(enumerate(testloader)):

            # volume_in, seg_vol, gmdm, wmdm, v_gt, f_gt, v_in, f_in, v_mid, f_mid, world2vox_affine = data 
            volume_in, seg_vol, gmdm, wmdm, v_gt, f_gt, v_in, f_in, v_mid, f_mid, subid, sub_surf_hemi, world2vox_affine = data 

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

            a=time.time()
            v_out, v_out2, v_mid, v_gm2, v_wm2, Dmid, Dgm, Dwm, Dg2m, Dw2m = model(v=v_mid, f=f_mid, volume=combined_in, n_smooth=config.n_smooth, lambd=config.lambd)
            b=time.time()
            inf_time.append(b-a)

            # gm 
            v_gm = v_out[0].cpu().numpy() 
            f_gm = f_in[0].cpu().numpy()
            v_gt_eval = v_gt[0].cpu().numpy() 
            f_gt_eval = f_gt[0].cpu().numpy()
            n_test_pts = v_gm.shape[0]
            world2vox_affine = world2vox_affine[0].numpy()

            v_gm = inv_process_surface(v_gm,'adni')
            new_v_gm, new_f_gm = transform_mesh_affine(v_gm, f_gm, np.linalg.inv(world2vox_affine))
            temp_v_gm = torch.from_numpy(new_v_gm).unsqueeze(0)
            temp_v_gm = temp_v_gm.to(device)
            tem_f_gm = torch.from_numpy(new_f_gm).unsqueeze(0)
            tem_f_gm = tem_f_gm.to(device)
            new_n_gm = Meshes(verts=list(temp_v_gm),faces=list(tem_f_gm)).verts_normals_list()[0]
            new_n_gm = new_n_gm.cpu().numpy()
            path_save_mesh = "./ckpts/eval/E{:04}_{}_{}_gm.obj".format(eval_epoch_idx, subid[0], config.hemisphere)

            v_gt_eval = inv_process_surface(v_gt_eval,'adni')
            v_gm_gt, f_gm_gt = transform_mesh_affine(v_gt_eval, f_gt_eval, np.linalg.inv(world2vox_affine))
            temp_v_gm = torch.from_numpy(v_gm_gt).unsqueeze(0)
            temp_v_gm = temp_v_gm.to(device)
            tem_f_gm = torch.from_numpy(f_gm_gt).unsqueeze(0)
            tem_f_gm = tem_f_gm.to(device)
            new_n_gm = Meshes(verts=list(temp_v_gm),faces=list(tem_f_gm)).verts_normals_list()[0]
            new_n_gm = new_n_gm.cpu().numpy()
            path_save_mesh = "./ckpts/eval/GT_{}_{}_gm.obj".format(subid[0], config.hemisphere)
            #save_mesh_obj(v_gm_gt, f_gm_gt, new_n_gm, path_save_mesh)

            '''
            cd, assd, hd = compute_distance(new_v_gm,v_gm_gt, new_f_gm,f_gm_gt, n_test_pts) 
            CD_gm.append(cd)
            AD_gm.append(assd)
            HD_gm.append(hd)
            '''

            new_v_gm = torch.Tensor(new_v_gm).unsqueeze(0).to(device)
            new_f_gm = torch.LongTensor(new_f_gm).unsqueeze(0).to(device)
            sif = 0 
            SIF_gm.append(sif)
            #print("{} GM\t CD {}\t AD {}\t HD {}\t sif {}".format(subid[0], cd, assd, hd, sif))
            #print('gm\t {}'.format(sif))

            # wm 
            v_wm = v_out2[0].cpu().numpy() 
            f_wm = f_in[0].cpu().numpy()
            v_in_eval = v_in[0].cpu().numpy() 
            f_in_eval = f_in[0].cpu().numpy() 
            #print('input vm:',v_wm.shape(), f_wm.shape(), v_in_eval.shape(), f_in_eval.shape())
            n_test_pts = v_wm.shape[0]

            v_wm = inv_process_surface(v_wm,'adni')
            new_v_wm, new_f_wm = transform_mesh_affine(v_wm, f_wm, np.linalg.inv(world2vox_affine))
            temp_v_wm = torch.from_numpy(new_v_wm).unsqueeze(0)
            temp_v_wm = temp_v_wm.to(device)
            tem_f_wm = torch.from_numpy(new_f_wm).unsqueeze(0)
            tem_f_wm = tem_f_wm.to(device)
            new_n_wm = Meshes(verts=list(temp_v_wm),faces=list(tem_f_wm)).verts_normals_list()[0]
            new_n_wm = new_n_wm.cpu().numpy()
            path_save_mesh = "./ckpts/eval/E{:04}_{}_{}_wm.obj".format(eval_epoch_idx, subid[0], config.hemisphere)
            #save_mesh_obj(new_v_wm, new_f_wm, new_n_wm, path_save_mesh)

            v_in_eval = inv_process_surface(v_in_eval,'adni')
            v_wm_gt, f_wm_gt = transform_mesh_affine(v_in_eval, f_in_eval, np.linalg.inv(world2vox_affine))
            temp_v_wm = torch.from_numpy(v_wm_gt).unsqueeze(0)
            temp_v_wm = temp_v_wm.to(device)
            tem_f_wm = torch.from_numpy(f_wm_gt).unsqueeze(0)
            tem_f_wm = tem_f_wm.to(device)
            new_n_wm = Meshes(verts=list(temp_v_wm),faces=list(tem_f_wm)).verts_normals_list()[0]
            new_n_wm = new_n_wm.cpu().numpy()
            path_save_mesh = "./ckpts/eval/GT_{}_{}_wm.obj".format(subid[0], config.hemisphere)
            #save_mesh_obj(v_wm_gt, f_wm_gt, new_n_wm, path_save_mesh)

            '''
            cd, assd, hd = compute_distance(new_v_wm,v_wm_gt, new_f_wm,f_wm_gt, n_test_pts) 
            CD_wm.append(cd)
            AD_wm.append(assd)
            HD_wm.append(hd)
            '''
            new_v_wm = torch.Tensor(new_v_wm).unsqueeze(0).to(device)
            new_f_wm = torch.LongTensor(new_f_wm).unsqueeze(0).to(device)
            sif = 0 
            SIF_wm.append(sif)
            #print("{} WM\t CD {}\t AD {}\t HD {}\t sif {}".format(subid[0], cd, assd, hd, sif))
            #print('wm\t {}'.format(sif))
            
            
    print('time:{} ({})'.format(np.mean(inf_time), np.std(inf_time)))

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


