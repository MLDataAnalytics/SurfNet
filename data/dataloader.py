import torch
import torch.nn as nn
from torch.utils.data import Dataset

import os
import numpy as np
from tqdm import tqdm
import nibabel as nib
import trimesh
from data.preprocess import process_volume, process_surface
from util.mesh import laplacian_smooth, compute_normal


# ----------------------------
#  for segmentation
# ----------------------------

class SegData():
    def __init__(self, vol, seg):
        self.vol = torch.Tensor(vol)
        self.seg = torch.Tensor(seg)

        vol = []
        seg = []

        
class SegDataset(Dataset):
    def __init__(self, data):
        super(Dataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        brain = self.data[i]
        return brain.vol, brain.seg
    
    
def load_seg_data(config, data_usage='train'):
    """
    data_dir: the directory of your dataset
    data_name: [hcp, adni, dhcp, ...]
    data_usage: [train, valid, test]
    """
    
    data_name = config.data_name
    data_dir = config.data_dir
    data_dir = data_dir + data_usage + '/'

    subject_list = sorted(os.listdir(data_dir))
    data_list = []

    for i in tqdm(range(len(subject_list))):
        subid = subject_list[i]

        if data_name == 'hcp' or data_name == 'adni':
            brain = nib.load(data_dir+subid+'/mri/orig.mgz')
            brain_arr = brain.get_fdata()
            brain_arr = (brain_arr / 255.).astype(np.float32)
            brain_arr = process_volume(brain_arr, data_name)

            seg = nib.load(data_dir+subid+'/mri/ribbon.mgz')
            seg_arr = seg.get_fdata()
            seg_arr = process_volume(seg_arr, data_name)[0]
            seg_left = (seg_arr == 2).astype(int)    # left wm
            seg_right = (seg_arr == 41).astype(int)  # right wm

            seg_arr = np.zeros_like(seg_left, dtype=int)  # final label
            seg_arr += 1 * seg_left
            seg_arr += 2 * seg_right
    
        elif data_name == 'dhcp':
            brain = nib.load(data_dir+subid+'/'+subid+'_T2w.nii.gz')
            brain_arr = brain.get_fdata()
            brain_arr = (brain_arr / 20).astype(np.float32)
            brain_arr = process_volume(brain_arr, data_name)
            
            # wm_label is the generated segmentation by projecting surface into the volume
            seg_arr = np.load(data_dir+subid+'/'+subid+'_wm_label.npy', allow_pickle=True)
            seg_arr = process_volume(seg_arr, data_name)[0]
            
        segdata = SegData(vol=brain_arr, seg=seg_arr)
        # add to data list
        data_list.append(segdata)

    # make dataset
    dataset = SegDataset(data_list)
    
    return dataset


'''
# ----------------------------
#  for surface reconstruction
# ----------------------------
'''

class load_surf_data(Dataset):

    def __init__(self, config, data_usage='train'):
        self.data_dir = config.data_dir + data_usage + '/' 
        self.data_name = config.data_name

        # self.init_dir = config.init_dir + data_usage + '/'
        self.surf_type = config.surf_type   # surf_type: [wm, gm]
        self.surf_hemi = config.surf_hemi   # surf_hemi: [lh, rh, ths]  ths = two hemispheres
        self.device = config.device

        self.n_inflate = config.n_inflate   # 2
        self.rho = config.rho    # 0.002
        self.lambd = config.lambd

        self.subject_list = sorted(os.listdir(self.data_dir))

    def __len__(self):
        return len(self.subject_list)

    def __getitem__(self, idx):
        subid = self.subject_list[idx]
        if self.surf_hemi in ['lh','rh']:
            surf_hemi = self.surf_hemi
        else:
            l_r_choice = np.random.rand()
            if l_r_choice>0.5:
                surf_hemi = 'lh'
            else:
                surf_hemi = 'rh'

        # ------- load brain MRI ------- 
        if self.data_name == 'adni':
            # print(self.data_dir + subid + '/mri/orig.mgz')
            brain = nib.load(self.data_dir + subid + '/mri/orig.mgz')
            brain_arr = brain.get_fdata()
            brain_arr = (brain_arr / 255.).astype(np.float32)
        
        brain_arr = process_volume(brain_arr, self.data_name)

        # ------- load distance map (WM & GM) ------- 
        if self.data_name == 'adni':
            # print(self.data_dir + subid + '/mri/' + surf_hemi+'_wm_dist.mgz')
            wm_vol_dist = nib.load(self.data_dir + subid + '/mri/' + surf_hemi+'_wm_dist.mgz')
            wm_vol_dist = wm_vol_dist.get_fdata() / 1.
            temp_min = wm_vol_dist.min()
            temp_max = wm_vol_dist.max()
            # print('dist wm:', temp_min, temp_max)
            wm_vol_dist = (wm_vol_dist-temp_min)/(temp_max-temp_min)

            # print(self.data_dir + subid + '/mri/' + surf_hemi+'_gm_dist.mgz')
            gm_vol_dist = nib.load(self.data_dir + subid + '/mri/' + surf_hemi+'_gm_dist.mgz')
            gm_vol_dist = gm_vol_dist.get_fdata() / 1. 
            temp_min = gm_vol_dist.min()
            temp_max = gm_vol_dist.max()
            # print('dist gm:', temp_min, temp_max)
            gm_vol_dist = (gm_vol_dist-temp_min)/(temp_max-temp_min)


        wm_vol_dist = process_volume(wm_vol_dist, self.data_name)
        gm_vol_dist = process_volume(gm_vol_dist, self.data_name)

        # ------- load ribbon seg map ------- 
        if self.data_name == 'adni':
            # print(self.data_dir + subid + '/mri/ribbon.mgz')
            seg_vol = nib.load(self.data_dir + subid + '/mri/ribbon.mgz') 
            seg_vol = seg_vol.get_fdata() / 1.
            seg_vol_map = np.zeros_like(seg_vol)
            if surf_hemi == 'lh':
                seg_vol_map[seg_vol==2] = 0.5
                seg_vol_map[seg_vol==3] = 1
            else:
                seg_vol_map[seg_vol==41] = 0.5
                seg_vol_map[seg_vol==42] = 1
        
        seg_vol_map = process_volume(seg_vol_map, self.data_name) 

        # ------- load WM surface ------- 
        if self.data_name == 'adni':
            # print(self.data_dir + subid + '/surf/' + surf_hemi + '.white')
            v_in, f_in = nib.freesurfer.io.read_geometry(self.data_dir + subid + '/surf/' + surf_hemi + '.white')
        #v_in = np.clip(v_in, [-79.9,-79.9,-95.9], [79.9,79.9,95.9])
        v_in, f_in = process_surface(v_in, f_in, self.data_name)

        
        if self.data_name == 'adni':
            # print(self.data_dir+subid+'/surf/'+surf_hemi+'.mid')
            v_mid, f_mid = nib.freesurfer.io.read_geometry(self.data_dir+subid+'/surf/'+surf_hemi+'.mid')
        v_mid, f_mid = process_surface(v_mid, f_mid, self.data_name)


        # ------- load GM surface ------- 
        if self.data_name == 'adni':
            # print(self.data_dir + subid + '/surf/' + surf_hemi + '.pial')
            v_gt, f_gt = nib.freesurfer.io.read_geometry(self.data_dir + subid + '/surf/' + surf_hemi + '.pial') 
        #v_gt = np.clip(v_gt, [-79.9,-79.9,-95.9], [79.9,79.9,95.9])
        v_gt, f_gt = process_surface(v_gt, f_gt, self.data_name)


        v_in = torch.Tensor(v_in)
        f_in = torch.LongTensor(f_in)
        v_gt = torch.Tensor(v_gt)
        f_gt = torch.LongTensor(f_gt)
        v_mid = torch.Tensor(v_mid)
        f_mid = torch.LongTensor(f_mid)

        brain_arr = torch.from_numpy(brain_arr)
        wm_vol_dist = torch.from_numpy(wm_vol_dist)
        gm_vol_dist = torch.from_numpy(gm_vol_dist)
        seg_vol_map = torch.from_numpy(seg_vol_map)


        return brain_arr,wm_vol_dist,gm_vol_dist,seg_vol_map,\
               v_in, v_mid, v_gt, f_in, f_mid, f_gt , subid, surf_hemi

        



