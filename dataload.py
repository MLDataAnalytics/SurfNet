import os
import numpy as np
import torch
from tqdm import tqdm
import nibabel as nib
from torch.utils.data import Dataset

from typing import Union, Tuple
import trimesh
from trimesh import Trimesh
from trimesh.scene.scene import Scene
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

"""
volume: brain MRI volume
v_in: vertices of input white matter surface
f_in: faces of ground truth pial surface
v_gt: vertices of input white matter surface
f_gt: faces of ground truth pial surface
"""

from nibabel.affines import apply_affine
import numpy.linalg as npl
def surf_v2vox(v, Torig):
    # Torig is to be obtained from a mgz image img
    # img = nibabel.load('img.mgz')
    # Torig = img.header.get_vox2ras_tkr()
    vox = apply_affine(npl.inv(Torig), v)
    return vox

def vox2surf_v(vox, Torig):
    # Torig is to be obtained from a mgz image img
    # img = nibabel.load('img.mgz')
    # Torig = img.header.get_vox2ras_tkr()
    v = apply_affine(Torig, vox)
    return v
  
def load_mri(path):
    
    brain = nib.load(path)
    brain_arr = brain.get_fdata()
    brain_arr = brain_arr / 255.
    
    # ====== change to your own transformation ======
    # transpose and clip the data to [192,224,192]
    brain_arr = brain_arr.transpose(1,2,0)
    #brain_arr = brain_arr[::-1,:,:]
    #brain_arr = brain_arr[:,:,::-1]
    #brain_arr = brain_arr[32:-32, 16:-16, 32:-32]
    #================================================
    
    return brain_arr.copy()

def load_distmap(path):
    
    brain = nib.load(path)
    brain_arr = brain.get_fdata()
    brain_arr = 1.0 - np.exp(-np.abs(brain_arr))
    
    # ====== change to your own transformation ======
    # transpose and clip the data to [192,224,192]
    brain_arr = brain_arr.transpose(1,2,0)
    brain_arr = brain_arr[::-1,:,:]
    brain_arr = brain_arr[:,:,::-1]
    brain_arr = brain_arr[32:-32, 16:-16, 32:-32]
    #================================================
   
    return brain_arr.copy()

def load_surf(path):
    v, f = nib.freesurfer.io.read_geometry(path)

    # ====== change to your own transformation ======
    # transpose and clip the data to [192,224,192]
    v = v[:,[0,2,1]]
    
    v[:,0] = v[:,0] - 32
    v[:,1] = - v[:,1] - 15
    v[:,2] = v[:,2] - 32

    # normalize to [-1, 1]
    v = v + 128
    v = (v - [96, 112, 96]) / 112
    f = f.astype(np.int32)
    #================================================

    return v, f   

def save_surf(file, surf, fc, Torig):
    # surf is vertices from data loading
    # fc is face information
    # Torig is to be obtained from a mgz image img
    # img = nibabel.load('img.mgz')
    # Torig = img.header.get_vox2ras_tkr()

    # map vertices back to int values in voxel space
    surf[:,0] = (surf[:,0] + 1) / 2 * 191 + 32
    surf[:,1] = (surf[:,1] + 1) / 2 * 223 + 16
    surf[:,2] = (surf[:,2] + 1) / 2 * 191 + 32
    # map them back to surface space
    surf = vox2surf_v(surf, Torig)
    nib.freesurfer.io.write_geometry(file, surf, fc)
    return surf 


def transform_mesh_affine(vertices: Union[np.ndarray, torch.Tensor],
                          faces: Union[np.ndarray, torch.Tensor],
                          transformation_matrix: Union[np.ndarray, torch.Tensor]):
    """ Transform vertices of shape (V, D) or (S, V, D) using a given
    transformation matrix such that v_new = (mat @ v.T).T. """

    ndims = vertices.shape[-1]
    if (tuple(transformation_matrix.shape) != (ndims + 1, ndims + 1)):
        raise ValueError("Wrong shape of transformation matrix.")

    # Convert to torch if necessary
    if isinstance(vertices, np.ndarray):
        vertices_ = torch.from_numpy(vertices).float()
    else:
        vertices_ = vertices
    if isinstance(faces, np.ndarray):
        faces_ = torch.from_numpy(faces).long()
    else:
        faces_ = faces
    if isinstance(transformation_matrix, np.ndarray):
        transformation_matrix = torch.from_numpy(
            transformation_matrix
        ).float().to(vertices_.device)
    vertices_ = vertices_.view(-1, ndims)
    faces_ = faces_.view(-1, ndims)
    coords = torch.cat(
        (vertices_.T, torch.ones(1, vertices_.shape[0]).to(vertices_.device)),
        dim=0
    )

    # Transform
    new_coords = (transformation_matrix @ coords)

    # Adapt faces s.t. normal convention is still fulfilled
    if torch.sum(torch.sign(torch.diag(transformation_matrix)) == -1) % 2 == 1:
        new_faces = faces_.flip(dims=[1])
    else: # No flip required
        new_faces = faces_

    # Correct shape
    new_coords = new_coords.T[:,:-1].view(vertices.shape)
    new_faces = new_faces.view(faces.shape)

    # Correct data type
    if isinstance(vertices, np.ndarray):
        new_coords = new_coords.numpy()
    if isinstance(faces, np.ndarray):
        new_faces = new_faces.numpy()

    return new_coords, new_faces

def process_volume(x, data_name='adni'):
    if data_name == 'adni':
        # pad from [182, 218, 182] to [192,224,192] by 10/2,6/2,10/2
        return torch.nn.functional.pad(x,(5,5,3,3,5,5))
    else:
        raise ValueError("data_name should be in ['hcp','adni','dhcp']")

def process_distmap(x, data_name='adni'):
    if data_name == 'adni':
        # pad from [1,182,218,182] to [1,192,224,192] by 10/2,6/2,10/2
        return torch.nn.functional.pad(x,(5,5,3,3,5,5),mode='replicate')
    else:
        raise ValueError("data_name should be in ['hcp','adni','dhcp']")


def process_surface(v, data_name='adni'):
    if data_name == 'adni':
        # clip/pad the surface according to the volume & normalize to [-1, 1]
        v[:,0] = (v[:,0] + 5)/192 * 2 - 1
        v[:,1] = (v[:,1] + 3)/224 * 2 - 1
        v[:,2] = (v[:,2] + 5)/192 * 2 - 1
    else:
        raise ValueError("data_name should be in ['hcp','adni','dhcp']")
    return v

class ADNIDataset2(Dataset):
    """ADNI dataset."""

    def __init__(self, root_dir, hemisphere):
        self.root_dir = root_dir 
        self.subject_lists = sorted(os.listdir(self.root_dir))
        self.hemisphere = hemisphere

    def __len__(self):
        return len(self.subject_lists)

    def __getitem__(self, idx):
        subid = self.subject_lists[idx]
        if self.hemisphere in ['lh', 'rh']: 
            surf_hemi = self.hemisphere 
        else:
            l_r_choice = np.random.rand()
            if l_r_choice>0.5:
                surf_hemi = 'lh'
            else:
                surf_hemi = 'rh'

        # load brain MRI
        mri = nib.load(os.path.join(self.root_dir, subid ,'mri.nii.gz'))
        volume = mri.get_fdata() / 255.
        # get affine matrix from voxels to surf vertices
        # Torig = mri.header.get_vox2ras_tkr()

        vox2world_affine = mri.affine
        world2vox_affine = np.linalg.inv(vox2world_affine)

        # load pial surface & convert World --> voxel coordinates
        gm_mesh = trimesh.load_mesh(os.path.join(self.root_dir, subid, surf_hemi + "_pial.ply"))
        gm_voxel_verts, gm_voxel_faces = transform_mesh_affine(gm_mesh.vertices, gm_mesh.faces, world2vox_affine)

        # load mid-layer surface
        mid_mesh = trimesh.load_mesh(os.path.join(self.root_dir, subid, surf_hemi + "_mid2.ply"))
        mid_voxel_verts, mid_voxel_faces = transform_mesh_affine(mid_mesh.vertices, mid_mesh.faces, world2vox_affine)

        # load white matter surface
        wm_mesh = trimesh.load_mesh(os.path.join(self.root_dir, subid, surf_hemi + "_white.ply"))
        wm_voxel_verts, wm_voxel_faces = transform_mesh_affine(wm_mesh.vertices, wm_mesh.faces, world2vox_affine)

        # load distance map for pial surface 
        gm_vol_dist = nib.load(os.path.join(self.root_dir, subid , surf_hemi + '_gm_dist.nii.gz'))
        gm_vol_dist = gm_vol_dist.get_fdata() / 1. 
        temp_min = gm_vol_dist.min()
        temp_max = gm_vol_dist.max()
        gm_vol_dist = (gm_vol_dist-temp_min)/(temp_max-temp_min)

        # load distance map for pial surface 
        wm_vol_dist = nib.load(os.path.join(self.root_dir, subid , surf_hemi + '_wm_dist.nii.gz'))
        wm_vol_dist = wm_vol_dist.get_fdata() / 1.
        temp_min = wm_vol_dist.min()
        temp_max = wm_vol_dist.max()
        wm_vol_dist = (wm_vol_dist-temp_min)/(temp_max-temp_min)

        # load segmentation map and process 
        seg_vol = nib.load(os.path.join(self.root_dir, subid , 'ribbon.nii.gz')) 
        seg_vol = seg_vol.get_fdata() / 1.
        seg_vol_map = np.zeros_like(seg_vol)
        if surf_hemi == 'lh':
            seg_vol_map[seg_vol==2] = 0.5
            seg_vol_map[seg_vol==3] = 1
        else:
            seg_vol_map[seg_vol==41] = 0.5
            seg_vol_map[seg_vol==42] = 1

        
        gm_vol_dist = process_distmap(torch.Tensor(gm_vol_dist).unsqueeze(0))
        wm_vol_dist = process_distmap(torch.Tensor(wm_vol_dist).unsqueeze(0))
        #print(gm_vol_dist.shape, wm_vol_dist.shape) 
        seg_vol_map = process_distmap(torch.Tensor(seg_vol_map).unsqueeze(0))

        volume = process_volume(torch.Tensor(volume)).unsqueeze(0)
        
        gm_voxel_verts = process_surface(gm_voxel_verts)
        wm_voxel_verts = process_surface(wm_voxel_verts)
        mid_voxel_verts = process_surface(mid_voxel_verts) 
        

        return volume, seg_vol_map, gm_vol_dist, wm_vol_dist, \
               gm_voxel_verts, gm_voxel_faces, wm_voxel_verts, wm_voxel_faces, mid_voxel_verts,mid_voxel_faces, \
               subid, surf_hemi



class ADNIDataset2_test(Dataset):
    """ADNI dataset."""

    def __init__(self, root_dir, hemisphere):
        self.root_dir = root_dir
        self.subject_lists = sorted(os.listdir(self.root_dir))
        self.hemisphere = hemisphere

    def __len__(self):
        return len(self.subject_lists)

    def __getitem__(self, idx):
        subid = self.subject_lists[idx]

        # load brain MRI
        mri = nib.load(os.path.join(self.root_dir, subid ,'mri.nii.gz'))
        volume = mri.get_fdata() / 255.
        # get affine matrix from voxels to surf vertices
        # Torig = mri.header.get_vox2ras_tkr()

        vox2world_affine = mri.affine
        world2vox_affine = np.linalg.inv(vox2world_affine)

        # load pial surface & convert World --> voxel coordinates
        gm_mesh = trimesh.load_mesh(os.path.join(self.root_dir, subid, self.hemisphere + "_pial.ply"))
        gm_voxel_verts, gm_voxel_faces = transform_mesh_affine(gm_mesh.vertices, gm_mesh.faces, world2vox_affine)

        # load mid-layer surface
        mid_mesh = trimesh.load_mesh(os.path.join(self.root_dir, subid, self.hemisphere + "_mid2.ply"))
        mid_voxel_verts, mid_voxel_faces = transform_mesh_affine(mid_mesh.vertices, mid_mesh.faces, world2vox_affine)

        # load white matter surface
        wm_mesh = trimesh.load_mesh(os.path.join(self.root_dir, subid, self.hemisphere + "_white.ply"))
        wm_voxel_verts, wm_voxel_faces = transform_mesh_affine(wm_mesh.vertices, wm_mesh.faces, world2vox_affine)

        # load distance map for pial surface 
        gm_vol_dist = nib.load(os.path.join(self.root_dir, subid , self.hemisphere + '_gm_dist.nii.gz'))
        gm_vol_dist = gm_vol_dist.get_fdata() / 1.
        temp_min = gm_vol_dist.min()
        temp_max = gm_vol_dist.max()
        gm_vol_dist = (gm_vol_dist-temp_min)/(temp_max-temp_min)

        # load distance map for pial surface 
        wm_vol_dist = nib.load(os.path.join(self.root_dir, subid , self.hemisphere + '_wm_dist.nii.gz'))
        wm_vol_dist = wm_vol_dist.get_fdata() / 1.
        temp_min = wm_vol_dist.min()
        temp_max = wm_vol_dist.max()
        wm_vol_dist = (wm_vol_dist-temp_min)/(temp_max-temp_min)

        # load segmentation map and process 
        seg_vol = nib.load(os.path.join(self.root_dir, subid , 'ribbon.nii.gz')) 
        seg_vol = seg_vol.get_fdata() / 1.
        seg_vol_map = np.zeros_like(seg_vol)
        if self.hemisphere == 'lh':
            seg_vol_map[seg_vol==2] = 0.5
            seg_vol_map[seg_vol==3] = 1
        else:
            seg_vol_map[seg_vol==41] = 0.5
            seg_vol_map[seg_vol==42] = 1

        gm_vol_dist = process_distmap(torch.Tensor(gm_vol_dist).unsqueeze(0))
        wm_vol_dist = process_distmap(torch.Tensor(wm_vol_dist).unsqueeze(0))
        #print(gm_vol_dist.shape, wm_vol_dist.shape) 
        seg_vol_map = process_distmap(torch.Tensor(seg_vol_map).unsqueeze(0))

        volume = process_volume(torch.Tensor(volume))
        volume = volume.unsqueeze(0)
        
        gm_voxel_verts = process_surface(gm_voxel_verts)
        wm_voxel_verts = process_surface(wm_voxel_verts)
        mid_voxel_verts = process_surface(mid_voxel_verts) 


        return volume, seg_vol_map, gm_vol_dist, wm_vol_dist, \
               gm_voxel_verts, gm_voxel_faces, wm_voxel_verts, wm_voxel_faces, mid_voxel_verts,mid_voxel_faces, \
               subid, self.hemisphere, world2vox_affine 




      