import numpy as np
import pytorch3d
from pytorch3d.structures import Meshes
import trimesh
from trimesh.exchange.obj import export_obj
from scipy.spatial import cKDTree


def compute_normal(v, f):
    """v, f: Tensors"""
    normal = Meshes(verts=list(v),
                    faces=list(f)).verts_normals_list()[0]
    return normal.unsqueeze(0)


def save_mesh_obj(v, f, n, path):
    mesh_save = trimesh.Trimesh(vertices=v,
                                faces=f,
                                vertex_normals=n)
    obj_save  = export_obj(mesh_save, include_normals=True)
    with open(path, "w") as file:
        print(obj_save, file=file)


def compute_distance(v_pred, v_gt, f_pred, f_gt, n_samples=150000):
    """
    The results are evaluated based on three distances:
    1. Chamfer Distance (CD)
    2. Average Absolute Distance (AD)
    3. Hausdorff Distance (HD)

    Please see DeepCSR paper in details:
    https://arxiv.org/abs/2010.11423
    
    For original code, please see:
    https://bitbucket.csiro.au/projects/CRCPMAX/repos/deepcsr/browse/eval.py
    """
    
    # chamfer distance
    cd = 0
    kdtree = cKDTree(v_pred)
    cd += kdtree.query(v_gt)[0].mean()/2
    kdtree = cKDTree(v_gt)
    cd += kdtree.query(v_pred)[0].mean()/2

    # AD & HD
    mesh_pred = trimesh.Trimesh(vertices=v_pred, faces=f_pred)
    pts_pred = mesh_pred.sample(n_samples)
    mesh_gt = trimesh.Trimesh(vertices=v_gt, faces=f_gt)
    pts_gt = mesh_gt.sample(n_samples)

    _, P2G_dist, _ = trimesh.proximity.closest_point(mesh_pred, pts_gt)
    _, G2P_dist, _ = trimesh.proximity.closest_point(mesh_gt, pts_pred)

    # average absolute distance
    assd = ((P2G_dist.sum() + G2P_dist.sum()) / float(P2G_dist.size + G2P_dist.size))
    
    # Hausdorff distance
    hd = max(np.percentile(P2G_dist, 90), np.percentile(G2P_dist, 90))

    return cd, assd, hd

from mesh_intersection.bvh_search_tree import BVH

def check_self_intersect(v, f, collisions=8):
    """
    Check mesh self-intersections.
    
    We use the calculate_non_manifold_face_intersection function from
    the Neural Mesh Flow paper. For original code please see:
    - https://github.com/KunalMGupta/NeuralMeshFlow/blob/master/evaluation/tools.py
    """
    
    triangles = v[:, f[0]]
    bvh = BVH(max_collisions=collisions)
    outputs = bvh(triangles)
    outputs = outputs.detach().cpu().numpy().squeeze()
    collisions = outputs[outputs[:, 0] >= 0, :]  # the number of collisions
    
    # ------- old version ------- 
    # This just returns the ratio #collisions / #faces.
    # It will over-estimate the percentage of SIFs.
    # return collisions.shape[0] / f.shape[1] * 100.

    # ------- new version ------- 
    # Find all self-intersected faces using a set without overlapping
    sifs = len(set(collisions.reshape(-1)))
    return sifs / f.shape[1] * 100.

