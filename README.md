# SurfNet: Reconstruction of Cortical Surfaces via Coupled Diffeomorphic Deformations

This is the repository of [SurfNet for coupled cortical surface reconstruction](<https://doi.org/10.1101/2025.01.30.635814>). SurfNet takes as input a combination of MRI brain images, cortical ribbon segmentation maps, and a signed distance map of the midthickness surface. It simultaneously learns three diffeomorphic deformations to optimize the initial midthickness surface ($S_0$) to align with the target midthickness surface ($S_{M}$) using a diffeomorphic deformation model (DDM). Additionally, SurfNet deforms $S_{M}$ outward towards the pial surface ($S_G$) and inward towards the white matter surface ($S_W$) using two other DDMs. A cyclic constraint is applied to regularize the deformation trajectories, along with enforcing non-negative cortical thickness to ensure biological plausibility. This process is illustrated in the figure below. 

![Figure1](https://github.com/MLDataAnalytics/SurfNet/blob/main/F1.large.jpg)

## Get started 
```
conda env create -f environment.yaml
```

## Model Training 

For CNN-based SurfNet:
```
python3 surfNet_diff.py  --save_mesh_train True --hemisphere 'ths'   
```

For NODE-based SurfNet:
```
python surfNet_node.py --train_type='surf' --data_dir='/Documents/' --model_dir='./ckpts/experiment_1/model/' --data_name='adni'  --surf_hemi='lh' --surf_type='gm' --n_epochs=1000 --tag='exp1' --solver='euler' --step_size=0.1 --device='gpu'
```

## Model Inference 

For CNN-based SurfNet:
```
python3 eval_diff.py  --hemisphere 'lh'  
```

For NODE-based SurfNet:
```
python eval_node.py --test_type='eval'  --data_dir='/Documents/' --model_dir='./ckpts/experiment_1/model/' --result_dir='./ckpts/experiment_1/result/'  --data_name='adni' --surf_hemi='lh' --tag='exp1' --solver='euler' --step_size=0.1 --device='gpu'
```

## References

* Hao Zheng, Hongming Li, Yong Fan, [SurfNet: Reconstruction of Cortical Surfaces via Coupled Diffeomorphic Deformations](https://doi.org/10.1101/2025.01.30.635814), bioRxiv 2025.01.30.635814; doi: https://doi.org/10.1101/2025.01.30.635814
* Hao Zheng, Hongming Li, Yong Fan, [Coupled reconstruction of cortical surfaces by diffeomorphic mesh deformation](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ff0da832a110c6537e885cdfbac80a94-Abstract-Conference.html), Advances in Neural Information Processing Systems, 37, 2023; https://proceedings.neurips.cc/paper_files/paper/2023/file/ff0da832a110c6537e885cdfbac80a94-Paper-Conference.pdf
* Hao Zheng, Hongming Li, Yong Fan, [SurfNN: Joint reconstruction of multiple cortical surfaces from magnetic resonance images](https://doi.org/10.1109/isbi53787.2023.10230488), International Symposium on Biomedical Imaging, 2023; https://doi.org/10.1109/isbi53787.2023.10230488
* Hongming Li, Yong Fan, [MDReg-Net: Multi-resolution diffeomorphic image registration using fully convolutional networks with deep self-supervision](https://doi.org/10.1002/hbm.25782). Human Brain Mapping, 43 (7), 2218–2231; https://doi.org/10.1002/hbm.25782
