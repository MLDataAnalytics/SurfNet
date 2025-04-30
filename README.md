# SurfNet: Reconstruction of Cortical Surfaces via Coupled Diffeomorphic Deformations

This is the repository of SurfNet for coupled cortical surface reconstruction. SurfNet takes as input a combination of MRI brain images, cortical ribbon segmentation maps, and a signed distance map of the midthickness surface. It simultaneously learns three diffeomorphic deformations to optimize the initial midthickness surface (𝒮0) to align with the target midthickness surface (𝒮Mid) using a diffeomorphic deformation model (DDM). Additionally, SurfNet deforms 𝒮Mid outward towards the pial surface (𝒮G) and inward towards the white matter surface (𝒮W) using two other DDMs. A cyclic constraint is applied to regularize the deformation trajectories, along with enforcing non-negative cortical thickness to ensure biological plausibility. This process is illustrated in the figure below. 

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
