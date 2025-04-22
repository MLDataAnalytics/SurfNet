# SurfNet: Reconstruction of Cortical Surfaces via Coupled Diffeomorphic Deformations

This is the repository for the paper. 

## Get started 
```
conda env create -f environment.yaml
```

## Model Training 
```
python3 train_dist_diff.py  --save_mesh_train True --hemisphere 'ths'   
```

## Model Inference 
```
python3 evaluation2.py  --hemisphere 'lh'  
```
