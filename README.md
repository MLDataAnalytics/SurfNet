# SurfNet: Reconstruction of Cortical Surfaces via Coupled Diffeomorphic Deformations

This is the repository for the paper. 

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
