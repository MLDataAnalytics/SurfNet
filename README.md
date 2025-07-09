# 🧠 SurfNet: Reconstruction of Cortical Surfaces via Coupled Diffeomorphic Deformations

[![DOI](https://img.shields.io/badge/DOI-10.1101/2025.01.30.635814-blue?style=for-the-badge&logo=biorxiv)](https://doi.org/10.1101/2025.01.30.635814)
[![TMI](https://img.shields.io/badge/TMI-Pending_Publication-green?style=for-the-badge&logo=ieee)](https://doi.org/10.1109/TMI.2025.3585088)

This repository contains the official implementation of **SurfNet**, our novel method for coupled cortical surface reconstruction, as presented in our work available on [TMI](https://doi.org/10.1109/TMI.2025.3585088) and [bioRxiv](https://doi.org/10.1101/2025.01.30.635814).

**SurfNet** offers a robust approach to reconstructing high-quality cortical surfaces (white matter, midthickness, and pial) from MRI brain images. Unlike traditional methods, SurfNet simultaneously learns three diffeomorphic deformations, ensuring biological plausibility and topological consistency across the reconstructed surfaces.

---

## ✨ Core Concept

SurfNet takes as input MRI brain images, cortical ribbon segmentation maps, and a signed distance map of the midthickness surface. Its innovative approach involves:

1.  **Midthickness Surface Optimization:** It optimizes an initial midthickness surface ($S_0$) to precisely align with the target midthickness surface ($S_{M}$) using a dedicated Diffeomorphic Deformation Model (DDM).
2.  **Coupled Pial and White Matter Deformation:** Simultaneously, SurfNet deforms $S_{M}$ outward towards the pial surface ($S_G$) and inward towards the white matter surface ($S_W$) using two other DDMs.
3.  **Cyclic and Non-negative Thickness Constraints:** A crucial cyclic constraint is applied to regularize the deformation trajectories, while non-negative cortical thickness is enforced throughout the process to ensure anatomical accuracy and biological plausibility.

This intricate process is elegantly illustrated in the figure below:

![SurfNet Coupled Deformation Process](https://github.com/MLDataAnalytics/SurfNet/blob/main/F1.large.jpg)
* An illustration of SurfNet's coupled diffeomorphic deformation process for cortical surface reconstruction.*

---

## 🚀 Get Started

To set up your environment and begin using SurfNet, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/MLDataAnalytics/SurfNet.git](https://github.com/MLDataAnalytics/SurfNet.git)
    cd SurfNet
    ```
2.  **Create and activate the Conda environment:**
    ```bash
    conda env create -f environment.yaml
    conda activate surfnet
    ```

---

## 🏋️ Model Training

SurfNet provides two main architectures: a CNN-based model and a NODE-based model.

### For CNN-based SurfNet:

Train the model using the following command:

```bash```

```python3 surfNet_diff.py --save_mesh_train True --hemisphere 'ths' ```

### For NODE-based SurfNet:

Train the model with specified data and model directories:

```Bash```

```python surfNet_node.py --train_type='surf' \
                       --data_dir='/Documents/' \
                       --model_dir='./ckpts/experiment_1/model/' \
                       --data_name='adni' \
                       --surf_hemi='lh' \
                       --surf_type='gm' \
                       --n_epochs=1000 \
                       --tag='exp1' \
                       --solver='euler' \
                       --step_size=0.1 \
                       --device='gpu'
```

Note: Please adjust --data_dir to your actual dataset path and --model_dir to your desired checkpoint save location.

---

## 🧪 Model Inference
After training, you can use the trained models to perform inference and reconstruct cortical surfaces.

### For CNN-based SurfNet:

Run inference for a specified hemisphere:

```Bash```

``` python3 eval_diff.py --hemisphere 'lh' ```


### For NODE-based SurfNet:

Perform evaluation with your trained model:

```Bash```

```python eval_node.py --test_type='eval' \
                    --data_dir='/Documents/' \
                    --model_dir='./ckpts/experiment_1/model/' \
                    --result_dir='./ckpts/experiment_1/result/' \
                    --data_name='adni' \
                    --surf_hemi='lh' \
                    --tag='exp1' \
                    --solver='euler' \
                    --step_size=0.1 \
                    --device='gpu'
```

Note: Ensure --data_dir, --model_dir, and --result_dir are correctly set.

---

## 📄 References

Our work builds upon and relates to several key publications in the field of medical image analysis and diffeomorphic deformations.

### SurfNet (Our latest work):
* Zheng H, Li H, Fan Y. "SurfNet: Reconstruction of Cortical Surfaces via Coupled Diffeomorphic Deformations," IEEE Trans Med Imaging. 2025 Jul 2;PP. doi: https://doi.org/10.1109/TMI.2025.3585088. Epub ahead of print. PMID: 40601461.
* Hao Zheng, Hongming Li, Yong Fan, "SurfNet: Reconstruction of Cortical Surfaces via Coupled Diffeomorphic Deformations," bioRxiv 2025.01.30.635814; doi: https://doi.org/10.1101/2025.01.30.635814

### Related Works by the Authors:

* Hao Zheng, Hongming Li, Yong Fan, "Coupled reconstruction of cortical surfaces by diffeomorphic mesh deformation," Advances in Neural Information Processing Systems, 37, 2023; https://proceedings.neurips.cc/paper_files/paper/2023/file/ff0da832a110c6537e885cdfbac80a94-Paper-Conference.pdf

* Hao Zheng, Hongming Li, Yong Fan, "SurfNN: Joint reconstruction of multiple cortical surfaces from magnetic resonance images," International Symposium on Biomedical Imaging, 2023; https://doi.org/10.1109/isbi53787.2023.10230488

* Hongming Li, Yong Fan, "MDReg-Net: Multi-resolution diffeomorphic image registration using fully convolutional networks with deep self-supervision," Human Brain Mapping, 43 (7), 2218–2231; https://doi.org/10.1002/hbm.25782

If you find our work or this code useful for your research, please consider citing our paper:
```
@ARTICLE{11063456,
  author={Zheng, Hao and Li, Hongming and Fan, Yong},
  journal={IEEE Transactions on Medical Imaging}, 
  title={SurfNet: Reconstruction of Cortical Surfaces via Coupled Diffeomorphic Deformations}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Surface reconstruction;Deformation;Surface morphology;Image reconstruction;Magnetic resonance imaging;Topology;Estimation;Accuracy;Deformable models;Trajectory;Cortical surface reconstruction;diffeomorphic deformation;ODE;thickness estimation},
  doi={10.1109/TMI.2025.3585088}}
```

---

## 📬 Contact

For any questions, collaborations, or further information, please feel free to reach out by opening an issue in this repository.

---

## 🙏 Acknowledgment

This project has been generously supported in part by the National Institutes of Health (NIH) through grants **AG066650**, **U24NS130411**, and **R01EB022573**. We are grateful for their support in making this research possible.
