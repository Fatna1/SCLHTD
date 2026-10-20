# Volumetric Brain Tumor Segmentation

3D U-Net for brain tumor segmentation on MRI volumes, built with [MONAI](https://monai.io/) and trained on free-tier T4 GPUs via [Modal](https://modal.com/).

<img width="1432" height="510" alt="image" src="https://github.com/user-attachments/assets/4521408c-0e2c-4f38-90e9-d1bb4e8ddcbd" />


## Results

Evaluated on the validation split of the Decathlon BrainTumour dataset (96 volumes):

| Metric | Score |
|---|---|
| **Mean Dice** (excluding background) | **0.643** |
| **Mean HD95** | **12.4 voxels** |

Per-class breakdown:

| Class | Tumor subregion | Dice | HD95 (voxels) |
|---|---|---|---|
| 1 | Necrotic core | 0.698 | 10.81 |
| 2 | Edema | 0.493 | 13.96 |
| 3 | Enhancing tumor | 0.737 | 12.48 |

Edema is the known bottleneck on this dataset, diffuse boundaries and intensity overlap with healthy tissue make it consistently the hardest subregion. The model reflects that pattern.

## Stack

- **MONAI** — data pipeline, 3D U-Net, sliding-window inference, metrics
- **PyTorch** — training loop with AMP
- **Modal** — serverless T4 GPUs, persistent Volumes for dataset and checkpoints
- **matplotlib** — visualization

## Project structure

- config.py: hyperparameters, paths
- data_utils.py: transforms, dataset, dataloaders
- model_utils.py: model, loss, optimizer, save/load
-  train.py: training loop
-   valuate.py: per-class Dice + HD95
-   visualize.py: prediction overlay
-   modal_run.py: Modal wrapper (train / eval entrypoints)
-   requirements.txt


