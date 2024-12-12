# 3D Baseline Solvers and Evaluation Metrics for the RePAIR Dataset

This repository provides the implementation of 3D baseline solvers and evaluation metrics for the **RePAIR Dataset**, introduced in the paper:

**"Re-assembling the Past: The RePAIR Dataset and Benchmark for Realistic 2D and 3D Puzzle Solving"** (in NeurIPS 2024).

The RePAIR dataset represents a challenging benchmark for computational puzzle-solving, featuring realistic fragment reassembly scenarios based on archaeological fresco fragments from the Pompeii Archaeological Park. These solvers and metrics serve as benchmarks for evaluating the performance of computational methods in solving complex 3D puzzles with irregular and eroded fragments.
For more details about the RePAIR dataset paper and baselines, visit the [RePAIR NeurIPS Project Page](https://repairproject.github.io/RePAIR_dataset/).

---

## Overview

### Baseline Solvers
1. **Global**: Features are extracted for each part from the input point cloud, along with a global feature, using PointNet. Subsequently, the global feature is concatenated with the individual part features and passed through an MLP network with shared weights to estimate the SE(3) pose for each input point cloud.
2. **LSTM**: A bi-directional LSTM module is created to enhance the understanding of relationships between parts, with part features being processed and the SE(3) pose for each input point cloud sequentially predicted. This approach mirrors the step-by-step decision-making process utilized by humans during shape assembly.
3. **DGL**: GNNs capture part features through modules that reason over edge relationships and aggregate information from nodes. The node aggregation step is removed, which was originally designed to handle geometrically equivalent parts in DGL. This decision is made due to the distinct geometric properties of each piece in the dataset.
4. **SE(3)-Equiv.**: Takes a point cloud for each part and generates two representations: equivariant and invariant. It then computes correlations between parts to create an equivariant representation for each part. Using these representations, rotation and translation decoders predict each part's pose. Performance is further enhanced through additional techniques like adversarial training and reconstructing a canonical point cloud.
5. **DiffAssemble**: The pieces are modeled as nodes in a complete graph, with each node having its own feature encoder. Time-dependent noise is introduced to the translation and rotation of each piece to simulate a shuffled state, similar to scattering puzzle pieces or 3D fragments. During training, an Attention-based GNN processes this noisy graph to restore the original translation and rotation of the pieces. During inference, The pieces are randomized in positions and rotations, and the noise is removed iteratively to reassemble the pieces.

### Evaluation Metrics
The repository includes evaluation metrics to assess puzzle-solving performance. These metrics account for:
- **Q_pos**: It scores the shared areas/volume between ground truth fragments pose (translation and rotation) and the solution given by the evaluated methods.
- **RMSE**: Root Mean Square Error (RMSE) for both translation in millimeters (mm) and rotation in degrees(◦) computed relatively with respect to the ground truth.
- **Neighbor Consistency**: Assessing the accuracy of matching neighboring fragments using a ground-truth mating graph.

These metrics provide a comprehensive evaluation framework for the quality of puzzle-solving solutions.

---

## Installation

### Requirements
- Python 3.8 or later

### Steps
1. Clone the repository:
```
git clone https://github.com/RePAIRProject/3D-baselines.git
cd 3D-baselines
```

2. Download the [RePAIR dataset](https://drive.google.com/drive/folders/1G4ffmH5lxEqITZMNValiModByYUAO6yk).

3. For Global, LSTM, and DGL follow the instructions to install Python dependencies from [multi_part_assembly](https://github.com/Wuziyi616/multi_part_assembly).
4. For SE(3)-Equiv. follow the instructions to install Python dependencies from [SE(3)_Equiv.](https://github.com/crtie/Leveraging-SE-3-Equivariance-for-Learning-3D-Geometric-Shape-Assembly).
5. For DiffAssemble follow the instructions to install Python dependencies from [DiffAssemble](https://github.com/IIT-PAVIS/DiffAssemble).
---

## Usage

### Global
```
cd multi_part_assembly-withoutPivot
python scripts/train.py --cfg_file configs/global/global-32x1-cosine_200e-everyday.py
```

### LSTM
```
cd multi_part_assembly-withoutPivot
python scripts/train.py --cfg_file configs/lstm/lstm-32x1-cosine_200e-everyday.py
```

### DGL
```
cd multi_part_assembly-withoutPivot
python scripts/train.py --cfg_file configs/dgl/dgl-32x1-cosine_200e-everyday.py
```

### SE(3)-Equiv.
```
Leveraging-SE-3-Equivariance-for-Learning-3D-Geometric-Shape-Assembly-withoutPivot
python train.py --cfg_file configs/vnn/vnn-everyday.py
```

### DiffAssemble
```
cd Positional_Puzzle_RePAIR-withoutPivot
python puzzle_diff/train_3d.py --inference_ratio 10 --sampling DDIM --gpus 1 --max_epochs 500 --batch_size 1 --steps 600 --num_workers 12 --noise_weight 0 --predict_xstart True --backbone vn_dgcnn --max_num_part 44 --category all 
```

#### Arguments
- `--cfg_file`: Configuration file.
- `--inference_ratio`: Specifies the ratio for inference during training. 
- `--sampling`: Determines the sampling method to use.
- `--gpus`: Indicates the number of GPUs to use for training.
- `--max_epochs`: Sets the maximum number of epochs for training.
- `--batch_size`: Specifies the number of samples per batch during training.
- `--steps`: Pndicates the number of diffusion steps.
- `--num_workers`: Specifies the number of worker threads to use for data loading.
- `--noise_weight`: Sets the weight of the noise added during training.
- `--predict_xstart`: A boolean flag indicating whether the model should predict the initial latent variable directly in diffusion processes. 
- `--backbone`: Specifies the backbone architecture for the model.
- `--max_num_part`: Sets the maximum number of parts in a puzzle.
- `--category`: Specifies the dataset subset to use. Here, all indicates that the training should include all sub groups in the dataset.


### Evaluation

The following code can be used to test the baselines

### Global
```
cd multi_part_assembly-withoutPivot/
python test.py --cfg_file configs/global/global-32x1-cosine_200e-everyday.py --weight /path/last.ckpt
```

### LSTM
```
cd multi_part_assembly-withoutPivot
python test.py --cfg_file configs/lstm/lstm-32x1-cosine_200e-everyday.py --weight /path/last.ckpt
```

### DGL
```
cd multi_part_assembly-withoutPivot
python test.py --cfg_file configs/dgl/dgl-32x1-cosine_200e-everyday.py --weight /path/last.ckpt
```

### SE(3)-Equiv.
```
cd Leveraging-SE-3-Equivariance-for-Learning-3D-Geometric-Shape-Assembly-withoutPivot/BreakingBad/
python test.py --cfg_file configs/vnn/vnn-everyday.py --weight /path/last.ckpt
```

### DiffAssemble
```
cd Positional_Puzzle_RePAIR-withoutPivot
python puzzle_diff/train_3d.py --inference_ratio 10 --batch_size 1 --steps 600 --num_workers 8 --noise_weight 0 --predict_xstart True  --max_epochs 500 --backbone vn_dgcnn --max_num_part 44 --evaluate True --checkpoint_path /path/last.ckpt --adjth 0.5 
```

**Arguments**:
- `--evaluate`: Set it True for testing.
- `--checkpoint_path`: Path to the checkpoint file.
- `--adjth`: Threshold to compute the adjacency matrix for evaluation.
---

## Acknowledgments

This project has received funding from the European Union under the Horizon 2020 research and innovation program.

---

## Citation

If you use this code in your research, please cite the following paper:

```
@inproceedings{repair2024,
title={Re-assembling the Past: The RePAIR Dataset and Benchmark for Realistic 2D and 3D Puzzle Solving},
author={Tsesmelis, Theodore and Palmieri, Luca and Khoroshiltseva, Marina and Islam, Adeela and Elkin, Gur and Shahar, Ofir Itzhak and Scarpellini, Gianluca and Fiorini, Stefano and Ohayon, Yaniv and Alal, Nadav and Aslan, Sinem and Moretti, Pietro and Vascon, Sebastiano and Gravina, Elena and Napolitano, Maria Cristina and Scarpati, Giuseppe and Zuchtriegel, Gabriel and Spühler, Alexandra and Fuchs, Michel E. and James, Stuart and Ben-Shahar, Ohad and Pelillo, Marcello and Del Bue, Alessio},
booktitle={NeurIPS},
year={2024}
}
```

