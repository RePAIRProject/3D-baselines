This repository contains the official implementation of the baselines used in the paper *Re-assembling the past: The RePAIR dataset and
benchmark for real world 2D and 3D puzzle solving*.

We used **five state-of-the-art learning-based shape assembly methods**:
1. Global [1]
2. LSTM [1]
3. DGL [1]
4. SE(3)-Equiv. [2]
5. DiffAssemble [3]

## Train
The following code can be used to train the baselines

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
python scripts/train.py --cfg_file configs/dgl/dgl-32x1-cosine_200e-everyday.py```
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


## Test
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
python test.py --cfg_file configs/dgl/dgl-32x1-cosine_200e-everyday.py --weight /path/last.ckpt```
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


## References
<a id="1">[1]</a> 
Sellán, Silvia, et al. "Breaking bad: A dataset for geometric fracture and reassembly." Advances in Neural Information Processing Systems 35 (2022): 38885-38898.

<a id="2">[2]</a> 
Wu, Ruihai, et al. "Leveraging SE (3) Equivariance for Learning 3D Geometric Shape Assembly." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

<a id="3">[3]</a> 
Scarpellini, Gianluca, et al. "DiffAssemble: A Unified Graph-Diffusion Model for 2D and 3D Reassembly." arXiv preprint arXiv:2402.19302 (2024).
