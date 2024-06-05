import argparse
import os
import sys
from typing import Tuple

import torch_geometric

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

import argparse
import math

import pytorch_lightning as pl
import torch
from dataset.puzzle_dataset import Puzzle_Dataset
from model.spatial_diffusion import GNN_Diffusion
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchvision.datasets import CIFAR100, Caltech256, CelebA, Flickr30k

import wandb


def main(batch_size, gpus, steps):
    celeba_get_fn = lambda x: x[0]

    # celebA_tr = CIFAR100(
    #     root="./datasets",
    #     download=True,
    #     train=True,
    # )

    dt = Caltech256(root="./datasets", download=True)
    # dt_train = dt[:25000]
    # dt_test = dt[25000:]
    dt_train, dt_test = torch.utils.data.random_split(dt, [25000, len(dt) - 25000])

    puzzleDt_train = Puzzle_Dataset(
        dataset=dt_train,
        dataset_get_fn=celeba_get_fn,
        patch_per_dim=[(6, 6), (8, 8), (10, 10), (12, 12)],
    )

    puzzleDt_test = Puzzle_Dataset(
        dataset=dt_test,
        dataset_get_fn=celeba_get_fn,
        patch_per_dim=[(6, 6), (8, 8), (10, 10), (12, 12)],
    )

    dl_train = torch_geometric.loader.DataLoader(
        puzzleDt_train, batch_size=batch_size, num_workers=8, shuffle=False
    )
    dl_test = torch_geometric.loader.DataLoader(
        puzzleDt_test, batch_size=batch_size, num_workers=8, shuffle=False
    )

    # dl_train = dl_test  # TODO <----------------- CHANGE to train once debugging

    save_and_sample_every = 20000  # math.floor(len(dl_train) / gpus / 4)

    model = GNN_Diffusion(
        steps=steps, sampling="DDPM", save_and_sample_every=save_and_sample_every
    )
    model.initialize_torchmetrics([(6, 6), (8, 8), (10, 10), (12, 12)])

    wandb_logger = WandbLogger(
        project="Puzzle-Diff", settings=wandb.Settings(code_dir="."), offline=False
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="overall_acc", mode="max", save_top_k=2
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=gpus,
        strategy="ddp" if gpus > 1 else None,
        # limit_val_batches=10,
        # limit_train_batches=10,
        check_val_every_n_epoch=5,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, dl_train, dl_test)

    pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-batch_size", type=int, default=2)
    ap.add_argument("-gpus", type=int, default=1)
    ap.add_argument("-steps", type=int, default=300)

    args = ap.parse_args()
    main(batch_size=args.batch_size, gpus=args.gpus, steps=args.steps)