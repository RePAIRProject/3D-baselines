import colorsys
import enum
import math
import pickle

# from .backbones.Transformer_GNN import Transformer_GNN
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import einops
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pytorch_lightning as pl
import scipy
import timm

# from .network_modules import (
#     default,
#     partial,
#     SinusoidalPositionEmbeddings,
#     PreNorm,
#     Downsample,
#     Upsample,
#     Residual,
#     LinearAttention,
#     ConvNextBlock,
#     ResnetBlock,
#     Attention,
#     exists,
# )
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn.models
import torchmetrics
import torchvision
import torchvision.transforms.functional as trF
from kornia.geometry.transform import Rotate as krot
from PIL import Image
from scipy.stats import kendalltau
from torch import Tensor
from torch.optim import Adam
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor

import wandb

# from .backbones import Dark_TFConv, Eff_GAT
from .backbones.efficient_gat_text import Eff_GAT_TEXT

matplotlib.use("agg")


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


def interpolate_color1d(color1, color2, fraction):
    # color1 = [float(x) / 255 for x in color1]
    # color2 = [float(x) / 255 for x in color2]
    hsv1 = color1  #  colorsys.rgb_to_hsv(*color1)
    hsv2 = color2  #  colorsys.rgb_to_hsv(*color2)
    h = hsv1[0] + (hsv2[0] - hsv1[0]) * fraction
    s = hsv1[1] + (hsv2[1] - hsv1[1]) * fraction
    v = hsv1[2] + (hsv2[2] - hsv1[2]) * fraction
    return tuple(x for x in (h, s, v))


def interpolate_color(
    pos, col_1=(1, 0, 0), col_2=(1, 1, 0), col_3=(0, 0, 1), col_4=(0, 1, 0)
):
    f1 = float((pos[0] + 1) / 2)
    f2 = float((pos[1] + 1) / 2)
    c1 = interpolate_color1d(col_1, col_2, f1)
    c2 = interpolate_color1d(col_3, col_4, f1)
    return interpolate_color1d(c1, c2, f2)


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def rotate_images(patches, rot_vector):
    angle_vec = rot_vector  # x_noisy[:, -2:]
    angles = -torch.atan2(angle_vec[:, 1], angle_vec[:, 0]) / torch.pi * 180
    # rotated_patches = torch.stack(
    #     [trF.rotate(cond_img, rot.item()) for cond_img, rot in zip(patches, angles)]
    # )

    r = krot(angles, mode="nearest")
    rot2 = r(patches)
    return rot2


def cosine_beta_schedule(timesteps, s=0.08):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def extract(a, t, x_shape=None):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out[:, None]  # out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


@torch.jit.script
def greedy_cost_assignment(pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
    # Compute pairwise distances between positions
    dist = torch.norm(pos1[:, None] - pos2, dim=2)

    # Create a tensor to store the assignments
    assignments = torch.zeros(dist.size(0), 3, dtype=torch.int64)

    # Create a mask to keep track of assigned positions
    mask = torch.ones_like(dist, dtype=torch.bool)

    # Counter for keeping track of the number of assignments
    counter = 0

    # While there are still unassigned positions
    while mask.sum() > 0:
        # Find the minimum distance
        min_val, min_idx = dist[mask].min(dim=0)

        # Get the indices of the two dimensions
        idx = int(min_idx.item())
        ret = mask.nonzero()[idx, :]
        i = ret[0]
        j = ret[1]

        # Add the assignment to the tensor
        assignments[counter, 0] = i
        assignments[counter, 1] = j
        assignments[counter, 2] = min_val

        # Increase the counter
        counter += 1

        # Remove the assigned positions from the distance matrix and the mask
        mask[i, :] = 0
        mask[:, j] = 0

    return assignments[:counter]


class GNN_Diffusion(pl.LightningModule):
    def __init__(
        self,
        steps=600,
        inference_ratio=1,
        sampling="DDPM",
        learning_rate=1e-4,
        save_and_sample_every=1000,
        bb=None,
        classifier_free_prob=0,
        classifier_free_w=0,
        noise_weight=0.0,
        rotation=False,
        model_mean_type: ModelMeanType = ModelMeanType.EPSILON,
        warmup_steps=1000,
        max_train_steps=10000,
        architecture: str = "transformer",
        virt_nodes: int = 4,

        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model_mean_type = model_mean_type
        self.learning_rate = learning_rate
        self.save_and_sample_every = save_and_sample_every
        self.classifier_free_prob = classifier_free_prob
        self.classifier_free_w = classifier_free_w
        self.noise_weight = noise_weight
        self.rotation = rotation

        self.warmup_steps = warmup_steps
        self.max_train_steps = max_train_steps
        ### DIFFUSION STUFF

        if sampling == "DDPM":
            self.inference_ratio = 1
            self.p_sample = partial(
                self.p_sample,
                sampling_func=self.p_sample_ddpm,
            )
            self.eta = 1
        elif sampling == "DDIM":
            self.inference_ratio = inference_ratio
            self.p_sample = partial(
                self.p_sample,
                sampling_func=self.p_sample_ddim,
            )
            self.eta = 0

        # define beta schedule
        betas = linear_beta_schedule(timesteps=steps)
        # self.timesteps = torch.arange(0, 700).flip(0)
        self.register_buffer("betas", betas)
        # self.betas = cosine_beta_schedule(timesteps=steps)
        # define alphas
        alphas = 1.0 - self.betas
        self.register_buffer("alphas", alphas)
        alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.register_buffer("sqrt_recip_alphas", sqrt_recip_alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)

        self.register_buffer(
            "sqrt_recip_alphas_cumprod", np.sqrt(1.0 / self.alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", np.sqrt(1.0 / self.alphas_cumprod - 1)
        )

        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        self.architecture = architecture
        self.virt_nodes = virt_nodes


        self.steps = steps

        ### BACKBONE
        self.model = Eff_GAT_TEXT(steps=steps, input_channels=1, output_channels=1, architecture=self.architecture, virt_nodes=self.virt_nodes)


        self.save_hyperparameters()

    def initialize_torchmetrics(self):
        metrics = {}

        # for i in n_patches:
        #     metrics[f"{i}_acc"] = torchmetrics.MeanMetric()
        #     metrics[f"{i}_nImages"] = torchmetrics.SumMetric()
        metrics["accuracy"] = torchmetrics.MeanMetric()
        metrics["tau"] = torchmetrics.MeanMetric()
        metrics["pmr"] = torchmetrics.MeanMetric()
        metrics["overall_nImages"] = torchmetrics.SumMetric()
        self.metrics = nn.ModuleDict(metrics)

    def forward(self, xy_pos, time, patch_rgb, edge_index, batch) -> Any:
        return self.model(xy_pos, time, patch_rgb, edge_index, batch)
        # # mean = patch_rgb.new_tensor([0.4850, 0.4560, 0.4060])[None, :, None, None]
        # # std = patch_rgb.new_tensor([0.2290, 0.2240, 0.2250])[None, :, None, None]
        # # if patch_feats == None:

        # patch_rgb = (patch_rgb - self.mean) / self.std

        # # fe[3].reshape(fe[0].shape[0],-1)
        # patch_feats = self.visual_backbone.forward(patch_rgb)[3].reshape(
        #     patch_rgb.shape[0], -1
        # )
        # # patch_feats = patch_feats
        # time_feats = self.time_emb(time)
        # pos_feats = self.pos_mlp(xy_pos)
        # combined_feats = torch.cat([patch_feats, pos_feats, time_feats], -1)
        # combined_feats = self.mlp(combined_feats)
        # feats = self.gnn_backbone(x=combined_feats, edge_index=edge_index)
        # final_feats = self.final_mlp(feats + combined_feats)

        # return final_feats

    def forward_with_feats(
        self,
        xy_pos: Tensor,
        time: Tensor,
        patch_rgb: Tensor,
        edge_index: Tensor,
        patch_feats: Tensor,
        batch,
    ) -> Any:
        return self.model.forward_with_feats(
            xy_pos, time, patch_rgb, edge_index, patch_feats, batch
        )
        # mean = patch_rgb.new_tensor([0.4850, 0.4560, 0.4060])[None, :, None, None]
        # std = patch_rgb.new_tensor([0.2290, 0.2240, 0.2250])[None, :, None, None]
        # if patch_feats == None:

        # patch_rgb = (patch_rgb - self.mean) / self.std

        # # fe[3].reshape(fe[0].shape[0],-1)
        # patch_feats = self.visual_backbone.forward(patch_rgb)[3].reshape(
        # patch_rgb.shape[0], -1
        # )
        # patch_feats = patch_feats
        # time_feats = self.time_emb(time)
        # pos_feats = self.pos_mlp(xy_pos)
        # combined_feats = torch.cat([patch_feats, pos_feats, time_feats], -1)
        # combined_feats = self.mlp(combined_feats)
        # feats = self.gnn_backbone(x=combined_feats, edge_index=edge_index)
        # final_feats = self.final_mlp(feats + combined_feats)

        # return final_feats

    def visual_features(self, patch_rgb):
        # patch_rgb = (patch_rgb - self.mean) / self.std

        # # fe[3].reshape(fe[0].shape[0],-1)
        # patch_feats = self.visual_backbone.forward(patch_rgb)[3].reshape(
        #     patch_rgb.shape[0], -1
        # )
        # return patch_feats
        return self.model.visual_features(patch_rgb)

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    # def p_losses_old(self, x_start, t, noise=None, loss_type="l1", cond=None):
    #     if noise is None:
    #         noise = torch.randn_like(x_start)

    #     x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
    #     predicted_noise = self(x_noisy, t, cond)

    #     if loss_type == "l1":
    #         loss = F.l1_loss(noise, predicted_noise)
    #     elif loss_type == "l2":
    #         loss = F.mse_loss(noise, predicted_noise)
    #     elif loss_type == "huber":
    #         loss = F.smooth_l1_loss(noise, predicted_noise)
    #     else:
    #         raise NotImplementedError()

    #     return loss

    def p_losses(
        self,
        x_start,
        t,
        noise=None,
        loss_type="l1",
        cond=None,
        edge_index=None,
        batch=None,
    ):
        if noise is None:
            noise = torch.randn_like(x_start)

        if self.steps == 1:
            x_noisy = torch.zeros_like(x_start)
        else:
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # rotation of middle images doesn't work
        # if self.rotation:
        #     cond = rotate_images(cond, x_noisy[:, -2:])

        patch_feats = self.visual_features(cond)
        batch_size = batch.max() + 1
        batch_one_hot = torch.nn.functional.one_hot(batch)
        prob = (
            batch_one_hot.float() @ torch.rand(batch_size, device=self.device)
            > self.classifier_free_prob
        )
        classifier_free_patch_feats = prob[:, None] * patch_feats

        prediction = self.forward_with_feats(
            x_noisy,
            t,
            cond,
            edge_index,
            patch_feats=classifier_free_patch_feats,
            batch=batch,
        )

        target = {
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]

        if loss_type == "l1":
            loss = F.l1_loss(target, prediction)
        elif loss_type == "l2":
            loss = F.mse_loss(target, prediction)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(target, prediction)
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    def p_sample_ddpm(self, x, t, t_index, cond, edge_index, patch_feats, batch):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x
            - betas_t
            * self.forward_with_feats(
                x, t, cond, edge_index, patch_feats=patch_feats, batch=batch
            )
            / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def _get_variance_old(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (
            1 - alpha_prod_t / alpha_prod_t_prev
        )

        return variance

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = extract(
            self.alphas_cumprod, timestep
        )  # self.alphas_cumprod[timestep]

        alpha_prod_t_prev = (
            extract(self.alphas_cumprod, prev_timestep)
            if (prev_timestep >= 0).all()
            else alpha_prod_t * 0 + 1
        )

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (
            1 - alpha_prod_t / alpha_prod_t_prev
        )

        return variance

    @torch.no_grad()
    def p_sample_ddim(
        self, x, t, t_index, cond, edge_index, patch_feats, batch
    ):  # (self, x, t, t_index, cond):
        # if t[0] == 0:
        #     return x

        prev_timestep = t - self.inference_ratio

        eta = self.eta
        alpha_prod = extract(self.alphas_cumprod, t, x.shape)

        if (prev_timestep >= 0).all():
            alpha_prod_prev = extract(self.alphas_cumprod, prev_timestep, x.shape)
        else:
            alpha_prod_prev = alpha_prod * 0 + 1

        beta = 1 - alpha_prod
        beta_prev = 1 - alpha_prod_prev

        if self.classifier_free_prob > 0.0:
            model_output_cond = self.forward_with_feats(
                x, t, cond, edge_index, patch_feats=patch_feats, batch=batch
            )

            model_output_uncond = self.forward_with_feats(
                x,
                t,
                cond,
                edge_index,
                patch_feats=torch.zeros_like(patch_feats),
                batch=batch,
            )
            model_output = (
                1 + self.classifier_free_w
            ) * model_output_cond - self.classifier_free_w * model_output_uncond
        else:
            model_output = self.forward_with_feats(
                x, t, cond, edge_index, patch_feats=patch_feats, batch=batch
            )

        # estimate x_0

        x_0 = {
            ModelMeanType.EPSILON: (x - beta**0.5 * model_output) / alpha_prod**0.5,
            ModelMeanType.START_X: model_output,
        }[self.model_mean_type]
        eps = self._predict_eps_from_xstart(x, t, x_0)

        variance = self._get_variance(
            t, prev_timestep
        )  # (beta_prev / beta) * (1 - alpha_prod / alpha_prod_prev)
        std_eta = eta * variance**0.5

        # estimate "direction to x_t"
        pred_sample_direction = (1 - alpha_prod_prev - std_eta**2) ** (0.5) * eps

        # x_t-1 = a * x_0 + b * eps
        prev_sample = alpha_prod_prev ** (0.5) * x_0 + pred_sample_direction

        if eta > 0:
            noise = torch.randn(model_output.shape, dtype=model_output.dtype).to(
                self.device
            )
            prev_sample = prev_sample + std_eta * noise
        return prev_sample

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    # Algorithm 2 but save all images:
    @torch.no_grad()
    def p_sample_loop(self, shape, cond, edge_index, batch):
        # device = next(model.parameters()).device
        device = self.device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device) * self.noise_weight
        # img = einops.rearrange(
        #     img,
        #     "b c (w1 w) (h1 h) -> b (w1 h1) c w h",
        #     h1=self.patches,
        #     w1=self.patches,
        # )
        imgs = []

        patch_feats = self.visual_features(cond)

        # time_t = torch.full((b,), i, device=device, dtype=torch.long)

        # time_t = torch.full((b,), 0, device=device, dtype=torch.long)

        for i in tqdm(
            list(reversed(range(0, self.steps, self.inference_ratio))),
            desc="sampling loop time step",
        ):
            img = self.p_sample(
                img,
                torch.full((b,), i, device=device, dtype=torch.long),
                # time_t + i,
                i,
                cond=cond,
                edge_index=edge_index,
                patch_feats=patch_feats,
                batch=batch,
            )
            # uncomment to rotate images during sampling
            # if self.rotation:
            #     rotated_imgs = rotate_images(cond, img[:, -2:])
            #     patch_feats = self.visual_features(rotated_imgs)

            # imgs.append(img)
            # if i is not None:  # == 0:
            # if i == 0:
            #     img2 = img.clone()

            #     img2 = torch.cat((cond, img2), 2)
            #     img2 = einops.rearrange(
            #         img2,
            #         "b (w1 h1) c w h -> b c (w1 w) (h1 h)",
            #         h1=self.patches,
            #         w1=self.patches,
            #     )
            #     imgs.append(img2.cpu().numpy())
            imgs.append(img)
        return imgs

    @torch.no_grad()
    def p_sample(
        self, x, t, t_index, cond, edge_index, sampling_func, patch_feats, batch
    ):
        return sampling_func(x, t, t_index, cond, edge_index, patch_feats, batch)

    @torch.no_grad()
    def sample(
        self,
        image_size,
        batch_size=16,
        channels=3,
        cond=None,
        edge_index=None,
        batch=None,
    ):
        return self.p_sample_loop(
            shape=(batch_size, channels, image_size, image_size),
            cond=cond,
            edge_index=edge_index,
            batch=batch,
        )

    def configure_optimizers(self):
        # optimizer = torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)
        # optimizer = Adafactor(self.parameters())
        optimizer = Adafactor(self.parameters())
        return optimizer

        # optimizer = AdamW(self.parameters(), lr=5e-6)
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=self.warmup_steps,
        #     num_training_steps=self.max_train_steps,
        # )

        # lr_scheduler_config = {
        #     # REQUIRED: The scheduler instance
        #     "scheduler": scheduler,
        #     # The unit of the scheduler's step size, could also be 'step'.
        #     # 'epoch' updates the scheduler on epoch end whereas 'step'
        #     # updates it after a optimizer update.
        #     "interval": "step",
        #     # How many epochs/steps should pass between calls to
        #     # `scheduler.step()`. 1 corresponds to updating the learning
        #     # rate after every epoch/step.
        #     "frequency": 1,
        #     # Metric to to monitor for schedulers like `ReduceLROnPlateau`
        #     # If set to `True`, will enforce that the value specified 'monitor'
        #     # is available when the scheduler is updated, thus stopping
        #     # training if not found. If set to `False`, it will only produce a warning
        #     "strict": True,
        #     # If using the `LearningRateMonitor` callback to monitor the
        #     # learning rate progress, this keyword can be used to specify
        #     # a custom logged name
        # }
        # return [optimizer], [lr_scheduler_config]

    def training_step(self, batch, batch_idx):
        # return super().training_step(*args, **kwargs)
        batch_size = batch.batch.max().item() + 1
        t = torch.randint(0, self.steps, (batch_size,), device=self.device).long()

        new_t = torch.gather(t, 0, batch.batch)

        loss = self.p_losses(
            batch.x,
            new_t,
            loss_type="huber",
            cond=batch.phrases_text,
            edge_index=batch.edge_index,
            batch=batch.batch,
        )
        # if batch_idx == 0 and self.local_rank == 0:
        #     imgs = self.p_sample_loop(
        #         batch.x.shape, batch.patches, batch.edge_index, batch=batch.batch
        #     )
        #     img = imgs[-1]

        #     save_path = Path(f"results/{self.logger.experiment.name}/train")
        #     for i in range(
        #         min(batch.batch.max().item(), 4)
        #     ):  # save max 4 images during training loop
        #         idx = torch.where(batch.batch == i)[0]
        #         patches_rgb = batch.patches[idx]
        #         gt_pos = batch.x[idx]
        #         pos = img[idx]
        #         n_patches = batch.patches_dim[i]
        #         i_name = batch.ind_name[i]
        #         if self.rotation:
        #             gt_pos = batch.x[idx, :2]
        #             pos = img[idx, :2]
        #             pred_rot = img[idx, 2:]
        #             gt_rot = batch.x[idx, 2:]
        #             self.save_image_rotated(
        #                 patches_rgb=patches_rgb,
        #                 pos=pos,
        #                 gt_pos=gt_pos,
        #                 patches_dim=n_patches,
        #                 ind_name=i_name,
        #                 file_name=save_path,
        #                 gt_rotations=gt_rot,
        #                 pred_rotations=pred_rot,
        #             )
        #         else:
        #             self.save_image(
        #                 patches_rgb=patches_rgb,
        #                 pos=pos,
        #                 gt_pos=gt_pos,
        #                 patches_dim=n_patches,
        #                 ind_name=i_name,
        #                 file_name=save_path,
        #             )

        self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            imgs = self.p_sample_loop(
                batch.x.shape, batch.phrases_text, batch.edge_index, batch=batch.batch
            )

            img = imgs[-1]
            n = 0

            for i in range(batch.batch.max() + 1):
                pos = img[batch.batch == i]
                # Saving entire diff process
                print(f"{self.logger.experiment.name}")
                file_path = Path(f"results/{self.logger.experiment.name}/vals/")

                file_path.mkdir(parents=True, exist_ok=True)

                with open(
                    file_path / f"{batch_idx * len(batch.num_phrases) + n}.pkl", "wb"
                ) as fp:

                    pickle.dump(
                        {
                            "pos": [
                                imgs[n][batch.batch == i].cpu()
                                for n in range(len(imgs))
                            ],
                            "sentences": batch.phrases_text[i],
                        },
                        fp,
                    )
                n += 1

                self.metrics["overall_nImages"].update(1)

                correct = False
                if ((pos[1:] - pos[:-1]) > 0).all():
                    correct = True

                match = torch.argsort(pos.squeeze()) == torch.arange(len(pos)).to(
                    batch.x.device
                )
                pmr = match.all().float()
                acc = match.float().mean()
                tau = kendall_tau(
                    torch.argsort(pos.squeeze()).cpu().numpy(), np.arange(len(pos))
                )

                self.metrics["accuracy"].update(acc)
                self.metrics["tau"].update(tau)
                self.metrics["pmr"].update(pmr)
            # for i in range(batch.batch.max() + 1):
            #     idx = torch.where(batch.batch == i)[0]
            #     patches_rgb = batch.patches[idx]
            #     gt_pos = batch.x[idx, :1]
            #     pos = img[idx, :1]
            #     n_patches = # batch.patches_dim[i].tolist()
            #     i_name = batch.ind_name[i]

            #     y = torch.linspace(-1, 1, n_patches[0], device=self.device)
            #     x = torch.linspace(-1, 1, n_patches[1], device=self.device)
            #     xy = torch.stack(torch.meshgrid(x, y, indexing="xy"), -1)
            #     real_grid = einops.rearrange(xy, "x y c-> (x y) c")

            #     gt_ass = greedy_cost_assignment(gt_pos, real_grid)
            #     sort_idx = torch.sort(gt_ass[:, 0])[1]
            #     gt_ass = gt_ass[sort_idx]

            #     pred_ass = greedy_cost_assignment(pos, real_grid)
            #     sort_idx = torch.sort(pred_ass[:, 0])[1]
            #     pred_ass = pred_ass[sort_idx]

            #     correct = (gt_ass[:, 1] == pred_ass[:, 1]).all()
            #     if self.rotation:
            #         pred_rot = img[idx, 2:]
            #         gt_rot = batch.x[idx, 2:]
            #         rot_correct = (
            #             torch.cosine_similarity(pred_rot, gt_rot)
            #             > math.cos(math.pi / 4)
            #         ).all()
            #         correct = correct and rot_correct

            #     # assignement = greedy_cost_assignment(gt_pos, pos)

            #     # self.num_images += 1

            #     if (
            #         self.local_rank == 0
            #         and batch_idx < 10
            #         and i < min(batch.batch.max().item(), 4)
            #     ):
            #         save_path = Path(f"results/{self.logger.experiment.name}/val")

            #         if self.rotation:
            #             self.save_image_rotated(
            #                 patches_rgb=patches_rgb,
            #                 pos=pos,
            #                 gt_pos=gt_pos,
            #                 patches_dim=n_patches,
            #                 ind_name=i_name,
            #                 file_name=save_path,
            #                 correct=correct,
            #                 gt_rotations=gt_rot,
            #                 pred_rotations=pred_rot,
            #             )
            #         else:
            #             self.save_image(
            #                 patches_rgb=patches_rgb,
            #                 pos=pos,
            #                 gt_pos=gt_pos,
            #                 patches_dim=n_patches,
            #                 ind_name=i_name,
            #                 file_name=save_path,
            #                 correct=correct,
            #             )

            #     self.metrics[f"{tuple(n_patches)}_nImages"].update(1)
            #     self.metrics["overall_nImages"].update(1)
            #     if correct:
            #         # if (assignement[:, 0] == assignement[:, 1]).all():
            #         self.metrics[f"{tuple(n_patches)}_acc"].update(1)
            #         self.metrics["overall_acc"].update(1)
            #         # accuracy_dict[tuple(n_patches)].append(1)
            #     else:
            #         self.metrics[f"{tuple(n_patches)}_acc"].update(0)
            #         self.metrics["overall_acc"].update(0)
            #         # accuracy_dict[tuple(n_patches)].append(0)

            self.log_dict(self.metrics)
        # return accuracy_dict

    def validation_epoch_end(self, outputs) -> None:
        self.log_dict(self.metrics)

    def test_epoch_end(self, outputs) -> None:
        return self.validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
        # all_outs = self.all_gather(outputs)

        ## mean = torch.mean(all_outs)
        ## num_images = all_outs.shape[0]

        # if self.local_rank == 0:
        # out_dict = {}
        # for d in all_outs:
        # out_dict = {
        # k: out_dict.get(k, []) + d.get(k, [])
        # for k in out_dict.keys() | d.keys()
        # }

        # acc_dict = {}
        # overall_acc = []
        # for key, val in out_dict.items():
        ## acc_dict[key] = {}
        # arr = torch.stack(out_dict[key])
        # acc_dict[f"{key}_acc"] = arr.float().mean()
        # acc_dict[f"{key}_num_img"] = arr.shape[0]
        # overall_acc.append(arr.float().mean())
        # overall_acc = torch.stack(overall_acc).mean()  # torch.mean(overall_acc)
        # self.log(
        # "val",
        # {"epoch": self.current_epoch, "overall_acc": overall_acc, **acc_dict},
        # rank_zero_only=True,
        # )
        # self.log("val_acc", overall_acc, rank_zero_only=True)

    # def on_validation_epoch_start(self) -> None:
    #     self.accuracy_dict = defaultdict(lambda: [])

    # def validation_step(self, batch, batch_idx, *args, **kwargs):
    # return self.test_step(batch, batch_idx, *args, **kwargs)

    def create_image_from_patches(
        self, patches, pos, n_patches=(4, 4), i=0, rotations=None
    ):
        patch_size = 32
        height = patch_size * n_patches[0]
        width = patch_size * n_patches[1]
        new_image = Image.new("RGB", (width, height))
        for p in range(patches.shape[0]):
            patch = patches[p, :]
            patch = Image.fromarray(
                ((patch.permute(1, 2, 0)) * 255).cpu().numpy().astype(np.uint8)
            )
            if rotations is not None:
                deg_angle = (
                    torch.arctan2(rotations[p, 1], rotations[p, 0]) / torch.pi * 180
                )
                patch = patch.rotate(-deg_angle)
            x = pos[p, 0] * (1 - 1 / n_patches[0])
            y = pos[p, 1] * (1 - 1 / n_patches[1])
            x_pos = int((x + 1) * width / 2) - patch_size // 2
            y_pos = int((y + 1) * height / 2) - patch_size // 2
            new_image.paste(patch, (x_pos, y_pos))
        return new_image

    def save_image(
        self,
        patches_rgb,
        pos,
        gt_pos,
        patches_dim,
        ind_name,
        file_name: Path,
        correct=None,
    ):
        file_name.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(2, 2)

        gt_img = self.create_image_from_patches(
            patches_rgb, gt_pos, n_patches=patches_dim, i=ind_name
        )
        # assignement = greedy_cost_assignment(pos, gt_pos)

        pred_img = self.create_image_from_patches(
            patches_rgb, pos, n_patches=patches_dim, i=ind_name
        )
        col = list(map(interpolate_color, gt_pos))
        ax[0, 0].imshow(gt_img)
        ax[0, 1].imshow(pred_img)
        ax[1, 0].scatter(gt_pos[:, 0].cpu(), gt_pos[:, 1].cpu(), c=col)
        ax[1, 0].invert_yaxis()
        ax[1, 0].set_aspect("equal")

        ax[1, 1].scatter(pos[:, 0].cpu(), pos[:, 1].cpu(), c=col)

        ax[1, 1].invert_yaxis()
        ax[1, 1].set_aspect("equal")
        ax[0, 0].set_title(f"{self.current_epoch}-{ind_name}- correct:{correct}")

        ax[0, 1].set_title(f"{patches_dim}")

        fig.canvas.draw()
        im = PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        im = wandb.Image(im)
        self.logger.experiment.log(
            {f"{file_name.stem}": im, "global_step": self.global_step}
        )

        plt.savefig(f"{file_name}/asd_{self.current_epoch}-{ind_name}.png")
        plt.close()

    def save_image_rotated(
        self,
        patches_rgb,
        pos,
        gt_pos,
        patches_dim,
        ind_name,
        file_name: Path,
        gt_rotations,
        pred_rotations,
        correct=None,
    ):
        file_name.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(2, 3)

        gt_img_correct = self.create_image_from_patches(
            patches_rgb,
            gt_pos,
            n_patches=patches_dim,
            i=ind_name,
            rotations=gt_rotations,
        )

        gt_img = self.create_image_from_patches(
            patches_rgb,
            gt_pos,
            n_patches=patches_dim,
            i=ind_name,
        )

        gt_img = self.create_image_from_patches(
            patches_rgb,
            gt_pos,
            n_patches=patches_dim,
            i=ind_name,
        )
        # assignement = greedy_cost_assignment(pos, gt_pos)

        pred_img = self.create_image_from_patches(
            patches_rgb,
            pos,
            n_patches=patches_dim,
            i=ind_name,
            rotations=pred_rotations,
        )
        col = list(map(interpolate_color, gt_pos))

        ax[0, 0].imshow(gt_img_correct)
        ax[0, 1].imshow(gt_img)
        ax[0, 2].imshow(pred_img)

        ax[1, 1].quiver(
            gt_pos[:, 0].cpu(),
            gt_pos[:, 1].cpu(),
            gt_rotations[:, 0].cpu(),
            gt_rotations[:, 1].cpu(),
            color=col,
            pivot="middle",
            scale=10,
            width=0.01,
        )
        ax[1, 1].invert_yaxis()
        ax[1, 1].set_aspect("equal")

        ax[1, 2].quiver(
            pos[:, 0].cpu(),
            pos[:, 1].cpu(),
            pred_rotations[:, 0].cpu(),
            pred_rotations[:, 1].cpu(),
            color=col,
            pivot="middle",
            scale=10,
            width=0.01,
        )

        ax[1, 2].invert_yaxis()
        ax[1, 2].set_aspect("equal")
        ax[0, 0].set_title(f"{self.current_epoch}-{ind_name}- correct:{correct}")

        ax[0, 1].set_title(f"{patches_dim}")

        fig.canvas.draw()
        im = PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        im = wandb.Image(im)
        self.logger.experiment.log(
            {f"{file_name.stem}": im, "global_step": self.global_step}
        )

        plt.savefig(f"{file_name}/asd_{self.current_epoch}-{ind_name}.png")
        plt.close()


def kendall_tau(order, ground_truth):
    """
    Computes the kendall's tau metric
    between the predicted sentence order and true order
    Input:
            order: list of ints denoting the predicted output order
            ground_truth: list of ints denoting the true sentence order

    Returns:
            kendall's tau - float
    """

    if len(ground_truth) == 1:
        if ground_truth[0] == order[0]:
            return 1.0

    reorder_dict = {}

    for i in range(len(ground_truth)):
        reorder_dict[ground_truth[i]] = i

    new_order = [0] * len(order)
    for i in range(len(new_order)):
        if order[i] in reorder_dict.keys():
            new_order[i] = reorder_dict[order[i]]

    corr, _ = kendalltau(new_order, list(range(len(order))))
    return corr
