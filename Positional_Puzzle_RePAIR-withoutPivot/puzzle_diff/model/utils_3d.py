import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import distance
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_apply,
    quaternion_multiply,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
)
from pytorch3d.transforms import rotation_6d_to_matrix as rot6d_to_matrix
from torch.optim.lr_scheduler import _LRScheduler

from .chamfer_distance import ChamferDistance as chamfer_dist
import math
from vedo import Points, Mesh, show, mag
from vedo.pyplot import histogram
from scipy.spatial import Delaunay
import pymeshlab
from pytorch3d.transforms import rotation_6d_to_matrix as rot6d_to_matrix
from torch.optim.lr_scheduler import _LRScheduler
import open3d as o3d
from .chamfer_distance import ChamferDistance as chamfer_dist
import math
import trimesh
from pyntcloud import PyntCloud
import pandas as pd
import sys

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """CosineLR with Warmup.

    Code borrowed from:
        https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult (float): Cycle steps magnification. Default: 1.
        max_lr (float): First cycle's max learning rate. Default: 0.1.
        min_lr (float): Min learning rate. Default: 0.001.
        warmup_steps (int): Linear warmup step size. Default: 0.
        gamma (float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps
                + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.max_lr - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.step_in_cycle - self.warmup_steps)
                        / (self.cur_cycle_steps - self.warmup_steps)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult)
                    + self.warmup_steps
                )
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(
                        math.log(
                            (
                                epoch / self.first_cycle_steps * (self.cycle_mult - 1)
                                + 1
                            ),
                            self.cycle_mult,
                        )
                    )
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps
                        * (self.cycle_mult**n - 1)
                        / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * (
                        self.cycle_mult**n
                    )
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class Rotation3D:
    """Class for different 3D rotation representations, all util functions,
    model input-output will adopt this as the general interface.

    Supports common properties of torch.Tensor, e.g. shape, device, dtype.
    Also support indexing and slicing which are done to torch.Tensor.

    Currently, we support three rotation representations:
    - quaternion: (..., 4), real part first, unit quaternion
    - rotation matrix: (..., 3, 3), the input `rot` can be either rmat or 6D
        For 6D, it could be either (..., 6) or (..., 2, 3)
        Note that, we will convert it to 3x3 matrix in the constructor
    - axis-angle: (..., 3)
    - euler angles: this is NOT supported as a representation, but we can
        convert from supported representations to euler angles
    """

    ROT_TYPE = ["quat", "rmat", "axis"]
    ROT_NAME = {
        "quat": "quaternion",
        "rmat": "matrix",
        "axis": "axis_angle",
    }

    def __init__(self, rot, rot_type="quat"):
        self._rot = rot
        self._rot_type = rot_type

        self._check_valid()

    def _process_zero_quat(self):
        """Convert zero-norm quat to (1, 0, 0, 0)."""
        with torch.no_grad():
            norms = torch.norm(self._rot, p=2, dim=-1, keepdim=True)
            new_rot = torch.zeros_like(self._rot)
            new_rot[..., 0] = 1.0  # zero quat
            valid_mask = (norms.abs() > 0.5).repeat_interleave(4, dim=-1)
        self._rot = torch.where(valid_mask, self._rot, new_rot)

    def _normalize_quat(self):
        """Normalize quaternion."""
        self._rot = F.normalize(self._rot, p=2, dim=-1)

    def _check_valid(self):
        """Check the shape of rotation."""
        assert (
            self._rot_type in self.ROT_TYPE
        ), f"rotation {self._rot_type} is not supported"
        assert isinstance(self._rot, torch.Tensor), "rotation must be a tensor"
        # let's always make rotation in float32
        # otherwise quat won't be unit, and rmat won't be orthogonal
        self._rot = self._rot.float()
        if self._rot_type == "quat":
            assert self._rot.shape[-1] == 4, "wrong quaternion shape"
            # quat with norm == 0 are padded, make them (1, 0, 0, 0)
            # because (0, 0, 0, 0) convert to rmat will cause PyTorch bugs
            self._process_zero_quat()
            # too expensive to check
            # self._normalize_quat()
            # assert _is_normalized(self._rot, dim=-1), 'quaternion is not unit'
        elif self._rot_type == "rmat":
            if self._rot.shape[-1] == 3:
                if self._rot.shape[-2] == 3:  # 3x3 matrix
                    # assert _is_orthogonal(self._rot)
                    pass  # too expensive to check
                elif self._rot.shape[-2] == 2:  # 6D representation
                    # (2, 3): (b1, b2), rot6d_to_matrix will calculate b3
                    # and stack them vertically
                    self._rot = rot6d_to_matrix(self._rot.flatten(-2, -1))
                else:
                    raise ValueError("wrong rotation matrix shape")
            elif self._rot.shape[-1] == 6:  # 6D representation
                # this indeed doing `rmat = torch.stack((b1, b2, b3), dim=-2)`
                self._rot = rot6d_to_matrix(self._rot)
            else:
                raise NotImplementedError("wrong rotation matrix shape")
        else:  # axis-angle
            assert self._rot.shape[-1] == 3

    def apply_rotation(self, rot):
        """Apply `rot` to the current rotation, left multiply."""
        assert rot.rot_type in ["quat", "rmat"]
        rot = rot.convert(self._rot_type)
        if self._rot_type == "quat":
            new_rot = quaternion_multiply(rot.rot, self._rot)
        else:
            new_rot = rot.rot @ self._rot
        return self.__class__(new_rot, self._rot_type)

    def convert(self, rot_type):
        """Convert to a different rotation type."""
        assert rot_type in self.ROT_TYPE, f"unknown target rotation {rot_type}"
        src_type = self.ROT_NAME[self._rot_type]
        dst_type = self.ROT_NAME[rot_type]
        if src_type == dst_type:
            return self.clone()
        new_rot = eval(f"{src_type}_to_{dst_type}")(self._rot)
        return self.__class__(new_rot, rot_type)

    def to_quat(self):
        """Convert to quaternion and return the tensor."""
        return self.convert("quat").rot

    def to_rmat(self):
        """Convert to rotation matrix and return the tensor."""
        return self.convert("rmat").rot

    def to_axis_angle(self):
        """Convert to axis angle and return the tensor."""
        return self.convert("axis").rot

    def to_euler(self, order="zyx", to_degree=True):
        """Compute to euler angles and return the tensor."""
        quat = self.convert("quat")
        return qeuler(quat._rot, order=order, to_degree=to_degree)

    @property
    def rot(self):
        return self._rot

    @rot.setter
    def rot(self, rot):
        self._rot = rot
        self._check_valid()

    @property
    def rot_type(self):
        return self._rot_type

    @rot_type.setter
    def rot_type(self, rot_type):
        raise NotImplementedError("please use convert() for rotation type conversion")

    @property
    def shape(self):
        return self._rot.shape

    def reshape(self, *shape):
        return self.__class__(self._rot.reshape(*shape), self._rot_type)

    def view(self, *shape):
        return self.__class__(self._rot.view(*shape), self._rot_type)

    def squeeze(self, dim=None):
        return self.__class__(self._rot.squeeze(dim), self._rot_type)

    def unsqueeze(self, dim=None):
        return self.__class__(self._rot.unsqueeze(dim), self._rot_type)

    def flatten(self, *args, **kwargs):
        return self.__class__(self._rot.flatten(*args, **kwargs), self._rot_type)

    def unflatten(self, *args, **kwargs):
        return self.__class__(self._rot.unflatten(*args, **kwargs), self._rot_type)

    def transpose(self, *args, **kwargs):
        return self.__class__(self._rot.transpose(*args, **kwargs), self._rot_type)

    def permute(self, *args, **kwargs):
        return self.__class__(self._rot.permute(*args, **kwargs), self._rot_type)

    def contiguous(self):
        return self.__class__(self._rot.contiguous(), self._rot_type)

    @staticmethod
    def cat(rot_lst, dim=0):
        """Concat a list a Rotation3D object."""
        assert isinstance(rot_lst, (list, tuple))
        assert all([isinstance(rot, Rotation3D) for rot in rot_lst])
        rot_type = rot_lst[0].rot_type
        assert all([rot.rot_type == rot_type for rot in rot_lst])
        rot_lst = [rot.rot for rot in rot_lst]
        return Rotation3D(torch.cat(rot_lst, dim=dim), rot_type)

    @staticmethod
    def stack(rot_lst, dim=0):
        """Stack a list of Rotation3D object."""
        assert isinstance(rot_lst, (list, tuple))
        assert all([isinstance(rot, Rotation3D) for rot in rot_lst])
        rot_type = rot_lst[0].rot_type
        assert all([rot.rot_type == rot_type for rot in rot_lst])
        rot_lst = [rot.rot for rot in rot_lst]
        return Rotation3D(torch.stack(rot_lst, dim=dim), rot_type)

    def __getitem__(self, key):
        return self.__class__(self._rot[key], self._rot_type)

    def __len__(self):
        return self._rot.shape[0]

    @property
    def device(self):
        return self._rot.device

    def to(self, device):
        return self.__class__(self._rot.to(device), self._rot_type)

    def cuda(self, device=None):
        return self.__class__(self._rot.cuda(device), self._rot_type)

    @property
    def dtype(self):
        return self._rot.dtype

    def type(self, dtype):
        return self.__class__(self._rot.type(dtype), self._rot_type)

    def type_as(self, other):
        return self.__class__(self._rot.type_as(other), self._rot_type)

    def detach(self):
        return self.__class__(self._rot.detach(), self._rot_type)

    def clone(self):
        return self.__class__(self._rot.clone(), self._rot_type)


@torch.no_grad()
def trans_metrics(trans1, trans2, metric="rmse"):
    """Evaluation metrics for transformation.

    Metrics used in the NSM paper.

    Args:
        trans1: [B, P, 3]
        trans2: [B, P, 3]
        metric: str, 'mse', 'rmse' or 'mae'

    Returns:
        [B], metric per data in the batch
    """
    assert metric in ["mse", "rmse", "mae"]
    if metric == "mse":
        metric_per_data = (trans1 - trans2).pow(2).mean(dim=-1)  # [B, P]
    elif metric == "rmse":
        metric_per_data = (trans1 - trans2).pow(2).mean(dim=-1) ** 0.5
    else:
        metric_per_data = (trans1 - trans2).abs().mean(dim=-1)
    metric_per_data = metric_per_data.mean()  # [B]
    return metric_per_data


@torch.no_grad()
def test_loss(cond, pred_t, gt_t, pred_r, gt_r, training):
    trans_loss_w = 1.0
    rot_pt_cd_loss_w = 10.0
    transform_pt_cd_loss_w = 10.0
    # cosine regression loss on rotation
    rot_loss_w = 0.2
    # per-point l2 loss between rotated part point clouds
    rot_pt_l2_loss_w = 1.0
    trans_loss = trans_l2_loss(pred_t, gt_t)
    rot_pt_cd_loss = rot_points_cd_loss(cond, pred_r, gt_r)
    transform_pt_cd_loss = shape_cd_loss(
        cond, pred_t, gt_t, pred_r, gt_r, training=training
    )
    # rot_loss = rot_cosine_loss(pred_r, gt_r)
    rot_pt_l2_loss = rot_points_l2_loss(cond, pred_r, gt_r, valids, n_batch)

    # loss = F.smooth_l1_loss(target, prediction)
    loss = (
        trans_loss * trans_loss_w
        + rot_pt_cd_loss * rot_pt_cd_loss_w
        + transform_pt_cd_loss * transform_pt_cd_loss_w
        + rot_pt_l2_loss * rot_pt_l2_loss_w
    )
    # + rot_loss * rot_loss_w + \
    return loss.mean(dim=-1)


@torch.no_grad()
def rot_metrics(rot1, rot2, metric="rmse"):
    """Evaluation metrics for rotation in euler angle (degree) space.

    Metrics used in the NSM paper.

    Args:
        rot1: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        rot2: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        valids: [B, P], 1 for input parts, 0 for padded parts
        metric: str, 'mse', 'rmse' or 'mae'

    Returns:
        [B], metric per data in the batch
    """
    assert metric in ["mse", "rmse", "mae", "geodesic"]
    rot1 = Rotation3D(rot1)
    rot2 = Rotation3D(rot2)

    deg1 = rot1.to_euler(to_degree=True)  # [B, P, 3]
    deg2 = rot2.to_euler(to_degree=True)
    diff1 = (deg1 - deg2).abs()
    diff2 = 360.0 - (deg1 - deg2).abs()
    # since euler angle has the discontinuity at 180
    diff = torch.minimum(diff1, diff2)
    if metric == "mse":
        metric_per_data = diff.pow(2).mean(dim=-1)  # [B, P]
    elif metric == "rmse":
        metric_per_data = diff.pow(2).mean(dim=-1) ** 0.5
    elif metric == "geodesic":
        metric_per_data = geodesic_distance(rot1, rot2)
    else:
        metric_per_data = diff.abs().mean(dim=-1)
    metric_per_data = metric_per_data.mean()
    return metric_per_data


@torch.no_grad()
def _is_normalized(mat, dim=-1):
    """
    Check if one dim of a matrix is normalized.
    """
    norm = torch.norm(mat, p=2, dim=dim)
    return (norm - 1.0).abs().max() < EPS


@torch.no_grad()
def _is_orthogonal(mat):
    """
    Check if a matrix (..., 3, 3) is orthogonal.
    """
    mat = mat.view(-1, 3, 3)
    iden = torch.eye(3).unsqueeze(0).repeat(mat.shape[0], 1, 1).type_as(mat)
    mat = torch.bmm(mat, mat.transpose(1, 2))
    return (mat - iden).abs().max() < EPS


def qeuler(q, order, epsilon=0, to_degree=False):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == "xyz":
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == "yzx":
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == "zxy":
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == "xzy":
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == "yxz":
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == "zyx":
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise

    euler = torch.stack((x, y, z), dim=1).view(original_shape)
    if to_degree:
        euler = euler * 180.0 / np.pi
    return euler


import torch


def rot_pc(rot, pc, rot_type=None):
    """Rotate the 3D point cloud.

    If `rot_type` is specified, `rot` is torch.Tensor. Otherwise, it is a
        Rotation object and the type will be inferred from it.

    Args:
        rot (Rotation3D or torch.Tensor): quat and rmat are supported now.
    """
    if rot_type is None:
        assert isinstance(rot, Rotation3D)
        r = rot.rot
        rot_type = rot.rot_type
    else:
        assert isinstance(rot, torch.Tensor)
        r = rot
    if rot_type == "quat":
        return qrot(r, pc)
    # elif rot_type == 'rmat':
    #    return rmat_rot(r, pc)
    else:
        raise NotImplementedError(f"{rot.rot_type} is not supported")


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
        where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    # repeat to e.g. apply the same quat for all points in a point cloud
    # [4] --> [N, 4], [B, 4] --> [B, N, 4], [B, P, 4] --> [B, P, N, 4]
    if len(q.shape) == len(v.shape) - 1:
        q = q.unsqueeze(-2).repeat_interleave(v.shape[-2], dim=-2)
    assert q.shape[:-1] == v.shape[:-1]
    return quaternion_apply(q, v)


def qtransform(t, q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q,
        and then translate it by the translation described by t.
    Expects a tensor of shape (*, 3) for t, a tensor of shape (*, 4) for q and
        a tensor of shape (*, 3) for v, where * denotes any dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert t.shape[-1] == 3

    # repeat to e.g. apply the same trans for all points in a point cloud
    # [3] --> [N, 3], [B, 3] --> [B, N, 3], [B, P, 3] --> [B, P, N, 3]
    if len(t.shape) == len(v.shape) - 1:
        t = t.unsqueeze(-2).repeat_interleave(v.shape[-2], dim=-2)

    assert t.shape == v.shape

    qv = qrot(q, v)
    tqv = qv + t
    return tqv


def _valid_mean(loss_per_part, valids):
    """Average loss values according to the valid parts.

    Args:
        loss_per_part: [B, P]
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], loss per data in the batch, averaged over valid parts
    """
    valids = valids.float().detach()
    loss_per_data = (loss_per_part * valids).sum(1) / valids.sum(1)
    return loss_per_data


def transform_pc(trans, rot, pc, rot_type=None):
    """Rotate and translate the 3D point cloud.

    If `rot_type` is specified, `rot` is torch.Tensor. Otherwise, it is a
        Rotation object and the type will be inferred from it.

    Args:
        rot (Rotation3D or torch.Tensor): quat and rmat are supported now.
    """
    if rot_type is None:
        assert isinstance(rot, Rotation3D)
        r = rot.rot
        rot_type = rot.rot_type
    else:
        assert isinstance(rot, torch.Tensor)
        r = rot
    if rot_type == "quat":
        return qtransform(trans, r, pc)
    # elif rot_type == 'rmat':
    #    return rmat_transform(trans, r, pc)
    else:
        raise NotImplementedError(f"{rot_type} is not supported")


def rot_cosine_loss(rot1, rot2, valids, n_batch, n_parts=20):
    """Cosine loss for rotation.

    Args:
        rot1: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        rot2: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], loss per data in the batch
    """

    # if n_batch:
    # breakpoint()
    valid_mask = valids.reshape(n_batch, n_parts)  # [B, P]
    rot12 = torch.zeros(n_batch, n_parts, 4).type_as(rot1)
    rot12[valid_mask] = rot1

    rot21 = torch.zeros(n_batch, n_parts, 4).type_as(rot2)
    rot21[valid_mask] = rot2
    #    shape2 = pts21.flatten(1, 2)

    rot12 = Rotation3D(rot12)
    rot21 = Rotation3D(rot21)
    # assert rot1.rot_type == rot2.rot_type
    rot_type = rot12.rot_type

    # cosine distance
    if rot_type == "quat":
        quat1, quat2 = rot12.rot, rot21.rot
        loss_per_data = 1.0 - torch.abs(torch.sum(quat1 * quat2, dim=-1))
        loss_per_data = _valid_mean(loss_per_data, valids.reshape(n_batch, n_parts))
        # print(loss_per_data.mean())

        # rot1 = Rotation3D(rot1)
        # rot2 = Rotation3D(rot2)
        # quat1, quat2 = rot1.rot, rot2.rot
        # loss_per_data = 1. - torch.abs(torch.sum(quat1 * quat2, dim=-1))
        # valids = valids.reshape(n_batch, n_parts).float().detach()
        # loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)
        # print(loss_per_data.mean())
        # exit()
    # |I - R1^T @ R2|^2
    elif rot_type == "rmat":
        B = rot1.shape[0]
        rmat1, rmat2 = rot1.rot.view(-1, 3, 3), rot2.rot.view(-1, 3, 3)
        iden = torch.eye(3).unsqueeze(0).type_as(rmat1)
        loss_per_data = (
            (iden - torch.bmm(rmat1.transpose(1, 2), rmat2))
            .pow(2)
            .mean(dim=[-1, -2])
            .view(B, -1)
        )
    else:
        raise NotImplementedError(f"cosine loss not supported for {rot_type}")
    return loss_per_data


def rot_points_l2_loss(pts, rot1, rot2, valids, n_batch, n_parts=20, ret_pts=False):
    """L2 distance between point clouds transformed by rotations.

    Args:
        pts: [B, P, N, 3], model input point cloud to be transformed
        rot1: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        rot2: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        valids: [B, P], 1 for input parts, 0 for padded parts
        ret_pts: whether to return the rotated point clouds

    Returns:
        [B], loss per data in the batch
    """
    rot1 = Rotation3D(rot1)
    rot2 = Rotation3D(rot2)

    pts1 = rot_pc(rot1, pts)
    pts2 = rot_pc(rot2, pts)

    valid_mask = valids.reshape(n_batch, n_parts)  # [B, P]
    pts12 = torch.zeros(n_batch, n_parts, 1000, 3).type_as(pts1)
    pts12[valid_mask] = pts1

    pts21 = torch.zeros(n_batch, n_parts, 1000, 3).type_as(pts2)
    pts21[valid_mask] = pts2
    # pts1 = rot1.apply_rotation(pts)
    # pts2 = rot2.apply_rotation(pts)

    loss_per_data = (pts12 - pts21).pow(2).sum(-1).mean(-1)  # type: ignore # [B, P]
    loss_per_data = _valid_mean(loss_per_data, valids.reshape(n_batch, n_parts))

    if ret_pts:
        return loss_per_data, pts1, pts2
    return loss_per_data


def rot_points_cd_loss(
    pts, rot1, rot2, ret_pts=False, n_parts=20, n_batch=None, valids=None
):
    """Chamfer distance between point clouds transformed by rotations.

    Args:
        pts: [B, P, N, 3], model input point cloud to be transformed
        rot1: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        rot2: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        valids: [B, P], 1 for input parts, 0 for padded parts
        ret_pts: whether to return the rotated point clouds

    Returns:
        [B], loss per data in the batch
    """
    B = pts.shape[0]

    rot1 = Rotation3D(rot1)
    rot2 = Rotation3D(rot2)
    pts1 = rot_pc(rot1, pts)
    pts2 = rot_pc(rot2, pts)

    valid_mask = valids.reshape(n_batch, n_parts)  # [B, P]
    pts12 = torch.zeros(n_batch, n_parts, 1000, 3).type_as(pts1)
    pts12[valid_mask] = pts1
    # shape1 = pts12.flatten(0, 1)  # [B, P*N, 3]S
    pts21 = torch.zeros(n_batch, n_parts, 1000, 3).type_as(pts2)
    pts21[valid_mask] = pts2
    # shape2 = pts21.flatten(0, 1)

    dist1, dist2, _, _ = chamfer_dist()(
        pts12.flatten(0, 1), pts21.flatten(0, 1)
    )  # copy the repository
    loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
    loss_per_data = loss_per_data.view(n_batch, -1).type_as(pts)  # [B, P]
    loss_per_data = _valid_mean(loss_per_data, valids.reshape(n_batch, n_parts))

    # dist1, dist2, _, _ = chamfer_dist()(pts1, pts2) # copy the repository
    # breakpoint()
    # loss_per_data = (torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)) / n_parts

    # loss_per_data = (torch.mean(dist1)) + (torch.mean(dist2))
    # loss_per_data = loss_per_data.view(B, -1).type_as(pts)  # [B, P]
    # loss_per_data = _valid_mean(loss_per_data, valids)

    if ret_pts:
        return loss_per_data, pts1, pts2
    return loss_per_data  # media gia' data


def shape_cd_loss(
    pts,
    trans1,
    trans2,
    rot1,
    rot2,
    ret_pts=False,
    n_parts=20,
    training=True,
    n_batch=None,
    valids=None,
):
    """Chamfer distance between point clouds after rotation and translation.

    Args:
        pts: [B, P, N, 3], model input point cloud to be transformed
        trans1: [B, P, 3]
        trans2: [B, P, 3]
        rot1: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        rot2: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        valids: [B, P], 1 for input parts, 0 for padded parts
        ret_pts: whether to return the transformed point clouds
        training: at training time we divide the SCD by `P` as an automatic
            hard negative mining strategy; while at test time we divide by
            the correct number of parts per shape

    Returns:
        [B], loss per data in the batch
    """
    # BP, N, C = pts.shape

    # fill the padded points with very large numbers
    # so that they won't be matched to any point in CD
    # clone the points to avoid changing the original points
    pts = pts.detach().clone()
    # valid_mask = valids[..., None, None]  # [B, P, 1, 1]
    # pts = pts.masked_fill(valid_mask == 0, 1e3)

    rot1 = Rotation3D(rot1)
    rot2 = Rotation3D(rot2)

    pts1 = transform_pc(trans1, rot1, pts)
    pts2 = transform_pc(trans2, rot2, pts)

    if n_batch:
        valid_mask = valids.reshape(n_batch, n_parts)  # [B, P]
        pts12 = torch.ones(n_batch, n_parts, 1000, 3).type_as(pts1) * 1e3
        pts12[valid_mask] = pts1
        shape1 = pts12.flatten(1, 2)  # [B, P*N, 3]

        pts21 = torch.ones(n_batch, n_parts, 1000, 3).type_as(pts2) * 1e3
        pts21[valid_mask] = pts2
        shape2 = pts21.flatten(1, 2)

    else:
        shape1 = pts1  # .flatten(1, 2)  # [B, P*N, 3]
        shape2 = pts2  # .flatten(1, 2)
    dist1, dist2, _, _ = chamfer_dist()(
        shape1, shape2
    )  # chamfer_distance(shape1, shape2)  # [B, P*N]
    if n_batch:
        valids = valids.reshape(n_batch, n_parts).float().detach()
        valids = valids.unsqueeze(2).repeat(1, 1, 1000).view(n_batch, -1)
        dist1 = dist1 * valids
        dist2 = dist2 * valids
        # we divide the loss by a fixed number `P`
        # this is actually an automatic hard negative loss weighting mechanism
        # shapes with more parts will have higher loss
        # ablation shows better results than using the correct SCD for training

        # if not training:
        #    loss_per_data = (dist1 + dist2).mean(-1) # da implementare been
        # else:
        loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
    else:
        if not training:
            loss_per_data = (dist1 + dist2).mean(-1)
        else:
            loss_per_data = torch.mean(dist1 / n_parts, dim=1) + torch.mean(
                dist2 / n_parts, dim=1
            )

    # this is the correct SCD calculation
    # else:
    # valids = valids.float().detach()
    # dist = (dist1 + dist2).view(B, P, N).mean(-1)  # [B, P]
    # loss_per_data = _valid_mean(dist, valids)
    # if not training:
    #    loss_per_data = (dist1 + dist2).mean(-1)
    if ret_pts:
        return loss_per_data, pts1, pts2
    return loss_per_data


def trans_l2_loss(trans1, trans2, n_batch=None, valids=None, n_parts=20):
    """L2 loss for transformation.

    Args:
        trans1: [B, P, 3]
        trans2: [B, P, 3]
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], loss per data in the batch
    """
    valid_mask = valids.reshape(n_batch, n_parts)  # [B, P]
    trans12 = torch.zeros(n_batch, n_parts, 3).type_as(trans1)
    trans12[valid_mask] = trans1

    trans21 = torch.zeros(n_batch, n_parts, 3).type_as(trans2)
    trans21[valid_mask] = trans2

    loss_per_data = (trans12 - trans21).pow(2).sum(dim=-1)  # [B, P]
    loss_per_data = _valid_mean(loss_per_data, valids.reshape(n_batch, n_parts))

    # valids = valids.reshape(n_batch, n_parts).float().detach()
    # loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)
    # print(loss_per_data.mean())

    # loss_per_data = (trans1 - trans2).pow(2).sum(dim=-1)  # [B, P]
    # print(loss_per_data.mean())

    return loss_per_data


def rot_l2_loss(rot1, rot2):
    """L2 loss for rotation.

    Args:
        rot1: [B, P, 4], Rotation3D, should be quat
        rot2: [B, P, 4], Rotation3D, should be quat
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], loss per data in the batch
    """
    rot1 = Rotation3D(rot1)
    rot2 = Rotation3D(rot2)
    assert rot1.rot_type == rot2.rot_type == "quat"
    quat1, quat2 = rot1.rot, rot2.rot
    # since quat == -quat
    rot_l2_1 = (quat1 - quat2).pow(2).sum(dim=-1)  # [B, P]
    rot_l2_2 = (quat1 + quat2).pow(2).sum(dim=-1)
    loss_per_data = torch.minimum(rot_l2_1, rot_l2_2)

    return loss_per_data  # media da mettere


def geodesic_distance(rot1, rot2):
    """Compute geodesic distance between two rotations.

    Args:
        rot1: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        rot2: [B, P, 4/(3, 3)], Rotation3D, quat or rmat

    Returns:
        [B, P], geodesic distance per data and part
    """
    if isinstance(rot1, Rotation3D):
        pass
    elif rot1.shape[-1] == 3:
        rot1 = Rotation3D(rot1, "rmat")
    else:
        rot1 = Rotation3D(rot1, "quat")
    if isinstance(rot2, Rotation3D):
        pass
    elif rot2.shape[-1] == 3:
        rot2 = Rotation3D(rot2, "rmat")
    else:
        rot2 = Rotation3D(rot2, "quat")
    R1 = rot1.to_rmat()  # .reshape(B*P, 3, 3)
    R2 = rot2.to_rmat()  # .reshape(B*P, 3, 3)
    Rds = torch.bmm(R1.permute(0, 2, 1), R2)
    Rt = torch.sum(Rds[:, torch.eye(3).bool()], 1)  # batch trace
    # necessary or it might lead to nans and the likes
    theta = torch.clamp(0.5 * (Rt - 1), -1 + 1e-6, 1 - 1e-6)
    return torch.acos(theta)  # .reshape(B, P)


def loss_recons(recons_pts, part_pts, rot1, trans1):
    # recons_pts = pred_data["recon_pts"]  # (batch_size, num_parts, N, 3)
    # whole_recon_pcs = pred_data["whole_recon_pcs"]  # (batch_size, N, 3)
    # part_pcs = pts #batch_data["part_pcs"]
    # B, P, N, _ = part_pcs.shape  # [B, P, N, 3]
    # recon_pcs = recons_pts.reshape(B * P, N, 3)

    # part_valids = batch_data["part_valids"]

    # Ground truths
    rot_gt = Rotation3D(rot1)  # batch_data["part_rot"].to_rmat()
    trans_gt = trans1  # batch_data["part_trans"].float()

    transformed_pc_gt = transform_pc(
        trans_gt, rot_gt, part_pcs
    )  # .reshape(B * P, N, 3)
    dist1, dist2 = chamfer_dist()(transformed_pc_gt, recon_pcs)
    part_cham_loss = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
    part_cham_loss = part_cham_loss.view(B, -1).type_as(recon_pcs)  # [B, P]
    # part_cham_loss = _valid_mean(part_cham_loss, part_valids)
    # whole_pc_gt = transformed_pc_gt.reshape(B, P, N, 3).reshape(B, -1, 3)
    # dist3, dist4 = chamfer_distance(whole_pc_gt, whole_recon_pcs)
    # part_cham_loss = part_cham_loss.view(B, -1).type_as(recon_pcs)  # [B, P]
    # part_cham_loss = _valid_mean(part_cham_loss, part_valids)
    # whole_cham_loss = torch.mean(dist3, dim=1) + torch.mean(dist4, dim=1)  # [B]
    # cham_loss = part_cham_loss + whole_cham_loss
    return part_cham_loss


def orthogonalise(mat):
    """Orthogonalise rotation/affine matrices

    Ideally, 3D rotation matrices should be orthogonal,
    however during creation, floating point errors can build up.
    We SVD decompose our matrix as in the ideal case S is a diagonal matrix of 1s
    We then round the values of S to [-1, 0, +1],
    making U @ S_rounded @ V.T an orthonormal matrix close to the original.
    """
    orth_mat = mat.clone()
    u, s, v = torch.svd(mat[..., :3, :3])
    orth_mat[..., :3, :3] = u @ torch.diag_embed(s.round()) @ v.transpose(-1, -2)
    return orth_mat


def skew2vec(skew: torch.Tensor) -> torch.Tensor:
    vec = torch.zeros_like(skew[..., 0])
    vec[..., 0] = skew[..., 2, 1]
    vec[..., 1] = -skew[..., 2, 0]
    vec[..., 2] = skew[..., 1, 0]
    return vec


def vec2skew(vec: torch.Tensor) -> torch.Tensor:
    skew = torch.repeat_interleave(torch.zeros_like(vec).unsqueeze(-1), 3, dim=-1)
    skew[..., 2, 1] = vec[..., 0]
    skew[..., 2, 0] = -vec[..., 1]
    skew[..., 1, 0] = vec[..., 2]
    return skew - skew.transpose(-1, -2)


# See paper
# Exponentials of skew-symmetric matrices and logarithms of orthogonal matrices
# https://doi.org/10.1016/j.cam.2009.11.032
# For most of the derivatons here


# We use atan2 instead of acos here dut to better numerical stability.
# it means we get nicer behaviour around 0 degrees
# More effort to derive sin terms
# but as we're dealing with small angles a lot,
# the tradeoff is worth it.
def log_rmat(r_mat: torch.Tensor) -> torch.Tensor:
    skew_mat = r_mat - r_mat.transpose(-1, -2)  # costruisco la amtrice skew
    sk_vec = skew2vec(skew_mat)
    s_angle = (sk_vec).norm(p=2, dim=-1) / 2
    c_angle = (torch.einsum("...ii", r_mat) - 1) / 2
    angle = torch.atan2(s_angle, c_angle)  # trovo l'angolo di rotazione [B]
    scale = angle / (2 * s_angle)  # [B]
    # if s_angle = 0, i.e. rotation by 0 or pi (180), we get NaNs
    # by definition, scale values are 0 if rotating by 0.
    # This also breaks down if rotating by pi, fix further down
    scale[angle == 0.0] = 0.0
    log_r_mat = scale[..., None, None] * skew_mat  # [B x 3 x 3]

    # Check for NaNs caused by 180deg rotations.
    nanlocs = log_r_mat[..., 0, 0].isnan()
    nanmats = r_mat[nanlocs]
    # We need to use an alternative way of finding the logarithm for nanmats,
    # Use eigendecomposition to discover axis of rotation.
    # By definition, these are symmetric, so use eigh.
    # NOTE: linalg.eig() isn't in torch 1.8,
    #       and torch.eig() doesn't do batched matrices
    eigval, eigvec = torch.linalg.eigh(nanmats)
    # Final eigenvalue == 1, might be slightly off because floats, but other two are -ve.
    # this *should* just be the last column if the docs for eigh are true.
    nan_axes = eigvec[..., -1, :]
    nan_angle = angle[nanlocs]
    nan_skew = vec2skew(nan_angle[..., None] * nan_axes)
    log_r_mat[nanlocs] = nan_skew
    return log_r_mat


def so3_scale(rmat, scalars):
    """Scale the magnitude of a rotation matrix,
    e.g. a 45 degree rotation scaled by a factor of 2 gives a 90 degree rotation.

    This is the same as taking matrix powers, but pytorch only supports integer exponents

    So instead, we take advantage of the properties of rotation matrices
    to calculate logarithms easily. and multiply instead.
    """
    logs = log_rmat(rmat)
    scaled_logs = logs * scalars[..., None, None]
    out = torch.matrix_exp(scaled_logs)
    return out


def skew_to_rmat(vmat, check=False):
    """ """
    rmat = vec2skew(vmat)
    out = torch.matrix_exp(rmat)
    if not check:
        return out
    else:
        return orthogonalise(out)


def projection(x, data):
    # The transpose operation here here is due to the shape of self.data.
    # (A^T)^T = A
    # (AB)^T = B^T A^T
    # So for rotation R and data D:
    # (RD^T)^T = (D^T)^T R^T = D R^T
    R_T = x.transpose(-1, -2)
    return data @ R_T


def translation(x, data):
    return data + x[None].permute(1, 0, 2)

def precision(adj_pred, adj_true, areas_matrix):

    both = torch.tensor(np.logical_and(adj_pred, adj_true))
    both_areas = torch.sum(both * areas_matrix)
    true_areas = torch.sum(adj_true * areas_matrix)
    return both_areas / true_areas if true_areas > 0 else 0

def recall(adj_pred, adj_true, areas_matrix):
    both = torch.tensor(np.logical_and(adj_pred, adj_true))
    both_areas = torch.sum(both * areas_matrix)
    pred_areas = torch.sum(adj_pred * areas_matrix)
    return both_areas / pred_areas if pred_areas > 0 else 0
    
def f1(adj_pred, adj_true, areas_matrix):
    _prescision = precision(adj_pred, adj_true, areas_matrix)
    _recall = recall(adj_pred, adj_true, areas_matrix)
    return 2 * _prescision * _recall / (_prescision + _recall) if _prescision + _recall > 0 else 0




def create_mesh_from_points(pcd):
    # Create a Delaunay triangulation of the points
    #tri = Delaunay(points_array)
    # Create a Mesh from the triangulated points
    #mesh = Mesh([points_array, tri.simplices])
    #return Points.generate_mesh(points_array)
    # points = Points(points_array)
    # return pcd.reconstruct_surface(dims=100, radius=0.02)
    
    dists1 = []
    for p1 in pcd.coordinates:
        q1 = pcd.closest_point(p1, n=2)[1]
        dists1.append(mag(p1 - q1))
    histo1 = histogram(dists1, bins=25).clone2d()
    radius = histo1.mean * 10
    
    m = pcd.generate_delaunay3d(radius=radius)
    
    return m.tomesh().compute_normals()
    
    # return fill_holes(m.tomesh().compute_normals())

def compute_volume_value(vol1):
    dx1, dy1, dz1 = vol1.spacing() # voxel size
    counts1 = np.unique(vol1.pointdata[0], return_counts=True)
    n01, n11 = counts1[1]
    vol_value1 = dx1*dy1*dz1 * n11
    return vol_value1

def vedo2pymesh(vd_mesh):

    # m = pymeshlab.Mesh(vertex_matrix=vd_mesh.points(), face_matrix=vd_mesh.faces(), v_normals_matrix=vd_mesh.pointdata["Normals"], v_color_matrix=np.insert(vd_mesh.pointdata["RGB"]/255, 3, 1, axis=1))
    # vd_pcd.compute_normals_with_pca()
    m = pymeshlab.Mesh(vertex_matrix=vd_mesh.vertices, face_matrix=vd_mesh.faces(), v_normals_matrix=vd_mesh.point_normals)

    ms = pymeshlab.MeshSet()
    ms.add_mesh(m)

    return ms

def pymesh2vedo(mlab_mesh):
    # color = mlab_mesh.vertex_color_matrix()[:, 0:-1]
    reco_mesh = Mesh(mlab_mesh)
    # reco_mesh.pointdata["RGB"] = (color * 255).astype(np.uint8)
    # reco_mesh.pointdata["Normals"] = mlab_mesh.vertex_normal_matrix().astype(np.float32)
    # reco_mesh.pointdata.select("RGB")

    return reco_mesh
    
def create_mesh(mesh_):
    m = o3d2pymesh(mesh_)
    m.generate_surface_reconstruction_screened_poisson(depth=8, pointweight=1, preclean=True)

    mlab_mesh = m.current_mesh()

    reco_mesh = pymesh2o3d(mlab_mesh)

    return reco_mesh


def o3d2vedo(o3d_mesh):
    m = Mesh([np.array(o3d_mesh.vertices), np.array(o3d_mesh.triangles)])

    # you could also check whether normals and color are present in order to port with the above vertices/faces
    return m

def vedo2open3d(vd_mesh):
    """
    Return an `open3d.geometry.TriangleMesh` version of
    the current mesh.

    Returns
    ---------
    open3d : open3d.geometry.TriangleMesh
      Current mesh as an open3d object.
    """
    # create from numpy arrays
    o3d_mesh = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(vd_mesh.vertices))

    # I need to add some if check here in case color and normals info are not existing
    # o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vd_mesh.pointdata["RGB"]/255)
    # o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(vd_mesh.pointdata["Normals"])

    return o3d_mesh

def o3d2pymesh(o3d_mesh):
    m = pymeshlab.Mesh(vertex_matrix=np.array(o3d_mesh.vertices), face_matrix=np.array(o3d_mesh.triangles),
                       v_normals_matrix=np.array(o3d_mesh.vertex_normals))#, v_color_matrix=np.insert(np.array(o3d_mesh.vertex_colors), 3, 1, axis=1))

    ms = pymeshlab.MeshSet()
    ms.add_mesh(m)

    return ms

def pymesh2o3d(pymesh_):
    # create from numpy arrays
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(pymesh_.vertex_matrix()),
        triangles=o3d.utility.Vector3iVector(pymesh_.face_matrix()))

    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(pymesh_.vertex_color_matrix()[:, 0:-1])
    o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(pymesh_.vertex_normal_matrix())

    return o3d_mesh

def fill_holes(mesh_):
    m = vedo2pymesh(mesh_)
    m.meshing_close_holes(maxholesize=30, newfaceselected=False)

    mlab_mesh = m.current_mesh()

    reco_mesh = pymesh2vedo(mlab_mesh)

    return reco_mesh

def ball_pivoting_get_mesh(pcd):
    
    dists1 = []
    for p1 in pcd.coordinates:
        q1 = pcd.closest_point(p1, n=2)[1]
        dists1.append(mag(p1 - q1))
    radii = histogram(dists1, bins=25).bins.tolist() * 10
    
    point_cloud = vedo2open3d(pcd)
    
    point_cloud.estimate_normals()
    point_cloud.orient_normals_consistent_tangent_plane(50)
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(point_cloud, o3d.utility.DoubleVector(radii))
    
    
    return o3d2vedo(create_mesh(rec_mesh))

def Volume(points_array):
    # Create a Points object from the numpy array
    points = Points(points_array.cpu())
    
    # Create a mesh from the points
    # Note: This assumes the points form a closed surface (like a convex hull)
    mesh = Mesh(points).triangulate()
    
    # Compute the volume of the mesh
    volume = mesh.volume()
    
    return volume

    
def denormalize_point_cloud(normalized_points, min_vals, max_vals):
    # Calculate the range for each axis
    ranges = max_vals - min_vals
    # Find the axis with the greatest range
    max_range_axis = np.argmax(ranges)
    # Calculate the scaling factor
    scaling_factor = 1.0 / ranges[max_range_axis]
    # Calculate the midpoints
    midpoints = (min_vals + max_vals) / 2.0
    # Reverse the normalization

    denormalized_points = [(p / scaling_factor) + midpoints for p in normalized_points]
    
    return denormalized_points


@torch.no_grad()
def calc_part_acc(pts, trans1, trans2, rot1, rot2, valids, min_vals,  max_vals, gt_adjacency=None, max_num_part=44, adjac_dist_th = 0.01):
    adjac_dist_th = float(sys.argv[-1].strip())
    

    
    """Compute the `Part Accuracy` in the paper.

    We compute the per-part chamfer distance, and the distance lower than a
        threshold will be considered as correct.

    Args:
        pts: [B, P, N, 3], model input point cloud to be transformed
        trans1: [B, P, 3]
        trans2: [B, P, 3]
        rot1: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        rot2: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], accuracy per data in the batch
    """
    # NOT HERE
    denormalized_points = denormalize_point_cloud(pts, min_vals.cpu(), max_vals.cpu())
    pts_denormed = torch.stack([torch.tensor(p) for p in denormalized_points])
    pts_denormed = torch.tensor(pts_denormed, dtype=torch.float32)

    num_parts = pts.shape[0]
    gt_adjacency = torch.zeros((max_num_part, max_num_part))

    for part_i in range(num_parts):
        for part_j in range(num_parts):
            dist = distance.cdist(pts_denormed[part_i], pts_denormed[part_j]).min()
            if dist < adjac_dist_th:
               gt_adjacency[part_i, part_j] = 1
                
    
    B, P = pts.shape[:2]

    rot1_r = Rotation3D(rot1)
    rot2_r = Rotation3D(rot2)

    pts1 = transform_pc(trans1, rot1_r, pts)  # [B, P, N, 3]
    pts2 = transform_pc(trans2, rot2_r, pts)

    

    # BUT HERE
    denormalized_points1 = denormalize_point_cloud(pts1, min_vals.cpu(), max_vals.cpu())
    pts1 = torch.stack([torch.tensor(p) for p in denormalized_points1])
    pts1 = torch.tensor(pts1, dtype=torch.float32)

    denormalized_points2 = denormalize_point_cloud(pts2, min_vals.cpu(), max_vals.cpu())
    pts2 = torch.stack([torch.tensor(p) for p in denormalized_points2])
    pts2 = torch.tensor(pts2, dtype=torch.float32)

    # calculate the volume
  
    
    volumes = torch.zeros(gt_adjacency.shape[0])
    
    
    for idx, pc in enumerate(pts1):
        pynt = PyntCloud(pd.DataFrame(
            pc.cpu(), columns=['x', 'y', 'z']))
        voxel_id = pynt.add_structure('convex_hull')
        volume = pynt.structures[voxel_id].volume
        volumes[idx] = volume

    volumes_array = torch.tensor(volumes[:, np.newaxis] + volumes[np.newaxis, :])
    
    pred_adjacency = torch.zeros_like(gt_adjacency)
    
    for part_i in range(num_parts):
        for part_j in range(num_parts):
            dist = distance.cdist(pts1[part_i].cpu(), pts1[part_j].cpu()).min()
            if dist < adjac_dist_th:
                pred_adjacency[part_i, part_j] = 1
        
    
    
    # pred_adjacency = torch.tensor(gt_adjacency) 
    prec_calc = precision(pred_adjacency, gt_adjacency, volumes_array)
    recall_calc = recall(pred_adjacency, gt_adjacency, volumes_array)
    f1_calc = f1(pred_adjacency, gt_adjacency, volumes_array)
  
    intersection_volume_lst = []
    for t, (i, j) in enumerate(zip(pts1.cpu(), pts2.cpu())):
        
        pcd1 = Points(i)
        pcd2 = Points(j).c('r')
        
        msh1 = ball_pivoting_get_mesh(pcd1)
        msh2 = ball_pivoting_get_mesh(pcd2)
        try:
            surf1 = msh1
            surf1.color("blue5").alpha(0.1)
            vol1 = surf1.binarize()
            vol_value1 = compute_volume_value(vol1)

            surf2 = msh2
            surf2.color("red5").alpha(0.1)
            vol2 = surf2.binarize()
            vol_value2 = compute_volume_value(vol2)

            vol = vol1.operation('and', vol2)
            vol_value = compute_volume_value(vol)
            
            intersection_volume_lst.append((vol_value / vol_value2, vol_value2))
        except:
            intersection_volume_lst.append((0, vol_value2))
    
    total_volume = sum(x[1] for x in intersection_volume_lst)
    volume_weights = [x[1] / total_volume for x in intersection_volume_lst]
    
    Qpos = 0
    
    # breakpoint()
    
    for (i, j), w in zip(intersection_volume_lst, volume_weights):
        Qpos += w * i


    dist1, dist2, _, _ = chamfer_dist()(pts1, pts2)  # copy the repository
    loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)

    val = loss_per_data < 0.01
    partacc = val.sum() / len(val)
    return partacc, prec_calc, recall_calc, f1_calc, Qpos

