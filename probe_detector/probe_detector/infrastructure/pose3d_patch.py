"""Ultralytics patches for the pose3d probe model (kpt_shape [4, 4]: x, y, visibility, log Z).

Copied from oak_probe_vision/scripts/model/pose3d.py, which trained the weights; keep the two in
sync. Import this module before creating the YOLO model.
"""

import numpy as np
import torch
import torch.nn.functional as tf
from ultralytics.data import augment, base, dataset
from ultralytics.nn.modules import head
from ultralytics.utils import loss as uloss

DEPTH_GAIN = 2.0

_apply_keypoints = augment.RandomPerspective.apply_keypoints


def apply_keypoints(self, keypoints, M, size):
    if keypoints.shape[-1] <= 3:
        return _apply_keypoints(self, keypoints, M, size)
    out = _apply_keypoints(self, np.ascontiguousarray(keypoints[..., :3]), M, size)
    s = float(np.sqrt(abs(np.linalg.det(M[:2, :2]))))      # skala (bez obrotu/scinania w treningu)
    depth = keypoints[..., 3:4]
    depth = np.where(depth > 0, depth / s, 0).astype(keypoints.dtype)
    return np.concatenate([out, depth], -1)


_kpts_decode = head.Pose26.kpts_decode


def kpts_decode(self, kpts):
    if self.kpt_shape[1] != 4:
        return _kpts_decode(self, kpts)
    bs = kpts.shape[0]
    if self.export:
        y = kpts.view(bs, *self.kpt_shape, -1)
        a = (y[:, :, :2] + self.anchors) * self.strides
        a = torch.cat((a, y[:, :, 2:3].sigmoid(), y[:, :, 3:4]), 2)
        return a.view(bs, self.nk, -1)
    y = kpts.clone()
    y[:, 2::4] = y[:, 2::4].sigmoid()
    y[:, 0::4] = (y[:, 0::4] + self.anchors[0]) * self.strides
    y[:, 1::4] = (y[:, 1::4] + self.anchors[1]) * self.strides
    return y


_calc_kpts_loss = uloss.PoseLoss26.calculate_keypoints_loss
_pose_loss = uloss.PoseLoss26.loss


def calculate_keypoints_loss(self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes,
                             pred_kpts):
    if keypoints.shape[-1] != 4:
        return _calc_kpts_loss(self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes,
                               pred_kpts)
    # pred: x, y, v, logZ[, sigma_x, sigma_y] -> do zwyklej straty bez glebokosci
    pred3 = pred_kpts[..., [0, 1, 2, 4, 5]] if pred_kpts.shape[-1] == 6 else pred_kpts[..., :3]
    out = _calc_kpts_loss(self, masks, target_gt_idx, keypoints[..., :3].contiguous(), batch_idx, stride_tensor,
                          target_bboxes, pred3)
    depth_loss = pred_kpts[..., 3][..., :0].sum()           # zero w grafie (DDP)
    if masks.any():
        gt_d = self._select_target_keypoints(keypoints[..., 3:4].contiguous(), batch_idx, target_gt_idx,
                                             masks)[masks][..., 0]
        pd = pred_kpts[masks][..., 3]
        valid = gt_d > 0.01
        if valid.any():
            depth_loss = tf.smooth_l1_loss(pd[valid], torch.log(gt_d[valid]), beta=0.1)
    self._depth_loss = depth_loss
    return out


def pose_loss(self, preds, batch):
    self._depth_loss = None
    total, items = _pose_loss(self, preds, batch)
    if self.kpt_shape[1] != 4:
        return total, items
    d = self._depth_loss if self._depth_loss is not None else preds["kpts"][..., :0].sum()
    d = d * DEPTH_GAIN
    bs = preds["kpts"].shape[0]
    return torch.cat([total, (d * bs).reshape(1)]), {**items, "depth_loss": d.detach()}


_dataset_init = dataset.YOLODataset.__init__


def dataset_init(self, *args, data, task="detect", **kwargs):
    """YOLODataset odrzuca kpt_shape [N, 4] jeszcze przed wczytaniem etykiet - ta sama inicjalizacja
    bez tej kontroli (reszta, w tym verify_image_label, dziala dla dowolnego ndim)."""
    if task != "pose" or list(data.get("kpt_shape", (0, 0)))[1] != 4:
        return _dataset_init(self, *args, data=data, task=task, **kwargs)
    self.use_segments, self.use_keypoints, self.use_obb = False, True, False
    self.data = data
    base.BaseDataset.__init__(self, *args, channels=data.get("channels", 3), **kwargs)


dataset.YOLODataset.__init__ = dataset_init
augment.RandomPerspective.apply_keypoints = apply_keypoints
head.Pose26.kpts_decode = kpts_decode
uloss.PoseLoss26.calculate_keypoints_loss = calculate_keypoints_loss
uloss.PoseLoss26.loss = pose_loss
