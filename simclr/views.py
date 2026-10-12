"""SimCLR view construction: ``n_views`` random-resized crops per volume, each augmented once.

Every crop of a volume is drawn independently (own brain-voxel centre, own scale), so the
two views of a volume are different sub-regions at different scales rather than two
augmentations of one crop.  All ``B * n_views`` crops are then passed through the shared
GPU augmentation in a single call.  Crops are ordered volume-major, i.e. ``x[v*n_views + j]``
is view ``j`` of volume ``v``, and ``vol_ids`` records the volume index of every crop so the
loss can identify positives.
"""

import torch

from shared.augment import gpu_augment
from shared.data import gpu_crop_batch, prepare_volumes


def make_simclr_views(vols, bboxes, np_rng, aug_gen, device, cfg):
    """Build the SimCLR crop batch for one DataLoader batch.

    Args:
        vols: list of ``B`` CPU fp16 volumes (variable shapes).
        bboxes: per-volume crop-centre bounding boxes (the stored volume extent).
        np_rng: ``numpy.random.Generator`` driving crop centres and scales.
        aug_gen: ``torch.Generator`` on ``device`` driving the augmentation.
        device: target CUDA device.
        cfg: dict with ``crop_size`` (output edge ``S``), ``n_views`` (crops per volume),
            ``scale_range`` (relative window scale ``[lo, hi]``) and ``aug`` (augmentation
            probabilities and ranges, see :func:`shared.augment.gpu_augment`).

    Returns:
        ``(x [B*n_views, 1, S, S, S] fp32, M [B*n_views, 1, S, S, S] fp32 0/1,
        vol_ids [B*n_views] long)``; ``M`` is the crisp brain mask of each crop after the
        geometric augmentations and ``vol_ids[i]`` is the local volume index of crop ``i``.
    """
    S = int(cfg["crop_size"])
    n_views = int(cfg["n_views"])
    scale_range = tuple(cfg["scale_range"])
    vgs, bidx = prepare_volumes(vols, device)
    crops, masks = gpu_crop_batch(vgs, bidx, bboxes, np_rng, device,
                                  S=S, NS=n_views, scale_range=scale_range)
    x, M = gpu_augment(crops.float(), masks.float(), cfg["aug"], aug_gen)
    vol_ids = torch.arange(len(vols), device=device, dtype=torch.long).repeat_interleave(n_views)
    return x, M, vol_ids
