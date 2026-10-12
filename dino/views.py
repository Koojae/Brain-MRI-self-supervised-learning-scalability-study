"""Multi-crop view construction for DINO.

Per volume in the batch: ``n_global`` global crops (``global_size^3``, scale
``global_scale_range``) and ``n_local`` local crops (``local_size^3``, scale
``local_scale_range``), each a scale-based random resized crop centred on a brain voxel
(:func:`shared.data.gpu_crop_batch`, centre-fit placement) with a crisp brain mask tracked
through the resize.  Every crop is then augmented independently with
:func:`shared.augment.gpu_augment`.

The random stream is fixed by construction order: all global crops of the batch are drawn
first (volume-major, ``(centre, scale)`` per crop from ``np_rng``), then all local crops;
augmentation then consumes ``aug_gen`` one view slot at a time, globals first, so view slot
``i`` holds the ``i``-th crop of every volume.

The volume upload and the brain-voxel index are computed once per batch and shared by the
global and the local crop pass.
"""

from shared.augment import gpu_augment
from shared.data import gpu_crop_batch, prepare_volumes


def _split(crops, masks, B, NS):
    """``[B*NS, 1, S^3]`` (volume-major) -> ``NS`` tensors of ``[B, 1, S^3]`` (one per view slot)."""
    S = crops.shape[-1]
    crops = crops.view(B, NS, 1, S, S, S)
    masks = masks.view(B, NS, 1, S, S, S)
    return [crops[:, i] for i in range(NS)], [masks[:, i] for i in range(NS)]


def make_dino_views(vols, bboxes, np_rng, aug_gen, device, cfg):
    """Build the augmented multi-crop batch for one training step.

    Args:
        vols: list of ``B`` fp16 volume tensors (CPU, variable shapes).
        bboxes: per-volume ``[[lo, hi]] * 3`` bounds used only for volumes without brain voxels.
        np_rng: ``numpy.random.Generator`` for crop centres and scales.
        aug_gen: ``torch.Generator`` on ``device`` for the augmentation.
        device: target device.
        cfg: dict with ``n_global, n_local, global_size, local_size, global_scale_range,
            local_scale_range, aug``.

    Returns ``{"global_crops": [n_global x [B, 1, G^3]], "global_masks": [...],
    "local_crops": [n_local x [B, 1, L^3]], "local_masks": [...]}``; crops and masks are fp32,
    masks are crisp 0/1 and follow the geometric augmentation of their crop.
    """
    B = len(vols)
    ng, nl = cfg["n_global"], cfg["n_local"]

    vgs, bidx = prepare_volumes(vols, device)

    g_crops, g_masks = gpu_crop_batch(vgs, bidx, bboxes, np_rng, device, S=cfg["global_size"],
                                      NS=ng, scale_range=tuple(cfg["global_scale_range"]))
    l_crops, l_masks = gpu_crop_batch(vgs, bidx, bboxes, np_rng, device, S=cfg["local_size"],
                                      NS=nl, scale_range=tuple(cfg["local_scale_range"]))
    gc, gm = _split(g_crops, g_masks, B, ng)
    lc, lm = _split(l_crops, l_masks, B, nl)
    aug = cfg["aug"]
    out = {"global_crops": [], "global_masks": [], "local_crops": [], "local_masks": []}
    for c, m in zip(gc, gm):
        x, M = gpu_augment(c.float(), m.float(), aug, aug_gen)   # .float() = fresh fp32 copy
        out["global_crops"].append(x)
        out["global_masks"].append(M)
    for c, m in zip(lc, lm):
        x, M = gpu_augment(c.float(), m.float(), aug, aug_gen)
        out["local_crops"].append(x)
        out["local_masks"].append(M)
    return out
