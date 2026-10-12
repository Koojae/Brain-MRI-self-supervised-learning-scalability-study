"""Volume loading, GPU-side cropping / resizing and the brain-valid token mask.

Data format
-----------
One pre-processed volume = one blosc2 ``.b2nd`` file holding a 3D ``float16`` array that is
skull-stripped (background exactly 0), intensity-normalised to [0, 1] and tightly cropped to
the brain bounding box.  Typical shapes are around 130-180 voxels per axis at 1 mm isotropic
spacing.  A training index is a JSON list of file paths.

Loader contract
---------------
DataLoader workers only decode; each item is ``(volume fp16 tensor, bbox, index)`` and the
collate function returns Python lists (no stacking, volumes have different shapes).  The
index maps a batch entry back to the dataset's path list; the training loops in this
repository do not use it.  All cropping, resizing and augmentation happens on the GPU inside
the training step.

Two input pipelines are provided:

* **Whole-volume resize** (MAE, JEPA): the volume is zero-padded symmetrically to a cube
  whose edge is its longest axis and resized with one isotropic trilinear factor to
  ``S^3`` (``S = 96``).  There is no randomness in this step.
* **Scale-based random resized crop** (DINO, SimCLR): the crop centre is drawn uniformly
  from brain voxels, a cubic window with edge ``round(s * max(volume.shape))`` for
  ``s ~ U[scale_range]`` is placed *centre-fit* (shifted inside the volume when the axis is
  long enough, symmetrically zero-padded otherwise) and resized trilinearly to ``S^3``.

Both pipelines track a crisp brain mask ``M``: the exact source mask ``(window > 0)`` is
resized with the same trilinear operator and thresholded at 0.5, and the image is multiplied
by ``M`` to remove interpolation bleed into the background.

``token_valid(M, patch_size, threshold)`` marks a 16^3 patch as valid when at least
``threshold`` of its voxels are brain (``threshold = 0.1`` in all runs).  Encoders drop
invalid tokens, masking objectives draw only from valid tokens, and any row without a valid
token falls back to all-valid to avoid empty sequences.
"""

import json

import blosc2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ─────────────────────────────── dataset ───────────────────────────────
class BrainMRIDataset(Dataset):
    """Decode-only dataset: returns the whole ``float16`` volume, its bbox and its index."""

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        vol = blosc2.open(self.paths[i])[:]                       # fp16 numpy array
        t = torch.from_numpy(np.ascontiguousarray(vol))
        bb = [[0, t.shape[0]], [0, t.shape[1]], [0, t.shape[2]]]  # stored volume = brain bbox
        return t, bb, i


def collate_gpu(batch):
    """Keep volumes as a list (variable shapes); return ``(vols, bboxes, idxs)``.

    ``idxs`` are the dataset indices of the items; they are forwarded for the caller and not
    used by the training loops in this repository.
    """
    return [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch]


def worker_init_fn(_):
    """One CPU thread per worker: 8 ranks x several workers would otherwise oversubscribe."""
    torch.set_num_threads(1)
    blosc2.set_nthreads(1)


def load_index(index_json):
    """Read a JSON index file into a list of ``.b2nd`` paths.

    The file holds a list of path strings (as written by ``tools/make_index.py``); an entry
    may also be a dict with an ``img_path`` key.
    """
    with open(index_json, "r") as f:
        entries = json.load(f)
    return [e if isinstance(e, str) else e["img_path"] for e in entries]


# ─────────────────────────────── GPU volumes ───────────────────────────────
def brain_voxel_index(vg):
    """``[K, 3]`` coordinates of nonzero (brain) voxels of a GPU volume, or None if empty."""
    idx = (vg > 0).nonzero(as_tuple=False)
    return idx if idx.numel() else None


def prepare_volumes(vols, device):
    """Move volumes to the GPU once and cache their brain-voxel indices.

    Returns ``(vgs, bidx)``; both lists are shared by every crop call on this batch so the
    host-to-device copy and the ``nonzero`` are done once per volume.
    """
    vgs, bidx = [], []
    for vol in vols:
        vg = vol.to(device, non_blocking=True)
        vgs.append(vg)
        bidx.append(brain_voxel_index(vg))
    return vgs, bidx


# ─────────────────────────────── random resized crop ───────────────────────────────
def crop_window(vg, center, S):
    """Cut an ``S^3`` window of ``vg`` around ``center`` into a new ``[1, S, S, S]`` tensor (centre-fit).

    Per axis: if the axis is at least ``S`` long the window start is clamped to
    ``[0, axis_len - S]`` so the window lies entirely inside the data (no padding); if the
    axis is shorter than ``S`` the whole axis is centred in the window and the remainder is
    zero-padded symmetrically.
    """
    out = torch.zeros(1, S, S, S, device=vg.device, dtype=vg.dtype)
    sl_i, sl_o = [], []
    for ax in range(3):
        axis_len = vg.shape[ax]
        if axis_len >= S:
            ws = max(0, min(center[ax] - S // 2, axis_len - S))
        else:
            ws = (axis_len - S) // 2
        a = max(0, ws)
        b = min(axis_len, ws + S)
        sl_i.append((a, b))
        sl_o.append((a - ws, a - ws + (b - a)))
    out[0, sl_o[0][0]:sl_o[0][1], sl_o[1][0]:sl_o[1][1], sl_o[2][0]:sl_o[2][1]] = \
        vg[sl_i[0][0]:sl_i[0][1], sl_i[1][0]:sl_i[1][1], sl_i[2][0]:sl_i[2][1]]
    return out


def crop_resized(vg, center, out_S, s, out=None, mask_out=None):
    """Scale-based random resized crop with brain-mask tracking.

    The cubic window edge is ``Sw = round(s * max(vg.shape))``; the window is cut with
    :func:`crop_window` and resized trilinearly to ``out_S^3``.  The exact source mask
    ``(window > 0)`` is resized with the same operator and thresholded at 0.5 to a crisp
    mask ``M``; the image is multiplied by ``M``.

    Returns ``(img [1, out_S^3] fp16, M [1, out_S^3] fp16 0/1)``; when ``out`` / ``mask_out``
    are given the results are also copied into them.
    """
    Sw = max(1, int(round(s * int(max(vg.shape)))))
    win = crop_window(vg, center, Sw)                                    # [1, Sw, Sw, Sw]
    mwin = (win > 0).to(torch.float32)                                   # exact source mask
    img = F.interpolate(win[None].float(), size=(out_S, out_S, out_S),
                        mode="trilinear", align_corners=False)[0]
    mr = F.interpolate(mwin[None], size=(out_S, out_S, out_S),
                       mode="trilinear", align_corners=False)[0]         # fractional
    M = (mr >= 0.5).to(vg.dtype)                                         # crisp
    img = (img * M).to(vg.dtype)
    if out is not None:
        out.copy_(img)
    if mask_out is not None:
        mask_out.copy_(M)
    return img, M


def gpu_crop_batch(vgs, bidx, bboxes, rng, device, S=96, NS=1, scale_range=(0.5, 1.0)):
    """``NS`` random resized crops per volume.

    ``vgs`` / ``bidx`` come from :func:`prepare_volumes`.  For each crop the centre is a
    uniformly drawn brain voxel (uniform inside the bbox if the volume has no brain voxel)
    and the scale is ``s ~ U[scale_range]``; the random stream is consumed as
    ``(centre, scale)`` per crop.  The ``NS`` centres of a volume are fetched from the GPU in
    one transfer.

    Returns ``(crops [V*NS, 1, S, S, S] fp16, masks [V*NS, 1, S, S, S] fp16 0/1)`` in
    volume-major order.
    """
    V = len(vgs)
    out = torch.zeros(V * NS, 1, S, S, S, device=device, dtype=torch.float16)
    mout = torch.zeros(V * NS, 1, S, S, S, device=device, dtype=torch.float16)
    lo, hi = float(scale_range[0]), float(scale_range[1])
    k = 0
    for v, (vg, bb) in enumerate(zip(vgs, bboxes)):
        bi = bidx[v]
        rows, cpu_centers, scales = [], [], []
        for _ in range(NS):
            if bi is not None:
                rows.append(int(rng.integers(0, bi.shape[0])))
            else:
                cpu_centers.append(tuple(int(rng.integers(a, b)) if b > a else a for a, b in bb))
            scales.append(float(rng.uniform(lo, hi)) if hi > lo else lo)
        if bi is not None:
            sel = torch.as_tensor(rows, device=device, dtype=torch.long)
            centers = bi[sel].tolist()                                   # one D2H per volume
        else:
            centers = [list(c) for c in cpu_centers]
        for i in range(NS):
            crop_resized(vg, centers[i], S, scales[i], out=out[k], mask_out=mout[k])
            k += 1
    return out, mout


# ─────────────────────────────── whole-volume resize ───────────────────────────────
def whole_volume_pad_resize(vg, out_S):
    """Zero-pad ``vg`` to a cube (edge = longest axis, symmetric) and resize to ``out_S^3``.

    One isotropic trilinear factor preserves the aspect ratio; the brain mask is tracked
    like in :func:`crop_resized`.  Deterministic: a volume always gives the same output.
    Returns ``(img [1, out_S^3] fp16, M [1, out_S^3] fp16 0/1)``.
    """
    L = int(max(vg.shape))
    pads = []                                        # F.pad order: last axis first, (lo, hi)
    for ax in reversed(range(3)):
        diff = L - vg.shape[ax]
        lo = diff // 2
        pads += [lo, diff - lo]
    cube = F.pad(vg[None, None].float(), pads)       # [1, 1, L, L, L]
    mcube = (cube > 0).to(torch.float32)
    img = F.interpolate(cube, size=(out_S, out_S, out_S), mode="trilinear", align_corners=False)[0]
    mr = F.interpolate(mcube, size=(out_S, out_S, out_S), mode="trilinear", align_corners=False)[0]
    M = (mr >= 0.5).to(vg.dtype)
    img = (img * M).to(vg.dtype)
    return img, M


def gpu_wholevol_batch(vols, device, S=96):
    """Whole-volume batch: ``(x [V, 1, S, S, S] fp16, M [V, 1, S, S, S] fp16 0/1)``."""
    out = torch.zeros(len(vols), 1, S, S, S, device=device, dtype=torch.float16)
    mout = torch.zeros(len(vols), 1, S, S, S, device=device, dtype=torch.float16)
    for v, vol in enumerate(vols):
        vg = vol.to(device, non_blocking=True)
        img, M = whole_volume_pad_resize(vg, S)
        out[v], mout[v] = img, M
    return out, mout


# ─────────────────────────────── valid tokens ───────────────────────────────
def token_valid(x, patch_size=16, threshold=0.0):
    """Brain-valid token mask ``[B, N]`` (bool) from ``x [B, 1, S, S, S]``.

    A token is valid when the fraction of voxels with ``x > 0`` in its ``patch_size^3`` patch
    is at least ``threshold``.  Pass the crisp mask ``M`` for an exact brain fraction.  Rows
    with no valid token are set to all-True so no sequence is empty.
    """
    occ = F.avg_pool3d((x > 0).to(torch.float32), patch_size, patch_size)   # [B, 1, g, g, g]
    tv = (occ >= threshold).flatten(1)                                      # [B, N]
    empty = tv.sum(dim=1) < 1
    if empty.any():
        tv[empty] = True
    return tv
