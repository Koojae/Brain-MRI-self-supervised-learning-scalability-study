"""GPU-side 3D augmentation shared by all four SSL methods.

``gpu_augment`` operates on a batch of cubic crops ``x [B, 1, S, S, S]`` together with a
crisp brain mask ``M [B, 1, S, S, S]`` (0/1) that the cropping code tracked through the
trilinear resize.  Three op groups are applied in this order, each op sampled per sample:

1. **geometry**: random flip on each of the three spatial axes (``flip_p`` per axis) and
   random 90-degree rotation in each of the three axis pairs (``rotate_p`` per pair,
   ``k`` uniform in {1, 2, 3}).  ``x`` and ``M`` are transformed together.
2. **contrast**: gamma ``clamp(x, 0, 1) ** g`` with ``g ~ U[gamma]`` (``gamma_p``) and a
   monotone piecewise-linear histogram shift with fixed end points (``hist_shift_p``).
3. **image quality**: additive Gaussian noise with ``std ~ U[0, noise_std]`` (``noise_p``)
   and a per-sample separable Gaussian blur with ``sigma ~ U[blur_sigma]`` (``blur_p``).

Invariant: after all ops ``x`` is multiplied by the (possibly flipped/rotated) mask so the
background stays exactly zero.  The returned ``mask`` -- not ``x > 0`` -- is the brain
indicator that callers hand to ``token_valid``: additive noise can push dark in-brain voxels
to or below zero, so after augmentation ``x > 0`` implies brain but not the converse.  All
randomness comes from the ``torch.Generator`` passed as ``gen``.

Implementation note: the flip, gamma, noise and blur ops select samples with ``torch.where``
(no device-to-host synchronisation).  The 90-degree rotation and the histogram shift gather
the selected samples by index (``nonzero`` + advanced indexing, data-dependent shapes), and
four ``sel.any()`` gates guard ops that draw additional random numbers; these are the
synchronisation points.  Skipping those draws when no sample is selected is part of the
random-stream definition.
"""

import math

import torch
import torch.nn.functional as F


def _urand(M, lo, hi, device, gen):
    return torch.empty(M, device=device).uniform_(float(lo), float(hi), generator=gen)


def _gaussian_smooth3d_persample(x, sigmas, sigma_max):
    """Per-sample separable 3D Gaussian blur.  ``x [M, 1, S, S, S]``, ``sigmas [M]`` (voxels).

    The kernel length is fixed by ``sigma_max``; each sample's 1D kernel is evaluated on that
    support and applied as one grouped (depthwise) conv per axis.  ``sigma ~ 0`` is the identity.
    """
    M = x.shape[0]
    radius = max(1, int(math.ceil(2.5 * sigma_max)))
    ksize = 2 * radius + 1
    coords = torch.arange(ksize, device=x.device, dtype=torch.float32) - radius       # [ksize]
    s = sigmas.clamp_min(1e-6).to(torch.float32).view(M, 1)                            # [M, 1]
    k1d = torch.exp(-(coords.view(1, -1) ** 2) / (2.0 * s ** 2))                       # [M, ksize]
    k1d = (k1d / k1d.sum(dim=1, keepdim=True)).to(x.dtype)
    out = x.view(1, M, *x.shape[2:])                                                  # [1, M, S, S, S]
    for axis in range(3):
        wshape = [M, 1, 1, 1, 1]
        wshape[2 + axis] = ksize
        weight = k1d.view(*wshape)
        pad = [0, 0, 0]
        pad[axis] = radius
        out = F.conv3d(out, weight, padding=(pad[0], pad[1], pad[2]), groups=M)
    return out.view(M, 1, *x.shape[2:]).to(x.dtype)


def _rand_rot90_3pairs(x, fg, p, gen):
    """Random 90-degree rotations, independently for the axis pairs (2,3), (2,4), (3,4).

    For each pair, every sample is rotated with probability ``p`` by ``k ~ U{1, 2, 3}``
    quarter turns.  ``x`` and the boolean foreground ``fg`` are rotated together.  Requires
    cubic crops.
    """
    pairs = [(2, 3), (2, 4), (3, 4)]
    M = x.shape[0]
    for pair in pairs:
        sel = torch.rand(M, device=x.device, generator=gen) < p
        idx = sel.nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        kk = torch.randint(1, 4, (idx.numel(),), device=x.device, generator=gen)
        for k in (1, 2, 3):
            g = idx[kk == k]
            if g.numel():
                x[g] = torch.rot90(x[g], k, dims=list(pair))
                fg[g] = torch.rot90(fg[g], k, dims=list(pair))
    return x, fg


def _hist_shift_persample(x, fg, sel, n_ctrl, a, gen):
    """Monotone piecewise-linear intensity remap for the selected samples.

    ``n_ctrl`` control points span [0, 1]; the segment lengths are perturbed by
    ``U[1 - a, 1 + a]`` and renormalised, giving a monotone increasing look-up table with
    fixed end points (0 -> 0, 1 -> 1), so the background stays zero.  Vectorised gather.
    """
    idx = sel.nonzero(as_tuple=False).flatten()
    if idx.numel() == 0:
        return x
    xs, fgs = x[idx], fg[idx]                                                # [m, 1, S, S, S]
    m = xs.shape[0]
    n = max(3, int(n_ctrl))
    seg = torch.empty(m, n - 1, device=x.device).uniform_(1.0 - a, 1.0 + a, generator=gen)
    seg = seg / seg.sum(dim=1, keepdim=True)                                 # positive, sums to 1
    fp = torch.cat([torch.zeros(m, 1, device=x.device),
                    torch.cumsum(seg, dim=1)], dim=1).to(xs.dtype)           # [m, n] monotone 0..1
    v = xs.clamp(0, 1)
    pos = v * (n - 1)
    lo = pos.floor().clamp(0, n - 2).long()                                  # segment index
    t = pos - lo.to(v.dtype)
    flat = lo.reshape(m, -1)
    y_lo = torch.gather(fp, 1, flat).reshape_as(v)
    y_hi = torch.gather(fp, 1, (flat + 1)).reshape_as(v)
    x[idx] = ((y_lo + t * (y_hi - y_lo)) * fgs.to(v.dtype)).to(x.dtype)
    return x


def _smooth_sel(x, sel, sig_lo, sig_hi, gen):
    """Gaussian blur with ``sigma ~ U[sig_lo, sig_hi]`` for the selected samples (others: identity)."""
    M = x.shape[0]
    sig = _urand(M, sig_lo, sig_hi, x.device, gen)
    sig = torch.where(sel, sig, torch.full_like(sig, 1e-6))
    return _gaussian_smooth3d_persample(x, sig, float(sig_hi))


def gpu_augment(x, mask, aug, gen):
    """Augment a crop batch on the GPU.  See the module docstring for the op list and order.

    Args:
        x: ``[B, 1, S, S, S]`` intensities in [0, 1] with background exactly 0.
        mask: ``[B, 1, S, S, S]`` crisp brain mask (0/1), same dtype as ``x``.
        aug: dict of per-op probabilities and ranges (``flip_p``, ``rotate_p``, ``gamma_p``,
            ``gamma``, ``hist_shift_p``, ``hist_shift_ctrl``, ``hist_shift_a``, ``noise_p``,
            ``noise_std``, ``blur_p``, ``blur_sigma``).
        gen: ``torch.Generator`` on ``x.device``.

    Returns ``(x, mask)``, both reflecting the geometric transforms.
    """
    B = x.shape[0]

    # ── geometry: flips (x and mask together), then 90-degree rotations ──
    fp = aug.get("flip_p", 0.5)
    if fp and fp > 0:
        for ax in (2, 3, 4):
            m = (torch.rand(B, device=x.device, generator=gen) < fp).view(B, 1, 1, 1, 1)
            x = torch.where(m, torch.flip(x, dims=[ax]), x)
            mask = torch.where(m, torch.flip(mask, dims=[ax]), mask)
    rp = aug.get("rotate_p", 0.5)
    if rp and rp > 0:
        mb = mask > 0
        x, mb = _rand_rot90_3pairs(x, mb, rp, gen)
        mask = mb.to(x.dtype)
    fg = mask                                                     # crisp foreground (0/1)

    # ── contrast: gamma, then histogram shift (x only; background stays 0) ──
    gp = aug.get("gamma_p", 0.5)
    sel = torch.rand(B, device=x.device, generator=gen) < gp
    if bool(sel.any()):
        g0, g1 = aug.get("gamma", (0.5, 2.0))
        gam = _urand(B, g0, g1, x.device, gen).view(B, 1, 1, 1, 1).to(x.dtype)
        xc = x.clamp(0, 1).pow(gam)                               # 0 ** g = 0 keeps background
        x = torch.where(sel.view(B, 1, 1, 1, 1), xc, x)
    hp = aug.get("hist_shift_p", 0.5)
    sel = torch.rand(B, device=x.device, generator=gen) < hp
    if bool(sel.any()):
        x = _hist_shift_persample(x, fg, sel, aug.get("hist_shift_ctrl", 10),
                                  aug.get("hist_shift_a", 0.3), gen)

    # ── image quality: additive Gaussian noise, then Gaussian blur (x only) ──
    np_ = aug.get("noise_p", 0.5)
    sel = torch.rand(B, device=x.device, generator=gen) < np_
    if bool(sel.any()):
        std = _urand(B, 0.0, aug.get("noise_std", 0.05), x.device, gen).view(B, 1, 1, 1, 1).to(x.dtype)
        xn = x + torch.randn(x.shape, device=x.device, generator=gen, dtype=x.dtype) * std
        x = torch.where(sel.view(B, 1, 1, 1, 1), xn, x)
    bp = aug.get("blur_p", 0.5)
    sel = torch.rand(B, device=x.device, generator=gen) < bp
    if bool(sel.any()):
        b_lo, b_hi = aug.get("blur_sigma", (0.25, 1.0))
        x = _smooth_sel(x, sel, b_lo, b_hi, gen)

    # ── invariant: background exactly zero after every op ──
    x = x * fg
    return x, mask
