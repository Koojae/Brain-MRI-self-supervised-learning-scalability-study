"""Training scaffolding shared by the four ``train.py`` scripts.

Distributed set-up (``torchrun`` environment variables, NCCL), seeding, the volume
DataLoader, resume / final checkpoint helpers, and the small ``config -> kwargs`` adapters
that keep the four scripts reading the same config keys the same way.
"""

import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from shared.data import BrainMRIDataset, collate_gpu, worker_init_fn, load_index
from shared.vit3d import QK_NORM_EPS


# ─────────────────────────────── DDP ───────────────────────────────
def ddp_setup():
    """Initialise the process group from the ``torchrun`` env.  Returns ``(rank, world, local_rank, device)``."""
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device(f"cuda:{local}")
    else:
        device = torch.device("cpu")
    return rank, world, local, device


def seed_everything(seed=42, rank=0):
    """Seed ``random`` / NumPy / torch / CUDA with ``seed + rank`` and make cuDNN deterministic.

    ``PYTHONHASHSEED`` must already be set in the process environment (``launch.py`` does
    this for the ``torchrun`` children); setting it here only documents the intent.
    """
    eff = int(seed) + int(rank)
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(eff)
    np.random.seed(eff)
    torch.manual_seed(eff)
    torch.cuda.manual_seed_all(eff)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return eff


def is_main(rank):
    return rank == 0


def barrier(world):
    if world > 1 and dist.is_initialized():
        dist.barrier()


def cleanup(world):
    if world > 1 and dist.is_initialized():
        dist.destroy_process_group()


# ─────────────────────────────── data ───────────────────────────────
def build_loader(index_json, batch_size_per_gpu, world, rank, num_workers=4, drop_last=True):
    """Volume-level DataLoader (workers decode only).  Returns ``(loader, sampler, n_vol)``.

    ``DistributedSampler(shuffle=True, drop_last=True)`` for ``world > 1``; the caller must
    call ``sampler.set_epoch(epoch)`` every epoch.  Steps per epoch =
    ``((n_vol // world) // batch_size_per_gpu)``.
    """
    paths = load_index(index_json)
    ds = BrainMRIDataset(paths)
    sampler = (DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=drop_last)
               if world > 1 else None)
    kw = {}
    if num_workers > 0:
        kw["prefetch_factor"] = 2
        kw["persistent_workers"] = True
    loader = DataLoader(
        ds, batch_size=batch_size_per_gpu, sampler=sampler, shuffle=(sampler is None),
        num_workers=num_workers, pin_memory=True, drop_last=drop_last,
        collate_fn=collate_gpu, worker_init_fn=worker_init_fn, **kw,
    )
    return loader, sampler, len(paths)


# ─────────────────────────────── checkpoints ───────────────────────────────
def peak_mem_gb():
    """Peak allocated GPU memory of this rank in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def save_ckpt(path, obj):
    """Atomic ``torch.save`` (write to ``.tmp`` then rename).  Call on rank 0 only."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)
    return path


def save_resume(path, model, opt, gstep, epoch, extra=None):
    """Rolling resume checkpoint: full model + optimizer + step / epoch counters."""
    obj = {"model": model.state_dict(), "opt": opt.state_dict(),
           "gstep": int(gstep), "epoch": int(epoch)}
    if extra:
        obj.update(extra)
    return save_ckpt(path, obj)


def load_resume(path, model, opt, map_location="cpu"):
    """Restore a resume checkpoint on every rank.  Returns ``(gstep, epoch)``."""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    return int(ck["gstep"]), int(ck["epoch"])


def rm_file(path):
    try:
        os.remove(path)
    except OSError:
        pass


# ─────────────────────────────── config adapters ───────────────────────────────
def enc_kwargs(cfg, img_size, patch_size=16):
    """Config -> :class:`shared.vit3d.ViT3D` keyword arguments.

    ``pos_base_grid`` is the token grid of the method's reference crop (``img_size //
    patch_size`` per axis).  ``qk_norm`` / ``qk_norm_eps`` come straight from the config.
    """
    g = int(img_size) // int(patch_size)
    return {
        "pos_base_grid": (g, g, g),
        "qk_norm": bool(cfg.get("qk_norm", False)),
        "qk_norm_eps": float(cfg.get("qk_norm_eps", QK_NORM_EPS)),
    }


def opt_kwargs(cfg):
    """Config -> :func:`shared.optim.build_adamw` keyword arguments."""
    return {"wd": float(cfg["weight_decay"]),
            "betas": (float(cfg.get("adamw_beta1", 0.9)), float(cfg.get("adamw_beta2", 0.999))),
            "patch_embed_lr_mult": float(cfg.get("patch_embed_lr_mult", 1.0))}


def sched_kwargs(cfg):
    """Config -> :func:`shared.optim.lr_at` keyword arguments."""
    return {"warmup_frac": float(cfg.get("warmup_frac", 0.1)),
            "min_lr": float(cfg.get("min_lr", 1.0e-6))}


def ema_kwargs(cfg):
    """Config -> :func:`shared.optim.ema_momentum` keyword arguments."""
    return {"m0": float(cfg["ema_start"]), "m1": float(cfg["ema_end"]),
            "schedule": str(cfg["ema_schedule"])}


def clip_grad_max(cfg):
    """Gradient-clipping max norm (3.0 in all runs)."""
    return float(cfg.get("clip_grad", 3.0))


def stabilizer_summary(cfg, ek):
    """One-line description of the fixed encoder recipe and this run's optimizer settings."""
    return (f"pos=rope base_grid={ek['pos_base_grid']} init=stabilized depth_rescale=True"
            f" qk_norm={ek['qk_norm']}/{ek['qk_norm_eps']:g} clip={clip_grad_max(cfg)}"
            f" sched={sched_kwargs(cfg)} adamw={opt_kwargs(cfg)}")


def encoder_meta(cfg, method, vit, lr_sum, total_steps):
    """``meta`` dict stored next to every final encoder checkpoint.

    RoPE adds no parameters and QK-norm adds LayerNorm keys only when enabled, so a loader
    must rebuild :class:`shared.vit3d.ViT3D` from these fields rather than from the
    ``state_dict`` alone.
    """
    return {
        "method": method,
        "arch": vit.arch,
        "patch_size": int(vit.patch_size),
        "embed_dim": int(vit.embed_dim),
        "depth": int(vit.depth),
        "num_heads": int(vit.num_heads),
        "pos_base_grid": list(vit.pos_base_grid),
        "pos_embed_type": "rope",
        "init_scheme": "stabilized",
        "depth_rescale": True,
        "qk_norm": bool(vit.qk_norm),
        "qk_norm_eps": float(vit.qk_norm_eps),
        "valid_token_threshold": float(cfg["valid_token_threshold"]),
        "valid_token_threshold_enc": float(cfg["valid_token_threshold"]),
        "total_steps": int(total_steps),
        **lr_sum,
    }
