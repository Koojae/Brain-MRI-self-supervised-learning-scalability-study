"""SimCLR pre-training of the shared 3D ViT on brain MRI (multi-GPU DDP via ``torchrun``).

Each micro-batch holds ``batch_size_per_gpu`` volumes per rank and ``grad_accum`` micro-batches
form one optimizer step (the default configuration uses ``grad_accum`` = 1).  For every volume
two random-resized crops (``96^3``, scale ``[0.5, 1]`` of the longest axis, centre-fit) are cut
on the GPU and augmented independently; the two crops of a volume are the positive pair and
all other crops in the global batch are negatives.  Crops pass through the encoder with
brain-valid token dropping, the class token feeds a ``Linear -> BN -> ReLU -> Linear``
projection head (``SyncBatchNorm`` under DDP) and the NT-Xent loss (temperature 0.1) is
computed on the all-gathered embeddings inside the bf16 autocast region.

Optimisation: AdamW (weight decay 0.1, betas (0.9, 0.999)), peak learning rate
``base_lr * volumes_per_step / 256`` with 10 % linear warm-up and cosine decay to
``min_lr``, gradient clipping at 3.0, patch-embedding learning-rate multiplier 0.2.

Outputs in ``run_dir``: a rolling ``checkpoint_last.pt`` (model + optimizer, removed at the
end) and the final ``ckpt_final.pt`` holding ``backbone_state_dict`` (encoder weights with a
``backbone.`` prefix) and ``meta`` (encoder configuration, see
:func:`shared.trainutil.encoder_meta`).

Run: ``torchrun --standalone --nproc_per_node=8 simclr/train.py --config <run_dir>/config.yaml``
(``launch.py`` writes the config and issues this command).
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402

from shared import optim as O  # noqa: E402
from shared import trainutil as T  # noqa: E402
from simclr.loss import NTXentLoss  # noqa: E402
from simclr.model import SimCLRModel  # noqa: E402
from simclr.views import make_simclr_views  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="SimCLR pre-training of the shared 3D ViT")
    ap.add_argument("--config", required=True, help="resolved run config (YAML)")
    with open(ap.parse_args().config) as f:
        cfg = yaml.safe_load(f)

    rank, world, local, device = T.ddp_setup()
    T.seed_everything(cfg["seed"], rank)
    np_rng = np.random.default_rng(cfg["seed"] * 1000 + rank)      # crop centres / scales
    aug_gen = torch.Generator(device=device).manual_seed(cfg["seed"] + rank)   # augmentation

    model = SimCLRModel(
        arch=cfg["arch"], in_chans=cfg["in_chans"], patch_size=cfg["patch_size"],
        valid_token_threshold=cfg["valid_token_threshold"],
        proj_hidden=cfg["proj_hidden"], proj_dim=cfg["proj_dim"],
        enc_kwargs=T.enc_kwargs(cfg, cfg["crop_size"], cfg["patch_size"]),
    ).to(device)
    if world > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)     # projection-head BN across ranks
    ddp = (nn.parallel.DistributedDataParallel(model, device_ids=[local]) if world > 1 else model)
    core = ddp.module if world > 1 else ddp
    if cfg.get("grad_checkpointing"):
        core.backbone.vit.grad_checkpointing = True   # recompute block activations (memory only)
    criterion = NTXentLoss(temperature=cfg["temperature"], gather=True)

    loader, sampler, n_vol = T.build_loader(
        cfg["index_json"], cfg["batch_size_per_gpu"], world, rank, num_workers=cfg["num_workers"])
    grad_accum = cfg.get("grad_accum", 1)
    steps_per_epoch = max(1, len(loader) // grad_accum)
    total_steps = int(cfg["total_steps"])
    global_vol = cfg["batch_size_per_gpu"] * world * grad_accum     # volumes per optimizer step
    peak_lr = O.compute_peak_lr(cfg["base_lr"], global_vol)

    opt = O.build_adamw(core, peak_lr, **T.opt_kwargs(cfg))
    clip_max = T.clip_grad_max(cfg)
    wd = float(cfg["weight_decay"])
    _ek = T.enc_kwargs(cfg, cfg["crop_size"], cfg["patch_size"])
    n_views = int(cfg["n_views"])

    lr_sum = O.lr_scaling_summary(cfg["base_lr"], global_vol, peak_lr)
    if T.is_main(rank):
        print(f"[SimCLR] n_vol={n_vol} world={world} bs/gpu={cfg['batch_size_per_gpu']} accum={grad_accum} "
              f"global_vol={global_vol} crops/vol={n_views} total_crops={global_vol * n_views} "
              f"crop={cfg['crop_size']} scale={tuple(cfg['scale_range'])} tau={cfg['temperature']} "
              f"vt={cfg['valid_token_threshold']} "
              f"steps/ep={steps_per_epoch} total_steps={total_steps} peak_lr={peak_lr:.3e} "
              f"lr_scaling={lr_sum}", flush=True)
        print(f"[SimCLR] stabilizers: {T.stabilizer_summary(cfg, _ek)}", flush=True)
        _ph = core.proj
        _hp = sum(q.numel() for q in _ph.parameters())
        _bp = sum(q.numel() for q in core.backbone.vit.parameters())
        _nbn = len([m for m in _ph.net if isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm))])
        print(f"[SimCLR] proj head: "
              f"linears={len([m for m in _ph.net if isinstance(m, nn.Linear)])} bn={_nbn} "
              f"mlp_dim={_ph.mlp_dim} out_dim={_ph.out_dim} "
              f"head_params={_hp:,} backbone={_bp:,} head/backbone={_hp / _bp * 100:.1f}%", flush=True)

    view_cfg = {k: cfg[k] for k in ("n_views", "crop_size", "scale_range")}
    view_cfg["aug"] = cfg["aug"]
    bs = cfg["batch_size_per_gpu"]
    gstep = 0
    stop = False
    ckpt_every_steps = int(cfg.get("ckpt_every_steps", 5000))
    resume_path = os.path.join(cfg["run_dir"], "checkpoint_last.pt")
    start_epoch = 0
    if os.path.exists(resume_path):
        gstep, done_ep = T.load_resume(resume_path, core, opt, map_location=device)
        start_epoch = done_ep + 1
        if T.is_main(rank):
            print(f"[SimCLR] RESUME from ep{done_ep} gstep{gstep} (checkpoint_last)", flush=True)
    model.train()
    epoch = start_epoch
    while gstep < total_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        opt.zero_grad(set_to_none=True)
        ep_t0 = time.time()
        for it, (vols, bboxes, _vidx) in enumerate(loader):
            with torch.no_grad():
                x, M, vol_ids = make_simclr_views(vols, bboxes, np_rng, aug_gen, device, view_cfg)
                vol_ids = vol_ids + rank * bs                      # globally unique volume id
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                z, _ = ddp(x, M)
                loss = criterion(z.float(), vol_ids)
            (loss / grad_accum).backward()

            if (it + 1) % grad_accum == 0:
                gn = O.clip_grad_norm(core.parameters(), clip_max)
                lr = O.set_lr(opt, O.lr_at(gstep, total_steps, peak_lr, **T.sched_kwargs(cfg)))
                opt.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1
                if T.is_main(rank) and gstep % cfg.get("log_every", 20) == 0:
                    msg = (f"[SimCLR] ep{epoch} step{gstep}/{total_steps} loss={float(loss):.4f} "
                           f"gn={float(gn):.2f} lr={lr:.3e} wd={wd:.3f}")
                    if gstep % cfg.get("feat_every", 200) == 0:
                        # Collapse monitors (diagnostic only): feature-covariance rank of the
                        # class-token features and first-block attention statistics.
                        nb = min(128, x.shape[0])
                        h = O.feature_covariance_rank(core.backbone(x[:nb], M[:nb]))
                        msg += f" rank={h['rank']} eff_rank={h['eff_rank']:.1f}"
                        a = O.block0_attention_stats(core.backbone.vit, x[:nb])
                        msg += O.block0_msg(a)
                    print(msg, flush=True)
                if T.is_main(rank) and gstep % ckpt_every_steps == 0:
                    T.save_resume(resume_path, core, opt, gstep, epoch)
                    print(f"[SimCLR] checkpoint_last saved @ step{gstep}", flush=True)
                if gstep >= total_steps:
                    stop = True
                    break

        if T.is_main(rank):
            print(f"[SimCLR] EPOCH {epoch} time {time.time() - ep_t0:.1f}s", flush=True)
        if stop:
            break
        epoch += 1

    if T.is_main(rank):
        path = os.path.join(cfg["run_dir"], "ckpt_final.pt")
        sd = {"backbone." + k: v.cpu() for k, v in core.backbone.vit.state_dict().items()}
        T.save_ckpt(path, {"backbone_state_dict": sd,
                           "meta": T.encoder_meta(cfg, "simclr", core.backbone.vit, lr_sum, total_steps)})
        print(f"[SimCLR] saved {path}", flush=True)
        print(f"[SimCLR] peak_mem={T.peak_mem_gb():.1f} GB/gpu", flush=True)
        T.rm_file(resume_path)
    T.barrier(world)
    T.cleanup(world)


if __name__ == "__main__":
    main()
