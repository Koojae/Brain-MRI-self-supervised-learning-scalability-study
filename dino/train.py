"""DINO pre-training entry point.

Run through ``launch.py`` (which writes the resolved config) or directly::

    torchrun --standalone --nproc_per_node=8 dino/train.py --config <run_dir>/config.yaml

DINO must be launched through ``torchrun`` with at least two processes: the vendored iBOT
Sinkhorn-Knopp step all-reduces the masked-patch count unconditionally, so a single process
(no process group) would fail at the first training step.  ``main`` checks this and exits early.

One optimizer step = ``batch_size_per_gpu`` volumes per rank (x ``grad_accum``), each expanded
into ``n_global`` global and ``n_local`` local crops on the GPU (:mod:`dino.views`), a
student / teacher forward under bf16 autocast (:class:`dino.model.DINO3DSSL`), gradient
clipping, AdamW with the warm-up + cosine schedule, and the EMA teacher update.  The DINO /
iBOT ``last_layer`` gradients are dropped for the first ``freeze_last_layer_steps`` steps and
the teacher temperature warms up linearly over ``warmup_teacher_temp_frac`` of training.

Random streams per rank: the global torch / NumPy / Python seeds (``seed + rank``), a NumPy
generator for crop centres and scales (``seed * 1000 + rank``), a torch generator for the
augmentation (``seed + rank``) and one for the iBOT masks (``seed * 7 + rank``).

Outputs in ``run_dir``: a rolling ``checkpoint_last.pt`` (full model + optimizer, used to
resume; deleted at the end) and ``teacher_backbone_final.pt`` with the teacher encoder
``state_dict`` under ``"teacher_backbone"`` and the encoder ``meta`` dict.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from shared import optim as O
from shared import trainutil as T
from dino.model import DINO3DSSL
from dino.views import make_dino_views


def teacher_temp_at(step, total, warmup_frac, t0, t1):
    """Teacher temperature: linear ``t0 -> t1`` over ``warmup_frac * total`` steps, then ``t1``."""
    warm = max(1, int(round(warmup_frac * total)))
    if step >= warm:
        return t1
    return t0 + (t1 - t0) * (step / warm)


def main():
    ap = argparse.ArgumentParser(description="DINO pre-training of the shared 3D ViT")
    ap.add_argument("--config", required=True, help="resolved run config (YAML)")
    with open(ap.parse_args().config) as f:
        cfg = yaml.safe_load(f)

    rank, world, local, device = T.ddp_setup()
    if world < 2:
        raise SystemExit("[DINO] requires torchrun with >= 2 processes: the vendored iBOT "
                         "Sinkhorn-Knopp all-reduces the masked-patch count unconditionally")
    T.seed_everything(cfg["seed"], rank)
    np_rng = np.random.default_rng(cfg["seed"] * 1000 + rank)                 # crop centres / scales
    aug_gen = torch.Generator(device=device).manual_seed(cfg["seed"] + rank)   # augmentation
    mask_gen = torch.Generator(device=device).manual_seed(cfg["seed"] * 7 + rank)   # iBOT masks

    ek = T.enc_kwargs(cfg, cfg["global_size"], cfg["patch_size"])
    model = DINO3DSSL(
        arch=cfg["arch"], patch_size=cfg["patch_size"], in_chans=cfg["in_chans"],
        dino_out_dim=cfg["dino_head_n_prototypes"], ibot_out_dim=cfg["ibot_head_n_prototypes"],
        head_nlayers=cfg["head_nlayers"], head_hidden_dim=cfg["head_hidden_dim"],
        head_bottleneck_dim=cfg["head_bottleneck_dim"], student_temp=cfg["student_temp"],
        sinkhorn_iterations=cfg["sinkhorn_iterations"],
        valid_token_threshold=cfg["valid_token_threshold"],
        ibot_mask_ratio_min_max=tuple(cfg["ibot_mask_ratio_min_max"]),
        ibot_mask_sample_probability=cfg["ibot_mask_sample_probability"],
        koleo_weight=cfg["koleo_weight"], enc_kwargs=ek,
    )
    model = model.to(device)
    # Every student parameter receives a gradient each step (the loss keeps a zero-valued
    # path through the iBOT head when a crop has no masked patch), so DDP needs no
    # unused-parameter search.
    ddp = nn.parallel.DistributedDataParallel(model, device_ids=[local], find_unused_parameters=False)
    core = ddp.module
    if cfg.get("grad_checkpointing"):
        core.student_backbone.vit.grad_checkpointing = True

    loader, sampler, n_vol = T.build_loader(
        cfg["index_json"], cfg["batch_size_per_gpu"], world, rank, num_workers=cfg["num_workers"])
    grad_accum = cfg.get("grad_accum", 1)
    steps_per_epoch = max(1, len(loader) // grad_accum)
    total_steps = int(cfg["total_steps"])
    global_vol = cfg["batch_size_per_gpu"] * world * grad_accum
    peak_lr = O.compute_peak_lr(cfg["base_lr"], global_vol)

    student_mods = [core.student_backbone, core.student_dino_head, core.student_ibot_head]
    teacher_mods = [core.teacher_backbone, core.teacher_dino_head, core.teacher_ibot_head]
    opt = O.build_adamw(student_mods, peak_lr, **T.opt_kwargs(cfg))
    clip_max = T.clip_grad_max(cfg)
    wd = float(cfg["weight_decay"])
    ema = O.EmaUpdater(student_mods, teacher_mods)
    clip_params = [p for m in student_mods for p in m.parameters() if p.requires_grad]
    # The freeze below drops the gradients of ``last_layer.weight_g`` / ``weight_v``; those
    # attribute names exist only with ``torch.nn.utils.weight_norm`` (as vendored).  Fail
    # loudly rather than silently skipping the freeze if the head is ever re-parametrised.
    for h in (core.student_dino_head, core.student_ibot_head):
        assert hasattr(h.last_layer, "weight_g") and hasattr(h.last_layer, "weight_v"), \
            "DINOHead.last_layer must expose weight_g / weight_v (torch.nn.utils.weight_norm)"

    lr_sum = O.lr_scaling_summary(cfg["base_lr"], global_vol, peak_lr)
    if T.is_main(rank):
        print(f"[DINO] n_vol={n_vol} world={world} bs/gpu={cfg['batch_size_per_gpu']} accum={grad_accum} "
              f"global_vol={global_vol} steps/ep={steps_per_epoch} total_steps={total_steps} "
              f"peak_lr={peak_lr:.3e} lr_scaling={lr_sum}", flush=True)
        print(f"[DINO] {T.stabilizer_summary(cfg, ek)} wd={wd} ema={T.ema_kwargs(cfg)}"
              f" valid_token_threshold={cfg['valid_token_threshold']}"
              f" (ema_tensors={ema.n_params()})", flush=True)

    view_cfg = {k: cfg[k] for k in ("n_global", "n_local", "global_size", "local_size",
                                    "global_scale_range", "local_scale_range")}
    view_cfg["aug"] = cfg["aug"]
    freeze_last = cfg["freeze_last_layer_steps"]
    tt_frac, tt0, tt1 = cfg["warmup_teacher_temp_frac"], cfg["warmup_teacher_temp"], cfg["teacher_temp"]

    gstep = 0
    stop = False
    ckpt_every_steps = int(cfg.get("ckpt_every_steps", 5000))
    resume_path = os.path.join(cfg["run_dir"], "checkpoint_last.pt")
    start_epoch = 0
    if os.path.exists(resume_path):
        gstep, done_ep = T.load_resume(resume_path, core, opt, map_location=device)
        start_epoch = done_ep + 1
        if T.is_main(rank):
            print(f"[DINO] RESUME from ep{done_ep} gstep{gstep} (checkpoint_last)", flush=True)
    model.train()
    epoch = start_epoch
    log_every = cfg.get("log_every", 20)
    while gstep < total_steps:
        sampler.set_epoch(epoch)   # always a DistributedSampler: ``main`` exits when ``world < 2``
        opt.zero_grad(set_to_none=True)
        ep_t0 = time.time()
        for it, (vols, bboxes, _vidx) in enumerate(loader):
            with torch.no_grad():
                batch = make_dino_views(vols, bboxes, np_rng, aug_gen, device, view_cfg)
            ttemp = teacher_temp_at(gstep, total_steps, tt_frac, tt0, tt1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out = ddp(batch, ttemp, mask_gen)
            loss = out["loss"]
            (loss / grad_accum).backward()

            if (it + 1) % grad_accum == 0:
                if gstep < freeze_last:
                    # Freeze the prototype layer of both heads (weight-normalised Linear).
                    for h in (core.student_dino_head, core.student_ibot_head):
                        for pname in ("weight_g", "weight_v"):
                            p = getattr(h.last_layer, pname, None)
                            if p is not None and p.grad is not None:
                                p.grad = None
                gn = O.clip_grad_norm(clip_params, clip_max)      # tensor; read on log steps only
                lr = O.set_lr(opt, O.lr_at(gstep, total_steps, peak_lr, **T.sched_kwargs(cfg)))
                opt.step()
                opt.zero_grad(set_to_none=True)
                ema.step(O.ema_momentum(gstep, total_steps, **T.ema_kwargs(cfg)))
                gstep += 1
                if T.is_main(rank) and gstep % log_every == 0:
                    msg = (f"[DINO] ep{epoch} step{gstep}/{total_steps} loss={float(loss):.4f} "
                           f"dino={float(out['dino']):.3f} ibot={float(out['ibot']):.3f} "
                           f"koleo={float(out['koleo']):.3f} gn={float(gn):.2f} lr={lr:.2e} wd={wd:.3f} tt={ttemp:.3f}")
                    if gstep % cfg.get("feat_every", 200) == 0:
                        g0 = batch["global_crops"][0][: min(64, batch["global_crops"][0].shape[0])]
                        m0 = batch["global_masks"][0][: g0.shape[0]]
                        h = O.feature_covariance_rank(core.encode_features(g0, m0))
                        msg += f" rank={h['rank']} eff_rank={h['eff_rank']:.1f}"
                        a = O.block0_attention_stats(core.teacher_backbone.vit, g0)
                        msg += O.block0_msg(a)
                    print(msg, flush=True)
                if T.is_main(rank) and gstep % ckpt_every_steps == 0:
                    T.save_resume(resume_path, core, opt, gstep, epoch)
                    print(f"[DINO] checkpoint_last saved @ step{gstep}", flush=True)
                if gstep >= total_steps:
                    stop = True
                    break

        if T.is_main(rank):
            print(f"[DINO] EPOCH {epoch} time {time.time() - ep_t0:.1f}s", flush=True)
        if stop:
            break
        epoch += 1

    if T.is_main(rank):
        path = os.path.join(cfg["run_dir"], "teacher_backbone_final.pt")
        sd = {k: v.cpu() for k, v in core.teacher_backbone.vit.state_dict().items()}
        T.save_ckpt(path, {"teacher_backbone": sd,
                           "meta": T.encoder_meta(cfg, "dino", core.teacher_backbone.vit, lr_sum, total_steps)})
        print(f"[DINO] saved {path}", flush=True)
        print(f"[DINO] peak_mem={T.peak_mem_gb():.1f} GB/gpu", flush=True)
        T.rm_file(resume_path)
    T.barrier(world)
    T.cleanup(world)


if __name__ == "__main__":
    main()
