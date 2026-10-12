"""JEPA pre-training on whole-brain 96^3 volumes (``torchrun`` multi-GPU DDP).

Each volume is zero-padded to a cube and resized to ``img_size^3`` (no random crop; the
only randomness of the input pipeline is the GPU augmentation).  Brain-valid tokens are
computed from the resized brain mask, the NeuroVFM two-way masking draws a
``(context, target)`` pair per sample on the CPU, and :class:`jepa.model.JEPA3D` computes
the L1 latent-prediction loss under bf16 autocast.  After every optimizer step the target
encoder is updated by EMA.

Run: ``torchrun --standalone --nproc_per_node=8 jepa/train.py --config RUN_DIR/config.yaml``
(``launch.py`` writes the config and starts this command).

Files written to ``run_dir``:

* ``checkpoint_last.pt`` - rolling resume state (model + optimizer + counters) every
  ``ckpt_every_steps``; picked up automatically on restart and removed at the end.  The
  saved epoch is the one in progress at save time; a restart continues from the epoch
  after it (the remainder of the saved epoch is skipped, the step counter is exact).  The
  mask generator's call counter (mask seeds 0, 1, 2, ...) and the augmentation generator
  are not part of the saved state, so the masks and augmentations drawn after a resume
  are not those of an uninterrupted run.
* ``3d_jepa_vitb_96-final.pth.tar`` - ``{"encoder": {"module.vit.<k>": ...}, "meta": {...}}``
  holding the target (EMA) encoder.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import time

import torch
import torch.nn as nn
import yaml

from shared import optim as O
from shared import trainutil as T
from shared.data import gpu_wholevol_batch, token_valid
from shared.augment import gpu_augment
from jepa.model import JEPA3D
from jepa.masks import MultiBlockMaskGenerator


def main():
    ap = argparse.ArgumentParser(description="JEPA pre-training of the shared 3D ViT")
    ap.add_argument("--config", required=True, help="resolved run config (YAML)")
    with open(ap.parse_args().config) as f:
        cfg = yaml.safe_load(f)

    rank, world, local, device = T.ddp_setup()
    T.seed_everything(cfg["seed"], rank)
    aug_gen = torch.Generator(device=device).manual_seed(cfg["seed"] + rank)

    model = JEPA3D(
        arch=cfg["arch"], patch_size=cfg["patch_size"], in_chans=cfg["in_chans"],
        img_size=cfg["img_size"], pred_embed_dim=cfg["pred_embed_dim"],
        pred_depth=cfg["pred_depth"], pred_num_heads=cfg["pred_num_heads"],
        num_mask_tokens=cfg["num_mask_tokens"], valid_token_threshold=cfg["valid_token_threshold"],
        loss_exp=cfg["loss_exp"],
        enc_kwargs=T.enc_kwargs(cfg, cfg["img_size"], cfg["patch_size"]),
    ).to(device)
    # broadcast_buffers=False: the model has no buffers to synchronise, and the frozen
    # target encoder (outside the gradient path) evolves identically on every rank
    # through the EMA update.
    ddp = (nn.parallel.DistributedDataParallel(model, device_ids=[local], broadcast_buffers=False)
           if world > 1 else model)
    core = ddp.module if world > 1 else ddp
    if cfg.get("grad_checkpointing"):
        core.encoder.vit.grad_checkpointing = True   # activation recompute (memory only)

    mask_gen = MultiBlockMaskGenerator(cfg["mask"], volume_size=cfg["img_size"],
                                       patch_size=cfg["patch_size"],
                                       pad_enc=cfg["pad_enc"], pad_pred=cfg["pad_pred"])

    loader, sampler, n_vol = T.build_loader(
        cfg["index_json"], cfg["batch_size_per_gpu"], world, rank, num_workers=cfg["num_workers"])
    grad_accum = int(cfg.get("grad_accum", 1))
    steps_per_epoch = max(1, len(loader) // grad_accum)
    total_steps = int(cfg["total_steps"])                       # fixed step budget
    global_vol = cfg["batch_size_per_gpu"] * world * grad_accum
    peak_lr = O.compute_peak_lr(cfg["base_lr"], global_vol)

    # Only the context encoder and the predictor are optimised; the target encoder is frozen.
    opt = O.build_adamw([core.encoder, core.predictor], peak_lr, **T.opt_kwargs(cfg))
    ema = O.EmaUpdater([core.encoder], [core.target_encoder])
    clip_max = T.clip_grad_max(cfg)
    wd = float(cfg["weight_decay"])                             # constant over training
    ek = T.enc_kwargs(cfg, cfg["img_size"], cfg["patch_size"])

    S, ps, vt = cfg["img_size"], cfg["patch_size"], cfg["valid_token_threshold"]
    aug_cfg = cfg["aug"]

    lr_sum = O.lr_scaling_summary(cfg["base_lr"], global_vol, peak_lr)
    if T.is_main(rank):
        print(f"[JEPA] n_vol={n_vol} world={world} bs/gpu={cfg['batch_size_per_gpu']} accum={grad_accum} "
              f"global_vol={global_vol} steps/ep={steps_per_epoch} total_steps={total_steps} "
              f"peak_lr={peak_lr:.3e} lr_scaling={lr_sum}", flush=True)
        print(f"[JEPA] {T.stabilizer_summary(cfg, ek)} wd={wd} ema={T.ema_kwargs(cfg)}", flush=True)
        print(f"[JEPA] valid_token_threshold={vt} pad_enc={mask_gen.pad_enc} pad_pred={mask_gen.pad_pred} "
              f"predictor={cfg['pred_embed_dim']}/L{cfg['pred_depth']}/h{cfg['pred_num_heads']} "
              f"loss_exp={cfg['loss_exp']}", flush=True)
        print(f"[JEPA] input: whole volume, zero-pad to cube -> trilinear resize to {S}^3 "
              f"(no random crop)", flush=True)

    gstep = 0
    stop = False
    ckpt_every_steps = int(cfg.get("ckpt_every_steps", 5000))
    resume_path = os.path.join(cfg["run_dir"], "checkpoint_last.pt")
    start_epoch = 0
    if os.path.exists(resume_path):
        gstep, done_ep = T.load_resume(resume_path, core, opt, map_location=device)
        start_epoch = done_ep + 1
        if T.is_main(rank):
            print(f"[JEPA] RESUME from ep{done_ep} gstep{gstep} (checkpoint_last)", flush=True)
    model.train()
    epoch = start_epoch
    while gstep < total_steps:                                  # step-driven epoch loop
        if sampler is not None:
            sampler.set_epoch(epoch)
        opt.zero_grad(set_to_none=True)
        ep_t0 = time.time()
        for it, (vols, bboxes, vidx) in enumerate(loader):
            with torch.no_grad():
                crops, masks = gpu_wholevol_batch(vols, device, S=S)          # fp16
                x, M = gpu_augment(crops, masks, aug_cfg, aug_gen)            # fp16
                x = x.float()
                tv = token_valid(M, ps, vt)                                   # [B, N] bool
                enc, pred, pred_w, enc_w = mask_gen(tv.cpu())                 # two-way masking (CPU)
                enc = [e.to(device) for e in enc]
                pred = [p.to(device) for p in pred]
                pred_w = [w.to(device) for w in pred_w]
                enc_w = [w.to(device) for w in enc_w]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out = ddp(x, tv, enc, pred, pred_w, enc_w)
            loss = out["loss"]
            (loss / grad_accum).backward()

            if (it + 1) % grad_accum == 0:
                gn = O.clip_grad_norm([p for m in (core.encoder, core.predictor)
                                       for p in m.parameters() if p.requires_grad], clip_max)
                lr = O.set_lr(opt, O.lr_at(gstep, total_steps, peak_lr, **T.sched_kwargs(cfg)))
                opt.step()
                opt.zero_grad(set_to_none=True)
                ema.step(O.ema_momentum(gstep, total_steps, **T.ema_kwargs(cfg)))
                gstep += 1
                if T.is_main(rank) and gstep % cfg.get("log_every", 20) == 0:
                    msg = (f"[JEPA] ep{epoch} step{gstep}/{total_steps} loss={float(loss):.4f} "
                           f"gn={float(gn):.2f} lr={lr:.3e} wd={wd:.3f} valid_frac={float(tv.float().mean()):.3f} "
                           f"kmax={int(tv.sum(1).max())} kmean={float(tv.sum(1).float().mean()):.1f} "
                           f"encL={enc[0].shape[1]}/{float(enc_w[0].sum(1).float().mean()):.1f} "
                           f"predL={pred[0].shape[1]}/{float(pred_w[0].sum(1).float().mean()):.1f} "
                           f"encw_all={bool(enc_w[0].bool().all())} predw_all={bool(pred_w[0].bool().all())}")
                    # ovf_* > 0 means a batch exceeded the fixed padded length (shapes then
                    # change for that step); it should stay 0 for the whole run.
                    _ps = mask_gen.pad_stats()
                    msg += (f" pad[combos={_ps['n_shape_combos']} ovf={_ps['ovf_enc']}/{_ps['ovf_pred']}"
                            f" max={_ps['max_enc']}/{_ps['max_pred']}]")
                    if gstep % cfg.get("feat_every", 200) == 0:
                        nb = min(64, x.shape[0])
                        h = O.feature_covariance_rank(core.encode_features(x[:nb], M[:nb]))
                        msg += f" rank={h['rank']} eff_rank={h['eff_rank']:.1f}"
                        # Block-0 attention statistics of the target encoder (diagnostic only).
                        a = O.block0_attention_stats(core.target_encoder.vit, x)
                        msg += O.block0_msg(a)
                    print(msg, flush=True)
                if T.is_main(rank) and gstep % ckpt_every_steps == 0:
                    T.save_resume(resume_path, core, opt, gstep, epoch)
                    print(f"[JEPA] checkpoint_last saved @ step{gstep}", flush=True)
                if gstep >= total_steps:
                    stop = True
                    break

        if T.is_main(rank):
            print(f"[JEPA] EPOCH {epoch} time {time.time() - ep_t0:.1f}s", flush=True)
        if stop:
            break
        epoch += 1

    if T.is_main(rank):
        # Final checkpoint: the target (EMA) encoder only; the predictor is discarded.
        path = os.path.join(cfg["run_dir"], "3d_jepa_vitb_96-final.pth.tar")
        sd = {"module.vit." + k: v.cpu() for k, v in core.target_encoder.vit.state_dict().items()}
        T.save_ckpt(path, {"encoder": sd,
                           "meta": T.encoder_meta(cfg, "jepa", core.target_encoder.vit, lr_sum, total_steps)})
        print(f"[JEPA] saved {path}", flush=True)
        print(f"[JEPA] peak_mem={T.peak_mem_gb():.1f} GB/gpu", flush=True)
        T.rm_file(resume_path)
    T.barrier(world)
    T.cleanup(world)


if __name__ == "__main__":
    main()
