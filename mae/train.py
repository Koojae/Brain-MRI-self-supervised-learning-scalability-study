"""MAE pre-training of the shared 3D ViT encoder on whole brain volumes (DDP, one process per GPU).

Run through ``launch.py`` (it resolves the config and starts ``torchrun``), or directly:
``torchrun --standalone --nproc_per_node=8 mae/train.py --config RUN_DIR/config.yaml``.

Per micro-batch every rank: loads ``batch_size_per_gpu`` volumes, resizes each whole
volume to ``img_size^3`` (:func:`shared.data.gpu_wholevol_batch`, deterministic), augments
on the GPU (:func:`shared.augment.gpu_augment`, per-rank generator seeded ``seed + rank``),
and runs the MAE forward / backward under bf16 autocast; ``grad_accum`` micro-batches form
one optimizer step (1 in all runs of the study).  AdamW with the shared parameter
grouping, linear warm-up + cosine learning-rate schedule to ``min_lr``, constant weight
decay and global-norm gradient clipping.  Training is step-driven: it stops after
``total_steps`` optimizer steps regardless of epoch boundaries.

Files written to ``run_dir``:

* ``checkpoint_last.pt`` — rolling resume state (model + optimizer + counters) every
  ``ckpt_every_steps``; picked up automatically on restart and removed at the end.  The
  saved epoch is the one in progress at save time; a restart continues from the epoch
  after it (the remainder of the saved epoch is skipped, the step counter is exact).
* ``checkpoints/step{total_steps:06d}.ckpt`` — ``{"state_dict", "meta"}`` with the encoder
  ``state_dict`` under the ``model.encoder.`` prefix and the ``meta`` dict of
  :func:`shared.trainutil.encoder_meta` (enough to rebuild the encoder; see
  ``tools/load_encoder.py``).
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402

from shared import optim as O  # noqa: E402
from shared import trainutil as T  # noqa: E402
from shared.data import gpu_wholevol_batch  # noqa: E402
from shared.augment import gpu_augment  # noqa: E402
from mae.model import MAE3D  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="MAE pre-training of the shared 3D ViT encoder")
    ap.add_argument("--config", required=True, help="resolved YAML config (written by launch.py)")
    with open(ap.parse_args().config) as f:
        cfg = yaml.safe_load(f)

    rank, world, local, device = T.ddp_setup()
    T.seed_everything(cfg["seed"], rank)
    aug_gen = torch.Generator(device=device).manual_seed(cfg["seed"] + rank)

    # ── model ──
    ek = T.enc_kwargs(cfg, cfg["img_size"], cfg["patch_size"])
    model = MAE3D(
        arch=cfg["arch"], in_chans=cfg["in_chans"], img_size=cfg["img_size"],
        patch_size=cfg["patch_size"], decoder_dim=cfg["decoder_dim"],
        decoder_depth=cfg["decoder_depth"], decoder_heads=cfg["decoder_heads"],
        masking_ratio=cfg["masking_ratio"], valid_token_threshold=cfg["valid_token_threshold"],
        norm_pix_loss=cfg["norm_pix_loss"], enc_kwargs=ek,
    ).to(device)
    ddp = (nn.parallel.DistributedDataParallel(model, device_ids=[local]) if world > 1 else model)
    core = ddp.module if world > 1 else ddp
    if cfg.get("grad_checkpointing"):
        core.encoder.grad_checkpointing = True      # activation recompute; same maths

    # ── data ──
    loader, sampler, n_vol = T.build_loader(
        cfg["index_json"], cfg["batch_size_per_gpu"], world, rank, num_workers=cfg["num_workers"])
    grad_accum = cfg.get("grad_accum", 1)
    steps_per_epoch = max(1, len(loader) // grad_accum)
    total_steps = int(cfg["total_steps"])
    global_vol = cfg["batch_size_per_gpu"] * world * grad_accum   # volumes per optimizer step
    peak_lr = O.compute_peak_lr(cfg["base_lr"], global_vol)

    # ── optimizer ──
    opt = O.build_adamw(core, peak_lr, **T.opt_kwargs(cfg))
    clip_max = T.clip_grad_max(cfg)
    wd = float(cfg["weight_decay"])                 # constant; set once in the param groups
    amp_dtype = torch.bfloat16

    # ── logging ──
    lr_sum = O.lr_scaling_summary(cfg["base_lr"], global_vol, peak_lr)
    S = cfg["img_size"]
    aug_cfg = cfg["aug"]
    if T.is_main(rank):
        print(f"[MAE] n_vol={n_vol} world={world} bs/gpu={cfg['batch_size_per_gpu']} "
              f"accum={grad_accum} global_vol={global_vol} steps/ep={steps_per_epoch} "
              f"total_steps={total_steps} peak_lr={peak_lr:.3e}", flush=True)
        print(f"[MAE] lr_scaling={lr_sum}", flush=True)
        print(f"[MAE] encoder: arch={cfg['arch']} {T.stabilizer_summary(cfg, ek)}", flush=True)
        print(f"[MAE] objective: masking_ratio={cfg['masking_ratio']} norm_pix_loss={cfg['norm_pix_loss']}"
              f" decoder={cfg['decoder_dim']}x{cfg['decoder_depth']} heads={cfg['decoder_heads']}"
              f" valid_token_threshold={cfg['valid_token_threshold']}", flush=True)
        print(f"[MAE] input: whole volume, zero-padded to a cube and resized to {S}^3"
              f" (deterministic); aug={aug_cfg}", flush=True)

    # ── resume ──
    gstep = 0
    stop = False
    ckpt_every_steps = int(cfg.get("ckpt_every_steps", 5000))
    resume_path = os.path.join(cfg["run_dir"], "checkpoint_last.pt")
    start_epoch = 0
    if os.path.exists(resume_path):
        gstep, done_ep = T.load_resume(resume_path, core, opt, map_location=device)
        start_epoch = done_ep + 1
        if T.is_main(rank):
            print(f"[MAE] RESUME from ep{done_ep} gstep{gstep} (checkpoint_last)", flush=True)

    # ── training loop (step-driven) ──
    model.train()
    epoch = start_epoch
    while gstep < total_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        opt.zero_grad(set_to_none=True)
        ep_t0 = time.time()
        for it, (vols, _bboxes, _vidx) in enumerate(loader):
            with torch.no_grad():
                crops, masks = gpu_wholevol_batch(vols, device, S=S)   # fp16 [B,1,S,S,S] x 2
                x, M = gpu_augment(crops, masks, aug_cfg, aug_gen)     # augmented in fp16
                x = x.float()
            with torch.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                loss, _, aux = ddp(x, M)
            (loss / grad_accum).backward()

            if (it + 1) % grad_accum == 0:
                gn = O.clip_grad_norm(core.parameters(), clip_max)
                lr = O.set_lr(opt, O.lr_at(gstep, total_steps, peak_lr, **T.sched_kwargs(cfg)))
                opt.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1
                if T.is_main(rank) and gstep % cfg.get("log_every", 20) == 0:
                    msg = (f"[MAE] ep{epoch} step{gstep}/{total_steps} loss={float(loss):.4f} "
                           f"gn={float(gn):.2f} lr={lr:.3e} wd={wd:.3f} "
                           f"valid_frac={float(aux['valid_frac']):.3f}")
                    if gstep % cfg.get("feat_every", 200) == 0:
                        # Collapse monitor (feature covariance rank) and first-block
                        # attention statistics; diagnostics only, no effect on training.
                        n = min(64, x.shape[0])
                        feat = core.encode_features(x[:n], M[:n])
                        h = O.feature_covariance_rank(feat)
                        msg += f" rank={h['rank']} eff_rank={h['eff_rank']:.1f}"
                        msg += O.block0_msg(O.block0_attention_stats(core.encoder, x))
                    print(msg, flush=True)
                if T.is_main(rank) and gstep % ckpt_every_steps == 0:
                    T.save_resume(resume_path, core, opt, gstep, epoch)
                    print(f"[MAE] checkpoint_last saved @ step{gstep}", flush=True)
                if gstep >= total_steps:
                    stop = True
                    break

        if T.is_main(rank):
            print(f"[MAE] EPOCH {epoch} time {time.time() - ep_t0:.1f}s", flush=True)
        if stop:
            break
        epoch += 1

    # ── final checkpoint: encoder only ──
    if T.is_main(rank):
        enc_sd = core.encoder.state_dict()
        state = {"model.encoder." + k: v.cpu() for k, v in enc_sd.items()}
        path = os.path.join(cfg["run_dir"], "checkpoints", f"step{total_steps:06d}.ckpt")
        T.save_ckpt(path, {"state_dict": state,
                           "meta": T.encoder_meta(cfg, "mae", core.encoder, lr_sum, total_steps)})
        print(f"[MAE] saved {path}", flush=True)
        print(f"[MAE] peak_mem={T.peak_mem_gb():.1f} GB/gpu", flush=True)
        T.rm_file(resume_path)                       # final checkpoint written: drop the resume state
    T.barrier(world)
    T.cleanup(world)


if __name__ == "__main__":
    main()
