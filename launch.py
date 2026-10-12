#!/usr/bin/env python
"""Launch one pre-training run on a single multi-GPU node.

The four templates in ``configs/`` hold every hyper-parameter of the study except the ones
that vary between cells.  This script resolves a template into a complete config, writes it
to ``<run_dir>/config.yaml`` and starts ``torchrun`` on ``<method>/train.py``, capturing
stdout and stderr in ``<run_dir>/train.out``.  The training script reads nothing but that
config file, so a run is fully described by its run directory.

What the launcher fills in:

* ``arch`` and, with it, the MAE decoder size (:data:`MAE_DEC`) or the JEPA predictor size
  (:data:`JEPA_PRED`).
* ``qk_norm``: on for every ViT-Huge run and for every SimCLR run, off otherwise
  (``--qk-norm on|off`` overrides the rule).
* ``base_lr``, the data index (absolute path) and the loader knobs.
* ``total_steps`` = ``epochs * steps_per_epoch`` where one epoch is the number of optimizer
  steps the distributed loader yields (``--total-steps`` sets the budget directly).  An
  index too small for one optimizer step per epoch is rejected, because the training loop
  could never advance on it.
* ``run_dir`` = ``<logs_root>/<run_name>`` with the default run name
  ``{method}_vit-{arch}_data-{tag}_ep-{E}_lr-{base_lr}`` (``steps-{N}`` replaces ``ep-{E}``
  under ``--total-steps``; ``_qk-on`` / ``_qk-off`` is appended when ``--qk-norm`` deviates
  from the rule above).  ``tag`` defaults to the stem of the index file.

Launching again into an existing run directory (same arguments, or the same ``--run-name``)
rewrites ``config.yaml`` from the current arguments and truncates ``train.out``;
``<method>/train.py`` then resumes from ``checkpoint_last.pt`` when that file exists.  Pass
the same arguments when resuming: the config is re-resolved every time, so for example a
different ``--epochs`` changes ``total_steps`` and with it the learning-rate schedule of the
resumed run.

Example::

    python launch.py --method mae --arch base --index-json data/train_index.json --epochs 100

``build_config(args)`` is importable so the resolution logic can be tested without
launching anything.
"""

import argparse
import os
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from shared.data import load_index  # noqa: E402

METHODS = ("mae", "jepa", "dino", "simclr")
ARCHS = ("tiny", "base", "huge")

# MAE decoder per encoder architecture: (decoder_dim, decoder_depth, decoder_heads).
# Width grows with the encoder (tiny 64 / base 256 / huge 512); head_dim is 64 throughout.
MAE_DEC = {"tiny": (64, 4, 1), "base": (256, 6, 4), "huge": (512, 8, 8)}

# JEPA predictor per encoder architecture: (pred_embed_dim, pred_depth, pred_num_heads).
# Width never exceeds the encoder width; head_dim is 32 throughout.
JEPA_PRED = {"tiny": (192, 4, 6), "base": (384, 6, 12), "huge": (384, 12, 12)}


def qk_norm_default(method, arch):
    """QK-normalisation rule of the study: every ViT-Huge run and every SimCLR run."""
    return arch == "huge" or method == "simclr"


def steps_per_epoch(n_vol, world, batch_size_per_gpu, grad_accum=1):
    """Optimizer steps per epoch, matching the training loader.

    ``DistributedSampler(drop_last=True)`` gives ``n_vol // world`` volumes per rank,
    ``DataLoader(drop_last=True)`` turns them into ``(n_vol // world) // batch_size_per_gpu``
    batches, and ``grad_accum`` batches form one optimizer step.  The ``max(1, ...)`` floor
    mirrors the ``steps_per_epoch`` the training scripts compute for their log line; it is
    never reached from :func:`build_config`, which rejects an index that yields fewer than
    one optimizer step per epoch.
    """
    return max(1, ((int(n_vol) // int(world)) // int(batch_size_per_gpu)) // int(grad_accum))


def fmt_lr(lr):
    """Compact scientific notation for run names (``1e-4``, ``2.5e-4``, ``1e-3``)."""
    mant, exp = f"{float(lr):.6e}".split("e")
    mant = mant.rstrip("0").rstrip(".")
    return f"{mant}e{int(exp)}"


def build_parser():
    p = argparse.ArgumentParser(
        description="Resolve a method config template and launch torchrun on <method>/train.py.")
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--arch", required=True, choices=ARCHS)
    p.add_argument("--index-json", required=True,
                   help="JSON list of pre-processed .b2nd volume paths (see tools/make_index.py)")
    p.add_argument("--epochs", type=int, default=None,
                   help="training budget in epochs; total_steps = epochs * steps_per_epoch")
    p.add_argument("--total-steps", type=int, default=None,
                   help="training budget in optimizer steps (takes precedence over --epochs)")
    p.add_argument("--data-tag", default=None,
                   help="label of the data index in the run name (default: index file stem)")
    p.add_argument("--base-lr", type=float, default=1e-4,
                   help="base learning rate; peak_lr = base_lr * volumes_per_step / 256")
    p.add_argument("--nproc", type=int, default=8, help="GPUs (processes) on this node")
    p.add_argument("--logs-root", default="./runs", help="parent directory of run directories")
    p.add_argument("--qk-norm", default="auto", choices=("auto", "on", "off"),
                   help="QK-normalisation; auto = on for ViT-Huge and for SimCLR, off otherwise")
    p.add_argument("--batch-size", type=int, default=None,
                   help="volumes per GPU per micro-batch (template default 64)")
    p.add_argument("--grad-accum", type=int, default=None,
                   help="micro-batches per optimizer step (template default 1)")
    p.add_argument("--num-workers", type=int, default=None,
                   help="DataLoader workers per rank (template default 4)")
    p.add_argument("--run-name", default=None, help="run directory name (default: derived)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the resolved config and command without writing or launching")
    return p


def build_config(args):
    """Resolve ``configs/<method>.yaml`` for ``args``.  Returns ``(cfg, run_dir)``.

    ``args`` is the parsed namespace of :func:`build_parser`; attributes that are not present
    fall back to the parser defaults, so a minimal namespace with ``method``, ``arch``,
    ``index_json`` and ``epochs`` (or ``total_steps``) is enough.
    """
    method, arch = args.method, args.arch
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}; expected one of {ARCHS}")

    with open(os.path.join(ROOT, "configs", f"{method}.yaml"), "r") as f:
        cfg = yaml.safe_load(f)

    # ── architecture ──
    cfg["method"] = method
    cfg["arch"] = arch
    if method == "mae":
        cfg["decoder_dim"], cfg["decoder_depth"], cfg["decoder_heads"] = MAE_DEC[arch]
    elif method == "jepa":
        cfg["pred_embed_dim"], cfg["pred_depth"], cfg["pred_num_heads"] = JEPA_PRED[arch]
    qk_auto = qk_norm_default(method, arch)
    qk_choice = getattr(args, "qk_norm", "auto") or "auto"
    qk_norm = qk_auto if qk_choice == "auto" else (qk_choice == "on")
    cfg["qk_norm"] = bool(qk_norm)

    # ── optimisation / data ──
    cfg["base_lr"] = float(getattr(args, "base_lr", 1e-4))
    index_json = os.path.abspath(args.index_json)
    cfg["index_json"] = index_json
    for attr, key in (("batch_size", "batch_size_per_gpu"), ("grad_accum", "grad_accum"),
                      ("num_workers", "num_workers")):
        v = getattr(args, attr, None)
        if v is not None:
            cfg[key] = int(v)

    # ── budget (loader overrides above must be applied before this) ──
    nproc = int(getattr(args, "nproc", 8))
    n_vol = len(load_index(index_json))
    bs = int(cfg["batch_size_per_gpu"])
    accum = int(cfg.get("grad_accum", 1))
    # The loaders drop incomplete batches and the training loop counts a step only once
    # ``grad_accum`` batches have been seen within one epoch, so an index that yields no step
    # per epoch would leave every rank looping over empty epochs forever.  Refuse it here.
    per_rank_batches = (n_vol // nproc) // bs
    if per_rank_batches // accum == 0:
        raise ValueError(
            f"index {index_json} has {n_vol} volumes: with --nproc {nproc}, batch_size_per_gpu "
            f"{bs} and grad_accum {accum} each rank gets {n_vol // nproc} volumes = "
            f"{per_rank_batches} batch(es) per epoch, fewer than one optimizer step; reduce "
            f"--batch-size / --grad-accum / --nproc or use a larger index")
    spe = steps_per_epoch(n_vol, nproc, bs, accum)
    total_steps = getattr(args, "total_steps", None)
    epochs = getattr(args, "epochs", None)
    if total_steps is not None:
        cfg["total_steps"] = int(total_steps)
        budget = f"steps-{int(total_steps)}"
    elif epochs is not None:
        cfg["total_steps"] = int(epochs) * spe
        budget = f"ep-{int(epochs)}"
    else:
        raise ValueError("one of --epochs or --total-steps is required")

    # ── run name / directory ──
    data_tag = getattr(args, "data_tag", None) or os.path.splitext(os.path.basename(index_json))[0]
    run_name = getattr(args, "run_name", None)
    if not run_name:
        run_name = f"{method}_vit-{arch}_data-{data_tag}_{budget}_lr-{fmt_lr(cfg['base_lr'])}"
        if qk_norm != qk_auto:
            run_name += "_qk-on" if qk_norm else "_qk-off"
    run_dir = os.path.join(getattr(args, "logs_root", "./runs"), run_name)
    cfg["run_dir"] = run_dir
    return cfg, run_dir


def launch_env(nproc, seed):
    """Environment for the ``torchrun`` children.

    One BLAS / OpenMP thread per process (``nproc`` ranks plus their loader workers would
    otherwise oversubscribe the host), a fixed ``PYTHONHASHSEED`` (it must be set before the
    interpreter starts to take effect) and, unless the caller already restricted the visible
    GPUs, the first ``nproc`` devices.
    """
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1", "PYTHONHASHSEED": str(int(seed))})
    env.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(int(nproc))))
    return env


def build_command(method, nproc, cfg_path):
    return [sys.executable, "-m", "torch.distributed.run", "--standalone",
            f"--nproc_per_node={int(nproc)}", os.path.join(ROOT, method, "train.py"),
            "--config", cfg_path]


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.epochs is None and args.total_steps is None:
        raise SystemExit("launch.py: one of --epochs or --total-steps is required")
    try:
        cfg, run_dir = build_config(args)
    except ValueError as e:
        raise SystemExit(f"launch.py: {e}") from None
    cfg_path = os.path.join(run_dir, "config.yaml")
    cmd = build_command(args.method, args.nproc, cfg_path)
    env = launch_env(args.nproc, cfg.get("seed", 42))

    volumes_per_step = int(cfg["batch_size_per_gpu"]) * int(args.nproc) * int(cfg.get("grad_accum", 1))
    print(f"[launch] run_dir={run_dir}")
    print(f"[launch] total_steps={cfg['total_steps']} volumes_per_step={volumes_per_step}"
          f" qk_norm={cfg['qk_norm']}")
    print(f"[launch] cmd={' '.join(cmd)}")
    if args.dry_run:
        print("[launch] resolved config:")
        print(yaml.safe_dump(cfg, sort_keys=False), end="")
        print("[launch] dry run: nothing written, nothing launched")
        return 0

    # An existing run directory is reused: config.yaml is rewritten and train.out truncated;
    # train.py resumes from checkpoint_last.pt if it is present (see the module docstring).
    os.makedirs(run_dir, exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    out_path = os.path.join(run_dir, "train.out")
    print(f"[launch] config={cfg_path}")
    print(f"[launch] log={out_path}", flush=True)
    with open(out_path, "w") as out:
        result = subprocess.run(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, check=False)
    print(f"[launch] finished rc={result.returncode} log={out_path}")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
