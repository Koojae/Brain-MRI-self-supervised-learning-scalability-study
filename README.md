# Self-supervised pre-training of a shared 3D ViT on brain MRI

This repository contains the pre-training code of a controlled comparison of four
self-supervised learning (SSL) methods on skull-stripped 3D brain MRI: masked autoencoding
(**MAE-96**, `mae`), joint-embedding prediction (**JEPA-96**, `jepa`), self-distillation
(**DINO**, `dino`) and instance contrastive learning (**SimCLR-new**, `simclr`). All four
methods train the *same* 3D Vision Transformer encoder (`shared/vit3d.py`) with the same data
pipeline, GPU augmentation, optimizer, learning-rate schedule and seeding; only the objective
and the way input views are built differ. After pre-training, only the encoder is kept and
transferred (the MAE decoder, JEPA predictor, DINO heads and SimCLR projection head are
discarded). The study ran every method at three encoder sizes (ViT-Tiny / Base / Huge) and
several data-fraction x epoch budgets on one 8-GPU node. Below, MAE / JEPA / SimCLR refer to
MAE-96 / JEPA-96 / SimCLR-new.

## Repository layout

```
code_pretrain_final/
├── launch.py                  resolve a config template and start torchrun on <method>/train.py
├── configs/
│   ├── mae.yaml               MAE-96 hyper-parameters (ViT-Base template)
│   ├── jepa.yaml              JEPA-96 hyper-parameters
│   ├── dino.yaml              DINO hyper-parameters
│   └── simclr.yaml            SimCLR-new hyper-parameters
├── shared/                    code identical for all four methods
│   ├── vit3d.py               the 3D ViT encoder (RoPE, stabilized init, depth rescale, QK-norm, valid-token packing)
│   ├── data.py                .b2nd loading, whole-volume resize, random resized crops, brain-valid token mask
│   ├── augment.py             GPU augmentation (flip / rot90, gamma / histogram shift, noise / blur)
│   ├── optim.py               AdamW parameter groups, lr schedule, EMA, gradient clipping, diagnostics
│   └── trainutil.py           DDP set-up, seeding, DataLoader, checkpoint helpers, config adapters
├── mae/                       train.py, model.py            (MAE3D)
├── jepa/                      train.py, model.py, masks.py  (JEPA3D, NeuroVFM two-way masking)
├── dino/                      train.py, model.py, views.py  (DINO3DSSL, multi-crop views)
│   └── dinov2/                DINOv2 head and losses, vendored verbatim (Apache-2.0, LICENSE included)
├── simclr/                    train.py, model.py, loss.py, views.py  (SimCLRModel, NT-Xent)
├── tools/
│   ├── make_index.py          build the JSON data index
│   └── load_encoder.py        load the encoder from any of the four final checkpoints
├── requirements.txt
├── THIRD_PARTY_NOTICES.md
└── README.md
```

Every `train.py` reads exactly one resolved `config.yaml` (written by `launch.py`) and imports
only from `shared/` and its own package, so a run is fully described by its run directory.

## Installation

Python 3.11. Tested with `torch 2.11.0+cu128`, `numpy 2.4.4`, `blosc2 4.7.0` and
`PyYAML 6.0.3` (pinned in `requirements.txt`). Training uses `torch.distributed` with the
NCCL backend, one process per GPU; the launcher targets a single multi-GPU node
(`torchrun --standalone`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # pick the torch wheel matching your CUDA driver
```

## Data

The pre-processing pipeline is **not** part of this release. The training code expects one
file per volume in the following format:

* a [blosc2](https://www.blosc.org/python-blosc2/) `.b2nd` file holding a 3D `float16` array
  (opened with `blosc2.open(path)[:]`);
* skull-stripped, with the background **exactly 0** (the pipeline derives the brain mask from
  `x > 0`, and every masking objective and the encoder's valid-token logic depend on it);
* intensities normalised to `[0, 1]`;
* tightly cropped to the brain bounding box (typical shapes 130-180 voxels per axis);
* approximately 1 mm isotropic spacing.

The volumes used in the study were produced by N4 bias-field correction, resampling to 1 mm
isotropic, HD-BET brain extraction, per-volume normalisation to the brain 99th percentile
followed by clipping to `[0, 1]`, skull-stripping, bounding-box cropping and conversion to
fp16 blosc2.

A training set is a **JSON index**: a list of absolute `.b2nd` paths. Build one with

```bash
python tools/make_index.py --root /data/preprocessed --out data/train_index.json
# --pattern "**/*.b2nd" (default) is a recursive glob relative to --root
```

Paths are sorted before writing, so the index (and with it the data order of a seeded run)
does not depend on file-system enumeration order. The reader (`shared.data.load_index`) also
accepts entries that are dicts with an `img_path` key.

The data-fraction subsets of the study (100 / 30 / 10 / 3 %) were separate index files. The
code does no splitting of its own: **subsets and any held-out sets must be built by the user
at the subject level** (all scans of a subject in one split).

## Quick start

`launch.py` resolves `configs/<method>.yaml` into a complete config, writes it to
`<run_dir>/config.yaml` and starts `torchrun`:

```bash
# one cell of the study: MAE-96, ViT-Base, 100 epochs on 8 GPUs
python launch.py --method mae    --arch base --index-json data/train_index.json --epochs 100

# the other methods / sizes
python launch.py --method jepa   --arch tiny --index-json data/train_index.json --epochs 100
python launch.py --method dino   --arch huge --index-json data/train_index.json --epochs 100
python launch.py --method simclr --arch base --index-json data/train_index.json --epochs 100

# inspect the resolved config and the torchrun command without writing or launching
python launch.py --method dino --arch base --index-json data/train_index.json --epochs 100 --dry-run
```

Full CLI:

```
python launch.py --method {mae,jepa,dino,simclr} --arch {tiny,base,huge} --index-json PATH
                 (--epochs E | --total-steps N)
                 [--data-tag TAG] [--base-lr 1e-4] [--nproc 8] [--logs-root ./runs]
                 [--qk-norm {auto,on,off}] [--batch-size 64] [--grad-accum 1] [--num-workers 4]
                 [--run-name NAME] [--dry-run]
```

`--nproc` is the number of processes (GPUs) `torchrun` starts. **DINO must run with at least
two processes** (`--nproc >= 2`): the vendored iBOT Sinkhorn-Knopp step all-reduces
unconditionally and `dino/train.py` exits otherwise. MAE, JEPA and SimCLR also run
single-process (`--nproc 1`).

What the launcher fills in:

| item | rule |
|---|---|
| `arch` | `--arch`; also sets the MAE decoder (`MAE_DEC`) or JEPA predictor (`JEPA_PRED`) size for that arch |
| `qk_norm` | `auto` = on for ViT-Huge and for every SimCLR run, off otherwise; `on` / `off` override |
| `total_steps` | `epochs * steps_per_epoch` with `steps_per_epoch = ((n_vol // nproc) // batch_size) // grad_accum`, or `--total-steps` directly; `--total-steps` takes precedence when both are given (the run name then uses `steps-{N}`) |
| `run_dir` | `<logs_root>/<run_name>`; default run name `{method}_vit-{arch}_data-{tag}_ep-{E}_lr-{base_lr}` (e.g. `mae_vit-base_data-train_index_ep-100_lr-1e-4`); `steps-{N}` replaces `ep-{E}` under `--total-steps`; `_qk-on` / `_qk-off` is appended when `--qk-norm` deviates from the rule; `tag` defaults to the index file stem |
| environment | `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=1`, `PYTHONHASHSEED=<seed>`, and `CUDA_VISIBLE_DEVICES=0..nproc-1` unless already set |

`--batch-size`, `--grad-accum` and `--num-workers` override the template values (64 / 1 / 4)
*before* `steps_per_epoch` is computed. The launcher prints `[launch] ...` lines
(run directory, total steps, volumes per step, the command) and then blocks until training
ends, propagating the exit code of `torchrun`. **All training output (stdout and stderr of
every rank) goes to `<run_dir>/train.out`**, not to the terminal.

### Running `torchrun` directly

`launch.py` runs `python -m torch.distributed.run --standalone --nproc_per_node=<nproc>
<repo>/<method>/train.py --config <run_dir>/config.yaml`. Given a resolved config you can
start the same thing yourself:

```bash
torchrun --standalone --nproc_per_node=8 mae/train.py --config runs/<run>/config.yaml
```

The training script reads nothing but the config file (in particular `index_json`,
`total_steps` and `run_dir`, which are `null` in the templates), so use a config produced by
`launch.py` (or `--dry-run`) rather than a template. Set the environment variables listed
above yourself; `PYTHONHASHSEED` must be set before the interpreter starts. `dino/train.py`
requires `--nproc_per_node >= 2` (see above); the other three scripts also run with
`--nproc_per_node=1`.

### Resuming

Every `ckpt_every_steps` (5000) optimizer steps, rank 0 writes `<run_dir>/checkpoint_last.pt`
(full model, optimizer state, step and epoch counters; one rolling file, overwritten each
time). **If a run is launched again with the same run directory** (same arguments, or the same
`--run-name`), `train.py` finds `checkpoint_last.pt`, restores it on every rank and continues
from the saved step count at the beginning of the next epoch; the remainder of the
interrupted epoch is not replayed. `total_steps` stays the same only if the same budget flags
are given: `config.yaml` is re-resolved from the current arguments at every launch, so a
different `--epochs` / `--total-steps` (or `--base-lr`) changes `total_steps` and with it the
learning-rate, EMA and teacher-temperature schedules of the resumed run. The file is deleted once
the final encoder checkpoint has been written. A resumed run continues from identical weights
and optimizer state, but no random stream is saved: the augmentation, crop-centre / scale and
iBOT-mask generators, the JEPA mask call counter and the global torch RNG (MAE masking)
restart from their seeds, so a resumed run is not bit-identical to an uninterrupted one.
Relaunching into an existing run directory rewrites `config.yaml` and truncates `train.out`;
copy the previous log first if you need it.

## The study grid

One optimizer step processes `batch_size_per_gpu x nproc x grad_accum` volumes; the study
used 64 volumes per GPU on 8 GPUs (512 volumes per step). The budget is given in epochs and
converted to steps by the launcher, `steps_per_epoch = ((n_vol // 8) // 64)`:

| data fraction | volumes | steps / epoch |
|---|---|---|
| data-100 | 844,938 | 1,650 |
| data-30 | 253,481 | 495 |
| data-10 | 84,494 | 165 |
| data-3 | 25,348 | 49 |

| cell (data % x epochs) | total steps | launch flags |
|---|---|---|
| 100 x 100 | 165,000 | `--epochs 100` on the full index |
| 30 x 100 | 49,500 | `--epochs 100` on the 30 % index |
| 10 x 100 | 16,500 | `--epochs 100` on the 10 % index |
| 3 x 100 | 4,900 | `--epochs 100` on the 3 % index |
| 100 x 300 | 495,000 | `--epochs 300` |
| 100 x 30 | 49,500 | `--epochs 30` |
| 100 x 10 | 16,500 | `--epochs 10` |
| 100 x 3 | 4,950 | `--epochs 3` |

Each cell was run for the four methods and, where budget allowed, the three encoder sizes.
`base_lr` was `1e-4` for every method (peak lr `2e-4` at 512 volumes per step; see below).

**Encoder sizes** (`shared.vit3d.VIT_FAMILY`, patch size 16; `small` and `large` are defined
but were not part of the study and are not accepted by `launch.py`):

| arch | embed dim | MLP dim | heads (head dim) | depth | encoder parameters |
|---|---|---|---|---|---|
| tiny | 192 | 768 | 3 (64) | 12 | 6.13 M |
| base | 768 | 3072 | 12 (64) | 12 | 88.20 M |
| huge | 1280 | 5120 | 16 (80) | 32 | 634.93 M |

Counts with `qk_norm` off. QK-normalisation adds two `LayerNorm(head_dim)` layers (weight and
bias) per block, i.e. `4 x head_dim` parameters per block: +3,072 in total for tiny / base
(12 blocks) and +10,240 for huge (32 blocks). The ViT-Huge encoders of the study (`qk_norm` on
for every method) therefore have 634.94 M parameters and the SimCLR-Base encoder 88.21 M;
SimCLR-Tiny stays at 6.13 M when rounded.

**QK-normalisation rule.** The only encoder setting that differs between cells is `qk_norm`:
it is **on for every ViT-Huge run** (all four methods; without it the first-block attention
logits of the 32-block encoder diverge) and **on for every SimCLR run** (all three sizes), and
off otherwise. `launch.py --qk-norm auto` implements this rule. Everything else (initialisation,
RoPE, depth rescale, clipping, schedule, `min_lr`, `patch_embed_lr_mult`, weight decay, EMA)
is identical across the three sizes of a method.

## What is identical across methods

### Encoder (`shared/vit3d.py`)

* `Conv3d` patch embedding, patch `16^3`; a `96^3` input gives a `6^3 = 216` token grid.
* Pre-norm transformer blocks, attention through `F.scaled_dot_product_attention`, GELU MLP
  with hidden size 4x. No LayerScale, no DropPath, no register tokens, no SwiGLU.
* **3D axial RoPE** instead of an additive positional embedding. Queries and keys are rotated
  before the dot product; the head dimension is split into three axis groups of
  `2 * ((head_dim // 3) // 2)` channels (64 -> 20/20/20 with 4 unrotated channels, 80 ->
  26/26/26 with 2). Frequencies are interleaved so that adjacent channel pairs share one
  frequency (translation-invariant logits). The class token is never rotated. Coordinates of a
  smaller token grid (DINO local crops, `3^3`) are rescaled onto the base grid (`6^3`: 0, 2.5,
  5), so different crop sizes share one coordinate system. The MAE decoder and the JEPA
  predictor also use RoPE (no additive tables anywhere).
* **Stabilized initialisation**: every `nn.Linear` and the patch-embedding convolution use
  `trunc_normal(std=0.02)`; the fused `qkv` projection uses the separate-fan uniform rule of
  MoCo v3 (effective std `1/sqrt(D)`); biases are zero; LayerNorm keeps PyTorch defaults.
* **Residual depth rescaling** (V-JEPA 2): after initialisation, `attn.proj.weight` and
  `mlp.fc2.weight` of block `l` (1-indexed) are divided by `sqrt(2 l)`.
* **QK-normalisation** (optional, see the rule above): a per-head `LayerNorm` on q and k
  before the dot product, with **`eps = 1e-2`** instead of the PyTorch default `1e-5`.
  Skull-stripped background patches are exactly zero; with `eps = 1e-5` the LayerNorm backward
  on those tokens amplifies the patch-embedding gradient by more than an order of magnitude.
  When off, the norms are `nn.Identity` and the `state_dict` has no extra keys.
* `cls_token` and `mask_token` parameters exist in every encoder; which method uses or freezes
  them is listed per method below.

### Brain mask and valid tokens (`shared/data.py`)

Trilinear resizing bleeds signal into the background, so the pipeline never derives the brain
mask from a resized image. Every crop or resized volume `x` travels with a **crisp mask `M`**:
the exact source mask `(window > 0)` is resized with the same trilinear operator, thresholded
at 0.5, and `x` is multiplied by `M`. After all augmentation ops `x` is multiplied by the
(flipped / rotated) mask once more, so the background is exactly zero at the output.

`token_valid(M, 16, threshold=0.1)` marks a `16^3` patch as **brain-valid when at least 10 %
of its voxels are brain** (`valid_token_threshold: 0.1` in every run). Rows without a valid
token fall back to all-valid. Consequences:

* the encoder drops invalid tokens (left-packed gather), excludes padded slots from the
  attention keys and scatters the outputs back to the full grid with exact zeros at invalid
  positions; when every token is valid the plain full-grid path (flash kernel) is used;
* MAE masking, JEPA context / target blocks and iBOT masks are drawn from valid tokens only;
* the same threshold is stored in the checkpoint `meta` so downstream code can reproduce it.

### Augmentation (`shared/augment.py`)

Applied on the GPU, per sample, from a per-rank `torch.Generator`; identical settings in all
four configs. Order: geometry -> contrast -> image quality -> `x * M`.

| group | op | probability | range |
|---|---|---|---|
| geometry | flip, each of the 3 axes independently (`x` and `M`) | `flip_p 0.5` per axis | |
| geometry | 90-degree rotation, each of the 3 axis pairs independently (`x` and `M`) | `rotate_p 0.5` per pair | `k` uniform in {1, 2, 3} |
| contrast | gamma `clamp(x, 0, 1) ** g` | `gamma_p 0.5` | `g ~ U[0.5, 2.0]` |
| contrast | monotone piecewise-linear histogram shift, end points fixed | `hist_shift_p 0.5` | 10 control points, segment lengths `~ U[0.7, 1.3]` renormalised |
| image quality | additive Gaussian noise | `noise_p 0.5` | `std ~ U[0, 0.05]` |
| image quality | separable Gaussian blur | `blur_p 0.5` | `sigma ~ U[0.25, 1.0]` voxels |

MAE and JEPA augment the single whole-volume view; DINO and SimCLR augment every crop
independently with the same op set (no global / local asymmetry). The settings are identical,
but the arithmetic precision differs between the two input pipelines: the whole-volume pipeline
(MAE, JEPA) augments the fp16 tensors produced by the resize and casts to fp32 afterwards, the
crop pipeline (DINO, SimCLR) casts the crops to fp32 before augmenting.

### Optimizer and schedule (`shared/optim.py`)

* **AdamW**, `beta1 = 0.9`, `eps = 1e-8`; `beta2` and the weight decay are per method (table
  below). Two parameter classes: tensors with `ndim >= 2` receive weight decay -- the weights
  and also the class / mask tokens (`[1, 1, D]` / `[1, D]`) and the MAE decoder / JEPA
  predictor mask tokens; 1-D tensors (biases, LayerNorm affine parameters) receive none.
  Weight decay is constant over training.
* **Patch-embedding lr multiplier** `patch_embed_lr_mult = 0.2` on the encoder's
  `patch_embed` parameters.
* **Peak lr** `= base_lr * volumes_per_step / 256` (linear scaling in *volumes*; the number of
  crops per volume does not enter). With `base_lr 1e-4` and 512 volumes per step the peak is
  `2e-4`.
* **Schedule**: linear warm-up over the first 10 % of `total_steps` (factor `(step + 1) /
  warm`), then cosine decay to the absolute floor `min_lr = 1e-6`.
* **Gradient clipping** by global norm, `clip_grad = 3.0`, every step.
* **Mixed precision**: forward / backward under `torch.autocast(dtype=torch.bfloat16)`;
  parameters and optimizer state in fp32.
* `batch_size_per_gpu 64`, `grad_accum 1`, `num_workers 4` (DataLoader workers only decode;
  all cropping, resizing and augmentation happen on the GPU).

### Seeding and determinism

* `seed: 42`. Each rank seeds `random`, NumPy, torch and CUDA with `seed + rank`;
  `torch.backends.cudnn.deterministic = True`, `benchmark = False`; `PYTHONHASHSEED` is set by
  the launcher.
* Separate per-rank generators: augmentation `torch.Generator(seed + rank)` (all methods);
  crop centres and scales `numpy.random.default_rng(seed * 1000 + rank)` (DINO, SimCLR); iBOT
  masks `torch.Generator(seed * 7 + rank)` (DINO). The JEPA mask generator seeds a fresh
  `torch.Generator` with its call index (0, 1, 2, ...), so its masks depend only on the call
  order and the brain masks (the seed carries no rank offset, so all ranks draw the same mask
  stream; the resulting masks differ only through each rank's brain masks). MAE's random
  masking uses the global torch RNG of the rank (seeded `seed + rank` by `seed_everything`),
  not a dedicated generator.
* Data order comes from `DistributedSampler(shuffle=True, drop_last=True)` with
  `set_epoch(epoch)` every epoch (for `--nproc 1` there is no distributed sampler and the
  DataLoader shuffles with its own sampler drawn from the global torch RNG of rank 0).

### Logging

Rank 0 prints a line every `log_every` (20) steps tagged `[MAE]`, `[JEPA]`, `[DINO]` or
`[SimCLR]` with the loss (and its terms), pre-clip gradient norm, lr and weight decay, and
every `feat_every` (200) steps adds two diagnostics: the numerical / effective rank of the
feature covariance of the encoder features (mean of the brain patch tokens for MAE / JEPA /
DINO, the class token for SimCLR; representation collapse) and first-block
attention statistics (`ent0`, `dc0`, `kvar`, `qtv`, `lstd`; attention-logit blow-up). These
are diagnostics only and do not influence training. At the end, rank 0 prints the peak
allocated GPU memory.

## Methods

### MAE-96 (`mae/`)

* **Input.** Whole volume, zero-padded to a cube whose edge is its longest axis, resized with
  one trilinear factor to `96^3` (`img_size 96`; deterministic, no random crop), then
  augmented. About half of the 216 tokens are brain-valid for typical volumes.
* **Objective.** 3D port of the MAE objective. Among the brain-valid tokens of each volume a
  random 75 % (`masking_ratio 0.75`) is hidden; `n_keep = round(0.25 * n_valid)` (at least 1).
  Because `n_valid` varies per volume, the visible and masked index tables are left-aligned
  and padded to the batch maximum; padded slots are excluded from the attention keys of the
  encoder and the decoder and carry zero loss weight. Background tokens never enter the
  encoder; they enter the loss only in the degenerate case of a volume with a single brain
  token, where `n_mask` is clamped to 1 so the decoder is never empty. The encoder runs on
  the visible tokens only (no class token). The decoder input is
  `[mask_token x n_masked | enc_to_dec(visible)]`, positions through RoPE
  (same coordinates as the encoder, tables rebuilt for the decoder head dimension), then
  `decoder_depth` shared `Block`s (MLP 4x), LayerNorm and `to_pixels` (`16^3 = 4096` voxels
  per token). Target = per-patch normalised voxels (`norm_pix_loss: true`, mean / variance
  over the 4096 voxels, `eps 1e-6`); loss = mean squared error over the real masked slots.
* **Decoder per arch** (`launch.MAE_DEC`, head dim 64): tiny `64 / 4 blocks / 1 head`
  (0.48 M parameters), base `256 / 6 / 4` (5.99 M), huge `512 / 8 / 8` (27.98 M).
* **Optimisation.** `weight_decay 0.05`, `adamw_beta2 0.95`, no EMA. Optimizer over the whole
  model; the encoder's `cls_token` and `mask_token` are unused and frozen.
* **Transferred.** The encoder (`encoder`), saved as `checkpoints/step{total_steps:06d}.ckpt`.

### JEPA-96 (`jepa/`)

* **Input.** Same whole-volume `96^3` pipeline and augmentation as MAE-96; the brain-valid
  token mask `tv` is computed from `M`.
* **Masking** (`jepa/masks.py`, NeuroVFM two-way block masking on the `6^3` grid, CPU). For
  every sample one of two *ways* is drawn uniformly, and exactly one `(context, target)` index
  pair is produced (`num_mask_tokens 1`):

  | | way 1 | way 2 |
  |---|---|---|
  | `hw_block_scale` | `[0.7, 0.7]` | `[0.25, 0.25]` |
  | `switch_enc_pred` | `false`: block = **target**, complement = context | `true`: block = **context**, complement = target |

  Shared per-way knobs: `aspect_ratio [0.3, 3.0]`, `npred 1`, `enc_mask_scale 0.25`,
  `patch_drop_rate 0.2`, `full_depth true`. The block is a full-depth rectangle whose area is
  `hw_block_scale x (brain fraction of the sample)` of the `6 x 6` plane, placed so that it
  covers the brain (position redrawn while the covered foreground is below a ratio that
  starts at 0.95 and decays). Post-processing common to both ways: if the context covers more
  than `enc_mask_scale` of the brain tokens, a contiguous slab of depth slices, sized to bring
  it down to that fraction, is removed from the context at a random offset;
  each remaining context token is dropped with probability `patch_drop_rate` (dropped tokens
  join the target); both sides are confined to brain-valid tokens; a draw is accepted when the
  context holds 5-50 % of the valid tokens and both sides are non-empty (5 attempts, then a
  deterministic split).
* **Padding.** Index lists have per-sample lengths, so they are padded to the fixed lengths
  `pad_enc 64` (context) and `pad_pred 192` (target) by repeating an own index with weight 0;
  the weights are the loss weights (target side) *and* the attention key masks (both sides),
  so repeated indices never act as extra keys and tensor shapes are constant from step to
  step. If a batch exceeds a fixed length the batch maximum is used instead and counted; the
  counters appear in every log line as `pad[combos=... ovf=e/p max=e/p]` and stayed at
  `ovf=0/0` in all runs.
* **Objective.** The target encoder (EMA copy, frozen, no gradient) encodes all 216 tokens
  with the background removed from the attention keys, applies a parameter-free `LayerNorm`
  and gathers the target positions. The context encoder encodes the gathered context tokens
  (key mask = context weights). The predictor embeds them (`D -> P`), appends one learnable
  mask token per target slot, runs `pred_depth` shared `Block`s with RoPE over
  `cat([context_idx, target_idx])` and key mask `[w_enc | w_pred]`, applies LayerNorm and
  projects the target part back to `D`. Loss = **L1** (`loss_exp 1.0`) between prediction
  and target, averaged over real target tokens and channels.
* **Predictor per arch** (`launch.JEPA_PRED`, head dim 32, width never above the encoder
  width): tiny `192 / 4 blocks / 6 heads` (1.85 M parameters), base `384 / 6 / 12` (11.24 M),
  huge `384 / 12 / 12` (22.28 M).
* **Optimisation.** Optimizer over the context encoder and the predictor; after every step
  the target encoder is updated by EMA with **constant momentum 0.99925** (`ema_start =
  ema_end = 0.99925`, `ema_schedule const`). `weight_decay 0.04`, `adamw_beta2 0.999`. DDP
  with `broadcast_buffers=False`. `cls_token` / `mask_token` frozen.
* **Transferred.** The **target (EMA) encoder**, saved as `3d_jepa_vitb_96-final.pth.tar`.

### DINO (`dino/`)

* **Input.** Multi-crop random resized crops centred on a uniformly drawn brain voxel, window
  edge `s x (longest volume axis)`, *centre-fit* placement (the window is shifted inside the
  volume when the axis is long enough, symmetrically zero-padded otherwise), trilinear resize
  with mask tracking, each crop augmented independently:

  | view | count | size | `scale_range` | tokens |
  |---|---|---|---|---|
  | global | `n_global 2` | `global_size 96` | `[0.5, 1.0]` | `6^3 = 216` |
  | local | `n_local 8` | `local_size 48` | `[0.2, 0.5]` | `3^3 = 27` (RoPE coordinates rescaled onto the `6^3` grid) |

  At 64 volumes per GPU this is 128 global and 512 local crops per GPU per step.
* **Objective** (DINOv2, heads and losses vendored in `dino/dinov2/`). Student and teacher
  each have a backbone, a DINO head and a separate iBOT head (`DINOHead`: MLP `D -> 2048 ->
  2048 -> 256`, `head_nlayers 3`, L2 normalisation, weight-normalised prototype layer with
  `65536` prototypes for both heads; 23.14 M parameters per head at ViT-Base). Readout = class
  token. Per step: the teacher (no gradient) sees the two global crops without masks; its
  class-token and masked-patch outputs are centred with **Sinkhorn-Knopp** (`sinkhorn_iterations
  3`, normalised across ranks with `all_reduce`) at the teacher temperature. The student sees
  the two global crops with the iBOT mask and the eight local crops. Loss = `dino + ibot +
  koleo_weight * koleo` with
  * `dino`: cross-entropy of every student view against every *other* teacher global crop,
    averaged over the `(2 - 1 + 8) x 2 = 18` pairs (`student_temp 0.1`);
  * `ibot`: masked-patch cross-entropy on the global crops with DINOv2's `forward_masked`
    normalisation (per-sample `1 / n_masked` weights, divided by the number of global-crop
    samples), averaged over the two global crops;
  * `koleo`: KoLeo regulariser on the student's pre-head global class-token features, summed
    over the two global crops, `koleo_weight 0.1`.
* **iBOT masking.** Each global-crop sample is masked with probability
  `ibot_mask_sample_probability 0.5`; the masked fraction `~ U[0.2, 0.75]`
  (`ibot_mask_ratio_min_max`) applies to that sample's **valid** token count and the masked
  positions are a random subset of its valid tokens, replaced by `mask_token` in the student
  input.
* **Schedules.** Teacher temperature linear `0.04 -> 0.07` over the first 30 % of the steps
  (`warmup_teacher_temp`, `teacher_temp`, `warmup_teacher_temp_frac 0.3`), then constant. The
  prototype layers (`last_layer`) of both student heads receive no gradient for the first
  `freeze_last_layer_steps 1500` steps. Teacher = EMA of the student (backbone and both heads)
  with **cosine momentum `0.992 -> 1.0`** (`ema_start 0.992`, `ema_end 1.0`, `ema_schedule
  cosine`), updated after every optimizer step.
* **Optimisation.** Optimizer over the three student modules; `weight_decay 0.04`,
  `adamw_beta2 0.999`. `cls_token` (readout) and `mask_token` (iBOT replacement token) are
  both trained. Same-resolution crops are concatenated so the backbone runs three
  times per step (teacher globals, student globals, student locals) and the heads once per
  role; padded slots from valid-token packing are excluded from the keys, so this equals the
  per-crop computation. DDP with `find_unused_parameters=False` (the iBOT head keeps a
  zero-valued gradient path when a crop has no masked patch, and the head and Sinkhorn are
  always called so every rank issues the same collectives).
* **Transferred.** The **teacher backbone (EMA)**, saved as `teacher_backbone_final.pt`.

### SimCLR-new (`simclr/`)

* **Input.** `n_views 2` random resized crops per volume (`crop_size 96`, `scale_range [0.5,
  1.0]`, centre-fit, same crop operator as DINO), each with its own centre and scale, each
  augmented once (a single `gpu_augment` call over the `64 x 2 = 128` crops of a GPU). The two
  crops of a volume are the positive pair; every other crop in the global batch is a negative.
* **Model.** The shared encoder with valid-token dropping and class-token readout, followed by
  the projection head `Linear(D, 2048) -> BatchNorm1d -> ReLU -> Linear(2048, 128)`
  (`proj_hidden 2048`, `proj_dim 128`; 1.84 M parameters at ViT-Base). Under DDP the
  BatchNorm is converted to `SyncBatchNorm`. `mask_token` frozen; `cls_token` trained.
* **Objective.** NT-Xent with `temperature 0.1` on the all-gathered embeddings (the loss is
  called on `z.float()` inside the autocast region, so the similarity matmul runs in bf16 and
  the exponentials, logarithms and sums in fp32): the embeddings of all ranks are gathered
  through an autograd function whose
  backward all-reduces the gradient and returns the local shard, so negatives span the global
  batch (`8 x 128 = 1,024` crops: 1 positive and 1,022 negatives per anchor) while gradients
  flow to every rank's encoder. Volume ids are made globally unique (`vol_ids + rank *
  batch_size_per_gpu`); positives are all other crops with the same id (exactly one here).
  Logits are shifted by the detached per-row maximum; the softmax excludes the anchor itself.
* **Optimisation.** Optimizer over encoder and head; `weight_decay 0.1`, `adamw_beta2 0.999`,
  no EMA. `qk_norm` on for all three sizes (`configs/simclr.yaml` has `qk_norm: true`).
* **Transferred.** The encoder (`backbone`), saved as `ckpt_final.pt`.

## Hyper-parameters

Shared by all four (`configs/*.yaml`): `patch_size 16`, `in_chans 1`,
`valid_token_threshold 0.1`, `qk_norm_eps 1.0e-2`, `batch_size_per_gpu 64`, `grad_accum 1`,
`base_lr 1.0e-4`, `adamw_beta1 0.9`, `patch_embed_lr_mult 0.2`, `clip_grad 3.0`,
`warmup_frac 0.1`, `min_lr 1.0e-6`, `seed 42`, identical `aug` block, `log_every 20`,
`feat_every 200`, `ckpt_every_steps 5000`.

| | MAE-96 (`mae`) | JEPA-96 (`jepa`) | DINO (`dino`) | SimCLR-new (`simclr`) |
|---|---|---|---|---|
| paradigm | masked voxel reconstruction | masked latent prediction | self-distillation (class token + patches) | instance contrastive |
| input view | whole volume -> `96^3`, deterministic | whole volume -> `96^3`, deterministic | RRC `2 x 96^3` (scale `[0.5, 1]`) + `8 x 48^3` (`[0.2, 0.5]`) | RRC `2 x 96^3` (scale `[0.5, 1]`) |
| crops / volume | 1 | 1 | 10 | 2 |
| masking | random 75 % of brain tokens (`masking_ratio 0.75`) | two-way blocks, `pad_enc 64` / `pad_pred 192` | iBOT: `p 0.5`, ratio `U[0.2, 0.75]` of valid tokens | none |
| target | per-patch normalised voxels (`norm_pix_loss true`) | EMA encoder latent (LayerNorm, stop-grad) | EMA teacher probabilities (Sinkhorn-Knopp, 3 it.) | the other crop of the same volume |
| loss | MSE on masked patches | L1 (`loss_exp 1.0`) | CE + CE + `0.1` KoLeo | NT-Xent, `temperature 0.1` |
| auxiliary module (per arch) | decoder `64/4/1`, `256/6/4`, `512/8/8` | predictor `192/4/6`, `384/6/12`, `384/12/12` | 4 heads, each `D -> 2048 -> 2048 -> 256 -> 65536` (weight-normalised prototypes) | head `2048 -> 128` |
| method-specific schedule | - | - | teacher temp `0.04 -> 0.07` over 30 %; freeze last layer 1500 steps | - |
| `weight_decay` | **0.05** | 0.04 | 0.04 | **0.10** |
| `adamw_beta2` | **0.95** | 0.999 | 0.999 | 0.999 |
| EMA | none | target encoder, **0.99925 const** | teacher (backbone + heads), **0.992 -> 1.0 cosine** | none |
| `qk_norm` | huge only | huge only | huge only | **all sizes** |
| readout | - (class token unused) | - (class token unused) | class token | class token |
| batch dependence | none | none | Sinkhorn over the global batch | high (1,022 negatives via all-gather; SyncBN) |
| transferred encoder | encoder | **target encoder (EMA)** | **teacher backbone (EMA)** | encoder |

## Outputs

A run directory contains:

| file | content |
|---|---|
| `config.yaml` | the resolved config (the only input of `train.py`) |
| `train.out` | stdout + stderr of all ranks (log lines, diagnostics, final `peak_mem`) |
| `checkpoint_last.pt` | rolling resume state (removed after the final checkpoint is written) |
| final encoder checkpoint | see below; the only file needed downstream |

The final checkpoint holds the plain `shared.vit3d.ViT3D` `state_dict` of the transferred
encoder under a method-specific top-level key and key prefix, plus a `meta` dict:

| method | file | top-level key | key prefix |
|---|---|---|---|
| `mae` | `checkpoints/step{total_steps:06d}.ckpt` (e.g. `checkpoints/step165000.ckpt`) | `state_dict` | `model.encoder.` |
| `jepa` | `3d_jepa_vitb_96-final.pth.tar` | `encoder` | `module.vit.` |
| `dino` | `teacher_backbone_final.pt` | `teacher_backbone` | (none) |
| `simclr` | `ckpt_final.pt` | `backbone_state_dict` | `backbone.` |

The file names and key layouts are kept as they were, for compatibility with the authors'
downstream evaluation code (the JEPA file name is fixed and does not encode the architecture).

The encoder `state_dict` keys are `patch_embed.proj.*`, `cls_token`, `mask_token`,
`blocks.{i}.{norm1,attn.qkv,attn.proj,norm2,mlp.fc1,mlp.fc2}.*`, `norm.*`, plus
`blocks.{i}.attn.{q_norm,k_norm}.*` when `qk_norm` is on. RoPE adds no parameters, so the
encoder must be rebuilt from `meta` rather than from the tensors alone. `meta` fields
(`shared.trainutil.encoder_meta`):

| field | value |
|---|---|
| `method` | `mae` / `jepa` / `dino` / `simclr` |
| `arch`, `patch_size`, `embed_dim`, `depth`, `num_heads`, `pos_base_grid` | encoder geometry (`pos_base_grid = [6, 6, 6]` for `96^3 / 16`) |
| `pos_embed_type`, `init_scheme`, `depth_rescale` | `"rope"`, `"stabilized"`, `true` (fixed in this code base) |
| `qk_norm`, `qk_norm_eps` | QK-normalisation of this run |
| `valid_token_threshold`, `valid_token_threshold_enc` | brain-valid token threshold (0.1; the encoder used the same value) |
| `total_steps` | optimizer steps trained |
| `lr/base_lr`, `lr/eff_batch_volumes`, `lr/lr_scale`, `lr/peak_lr` | learning-rate scaling actually used |

`tools/load_encoder.py` recognises the four layouts, rebuilds `ViT3D` from `meta` (falling
back to tensor shapes when a field is missing) and loads the weights strictly:

```bash
python tools/load_encoder.py runs/<run>/teacher_backbone_final.pt            # prints a summary
python tools/load_encoder.py runs/<run>/ckpt_final.pt --device cuda --forward # + one forward pass on a synthetic 96^3 volume
```

```python
import sys, torch
sys.path.insert(0, "/path/to/code_pretrain_final")
from tools.load_encoder import load_encoder
from shared.data import token_valid

vit, meta = load_encoder("runs/<run>/checkpoints/step165000.ckpt", device="cuda")  # eval mode

x = ...                                    # [B, 1, 96, 96, 96] float32, background exactly 0, values in [0, 1]
tv = token_valid((x > 0).float(), vit.patch_size, meta["valid_token_threshold"])  # reproduce the training-time token set
with torch.no_grad():
    out = vit(x, token_valid=tv)
cls = out["x_norm_clstoken"]               # [B, D]
patches = out["x_norm_patchtokens"]        # [B, 216, D]; exact zeros at background positions
mean_pool = (patches * tv[..., None]).sum(1) / tv.sum(1, keepdim=True).clamp(min=1)
```

Passing `token_valid` is optional (`vit(x)` encodes the full grid), but the encoders were
trained on brain tokens only, so the mask reproduces the training-time input.

## Measured cost

Data-100 x ep-100 (165,000 steps), 8 x NVIDIA B200, 64 volumes per GPU, bf16 autocast,
`grad_checkpointing false`. Memory is the peak *allocated* memory of rank 0
(`torch.cuda.max_memory_allocated`, printed at the end of `train.out`); reserved memory is
roughly 1.1-1.35x higher.

| method | wall-clock (h) tiny / base / huge | peak allocated (GB per GPU) tiny / base / huge | s / step tiny / base / huge |
|---|---|---|---|
| MAE-96 | 7.6 / 7.5 / 9.4 | 2.4 / 5.6 / 23.4 | 0.165 / 0.164 / 0.206 |
| JEPA-96 | 8.8 / 10.3 / 16.3 | 2.5 / 6.4 / 27.4 | 0.192 / 0.224 / 0.355 |
| DINO | 14.2 / 17.9 / 48.9 | 15.6 / 30.2 / 112.9 | 0.309 / 0.386 / 1.068 |
| SimCLR-new | 8.8 / 11.4 / 25.3 | 6.5 / 21.6 / 94.5 | 0.193 / 0.248 / 0.550 |

Other budgets scale linearly with the step count (e.g. 100 x 300 = 495,000 steps is 3x). The
time cost of `qk_norm` grows with the number of attention forwards per step (measured
overhead: MAE about +5 %, JEPA about +17 %, DINO about +33 %). DINO and SimCLR keep the
activations of many crops (10 and 2 per volume) alive, which is why their memory is far above
MAE / JEPA; `grad_checkpointing: true` trades compute for memory without changing the maths.

## Reproducibility notes

* **Fixed by the code and config**: model construction order (and therefore the initial
  weights for a given seed), the data order (sorted index, `DistributedSampler` with
  `set_epoch`), every random stream (per-rank generators listed above), the dtype at every op
  (fp16 loading and resize, bf16 autocast for forward / backward, fp32 master weights and
  optimizer state), `cudnn.deterministic`.
  Two launches with the same code, index, `--nproc`, batch size, config and hardware / library
  versions consume identical random streams. Bitwise identity across different GPU models or
  library versions is not guaranteed (attention and GEMM kernels may differ).
* **What changes with the number of GPUs.** `steps_per_epoch = ((n_vol // nproc) //
  batch_size) // grad_accum`, so a given `--epochs` yields a different `total_steps`; the
  volumes per step (`batch_size x nproc x grad_accum`) set the peak lr (`base_lr x
  volumes_per_step / 256`). Keeping `volumes_per_step = 512` (for example 4 GPUs with
  `--grad-accum 2`) reproduces the schedule and the step count, but the per-rank random
  streams differ, the SimCLR contrastive batch (all-gather is per micro-batch), the DINO
  Sinkhorn normalisation and the SyncBatchNorm statistics change with the world size, so the
  optimisation trajectory is not identical. `--total-steps` fixes the budget in steps
  independently of the data size.
* **Resume** restores weights and optimizer state but saves no random stream: the
  augmentation, crop-centre / scale and iBOT-mask generators, the JEPA mask call counter and
  the global torch RNG (MAE masking) restart from their seeds (see *Resuming*).
* **Downstream inference** should use the same brain-valid token threshold as pre-training
  (`meta["valid_token_threshold"]`, 0.1) on `x > 0` masks, and must rebuild the encoder with
  `qk_norm` / `qk_norm_eps` from `meta` (`tools/load_encoder.py` does this).

## License

The original code of this repository (everything outside `dino/dinov2/`) is released under the
terms of the top-level `LICENSE` file. The third-party components keep their upstream terms,
detailed in `THIRD_PARTY_NOTICES.md`: `dino/dinov2/` is redistributed verbatim under Apache-2.0
(`dino/dinov2/LICENSE`); the RoPE attention, residual depth rescaling, predictor and L1 loss
ported from V-JEPA 2 are under MIT; the NT-Xent loss ported from SimCLR is under Apache-2.0;
the NeuroVFM masking port is subject to the terms of the upstream repository. No code from
the MAE repository (CC BY-NC 4.0) is included -- the MAE objective is an independent 3D
re-implementation -- but anyone combining this repository with the upstream MAE code must
respect its non-commercial terms.

## Acknowledgements

The DINO heads and losses are vendored verbatim from [DINOv2](https://github.com/facebookresearch/dinov2)
(Apache-2.0; `dino/dinov2/LICENSE`). The RoPE attention, residual depth rescaling, predictor
structure and L1 loss follow [V-JEPA 2](https://github.com/facebookresearch/vjepa2) (MIT); the
two-way block masking follows [NeuroVFM](https://github.com/MLNeurosurg/neurovfm); the MAE
objective is a 3D re-implementation of [MAE](https://github.com/facebookresearch/mae); the
NT-Xent loss is a PyTorch port of the [SimCLR](https://github.com/google-research/simclr)
objective. Upstream commits and license terms are listed in `THIRD_PARTY_NOTICES.md`.

## Citation

A citation for the accompanying paper will be added when it is published.
