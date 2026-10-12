# Third-party notices

This repository builds on the following open-source projects. For each one we list the
upstream repository and the exact commit we worked from, its license, and which parts of this
code base derive from it. Only DINOv2 code is redistributed verbatim; everything else was
ported or re-implemented for the 3D brain-MRI setting and the shared encoder.

| project | upstream commit | license | used how |
|---|---|---|---|
| DINOv2 | `facebookresearch/dinov2@7764ea0f912e53c92e82eb78a2a1631e92725fc8` | Apache-2.0 | head and loss modules vendored verbatim in `dino/dinov2/` |
| V-JEPA 2 | `facebookresearch/vjepa2@204698b45b3712590f06245fbfba32d3be539812` | MIT | RoPE attention, residual depth rescaling, predictor, `apply_masks`, L1 loss ported |
| NeuroVFM | `MLNeurosurg/neurovfm` | see upstream repository | two-way block masking ported |
| MAE | `facebookresearch/mae@efb2a8062c206524e35e47d04501ed4f544c0ae8` | CC BY-NC 4.0 (code) | objective re-implemented in 3D; no upstream code included |
| SimCLR | `google-research/simclr@383d4143fd8cf7879ae10f1046a9baeb753ff438` | Apache-2.0 | NT-Xent loss ported from TensorFlow to PyTorch |

---

## DINOv2

* Upstream: https://github.com/facebookresearch/dinov2, commit
  `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.
* Copyright (c) Meta Platforms, Inc. and affiliates.
* License: Apache License, Version 2.0. The full license text is included as
  `dino/dinov2/LICENSE`.

Files redistributed **verbatim**; the original copyright headers are preserved:

| this repository | upstream path |
|---|---|
| `dino/dinov2/__init__.py` | `dinov2/loss/__init__.py` (the three loss imports; the head is imported directly from `dino_head.py`) |
| `dino/dinov2/dino_head.py` | `dinov2/layers/dino_head.py` |
| `dino/dinov2/dino_clstoken_loss.py` | `dinov2/loss/dino_clstoken_loss.py` |
| `dino/dinov2/ibot_patch_loss.py` | `dinov2/loss/ibot_patch_loss.py` |
| `dino/dinov2/koleo_loss.py` | `dinov2/loss/koleo_loss.py` |

In addition, `dino/model.py` and `dino/train.py` re-implement the training logic of DINOv2's
`SSLMetaArch` (multi-crop DINO / iBOT / KoLeo objective, Sinkhorn-Knopp teacher centering,
teacher-temperature warm-up, freezing of the prototype layer, EMA teacher with cosine momentum,
`patch_embed` learning-rate multiplier) on top of the shared 3D encoder; these files are
derived works and are not verbatim copies.

## V-JEPA 2

* Upstream: https://github.com/facebookresearch/vjepa2, commit
  `204698b45b3712590f06245fbfba32d3be539812`.
* Copyright (c) Meta Platforms, Inc. and affiliates.
* License: MIT (reproduced below).

Ported components (re-written for the isotropic 3D token grid and the shared
`shared.vit3d.ViT3D`; not verbatim copies):

| this repository | derived from |
|---|---|
| `shared/vit3d.py`: `rope_dim_per_axis`, `rope_sincos`, `rope_fuse_axes`, `rope_rotate`, `rope_apply_qk`, RoPE handling in `Attention` | `src/models/utils/modules.py` (`RoPEAttention`, `rotate_queries_or_keys`; per-axis head-dimension split) |
| `shared/vit3d.py`: `ViT3D._rescale_blocks` | `src/models/vision_transformer.py` (`_rescale_blocks`) |
| `jepa/model.py`: `JEPAPredictor` (embed, mask tokens, blocks, norm, projection; RoPE positions) | `src/models/predictor.py` (`VisionTransformerPredictor`) |
| `jepa/masks.py`: `apply_masks` | `src/masks/utils.py` (`apply_masks`) |
| `jepa/model.py`: `JEPA3D.forward` loss (`abs(pred - target) ** loss_exp`, normalised, `loss_exp = 1`) and EMA target encoder | V-JEPA 2 training loss / target-encoder update |

```
MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## NeuroVFM

* Upstream: https://github.com/MLNeurosurg/neurovfm (no specific commit was pinned; the
  masking logic was ported in 2026).
* License: please consult the upstream repository for its license terms; no NeuroVFM source
  file is redistributed here.

`jepa/masks.py` (`MultiBlockMaskGenerator`, `_MaskGenerator`) is a port of NeuroVFM's
`_MultiBlockCollator` (`_sample_block_size`, `_sample_block_mask`, `_generate_masks`; the
two-way `switch_enc_pred` scheme, depth-slab dropping to `enc_mask_scale`, patch dropout and
the acceptance constants `MIN_ENC_FRAC`, `MAX_ENC_FRAC`, `MAX_N_ATTEMPTS`) to the isotropic
`6^3` token grid of this repository, confined to brain-valid tokens and extended with
fixed-length padding and weight masks.

## MAE (Masked Autoencoders)

* Upstream: https://github.com/facebookresearch/mae, commit
  `efb2a8062c206524e35e47d04501ed4f544c0ae8`.
* Copyright (c) Meta Platforms, Inc. and affiliates.
* License: the upstream code is released under the Creative Commons
  Attribution-NonCommercial 4.0 International license (CC BY-NC 4.0).

**No code from the MAE repository is included in this release.** `mae/model.py` is an
independent 3D re-implementation of the MAE objective described by He et al. and structured
after `models_mae.py` (random masking of 75 % of the tokens, encoder on the visible tokens
only, lightweight transformer decoder with learnable mask tokens, per-patch normalised pixel
targets, mean-squared error on the masked patches), with the masking restricted to
brain-valid tokens, `Conv3d` patch embedding and RoPE positions. Users who combine this
repository with the upstream MAE code must respect the non-commercial terms of that code.

## SimCLR

* Upstream: https://github.com/google-research/simclr, commit
  `383d4143fd8cf7879ae10f1046a9baeb753ff438`.
* Copyright 2020 The SimCLR Authors.
* License: Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).

`simclr/loss.py` (`NTXentLoss`) is a PyTorch port of the NT-Xent contrastive loss
`add_contrastive_loss` in `objective.py` (TensorFlow 1), generalised to an arbitrary number of
positives per anchor (positives = other crops of the same volume) and to a gradient-preserving
all-gather across DDP ranks (`GatherLayer`). The two-layer projection head
`Linear -> BatchNorm -> ReLU -> Linear` in `simclr/model.py` follows the reference
architecture. No TensorFlow code is redistributed.

## Other references (no code)

* The fused-`qkv` initialisation rule of `shared.vit3d.init_linear_` follows MoCo v3
  (Chen et al., 2021); the QK-normalisation option follows ViT-22B (Dehghani et al., 2023).
  Both were implemented independently from the papers.
* The Apache-2.0 and MIT terms above apply to the respective upstream components only, not
  to the original code of this repository.
