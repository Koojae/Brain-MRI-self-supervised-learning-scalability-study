#!/usr/bin/env python
"""Load the pre-trained shared encoder from any of the four final checkpoints.

Every training script saves the encoder that is meant for transfer as a plain
:class:`shared.vit3d.ViT3D` ``state_dict`` under a method-specific key and prefix, together
with a ``meta`` dict:

======  ================================  =======================  ================
method  file                              state_dict key           key prefix
======  ================================  =======================  ================
mae     checkpoints/step<N>.ckpt          ``state_dict``           ``model.encoder.``
jepa    3d_jepa_vitb_96-final.pth.tar     ``encoder``              ``module.vit.``
dino    teacher_backbone_final.pt         ``teacher_backbone``     (none)
simclr  ckpt_final.pt                     ``backbone_state_dict``  ``backbone.``
======  ================================  =======================  ================

The JEPA file name is the same for every architecture (``vitb`` is part of the fixed name,
not an architecture tag); ``<N>`` in the MAE name is ``total_steps`` zero-padded to six digits.

RoPE adds no parameters and QK-normalisation only adds LayerNorm keys when it is enabled,
so the encoder is rebuilt from ``meta`` (``arch``, ``patch_size``, ``pos_base_grid``,
``qk_norm``, ``qk_norm_eps``) rather than from the state_dict alone.  When a field is
missing from ``meta`` it is inferred from the tensor shapes (``arch`` from the embedding
width and the block count, ``patch_size`` from the patch-embedding kernel, ``qk_norm`` from
the presence of ``q_norm`` keys), ``qk_norm_eps`` defaults to
:data:`shared.vit3d.QK_NORM_EPS` (1e-2), ``in_chans`` always comes from the patch-embedding
kernel, and ``pos_base_grid`` falls back to the 96^3 / 16 grid.  A checkpoint whose ``meta``
declares a positional encoding other than RoPE is refused, since :class:`ViT3D` has no other.
Weights are loaded strictly.

Example::

    python tools/load_encoder.py runs/<run>/teacher_backbone_final.pt --forward
"""

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from shared.vit3d import QK_NORM_EPS, VIT_FAMILY, ViT3D  # noqa: E402

# (top-level key, prefix to strip) for the four checkpoint layouts.
LAYOUTS = (
    ("state_dict", "model.encoder."),
    ("encoder", "module.vit."),
    ("teacher_backbone", ""),
    ("backbone_state_dict", "backbone."),
)
DEFAULT_BASE_GRID = (6, 6, 6)


def extract_encoder_state(ck):
    """Return the bare ``ViT3D`` state_dict from a loaded checkpoint dict."""
    for key, prefix in LAYOUTS:
        if key in ck:
            sd = ck[key]
            if prefix:
                sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            if not sd:
                raise KeyError(f"checkpoint key {key!r} holds no tensors with prefix {prefix!r}")
            return sd
    raise KeyError(f"no encoder state found; top-level keys: {sorted(ck.keys())}")


def infer_arch(sd):
    """Architecture name from the embedding width and block count of a state_dict."""
    embed_dim = int(sd["patch_embed.proj.weight"].shape[0])
    depth = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    for name, (d, _, _, n) in VIT_FAMILY.items():
        if d == embed_dim and n == depth:
            return name
    raise ValueError(f"no ViT family entry with embed_dim={embed_dim}, depth={depth}")


def load_encoder(path, device="cpu"):
    """Load ``path`` and return ``(vit, meta)`` with ``vit`` in eval mode on ``device``."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    meta = dict(ck.get("meta") or {})
    sd = extract_encoder_state(ck)

    pos_type = meta.get("pos_embed_type", "rope")
    if pos_type != "rope":
        raise ValueError(f"unsupported positional encoding {pos_type!r}; this encoder uses RoPE")

    w = sd["patch_embed.proj.weight"]
    arch = meta.get("arch") or infer_arch(sd)
    in_chans = int(w.shape[1])
    patch_size = int(meta.get("patch_size") or w.shape[-1])
    pos_base_grid = tuple(int(g) for g in meta.get("pos_base_grid") or DEFAULT_BASE_GRID)
    qk_norm = bool(meta.get("qk_norm", any(k.startswith("blocks.0.attn.q_norm") for k in sd)))
    qk_norm_eps = float(meta.get("qk_norm_eps", QK_NORM_EPS))

    vit = ViT3D(arch, in_chans=in_chans, patch_size=patch_size, pos_base_grid=pos_base_grid,
                qk_norm=qk_norm, qk_norm_eps=qk_norm_eps)
    vit.load_state_dict(sd, strict=True)
    vit.to(device).eval()
    return vit, meta


def summary(vit, meta, path):
    n_params = sum(p.numel() for p in vit.parameters())
    lines = [
        f"checkpoint : {path}",
        f"method     : {meta.get('method', '?')}",
        f"arch       : {vit.arch} (embed_dim {vit.embed_dim}, depth {vit.depth}, heads {vit.num_heads})",
        f"patch_size : {vit.patch_size}  pos_base_grid {vit.pos_base_grid}  (RoPE)",
        f"qk_norm    : {vit.qk_norm} (eps {vit.qk_norm_eps:g})",
        f"params     : {n_params:,}",
        f"trained    : total_steps {meta.get('total_steps', '?')}  base_lr {meta.get('lr/base_lr', '?')}"
        f"  peak_lr {meta.get('lr/peak_lr', '?')}  volumes/step {meta.get('lr/eff_batch_volumes', '?')}",
        f"valid_token_threshold : {meta.get('valid_token_threshold', '?')}",
    ]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Load a final encoder checkpoint and print a summary.")
    ap.add_argument("path", help="checkpoint file written by one of the training scripts")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--forward", action="store_true",
                    help="also run one forward pass on a synthetic volume of the encoder's reference size "
                         "(patch_size x pos_base_grid voxels per axis; 96^3 for the study's encoders) "
                         "and print the output shapes")
    args = ap.parse_args(argv)

    vit, meta = load_encoder(args.path, device=args.device)
    print(summary(vit, meta, args.path))
    if args.forward:
        S = vit.patch_size * vit.pos_base_grid[0]
        x = torch.zeros(1, 1, S, S, S, device=args.device)
        lo, hi = S // 4, 3 * S // 4
        x[..., lo:hi, lo:hi, lo:hi] = 0.5                       # brain-like nonzero cube
        with torch.no_grad():
            out = vit(x)
        print(f"forward    : input {tuple(x.shape)} -> cls {tuple(out['x_norm_clstoken'].shape)}"
              f" patches {tuple(out['x_norm_patchtokens'].shape)} grid {out['grid']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
