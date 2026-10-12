"""Masked autoencoder (MAE) on the shared 3D ViT encoder.

3D port of the MAE objective of He et al. (``facebookresearch/mae``, ``models_mae.py``): the
encoder sees a random 25 % subset of the *brain* tokens of a 96^3 whole-volume input, a
light transformer decoder reconstructs the masked brain patches from ``mask_token`` slots,
and the loss is the mean squared error against per-patch normalised voxel intensities
(``norm_pix_loss``).  Only ``encoder`` (a :class:`shared.vit3d.ViT3D`) is transferred
downstream; the decoder, ``mask_token`` and ``to_pixels`` are discarded after pre-training.

Differences from the 2D reference implementation:

* Tokens are split into brain / background with :func:`shared.data.token_valid` on the
  crisp brain mask ``M`` that the input pipeline tracks (a token is brain when at least
  ``valid_token_threshold`` of its voxels are).  Masking is drawn among brain tokens only,
  so the numbers of visible and masked tokens vary per volume: sequences are left-aligned
  and padded to the batch maximum, padded slots are excluded from the attention keys and
  carry zero loss weight.  Background tokens never enter the encoder; they enter the loss
  only in the degenerate case of a volume with a single brain token, where ``n_mask`` is
  clamped to 1 so the decoder is never empty.
* Positions enter through RoPE in the encoder and in the decoder alike.  The decoder has no
  additive positional table; its rotation tables are built for the decoder head dimension
  from the encoder's coordinate convention (same grid, same token indices).
* The encoder's class and mask tokens are unused and frozen (``requires_grad=False``), so
  they are absent from the optimizer and from DDP's gradient buckets.

Shapes: ``x [B, 1, 96, 96, 96]`` -> ``N = 6^3 = 216`` tokens of ``patch_dim = 16^3`` voxels.
"""

import torch
import torch.nn as nn

from shared.vit3d import ViT3D, Block, INIT_STD, rope_arg_from_coords
from shared.data import token_valid


class MAE3D(nn.Module):
    """MAE with a :class:`shared.vit3d.ViT3D` encoder and a small transformer decoder.

    Args:
        arch: encoder size, a key of :data:`shared.vit3d.VIT_FAMILY`.
        in_chans: input channels (1 for MRI).
        img_size: cubic input edge in voxels (96).
        patch_size: cubic patch edge in voxels (16).
        decoder_dim, decoder_depth, decoder_heads: decoder width / number of blocks / heads.
            The decoder MLP hidden size is ``4 * decoder_dim``.
        masking_ratio: fraction of brain tokens that is masked (0.75).
        valid_token_threshold: minimum brain-voxel fraction for a token to count as brain.
        norm_pix_loss: normalise each target patch to zero mean / unit variance.
        enc_kwargs: extra :class:`shared.vit3d.ViT3D` arguments
            (``pos_base_grid``, ``qk_norm``, ``qk_norm_eps``; see
            :func:`shared.trainutil.enc_kwargs`).
    """

    def __init__(self, arch="base", in_chans=1, img_size=96, patch_size=16,
                 decoder_dim=256, decoder_depth=6, decoder_heads=4,
                 masking_ratio=0.75, valid_token_threshold=0.1, norm_pix_loss=True,
                 enc_kwargs=None):
        super().__init__()
        assert 0 < masking_ratio < 1
        assert 0.0 <= valid_token_threshold <= 1.0
        self.patch = patch_size
        self.mask_ratio = masking_ratio
        self.vt = valid_token_threshold
        self.norm_pix = norm_pix_loss
        self.patch_dim = in_chans * patch_size ** 3

        # ── encoder: the shared ViT3D (transferred downstream) ──
        self.encoder = ViT3D(arch=arch, in_chans=in_chans, patch_size=patch_size,
                             **(enc_kwargs or {}))
        self.embed_dim = self.encoder.embed_dim
        # MAE uses neither the class token nor the encoder's mask token: freeze both.
        self.encoder.cls_token.requires_grad_(False)
        self.encoder.mask_token.requires_grad_(False)

        # ── decoder (discarded after pre-training); reuses the shared transformer block ──
        # Module construction order matters for reproducibility: every nn.Linear draws its
        # default initialisation from the global RNG before the stabilised init overwrites it
        # (``Block.__init__`` for the decoder blocks, ``_init_weights`` for ``enc_to_dec`` /
        # ``to_pixels``).
        self.dec_head_dim = decoder_dim // decoder_heads
        self.enc_to_dec = (nn.Linear(self.embed_dim, decoder_dim)
                           if self.embed_dim != decoder_dim else nn.Identity())
        self.decoder_blocks = nn.ModuleList(
            [Block(decoder_dim, decoder_dim * 4, decoder_heads) for _ in range(decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        # Shape [1, 1, D] (ndim 3), as in the reference implementation: the shared
        # parameter grouping applies weight decay to tensors with ndim > 1, so the mask
        # token is decayed like a weight rather than treated as a bias.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.to_pixels = nn.Linear(decoder_dim, self.patch_dim)

        g = img_size // patch_size
        self.grid = (g, g, g)
        self._init_weights()

    def _init_weights(self):
        """Decoder-side init: ``trunc_normal(std=0.02)`` weights, zero biases, unit LayerNorm.

        The decoder blocks are already initialised by :class:`shared.vit3d.Block` (same
        scheme as the encoder, including the fused-qkv rule) and are not touched here.
        """
        nn.init.trunc_normal_(self.mask_token, std=INIT_STD)
        for mod in (self.enc_to_dec, self.to_pixels):
            if isinstance(mod, nn.Linear):
                nn.init.trunc_normal_(mod.weight, std=INIT_STD)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
        nn.init.zeros_(self.decoder_norm.bias)
        nn.init.ones_(self.decoder_norm.weight)

    def _dec_rope(self, idx, device):
        """RoPE tables for the decoder sequence whose token indices are ``idx [B, K]``.

        Coordinates come from the encoder (same grid and base-grid convention); only the
        sin / cos tables are rebuilt for the decoder head dimension.  No prefix token.
        """
        coords = self.encoder.rope_coords(self.grid, idx, device)
        return rope_arg_from_coords(coords, self.dec_head_dim, 0)

    def patchify(self, x):
        """``x [B, C, D, H, W]`` -> ``[B, N, C * p^3]`` in the token order of the patch embedding (d, h, w)."""
        p = self.patch
        B, C, D, H, W = x.shape
        gd, gh, gw = D // p, H // p, W // p
        x = x.reshape(B, C, gd, p, gh, p, gw, p)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B, gd * gh * gw, C * p * p * p)
        return x

    def forward(self, x, mask=None):
        """One MAE step.

        Args:
            x: ``[B, 1, S, S, S]`` volume batch (fp32; run under autocast by the trainer).
            mask: ``[B, 1, S, S, S]`` crisp brain mask (0/1).  When ``None`` the brain mask
                is taken as ``x > 0``.

        Returns ``(loss, pred, aux)`` where ``pred [B, max_mask, patch_dim]`` are the decoder
        outputs for the masked slots (padded slots included) and ``aux`` holds
        ``valid_frac`` (brain-token fraction of the batch) and ``keep_frac`` (mean visible
        fraction of the full grid).
        """
        device = x.device
        patches = self.patchify(x)                                  # [B, N, patch_dim] targets
        B, N, _ = patches.shape
        vsrc = mask if mask is not None else (x > 0).to(x.dtype)
        valid = token_valid(vsrc, self.patch, self.vt)              # [B, N] bool, brain tokens
        tokens, _ = self.encoder.tokenize(x)                        # [B, N, embed_dim]
        br = torch.arange(B, device=device)[:, None]

        # ── masking among brain tokens: random order for brain tokens, background last ──
        key = torch.rand(B, N, device=device).masked_fill(~valid, 2.0)
        sidx = key.argsort(dim=-1)                                  # brain (shuffled) | background
        n_valid = valid.sum(dim=-1)
        n_keep = (n_valid.float() * (1.0 - self.mask_ratio)).round().clamp(min=1).long()
        n_mask = (n_valid - n_keep).clamp(min=1)
        max_keep, max_mask = int(n_keep.max()), int(n_mask.max())

        # Left-aligned index tables padded to the batch maximum.  Padded slots repeat real
        # indices (``% n``) so the gather is always in range; ``vis_real`` / ``mask_weight``
        # mark the real slots.
        ck = torch.arange(max_keep, device=device)[None, :]
        visible_idx = sidx.gather(1, ck % n_keep[:, None])          # [B, max_keep]
        vis_real = ck < n_keep[:, None]
        cm = torch.arange(max_mask, device=device)[None, :]
        masked_idx = sidx.gather(1, n_keep[:, None] + (cm % n_mask[:, None]))  # [B, max_mask]
        mask_weight = (cm < n_mask[:, None]).to(patches.dtype)

        # ── encoder on the visible brain tokens; padded slots are dropped from the keys ──
        enc = tokens[br, visible_idx]                               # [B, max_keep, embed_dim]
        enc_mask = None if bool(vis_real.all()) else vis_real
        enc = self.encoder.run_blocks(
            enc, attn_mask=enc_mask, norm=True,
            rope=self.encoder.rope_arg(self.grid, visible_idx, 0, device))

        # ── decoder over [masked mask_token | visible]; positions through RoPE ──
        drope = self._dec_rope(torch.cat([masked_idx, visible_idx], dim=1), device)
        dec_vis = self.enc_to_dec(enc)
        mask_tok = self.mask_token.expand(B, max_mask, -1)
        dec_in = torch.cat([mask_tok, dec_vis], dim=1)              # [B, max_mask + max_keep, dec]
        dec_keep = torch.cat([mask_weight.bool(), vis_real], dim=1)  # True = real slot
        dec_attn = None if bool(dec_keep.all()) else dec_keep
        for blk in self.decoder_blocks:
            dec_in = blk(dec_in, attn_mask=dec_attn, rope=drope)
        decoded = self.decoder_norm(dec_in)
        pred = self.to_pixels(decoded[:, :max_mask])               # [B, max_mask, patch_dim]

        # ── loss: MSE on (normalised) masked brain patches; padded slots weigh zero ──
        target = patches[br, masked_idx]                           # [B, max_mask, patch_dim]
        if self.norm_pix:
            mu = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mu) / (var + 1e-6).sqrt()
        per = ((pred - target) ** 2).mean(dim=-1)                  # [B, max_mask]
        loss = (per * mask_weight).sum() / mask_weight.sum().clamp(min=1.0)
        aux = {"valid_frac": valid.float().mean().detach(),
               "keep_frac": (n_keep.float() / N).mean().detach()}
        return loss, pred, aux

    @torch.no_grad()
    def encode_features(self, x, mask=None):
        """Mean of the brain patch tokens of the full encoder forward, ``[B, embed_dim]``.

        Used by the trainer's collapse monitor: the encoder runs on the brain-valid tokens
        only (same valid-token definition as the objective) and the readout is the mean of
        those patch tokens.
        """
        vsrc = mask if mask is not None else (x > 0).to(x.dtype)
        tv = token_valid(vsrc, self.patch, self.vt)
        out = self.encoder(x, token_valid=tv)
        patch = out["x_norm_patchtokens"]                          # [B, N, embed_dim]; 0 at background
        w = tv.unsqueeze(-1).to(patch.dtype)
        return (patch * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
