"""JEPA for 3D brain MRI: context encoder, EMA target encoder, predictor and L1 loss.

The predictor / EMA-target / L1-loss structure follows V-JEPA 2 (facebookresearch/vjepa2);
the masking follows NeuroVFM (``jepa/masks.py``).  The encoder is the shared
:class:`shared.vit3d.ViT3D`.

One training step on a batch ``x [B, 1, 96, 96, 96]`` with brain-valid tokens
``tv [B, 216]`` and a ``(context, target)`` index pair per sample:

* **target encoder** (EMA copy of the context encoder, no gradient): encodes all 216
  tokens with the background tokens removed from the attention keys (``attn_mask = tv``),
  applies a parameter-free LayerNorm to the output and gathers the target positions.
* **context encoder** (trained): encodes only the gathered context tokens; padded slots
  are removed from the attention keys.  Positions enter through RoPE, so the gathered
  token indices are passed as coordinates.
* **predictor** (trained, discarded after pre-training): a narrow transformer over
  ``[embed(context) | mask token x n_target]`` with RoPE coordinates
  ``cat([enc_idx, pred_idx])`` and the key mask ``[enc_w | pred_w]``; the target part of the
  output is projected back to the encoder width.
* **loss**: L1 between prediction and target (``loss_exp = 1``), averaged over real target
  tokens and channels (padded slots have weight 0).

Only the target encoder is transferred downstream.  Its ``cls_token`` / ``mask_token`` are
never used by the objective and are frozen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.vit3d import ViT3D, Block, rope_coords_from_idx, rope_arg_from_coords
from shared.data import token_valid
from jepa.masks import apply_masks


class JEPAEncoder(nn.Module):
    """The shared ViT3D used as context or target encoder.

    ``forward(x, masks=None, attn_mask=None, masks_attn=None)``:

    * ``masks is None`` (target path): all tokens are encoded; ``attn_mask [B, N]`` (bool,
      True = attend) removes background tokens from the attention keys.  Returns ``[B, N, D]``.
    * otherwise (context path; ``masks_attn`` must be given, one entry per mask): for
      each index mask ``m [B, K]`` in ``masks`` only the gathered tokens are encoded,
      with the matching ``masks_attn`` entry (1 = real, 0 = padded) as key mask (dropped
      when every slot is real so the flash kernel is used).  Returns a list of
      ``[B, K, D]``.
    """

    def __init__(self, arch, in_chans=1, patch_size=16, img_size=96, enc_kwargs=None):
        super().__init__()
        self.vit = ViT3D(arch=arch, in_chans=in_chans, patch_size=patch_size, **(enc_kwargs or {}))
        self.vit.cls_token.requires_grad_(False)     # not used by the JEPA objective
        self.vit.mask_token.requires_grad_(False)
        self.embed_dim = self.vit.embed_dim
        g = img_size // patch_size
        self.num_patches = g ** 3

    def forward(self, x, masks=None, attn_mask=None, masks_attn=None):
        tokens, grid = self.vit.tokenize(x)           # [B, N, D], no additive position
        if masks is None:                             # target: all tokens, background key mask
            return self.vit.run_blocks(tokens, attn_mask=attn_mask, norm=True,
                                       rope=self.vit.rope_arg(grid, None, 0, x.device))
        # Context only.  The padding weights become the attention key mask so that the
        # repeated (padded) indices do not act as extra keys.
        out = []
        for m, am in zip(masks, masks_attn):
            am = am.bool()
            if bool(am.all()):
                am = None
            out.append(self.vit.run_blocks(
                apply_masks(tokens, [m]), attn_mask=am, norm=True,
                rope=self.vit.rope_arg(grid, m, 0, x.device)))   # token indices as coordinates
        return out


class JEPAPredictor(nn.Module):
    """Context tokens + target positions -> predicted target latents.

    Args:
        num_patches: tokens of the full grid (``G^3``); the predictor grid equals the
            encoder grid.
        embed_dim: encoder width (input and output width of the predictor).
        predictor_embed_dim, depth, num_heads: predictor transformer size.
        num_mask_tokens: learnable target queries (one per mask; the two-way masking uses 1).

    Blocks are the shared :class:`shared.vit3d.Block` with the same stabilized
    initialisation as the encoder; ``embed`` / ``proj`` use ``trunc_normal(std=0.02)`` with
    zero bias.  Positions enter through RoPE only (no additive positional table).
    """

    def __init__(self, num_patches, embed_dim, predictor_embed_dim=384, depth=6,
                 num_heads=12, num_mask_tokens=1):
        super().__init__()
        self.head_dim = predictor_embed_dim // num_heads
        self.embed = nn.Linear(embed_dim, predictor_embed_dim)
        g = round(num_patches ** (1.0 / 3.0))
        assert g ** 3 == num_patches, f"non-cubic grid: {num_patches}"
        self.grid = (g, g, g)
        self.mask_tokens = nn.Parameter(torch.zeros(num_mask_tokens, predictor_embed_dim))
        self.blocks = nn.ModuleList(
            [Block(predictor_embed_dim, predictor_embed_dim * 4, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(predictor_embed_dim)
        self.proj = nn.Linear(predictor_embed_dim, embed_dim)
        nn.init.trunc_normal_(self.mask_tokens, std=0.02)
        for mod in (self.embed, self.proj):
            nn.init.trunc_normal_(mod.weight, std=0.02)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)

    def _rope(self, idx, device):
        """RoPE tables for the ``[context | target]`` sequence; ``idx [B, Kc + Kt]`` token indices."""
        c = rope_coords_from_idx(self.grid, idx, self.grid, device)
        return rope_arg_from_coords(c, self.head_dim, 0)

    def forward(self, ctxt_list, masks_enc, masks_pred, w_enc, w_pred):
        """``ctxt_list``: context latents ``[B, Kc, D]`` per mask; ``masks_enc`` / ``masks_pred``:
        index tensors ``[B, Kc]`` / ``[B, Kt]``; ``w_enc`` / ``w_pred``: padding weights
        (1 = real, 0 = padded) used as the attention key mask.  Returns a list of
        ``[B, Kt, D]`` predictions.
        """
        preds = []
        for i, (z, me, mp) in enumerate(zip(ctxt_list, masks_enc, masks_pred)):
            B = z.size(0)
            c = self.embed(z)                                           # [B, Kc, P]
            mt = self.mask_tokens[i].view(1, 1, -1).expand(B, mp.size(1), -1)
            x = torch.cat([c, mt], dim=1)                               # [B, Kc + Kt, P]
            rope = self._rope(torch.cat([me, mp], dim=1), x.device)
            am = torch.cat([w_enc[i].bool(), w_pred[i].bool()], dim=1)  # [B, Kc + Kt]
            if bool(am.all()):
                am = None                                               # no padding: flash path
            for blk in self.blocks:
                x = blk(x, attn_mask=am, rope=rope)
            x = self.norm(x)
            preds.append(self.proj(x[:, c.size(1):]))                  # target part only
        return preds


class JEPA3D(nn.Module):
    """Context encoder + EMA target encoder + predictor with the V-JEPA 2 L1 loss.

    The target encoder is initialised as a copy of the context encoder and frozen; the
    training script updates it by EMA after every optimizer step.
    """

    def __init__(self, arch="base", patch_size=16, in_chans=1, img_size=96,
                 pred_embed_dim=384, pred_depth=6, pred_num_heads=12, num_mask_tokens=1,
                 valid_token_threshold=0.1, loss_exp=1.0, enc_kwargs=None):
        super().__init__()
        self.patch = patch_size
        self.vt = valid_token_threshold
        self.loss_exp = loss_exp
        enc_kwargs = dict(enc_kwargs or {})
        self.encoder = JEPAEncoder(arch, in_chans, patch_size, img_size, enc_kwargs)         # context
        self.target_encoder = JEPAEncoder(arch, in_chans, patch_size, img_size, enc_kwargs)  # EMA
        self.predictor = JEPAPredictor(self.encoder.num_patches, self.encoder.embed_dim,
                                       pred_embed_dim, pred_depth, pred_num_heads, num_mask_tokens)
        self._sync_target()

    @torch.no_grad()
    def _sync_target(self):
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    def forward(self, x, tv, masks_enc, masks_pred, masks_pred_w, masks_enc_w):
        """``x [B, 1, S, S, S]``, ``tv [B, N]`` bool, ``masks_enc`` / ``masks_pred``: lists of
        ``[B, K]`` index tensors, ``masks_pred_w`` / ``masks_enc_w``: lists of ``[B, K]``
        weights (1 = real, 0 = padded).  The weights are the loss weights (target side) and
        the attention key masks of the context encoder and the predictor (both sides).
        Returns ``{"loss": loss}`` (scalar).
        """
        # Target (EMA, no gradient): full encode with background key mask -> LayerNorm -> gather.
        with torch.no_grad():
            h = self.target_encoder(x, masks=None, attn_mask=tv)        # [B, N, D]
            h = F.layer_norm(h, (h.size(-1),))
            h = apply_masks(h, masks_pred)                              # [B, Kt, D]
        # Context encode + predict (padded slots excluded from the attention keys).
        z_list = self.encoder(x, masks=masks_enc, masks_attn=masks_enc_w)   # list of [B, Kc, D]
        preds = self.predictor(z_list, masks_enc, masks_pred, masks_enc_w, masks_pred_w)
        z = preds[0]                                                    # [B, Kt, D]
        # L1 loss over real target tokens (V-JEPA 2 ``loss_exp`` form).
        w = masks_pred_w[0].unsqueeze(-1)                               # [B, Kt, 1]
        diff = torch.abs(z - h) ** self.loss_exp
        loss = (diff * w).sum() / (w.sum() * z.size(-1)).clamp(min=1.0) / self.loss_exp
        return {"loss": loss}

    @torch.no_grad()
    def encode_features(self, x, M):
        """Mean of the brain-valid target-encoder tokens, ``[B, D]`` (collapse monitor).

        ``M`` is the crisp brain mask of ``x``; the encoder sees the brain tokens
        (background removed from the keys) and the same tokens are averaged.
        """
        tv = token_valid(M, self.patch, self.vt)
        h = self.target_encoder(x, masks=None, attn_mask=tv)            # [B, N, D]
        w = tv.unsqueeze(-1).to(h.dtype)
        return (h * w).sum(1) / w.sum(1).clamp_min(1.0)
