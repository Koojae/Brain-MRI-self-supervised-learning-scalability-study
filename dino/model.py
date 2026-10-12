"""DINO / iBOT self-distillation on the shared 3D ViT (DINOv2 objective).

The objective is the one of facebookresearch/dinov2: DINO class-token distillation across
views, iBOT masked-patch distillation on the global crops, and the KoLeo regularizer, with an
EMA teacher and Sinkhorn-Knopp centering of the teacher outputs.  The loss and head modules
are vendored from DINOv2 (``dino/dinov2/``); only the backbone is replaced by the shared
:class:`shared.vit3d.ViT3D` so that all four methods in this repository train the same
encoder.  Only the teacher backbone is transferred downstream.

Per step, with ``n_global = 2`` global crops (96^3) and ``n_local = 8`` local crops (48^3):

* **Valid tokens.**  ``token_valid(M, patch_size, valid_token_threshold)`` on the crisp brain
  mask of every crop marks brain patches.  The backbone drops the other tokens, and the iBOT
  mask is sampled from the valid tokens only.
* **iBOT mask.**  Each global-crop sample is masked with probability
  ``ibot_mask_sample_probability``; the mask ratio ``~U[ibot_mask_ratio_min_max]`` is applied
  to that sample's *valid* token count and the masked positions are a random subset of its
  valid tokens (replaced by ``mask_token`` in the student input).
* **Teacher** (no grad): the global crops are run through the teacher backbone and the class
  tokens through the teacher DINO head; the masked patch tokens go through the teacher iBOT
  head.  Both outputs are centred with Sinkhorn-Knopp over the concatenated global crops of all
  ranks (all-reduced inside the vendored loss); the iBOT run normalises by the all-reduced total
  number of masked patches.
* **Student**: global crops (with the iBOT mask) and local crops through the student
  backbone; class tokens through the DINO head, masked patch tokens through the iBOT head.
* **Loss** = ``dino + ibot + koleo_weight * koleo`` where ``dino`` averages the cross-entropy
  over every (student view, other teacher global) pair (``(n_global - 1 + n_local) * n_global
  = 18`` terms), ``ibot`` follows DINOv2 ``forward_masked`` (per-sample ``1 / n_masked``
  weights, divided by the number of global-crop samples), and ``koleo`` is summed over the
  two global crops of the student class-token features.

Crops of equal resolution are concatenated and run through the backbone / heads in one call
(three backbone calls per step: teacher globals, student globals, student locals); the
outputs are split back per crop before the per-view loss terms.  Padding slots introduced by
the batch-wide valid-token packing are excluded from the attention keys, so this is
mathematically the per-crop computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.data import token_valid
from shared.vit3d import ViT3D
from dino.dinov2.dino_head import DINOHead
from dino.dinov2.dino_clstoken_loss import DINOLoss
from dino.dinov2.ibot_patch_loss import iBOTPatchLoss
from dino.dinov2.koleo_loss import KoLeoLoss


class ViT3DBackbone(nn.Module):
    """The shared encoder as a DINO backbone.  ``forward(x, ibot_mask, token_valid)`` -> ViT3D dict."""

    def __init__(self, arch, patch_size=16, in_chans=1, enc_kwargs=None):
        super().__init__()
        self.vit = ViT3D(arch=arch, in_chans=in_chans, patch_size=patch_size, **(enc_kwargs or {}))
        self.embed_dim = self.vit.embed_dim
        self.patch_size = patch_size

    def forward(self, x, ibot_mask=None, token_valid=None):
        return self.vit(x, mask=ibot_mask, token_valid=token_valid)


def _chunk_out(out, n):
    """Split a batched backbone output dict into ``n`` per-crop dicts.

    Each per-crop dict mirrors the :class:`shared.vit3d.ViT3D` output layout (``grid`` and, when
    present, ``token_valid`` are carried along); the loss only reads ``x_norm_clstoken`` and
    ``x_norm_patchtokens``.  ``chunk`` returns views, so nothing is copied.
    """
    cls = out["x_norm_clstoken"].chunk(n, 0)
    patch = out["x_norm_patchtokens"].chunk(n, 0)
    tv = out.get("token_valid")
    tvc = tv.chunk(n, 0) if tv is not None else [None] * n
    res = []
    for i in range(n):
        d = {"x_norm_clstoken": cls[i], "x_norm_patchtokens": patch[i], "grid": out["grid"]}
        if tvc[i] is not None:
            d["token_valid"] = tvc[i]
        res.append(d)
    return res


class DINO3DSSL(nn.Module):
    """Student / teacher pair with DINO and iBOT heads and the three loss terms.

    Args:
        arch, patch_size, in_chans, enc_kwargs: backbone specification (see
            :class:`shared.vit3d.ViT3D`; ``enc_kwargs`` carries ``pos_base_grid``, ``qk_norm``,
            ``qk_norm_eps``).
        dino_out_dim, ibot_out_dim: number of prototypes of the DINO and iBOT heads.
        head_nlayers, head_hidden_dim, head_bottleneck_dim: :class:`DINOHead` MLP shape.
        student_temp: student softmax temperature (teacher temperature is a forward argument).
        sinkhorn_iterations: Sinkhorn-Knopp iterations for teacher centering.
        valid_token_threshold: minimum brain-voxel fraction (inclusive, ``>=``) of a patch for
            its token to count as valid (see :func:`shared.data.token_valid`).
        ibot_mask_ratio_min_max, ibot_mask_sample_probability: iBOT mask sampling.
        dino_weight, ibot_weight, koleo_weight: loss weights.
    """

    def __init__(self, arch="base", patch_size=16, in_chans=1,
                 dino_out_dim=65536, ibot_out_dim=65536,
                 head_nlayers=3, head_hidden_dim=2048, head_bottleneck_dim=256,
                 student_temp=0.1, sinkhorn_iterations=3, valid_token_threshold=0.1,
                 ibot_mask_ratio_min_max=(0.2, 0.75), ibot_mask_sample_probability=0.5,
                 dino_weight=1.0, ibot_weight=1.0, koleo_weight=0.1, enc_kwargs=None):
        super().__init__()
        self.valid_thr = valid_token_threshold
        self.sinkhorn_iterations = sinkhorn_iterations
        self.mask_ratio_min, self.mask_ratio_max = ibot_mask_ratio_min_max
        self.mask_sample_prob = ibot_mask_sample_probability
        self.dino_weight, self.ibot_weight, self.koleo_weight = dino_weight, ibot_weight, koleo_weight

        enc_kwargs = dict(enc_kwargs or {})
        # Construction order fixes the random initialisation stream: backbones, then heads.
        self.student_backbone = ViT3DBackbone(arch, patch_size, in_chans, enc_kwargs)
        self.teacher_backbone = ViT3DBackbone(arch, patch_size, in_chans, enc_kwargs)
        embed = self.student_backbone.embed_dim
        hkw = dict(nlayers=head_nlayers, hidden_dim=head_hidden_dim, bottleneck_dim=head_bottleneck_dim)
        self.student_dino_head = DINOHead(embed, dino_out_dim, **hkw)
        self.teacher_dino_head = DINOHead(embed, dino_out_dim, **hkw)
        self.student_ibot_head = DINOHead(embed, ibot_out_dim, **hkw)   # separate iBOT head
        self.teacher_ibot_head = DINOHead(embed, ibot_out_dim, **hkw)

        self.dino_loss = DINOLoss(dino_out_dim, student_temp)
        self.ibot_loss = iBOTPatchLoss(ibot_out_dim, student_temp)
        self.koleo_loss = KoLeoLoss()

        self._sync_teacher_from_student()

    # ── teacher = EMA of student (the per-step update lives in shared.optim.EmaUpdater) ──
    def _student_modules(self):
        return [self.student_backbone, self.student_dino_head, self.student_ibot_head]

    def _teacher_modules(self):
        return [self.teacher_backbone, self.teacher_dino_head, self.teacher_ibot_head]

    @torch.no_grad()
    def _sync_teacher_from_student(self):
        """Copy the student weights into the teacher and freeze the teacher."""
        for s, t in zip(self._student_modules(), self._teacher_modules()):
            t.load_state_dict(s.state_dict())
            for p in t.parameters():
                p.requires_grad_(False)

    # ── pooling: class token ──
    @staticmethod
    def _pool(out):
        return out["x_norm_clstoken"]

    # ── iBOT mask: random subset of each sample's valid (brain) tokens ──
    def _make_ibot_mask(self, tv, gen):
        """``tv [B, N]`` bool -> iBOT mask ``[B, N]`` bool, drawn from ``gen`` sample by sample."""
        B, N = tv.shape
        device = tv.device
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for b in range(B):
            if torch.rand(1, generator=gen, device=device).item() >= self.mask_sample_prob:
                continue
            valid = tv[b].nonzero(as_tuple=False).flatten()
            if valid.numel() == 0:
                continue
            ratio = self.mask_ratio_min + (self.mask_ratio_max - self.mask_ratio_min) * \
                torch.rand(1, generator=gen, device=device).item()
            # The ratio applies to the number of valid tokens, not to the full grid.
            n_mask = max(1, min(int(round(valid.numel() * ratio)), int(valid.numel())))
            perm = torch.randperm(valid.numel(), generator=gen, device=device)[:n_mask]
            mask[b, valid[perm]] = True
        return mask

    def _tv(self, M, ps):
        """Brain-valid token mask of a crop batch from its crisp mask ``M``."""
        return token_valid(M, ps, self.valid_thr)

    # ── batched backbone / head calls ──
    def _run_backbone(self, backbone, crops, tvs, masks=None):
        """Concatenate same-resolution crops, run the backbone once, split per crop."""
        n = len(crops)
        out = backbone(torch.cat(crops, 0), token_valid=torch.cat(tvs, 0),
                       ibot_mask=(None if masks is None else torch.cat(masks, 0)))
        return _chunk_out(out, n)

    @staticmethod
    def _run_head(head, feats):
        """Run a list of pooled features through ``head`` in one call (the head is row-wise)."""
        return list(head(torch.cat(feats, 0)).chunk(len(feats), 0))

    # ── one training step ──
    def forward(self, batch, teacher_temp, gen):
        """``batch`` from :func:`dino.views.make_dino_views`; ``gen`` draws the iBOT masks.

        Returns ``{"loss", "dino", "ibot", "koleo"}`` (the last three detached).
        """
        globals_, gmasks = batch["global_crops"], batch["global_masks"]
        locals_, lmasks = batch["local_crops"], batch["local_masks"]
        ps = self.student_backbone.patch_size
        n_global = len(globals_)

        tv_g = [self._tv(m, ps) for m in gmasks]
        tv_l = [self._tv(m, ps) for m in lmasks]
        masks = [self._make_ibot_mask(tv_g[gi], gen) for gi in range(n_global)]

        # ── teacher (no grad): global crops without the iBOT mask ──
        with torch.no_grad():
            t_outs = self._run_backbone(self.teacher_backbone, globals_, tv_g)
            t_cls_h = self._run_head(self.teacher_dino_head, [self._pool(o) for o in t_outs])
            # Masked patch features have a different row count per crop: concatenate, run the
            # head once, split by count.  The head accepts zero rows, so the concatenation and
            # the single Sinkhorn-Knopp call below run unconditionally and every rank issues the
            # same collectives even when a crop has no masked patch.
            t_masked = [t_outs[gi]["x_norm_patchtokens"][masks[gi]] for gi in range(n_global)]
            t_counts = [f.shape[0] for f in t_masked]
            t_ph = list(torch.split(self.teacher_ibot_head(torch.cat(t_masked, 0)), t_counts, 0))
            # Sinkhorn-Knopp over the concatenated global crops (crop-major), then split.
            teacher_cls_probs = list(self.dino_loss.sinkhorn_knopp_teacher(
                torch.cat(t_cls_h, 0), teacher_temp,
                n_iterations=self.sinkhorn_iterations).chunk(n_global, 0))
            patch_cat = torch.cat(t_ph, 0)
            n_masked_t = torch.tensor(sum(t_counts), device=patch_cat.device)
            patch_cat = self.ibot_loss.sinkhorn_knopp_teacher(
                patch_cat, teacher_temp, n_masked_t, n_iterations=self.sinkhorn_iterations)
            teacher_patch_probs = list(torch.split(patch_cat, t_counts, 0))

        # ── student: globals (masked) + locals ──
        s_gouts = self._run_backbone(self.student_backbone, globals_, tv_g, masks=masks)
        student_global_cls_feat = [self._pool(o) for o in s_gouts]
        s_louts = self._run_backbone(self.student_backbone, locals_, tv_l)
        student_local_cls_feat = [self._pool(o) for o in s_louts]
        all_cls = self._run_head(self.student_dino_head,
                                 student_global_cls_feat + student_local_cls_feat)
        student_global_cls_after = all_cls[:n_global]
        student_local_cls_after = all_cls[n_global:]
        s_masked = [s_gouts[gi]["x_norm_patchtokens"][masks[gi]] for gi in range(n_global)]
        s_counts = [f.shape[0] for f in s_masked]
        student_patch_after = list(torch.split(
            self.student_ibot_head(torch.cat(s_masked, 0)), s_counts, 0))

        # ── DINO class-token loss: every student view against every *other* teacher global ──
        dino = globals_[0].new_zeros(())
        n_terms = 0
        for ti in range(n_global):
            s_others = [student_global_cls_after[sj] for sj in range(n_global) if sj != ti] \
                + student_local_cls_after
            if s_others:
                dino = dino + self.dino_loss(s_others, [teacher_cls_probs[ti]])
                n_terms += len(s_others)
        dino = dino / max(1, n_terms)

        # ── iBOT patch loss (DINOv2 ``forward_masked`` normalisation) ──
        ibot = globals_[0].new_zeros(())
        for gi in range(n_global):
            tp, sp, mk = teacher_patch_probs[gi], student_patch_after[gi], masks[gi]
            if tp.shape[0] == 0:
                # No masked patch in this crop: keep a (zero-valued) gradient path through the
                # iBOT head so that DDP with find_unused_parameters=False sees every parameter.
                ibot = ibot + sp.sum() * 0
                continue
            w = (1.0 / mk.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(mk)[mk]   # masks_weight
            ce = -torch.sum(tp * F.log_softmax(sp / self.ibot_loss.student_temp, dim=-1), dim=-1)
            ibot = ibot + (ce * w).sum() / mk.shape[0]
        ibot = ibot / n_global

        # ── KoLeo on the student global class-token features, summed over the global crops ──
        koleo = sum(self.koleo_loss(f) for f in student_global_cls_feat)

        total = self.dino_weight * dino + self.ibot_weight * ibot + self.koleo_weight * koleo
        return {"loss": total, "dino": dino.detach(), "ibot": ibot.detach(), "koleo": koleo.detach()}

    @torch.no_grad()
    def encode_features(self, x, M):
        """Teacher features for the collapse monitor: mean of the valid patch tokens, ``[B, D]``."""
        ps = self.teacher_backbone.patch_size
        tv = self._tv(M, ps)
        out = self.teacher_backbone(x, ibot_mask=None, token_valid=tv)
        patch = out["x_norm_patchtokens"]
        w = tv.unsqueeze(-1).to(patch.dtype)
        return (patch * w).sum(1) / w.sum(1).clamp(min=1.0)
