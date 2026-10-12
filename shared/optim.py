"""Optimizer, schedules, EMA and training diagnostics shared by all four SSL methods.

Recipe (identical across methods unless a method's config says otherwise):

* **AdamW**, ``eps = 1e-8``.  Two parameter classes: tensors with ``ndim >= 2`` receive
  weight decay -- the weights, and also the class / mask tokens, which are stored as
  ``[1, 1, D]`` / ``[1, D]`` (likewise the MAE decoder and JEPA predictor mask tokens);
  1-D tensors (biases, LayerNorm affine parameters) receive none.  The patch-embedding
  parameters of the shared encoder get a learning-rate multiplier
  (``patch_embed_lr_mult = 0.2`` in all runs).  Weight decay is constant over training.
* **Peak learning rate** = ``base_lr * global_volumes / 256`` (linear scaling with the
  number of *volumes* per optimizer step; the number of crops per volume does not enter).
  With 8 GPUs x 64 volumes the peak is ``2 * base_lr``.
* **Schedule**: linear warm-up over the first ``warmup_frac`` of the steps
  (``(step + 1) / warm``), then cosine decay to an absolute floor ``min_lr``.
* **EMA** momentum for teacher/target encoders: ``cosine`` from ``m0`` to ``m1`` (DINO,
  0.992 -> 1.0) or ``const`` (JEPA, 0.99925).  Updated with ``torch._foreach_*`` kernels.
* **Gradient clipping** by global norm (``clip_grad = 3.0``) every step.

Diagnostics logged periodically to stdout: the numerical / effective rank of the feature
covariance (representation collapse) and first-block attention statistics (attention-logit
blow-up), see :func:`feature_covariance_rank` and :func:`block0_attention_stats`.
"""

import math

import torch

from shared.vit3d import ViT3D, rope_apply_qk

BASE_BATCH = 256          # volumes per step at which peak_lr == base_lr
EPS = 1e-8


# ─────────────────────────────── optimizer ───────────────────────────────
def _patch_embed_lr_mults(modules, patch_embed_lr_mult):
    """param id -> lr multiplier for the patch embedding of every :class:`ViT3D` found."""
    out = {}
    if float(patch_embed_lr_mult) == 1.0:
        return out
    for top in modules:
        for _, sub in top.named_modules():
            if not isinstance(sub, ViT3D):
                continue
            for n, p in sub.named_parameters(recurse=True):
                if "patch_embed" in n:
                    out[id(p)] = float(patch_embed_lr_mult)
    return out


def build_param_groups(modules, wd, patch_embed_lr_mult=1.0):
    """Split parameters into (lr multiplier, decay/no-decay) groups.

    Only parameters with ``requires_grad`` are included, each exactly once.  Groups are
    ordered decay-first and by ascending multiplier.  Returns ``(groups, lr_mults)``; the
    multipliers are kept outside the param-group dicts so that loading an optimizer state
    cannot overwrite them.
    """
    if not isinstance(modules, (list, tuple)):
        modules = [modules]
    mmap = _patch_embed_lr_mults(modules, patch_embed_lr_mult)
    buckets, seen = {}, set()
    for m in modules:
        for p in m.parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            buckets.setdefault((mmap.get(id(p), 1.0), p.ndim > 1), []).append(p)
    groups, mults = [], []
    for (mult, is_decay), ps in sorted(buckets.items(), key=lambda kv: (not kv[0][1], kv[0][0])):
        groups.append({"params": ps, "weight_decay": float(wd) if is_decay else 0.0})
        mults.append(mult)
    return groups, mults


def build_adamw(modules, peak_lr, wd, betas, patch_embed_lr_mult=1.0, eps=EPS):
    """AdamW over ``modules`` (a module or list of modules) with the shared param grouping."""
    groups, mults = build_param_groups(modules, wd=wd, patch_embed_lr_mult=patch_embed_lr_mult)
    opt = torch.optim.AdamW(groups, lr=peak_lr, betas=tuple(betas), eps=eps)
    opt._lr_mults = mults          # plain attribute: not part of state_dict
    return opt


# ─────────────────────────────── learning rate ───────────────────────────────
def compute_peak_lr(base_lr, global_volumes, base_batch=BASE_BATCH):
    """``peak_lr = base_lr * global_volumes / base_batch``."""
    return float(base_lr) * float(global_volumes) / float(base_batch)


def lr_at(step, total, peak_lr, warmup_frac=0.1, min_lr=1e-6):
    """Learning rate at 0-indexed ``step``: linear warm-up, then cosine decay to ``min_lr``.

    ``warm = round(warmup_frac * total)`` steps of warm-up with factor ``(step + 1) / warm``;
    afterwards ``min_lr + (peak_lr - min_lr) * 0.5 * (1 + cos(pi * progress))`` where
    ``progress`` runs from 0 at the end of warm-up to 1 at ``step == total``.
    """
    total = max(1, int(total))
    warm = max(1, int(round(warmup_frac * total)))
    if step < warm:
        factor = (step + 1) / warm
    else:
        prog = (step - warm) / max(1, total - warm)
        prog = min(1.0, max(0.0, prog))
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        eta = float(min_lr) / float(peak_lr) if peak_lr else 0.0
        factor = eta + (1.0 - eta) * cos
    return peak_lr * factor


def set_lr(optimizer, lr):
    """Set ``lr * multiplier`` on every param group; returns the reference ``lr``."""
    mults = getattr(optimizer, "_lr_mults", None)
    for i, pg in enumerate(optimizer.param_groups):
        pg["lr"] = lr * (mults[i] if mults else 1.0)
    return lr


def lr_scaling_summary(base_lr, global_volumes, peak_lr=None):
    """Dict logged once at start-up and stored in checkpoint ``meta``."""
    if peak_lr is None:
        peak_lr = compute_peak_lr(base_lr, global_volumes)
    return {
        "lr/base_lr": float(base_lr),
        "lr/eff_batch_volumes": int(global_volumes),
        "lr/lr_scale": float(global_volumes) / float(BASE_BATCH),
        "lr/peak_lr": float(peak_lr),
    }


# ─────────────────────────────── EMA ───────────────────────────────
def ema_momentum(step, total, m0, m1, schedule):
    """Teacher / target EMA momentum at ``step``.

    ``schedule="cosine"``: ``m1 + (m0 - m1) * 0.5 * (1 + cos(pi * step / total))``
    (DINOv2 ``CosineScheduler``, no warm-up).  ``schedule="const"``: ``m0``.
    """
    total = max(1, int(total))
    if schedule == "const":
        return float(m0)
    if schedule == "cosine":
        prog = min(1.0, max(0.0, step / total))
        return float(m1) + (float(m0) - float(m1)) * 0.5 * (1.0 + math.cos(math.pi * prog))
    raise ValueError(f"unknown ema_schedule: {schedule!r}")


class EmaUpdater:
    """``teacher = m * teacher + (1 - m) * student`` over paired module lists.

    Parameter pairs are collected once; the update runs as fused ``_foreach`` kernels.
    Buffers (none in the shared encoder or the DINO heads) are copied from the student.
    """

    def __init__(self, student_mods, teacher_mods):
        self.s, self.t = [], []
        for sm, tm in zip(student_mods, teacher_mods):
            for ps, pt in zip(sm.parameters(), tm.parameters()):
                self.s.append(ps)
                self.t.append(pt)
        self.sb, self.tb = [], []
        for sm, tm in zip(student_mods, teacher_mods):
            for bs, bt in zip(sm.buffers(), tm.buffers()):
                self.sb.append(bs)
                self.tb.append(bt)

    @torch.no_grad()
    def step(self, m):
        torch._foreach_mul_(self.t, m)
        torch._foreach_add_(self.t, self.s, alpha=1.0 - m)
        if self.tb:
            torch._foreach_copy_(self.tb, self.sb)

    def n_params(self):
        return len(self.t)


# ─────────────────────────────── gradient clipping ───────────────────────────────
def clip_grad_norm(params, max_norm):
    """Clip by global norm; returns the pre-clip total norm as a tensor (no host sync)."""
    if not isinstance(params, (list, tuple)):
        params = list(params)
    return torch.nn.utils.clip_grad_norm_(params, max_norm)


# ─────────────────────────────── diagnostics ───────────────────────────────
EIG_REL_EPS = 1e-3        # eigenvalues below this fraction of the largest do not count towards ``rank``


@torch.no_grad()
def feature_covariance_rank(feats):
    """Numerical rank and effective rank of the feature covariance (collapse monitor).

    ``feats [N, D]`` (one feature vector per volume).  ``rank`` counts eigenvalues above
    ``EIG_REL_EPS * max``; ``eff_rank`` is the effective rank ``exp(H)`` of the normalised
    eigenvalue distribution (its perplexity; Roy & Vetterli, 2007).  Diagnostic only: on
    non-finite features or a failed eigendecomposition it returns sentinels instead of
    raising.
    """
    f = feats.detach().float()
    N, D = f.shape
    if N < 2:
        return {"rank": 0, "eff_rank": 0.0}
    if not torch.isfinite(f).all():
        return {"rank": -1, "eff_rank": float("nan")}
    f = f - f.mean(dim=0, keepdim=True)
    cov = (f.t() @ f) / (N - 1)
    try:
        ev = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    except Exception:
        try:
            jit = (float(cov.diagonal().mean().abs()) + 1e-6) * 1e-4
            eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
            ev = torch.linalg.eigvalsh(cov + jit * eye).clamp_min(0.0)
        except Exception:
            try:
                ev = torch.linalg.svdvals(cov).clamp_min(0.0)
            except Exception:
                return {"rank": -1, "eff_rank": float("nan")}
    total = float(ev.sum())
    if not (total > 0):
        return {"rank": 0, "eff_rank": 0.0}
    rank = int((ev > EIG_REL_EPS * float(ev.max())).sum().item())
    p = ev / ev.sum()
    p = p[p > 0]
    eff = float(torch.exp(-(p * p.log()).sum()).item())
    return {"rank": rank, "eff_rank": eff}


@torch.no_grad()
def block0_attention_stats(vit, x, max_vols: int = 8):
    """First-block attention statistics on the full token grid (attention blow-up monitor).

    Recomputes block-0 attention logits of ``vit`` (a :class:`ViT3D`) on up to ``max_vols``
    inputs with the same QK-norm and RoPE as the training path.  Returns:

    * ``attn_entn``: mean attention entropy divided by ``log(N)`` (healthy ~0.9, collapse < 0.2)
    * ``attn_ent``: the raw entropy in nats
    * ``dc_gain``: RMS over output channels of the patch-embedding kernel sums
    * ``key_var``: number of distinct arg-max keys / N (query dependence of attention)
    * ``query_tv``: mean total-variation distance between each query's attention row and the
      mean row (0 when every query attends identically)
    * ``logit_std``: standard deviation of the attention logits

    Diagnostic only: never raises; returns NaNs on failure.
    """
    nan = {"attn_entn": float("nan"), "attn_ent": float("nan"), "dc_gain": float("nan"),
           "key_var": float("nan"), "query_tv": float("nan"), "logit_std": float("nan")}
    try:
        vit = getattr(vit, "module", vit)
        blk = vit.blocks[0]
        pdt = vit.patch_embed.proj.weight.dtype
        xs = x[:max_vols].detach().to(pdt)
        content, grid = vit.patch_embed(xs)
        y = blk.norm1(content)
        B, N, _ = y.shape
        nh, hd = blk.attn.num_heads, blk.attn.head_dim
        qkv = blk.attn.qkv(y).float().reshape(B, N, 3, nh, hd).permute(2, 0, 3, 1, 4)
        q0, k0 = qkv[0], qkv[1]
        q0 = blk.attn.q_norm(q0.to(y.dtype)).float()
        k0 = blk.attn.k_norm(k0.to(y.dtype)).float()
        q0, k0 = rope_apply_qk(q0, k0, vit.rope_arg(grid, None, 0, y.device))
        logits = (q0 @ k0.transpose(-2, -1)) / math.sqrt(hd)
        p = logits.softmax(-1)
        ent = float(-(p * (p + 1e-12).log()).sum(-1).mean())
        am = logits.argmax(-1)                                                # [B, nh, N]
        hit = torch.zeros(B, nh, N, dtype=torch.bool, device=p.device).scatter_(-1, am, True)
        kvar = float(hit.sum(-1).float().mean()) / N
        qtv = 0.5 * float((p - p.mean(-2, keepdim=True)).abs().sum(-1).mean())
        lstd = float(logits.std())
        w = vit.patch_embed.proj.weight.detach().float()
        dc = float(w.flatten(1).sum(1).pow(2).mean().sqrt())
        if not math.isfinite(ent):
            return nan
        return {"attn_entn": ent / math.log(N), "attn_ent": ent, "dc_gain": dc,
                "key_var": kvar, "query_tv": qtv, "logit_std": lstd}
    except Exception:
        return nan


def block0_msg(a) -> str:
    """Log-line fragment for :func:`block0_attention_stats`."""
    return (f" ent0={a['attn_entn']:.3f} dc0={a['dc_gain']:.2f}"
            f" kvar={a['key_var']:.3f} qtv={a['query_tv']:.3f} lstd={a['logit_std']:.3g}")
