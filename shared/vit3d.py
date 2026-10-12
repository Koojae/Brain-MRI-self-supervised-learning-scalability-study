"""Shared 3D Vision Transformer encoder used by all four SSL methods.

Every method in this repository (MAE, JEPA, DINO, SimCLR) trains exactly this encoder, so
the downstream comparison isolates the self-supervised objective.  Design, fixed for all runs:

* ``Conv3d`` patch embedding with patch size 16.  A 96^3 input gives a 6^3 = 216 token grid.
* Pre-norm transformer blocks: multi-head self-attention through
  ``F.scaled_dot_product_attention`` (flash kernels), GELU MLP with hidden size 4x.
* **3D axial RoPE** instead of an additive positional embedding.  Queries and keys are
  rotated before attention; the head dimension is split into three equal axis groups of
  ``2 * ((head_dim // 3) // 2)`` channels each (64 -> 20/20/20 with 4 unrotated channels,
  80 -> 26/26/26 with 2 unrotated).  Frequencies are expanded with ``repeat_interleave`` so
  that each adjacent channel pair shares one frequency, which makes attention logits
  translation-invariant.  The class token is never rotated.  Coordinates of a smaller grid
  (DINO local crops, 3^3) are rescaled onto the base grid (6^3) so both crop sizes share one
  coordinate system.
* **Stabilized initialisation**: every ``nn.Linear`` and the patch-embedding convolution use
  ``trunc_normal(std=0.02)``; the fused ``qkv`` projection uses the separate-fan uniform
  initialisation of MoCo v3 (effective std ``1/sqrt(D)``).  LayerNorm keeps PyTorch defaults.
* **Residual depth rescaling** (V-JEPA 2 ``_rescale_blocks``): after initialisation the
  output projections of block ``l`` (``attn.proj`` and ``mlp.fc2``) are divided by
  ``sqrt(2 * l)`` with ``l`` 1-indexed.
* Optional **QK-normalisation** (ViT-22B): a per-head LayerNorm on q and k before the dot
  product.  It is enabled for all ViT-Huge runs and for every SimCLR run.  ``eps`` is 1e-2
  rather than the PyTorch default because skull-stripped background patches are exactly
  zero; with ``eps=1e-5`` the LayerNorm backward on those tokens amplifies the patch-embedding
  gradient by more than an order of magnitude.
* **Valid-token encoding**: when ``token_valid`` is given, background tokens are dropped
  from the sequence (left-packed gather), padded slots are excluded from the attention keys,
  and the outputs are scattered back to the full grid with zeros at invalid positions.

Model family (``embed_dim, mlp_dim, num_heads, depth``): tiny (192, 768, 3, 12),
base (768, 3072, 12, 12), huge (1280, 5120, 16, 32).  Small and large are listed for
completeness but were not part of the study.

The ``state_dict`` layout is the transfer contract with the downstream evaluation code:
``patch_embed.proj.*``, ``cls_token``, ``mask_token``, ``blocks.{i}.{norm1,attn.qkv,attn.proj,
norm2,mlp.fc1,mlp.fc2}.*``, ``norm.*``, plus ``blocks.{i}.attn.{q_norm,k_norm}.*`` when
``qk_norm`` is on.  RoPE adds no parameters, so a checkpoint's ``meta`` must be consulted to
rebuild the encoder with the right options.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# (embed_dim, mlp_dim, num_heads, depth) for patch size 16.
VIT_FAMILY = {
    "tiny": (192, 768, 3, 12),
    "small": (384, 1536, 6, 12),
    "base": (768, 3072, 12, 12),
    "large": (1024, 4096, 16, 24),
    "huge": (1280, 5120, 16, 32),
}

INIT_STD = 0.02        # trunc_normal std shared by DINOv2 and V-JEPA 2
QK_NORM_EPS = 1e-2     # see module docstring; not the PyTorch LayerNorm default
ROPE_THETA = 10000.0


# ─────────────────────────────── initialisation ───────────────────────────────
def init_linear_(m: nn.Linear, qkv: bool = False):
    """Stabilized ``nn.Linear`` init.

    ``qkv=True`` applies the MoCo v3 separate-fan rule to the fused ``[3D, D]`` projection:
    ``fan_out`` is taken as ``shape[0] // 3`` so the scale equals that of three independent
    projections (effective std ``1/sqrt(D)``).  Otherwise ``trunc_normal(std=0.02)``.
    Biases are zeroed.
    """
    if qkv:
        val = math.sqrt(6.0 / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
        nn.init.uniform_(m.weight, -val, val)
    else:
        nn.init.trunc_normal_(m.weight, std=INIT_STD)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


# ─────────────────────────────────── RoPE ───────────────────────────────────
def rope_dim_per_axis(head_dim: int) -> int:
    """Rotated channels per spatial axis: ``2 * ((head_dim // 3) // 2)`` (V-JEPA 2 rule)."""
    return int(2 * ((head_dim // 3) // 2))


def rope_sincos(pos: torch.Tensor, dim: int):
    """Precompute ``(sin, cos)`` for one axis.

    ``pos`` is ``[N]`` or ``[B, N]`` (float coordinates); the result broadcasts over the head
    axis as ``[..., 1, N, dim]``.  Angles are computed in fp32 even under bf16 autocast, and
    frequencies are interleaved so that channel pairs ``(2i, 2i+1)`` share one frequency.
    Computed once per forward and reused by every block.  ``dim`` is the per-axis rotated
    width from :func:`rope_dim_per_axis` (even and positive for every supported head size).
    """
    if pos.dim() == 2:
        pos = pos.unsqueeze(1)                                    # [B, 1, N]
    omega = torch.arange(dim // 2, dtype=torch.float32, device=pos.device) / (dim / 2.0)
    omega = 1.0 / (ROPE_THETA ** omega)
    freq = pos.to(torch.float32).unsqueeze(-1) * omega            # [..., N, dim/2]
    return freq.sin().repeat_interleave(2, dim=-1), freq.cos().repeat_interleave(2, dim=-1)


def rope_fuse_axes(sc, head_dim: int):
    """Concatenate the three per-axis ``(sin, cos)`` tables into one ``head_dim`` table.

    Unrotated tail channels (``head_dim - 3 * dim_axis``) get ``sin=0, cos=1``, which is the
    identity under the rotation formula, so a single rotation call covers the whole head.
    """
    sins = [t[0] for t in sc]
    coss = [t[1] for t in sc]
    used = sum(t.shape[-1] for t in sins)
    tail = head_dim - used
    assert tail >= 0 and tail % 2 == 0, (head_dim, used, tail)
    ref = sins[0]
    if tail:
        shape = list(ref.shape[:-1]) + [tail]
        sins.append(torch.zeros(shape, device=ref.device, dtype=ref.dtype))
        coss.append(torch.ones(shape, device=ref.device, dtype=ref.dtype))
    return torch.cat(sins, dim=-1), torch.cat(coss, dim=-1)


def rope_rotate(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent channel pairs of ``x`` with precomputed ``(sin, cos)``."""
    s_, c_ = sin.to(x.dtype), cos.to(x.dtype)
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    return x * c_ + y * s_


def rope_coords_from_idx(grid, idx, base_grid, device=None):
    """Token indices -> float ``(d, h, w)`` coordinates ``[..., 3]``.

    ``idx`` (``[N]`` or ``[B, N]`` long) selects a token subset; ``None`` means the full
    grid.  Index decomposition follows the ``flatten(2)`` order of the patch embedding
    (row-major over ``gd, gh, gw``).  When ``grid`` differs from ``base_grid`` the
    coordinates are rescaled linearly onto the base grid (a 3^3 grid maps to 0, 2.5, 5 on a
    6^3 base), so crops of different sizes share one coordinate system.
    """
    gd, gh, gw = (int(g) for g in grid)
    i = torch.arange(gd * gh * gw, device=device) if idx is None else idx
    per_frame, per_row = gh * gw, gw
    d = torch.div(i, per_frame, rounding_mode="floor")
    r = i - d * per_frame
    h = torch.div(r, per_row, rounding_mode="floor")
    w = r - h * per_row
    c = torch.stack([d, h, w], dim=-1).float()
    scale = [((b - 1.0) / (g - 1.0)) if (g != b and g > 1) else 1.0
             for g, b in zip((gd, gh, gw), base_grid)]
    if any(sc != 1.0 for sc in scale):
        c = c * torch.tensor(scale, device=c.device, dtype=c.dtype)
    return c


def rope_arg_from_coords(coords: torch.Tensor, head_dim: int, n_prefix: int = 0):
    """Build the ``rope`` tuple consumed by :class:`Block` / :class:`Attention`.

    ``coords`` is ``[N, 3]`` or ``[B, N, 3]``; ``n_prefix`` leading tokens (the class token)
    are left unrotated.  Returns ``(sin, cos, n_prefix)`` with fused ``head_dim`` tables.
    """
    d = rope_dim_per_axis(head_dim)
    assert d > 0, f"head_dim {head_dim} is too small for 3D RoPE (needs at least 6)"
    sc = [rope_sincos(coords[..., a], d) for a in range(3)]
    sin, cos = rope_fuse_axes(sc, head_dim)
    return sin, cos, int(n_prefix)


def rope_apply_qk(q, k, rope):
    """Apply the fused rotation to ``q`` and ``k`` (``[B, heads, N, head_dim]``)."""
    sin, cos, n_prefix = rope
    out = []
    for t in (q, k):
        body = t[:, :, n_prefix:] if n_prefix else t
        body = rope_rotate(body, sin, cos)
        out.append(torch.cat([t[:, :, :n_prefix], body], dim=2) if n_prefix else body)
    return out[0], out[1]


# ─────────────────────────────────── blocks ───────────────────────────────────
class Attention(nn.Module):
    """Multi-head self-attention with optional QK-normalisation and RoPE.

    Order of operations: fused qkv projection -> per-head LayerNorm on q and k (when
    ``qk_norm``) -> RoPE rotation of q and k -> ``scaled_dot_product_attention`` -> output
    projection.  ``attn_mask`` is a boolean key mask ``[B, N]`` (True = attend to this key);
    it is broadcast over queries and heads.  When ``qk_norm`` is off the norms are
    ``nn.Identity`` and the ``state_dict`` gains no extra keys.
    """

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True,
                 qk_norm: bool = False, qk_norm_eps: float = QK_NORM_EPS):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        init_linear_(self.qkv, qkv=True)
        init_linear_(self.proj)
        self.qk_norm = bool(qk_norm)
        self.qk_norm_eps = float(qk_norm_eps)
        self.q_norm = nn.LayerNorm(self.head_dim, eps=self.qk_norm_eps) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim, eps=self.qk_norm_eps) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None,
                rope: tuple | None = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                       # each [B, heads, N, head_dim]
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = rope_apply_qk(q, k, rope)
        sdpa_mask = None
        if attn_mask is not None:
            sdpa_mask = attn_mask[:, None, None, :].to(torch.bool)   # key mask, broadcast
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        init_linear_(self.fc1)
        init_linear_(self.fc2)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm transformer block.  Also reused by the MAE decoder and the JEPA predictor."""

    def __init__(self, dim: int, mlp_dim: int, num_heads: int, qkv_bias: bool = True,
                 qk_norm: bool = False, qk_norm_eps: float = QK_NORM_EPS):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_norm=qk_norm, qk_norm_eps=qk_norm_eps)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_dim)

    def forward(self, x, attn_mask: torch.Tensor | None = None, rope: tuple | None = None):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask, rope=rope)
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed3D(nn.Module):
    """Non-overlapping ``Conv3d`` patch embedding; returns tokens ``[B, N, D]`` and the grid."""

    def __init__(self, in_chans: int, embed_dim: int, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        nn.init.trunc_normal_(self.proj.weight, std=INIT_STD)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)                                    # [B, D, gd, gh, gw]
        grid = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)                    # [B, N, D]
        return x, tuple(int(g) for g in grid)


# ─────────────────────────────────── encoder ───────────────────────────────────
class ViT3D(nn.Module):
    """Shared 3D ViT encoder (see module docstring for the fixed design choices).

    Args:
        arch: one of :data:`VIT_FAMILY`.
        in_chans: input channels (1 for MRI).
        patch_size: cubic patch edge in voxels.
        qkv_bias: bias in the qkv projection.
        pos_base_grid: token grid of the method's reference crop (``img_size // patch_size``
            per axis).  RoPE coordinates of other grids are rescaled onto it.
        qk_norm, qk_norm_eps: QK-normalisation switch and LayerNorm eps.
    """

    def __init__(self, arch: str = "base", in_chans: int = 1, patch_size: int = 16,
                 qkv_bias: bool = True, pos_base_grid: tuple = (6, 6, 6),
                 qk_norm: bool = False, qk_norm_eps: float = QK_NORM_EPS):
        super().__init__()
        embed_dim, mlp_dim, num_heads, depth = VIT_FAMILY[arch]
        self.arch = arch
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.patch_size = patch_size
        self.qk_norm = bool(qk_norm)
        self.qk_norm_eps = float(qk_norm_eps)
        self.patch_embed = PatchEmbed3D(in_chans, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))    # iBOT mask token (DINO); frozen by MAE, JEPA, SimCLR
        self.blocks = nn.ModuleList(
            [Block(embed_dim, mlp_dim, num_heads, qkv_bias, qk_norm=qk_norm,
                   qk_norm_eps=qk_norm_eps) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_base_grid = tuple(int(g) for g in pos_base_grid)
        self.head_dim = embed_dim // num_heads
        self.grad_checkpointing = False   # recompute block activations (memory only, same math)
        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=INIT_STD)
        nn.init.zeros_(self.mask_token)
        self._rescale_blocks()

    def _rescale_blocks(self):
        """Divide the residual output weights of block ``l`` by ``sqrt(2 * l)`` (1-indexed)."""
        for lid, blk in enumerate(self.blocks):
            f = math.sqrt(2.0 * (lid + 1))
            blk.attn.proj.weight.data.div_(f)
            blk.mlp.fc2.weight.data.div_(f)

    # ── RoPE helpers ──
    def rope_coords(self, grid, idx: torch.Tensor | None = None, device=None):
        """Coordinates for ``grid`` (optionally a token subset ``idx``) on this encoder's base grid."""
        return rope_coords_from_idx(grid, idx, self.pos_base_grid, device)

    def rope_arg(self, grid, idx=None, n_prefix: int = 0, device=None):
        """The ``rope`` tuple for :meth:`run_blocks`; computed once per forward."""
        return rope_arg_from_coords(self.rope_coords(grid, idx, device), self.head_dim, n_prefix)

    # ── forward pieces ──
    def tokenize(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """``x [B, 1, D, H, W]`` -> ``(tokens [B, N, D], grid)``.

        No class token and no additive positional term (positions enter through RoPE).
        ``mask [B, N]`` bool replaces the selected tokens with ``mask_token`` (iBOT).
        """
        tokens, grid = self.patch_embed(x)
        if mask is not None:
            tokens = torch.where(
                mask.unsqueeze(-1), self.mask_token.to(tokens.dtype)[None], tokens
            )
        return tokens, grid

    def run_blocks(self, tokens: torch.Tensor, attn_mask: torch.Tensor | None = None,
                   norm: bool = True, rope: tuple | None = None) -> torch.Tensor:
        """Run ``tokens [B, K, D]`` through all blocks (+ final norm).  ``attn_mask [B, K]`` bool."""
        ckpt = self.grad_checkpointing and self.training and torch.is_grad_enabled()
        for blk in self.blocks:
            if ckpt:
                tokens = torch.utils.checkpoint.checkpoint(blk, tokens, attn_mask, rope,
                                                           use_reentrant=False)
            else:
                tokens = blk(tokens, attn_mask=attn_mask, rope=rope)
        return self.norm(tokens) if norm else tokens

    @staticmethod
    def _pack_valid(tokens: torch.Tensor, token_valid: torch.Tensor):
        """Left-pack valid tokens.

        Returns ``(packed [B, Kmax, D], pad_mask [B, Kmax] bool, gather_idx [B, Kmax])`` where
        ``pad_mask`` is True for real tokens and ``gather_idx`` maps packed slots back to grid
        positions.  Original token order is preserved (stable sort).
        """
        B, N, D = tokens.shape
        counts = token_valid.sum(dim=1)                                          # [B]
        Kmax = int(counts.max().clamp_min(1).item())
        gather_idx = torch.argsort(token_valid.to(torch.int8), dim=1,
                                   descending=True, stable=True)[:, :Kmax]       # [B, Kmax]
        packed = torch.gather(tokens, 1, gather_idx[..., None].expand(-1, -1, D))
        pad_mask = torch.arange(Kmax, device=tokens.device)[None, :] < counts[:, None]
        return packed, pad_mask, gather_idx

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None,
                token_valid: torch.Tensor | None = None) -> dict:
        """Full forward with a class token.

        ``token_valid [B, N]`` bool drops background tokens from the sequence; padded slots
        are masked out of the attention keys and the patch outputs are scattered back to
        ``[B, N, D]`` with exact zeros at invalid positions.  If every token is valid the
        full-grid path is used (no gather, ``attn_mask=None``, flash kernel).

        Returns ``{"x_norm_clstoken": [B, D], "x_norm_patchtokens": [B, N, D], "grid": grid}``
        plus ``"token_valid"`` when tokens were dropped.
        """
        tokens, grid = self.tokenize(x, mask)
        B, N, D = tokens.shape
        cls = self.cls_token.expand(B, -1, -1).to(tokens.dtype)

        if token_valid is not None and bool(token_valid.all()):
            token_valid = None

        if token_valid is None:
            seq = torch.cat([cls, tokens], dim=1)                                # [B, 1+N, D]
            seq = self.run_blocks(seq, attn_mask=None, norm=True,
                                  rope=self.rope_arg(grid, None, 1, x.device))
            return {"x_norm_clstoken": seq[:, 0], "x_norm_patchtokens": seq[:, 1:], "grid": grid}

        packed, pad_mask, gather_idx = self._pack_valid(tokens, token_valid)     # [B, Kmax, D]
        seq = torch.cat([cls, packed], dim=1)                                    # [B, 1+Kmax, D]
        attn = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device), pad_mask], dim=1)
        # RoPE coordinates follow the gather order; padded slots are excluded from the keys
        # and their outputs are discarded below.
        seq = self.run_blocks(seq, attn_mask=attn, norm=True,
                              rope=self.rope_arg(grid, gather_idx, 1, x.device))
        cls_out, patch_valid = seq[:, 0], seq[:, 1:]                             # [B, D], [B, Kmax, D]
        patch_full = torch.zeros(B, N, D, device=x.device, dtype=patch_valid.dtype)
        patch_full.scatter_(1, gather_idx[..., None].expand(-1, -1, D), patch_valid)
        patch_full = patch_full * token_valid.unsqueeze(-1).to(patch_full.dtype)
        return {
            "x_norm_clstoken": cls_out,
            "x_norm_patchtokens": patch_full,   # [B, N, D], zeros at invalid positions
            "token_valid": token_valid,
            "grid": grid,
        }
