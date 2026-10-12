"""SimCLR model: shared 3D ViT encoder (class-token readout) + 2-layer projection head.

Only the encoder (``backbone.vit``) is transferred downstream; the projection head is a
training-time module.  Under DDP the head's ``BatchNorm1d`` is converted to
``SyncBatchNorm`` by the training script so batch statistics span all ranks.
"""

import torch.nn as nn

from shared.data import token_valid
from shared.vit3d import ViT3D


class ViTEncoder(nn.Module):
    """Shared :class:`shared.vit3d.ViT3D` with brain-valid token dropping and class-token pooling.

    Args:
        arch: ViT family member (``tiny`` / ``base`` / ``huge``).
        in_chans: input channels (1).
        patch_size: cubic patch edge (16).
        valid_token_threshold: minimum brain fraction for a patch to enter the sequence.
        enc_kwargs: extra :class:`ViT3D` options (``pos_base_grid``, ``qk_norm``, ``qk_norm_eps``).

    The encoder's ``mask_token`` (used only by the DINO / iBOT objective) is unused by SimCLR
    and frozen; the class token is trained.
    """

    def __init__(self, arch="base", in_chans=1, patch_size=16, valid_token_threshold=0.1,
                 enc_kwargs=None):
        super().__init__()
        self.vit = ViT3D(arch=arch, in_chans=in_chans, patch_size=patch_size, **(enc_kwargs or {}))
        self.vt = float(valid_token_threshold)
        self.patch = patch_size
        self.embed_dim = self.vit.embed_dim
        self.vit.mask_token.requires_grad_(False)

    def forward(self, x, M):
        """``x [B, 1, S, S, S]``, crisp brain mask ``M`` (same shape) -> class token ``[B, D]``."""
        tv = token_valid(M, self.patch, self.vt)
        out = self.vit(x, token_valid=tv)
        return out["x_norm_clstoken"]


class ProjectionHead(nn.Module):
    """``Linear(in_dim, hidden) -> BatchNorm1d -> ReLU -> Linear(hidden, out)``."""

    def __init__(self, in_dim, hidden=2048, out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out),
        )
        self.mlp_dim, self.out_dim = int(hidden), int(out)

    def forward(self, x):
        return self.net(x)


class SimCLRModel(nn.Module):
    """Encoder + projection head.  ``forward(x, M) -> (z [B, proj_dim], h [B, embed_dim])``."""

    def __init__(self, arch="base", in_chans=1, patch_size=16, valid_token_threshold=0.1,
                 proj_hidden=2048, proj_dim=128, enc_kwargs=None):
        super().__init__()
        self.backbone = ViTEncoder(arch, in_chans, patch_size, valid_token_threshold,
                                   enc_kwargs=enc_kwargs)
        self.proj = ProjectionHead(self.backbone.embed_dim, proj_hidden, proj_dim)

    def forward(self, x, M):
        h = self.backbone(x, M)      # [B, embed_dim]
        z = self.proj(h)             # [B, proj_dim]
        return z, h
