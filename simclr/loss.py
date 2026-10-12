"""NT-Xent contrastive loss with gradient-preserving all-gather across DDP ranks.

The loss is Eq. (1) of the SimCLR paper -- the mean of the per-anchor NT-Xent terms over all
``2B`` crops of a batch of ``B`` volumes -- generalised to an arbitrary number of positives
per anchor: for every crop, the positives are all other crops of the same volume and the
negatives are every remaining crop in the batch.  With two views per volume each anchor has
exactly one positive and ``2B - 2`` negatives, which is the standard SimCLR pairing.

Note on normalisation: the reference implementation (google-research/simclr,
``objective.py::add_contrastive_loss``) computes ``loss_a + loss_b`` where each term is
``tf.losses.softmax_cross_entropy`` over one view, i.e. a mean over the ``B`` anchors of that
view; the sum is therefore exactly twice the mean over all ``2B`` anchors returned here.  Keep
this factor in mind when comparing logged loss values, or the gradient scale implied by a
learning rate, against the reference code.

Under DDP the embeddings of all ranks are gathered so that negatives span the global batch.
The embeddings are gathered through :class:`GatherLayer`, a custom autograd function whose
backward all-reduces the gradient and returns the local shard, so gradients flow to every
rank's encoder exactly as if the whole batch were on one device.  The volume ids are
gathered without gradient; the caller must make them globally unique (rank offset).
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _dist_ready():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


class GatherLayer(torch.autograd.Function):
    """All-gather along dim 0 that keeps the gradient for the local shard.

    Identity when no multi-rank process group is initialised, so the layer is safe to use
    in a single-process run.
    """

    @staticmethod
    def forward(ctx, x):
        if not _dist_ready():
            return x
        out = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(out, x.contiguous())
        return torch.cat(out, dim=0)

    @staticmethod
    def backward(ctx, grad):
        if not _dist_ready():
            return grad
        grad = grad.contiguous()
        dist.all_reduce(grad)
        ws, rank = dist.get_world_size(), dist.get_rank()
        n = grad.shape[0] // ws
        return grad[rank * n:(rank + 1) * n]


def _gather_nograd(t):
    """All-gather along dim 0 without gradient (labels); identity without a process group."""
    if not _dist_ready():
        return t
    out = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out, t.contiguous())
    return torch.cat(out, dim=0)


class NTXentLoss(nn.Module):
    """NT-Xent over L2-normalised embeddings; positives = other crops of the same volume.

    Args:
        temperature: softmax temperature ``tau`` (0.1).
        gather: all-gather embeddings and ids across ranks before computing the loss.

    ``forward(z [m, D], vol_ids [m])`` returns the mean over anchors that have at least one
    positive of ``-mean_{p in positives} log softmax(sim / tau)[anchor, p]``, where the
    softmax runs over all other crops (self excluded).  Logits are shifted by the per-row
    maximum (detached) for numerical stability.  When called under autocast the similarity
    matrix is computed in the autocast dtype.
    """

    def __init__(self, temperature=0.1, gather=True):
        super().__init__()
        self.t = temperature
        self.gather = gather

    def forward(self, z, vol_ids):
        if self.gather and _dist_ready():
            z = GatherLayer.apply(z)
            vol_ids = _gather_nograd(vol_ids)
        z = F.normalize(z, dim=1)
        m = z.shape[0]
        sim = (z @ z.t()) / self.t
        self_mask = torch.eye(m, dtype=torch.bool, device=z.device)
        logits = sim - sim.masked_fill(self_mask, float("-inf")).max(dim=1, keepdim=True).values.detach()
        exp = torch.exp(logits).masked_fill(self_mask, 0.0)
        log_prob = logits - torch.log(exp.sum(dim=1, keepdim=True) + 1e-12)

        pos = vol_ids[:, None] == vol_ids[None, :]
        pos = pos & ~self_mask
        pos_count = pos.sum(dim=1)
        loss = -(pos * log_prob).sum(dim=1) / pos_count.clamp(min=1)
        valid = pos_count > 0
        return loss[valid].mean() if valid.any() else sim.sum() * 0.0
