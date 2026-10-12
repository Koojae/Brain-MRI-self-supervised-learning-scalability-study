"""NeuroVFM two-way block masking on the brain-valid 3D token grid.

Port of the ``_MultiBlockCollator`` of NeuroVFM (MLNeurosurg/neurovfm) to the isotropic
``(G, G, G)`` token grid of this repository (``G = volume_size // patch_size = 6`` for a
96^3 volume with 16^3 patches) and to the brain-valid token mask ``token_valid``.  The
generator runs on the CPU inside the training loop, after the GPU resize / augmentation
has produced the brain mask of the batch.

Two masking *ways* are configured and one of them is drawn uniformly **per sample**:

* **way 1** (``switch_enc_pred: false``): ``npred`` rectangular HW blocks spanning the full
  depth axis are sampled; their union is the *target* and the complement is the *context*
  ("mask large blocks, predict them from the rest").
* **way 2** (``switch_enc_pred: true``): the sampled block becomes the *context* and its
  complement the *target* ("a small crop as context, predict everything else").

Both ways then share the same post-processing:

1. **depth drop** - if the context still covers more than ``enc_mask_scale`` of the
   foreground tokens, a contiguous slab of depth slices is removed from the context;
2. **patch dropout** - each surviving context token is kept with probability
   ``1 - patch_drop_rate``; dropped tokens become additional target tokens;
3. **brain confinement** - ``context = ctx & valid``, ``target = ~ctx & valid``, so
   background tokens appear on neither side (except in the degenerate fallback of step 4);
4. **acceptance** - the context must hold between ``MIN_ENC_FRAC`` and ``MAX_ENC_FRAC`` of
   the valid tokens and both sides must be non-empty; up to ``MAX_N_ATTEMPTS`` retries,
   then a deterministic split of the valid tokens is used as a fallback (a sample with
   fewer than two valid tokens is split over all ``G^3`` tokens, background included).

Every sample yields exactly one ``(context, target)`` index pair, so the predictor uses a
single mask token (``num_mask_tokens = 1``).

Randomness: each call creates a fresh ``torch.Generator`` seeded with the call index
(0, 1, 2, ...), so the mask sequence of a run depends only on the order of calls and on
the brain masks, not on the global torch RNG.

Padding: per-sample index lists have different lengths.  Each list is padded by cycling
its own indices, and a weight tensor (1 = real, 0 = padded) is returned for both sides.
``pred_weight`` excludes padded slots from the loss and both weights are used as attention
key masks, so repeated indices never act as extra keys.  Lengths are padded to the fixed
``pad_enc`` / ``pad_pred`` so that tensor shapes are constant from step to step; a batch
whose longest list exceeds the fixed length falls back to the batch maximum for that step
and is counted (see :meth:`MultiBlockMaskGenerator.pad_stats`).
"""

import math

import torch

# NeuroVFM constants (``_MultiBlockCollator.__call__`` / ``_sample_block_mask``).
MIN_ENC_FRAC = 0.05
MAX_ENC_FRAC = 0.5
MAX_N_ATTEMPTS = 5
INIT_RATIO = 0.95   # required foreground coverage of a placed block; decays by 0.05 every 5 tries


class MultiBlockMaskGenerator:
    """Two-way NeuroVFM mask generator with fixed-length padding.

    Args:
        cfgs_mask: list of way dicts (keys ``hw_block_scale``, ``aspect_ratio``, ``npred``,
            ``switch_enc_pred``, ``enc_mask_scale``, ``patch_drop_rate``, ``full_depth``).
        volume_size, patch_size: define the token grid ``G = volume_size // patch_size``.
        pad_enc, pad_pred: fixed padded lengths of the context and target index lists.

    Calling the generator with ``token_valid [B, G^3]`` (bool, CPU) returns four length-1
    lists ``([enc_idx], [pred_idx], [pred_weight], [enc_weight])`` with ``enc_idx
    [B, pad_enc]`` (long), ``pred_idx [B, pad_pred]`` (long) and float weights of the same
    shapes (1 = real token, 0 = padded slot).
    """

    def __init__(self, cfgs_mask, volume_size=96, patch_size=16, pad_enc=64, pad_pred=192):
        grid = volume_size // patch_size
        self.gen = _MaskGenerator(grid=grid, ways=cfgs_mask)
        self.pad_enc = int(pad_enc)
        self.pad_pred = int(pad_pred)
        # Fallback statistics: the overflow counters must stay at 0 for the padded lengths
        # to be constant over the whole run.
        self.n_overflow_enc = 0
        self.n_overflow_pred = 0
        self.max_enc_seen = 0
        self.max_pred_seen = 0
        self.shape_combos = set()

    def pad_stats(self):
        """Padding statistics for logging.  ``ovf_*`` > 0 means a fixed length was exceeded."""
        return {"ovf_enc": self.n_overflow_enc, "ovf_pred": self.n_overflow_pred,
                "max_enc": self.max_enc_seen, "max_pred": self.max_pred_seen,
                "n_shape_combos": len(self.shape_combos)}

    def __call__(self, token_valid_flat):
        me, mp, mw, ew = self.gen(token_valid_flat.size(0), token_valid_flat)
        self.max_enc_seen = max(self.max_enc_seen, int(me.shape[1]))
        self.max_pred_seen = max(self.max_pred_seen, int(mp.shape[1]))
        # Grow the batch-max padded tensors to the fixed lengths.  If a batch exceeds the
        # fixed length the batch maximum is kept (no index is ever dropped) and counted.
        if int(me.shape[1]) > self.pad_enc:
            self.n_overflow_enc += 1
        if int(mp.shape[1]) > self.pad_pred:
            self.n_overflow_pred += 1
        ke = max(self.pad_enc, int(me.shape[1]))
        kp = max(self.pad_pred, int(mp.shape[1]))

        def _grow(idx, w, k):
            B, K = idx.shape
            if k == K:
                return idx, w
            # Padded slots repeat the sample's first real index (always in range) with
            # weight 0, which removes them from both the loss and the attention keys.
            pad_i = idx[:, :1].expand(B, k - K)
            pad_w = torch.zeros(B, k - K, dtype=w.dtype)
            return torch.cat([idx, pad_i], 1), torch.cat([w, pad_w], 1)

        me, ew = _grow(me, ew, ke)
        mp, mw = _grow(mp, mw, kp)
        self.shape_combos.add((int(me.shape[1]), int(mp.shape[1])))
        return [me], [mp], [mw], [ew]


class _MaskGenerator:
    """NeuroVFM two-way single-block mask worker, confined to brain-valid tokens."""

    def __init__(self, grid=6, ways=None):
        assert ways, "at least one masking way is required"
        self.grid = grid
        self.N = grid ** 3
        self.ways = [self._norm_way(w) for w in ways]
        self._itr_counter = -1

    @staticmethod
    def _norm_way(w):
        """Normalise one way dict: coerce the seven NeuroVFM per-way knobs to plain types.

        Every key is required (a missing key raises ``KeyError``); the shipped
        ``configs/jepa.yaml`` lists all of them for both ways.  ``full_depth`` is accepted so
        that configs list the complete way specification; blocks always span the whole depth
        axis in this implementation.
        """
        return dict(
            hw_block_scale=tuple(w['hw_block_scale']),
            aspect_ratio=tuple(w['aspect_ratio']),
            npred=int(w['npred']),
            switch_enc_pred=bool(w['switch_enc_pred']),
            enc_mask_scale=float(w['enc_mask_scale']),
            patch_drop_rate=float(w['patch_drop_rate']),
            full_depth=bool(w['full_depth']),
        )

    def step(self):
        """Advance the call counter and return its value (the seed of this call)."""
        self._itr_counter += 1
        return self._itr_counter

    def _sample_block_size(self, way, filt_ratio, generator):
        """HW rectangle spanning the full depth axis (NeuroVFM ``_sample_block_size``).

        The HW area is ``hw_block_scale * filt_ratio`` of the ``G*G`` plane (``filt_ratio`` =
        foreground fraction of the sample); the aspect ratio is drawn from ``aspect_ratio``.
        Returns ``(d, h, w)`` on the isotropic ``(G, G, G)`` grid.
        """
        G = self.grid
        d = G   # full depth (axis 0); NeuroVFM d_pred_mask_scale = (1, 1)

        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = way['hw_block_scale']
        hw_scale = (min_s + _rand * (max_s - min_s)) * filt_ratio
        hw_num = max(1, int(G * G * hw_scale))   # floor, matching NeuroVFM int(H*W*hw_scale)

        _rand = torch.rand(1, generator=generator).item()
        min_ar, max_ar = way['aspect_ratio']
        ar = min_ar + _rand * (max_ar - min_ar)

        h = int(round(math.sqrt(hw_num * ar)))
        w = int(round(math.sqrt(hw_num / ar)))
        h = max(1, min(h, G))
        w = max(1, min(w, G))
        return d, h, w

    def _place_hw_block(self, d, h, w, valid_3d, generator):
        """Place a full-depth HW block (NeuroVFM ``_sample_block_mask``).

        Returns a bool ``[G, G, G]`` tensor (True inside the block).  The position is
        re-drawn until the block covers at least ``_ratio`` of the foreground, so the block
        sits on the brain; ``_ratio`` starts at ``INIT_RATIO`` and decays by 0.05 every 5
        tries, and any placement is accepted once it drops below 0.1 (NeuroVFM fallback).
        """
        G = self.grid
        fg_sum = int(valid_3d.sum().item())
        _ratio = INIT_RATIO
        attempts = 0
        while True:
            top = torch.randint(0, G - h + 1, (1,), generator=generator).item()
            left = torch.randint(0, G - w + 1, (1,), generator=generator).item()
            block = torch.zeros(G, G, G, dtype=torch.bool)
            block[0:d, top:top + h, left:left + w] = True
            covered = int((block & valid_3d).sum().item())
            if fg_sum == 0 or _ratio < 0.1 or covered >= fg_sum * _ratio:
                break
            attempts += 1
            if attempts >= 5:
                attempts = 0
                _ratio -= 0.05
        return block

    def _generate_one(self, valid_bool, valid_3d, n_valid, filt_ratio, way, generator):
        """One way of NeuroVFM ``_generate_masks``: blocks -> optional switch -> depth drop
        -> patch dropout -> ``(enc_idx, pred_idx)``.  Returns ``(None, None)`` when every
        attempt is rejected.
        """
        G = self.grid
        for _ in range(MAX_N_ATTEMPTS):
            # The context starts as the full grid; carve out npred full-depth HW blocks.
            ctx_3d = torch.ones(G, G, G, dtype=torch.bool)
            for _p in range(way['npred']):
                block = self._place_hw_block(
                    *self._sample_block_size(way, filt_ratio, generator),
                    valid_3d=valid_3d, generator=generator)
                ctx_3d = ctx_3d & ~block

            # switch_enc_pred: the block(s) become the context (NeuroVFM way 2).
            if way['switch_enc_pred']:
                ctx_3d = ~ctx_3d

            # Depth drop: shrink the context along the depth axis to about enc_mask_scale
            # of the foreground.
            ctx_fg = int((ctx_3d & valid_3d).sum().item())
            curr_frac = ctx_fg / max(1, n_valid)
            if curr_frac > way['enc_mask_scale']:
                d_drop_scale = 1.0 - (way['enc_mask_scale'] / curr_frac)
                n_dad = int(math.floor(d_drop_scale * G))
                if n_dad > 0:
                    offset = torch.randint(0, G - n_dad + 1, (1,), generator=generator).item()
                    ctx_3d[offset:offset + n_dad] = False

            # Patch dropout on the surviving context tokens (dropped tokens become target).
            ctx_flat = ctx_3d.flatten()
            if way['patch_drop_rate'] > 0:
                keep = torch.bernoulli(
                    torch.full((self.N,), 1.0 - way['patch_drop_rate']),
                    generator=generator).bool()
                ctx_flat = ctx_flat & keep

            # Confine to brain-valid tokens; target = foreground not in the context.
            enc_bool = ctx_flat & valid_bool
            pred_bool = (~ctx_flat) & valid_bool
            enc_idx = enc_bool.nonzero().squeeze(1)
            pred_idx = pred_bool.nonzero().squeeze(1)

            if enc_idx.numel() == 0 or pred_idx.numel() == 0 or n_valid == 0:
                continue
            frac = enc_idx.numel() / n_valid
            if frac < MIN_ENC_FRAC or frac > MAX_ENC_FRAC:
                continue
            return enc_idx, pred_idx
        return None, None

    def __call__(self, batch_size, token_valid_flat):
        """Per sample: draw one way (NeuroVFM ``cfg_idx``) and generate ``(context, target)``.

        Pads both index lists to the batch maximum and returns weight masks for both sides
        (1 = real, 0 = padded): ``(enc, pred, pred_w, enc_w)``.
        """
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        collated_masks_enc, collated_masks_pred = [], []
        for b in range(batch_size):
            valid_bool = token_valid_flat[b]  # [N] bool (brain tokens only)
            valid_3d = valid_bool.reshape(self.grid, self.grid, self.grid)
            n_valid = int(valid_bool.sum().item())
            filt_ratio = (n_valid / self.N) if self.N > 0 else 0.0

            way = self.ways[torch.randint(len(self.ways), (1,), generator=g).item()]
            enc_idx, pred_idx = self._generate_one(valid_bool, valid_3d, n_valid, filt_ratio, way, g)

            # Fallback (retries exhausted / empty side): deterministic enc_mask_scale split
            # of the valid tokens; a sample with fewer than two valid tokens is split over
            # all G^3 tokens, background included.
            if enc_idx is None or enc_idx.numel() < 1 or pred_idx.numel() < 1:
                valid_idx = valid_bool.nonzero().squeeze(1)
                if valid_idx.numel() < 2:
                    valid_idx = torch.arange(self.N)
                n = valid_idx.numel()
                n_ctx = int(round(n * way['enc_mask_scale']))
                n_ctx = min(max(1, n_ctx), n - 1)
                enc_idx, pred_idx = valid_idx[:n_ctx], valid_idx[n_ctx:]

            collated_masks_enc.append(enc_idx)
            collated_masks_pred.append(pred_idx)

        # Pad each sample's enc/pred to the batch maximum by cycling its own indices and
        # build the weight masks.
        max_enc = max(len(e) for e in collated_masks_enc)
        max_pred = max(len(p) for p in collated_masks_pred)

        def _pad(idx, k):
            if idx.numel() == 0:  # defensive: the fallback above guarantees >= 1 index
                idx = torch.zeros(1, dtype=torch.long)
            return idx[torch.arange(k) % len(idx)]

        enc = torch.stack([_pad(e, max_enc) for e in collated_masks_enc])
        pred = torch.stack([_pad(p, max_pred) for p in collated_masks_pred])
        pred_w = torch.zeros(batch_size, max_pred)
        enc_w = torch.zeros(batch_size, max_enc)
        for b, (e, p) in enumerate(zip(collated_masks_enc, collated_masks_pred)):
            enc_w[b, :len(e)] = 1.0
            pred_w[b, :len(p)] = 1.0
        return enc, pred, pred_w, enc_w


def apply_masks(x, masks):
    """Gather the tokens selected by each index mask (V-JEPA 2 ``apply_masks``).

    ``x [B, N, D]``; ``masks`` is a list of ``[B, K]`` long index tensors.  The gathered
    ``[B, K, D]`` tensors are concatenated along the batch axis.  Used for the context
    tokens (``apply_masks(tokens, [enc_idx])``) and for the target gather
    (``apply_masks(h, [pred_idx])``).
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).expand(-1, -1, x.size(-1))
        all_x.append(torch.gather(x, dim=1, index=mask_keep))
    return torch.cat(all_x, dim=0)
