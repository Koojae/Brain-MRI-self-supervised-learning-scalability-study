"""JEPA pre-training of the shared 3D ViT on whole-brain 96^3 volumes.

A context encoder sees a subset of brain tokens, a predictor regresses the latent
representation of the remaining brain tokens, and the regression target comes from an
EMA copy of the encoder.  The EMA (target) encoder is the transferable output.  See
``jepa/model.py`` for the objective, ``jepa/masks.py`` for the two-way block masking and
``jepa/train.py`` for the training loop.
"""
