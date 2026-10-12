"""SimCLR pre-training of the shared 3D ViT encoder on brain MRI.

Two random-resized crops per volume, each augmented once, form the positive pair; every
other crop in the (all-gathered) batch is a negative.  The NT-Xent objective is applied to
a 2-layer projection head on top of the class token.  See :mod:`simclr.train`.
"""
