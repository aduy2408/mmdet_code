# Dynamic Background Subspace Suppression

DBSS is a project-local FCOS ablation for LEVIR-Ship. It builds diverse,
image-conditioned background bases from valid P3 regions, projects the P3
embedding onto those bases, and applies a bounded residual displacement.

The main config is `configs/fcos_dbss_ridge.py`. Softmax prototype mixture and
Haar reliability are isolated in their own ablation configs.
