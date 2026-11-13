"""
OmniSeg3D on volume slices (HHGS, 2026-11-12): the training targets of one image, built from one slice's mask tree
instead of run_sam.py's per-image SAM masks, and OmniSeg3D's own loss (utils/omniseg_loss.py, unchanged).

run_sam.py, per image: the SAM masks, small ones absorbed by a larger one they lie 90 % inside (nesting), then
  patches  = the distinct mask-membership vectors of the pixels (index 0 = the all-zero vector, no mask, when present)
  accpatch = patch x patch number of masks containing both (unique.T @ unique)
Here a slice comes as a tree (HHGS training/labels.py: leaf (A*B,) = deepest mask of each pixel, -1 none; parent (N,)),
nested already, so a pixel's membership vector is the ancestor chain of its deepest mask:
  patch of a pixel = its deepest mask + 1 (0 = no mask, always index 0, so the loss's `sam > 0` drops exactly the
                     uncovered pixels)
  accpatch[p, q]   = |ancestors(p) & ancestors(q)| (each including itself), row / column 0 all zero
The volume renderer (HHGS GS field) replaces the rasteriser; everything after the rendered per-pixel features is
OmniSeg3D's: OmniSegLoss on the sampled pixels (features L2-normalised inside it; ProtoNCE with the patch hierarchy).
"""
import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("omniseg_loss", Path(__file__).resolve().parents[1] / "utils" /
                                               "omniseg_loss.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
OmniSegLoss = _mod.OmniSegLoss                      # the original loss, imported by path (no package clash)


def slice_patches(leaf, parent):
    """one slice's tree -> patch id per pixel (A*B,) int64 (0 = no mask), accpatch (N + 1, N + 1) uint8"""
    n = len(parent)
    anc = np.zeros((n + 1, n), np.uint8)            # row p + 1: masks containing patch p (its ancestor chain)
    for i in range(n):
        j = i
        while j >= 0:
            anc[i + 1, j] = 1
            j = int(parent[j])
    acc = anc.astype(np.int32) @ anc.T.astype(np.int32)
    return np.asarray(leaf, np.int64) + 1, np.minimum(acc, 255).astype(np.uint8)
