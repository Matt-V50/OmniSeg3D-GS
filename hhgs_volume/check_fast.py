"""
Check hhgs_volume/omniseg_loss_fast.py against utils/omniseg_loss.py (the original): same loss and same gradient
w.r.t. the features, on slices of real HHGS cases (random features, 32768 random pixels per slice).

usage:  python hhgs_volume/check_fast.py <hhgs repo> <case> <labels> [n slices]      (in the hhgs environment)
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from omniseg_slices import OmniSegLoss as original, slice_patches  # noqa: E402
from omniseg_loss_fast import OmniSegLoss as fast  # noqa: E402


def main(repo, case, labels, n=20):
    sys.path.insert(0, repo)
    from utils.case import Case
    _, _, _, slices = Case(case).slices(labels, "base")
    rng = np.random.default_rng(0)
    slices = [s for s in slices if (s[3] >= 0).any()]
    worst, t_o, t_f = 0.0, 0.0, 0.0
    for a, k, ab, leaf, parent in [slices[i] for i in rng.choice(len(slices), min(n, len(slices)), replace=False)]:
        patch, acc = slice_patches(leaf, parent)
        pix = torch.as_tensor(rng.integers(0, ab[0] * ab[1], 32768))
        tgt = {"sam": torch.as_tensor(patch)[pix].cuda(), "accpatch": torch.as_tensor(acc).cuda().long()}
        f0 = torch.randn(32768, 16, device="cuda")
        out = []
        for fn in (original, fast):
            f = f0.clone().requires_grad_(True)
            torch.cuda.synchronize(); t = time.time()
            l = sum(v.sum() for v in fn({"semantic": f}, tgt).values())
            if l.requires_grad:                                         # no hierarchy level: a constant 0
                l.backward()
            torch.cuda.synchronize()
            out.append((float(l), f.grad.clone() if f.grad is not None else torch.zeros_like(f), time.time() - t))
        (lo, go, to), (lf, gf, tf) = out
        rel = abs(lo - lf) / max(abs(lo), 1e-12)
        grel = float((go - gf).norm() / go.norm().clamp_min(1e-12))
        worst = max(worst, rel, grel)
        t_o, t_f = t_o + to, t_f + tf
        print(f"slice {a}/{k}: patches {len(parent) + 1}  loss {lo:.6f} / {lf:.6f}  rel {rel:.1e}  grad rel {grel:.1e}  "
              f"[{to * 1000:.0f} / {tf * 1000:.0f} ms]", flush=True)
    print(f"worst relative difference {worst:.1e}; time original {t_o:.1f} s, fast {t_f:.1f} s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], *(int(x) for x in sys.argv[4:5]))
