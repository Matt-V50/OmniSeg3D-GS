"""
utils/omniseg_loss.py OmniSegLoss (semantic_only, patch_flag, cal_opt 3: the configuration train.py uses), vectorised
over the clusters (HHGS, 2026-11-12): the original loops over clusters and hierarchy levels in Python, ~2.4 s per step
for 16 slices; this computes the same quantities with one (pixels x clusters) matrix and a loop over the levels.
Checked equal to the original (loss and gradient) by hhgs_volume/check_fast.py.

Per cluster c (patches with > min_pixnum sampled pixels): prototype u_c (mean of its normalised features), phi_c =
clip(10 * sum |x - u_c| / (n log(n + 10)), .5, 1) (detached); dist[x, j] = exp(x . u_j / phi_j);
log[x, j] = -log(dist[x, j] / (sum_j' dist[x, j'] + 1e-6)). Levels of c: the distinct accpatch[c, j] over the clusters,
descending, the smallest dropped; S_l = clusters at level l. For a pixel x of c:
  l = 0   term = mean_{j in S_0} log[x, j],                     M_0 = max_{S_0} log[x, .]
  l >= 1  term = mean_{j in S_l} max(log[x, j], M_{l-1}),       M_l = max(M_{l-1}, max_{S_l} log[x, .])
ProtoNCE = sum_x sum_l term * 5^-l; returned as {'semantic_render': 1e-3 * ProtoNCE}.
"""
import torch


def OmniSegLoss(results, target, lambda_sam=1e-3, sam_level_weight=5, min_pixnum=20):
    valid = target['sam'] > 0
    sam_t = target['sam'][valid].long()
    sam_o = results['semantic'][valid.squeeze(), :]
    sam_o = sam_o / (torch.norm(sam_o, dim=-1, keepdim=True) + 1e-6).detach()

    ids, counts = torch.unique(sam_t, return_counts=True)
    keep = counts > min_pixnum
    ids, cn = ids[keep], counts[keep]
    C = ids.shape[0]
    if C == 0:
        return {'semantic_render': torch.zeros([1], device=sam_o.device)}
    lut = torch.full((int(sam_t.max()) + 1,), -1, dtype=torch.long, device=sam_t.device)
    lut[ids] = torch.arange(C, device=sam_t.device)
    c = lut[sam_t]
    inside = c >= 0                                       # pixels of the kept clusters (the original's loop)
    x, c = sam_o[inside], c[inside]

    u = torch.zeros(C, x.shape[1], device=x.device, dtype=x.dtype).index_add(0, c, x) / cn[:, None].to(x.dtype)
    dev = torch.norm(x - u[c], dim=1)
    s = torch.zeros(C, device=x.device, dtype=x.dtype).index_add(0, c, dev)
    phi = (s / (cn * torch.log(cn + 10))).detach()
    phi = torch.clip(phi * 10, min=0.5, max=1.0)

    dist = torch.exp(x @ u.T / phi[None, :])                                   # (N, C)
    log = -torch.log(dist / (dist.sum(dim=1, keepdim=True) + 1e-6))

    acc = target['accpatch'][ids][:, ids]                                      # (C, C)
    # level of every (cluster, cluster): the number of distinct values of the row greater than it (acc holds small
    # integers, shared mask counts); the row's smallest distinct value -> -1 (dropped, the original's [:-1])
    acc = acc.long()
    present = torch.zeros(C, int(acc.max()) + 2, dtype=torch.long, device=acc.device)
    present.scatter_(1, acc, 1)
    above = present.flip(1).cumsum(1).flip(1) - present                        # distinct values > v, per row
    lev = above.gather(1, acc)
    n_lev = present.sum(1)
    lev = torch.where(lev == (n_lev - 1)[:, None], torch.full_like(lev, -1), lev)
    L = int(n_lev.max()) - 1

    lev_x = lev[c]                                                             # (N, C)
    total = torch.zeros((), device=x.device, dtype=x.dtype)
    M = None
    for l in range(L):
        m = lev_x == l
        cnt = m.sum(1)
        has = cnt > 0
        if not has.any():
            continue
        if l == 0:
            val = torch.where(m, log, torch.zeros_like(log))
        else:
            val = torch.where(m, torch.maximum(log, M[:, None]), torch.zeros_like(log))
        term = val.sum(1) / cnt.clamp_min(1)
        total = total + (term[has]).sum() * sam_level_weight ** (-l)
        mx = torch.where(m, log, torch.full_like(log, -float('inf'))).max(1).values
        M = mx if M is None else torch.maximum(M, mx)
    return {'semantic_render': lambda_sam * total.reshape(1)}
