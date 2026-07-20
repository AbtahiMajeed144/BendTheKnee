"""Probe-point construction.

Two evaluation sets that must NEVER be mixed (see execution_plan.md sec 3):

  P1 : noised real data, x_t = (1 - t) * eps + t * x_1,  eps ~ N(0, I).
       This is p_t, where the CFM loss lives and where v* is the exact minimizer.
       Used for E1 / E2 / E3 / E6 — every experiment that compares against v*.

  P2 : points sampled along generated trajectories (Euler / dopri5 from noise).
       Used ONLY for E4 / E5. Implemented in Stage 4; a guarded stub here so the
       API exists and misuse is loud.

t is drawn stratified on [0.02, 0.98] (the 1/(1-t) singularity is excluded).
"""

from __future__ import annotations

from typing import Tuple

import torch

T_LO = 0.02
T_HI = 0.98


def stratified_t(n_knots: int, lo: float = T_LO, hi: float = T_HI, jitter: bool = False,
                 generator: torch.Generator | None = None, device=None) -> torch.Tensor:
    """n_knots values on [lo, hi]. Bin centres by default; one uniform draw per bin if jitter."""
    edges = torch.linspace(lo, hi, n_knots + 1, device=device)
    if jitter:
        u = torch.rand(n_knots, generator=generator, device=device)
        return edges[:-1] + u * (edges[1:] - edges[:-1])
    return 0.5 * (edges[:-1] + edges[1:])


def p1_noised(
    x1: torch.Tensor,          # (B, *dim) real data in [-1, 1]
    t: float | torch.Tensor,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """x_t = (1 - t) eps + t x1. Returns (x_t, eps). t scalar or per-sample (B,)."""
    eps = torch.randn(x1.shape, dtype=x1.dtype, device=x1.device, generator=generator)
    if torch.is_tensor(t):
        tt = t.reshape((-1,) + (1,) * (x1.dim() - 1)).to(x1.dtype)
    else:
        tt = float(t)
    x_t = (1.0 - tt) * eps + tt * x1
    return x_t, eps


def basin_target(imgs, n, k=2, alpha=0.5, labels=None, different_class=True, generator=None):
    """Clean targets that force a NON-degenerate v* (E2b: genuine basin boundaries).

    Single-image targets make v* collapse to -I/(1-t) (Cov_w -> 0) at large t. Convex
    combinations of k *distinct* training images sit between modes, so the posterior over
    training points stays multi-modal and Cov_w is large -- the regime E2 never entered.

      k == 2 : interpolation  (1-alpha) x_a + alpha x_b   (alpha=0.5 -> the midpoint / boundary)
      k  > 2 : uniform mean of k images.

    If `labels` is given and different_class=True, the k images come from distinct classes
    (distant clusters -> steeper weight transition). Returns m: (n, *dim).
    """
    dev = imgs.device
    N = imgs.shape[0]
    picks = torch.empty(n, k, dtype=torch.long, device=dev)
    if labels is not None and different_class:
        classes = labels.unique()
        for j in range(n):
            cls = classes[torch.randperm(len(classes), generator=generator, device=dev)[:k]]
            for a in range(k):
                pool = (labels == cls[a]).nonzero(as_tuple=True)[0]
                picks[j, a] = pool[torch.randint(len(pool), (1,), generator=generator, device=dev)]
    else:
        for j in range(n):
            picks[j] = torch.randperm(N, generator=generator, device=dev)[:k]
    grp = imgs[picks]                                    # (n, k, *dim)
    if k == 2:
        return (1.0 - alpha) * grp[:, 0] + alpha * grp[:, 1]
    return grp.mean(dim=1)


def p2_trajectory(*args, **kwargs):  # pragma: no cover - Stage 4
    raise NotImplementedError(
        "P2 (generated-trajectory points) is Stage 4 (E4/E5). Do not use it for any "
        "experiment that compares against v* — that comparison is only valid on P1."
    )
