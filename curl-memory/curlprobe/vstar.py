"""Exact closed-form I-CFM empirical target field v* (sigma = 0).

For x_t = (1-t) x_0 + t x_1 with x_0 ~ N(0, I) independent of x_1 drawn uniformly from the
N training points {x_i}, the exact empirical minimizer is

    w_i(x,t)  proportional to  exp( -||x - t x_i||^2 / (2 (1-t)^2) )
    mu(x,t)   = sum_i w_i x_i
    v*(x,t)   = (mu - x) / (1 - t)
    grad v*   = 1/(1-t) ( t/(1-t)^2 * Cov_w - I )               (symmetric -> asym == 0)

`vstar_field` is a pure, autodiff-compatible function of x, so the estimators in
estimator.py can be run on it directly (E1c: they must read rho < 1e-5 at d=3072).
`vstar_jvp` gives the analytic Jacobian action grad v* @ eps without forming the d x d
Cov_w matrix (two matvecs against the data matrix) — used for the E2 tightness bound.

Numerics: weights via float64 log-sum-exp; distances chunked over the N data points so the
(P x N) logit matrix never blows memory. Restrict t to [0.02, 0.98] (the 1/(1-t) singularity).
"""

from __future__ import annotations

from typing import Tuple

import torch


def _pairwise_logits(
    x_flat: torch.Tensor,      # (P, d)  query points, flattened, in the model's [-1,1] space
    data_flat: torch.Tensor,   # (N, d)  training data, same space
    t: float,
    chunk: int = 4096,
) -> torch.Tensor:
    """logit_{p,i} = -||x_p - t x_i||^2 / (2 (1-t)^2), computed in float64, chunked over N."""
    inv = 1.0 / (2.0 * (1.0 - t) ** 2)
    x64 = x_flat.to(torch.float64)
    xsq = x64.pow(2).sum(dim=1, keepdim=True)             # (P,1)
    parts = []                                            # functional (no in-place) -> autodiff-safe
    for s in range(0, data_flat.shape[0], chunk):
        d64 = data_flat[s : s + chunk].to(torch.float64)  # (c, d)
        dsq = (t * t) * d64.pow(2).sum(dim=1)[None, :]     # (1,c)
        cross = (2.0 * t) * (x64 @ d64.t())                # (P,c)
        dist_sq = xsq + dsq - cross                        # ||x - t d||^2
        parts.append(-dist_sq * inv)
    return torch.cat(parts, dim=1)


def weights(x_flat: torch.Tensor, data_flat: torch.Tensor, t: float, chunk: int = 4096) -> torch.Tensor:
    """Posterior weights w (P, N), float64, numerically stable (softmax over N)."""
    logits = _pairwise_logits(x_flat, data_flat, t, chunk=chunk)
    return torch.softmax(logits, dim=1)


def vstar_field(
    x: torch.Tensor,           # (P, *dim) or (P, d)
    data_flat: torch.Tensor,   # (N, d)
    t: float,
    chunk: int = 4096,
) -> torch.Tensor:
    """Exact v*(x, t), returned in the same shape/dtype as x. Differentiable in x."""
    shape = x.shape
    x_flat = x.reshape(shape[0], -1)
    w = weights(x_flat, data_flat, t, chunk=chunk)               # (P,N) float64
    mu = (w @ data_flat.to(torch.float64))                       # (P,d) float64
    v = (mu - x_flat.to(torch.float64)) / (1.0 - t)
    return v.to(x.dtype).reshape(shape)


def vstar_jvp(
    x: torch.Tensor,           # (P, d) flattened
    eps: torch.Tensor,         # (P, d) probe(s)
    data_flat: torch.Tensor,   # (N, d)
    t: float,
    chunk: int = 4096,
) -> torch.Tensor:
    """Analytic grad v* @ eps = 1/(1-t) ( t/(1-t)^2 Cov_w eps - eps ), no d x d matrix.

    Cov_w eps = sum_i w_i (x_i - mu) <x_i - mu, eps>
              = (s @ X) - (sum_i s_i) mu,   s_i = w_i * (<x_i,eps> - <mu,eps>).
    """
    x64 = x.to(torch.float64)
    eps64 = eps.to(torch.float64)
    X = data_flat.to(torch.float64)                             # (N,d)
    w = weights(x, data_flat, t, chunk=chunk)                   # (P,N)
    mu = w @ X                                                  # (P,d)
    mu_eps = (mu * eps64).sum(dim=1, keepdim=True)              # (P,1)
    X_eps = eps64 @ X.t()                                       # (P,N) = <x_i, eps>
    c = X_eps - mu_eps                                          # (P,N)
    s = w * c                                                   # (P,N)
    cov_eps = s @ X - s.sum(dim=1, keepdim=True) * mu           # (P,d)
    jv = (t / (1.0 - t) ** 2 * cov_eps - eps64) / (1.0 - t)
    return jv.to(x.dtype)


def trace_cov_w(x: torch.Tensor, data_flat: torch.Tensor, t: float, chunk: int = 4096) -> torch.Tensor:
    """tr Cov_w(x,t) per query point (a difficulty coordinate for E4's mechanism test)."""
    x_flat = x.reshape(x.shape[0], -1)
    w = weights(x_flat, data_flat, t, chunk=chunk)             # (P,N)
    X = data_flat.to(torch.float64)
    mu = w @ X                                                 # (P,d)
    ex2 = w @ X.pow(2).sum(dim=1)                              # (P,)  E_w[||x_i||^2]
    return (ex2 - mu.pow(2).sum(dim=1)).to(x.dtype)            # E||x||^2 - ||mu||^2
