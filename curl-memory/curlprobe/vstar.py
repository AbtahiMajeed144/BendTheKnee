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
Cov_w matrix — used for the E2 tightness/calibration.

Scale: every matvec against the data is chunked over N and the float64 data matrix is never
materialized in full, so this runs against the full 50k CIFAR training set on a 16 GB GPU.
For E2 the weights depend only on (x, t) not on the probe, so `prepare` computes (w, mu)
once per point and `jvp_prepared` reuses them across all probes.

Numerics: weights via float64 log-sum-exp; matvecs in float64; t restricted to [0.02, 0.98].
"""

from __future__ import annotations

from typing import Tuple

import torch


# ----------------------------------------------------------------------------------------
# weights
# ----------------------------------------------------------------------------------------
def _pairwise_logits(x_flat, data_flat, t, chunk=8192):
    """logit_{p,i} = -||x_p - t x_i||^2 / (2 (1-t)^2), float64, chunked over N (no in-place)."""
    inv = 1.0 / (2.0 * (1.0 - t) ** 2)
    x64 = x_flat.to(torch.float64)
    xsq = x64.pow(2).sum(dim=1, keepdim=True)             # (P,1)
    parts = []
    for s in range(0, data_flat.shape[0], chunk):
        d64 = data_flat[s : s + chunk].to(torch.float64)  # (c,d)
        dsq = (t * t) * d64.pow(2).sum(dim=1)[None, :]     # (1,c)
        cross = (2.0 * t) * (x64 @ d64.t())                # (P,c)
        parts.append(-(xsq + dsq - cross) * inv)
    return torch.cat(parts, dim=1)


def weights(x_flat, data_flat, t, chunk=8192) -> torch.Tensor:
    """Posterior weights w (P, N), float64, numerically stable (softmax over N)."""
    return torch.softmax(_pairwise_logits(x_flat, data_flat, t, chunk=chunk), dim=1)


def _wX(w, data_flat, chunk=8192):
    """mu = w @ X in float64, chunked (never materializes the full float64 X)."""
    mu = torch.zeros(w.shape[0], data_flat.shape[1], dtype=torch.float64, device=w.device)
    for s in range(0, data_flat.shape[0], chunk):
        mu = mu + w[:, s : s + chunk] @ data_flat[s : s + chunk].to(torch.float64)
    return mu


# ----------------------------------------------------------------------------------------
# field + Jacobian action (standalone; recompute weights each call)
# ----------------------------------------------------------------------------------------
def vstar_field(x, data_flat, t, chunk=8192) -> torch.Tensor:
    """Exact v*(x, t), same shape/dtype as x. Differentiable in x."""
    shape = x.shape
    x_flat = x.reshape(shape[0], -1)
    w = weights(x_flat, data_flat, t, chunk=chunk)
    mu = _wX(w, data_flat, chunk=chunk)
    v = (mu - x_flat.to(torch.float64)) / (1.0 - t)
    return v.to(x.dtype).reshape(shape)


def vstar_jvp(x_flat, eps_flat, data_flat, t, chunk=8192) -> torch.Tensor:
    """Analytic grad v* @ eps for flattened inputs (recomputes weights)."""
    w = weights(x_flat, data_flat, t, chunk=chunk)
    mu = _wX(w, data_flat, chunk=chunk)
    return jvp_prepared(w, mu, eps_flat, data_flat, t, chunk=chunk).to(eps_flat.dtype)


# ----------------------------------------------------------------------------------------
# prepared path (E2): weights computed once per point, reused across probes
# ----------------------------------------------------------------------------------------
def prepare(x_flat, data_flat, t, chunk=8192) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (w, mu) for a batch of points — the per-point state that probes reuse."""
    w = weights(x_flat, data_flat, t, chunk=chunk)
    mu = _wX(w, data_flat, chunk=chunk)
    return w, mu


def jvp_prepared(w, mu, eps_flat, data_flat, t, chunk=8192) -> torch.Tensor:
    """grad v* @ eps = 1/(1-t) ( t/(1-t)^2 Cov_w eps - eps ), chunked, from prepared (w, mu).

    Cov_w eps = sum_i w_i (x_i - mu) <x_i - mu, eps>
              = (sum_i s_i x_i) - (sum_i s_i) mu,   s_i = w_i (<x_i,eps> - <mu,eps>).
    """
    eps64 = eps_flat.to(torch.float64)
    mu_eps = (mu * eps64).sum(dim=1, keepdim=True)                 # (P,1)
    cov_eps = torch.zeros_like(mu)                                 # (P,d)
    s_sum = torch.zeros(w.shape[0], 1, dtype=torch.float64, device=w.device)
    for a in range(0, data_flat.shape[0], chunk):
        Xc = data_flat[a : a + chunk].to(torch.float64)           # (c,d)
        Xeps = eps64 @ Xc.t()                                     # (P,c) = <x_i, eps>
        sc = w[:, a : a + chunk] * (Xeps - mu_eps)                # (P,c)
        cov_eps = cov_eps + sc @ Xc                               # (P,d)
        s_sum = s_sum + sc.sum(dim=1, keepdim=True)
    cov_eps = cov_eps - s_sum * mu
    jv = (t / (1.0 - t) ** 2 * cov_eps - eps64) / (1.0 - t)
    return jv


def trace_cov_from_prepared(w, mu, data_sqnorm) -> torch.Tensor:
    """tr Cov_w = E_w[||x_i||^2] - ||mu||^2, using a precomputed per-row ||x_i||^2 (N,)."""
    ex2 = w @ data_sqnorm.to(torch.float64)                       # (P,)
    return ex2 - mu.pow(2).sum(dim=1)


def trace_cov_w(x, data_flat, t, chunk=8192) -> torch.Tensor:
    """tr Cov_w(x,t) per point — standalone convenience (recomputes weights)."""
    x_flat = x.reshape(x.shape[0], -1)
    w = weights(x_flat, data_flat, t, chunk=chunk)
    mu = _wX(w, data_flat, chunk=chunk)
    sqn = data_flat.pow(2).sum(dim=1)
    return trace_cov_from_prepared(w, mu, sqn)
