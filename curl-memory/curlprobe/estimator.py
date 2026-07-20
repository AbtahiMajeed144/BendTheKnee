"""Vorticity / Jacobian-norm estimators.

Two estimators of the *per-sample* quantities

    A_fro_sq   = ||A||_F^2      with  A = asym(J) = 1/2 (J - J^T),   J = grad_x v
    gradv_fro_sq = ||J||_F^2

from which score.py forms  rho = ||A||_F / ||J||_F  in [0, 1].

Both are unbiased Hutchinson-style estimators using random probes eps with E[eps eps^T] = I
(standard normal). Identities used:

    E_eps || J eps ||^2                       = ||J||_F^2
    E_eps || 1/2 (J - J^T) eps ||^2           = ||A||_F^2        (primary, JVP+VJP)
    1/4 E_{eps,eta} ( eps^T J eta - eta^T J eps )^2 = ||A||_F^2   (fallback, AD-free)

The primary form needs the transpose action J^T eps (a VJP), so it needs reverse-mode AD.
The fallback needs only forward evaluations of v (4 per pair via central differences), so it
works on compiled / quantized / black-box models and sidesteps any missing forward-mode
(jvp) rule for attention ops. E1a picks which one we trust on this model (gate G1a').
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

import torch


def _flat_sum_sq(z: torch.Tensor) -> torch.Tensor:
    """Sum of squares over all non-batch dims -> shape (B,)."""
    return z.reshape(z.shape[0], -1).pow(2).sum(dim=1)


def _flat_inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-sample inner product over all non-batch dims -> shape (B,)."""
    return (a.reshape(a.shape[0], -1) * b.reshape(b.shape[0], -1)).sum(dim=1)


def default_fd_step(dtype: torch.dtype) -> float:
    """Central-difference step ~ eps_mach^(1/3), balancing truncation vs roundoff."""
    eps = torch.finfo(dtype).eps
    return float(eps ** (1.0 / 3.0))


# --------------------------------------------------------------------------------------
# Primary estimator: JVP + VJP  (needs reverse-mode AD; forward-mode via torch.func.jvp)
# --------------------------------------------------------------------------------------
def frobenius_jvp(
    vf: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    n_probes: int = 8,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """||A||_F^2 and ||J||_F^2 per sample via torch.func.jvp (J eps) and vjp (J^T eps).

    vf: pure function mapping x -> v(x) with x.shape == v.shape == (B, *dim).
    Returns (A_fro_sq, gradv_fro_sq), each shape (B,).
    """
    B = x.shape[0]
    A_sq = x.new_zeros(B)
    G_sq = x.new_zeros(B)
    for _ in range(n_probes):
        eps = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
        # J eps  (forward-mode)
        _, jvp_out = torch.func.jvp(vf, (x,), (eps,))
        # J^T eps (reverse-mode)
        _, vjp_fn = torch.func.vjp(vf, x)
        vjp_out = vjp_fn(eps)[0]
        a_eps = 0.5 * (jvp_out - vjp_out)
        A_sq = A_sq + _flat_sum_sq(a_eps)
        G_sq = G_sq + _flat_sum_sq(jvp_out)
    return A_sq / n_probes, G_sq / n_probes


# --------------------------------------------------------------------------------------
# Fallback estimator: 4 forward passes per pair (AD-free, central differences)
# --------------------------------------------------------------------------------------
def frobenius_fd(
    vf: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    n_probes: int = 8,
    h: float | None = None,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """||A||_F^2 and ||J||_F^2 per sample using only forward evaluations of vf.

    For each pair (eps, eta): J eta and J eps via central differences (4 forwards), then
        eps^T J eta - eta^T J eps  is the antisymmetric bilinear form.
    Returns (A_fro_sq, gradv_fro_sq), each shape (B,). Higher variance than the JVP form.
    """
    if h is None:
        h = default_fd_step(x.dtype)
    B = x.shape[0]
    A_sq = x.new_zeros(B)
    G_sq = x.new_zeros(B)
    with torch.no_grad():
        for _ in range(n_probes):
            eps = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
            eta = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
            j_eta = (vf(x + h * eta) - vf(x - h * eta)) / (2.0 * h)  # J eta
            j_eps = (vf(x + h * eps) - vf(x - h * eps)) / (2.0 * h)  # J eps
            eps_J_eta = _flat_inner(eps, j_eta)
            eta_J_eps = _flat_inner(eta, j_eps)
            A_sq = A_sq + 0.25 * (eps_J_eta - eta_J_eps) ** 2
            G_sq = G_sq + _flat_sum_sq(j_eps)
    return A_sq / n_probes, G_sq / n_probes


def frobenius_estimates(
    vf: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    n_probes: int = 8,
    method: str = "jvp",
    h: float | None = None,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dispatch to the chosen estimator. method in {'jvp', 'fd'}."""
    if method == "jvp":
        return frobenius_jvp(vf, x, n_probes=n_probes, generator=generator)
    if method == "fd":
        return frobenius_fd(vf, x, n_probes=n_probes, h=h, generator=generator)
    raise ValueError(f"unknown method {method!r}; use 'jvp' or 'fd'")


# --------------------------------------------------------------------------------------
# Exact reference (small dimension only) — for the E1a agreement test
# --------------------------------------------------------------------------------------
def frobenius_exact_smalldim(
    vf_single: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact ||A||_F^2 and ||J||_F^2 per sample from the full Jacobian (d-dim, small d).

    vf_single: maps a single vector (d,) -> (d,).  x: (B, d).
    Returns (A_fro_sq, gradv_fro_sq), each (B,).  Uses functorch jacrev + vmap.
    """
    jac = torch.func.vmap(torch.func.jacrev(vf_single))(x)  # (B, d, d)
    jt = jac.transpose(-1, -2)
    A = 0.5 * (jac - jt)
    A_sq = A.pow(2).sum(dim=(-1, -2))
    G_sq = jac.pow(2).sum(dim=(-1, -2))
    return A_sq, G_sq


# --------------------------------------------------------------------------------------
# Rotation injection — for the E1b positive control
# --------------------------------------------------------------------------------------
def make_rotation_operator(
    dim: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], float]:
    """A fixed antisymmetric linear operator R(x) with known ||R||_F = 1.

    R = (a b^T - b a^T) / sqrt(2) with a, b orthonormal in flattened space, so
        ||R||_F^2 = 2 (||a||^2 ||b||^2 - <a,b>^2) / 2 = 1.
    Since R is antisymmetric, asym(J_{v + amp*R}) = asym(J_v) + amp * R.
    Returns (R_op, R_fro) with R_fro == 1.0.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    d = int(math.prod(dim))
    a = torch.randn(d, generator=g, device=device, dtype=dtype)
    b = torch.randn(d, generator=g, device=device, dtype=dtype)
    a = a / a.norm()
    b = b - (a @ b) * a          # orthogonalize
    b = b / b.norm()
    scale = 1.0 / math.sqrt(2.0)  # -> ||R||_F = 1

    def R_op(x: torch.Tensor) -> torch.Tensor:
        xf = x.reshape(x.shape[0], -1)
        rx = (xf @ b)[:, None] * a[None, :] - (xf @ a)[:, None] * b[None, :]
        return (scale * rx).reshape(x.shape)

    return R_op, 1.0
