"""Scores derived from the Frobenius estimates.

rho = ||A||_F / ||grad v||_F  in [0, 1] is the headline quantity: it is scale-free (cancels
the 1/(1-t) blow-up to leading order) and is a certified lower bound on *relative* Jacobian
error. Always report the t-resolved profile alongside any scalar path score.
"""

from __future__ import annotations

from typing import Tuple

import torch


def rho_from_sq(A_fro_sq: torch.Tensor, gradv_fro_sq: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    """rho = sqrt(||A||_F^2 / ||grad v||_F^2) per sample, guarded against 0/0."""
    return torch.sqrt(A_fro_sq.clamp_min(0.0) / gradv_fro_sq.clamp_min(eps))


def path_score(
    t_knots: torch.Tensor,     # (K,) sorted
    rho_tk: torch.Tensor,      # (K, B) rho at each knot for each sample
    log_snr_weight: bool = False,
) -> torch.Tensor:
    """bar_rho(x0) = sqrt( int rho^2 w(t) dt ), trapezoid over the knots. Returns (B,).

    Default weight is uniform in t. With log_snr_weight, weight by d(log SNR)/dt for the
    linear path (SNR = (t/(1-t))^2 -> weight proportional to 1/(t(1-t))).
    """
    t = t_knots.to(rho_tk.dtype)
    r2 = rho_tk.pow(2)                                  # (K,B)
    if log_snr_weight:
        w = 1.0 / (t * (1.0 - t))
        w = w / torch.trapz(w, t)                       # normalize
        integ = torch.trapz(r2 * w[:, None], t, dim=0)  # (B,)
    else:
        integ = torch.trapz(r2, t, dim=0) / (t[-1] - t[0])
    return torch.sqrt(integ.clamp_min(0.0))


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation between two 1-D tensors (for the E1e cost-fidelity gate)."""
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = ra.norm() * rb.norm()
    if denom == 0:
        return float("nan")
    return float((ra @ rb) / denom)
