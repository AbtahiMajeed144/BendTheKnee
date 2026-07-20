"""curlprobe — shared library for the curl-memory certificate program.

Primitives (build once, every experiment imports these):
  estimator    : |A|_F^2 and |grad v|_F^2 via JVP+VJP (primary) or AD-free FD (fallback)
  vstar        : exact closed-form I-CFM empirical target v* and its Jacobian action
  probe_points : P1 (noised real data) vs P2 (generated trajectories) — never mixed
  score        : scale-free rho = |A|_F / |grad v|_F and the path score
  models       : load the pretrained torchcfm UNet (cfm_ = I-CFM certificate track)

See ../execution_plan.md — this code is written against that contract.
"""

from . import estimator, vstar, probe_points, score, models  # noqa: F401

__all__ = ["estimator", "vstar", "probe_points", "score", "models"]
