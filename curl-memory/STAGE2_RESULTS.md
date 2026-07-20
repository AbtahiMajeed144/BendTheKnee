# Stage 2 Results — Calibration (E2)

**Date:** 2026-07-21 · **Hardware:** Kaggle Tesla T4 · **torch** 2.10.0+cu128
**Model:** `cfm_cifar10_weights_step_400000.pt` (I-CFM) · **d=3072** · null τ² = 0.49984
**Setup:** 256 P1 points (noised real data), v* over all 50 000 training points, 16 probes,
10 t-knots, seed 0.

---

## Headline

- **pooled τ² = 0.0073**, 95% CI **[0.0070, 0.0077]** — vs null 0.49984
- overall **ρ = 0.0385** (cross-checks Stage-1 G0: 0.02–0.06), relative Jacobian error ‖E‖/‖J*‖ = **0.39**
- `Spearman(tr Cov_w, per-point τ²) = 0.120` (confound present, weak)

τ² = fraction of the *exact* Jacobian error `E = ∇v_θ − ∇v*` that is rotational. Measured with
no ground-truth label — v* is closed-form on CIFAR (σ=0 makes it exact).

---

## Two robust, unconfounded conclusions

1. **The "architectural failure → rotational garbage" hypothesis is falsified.** If the network
   spun out into curl, τ² would exceed 0.5. It is ≤ 0.11 in *every* regime probed. The network
   does not spin out; its curl is small both absolutely (ρ≈0.04) and as a share of its error.
2. **The certificate is precision, not magnitude.** ‖A_θ‖ captures <1% of the Jacobian error
   vs the empirical minimizer. High curl certifies error; low curl certifies nothing.

---

## The generalization confound — quantified, and it does not overturn the result

`v*_emp` at a noised *training* point goes degenerate as t grows: weights collapse onto the
source image, `Cov_w → 0`, `∇v*_emp → −I/(1−t)`. The smooth generalizing network then shows real
symmetric strain that counts entirely as "error", inflating ‖E‖ and deflating τ². The data show
this directly — **tr Cov_w ≈ 0 for 7 of 10 t-knots** (t ≥ 0.36):

### τ² resolved in t
| t | τ² | rho | rel.err | tr Cov_w |
|---|---|---|---|---|
| 0.068 | 0.1106 | 0.0121 | **0.0363** | **5.73e+02** |
| 0.164 | 0.0085 | 0.0345 | 0.3556 | 2.22e+02 |
| 0.260 | 0.0430 | 0.0626 | 0.2976 | 1.03e+01 |
| 0.356 | 0.0522 | 0.0581 | 0.2514 | 3.51e-01 |
| 0.452–0.932 | 0.051 → 0.006 | ~0.05 → 0.04 | 0.25 → 0.42 | ≈ 0 (degenerate) |

### τ² resolved in tr Cov_w (quartiles)
| tr Cov_w bin | n | τ² | 95% CI | rho | rel.err |
|---|---|---|---|---|---|
| [−1.6e-4, −1.0e-5] | 634 | 0.0072 | [.0068,.0078] | 0.0378 | 0.385 |
| [−1.0e-5, 1.3e-5] | 644 | 0.0073 | [.0068,.0079] | 0.0387 | 0.392 |
| [1.3e-5, 5.9e-3] | 636 | 0.0074 | [.0069,.0081] | 0.0396 | 0.397 |
| [5.9e-3, 7.3e+2] | 640 | **0.0124** | [.0097,.0161] | 0.0323 | 0.281 |

**Reading:** τ² rises with target structure (0.007 → 0.012 across quartiles; Spearman +0.12),
confirming the confound exists. But **even at the highest, least-confounded structure it stays
far below 0.5** — the single cleanest point is t=0.068 (near-population target, rel.err=0.036,
Cov_w=573) at **τ²=0.11**. The pooled 0.007 is a deflated *underestimate*; the confound-corrected
value is ~0.1, still ≪ 0.5.

---

## Verdict

**The network is implicitly conservative: its Jacobian error is symmetric-dominated in every
regime probed (τ² ∈ [0.006, 0.11] ≪ 0.5, no rotational excess anywhere).** The pooled magnitude
is deflated by the empirical minimizer's degeneracy at large t, but correcting for that (small-t /
high-Cov_w points) leaves τ² ≈ 0.1 — still sub-isotropic. This is the "different, stronger paper"
outcome: not a curl-driven memorization/OOD detector, but a certified statement that trained flow
networks stay near gradient-field structure.

## Methodological finding (feeds E4)

P1 points noised from *single* training images do not populate genuine **basin boundaries**
(mid-density, high Cov_w): they are either spread-at-small-t or concentrated-at-large-t. 7/10
t-knots had a degenerate target. A clean E2/E4 needs a probe construction that hits mid-density
inter-cluster points, or a smoothed (KDE plug-in) target that is not degenerate.

Raw per-point data saved to `STAGE2_RESULTS_raw.npz` (A, E, Jt, Js, trcov, t) for re-analysis.
