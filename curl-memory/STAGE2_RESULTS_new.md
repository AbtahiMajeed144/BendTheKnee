# Stage 2 — Calibration (E2): Results & Record

*(Curated human record. The script also auto-writes `STAGE2_RESULTS.md` + `STAGE2_RESULTS_raw.npz`
on each run; this file is the interpreted canonical version.)*

**Date:** 2026-07-21 · **Hardware:** Kaggle Tesla T4 (15.6 GB) · **torch** 2.10.0+cu128
**Model:** `cfm_cifar10_weights_step_400000.pt` — I-CFM (independent coupling, the certificate track)

---

## What the experiment does

E2 asks: *of the trained network's Jacobian error, how much is rotational (curl)?*

On CIFAR with σ=0 the exact I-CFM target field is **closed-form**:
`v*=(μ−x)/(1−t)` with softmax weights `w_i ∝ exp(−‖x−t·x_i‖²/2(1−t)²)` over all 50 000 training
images. So we can compute the true Jacobian error `E = ∇v_θ − ∇v*` and measure

    τ² = ‖A_θ‖²_F / ‖E‖²_F          (A_θ = asym(∇v_θ) = the curl; asym(∇v*) ≡ 0 exactly)

i.e. the **antisymmetric fraction of the Jacobian error**. Null (an error matrix with i.i.d.
entries) is `(d−1)/2d = 0.49984` at d=3072. Estimated via shared Hutchinson probes: `J_θ ε` (JVP),
`J_θᵀ ε` (VJP), and `J* ε` (analytic, closed-form). Evaluated on **P1** = noised real data
`x_t=(1−t)ε+t·x_1`, resolved in `t` and in `tr Cov_w`, with bootstrap CIs.

**Setup:** 256 P1 points, v* over all 50 000 training points, 16 probes, 10 t-knots (t∈[0.02,0.98]),
seed 0. Local validation beforehand: `calibration_estimates` recovers a known τ² to 0.06%.

---

## Results

**Headline**
| quantity | value |
|---|---|
| pooled τ² | **0.0073**  (95% CI [0.0070, 0.0077]) |
| null τ² | 0.49984 |
| overall ρ = ‖A_θ‖/‖∇v_θ‖ | **0.0385**  (cross-checks Stage-1 G0: 0.02–0.06) |
| relative Jacobian error ‖E‖/‖J*‖ | 0.39 |
| Spearman(tr Cov_w, per-point τ²) | 0.120 |

**Resolved in t**
| t | τ² | ρ | rel.err | tr Cov_w |
|---|---|---|---|---|
| 0.068 | 0.1106 | 0.0121 | 0.0363 | 5.73e+02 |
| 0.164 | 0.0085 | 0.0345 | 0.3556 | 2.22e+02 |
| 0.260 | 0.0430 | 0.0626 | 0.2976 | 1.03e+01 |
| 0.356 | 0.0522 | 0.0581 | 0.2514 | 3.51e-01 |
| 0.452 | 0.0512 | 0.0566 | 0.2452 | ≈0 |
| 0.548 | 0.0376 | 0.0493 | 0.2465 | ≈0 |
| 0.644 | 0.0238 | 0.0405 | 0.2516 | ≈0 |
| 0.740 | 0.0161 | 0.0367 | 0.2726 | ≈0 |
| 0.836 | 0.0104 | 0.0353 | 0.3157 | ≈0 |
| 0.932 | 0.0058 | 0.0378 | 0.4186 | ≈0 |

**Resolved in tr Cov_w (quartiles) — the confound diagnostic**
| tr Cov_w bin | n | τ² | 95% CI | ρ | rel.err |
|---|---|---|---|---|---|
| [−1.6e-4, −1.0e-5] | 634 | 0.0072 | [.0068,.0078] | 0.0378 | 0.385 |
| [−1.0e-5, 1.3e-5] | 644 | 0.0073 | [.0068,.0079] | 0.0387 | 0.392 |
| [1.3e-5, 5.9e-3] | 636 | 0.0074 | [.0069,.0081] | 0.0396 | 0.397 |
| [5.9e-3, 7.3e+2] | 640 | **0.0124** | [.0097,.0161] | 0.0323 | 0.281 |

---

## Interpretation

**Two robust, unconfounded conclusions**
1. **"Architectural failure → rotational garbage" is FALSIFIED.** If the network spun out into
   curl, τ² would exceed 0.5. It is ≤ 0.11 in every regime. The network's curl is small both
   absolutely (ρ≈0.04) and as a fraction of its error.
2. **The certificate is precision, not magnitude.** ‖A_θ‖ captures <1% of the Jacobian error vs
   the empirical minimizer — high curl certifies error, low curl certifies nothing.

**The generalization confound — quantified, does not overturn the result.** `v*_emp` at a noised
*training* point becomes degenerate as t grows (weights collapse onto the source image,
`Cov_w→0`, `∇v*→−I/(1−t)`); the smooth generalizing network's real symmetric strain then counts
as "error", deflating τ². The data confirm it: **tr Cov_w ≈ 0 for 7 of 10 t-knots**, and τ² rises
with target structure (0.007→0.012 across quartiles, Spearman +0.12). But even at the highest,
least-confounded structure it stays ≪ 0.5 — the cleanest point, t=0.068 (near-population target,
rel.err=0.036, Cov_w=573), gives **τ²=0.11**. The pooled 0.007 is a deflated underestimate; the
confound-corrected value is ~0.1, still sub-isotropic.

**Verdict.** The trained I-CFM network is **implicitly conservative**: its Jacobian error is
symmetric-dominated in every regime probed (τ² ∈ [0.006, 0.11] ≪ 0.5, no rotational excess). This
is the "different, stronger paper" branch pre-registered in `seed_experiment1.md` — a certified
statement that trained flow networks stay near gradient-field structure, rather than a curl-driven
memorization/OOD detector.

---

## Methodological finding (carries into E4)

P1 points noised from *single* training images do not populate genuine **basin boundaries**
(mid-density, high Cov_w): they are spread-at-small-t or concentrated-at-large-t, so 7/10 t-knots
had a degenerate target. A clean E2/E4 needs a probe construction that hits mid-density
inter-cluster points, or a smoothed (KDE plug-in) target that is not degenerate.

## Reproduce

```bash
python stage2_calibration.py            # cfm_ checkpoint, 50k v*, 256 P1 pts, 16 probes, 10 knots
```
Raw per-point arrays (A, E, Jt, Js, trcov, t) are in `STAGE2_RESULTS_raw.npz`.
