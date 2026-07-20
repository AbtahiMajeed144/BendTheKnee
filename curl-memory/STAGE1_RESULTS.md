# Stage 1 Results — Instrument Validation (PASSED)

**Date:** 2026-07-21 · **Hardware:** Kaggle Tesla T4 (15.6 GB) · **torch** 2.10.0+cu128
**Model:** `cfm_cifar10_weights_step_400000.pt` (I-CFM, independent coupling — the certificate track)
**Verdict:** ✅ all three hard gates pass → **PROCEED to Stage 2 (E2 calibration)**

---

## What Stage 1 tested

Stage 1 answers one question: *does our instrument actually measure the vorticity
`||A||_F = ||asym(∇v)||_F` of a velocity field, with known power and a known floor?* It is
pure inference/matvecs — no training. Six checks, in dependency order:

| ID | Experiment | What it establishes |
|---|---|---|
| E1a | FD & JVP+VJP estimators vs the **exact** autodiff Jacobian on a small MLP | the estimator is unbiased; **selects which estimator to trust** (gate G1a′) |
| G0 | raw `ρ = ||A||_F/||∇v||_F` on the pretrained model, 3 timesteps, 100 noised real images | **is there any curl to measure?** (if ~0 the program is dead) |
| E1b | inject a **known** rotation `ε·R` into the model and sweep ε over 8 decades | can the instrument **recover a known antisymmetric signal** at the right magnitude? |
| E1c | run the estimator on the **exact closed-form** irrotational field `v*` at d=3072 | does the instrument correctly **read ≈0** on a field that has zero curl by construction? |
| E1d | sweep finite-difference step `h` and dtype on `v*` | the instrument's **numerical floor** as a concrete number |
| E1e | rank P1 points by path-score under cheap vs dense probe budgets | how cheap can certification be (Spearman vs a dense reference)? |

`ρ` is the scale-free score: `ρ = ||A||_F / ||∇v||_F ∈ [0,1]`, a certified lower bound on
*relative* Jacobian error. `v*` is the exact I-CFM empirical target
`v*=(μ−x)/(1−t)` with softmax weights over the CIFAR training set (σ=0, so it is exact).

---

## Results

### E1a — Exactness → estimator selection ✅
```
exact   mean |A|^2 = 1.6983e+02   mean |J|^2 = 3.9390e+02
JVP+VJP rel.err     |A|^2 = 0.11%    |J|^2 = 0.25%
FD (4 fwd) rel.err  |A|^2 = 0.46%    |J|^2 = 0.01%
```
Both estimators agree with the exact Jacobian to well under 1%. **Selected: `jvp`**
(JVP+VJP); the AD-free FD form is the validated black-box fallback.

### G0 — Hour-zero (HARD GATE) ✅
```
t=0.10   mean ρ = 1.82e-02   (median 1.56e-02)
t=0.30   mean ρ = 5.65e-02   (median 5.15e-02)
t=0.50   mean ρ = 4.76e-02   (median 4.42e-02)
peak mean ρ = 5.65e-02  ->  PASS
```
**The pretrained model carries real, measurable curl** (ρ ≈ 2–6%). The program is alive.

### E1b — Positive control (HARD GATE) ✅
```
plateau ||A_theta||_F (model curl floor) = 1.28    knee ~ amp 1.28
asymptotic slope (floor-subtracted) = 1.020    recovered ||R||_F = 1.044  (true 1.0)
PASS
```
Injected rotation recovered to within ~4%; response is a clean hockey-stick (flat at the
model's own floor, then slope 1). The plateau `||A_θ||_F≈1.28` is consistent with G0
(ρ≈0.018 × `||∇v||_F≈65 ≈ 1.2`). Instrument has calibrated power.
*(Note: the first run failed this gate due to a slope-window bug — the fit straddled the
knee. Fixed by subtracting the floor in quadrature and fitting the asymptotic tail only.)*

### E1c — Negative control at d=3072 (HARD GATE) ✅
```
d=3072  N=4096  estimator=jvp  float64
mean ρ(v*) = 1.85e-15   (threshold 1e-04)  ->  PASS
```
**The headline number.** On the exact irrotational field, in the real image dimension, the
instrument reads `ρ = 2e-15` — pure numerical zero. Against the model's `ρ ≈ 5e-2`, that is a
**~13-order-of-magnitude gap** between "the exact field" and "the trained network." That gap
is the entire thesis, and the instrument resolves it cleanly.

### E1d — Precision floor
```
instrument floor: dtype=float64  h=1e-05  ->  ρ(v*) = 3.5e-10
```
The AD-free floor (~3.5e-10) sits ~8 orders of magnitude below the real signal (ρ≈5e-2), and
the primary JVP estimator does even better (2e-15). Ample headroom.

### E1e — Cost / fidelity (informational, not a gate)
No cheap probe×knot config reached Spearman 0.95 vs the dense reference (best ≈ 0.77 at
8 probes × 6 knots). **Reading:** among in-distribution noised-real points the ρ *ranking* is
estimator-variance-limited at M≤16 probes — magnitude is cheap, but *ranking* needs dense
estimation. Consistent with the plan's warning that "near-free certification is an overclaim."
Implication for later stages: ranking-based tasks (E4/E5) need more probes than magnitude-based
ones (E2).

---

## Gate summary

| Gate | Result |
|---|---|
| G0 (curl exists) | **PASS** |
| G1b (recovers known rotation) | **PASS** |
| G1c (reads zero on irrotational v*) | **PASS** |

The instrument measures what it claims, with power (E1b), a clean null (E1c), and a
characterized floor (E1d). **Cleared to scale to Stage 2.**
