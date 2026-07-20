# Execution Plan v1.0 — Curl Certificate for Flow-Matching Velocity Fields

**Status:** pre-implementation. This document is the contract. Code is written against it, not ahead of it.

**One-line thesis:** $\|A_\theta\|_F$ is a **certified, training-data-free lower bound on the relative Jacobian error** of a learned flow-matching velocity field. "Memorization / OOD / hallucination" is a *downstream empirical bridge* (E4–E6), never assumed.

**Naming discipline (throughout paper and code):** *certified lower bound on relative Jacobian error*. The certificate proves the model's **field** is wrong at $x_t$; that the resulting image is bad is a separate claim E4/E5 either earn or don't.

---

## 0. The central certificate (what every experiment serves)

For $x_t=(1-t)x_0+t x_1$ with $x_0\sim\mathcal N(0,I)$ **isotropic** and **independent** of $x_1$ (i.e. I-CFM / rectified flow), for *any* target $p_1$:
$$v^\star(x,t)=\nabla\Phi_t(x)\quad\Rightarrow\quad \mathrm{asym}(\nabla v^\star)\equiv 0\ \text{exactly, everywhere.}$$
Since $\mathrm{asym}$ is an orthogonal projection and $A_\theta=\mathrm{asym}(\nabla v_\theta-\nabla v^\star)$:
$$\boxed{\ \|A_\theta\|_F\ \le\ \|\nabla v_\theta-\nabla v^\star\|_F\ }$$
High curl **certifies** Jacobian error (precision). Zero curl certifies **nothing** (the symmetric part can be arbitrarily wrong) — this is a one-sided instrument, sold as precision, never recall.

The correct motivation for *why curl exists at all*: the CFM loss is an exact Bregman identity controlling $\|v-v^\star\|_{L^2(p_t)}$ but **not** $\|\nabla v-\nabla v^\star\|$. Curl lives in the $H^1$ part the objective never sees. (The old "network bleeds energy into unpenalized rotational fields" story is false and is retired.)

**Deleted before any code:** hypothesis H4 (curl vanishes for N=2, nonzero for N≥3) is falsified for all N by the covariance-cancellation argument.

---

## 1. Model selection — decided by the math

The theorem holds **only for independent coupling**. Therefore:

| Track | Checkpoint (release 1.0.4) | Coupling | Uses |
|---|---|---|---|
| **Certificate (spine)** | `cfm_cifar10_weights_step_400000.pt` | I-CFM (independent) | E1c, **E2**, E6 |
| **Bias-meter (secondary)** | `otcfm_cifar10_weights_step_400000.pt` | Exact-OT (non-independent) | E3 only |

- `cfm_` is the repo's base `ConditionalFlowMatcher` = independent coupling. Its target field is provably irrotational → the certificate is valid.
- `otcfm_` has coupling-induced curl bias; using it to *certify* entangles architecture error with OT bias. It is used **only** to *measure* that bias ($\Delta\rho$, E3).
- The prior Kaggle probe used `otcfm_` — the wrong file. Switching to `cfm_` is the highest-value single fix.

Download:
```
https://github.com/atong01/conditional-flow-matching/releases/download/1.0.4/cfm_cifar10_weights_step_400000.pt
https://github.com/atong01/conditional-flow-matching/releases/download/1.0.4/otcfm_cifar10_weights_step_400000.pt
```

---

## 2. Confirmed engineering facts (verified against the repo; silent-corruptor class)

| Fact | Value | Why it matters |
|---|---|---|
| Training `sigma` | **0.0** | Path is exactly $(1-t)x_0+tx_1$; closed-form $v^\star$ is **exact**, not approximate. E2 is a clean calibration. |
| Normalization | **GroupNorm** (ADM UNet) | Per-sample ⇒ `torch.func.jvp/vjp` over a batch is *correct*. BatchNorm would silently corrupt every Jacobian. |
| Parameterization | **velocity** `v(t,x)` | No $\hat x_1$-prediction trap; take the Jacobian directly. |
| Signature | `model(t, x)` | t scalar or per-sample tensor. |
| Weights | `checkpoint["ema_model"]` | EMA = converged. Strip `module.` prefix if DataParallel-saved. |
| Data range | **[-1, 1]** (`Normalize(0.5,0.5)`) | Model and $v^\star$ must see identical normalization or comparison is meaningless. |
| UNet config | dim=(3,32,32), num_res_blocks=2, num_channels=128, channel_mult=[1,2,2,2], num_heads=4, num_head_channels=64, attention_resolutions="16", dropout=0.1 | Same for both checkpoints. |

**Closed-form target (sigma=0):**
$$w_i \propto \exp\!\Big(-\tfrac{\|x_t - t\,x_i\|^2}{2(1-t)^2}\Big),\quad \mu=\sum_i w_i x_i,\quad v^\star=\frac{\mu-x_t}{1-t},\quad \nabla v^\star=\frac{1}{1-t}\Big(\tfrac{t}{(1-t)^2}\mathrm{Cov}_w-I\Big).$$
Apply $\nabla v^\star$ to a probe vector without forming the $3072^2$ matrix:
$$\mathrm{Cov}_w\,\varepsilon=\sum_i w_i (x_i-\mu)\langle x_i-\mu,\varepsilon\rangle\quad(\text{two matvecs against the }50\text{k}\times3072\text{ data matrix}).$$
Weights computed via **float64 logsumexp**; probe $t$ restricted to $[0.02, 0.98]$ (the $1/(1-t)$ singularity).

---

## 3. Shared probe library `curlprobe/` (build once, every stage imports)

Four unit-tested primitives. The prior scripts' failure mode was re-implementing the core per phase.

1. **`estimator`** — $\|A\|_F^2=\tfrac14\mathbb E_\varepsilon\|(J-J^\top)\varepsilon\|^2$ and $\|\nabla v\|_F^2=\mathbb E_\varepsilon\|J\varepsilon\|^2$ (shared JVP+VJP, M≈8 Gaussian probes). **Also** the AD-free 4-forward-pass FD form $\|A\|_F^2=\tfrac14\mathbb E_{\varepsilon,\eta}[(\varepsilon^\top J\eta-\eta^\top J\varepsilon)^2]$. Primary-vs-fallback **decided at hour zero** by the E1a agreement test (attention ops may lack a forward-mode rule).
2. **`vstar`** — exact $v^\star$ and $\nabla v^\star\varepsilon$ per §2, float64 logsumexp.
3. **`probe_points`** — **P1** (noised real data $x_t=(1-t)\epsilon+t x_1$; $v^\star$ is the minimizer here → E1/E2/E3/E6) vs **P2** (generated-trajectory points → E4/E5). *Never mixed.* Stratified $t\in[0.02,0.98]$, 8–16 knots.
4. **`score`** — scale-free $\rho=\|A\|_F/\|\nabla v\|_F\in[0,1]$; path score $\bar\rho^2=\int\rho^2 w(t)\,dt$; always emit the $t$-resolved profile.

---

## 4. Staged execution with hard gates (Kaggle T4 / T4×2)

**Stage 1 — Instrument (E0+E1).** One notebook, mostly CPU + minutes GPU.
- *Hour-zero:* raw $\rho$ on `cfm_`, 3 timesteps, 100 noised real points.
- E1a exactness (FD & JVP+VJP vs full autodiff Jacobian, 2D MLP) — **also selects the CIFAR estimator.**
- E1b positive control (inject $\epsilon Rx$, 6 decades).
- E1c negative control at $d=3072$ (estimator on exact $v^\star$).
- E1d precision floor. E1e cost–fidelity curve.
- **Gate below.**

**Stage 2 — Calibration (E2).** ~1 day GPU on `cfm_`. Measure $\tau^2=\|A_\theta\|_F^2/\|E\|_F^2$ vs null $0.5$, resolved in $t$ and $\mathrm{tr\,Cov}_w$, bootstrap CIs. **Paper exists from here regardless of outcome.**

**Stage 3 — Attribution (E3).** MNIST/FashionMNIST farm (~20 models, one afternoon): checkpoint / capacity / seed sweeps; `cfm_`/`otcfm_` twins for $\Delta\rho$; the $\rho(B)$ hump.

**Stage 4 — Utility (E4/E5/E6).** Rebuild §7 with energy + dopri5 controls (kill gate); $\|a\|$ showdown (AURC); external-input OOD (needs Kaggle internet for DINOv2/SSCD).

Resource note: single-GPU MVP throughout; T4×2 optional (data matrix on GPU-1). P1 evaluation has no sampling loop → far cheaper than the old generation+probe combo.

---

## 5. Pre-registration table (numeric thresholds fixed *before* running)

### Instrument gates — Stage 1 (all must pass or STOP)
| ID | Test | Pass condition | On fail |
|---|---|---|---|
| G0 | Hour-zero raw $\rho$ (`cfm_`) | $\rho > 10^{-5}$ (expect $10^{-3}$–$10^{-1}$) | **Program dead — stop in 1 hour** |
| G1a | FD & JVP+VJP vs autodiff Jacobian (2D MLP) | rel. err < 1% | Debug estimator |
| G1a′ | Estimator selection | Pick JVP+VJP if it matches autodiff <1%; else FD fallback | — |
| G1b | Positive control, $\epsilon\in[10^{-6},10^0]$ | Hockey-stick: flat at $\|A_\theta\|_F$, then slope 1; knee = model curl floor | Instrument has no power |
| G1c | Negative control on exact $v^\star$, $d{=}3072$ | $\rho < 10^{-5}$ | Bug in $v^\star$ or estimator |
| G1d | Precision floor ($h\sim\varepsilon_{\text{mach}}^{1/3}$, fp32/fp64) | Report as a number; floor $\ll$ measured $\rho$ | Raise dtype |
| G1e | Cost–fidelity (M × knots vs dense ref) | cheapest config with rank-corr $\tau>0.95$ | Use dense |

### E2 — three-way readout (null $\tau^2=(d-1)/2d=0.49984$ at $d{=}3072$)
| Measured $\tau^2$ (bootstrap CI) | Registered reading |
|---|---|
| CI ⊂ $[0.45,0.55]$ | Isotropic error; certificate captures ~half. Clean, sufficient. |
| CI lower bound $> 0.55$ | **Rotational excess** — §6 thesis survives on the correct mechanism ($L^2$ not $H^1$). |
| CI upper bound $< 0.45$ | Networks **implicitly conservative** — a better paper than intended. |

### E3 — attribution (pre-registered signatures)
| Control | Registered prediction (approximation-error hypothesis) |
|---|---|
| Checkpoint sweep | $\rho$ falls monotonically with training (Spearman $<0$, CI excludes 0) |
| Capacity sweep (width/depth) | $\rho$ falls monotonically |
| Two seeds, identical probe pts | spatial Spearman: $\approx 0$ ⇒ optimization noise; $\gg 0$ ⇒ data geometry |
| `cfm_`/`otcfm_` twins | $\Delta\rho=\rho_{\text{otcfm}}-\rho_{\text{icfm}}$ = coupling bias; if $\Delta\rho\ll\rho_{\text{measured}}$, pretrained `otcfm_` usable elsewhere with stated correction |
| $\rho(B)$, $B\in\{1,2,4,\dots,256\}$ | **Non-monotone hump**: $\approx 0$ at $B{=}1$, $\approx 0$ as $B\to\infty$, positive between (peak location = free finding) |

### E4 — §7 rebuild (kill gate)
- **Kill gate:** $\bar\rho$–vs–distance-to-data association must survive **both** (a) energy-matching (partial correlation controlling pixel-std, TV, HF energy) **and** (b) dopri5 high-NFE re-run. Else delete §7–§8, pivot to E2+E6.
- Mechanism test: regress $\rho$ on $\mathrm{tr\,Cov}_w$ and softmax entropy at every probe point. **Registered prediction: hump** ($\rho$ low near data, peaks at basin boundaries / mid-density). Monotone line ⇒ suspect energy confound.
- Statistics: 5k samples, KID on ≥1k buckets (never FID on 100), bootstrap CIs, full binned scatter.

### E5 — the $\|a\|$ showdown
- Do **not** run naive $\|a\|$ correlation ($2Av$ is a summand of $a$ — mechanically guaranteed, answers nothing).
- Registered positive: nested logistic $\Delta$AUC of $\bar\rho$ over $\|a\|$ with **bootstrap CI excluding 0** (target $\Delta\text{AUC}\ge0.02$).
- Lamb ablation ($\|a\|$, $\|2Av\|$, $\|a-2Av\|$): if the residual carries everything, curl is decoration — say so.
- Selective generation: report **AURC**; baselines $\|a\|$, $\|s_\theta\|$, $\|\nabla v_\theta\|_F$, ODE likelihood, 2-seed ensemble, trained artifact classifier (cheating ceiling).
- Register the **spectral bound** $\nabla v_\theta\succeq -I/(1-t)$ (nearly free, few Lanczos steps) and continuity residual $\mathcal R_\theta$ as the genuinely *independent* certificates. The acceleration certificate shares curl's null set — it is a reweighting, not an independent test. Claim three tests, one with two realizations.

### E6 — external-input OOD (the differentiator)
- CIFAR-10 `cfm_` model, noise real images to $x_t$, average $\rho$ over $t$ and noise draws.
- Registered metric: AUROC vs SVHN / CIFAR-100 / Places / CelebA, reported per dataset.
- Baselines: ODE likelihood (famously fails here), typicality test, $\|s_\theta\|$, ensemble disagreement.

---

## 6. Risk register

| # | Risk | Mitigation | Caught by |
|---|---|---|---|
| R1 | Wrong checkpoint (OT vs I-CFM) | `cfm_` certifies; `otcfm_` only measures bias | load |
| R2 | `jvp` unsupported op (attention) | FD fallback, chosen at hour zero (G1a′) | G1a |
| R3 | T4 OOM (jvp+vjp ≈3× mem) | separate P1 eval from generation; batch 8→down; `empty_cache` | Stage 1 |
| R4 | $v^\star$ overflow (3072-dim exp, $1/(1-t)^2$) | float64 logsumexp; $t\le0.98$ | G1c |
| R5 | Normalization mismatch model vs $v^\star$ | enforce [-1,1], single data-load path | unit test |
| R6 | BatchNorm corrupting JVP | confirmed GroupNorm | ✓ resolved |
| R7 | P1/P2 mixing invalidates $\tau^2$ | enforced in `probe_points` API | by design |
| R8 | Raw $\|A\|_F$ endpoint blowup dominates | scale-free $\rho$; $t$-resolved profile | E0 |
| R9 | DataParallel `module.` prefix | strip on load | Stage 1 |
| R10 | EMA vs raw weights | use `ema_model` | load |

---

## 7. Immediate next action

Build **Stage 1** (the `curlprobe/` core + E0 hour-zero + E1 controls) as one self-contained Kaggle notebook. It is the cheapest possible test of whether the program is real, and it forces the shared library into existence correctly. Estimator (JVP+VJP vs FD) is selected inside it by G1a′.

**Gate:** proceed to Stage 2 only if G0, G1b, G1c pass.
