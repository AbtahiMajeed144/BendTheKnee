## What the experiments must establish

Four claims, in dependency order. Every experiment below maps to exactly one.

1. **Instrument**: the estimator measures $\|A_\theta\|_F$ with known power, known floor, known cost.
2. **Calibration**: $\rho$ is a *quantitatively meaningful* fraction of true Jacobian error, not just a positive number.
3. **Attribution**: the measured curl is approximation/optimization error — not coupling bias, not data geometry, not the score's own artifacts.
4. **Utility**: it predicts something, better than $\|a\|$ and the standard uncertainty baselines.

Claims 1–3 are cheap and mostly deterministic. Claim 4 is the gamble. So the plan front-loads the cheap kills.

---

## E0 — Fix the score (half a day, CPU + a few GPU minutes)

**Primitive.** $\rho(x,t)=\|A_\theta(x,t)\|_F/\|\nabla v_\theta(x,t)\|_F\in[0,1]$.

This is not cosmetic. As $t\to1$, $\nabla v^\star\to -I/(1-t)$ so $\|\nabla v_\theta\|_F\sim\sqrt d/(1-t)$, and $\|A_\theta\|_F$ carries the same $(1-t)^{-1}$: **the ratio cancels the blowup to leading order**. It is also directly interpretable — $\rho$ is a certified lower bound on *relative* Jacobian error. That's your abstract number.

**Path score.** $\bar\rho^2(x_0)=\int_0^1\rho^2\,w(t)\,dt$, $w$ a normalized weight (default uniform in $t$; also report uniform in log-SNR). Always publish the $t$-resolved profile alongside the scalar.

**Deliverable figure (worth an appendix panel):** old $V$ vs NFE $\in\{50,100,200,400\}$ showing $V\propto$ NFE and last-step dominance; $\bar\rho$ flat. Plus Spearman$(V,\bar\rho)$ on the existing 10k run — if it's low, every number in §7 is about the final Euler step.

**Probe design (fix once, use everywhere).** Stratified $t\in[0.02,0.98]$, 8–16 knots. Two evaluation sets, never mixed: **(P1)** noised real data $x_t=(1-t)\epsilon+tx_1$ — this is $p_t$, where the loss lives and where $v^\star$ is the minimizer, used for E1/E2/E3/E6; **(P2)** points on generated trajectories, used only for E4/E5.

---

## E1 — Instrument validation (1 day)

**Hour zero, before anything:** raw $\rho$ on the pretrained CIFAR-10 checkpoint at 3 timesteps, 100 points. If $\rho\lesssim10^{-5}$, the program is dead in an hour and you saved a month.

| Test | Protocol | Gate |
|---|---|---|
| (a) Exactness | FD and JVP+VJP estimators vs full autodiff Jacobian, 2D MLP | rel. err < 1% |
| (b) **Positive control** | $v\leftarrow v_\theta+\epsilon Rx$, $R$ antisym., $\epsilon$ over 6 decades | Hockey-stick: flat at $\|A_\theta\|_F$, then slope 1. The knee independently locates the model's own curl |
| (c) **Negative control at scale** | Estimator run on the *exact empirical* $v^\star$ on CIFAR-10, $d{=}3072$ | $\rho<10^{-5}$. This is Phase A redone in the real dimension — far stronger than the 2D $10^{-29}$ |
| (d) Precision floor | $h$-sweep, dtype fp32/fp64, cancellation analysis | Report the floor as a number |
| (e) **Cost–fidelity curve** | Spearman of $\bar\rho$ ranking vs (probes $M$, #$t$-knots) against a dense reference | Find the cheapest config with $\tau_{\text{rank}}>0.95$ |

Primary estimator: $\|A\|_F^2=\tfrac14\mathbb E_\varepsilon\|(\nabla v-\nabla v^\top)\varepsilon\|^2$ (1 JVP + 1 VJP, shared $\varepsilon$). The four-forward-pass form is a *black-box fallback*, validated against the primary in (a) — its variance ratio is $3+r_{\text{stab}}$, so it is not a free upgrade and shouldn't be sold as one.

(e) matters more than it looks. At $M{=}8$ probes × 10 knots, certification costs ~$10^1$–$10^2$ forward-equivalents per sample — the same order as sampling itself. "Near-free" is an overclaim until (e) gives you the honest number.

---

## E2 — Calibration: tightness and the null test, merged (1 day)

The single most valuable asset in this project: **on CIFAR-10 the exact target is closed-form.** $v^\star=(\mu-x)/(1-t)$, $\nabla v^\star=\frac{1}{1-t}\big(\frac{t}{(1-t)^2}\mathrm{Cov}_w-I\big)$, and $\nabla v^\star\varepsilon$ is two matvecs against the $50\text{k}\times3072$ data matrix. Nobody has exploited this.

Since $\mathrm{asym}(\nabla v^\star)=0$, we have **exactly** $A_\theta=\mathrm{asym}(E)$ where $E=\nabla v_\theta-\nabla v^\star$. So

$$\tau^2 \;=\; \frac{\|A_\theta\|_F^2}{\|E\|_F^2} \;=\; \text{antisymmetric fraction of the Jacobian error},\qquad \text{null: } \tfrac{d-1}{2d}=0.49984 .$$

This is where the earlier $\rho$-vs-0.5 test was broken — the null belongs on $E$, not on $\nabla v_\theta$. With $v^\star$ in hand, the test becomes valid.

Measure $\tau^2$ on P1, resolved in $t$ and in $\mathrm{tr}\,\mathrm{Cov}_w$. Two caveats to state in the paper: $v^\star_{\text{emp}}$ is the *training-loss* minimizer, so $E$ includes desirable generalization — but the bound holds against the population field too (the theorem needs no assumption on $p_1$), and measuring against one family member *underestimates* true tightness.

**All three outcomes are publishable, which is why this experiment is worth running first:**

| Result | Reading |
|---|---|
| $\tau^2\approx0.5$ | Error is isotropic; certificate captures half of it. Clean, quantitative, sufficient. |
| $\tau^2\gg0.5$ | Rotational excess — §6's thesis survives, rebuilt on the correct mechanism (the loss controls $L^2$, not $H^1$; curl lives where the objective can't see). |
| $\tau^2\ll0.5$ | Networks are *implicitly conservative*. Certificate is weak in magnitude, but this is a better paper than the one you set out to write. |

---

## E3 — Attribution (5–8 GPU-hours, MNIST/FashionMNIST farm)

Causal claims need ~20 models, so they get made at a scale where 20 models cost one afternoon. The pretrained CIFAR model then carries external validity. That split is the whole resource strategy.

| Control | Prediction if curl = approximation error |
|---|---|
| Checkpoint sweep | $\rho$ falls monotonically with training |
| Capacity sweep (width/depth) | $\rho$ falls monotonically |
| Two seeds, **identical probe points** | Hotspots *differ* ⇒ optimization noise; *coincide* ⇒ data geometry. Report spatial Spearman |
| I-CFM / OT-CFM twins, matched everything | $\Delta\rho$ = the coupling contribution, measured not assumed |
| $\rho(B)$, $B\in\{1,2,4,\dots,256\}$ | **Non-monotone**: zero at $B{=}1$ (independent coupling), zero as $B\to\infty$ (Brenier), hump between |

The twins experiment is the cheap fix for the OT-CFM mismatch: rather than retraining CIFAR to get an I-CFM checkpoint, **bound the confound**. If $\Delta\rho\ll\rho_{\text{measured}}$ at MNIST scale, the pretrained OT-CFM checkpoint is usable for E4/E5 with a stated correction. Check first whether torchcfm ships I-CFM CIFAR weights — if it does, this becomes a direct check instead of an extrapolation.

The $\rho(B)$ hump is a genuine pre-registered signature. A monotone decay would be confoundable with anything else that varies with $B$; a curve that is zero at both ends is not.

---

## E4 — Rebuild §7 honestly (1–2 days, inference only)

Re-run at 5k samples (enough for KID, bootstrap CIs) rather than 10k, with $\bar\rho$ replacing $V$:

- **Full binned scatter** with bootstrap CIs — not two tail buckets.
- **Energy controls**: per-bin pixel-std, TV, high-frequency energy; partial correlation controlling for all three; plus an energy-matched resample.
- **Solver control**: dopri5 at tight tolerance. If the effect dies, it was truncation error.
- **Feature-space NN**: DINOv2 or SSCD, and the $d_1/d_2$ ratio. Pixel $L_2$ at 0.084 vs 0.167 per-pixel RMS is a typicality axis, not a memorization axis.
- **KID** on ≥1k buckets. Never FID on 100.

**The mechanism test, which the two-bucket design structurally cannot see.** Far from data the softmax goes one-hot, $\mathrm{Cov}_w\to0$, $\nabla v^\star\to-I/(1-t)$ — the far tail is where the target is *easiest*. Difficulty lives at basin boundaries: large $\mathrm{Cov}_w$, steep weight transitions, mid-density. And because $v^\star$ is closed-form, you can compute $\mathrm{tr}\,\mathrm{Cov}_w$ and softmax entropy **at every probe point** and regress $\rho$ on them directly.

Prediction: hump. Monotone line ⇒ suspect the energy confound. This converts a proxy argument into a direct measurement, and it costs one extra matvec per probe.

**Kill gate:** if $\bar\rho$'s association with distance-to-data does not survive energy-matching *and* the high-NFE re-run, the OOD framing is dead. Pivot to E2 + E6 and delete §7–§8. Better to learn this in week two.

---

## E5 — The $\|a\|$ showdown (1 day)

Do not run the naive correlation. $a=\partial_t v+\nabla\tfrac12|v|^2+2Av$ — the Lamb term *is a summand of the acceleration*, so correlation is mechanically guaranteed and answers nothing.

1. **Nested model**: does $\bar\rho$ add AUC over $\|a\|$? Partial correlation + nested logistic regression, bootstrap CI on $\Delta$AUC.
2. **Lamb ablation**: compare $\|a\|$, $\|2Av\|$, $\|a-2Av\|$ as predictors. If the residual carries everything, curl is decoration and you say so.
3. **Selective generation** (the native evaluation for a one-sided certificate — abstention needs precision, which is exactly what you have and per E7 can never upgrade to recall): reject top-$\alpha$, plot quality-vs-retention, report **AURC**. Baselines: $\|a\|$, $\|s_\theta\|$, $\|\nabla v_\theta\|_F$, ODE likelihood, 2-seed ensemble disagreement, and a trained artifact classifier as the cheating ceiling. Quality metric: KID on retained set + artifact-classifier rate.

Human raters only *after* this passes, and only on the fixed score. Skip curl-adaptive NFE — an embedded RK controller already estimates local truncation error more cheaply, and you'd lose that comparison.

**Also register the acceleration certificate, and be precise about it.** $a^\star=\nabla(\partial_t\Phi+\tfrac12|\nabla\Phi|^2)$ (Hamilton–Jacobi), so $\mathrm{asym}(\nabla a^\star)\equiv0$. But it shares a null set with curl — $v_\theta$ curl-free $\Rightarrow$ $a_\theta$ curl-free. It's a *reweighting*, not an independent test. The genuinely independent members are $\mathcal R_\theta$ (continuity) and the spectral bound $\nabla v_\theta\succeq-I/(1-t)$, which a curl-free model can violate. Claim three tests, one with two realizations — not four certificates.

The spectral bound is nearly free (a few Lanczos steps on the symmetric part) and holds for arbitrary $p_1$. Take it.

---

## E6 — External-input certification (1 day, inference only)

The differentiator, and cheaper than E4 since there's no sampling loop. Noise real images to $x_t$, average $\rho$ over $t$ and noise draws, run CIFAR-10 model → SVHN / CIFAR-100 / Places / CelebA. Report AUROC against ODE likelihood (famously fails here), typicality test, $\|s_\theta\|$, ensemble.

Scoring the model's own samples was always strange — they're in-distribution by construction. This is the version where "self-certifying generative model" gets a hard number.

---

## Budget and sequencing

| Week | Work | Gate |
|---|---|---|
| 1 | Hour-zero check, E0, E1 | Instrument passes (b),(c) or stop |
| 1–2 | **E2** | Report $\tau^2$ whatever it is — paper exists from here |
| 2 | E3 | Attribution + coupling bound |
| 3 | E4 | Survives confounds, or §7 is deleted |
| 3–4 | E5, E6 | $\Delta$AUC over $\|a\|$; AURC; OOD AUROC |

Assumes one 24GB consumer GPU. Roughly one GPU-week of compute; the only genuinely expensive item is E4's sampling, and 5k samples suffices for KID with CIs. Everything else is inference and matvecs.

**Pre-register before running:** the $\tau^2$ three-way readout, the $\rho(B)$ hump, the non-monotone $\rho$-vs-density profile, and the $\Delta$AUC-over-$\|a\|$ threshold you'll accept as a positive. Stating these in advance is much of what separates this from a correlation plot.

**Naming discipline throughout:** *certified lower bound on relative Jacobian error*. Not hallucination, not memorization, not OOD. The certificate proves the model's **field** is wrong at $x_t$; that the resulting image is bad is a separate empirical bridge that E4 and E5 either build or don't.

Want me to write E1's validation script (positive control + the $d{=}3072$ negative control against exact $v^\star$), or draft the pre-registration table with explicit numeric thresholds first?