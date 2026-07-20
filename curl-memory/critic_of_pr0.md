Read it line by line and re-derived everything independently. The central mathematical claim in §5 is **correct** — and it's the best thing in the document. But it's not new, it's mis-attributed to the wrong estimator (OT-CFM), and §6's causal story built on top of it contains a load-bearing error. Also, one of the claims it refutes is *mine*, so let me start there.

---

## 0. Retraction (my error, your report caught it)

Last turn I claimed the empirical RF curl is $\sum_i v_i \wedge \nabla w_i$, vanishing for $N=2$ but generically nonzero for $N\ge3$ non-collinear points. **That is wrong.** Your covariance argument kills it for all $N$:

$$\nabla w_i = \tfrac{t\,w_i}{(1-t)^2}\big(x^{(i)}-\mu\big),\quad \mu=\textstyle\sum_j w_j x^{(j)} \;\Longrightarrow\; \sum_i v_i(\nabla w_i)^\top = \tfrac{t}{(1-t)^3}\Big[\mathrm{Cov}_w(X) + (\mu-x)\underbrace{\textstyle\sum_i w_i(x^{(i)}-\mu)^\top}_{=0}\Big]$$

Symmetric for every $N$, every configuration, every $t$. Hypothesis **H4 in the RUNBOOK is falsified and must be deleted** before that seed generates any code.

---

## 1. The theorem, stated with its actual hypotheses

Your §5 proves a special case of something worth stating in full generality, because the generality is exactly what makes it a *certificate*:

> **Theorem.** Let $x_t=\alpha_t x_0+\beta_t x_1$ with $x_0\sim\mathcal N(0,\sigma^2 I)$ **isotropic** and **independent** of $x_1$. Then for *any* $p_1$ whatsoever (empirical, mixture, natural images):
> $$v^\star(x,t)=\frac{\dot\beta_t}{\beta_t}x+\frac{\sigma^2\alpha_t(\dot\beta_t\alpha_t-\dot\alpha_t\beta_t)}{\beta_t}\nabla\log p_t(x)=\nabla\Phi_t(x)$$
> Hence $\mathrm{asym}(\nabla v^\star)\equiv 0$, exactly, everywhere.

Proof: $v^\star=\frac{\mathbb E[x_1|x_t]-\ldots}{}$ reduces to the score by Tweedie, which requires precisely isotropy + independence. Your covariance cancellation is this identity in coordinates.

**What breaks it** (each is a real experimental hazard, not a footnote):

| Perturbation | Still curl-free? |
|---|---|
| Any data distribution, any $N$, any outliers | **Yes** |
| Any affine schedule $(\alpha_t,\beta_t)$ | **Yes** |
| Anisotropic source $\mathcal N(0,\Sigma)$ | **No** — $\nabla\mu\propto\mathrm{Cov}_w\Sigma^{-1}$, product of symmetrics isn't symmetric |
| Non-Gaussian source (image-to-image) | **No** |
| **Non-independent coupling (minibatch OT)** | **No** in general; yes in the $B\to\infty$ limit (true OT ⇒ Brenier gradient) |

Also: this is known. Lim states the gradient-field property explicitly; and it's implicit in every probability-flow ODE for affine-Gaussian diffusions ($f$ linear $+\;g^2\nabla\log p_t$ — manifestly a gradient). Frame §5 as **a correct rederivation whose novel use is as an error certificate**, not as a discovery. Petrović et al.'s Curly FM is your ally here: they motivate their whole method by the fact that standard FM can only learn gradient fields.

---

## 2. Errors, ranked by how much damage they do

**E1 — "the network bleeds energy into unpenalized, flux-null rotational fields." This is false, twice over.**

(a) *Nothing is unpenalized.* The CFM loss satisfies the exact Bregman identity $\mathcal L(v)=\mathcal L(v^\star)+\|v-v^\star\|^2_{L^2(p_t)}$. Every deviation — rotational, symmetric, whatever — is penalized identically. There is no cheap direction in the loss. (b) *Curl $\ne$ flux-null.* These are different decompositions: $A\ne0$ means non-gradient; flux-null means $\nabla\!\cdot\!(p_t w)=0$, which for $w$ with nonzero divergence requires the pointwise condition $\nabla\!\cdot\!w+w\!\cdot\!\nabla\log p_t=0$. A rotational error is generically *not* flux-null — it moves the marginals and costs FM loss.

The defensible replacement: the loss controls $\|v-v^\star\|_{L^2}$ but **not** $\|\nabla v-\nabla v^\star\|$. Curl lives in the $H^1$ part the objective never sees. That's a stronger motivation than the one you have, and it's true.

**E2 — OT-CFM mislabel is an experiment/theory mismatch, not a naming quibble.** Your §5 derivation is the *independent-coupling* (I-CFM / rectified flow) field. §10 commits to torchcfm's exact-OT matcher, whose induced coupling is non-independent, so its target field is **not** provably irrotational. Measured curl there = architectural error **+ coupling-induced bias, entangled**. Fix: train/obtain an I-CFM checkpoint for the certificate experiments. Silver lining — the entanglement is itself a finding: **curl as a minibatch-OT bias meter**, $\rho(B)$ should decay as batch size $B\to\infty$ toward the Brenier gradient limit. Nobody has quantified minibatch-OT bias this way, and it costs only forward passes.

**E3 — §9 says curl is a "finite-sample artifact."** Your own §5 disproves that: the *exact empirical* field is curl-free. Curl is purely approximation/optimization error (or coupling bias). Internal contradiction; one-line fix.

**E4 — $1.4\times10^{-29}$ is not machine epsilon.** float64 eps is $2.2\times10^{-16}$; $10^{-29}$ is consistent with a *squared* quantity whose entries are $\sim10^{-15}$, i.e. roundoff — self-consistent, but you proved nothing until you show (i) the **relative** figure $\|A\|_F/\|\nabla v\|_F$, and (ii) a **positive control**: add a known rotation $w=\epsilon(x_2,-x_1)$ and confirm the estimator recovers it at the right magnitude. A null result from an instrument with undemonstrated power is not a result. Keep the logsumexp fix regardless — you need it in the far tail where all weights genuinely underflow to $0/0$.

**E5 — no null model, so "high curl" is uncalibrated.** For an error matrix with i.i.d. entries, $\mathbb E\|A\|_F^2/\mathbb E\|\nabla v\|_F^2=(d-1)/2d\approx 0.5$. So the whole "architectural approximation failure" thesis reduces to one measurable number: is $\rho=\|A\|_F^2/\|\nabla v\|_F^2$ **above** $0.5$ (rotational excess — your thesis), **at** $0.5$ (isotropic garbage — no special story), or **below** (nets are implicitly conservative — arguably a more interesting paper). This is a ten-line experiment and it decides whether §6 is a thesis or a vibe.

**E6 — the estimator as specified will be variance-dominated.** Do **not** compute $\|A\|_F^2=\frac12(\|\nabla v\|_F^2-\mathrm{tr}((\nabla v)^2))$: both terms are large and nearly equal, so you're subtracting to find a small residual. Use the antisymmetrized form
$$\|A\|_F^2=\tfrac14\,\mathbb E_{\varepsilon}\big\|(\nabla v-\nabla v^\top)\varepsilon\big\|^2 \quad\text{(1 JVP + 1 VJP, shared }\varepsilon)$$
and note you don't need autodiff at all:
$$\|A\|_F^2=\tfrac14\,\mathbb E_{\varepsilon,\eta}\Big[\big(\varepsilon^\top(\nabla v)\eta-\eta^\top(\nabla v)\varepsilon\big)^2\Big],\qquad \varepsilon^\top(\nabla v)\eta=\tfrac{d}{dh}\big|_{0}\langle\varepsilon,\,v(x+h\eta,t)\rangle$$
— **four forward passes, no AD**, so it works on compiled/quantized/black-box models. Use fp32 minimum and pick $h\sim\varepsilon_{\text{mach}}^{1/3}$; if $\|A\|_F/\|S\|_F\sim10^{-2}$ you lose two digits to cancellation, which sets your precision floor.

**E7 — "low curl ⇒ the model knows what it's doing" is not supported.** The certificate is strictly one-sided. Since $\mathrm{asym}(\nabla v^\star)=0$ and $\mathrm{asym}$ is an orthogonal projection:
$$\boxed{\;\|A_\theta\|_F\;\le\;\|\nabla v_\theta-\nabla v^\star\|_F\;}$$
High curl **certifies** Jacobian error, with no ground truth needed, against *every* member of the affine-Gaussian target family simultaneously. Zero curl certifies **nothing** — the symmetric part can be arbitrarily wrong. Sell it as precision, not recall.

---

## 3. The §7 empirical result: I think there's a good chance it's a blurriness detector

Bottom-100 curl → NN $L_2$ = 4.66; top-100 → 9.27. Before this becomes a paper, four confounds, in order of how likely they are to kill it:

1. **Image-energy confound (most dangerous).** Blurry, low-contrast, near-mean images sit in flat regions (small $\|\nabla v\|$, hence small $\|A\|_F$ — the metric isn't scale-invariant) *and* are close to everything in pixel $L_2$. That single mechanism produces your entire correlation with no epistemics involved. **Controls:** use the scale-free ratio $\|A\|_F/\|\nabla v\|_F$; report per-bucket pixel-std, TV, and high-frequency energy; recompute the correlation matched on image energy.
2. **Discretization mechanism.** High curl ⇒ larger local truncation error ⇒ at fixed NFE the sample lands off the true ODE solution ⇒ atypical ⇒ far in $L_2$. That's a *sampler* artifact, not an epistemic one. **Control:** repeat with a high-NFE/adaptive solver.
3. **Pixel $L_2$ NN distance is a weak proxy.** Use DINOv2/SSCD feature distance and the $d_1/d_2$ ratio; 4.66 over 3072 dims ($\approx0.08$–$0.18$ per-pixel RMS) is "generic-similar," not duplicate — so the honest reading of your own data is a **typicality** detector, not memorization and not OOD.
4. **Statistics and weighting.** 100-vs-100 tails with no CIs; FID on 100 images is meaningless (use KID or 1k buckets). And $\|A\|_F\sim(1-t)^{-1}$ blows up at the endpoint, so $V(x_0)=\int\|A\|_F^2dt$ may be measuring one narrow $t$-window. Report the $t$-resolved profile and a normalized score.

There's also a **theoretical tension your two-bucket design can't see.** Far from data, the softmax collapses to one-hot, $\mathrm{Cov}_w\to0$, and the ideal Jacobian $\to-I/(1-t)$ — the far tail is where the target is *easiest*. Hardness lives at **basin boundaries** (large $\mathrm{Cov}_w$, steep weight transitions), which are mid-density. So competence-failure predicts a **non-monotonic** curl-vs-distance profile: low near data, peaked at inter-mode boundaries, then rising again only where supervision truly vanishes. Plot the full binned scatter — if you see the hump, you have a mechanism; if you see a monotone line, suspect confound 1.

---

## 4. The upgrade that makes this A*-shaped

Right now you have one certificate. There is a **complete family**, and framing it as *a posteriori error estimation for generative models* (the FEM residual-estimator analogy) is a much stronger paper than "OOD detector."

Given only $v_\theta$, define its implied score $s_\theta=(v_\theta-\frac{\dot\beta}{\beta}x)/c_t$. An exact model must satisfy both the gradient property **and** the continuity equation. Eliminating $\log p_t$ gives a fully self-referential residual:
$$\mathcal R_\theta:=\partial_t s_\theta+\nabla(\nabla\!\cdot\!v_\theta)+\nabla(v_\theta\!\cdot\!s_\theta)\;\overset{!}{=}\;0$$
Certificate 1 ($\|A_\theta\|$) detects the **rotational** error; $\mathcal R_\theta$ detects the **mass/divergence** error that curl is blind to — and recall from E1 that only the non-flux-null part of the error corrupts the sampled distribution, which is exactly what $\mathcal R_\theta$ sees. Evaluate $\mathcal R_\theta$ in weak form against random test functions to avoid third derivatives. A third, nearly free certificate: $\nabla v^\star\succeq-\frac{I}{1-t}$ (since $\mathrm{Cov}_w\succeq0$), so any eigenvalue below that bound certifies error too.

And the experiment that would actually land it: stop scoring the model's own samples (a strange object — they're in-distribution by construction) and score **external inputs**. Noise a real image to $x_t$, average the certificate over $t$ and noise draws, and run the standard OOD suite (CIFAR-10 model → SVHN / CIFAR-100 / Places). Generative-model *likelihoods* are famously terrible at this; an approximation-error certificate is a genuinely different signal and has a real shot at beating them. Mandatory baselines: $\|a\|$ (acceleration — cheaper and also "geometric"), $\|s_\theta\|$, ODE likelihood, ensemble disagreement. If $\|a\|$ matches $\|A\|_F$, the curl framing is decoration.

---

## 5. Next six experiments, ordered

1. **Instrument validation** (~1 h, CPU): positive control with injected rotation; FD estimator vs exact autodiff Jacobian on a 2D MLP; report relative error and the cancellation floor. *Gate: nothing proceeds until this passes.*
2. **The null-model measurement** (~2 h): $\rho(x,t)=\|A_\theta\|_F^2/\|\nabla v_\theta\|_F^2$ vs the $0.5$ isotropic baseline, resolved in $t$, on an **I-CFM** checkpoint. This is the make-or-break number for §6.
3. **Exact error decomposition on CIFAR-10** (~3 h): the empirical $v^\star$ is closed-form and cheap — $\mu$ is a $50\text{k}\times3072$ softmax-weighted matvec on GPU. Compute the true $\nabla v_\theta-\nabla v^\star$ and verify directly how tight the bound $\|A_\theta\|_F\le\|\nabla v_\theta-\nabla v^\star\|_F$ is. This turns the certificate from an inequality into a calibrated estimator.
4. **Confound sweep on §7** (~2 h): scale-free score, energy-matched correlation, high-NFE re-run, full binned scatter, feature-space NN. Expect the effect to shrink; the question is whether it survives.
5. **Curl-vs-training-run controls** (~4 h): curl across checkpoints (must fall with training and capacity if it's approximation error) and across two independently-seeded models (hotspots should *differ* if optimization error, *coincide* if data-geometry-driven). Cleanest causal discriminator you can buy.
6. **$\rho(B)$: minibatch-OT bias meter** (~4 h) and the external-input OOD benchmark.

Want me to patch the seed docs to match — delete H4, split the backbone spec into I-CFM (certificate track) vs OT-CFM (bias-meter track), and rewrite `specs/01_estimators.md` around the four-forward-pass antisymmetric estimator with the positive-control test as a hard gate?