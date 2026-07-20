# Stage 1 — Instrument Validation (Kaggle runbook)

Implements E0 + E1 from `execution_plan.md`. Purpose: the cheapest possible test of whether
the whole program is real. Runs in minutes on a single T4. **Proceed to Stage 2 only if the
three hard gates pass: G0, G1b, G1c.**

## What it does

| Step | Test | Gate |
|---|---|---|
| E1a | FD & JVP+VJP vs exact autodiff Jacobian on an MLP → **selects the estimator (G1a′)** | rel err < 1% |
| G0 | raw ρ on the **I-CFM** checkpoint, 3 timesteps, 100 noised real points | **ρ > 1e-5** or STOP |
| E1b | inject known rotation εR → hockey-stick + slope-1 recovery; plateau = model curl floor | **slope∈[0.9,1.1], ‖R‖ rec∈[0.8,1.25]** |
| E1c | estimator on the exact `v*` at d=3072 → must read ρ≈0 | **ρ < 1e-4 (jvp)** |
| E1d | h / dtype sweep → the instrument's numerical floor as a number | (reported) |
| E1e | cheapest probes×knots whose ranking matches a dense reference | Spearman > 0.95 |

## Setup (Kaggle notebook, GPU T4 enabled)

```bash
!pip install torchcfm torchvision tqdm
```

**Download the I-CFM checkpoint — this is the certificate track. Do NOT use `otcfm_`.**
```bash
!mkdir -p /kaggle/working/otcfm-weights
# cfm_ = ConditionalFlowMatcher = independent coupling = provably irrotational target
!wget -O /kaggle/working/otcfm-weights/cfm_cifar10_weights_step_400000.pt \
  https://github.com/atong01/conditional-flow-matching/releases/download/1.0.4/cfm_cifar10_weights_step_400000.pt
```

Get this repo's `curl-memory/` into the notebook (clone your repo, or upload the
`curlprobe/` folder + `stage1_instrument.py`), then:

```bash
!cd /kaggle/working/curl-memory && python stage1_instrument.py \
  --weights /kaggle/working/otcfm-weights/cfm_cifar10_weights_step_400000.pt
```

CIFAR-10 auto-downloads via torchvision (enable notebook internet), or is read from
`/kaggle/input/.../cifar10/train` if that dataset is attached.

## Reading the output

- **G0 FAIL (ρ ≲ 1e-5):** the pretrained model has no measurable curl — the program is dead;
  stop here (you saved a month). Expected healthy range: ρ ~ 1e-2 to 1e-1.
- **G1c must be far below G0.** The instrument reads ~0 on the exact irrotational field but a
  real ρ on the model — that gap is the whole thesis. (Locally, JVP hits ρ(v*) ≈ 2e-15 at
  d=3072.)
- **E1b knee** locates the model's own curl floor independently of G0 — they should agree in
  order of magnitude.
- **G1a′** prints which estimator was selected (`jvp` normally; `fd` only if forward-mode AD
  breaks on an op). Everything downstream uses that choice.

The final `STAGE 1 GATE SUMMARY` prints PASS/FAIL for G0/G1b/G1c and a PROCEED/STOP verdict.

## Notes / knobs

- `--n-cifar N` (default 256) images loaded. `--skip-e1e` to skip the cost-fidelity sweep
  (the heaviest, least-critical step). `--seed`.
- Memory: the UNet estimator runs in mini-batches of 16 with `empty_cache`; jvp+vjp is ≈3×
  forward memory. If OOM, lower the batch in `estimate_over_points`.
- E1c/E1d default to 4096 training points for `v*` (dimension d=3072 is what the claim needs,
  not N=50k). Raise via the function args for the full-set check.

## Locally verified before shipping (no checkpoint needed)

`torch 2.8 / CUDA`, env `loope`: E1a JVP 0.26% / FD 0.37% error; ρ(v*)=2e-15 (JVP, d=3072);
analytic `vstar_jvp` vs autodiff = 5e-15; rotation recovered ‖R‖≈0.94–1.0 over four decades.
Only the UNet-dependent steps (G0, E1b on the real model) require Kaggle + `torchcfm`.
```
