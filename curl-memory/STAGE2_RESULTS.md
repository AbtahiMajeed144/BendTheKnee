# Stage 2 Results — Calibration (E2)

**Model:** `cfm_cifar10_weights_step_400000.pt` (I-CFM) · **d=3072** · null tau^2 = 0.49984
**Setup:** 256 P1 points (noised real data), v* over 50000 training points,
16 probes, 10 t-knots, seed 0.

## Headline

- **pooled tau^2 = 0.0073**, 95% CI **[0.0070, 0.0077]** (null 0.49984)
- overall rho = 0.0385, relative Jacobian error = 0.3896
- **READOUT: IMPLICITLY CONSERVATIVE networks (a different, stronger paper)**

`tau^2` is the fraction of the *exact* Jacobian error `E = grad v_theta - grad v*` that is
rotational. It is measured with no ground-truth label — v* is closed-form on CIFAR.

## Resolved in t

| t | tau^2 | 95% CI | rho | rel. err | tr Cov_w |
|---|---|---|---|---|---|
| 0.068 | 0.1106 | [0.0941, 0.1281] | 0.0121 | 0.0363 | 5.731e+02 |
| 0.164 | 0.0085 | [0.0066, 0.0109] | 0.0345 | 0.3556 | 2.215e+02 |
| 0.260 | 0.0430 | [0.0291, 0.0604] | 0.0626 | 0.2976 | 1.033e+01 |
| 0.356 | 0.0522 | [0.0472, 0.0570] | 0.0581 | 0.2514 | 3.510e-01 |
| 0.452 | 0.0512 | [0.0466, 0.0565] | 0.0566 | 0.2452 | -2.957e-06 |
| 0.548 | 0.0376 | [0.0336, 0.0430] | 0.0493 | 0.2465 | -2.957e-06 |
| 0.644 | 0.0238 | [0.0227, 0.0249] | 0.0405 | 0.2516 | -2.957e-06 |
| 0.740 | 0.0161 | [0.0153, 0.0170] | 0.0367 | 0.2726 | -2.957e-06 |
| 0.836 | 0.0104 | [0.0100, 0.0109] | 0.0353 | 0.3157 | -2.957e-06 |
| 0.932 | 0.0058 | [0.0055, 0.0060] | 0.0378 | 0.4186 | -2.957e-06 |

## Reading

- **rho** is the certified lower bound on relative Jacobian error (the headline of the whole
  program); **rel. err** = ||E||/||J*|| is the *actual* relative error, known here because v*
  is exact. `rho <= relerr` always (the certificate is one-sided).
- **tau^2** places the curl within the error: above 0.5 means the error is rotationally biased
  (curl sees more than its isotropic share); ~0.5 means isotropic; below means conservative.
