"""Stage 2 — Calibration (E2): the tau^2 measurement.

The crown-jewel asset: on CIFAR the exact I-CFM target v* is closed-form, so we can compute
the true Jacobian error E = grad v_theta - grad v* and ask what fraction of it is rotational:

    tau^2 = ||A_theta||_F^2 / ||E||_F^2 = antisymmetric fraction of the Jacobian error.

Because asym(grad v*) = 0 exactly, A_theta = asym(E), so this is well defined. The null (an
error matrix with i.i.d. entries) is (d-1)/(2d) = 0.49984 at d=3072. Three pre-registered
readouts:

    tau^2 ~ 0.5   -> error is isotropic; certificate captures ~half of it (clean, sufficient).
    tau^2 >> 0.5  -> rotational excess; the "loss controls L^2 not H^1" thesis survives.
    tau^2 << 0.5  -> networks are implicitly conservative; a different (better) paper.

Evaluated on P1 (noised real data), resolved in t and in tr Cov_w, with bootstrap CIs.

Usage (Kaggle):  python stage2_calibration.py \
    --weights /kaggle/working/otcfm-weights/cfm_cifar10_weights_step_400000.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from curlprobe import estimator as est          # noqa: E402
from curlprobe import models, probe_points, vstar  # noqa: E402
from stage1_instrument import load_cifar, banner  # reuse loaders  # noqa: E402

NULL_TAU2 = None  # set once d is known: (d-1)/(2d)


def bootstrap_ratio_ci(num, den, n_boot=2000, gen=None, lo=2.5, hi=97.5):
    """Bootstrap CI for the pooled ratio sum(num)/sum(den) by resampling points."""
    B = num.shape[0]
    idx = torch.randint(0, B, (n_boot, B), generator=gen, device=num.device)
    ratios = num[idx].sum(dim=1) / den[idx].sum(dim=1).clamp_min(1e-30)
    qs = torch.quantile(ratios, torch.tensor([lo / 100, hi / 100], device=num.device))
    return float(qs[0]), float(qs[1])


def readout(lo, hi):
    if lo > 0.55:
        return "ROTATIONAL EXCESS (thesis survives: loss controls L^2 not H^1)"
    if hi < 0.45:
        return "IMPLICITLY CONSERVATIVE networks (a different, stronger paper)"
    return "ISOTROPIC error (certificate captures ~half; clean and sufficient)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/kaggle/working/otcfm-weights/cfm_cifar10_weights_step_400000.pt")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--n-vstar", type=int, default=50000, help="training points in the v* data matrix")
    ap.add_argument("--n-pts", type=int, default=256, help="P1 probe points")
    ap.add_argument("--n-probes", type=int, default=16)
    ap.add_argument("--n-knots", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="STAGE2_RESULTS.md")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(args.seed)
    print(f"device={device}  torch={torch.__version__}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU: {p.name}  {p.total_memory/1e9:.1f} GB  |  batch={args.batch}")

    banner("Loading CIFAR-10 (train, [-1,1]) and the I-CFM UNet")
    n_load = max(args.n_vstar, args.n_pts)
    imgs, flat = load_cifar(args.data_root, n_load, device)
    perm = torch.randperm(imgs.shape[0], generator=gen, device=device)
    X = flat[perm[: args.n_vstar]].contiguous()                 # data matrix (N, 3072), fp32
    data_sqnorm = X.pow(2).sum(dim=1)                            # (N,)
    p1_src = imgs[perm[: args.n_pts]].contiguous()              # P1 source images (subset of X)
    d = X.shape[1]
    global NULL_TAU2
    NULL_TAU2 = (d - 1) / (2 * d)
    print(f"  v* data matrix: {tuple(X.shape)}  |  P1 points: {args.n_pts}  |  null tau^2={NULL_TAU2:.5f}")
    if not os.path.exists(args.weights):
        print(f"  !! weights not found at {args.weights} — results INVALID (random model).")
    model = models.load_cifar_unet(args.weights, device=device)

    t_knots = probe_points.stratified_t(args.n_knots, device=device).tolist()

    # per-(t, point) accumulators
    per_t = {}
    all_A, all_E, all_Jt, all_Js = [], [], [], []
    banner("E2  measuring tau^2 = ||A_theta||^2 / ||E||^2  (E = grad v_theta - grad v*)")
    for t in t_knots:
        A_l, E_l, Jt_l, Js_l, tc_l = [], [], [], [], []
        x_t_all, _ = probe_points.p1_noised(p1_src, t, generator=gen)
        for s in range(0, x_t_all.shape[0], args.batch):
            xb = x_t_all[s : s + args.batch]
            xf = xb.reshape(xb.shape[0], -1)
            w, mu = vstar.prepare(xf, X, t)
            tc = vstar.trace_cov_from_prepared(w, mu, data_sqnorm)

            def jstar(eps, _w=w, _mu=mu):
                e = eps.reshape(eps.shape[0], -1)
                return vstar.jvp_prepared(_w, _mu, e, X, t).to(eps.dtype).reshape(eps.shape)

            vf = models.make_velocity_fn(model, t)
            A, E, Jt, Js = est.calibration_estimates(vf, jstar, xb, n_probes=args.n_probes, generator=gen)
            A_l.append(A.detach()); E_l.append(E.detach())
            Jt_l.append(Jt.detach()); Js_l.append(Js.detach()); tc_l.append(tc.detach())
        A = torch.cat(A_l); E = torch.cat(E_l); Jt = torch.cat(Jt_l); Js = torch.cat(Js_l)
        tc = torch.cat(tc_l)
        tau2 = float(A.sum() / E.sum())
        lo, hi = bootstrap_ratio_ci(A, E, gen=gen)
        rho = float((A.sum() / Jt.sum()).sqrt())
        relerr = float((E.sum() / Js.sum()).sqrt())
        per_t[t] = dict(tau2=tau2, lo=lo, hi=hi, rho=rho, relerr=relerr, trcov=float(tc.mean()))
        print(f"  t={t:.3f}  tau^2={tau2:.4f}  CI[{lo:.4f},{hi:.4f}]  rho={rho:.4f}  relerr={relerr:.4f}")
        all_A.append(A); all_E.append(E); all_Jt.append(Jt); all_Js.append(Js)

    A = torch.cat(all_A); E = torch.cat(all_E); Jt = torch.cat(all_Jt); Js = torch.cat(all_Js)
    tau2 = float(A.sum() / E.sum())
    lo, hi = bootstrap_ratio_ci(A, E, gen=gen)
    verdict = readout(lo, hi)

    banner("E2  RESULT")
    print(f"  pooled tau^2 = {tau2:.4f}   95% CI [{lo:.4f}, {hi:.4f}]   (null {NULL_TAU2:.5f})")
    print(f"  overall rho = {float((A.sum()/Jt.sum()).sqrt()):.4f}   relerr = {float((E.sum()/Js.sum()).sqrt()):.4f}")
    print(f"  READOUT: {verdict}")

    write_results(args, d, t_knots, per_t, tau2, lo, hi, verdict, A, E, Jt, Js)
    return 0


def write_results(args, d, t_knots, per_t, tau2, lo, hi, verdict, A, E, Jt, Js):
    rows = "\n".join(
        f"| {t:.3f} | {per_t[t]['tau2']:.4f} | [{per_t[t]['lo']:.4f}, {per_t[t]['hi']:.4f}] | "
        f"{per_t[t]['rho']:.4f} | {per_t[t]['relerr']:.4f} | {per_t[t]['trcov']:.3e} |"
        for t in t_knots
    )
    md = f"""# Stage 2 Results — Calibration (E2)

**Model:** `{os.path.basename(args.weights)}` (I-CFM) · **d={d}** · null tau^2 = {NULL_TAU2:.5f}
**Setup:** {args.n_pts} P1 points (noised real data), v* over {args.n_vstar} training points,
{args.n_probes} probes, {args.n_knots} t-knots, seed {args.seed}.

## Headline

- **pooled tau^2 = {tau2:.4f}**, 95% CI **[{lo:.4f}, {hi:.4f}]** (null {NULL_TAU2:.5f})
- overall rho = {float((A.sum()/Jt.sum()).sqrt()):.4f}, relative Jacobian error = {float((E.sum()/Js.sum()).sqrt()):.4f}
- **READOUT: {verdict}**

`tau^2` is the fraction of the *exact* Jacobian error `E = grad v_theta - grad v*` that is
rotational. It is measured with no ground-truth label — v* is closed-form on CIFAR.

## Resolved in t

| t | tau^2 | 95% CI | rho | rel. err | tr Cov_w |
|---|---|---|---|---|---|
{rows}

## Reading

- **rho** is the certified lower bound on relative Jacobian error (the headline of the whole
  program); **rel. err** = ||E||/||J*|| is the *actual* relative error, known here because v*
  is exact. `rho <= relerr` always (the certificate is one-sided).
- **tau^2** places the curl within the error: above 0.5 means the error is rotationally biased
  (curl sees more than its isotropic share); ~0.5 means isotropic; below means conservative.
"""
    with open(args.out, "w") as f:
        f.write(md)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
