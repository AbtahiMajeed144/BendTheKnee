"""Stage 2b — tau^2 at genuine basin boundaries (the regime E2 never measured).

E2 probed points noised from *single* training images; at large t the posterior collapsed
onto that one image (Cov_w -> 0, v* -> -I/(1-t)), so 7/10 knots had a degenerate target and
the pooled tau^2=0.007 was an artifact. The surviving hypothesis -- curl concentrates at
mid-density basin boundaries (large Cov_w, steep weight transitions) -- was never tested.

E2b forces NON-degenerate targets by building probe points from convex combinations of
distinct-class training images (interpolations / k-mixtures), then sweeps tau^2 INTO the
high-Cov_w regime. Everything else (closed-form v*, calibration estimator) is reused verbatim.

The fork this decides:
  tau^2 stays << 0.5 even at real boundaries  -> implicit conservativity is ROBUST  (fork 1)
  tau^2 climbs toward/past 0.5 at boundaries  -> curl-excess ALIVE but LOCALIZED   (fork 2)

Usage (Kaggle):  python stage2b_basin.py
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from curlprobe import estimator as est          # noqa: E402
from curlprobe import models, probe_points, vstar  # noqa: E402
from curlprobe.score import spearman            # noqa: E402
from stage1_instrument import load_cifar, banner  # noqa: E402
from stage2_calibration import bootstrap_ratio_ci  # noqa: E402

# go/no-go on the top-Cov_w (boundary) tau^2
FORK2_EXCESS = 0.40   # >= this -> rotational excess is alive at boundaries (fork 2)
FORK1_ROBUST = 0.20   # <  this -> conservative even at boundaries (fork 1)


def measure(model, X, data_sqnorm, x_t_all, t, n_probes, batch, gen):
    """Per-point (A, E, Jt, Js, trCov) at fixed t over a set of probe points. Reuses E2 machinery."""
    A_l, E_l, Jt_l, Js_l, tc_l = [], [], [], [], []
    for s in range(0, x_t_all.shape[0], batch):
        xb = x_t_all[s : s + batch]
        xf = xb.reshape(xb.shape[0], -1)
        w, mu = vstar.prepare(xf, X, t)
        tc = vstar.trace_cov_from_prepared(w, mu, data_sqnorm)

        def jstar(eps, _w=w, _mu=mu):
            e = eps.reshape(eps.shape[0], -1)
            return vstar.jvp_prepared(_w, _mu, e, X, t).to(eps.dtype).reshape(eps.shape)

        vf = models.make_velocity_fn(model, t)
        A, E, Jt, Js = est.calibration_estimates(vf, jstar, xb, n_probes=n_probes, generator=gen)
        A_l.append(A.detach()); E_l.append(E.detach())
        Jt_l.append(Jt.detach()); Js_l.append(Js.detach()); tc_l.append(tc.detach())
    return (torch.cat(A_l), torch.cat(E_l), torch.cat(Jt_l), torch.cat(Js_l), torch.cat(tc_l))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/kaggle/working/otcfm-weights/cfm_cifar10_weights_step_400000.pt")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--n-vstar", type=int, default=50000)
    ap.add_argument("--n-pts", type=int, default=256)
    ap.add_argument("--n-probes", type=int, default=16)
    ap.add_argument("--k", type=int, default=2, help="images per target (2 = interpolation)")
    ap.add_argument("--alphas", default="0.0,0.25,0.5", help="k=2 interpolation coeffs (0=single, 0.5=midpoint)")
    ap.add_argument("--ts", default="0.3,0.5,0.7", help="timesteps (mid-large: where E2 was degenerate)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="STAGE2B_RESULTS.md")
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

    alphas = [float(a) for a in args.alphas.split(",")]
    ts = [float(a) for a in args.ts.split(",")]

    banner("Loading CIFAR-10 (train, [-1,1], with labels) and the I-CFM UNet")
    n_load = max(args.n_vstar, args.n_pts * 4)
    imgs, flat, labels = load_cifar(args.data_root, n_load, device, with_labels=True)
    perm = torch.randperm(imgs.shape[0], generator=gen, device=device)
    X = flat[perm[: args.n_vstar]].contiguous()
    data_sqnorm = X.pow(2).sum(dim=1)
    d = X.shape[1]
    null = (d - 1) / (2 * d)
    print(f"  v* data matrix: {tuple(X.shape)}  |  targets: k={args.k}, alphas={alphas}, ts={ts}  |  null tau^2={null:.5f}")
    if not os.path.exists(args.weights):
        print(f"  !! weights not found at {args.weights} — results INVALID (random model).")
    model = models.load_cifar_unet(args.weights, device=device)

    banner("E2b  tau^2 at basin boundaries  (k-image targets -> non-degenerate v*)")
    cells = {}
    ALL = {k: [] for k in ("A", "E", "Jt", "Js", "tc", "alpha", "t")}
    for a in alphas:
        for t in ts:
            m = probe_points.basin_target(imgs, args.n_pts, k=args.k, alpha=a,
                                          labels=labels, different_class=True, generator=gen)
            x_t, _ = probe_points.p1_noised(m, t, generator=gen)
            A, E, Jt, Js, tc = measure(model, X, data_sqnorm, x_t, t, args.n_probes, args.batch, gen)
            tau2 = float(A.sum() / E.sum())
            rho = float((A.sum() / Jt.sum()).sqrt())
            relerr = float((E.sum() / Js.sum()).sqrt())
            cells[(a, t)] = dict(tau2=tau2, rho=rho, relerr=relerr, trcov=float(tc.mean()),
                                 trcov_max=float(tc.max()))
            print(f"  alpha={a:.2f} t={t:.2f}  tau^2={tau2:.4f}  rho={rho:.4f}  relerr={relerr:.4f}  "
                  f"trCov(mean/max)={float(tc.mean()):.2e}/{float(tc.max()):.2e}")
            for name, val in zip(("A", "E", "Jt", "Js", "tc"), (A, E, Jt, Js, tc)):
                ALL[name].append(val)
            ALL["alpha"].append(torch.full_like(tc, a)); ALL["t"].append(torch.full_like(tc, t))

    A = torch.cat(ALL["A"]); E = torch.cat(ALL["E"]); Jt = torch.cat(ALL["Jt"])
    Js = torch.cat(ALL["Js"]); TC = torch.cat(ALL["tc"])
    AL = torch.cat(ALL["alpha"]); TT = torch.cat(ALL["t"])

    # ---- the key sweep: tau^2 resolved in tr Cov_w, INTO the high-structure regime ----
    banner("E2b  tau^2(tr Cov_w) sweep  (does curl concentrate where the target has structure?)")
    edges = torch.quantile(TC.float(), torch.linspace(0, 1, 7, device=TC.device))
    bin_rows = []
    for i in range(6):
        m = (TC >= edges[i]) & (TC <= edges[i + 1] if i == 5 else TC < edges[i + 1])
        if m.sum() < 5:
            continue
        t2 = float(A[m].sum() / E[m].sum())
        blo, bhi = bootstrap_ratio_ci(A[m], E[m], gen=gen)
        r = float((A[m].sum() / Jt[m].sum()).sqrt())
        bin_rows.append((float(edges[i]), float(edges[i + 1]), int(m.sum()), t2, blo, bhi, r))
        print(f"  Cov_w[{edges[i]:.2e},{edges[i+1]:.2e}]  n={int(m.sum()):4d}  tau^2={t2:.4f}  "
              f"CI[{blo:.4f},{bhi:.4f}]  rho={r:.4f}")

    # boundary estimate = top Cov_w quartile
    thr = torch.quantile(TC.float(), 0.75)
    bnd = TC >= thr
    tau2_bnd = float(A[bnd].sum() / E[bnd].sum())
    bnd_lo, bnd_hi = bootstrap_ratio_ci(A[bnd], E[bnd], gen=gen)
    sp = spearman(TC, A / E.clamp_min(1e-30))
    covmax = float(TC.max())

    if bnd_lo >= FORK2_EXCESS:
        verdict = f"FORK 2 — ROTATIONAL EXCESS AT BOUNDARIES (curl alive, localized). tau^2_boundary CI >= {FORK2_EXCESS}"
    elif tau2_bnd < FORK1_ROBUST:
        verdict = f"FORK 1 — CONSERVATIVE EVEN AT BOUNDARIES (implicit conservativity robust). tau^2_boundary < {FORK1_ROBUST}"
    else:
        verdict = "INTERMEDIATE — partial localization; curl elevated at boundaries but sub-isotropic"

    banner("E2b  RESULT")
    print(f"  boundary tau^2 (top-Cov_w quartile) = {tau2_bnd:.4f}   95% CI [{bnd_lo:.4f}, {bnd_hi:.4f}]   (null {null:.5f})")
    print(f"  Spearman(tr Cov_w, per-point tau^2) = {sp:.3f}")
    print(f"  max tr Cov_w reached = {covmax:.2e}   (E2 single-image runs: ~5.7e2)")
    print(f"  VERDICT: {verdict}")

    import numpy as np
    npz = os.path.splitext(args.out)[0] + "_raw.npz"
    np.savez(npz, A=A.cpu().numpy(), E=E.cpu().numpy(), Jt=Jt.cpu().numpy(), Js=Js.cpu().numpy(),
             trcov=TC.cpu().numpy(), alpha=AL.cpu().numpy(), t=TT.cpu().numpy())
    write_results(args, d, null, alphas, ts, cells, bin_rows, tau2_bnd, bnd_lo, bnd_hi, sp, covmax, verdict)
    print(f"  wrote {npz}")
    return 0


def write_results(args, d, null, alphas, ts, cells, bin_rows, tau2_bnd, bnd_lo, bnd_hi, sp, covmax, verdict):
    cell_tbl = "\n".join(
        f"| {a:.2f} | {t:.2f} | {cells[(a,t)]['tau2']:.4f} | {cells[(a,t)]['rho']:.4f} | "
        f"{cells[(a,t)]['relerr']:.4f} | {cells[(a,t)]['trcov']:.2e} | {cells[(a,t)]['trcov_max']:.2e} |"
        for a in alphas for t in ts
    )
    bin_tbl = "\n".join(
        f"| [{lo:.2e}, {hi:.2e}] | {n} | {t2:.4f} | [{blo:.4f}, {bhi:.4f}] | {r:.4f} |"
        for (lo, hi, n, t2, blo, bhi, r) in bin_rows
    )
    md = f"""# Stage 2b Results — tau^2 at basin boundaries (E2b)

**Model:** `{os.path.basename(args.weights)}` (I-CFM) · **d={d}** · null tau^2 = {null:.5f}
**Setup:** {args.n_pts} points/config, k={args.k}-image distinct-class targets, alphas={alphas},
ts={ts}, v* over {args.n_vstar} training pts, {args.n_probes} probes, seed {args.seed}.

## Why E2b

E2's pooled tau^2=0.007 was an artifact: 7/10 t-knots had a degenerate target (Cov_w~0, v*->-I),
so legitimate generalization counted as symmetric error. The surviving hypothesis -- curl
concentrates at basin boundaries (high Cov_w) -- was never tested. E2b builds non-degenerate
targets from convex combinations of distinct-class images and sweeps tau^2 into that regime.

## Headline

- **boundary tau^2 (top Cov_w quartile) = {tau2_bnd:.4f}**, 95% CI **[{bnd_lo:.4f}, {bnd_hi:.4f}]** (null {null:.5f})
- Spearman(tr Cov_w, tau^2) = {sp:.3f}
- max tr Cov_w reached = {covmax:.2e}  (vs E2 single-image ~5.7e2)
- **VERDICT: {verdict}**

## tau^2(tr Cov_w) sweep

| tr Cov_w bin | n | tau^2 | 95% CI | rho |
|---|---|---|---|---|
{bin_tbl}

## Per (alpha, t) — alpha=0 is the single-image degenerate control

| alpha | t | tau^2 | rho | rel.err | tr Cov_w (mean) | tr Cov_w (max) |
|---|---|---|---|---|---|---|
{cell_tbl}

## Reading

- **alpha=0** reproduces E2 (single-image, degenerate at large t). **alpha=0.5** is the midpoint
  between two distinct-class images -- a genuine basin boundary with large Cov_w.
- If tau^2 climbs toward 0.5 as Cov_w grows -> curl is a **localized boundary** phenomenon
  (fork 2, sharper mechanism paper). If it stays << 0.5 -> **implicit conservativity is robust**
  (fork 1, clean scientific-finding paper). Thresholds: fork2 if boundary CI >= {FORK2_EXCESS};
  fork1 if boundary tau^2 < {FORK1_ROBUST}.

Raw per-point arrays in `{os.path.splitext(args.out)[0]}_raw.npz`.
"""
    with open(args.out, "w") as f:
        f.write(md)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
