"""Stage 1 — Instrument validation (E0 + E1) for the curl certificate.

Runs, in order, the cheap kills from execution_plan.md sec 4-5:

  E1a  exactness      : FD & JVP+VJP vs full autodiff Jacobian on an MLP.  -> selects estimator (G1a')
  G0   hour-zero      : raw rho on the cfm_ checkpoint. If ~0 the program is dead in an hour.
  E1b  positive ctrl  : inject known rotation, expect hockey-stick + slope-1 recovery.
  E1c  negative ctrl  : estimator on the exact v* at d=3072, must read rho ~ 0.
  E1d  precision floor: h / dtype sweep -> the instrument's numerical floor as a number.
  E1e  cost-fidelity  : cheapest (probes x knots) whose ranking matches a dense reference.

HARD GATES to proceed to Stage 2:  G0, G1b (positive), G1c (negative).

Usage (Kaggle):  python stage1_instrument.py \
    --weights /kaggle/working/otcfm-weights/cfm_cifar10_weights_step_400000.pt
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from curlprobe import estimator as est   # noqa: E402
from curlprobe import models, probe_points, score, vstar  # noqa: E402


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------
def banner(msg: str) -> None:
    print("\n" + "=" * 78 + f"\n{msg}\n" + "=" * 78)


def load_cifar(data_root: str, n: int, device, dtype=torch.float32, with_labels=False):
    """CIFAR-10 train images normalized to [-1, 1]. Returns (imgs (n,3,32,32), flat (n,3072)).

    with_labels=True also returns class labels (n,) as a third element (E2b uses them to pick
    distinct-class image pairs for basin-boundary targets).
    """
    from torchvision import datasets, transforms

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    kaggle_path = "/kaggle/input/datasets/ayush1220/cifar10/cifar10/train"
    if os.path.isdir(kaggle_path):
        ds = datasets.ImageFolder(root=kaggle_path, transform=tf)
    else:
        ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False, num_workers=2)
    imgs, labs, got = [], [], 0
    for xb, yb in loader:
        imgs.append(xb)
        labs.append(yb)
        got += xb.shape[0]
        if got >= n:
            break
    imgs = torch.cat(imgs, 0)[:n].to(device=device, dtype=dtype)
    flat = imgs.reshape(imgs.shape[0], -1)
    if with_labels:
        return imgs, flat, torch.cat(labs, 0)[:n].to(device)
    return imgs, flat


def estimate_over_points(vf, x, method, n_probes, batch=32, gen=None):
    """Run an estimator over x in mini-batches (memory safety for the UNet). Returns rho (B,).

    No per-batch empty_cache(): the caching allocator reuses freed memory, and forcing a
    cache flush every mini-batch serializes the GPU and kills throughput.
    """
    A_all, G_all = [], []
    for s in range(0, x.shape[0], batch):
        xb = x[s : s + batch]
        A, G = est.frobenius_estimates(vf, xb, n_probes=n_probes, method=method, generator=gen)
        A_all.append(A.detach())
        G_all.append(G.detach())
    A = torch.cat(A_all)
    G = torch.cat(G_all)
    return score.rho_from_sq(A, G), A, G


# ----------------------------------------------------------------------------------------
# E1a — exactness on an MLP (also selects the estimator: gate G1a')
# ----------------------------------------------------------------------------------------
def e1a_exactness(device, d=8, B=32, n_probes=4000, seed=0):
    banner("E1a  Exactness  (FD & JVP+VJP vs full autodiff Jacobian)  ->  estimator selection")
    g = torch.Generator(device=device).manual_seed(seed)
    W1 = torch.randn(32, d, generator=g, device=device)
    b1 = torch.randn(32, generator=g, device=device)
    W2 = torch.randn(d, 32, generator=g, device=device)
    b2 = torch.randn(d, generator=g, device=device)

    def vf_single(x):  # (d,) -> (d,)
        return W2 @ torch.tanh(W1 @ x + b1) + b2

    def vf_batch(x):   # (B,d) -> (B,d)
        return torch.tanh(x @ W1.t() + b1) @ W2.t() + b2

    x = torch.randn(B, d, generator=g, device=device)
    A_ex, G_ex = est.frobenius_exact_smalldim(vf_single, x)
    A_jvp, G_jvp = est.frobenius_jvp(vf_batch, x, n_probes=n_probes, generator=g)
    A_fd, G_fd = est.frobenius_fd(vf_batch, x, n_probes=n_probes, generator=g)

    def rel(a, b):
        return float((a.mean() - b.mean()).abs() / b.mean().clamp_min(1e-30))

    r_jvp_A, r_fd_A = rel(A_jvp, A_ex), rel(A_fd, A_ex)
    r_jvp_G, r_fd_G = rel(G_jvp, G_ex), rel(G_fd, G_ex)
    print(f"  exact   mean |A|^2 = {A_ex.mean():.6e}   mean |J|^2 = {G_ex.mean():.6e}")
    print(f"  JVP+VJP rel.err |A|^2 = {r_jvp_A:.2%}   |J|^2 = {r_jvp_G:.2%}")
    print(f"  FD (4fwd) rel.err |A|^2 = {r_fd_A:.2%}   |J|^2 = {r_fd_G:.2%}")

    jvp_ok = max(r_jvp_A, r_jvp_G) < 0.01
    fd_ok = max(r_fd_A, r_fd_G) < 0.01
    method = "jvp" if jvp_ok else ("fd" if fd_ok else "jvp")
    print(f"  G1a exactness: JVP {'PASS' if jvp_ok else 'FAIL'} | FD {'PASS' if fd_ok else 'FAIL'}")
    print(f"  G1a' selected estimator -> '{method}'")
    return method, {"jvp_ok": jvp_ok, "fd_ok": fd_ok}


# ----------------------------------------------------------------------------------------
# G0 — hour-zero raw rho on the pretrained I-CFM checkpoint
# ----------------------------------------------------------------------------------------
def g0_hour_zero(model, data_imgs, device, method, ts=(0.1, 0.3, 0.5), n_pts=100,
                 n_probes=8, batch=32, gen=None):
    banner("G0  Hour-zero  (raw rho on cfm_ checkpoint)  [HARD GATE]")
    x1 = data_imgs[:n_pts]  # image-shaped (B,3,32,32) — the UNet needs 4D input
    out = {}
    for t in ts:
        x_t, _ = probe_points.p1_noised(x1, t, generator=gen)
        vf = models.make_velocity_fn(model, t)
        rho, _, _ = estimate_over_points(vf, x_t, method, n_probes, batch=batch, gen=gen)
        out[t] = float(rho.mean())
        print(f"  t={t:.2f}   mean rho = {out[t]:.4e}   (median {rho.median():.4e})")
    peak = max(out.values())
    passed = peak > 1e-5
    print(f"  G0: peak mean rho = {peak:.4e}  ->  {'PASS' if passed else 'FAIL (program dead)'}")
    return passed, out


# ----------------------------------------------------------------------------------------
# E1b — positive control: inject a known rotation
# ----------------------------------------------------------------------------------------
def e1b_positive_control(model, data_imgs, device, method, t=0.1, n_pts=16, n_probes=8,
                         gen=None):
    banner("E1b  Positive control  (inject epsilon*R, expect hockey-stick + slope-1)  [HARD GATE]")
    x1 = data_imgs[:n_pts]
    x_t, _ = probe_points.p1_noised(x1, t, generator=gen)
    vf = models.make_velocity_fn(model, t)
    R_op, R_fro = est.make_rotation_operator((3, 32, 32), device, dtype=x_t.dtype, seed=1)
    # Sweep well past the knee (which sits at amp ~ ||A_theta||_F / ||R||) so there is a clean
    # asymptotic decade. The model's own floor can be O(1), so go to 1e2.
    amps = torch.logspace(-6, 2, 17).tolist()
    curl_sq = []                                         # mean ||A||_F^2 across points, per amp
    for amp in amps:
        def vf_pert(x, _amp=amp):
            return vf(x) + _amp * R_op(x)
        A, _ = est.frobenius_estimates(vf_pert, x_t, n_probes=n_probes, method=method, generator=gen)
        curl_sq.append(float(A.clamp_min(0).mean()))

    plateau_sq = sum(curl_sq[:4]) / 4.0                  # ||A_theta||_F^2 (model curl floor)
    plateau = math.sqrt(max(plateau_sq, 0.0))
    # Floor subtraction isolates the injected rotation:
    #   ||A_theta + amp R||_F^2 - ||A_theta||_F^2  ~  amp^2 ||R||^2   (cross term ~0 in high-d)
    excess = [math.sqrt(max(c - plateau_sq, 0.0)) for c in curl_sq]
    for amp, c, e in zip(amps, curl_sq, excess):
        print(f"  amp={amp:.1e}   mean |A|_F = {math.sqrt(c):.4e}   excess(inj) = {e:.4e}")

    # Slope + recovery on the asymptotic tail only (amp >> knee).
    tail = [(a, e) for a, e in zip(amps, excess) if a >= 3.0 * max(plateau, 1e-6) and e > 0]
    if len(tail) >= 2:
        la = [math.log(a) for a, _ in tail]
        le = [math.log(e) for _, e in tail]
        n = len(la); mx = sum(la) / n; my = sum(le) / n
        slope = sum((a - mx) * (e - my) for a, e in zip(la, le)) / sum((a - mx) ** 2 for a in la)
        recovered_R = sum(e / a for a, e in tail) / len(tail)  # -> ||R||_F = 1
    else:
        slope = recovered_R = float("nan")
    print(f"  plateau ||A_theta||_F (model curl floor) = {plateau:.4e}   knee ~ amp {plateau/R_fro:.2f}")
    print(f"  asymptotic slope (floor-subtracted) = {slope:.3f}   recovered ||R||_F = {recovered_R:.3f}  (true {R_fro:.1f})")
    passed = (0.85 <= slope <= 1.15) and (0.8 <= recovered_R <= 1.25)
    print(f"  G1b: {'PASS' if passed else 'FAIL'}")
    return passed, {"plateau": plateau, "slope": slope, "recovered_R": recovered_R, "amps": amps,
                    "curl_sq": curl_sq, "excess": excess}


# ----------------------------------------------------------------------------------------
# E1c — negative control: estimator on the exact v* at d=3072
# ----------------------------------------------------------------------------------------
def e1c_negative_control(data_flat, data_imgs, device, method, t=0.1, n_pts=16, n_probes=4,
                         n_data=4096, gen=None):
    banner("E1c  Negative control  (estimator on exact v*, d=3072, must read rho~0)  [HARD GATE]")
    X = data_flat[:n_data].to(torch.float64)             # irrotational for ANY data/N
    x1 = data_imgs[:n_pts]
    x_t, _ = probe_points.p1_noised(x1, t, generator=gen)
    x_t = x_t.to(torch.float64)

    def vf_star(x):
        return vstar.vstar_field(x, X, t)

    try:
        A, G = est.frobenius_estimates(vf_star, x_t, n_probes=n_probes, method=method, generator=gen)
        used = method
    except Exception as e:  # forward-mode may not like some op -> AD-free fallback
        print(f"  ({method} failed on v*: {type(e).__name__}: {e}; falling back to FD)")
        A, G = est.frobenius_fd(vf_star, x_t, n_probes=n_probes, generator=gen)
        used = "fd"
    rho = score.rho_from_sq(A, G)
    val = float(rho.mean())
    thresh = 1e-4 if used == "jvp" else 1e-2             # FD has a higher intrinsic floor
    passed = val < thresh
    print(f"  d={data_flat.shape[1]}  N={n_data}  estimator={used}  float64")
    print(f"  mean rho(v*) = {val:.4e}   (threshold {thresh:.0e})  ->  {'PASS' if passed else 'FAIL'}")
    return passed, {"rho_vstar": val, "used": used}


# ----------------------------------------------------------------------------------------
# E1d — precision floor
# ----------------------------------------------------------------------------------------
def e1d_precision_floor(data_flat, data_imgs, device, t=0.1, n_pts=8, n_data=4096, gen=None):
    banner("E1d  Precision floor  (h / dtype sweep of the AD-free estimator on v*)")
    X64 = data_flat[:n_data].to(torch.float64)
    X32 = data_flat[:n_data].to(torch.float32)
    x1 = data_imgs[:n_pts]
    x_t, _ = probe_points.p1_noised(x1, t, generator=gen)
    best = (None, None, float("inf"))
    for dtype, X in [(torch.float32, X32), (torch.float64, X64)]:
        xd = x_t.to(dtype)
        for h in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
            def vf_star(x):
                return vstar.vstar_field(x, X, t)
            A, G = est.frobenius_fd(vf_star, xd, n_probes=4, h=h, generator=gen)
            rho = float(score.rho_from_sq(A, G).mean())
            print(f"  dtype={str(dtype).split('.')[-1]:>7}  h={h:.0e}  rho(v*)={rho:.3e}")
            if rho < best[2]:
                best = (dtype, h, rho)
    print(f"  instrument floor: dtype={best[0]} h={best[1]:.0e} -> rho={best[2]:.3e}")
    return {"floor_rho": best[2], "floor_dtype": str(best[0]), "floor_h": best[1]}


# ----------------------------------------------------------------------------------------
# E1e — cost / fidelity
# ----------------------------------------------------------------------------------------
def e1e_cost_fidelity(model, data_imgs, device, method, n_pts=32, ref_probes=16, ref_knots=12,
                      batch=32, gen=None):
    banner("E1e  Cost-fidelity  (cheapest probes x knots with rank-corr > 0.95)")

    def path_rho(x1, n_probes, n_knots):
        tks = probe_points.stratified_t(n_knots, device=device)
        rows = []
        for tv in tks.tolist():
            x_t, _ = probe_points.p1_noised(x1, tv, generator=gen)
            vf = models.make_velocity_fn(model, tv)
            rho, _, _ = estimate_over_points(vf, x_t, method, n_probes, batch=batch, gen=gen)
            rows.append(rho)
        rho_tk = torch.stack(rows)                       # (K, B)
        return score.path_score(tks, rho_tk)

    x1 = data_imgs[:n_pts]
    ref = path_rho(x1, ref_probes, ref_knots)
    print(f"  reference: probes={ref_probes} knots={ref_knots}")
    results = {}
    cheapest = None
    for np_ in [2, 4, 8]:
        for nk in [4, 6, 8]:
            cand = path_rho(x1, np_, nk)
            tau = score.spearman(cand, ref)
            cost = np_ * nk
            results[(np_, nk)] = tau
            print(f"  probes={np_} knots={nk}  cost~{cost:>3}  spearman={tau:.3f}")
            if tau > 0.95 and (cheapest is None or cost < cheapest[2]):
                cheapest = (np_, nk, cost, tau)
    if cheapest:
        print(f"  cheapest >0.95: probes={cheapest[0]} knots={cheapest[1]} (cost~{cheapest[2]}, tau={cheapest[3]:.3f})")
    else:
        print("  no cheap config reached 0.95 — use the dense reference")
    return {"cheapest": cheapest, "results": {f"{k[0]}x{k[1]}": v for k, v in results.items()}}


# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/kaggle/working/otcfm-weights/cfm_cifar10_weights_step_400000.pt",
                    help="I-CFM (cfm_) checkpoint — the certificate track")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--n-cifar", type=int, default=4096,
                    help=">= the v* data-matrix size used by E1c/E1d (4096)")
    ap.add_argument("--batch", type=int, default=32,
                    help="UNet mini-batch for the estimator; raise for speed, lower if OOM")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-e1e", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(args.seed)
    print(f"device={device}  torch={torch.__version__}")
    if device.type == "cuda":
        # Tensor-core / cuDNN acceleration for the fp32 UNet forwards (the fp64 v* path,
        # and thus E1c/E1d precision, is unaffected by TF32).
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name}  {props.total_memory/1e9:.1f} GB  |  estimator batch={args.batch}")
    else:
        print("  !! No CUDA detected — G0/E1b/E1e run the UNet and will be SLOW on CPU.")

    gates = {}

    # E1a first — it chooses the estimator used everywhere below.
    method, _ = e1a_exactness(device, seed=args.seed)

    # Load data + model.
    banner("Loading CIFAR-10 (train, [-1,1]) and the I-CFM UNet")
    imgs, flat = load_cifar(args.data_root, args.n_cifar, device)
    print(f"  cifar: {imgs.shape}  range [{imgs.min():.2f}, {imgs.max():.2f}]")
    if not os.path.exists(args.weights):
        print(f"  !! weights not found at {args.weights} — G0/E1b use RANDOM weights (invalid gate).")
    model = models.load_cifar_unet(args.weights, device=device)

    gates["G0"], _ = g0_hour_zero(model, imgs, device, method, batch=args.batch, gen=gen)
    gates["G1b"], _ = e1b_positive_control(model, imgs, device, method, gen=gen)
    gates["G1c"], _ = e1c_negative_control(flat, imgs, device, method, gen=gen)
    e1d_precision_floor(flat, imgs, device, gen=gen)
    if not args.skip_e1e:
        e1e_cost_fidelity(model, imgs, device, method, batch=args.batch, gen=gen)

    banner("STAGE 1 GATE SUMMARY")
    for k in ("G0", "G1b", "G1c"):
        print(f"  {k}: {'PASS' if gates.get(k) else 'FAIL'}")
    all_pass = all(gates.get(k) for k in ("G0", "G1b", "G1c"))
    print("\n  >>> " + ("PROCEED to Stage 2 (E2 calibration)." if all_pass
                        else "STOP — a hard gate failed. Do not scale."))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
