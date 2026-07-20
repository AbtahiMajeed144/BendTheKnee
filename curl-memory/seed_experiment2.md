This is the most important decision point in the project so far, and the data is genuinely decisive in one direction and genuinely ambiguous in another. Let me separate those cleanly, because the easy reading of E2 would send you down the wrong path.

## The honest ledger of what Stages 1–2 bought you

**Bankable, not in question:** the instrument is real. E1c reading 2e-15 on the exact irrotational field at d=3072, against a model signal of 5e-2, is a 13-order-of-magnitude null/signal separation — that's a cleaner instrument-validation than most published detectors ever show. E1b recovers injected rotation to 4%. Stage 1 is a paper-grade methods section on its own. Keep it.

**Falsified, and correctly so:** the §6 thesis — "architecture fails → network spins out into rotational garbage" — predicted τ² > 0.5. You measured ≤ 0.11 everywhere. The global rotational-excess story is dead. Good; you killed it in an afternoon instead of in review.

But now the part that matters for the decision, and that the write-up half-sees but under-weights:

## E2 measured the wrong points. The headline 0.007 is an artifact of probe construction.

Your own methodological finding says it: **7 of 10 t-knots had Cov_w ≈ 0** — degenerate targets where v\* has collapsed to −I/(1−t), pure symmetric, so the network's legitimate generalization gets counted as "symmetric error" and τ² is mechanically deflated. The pooled 0.0073 is dominated by points where the measurement is meaningless.

Look at what happens when you strip that:

- Cleanest, least-degenerate point (t=0.068, Cov_w=573): **τ² = 0.11**, not 0.007. Sixteen times higher.
- Highest-Cov_w quartile: τ² = 0.0124, elevated above the other three (0.0072–0.0074).
- Spearman(Cov_w, τ²) = +0.12, positive.

Every one of those says the same thing: **τ² rises with target structure, and your probes systematically avoided the high-structure regime.** And that regime — mid-density basin boundaries, large Cov_w, steep weight transitions — is *exactly* where the original theoretical argument (two critiques ago) predicted curl should concentrate. So the pre-registered basin-boundary hypothesis wasn't tested and failed; **it wasn't tested at all.** P1 points noised from single training images can't reach basin boundaries by construction — they're spread-at-small-t or collapsed-at-large-t, never in between.

So the correct statement is not "the network is implicitly conservative." It's: "the network is symmetric-dominated in the low-and-mid-density regime we could cleanly probe (τ² ≤ 0.11), and the one regime where the surviving hypothesis lives is unmeasured." Those are very different conclusions, and only the second is supported.

## What is genuinely damaged — say it plainly

Two things took real hits, and you should not talk yourself out of them:

**The A\*-via-curl-correction lever is probably gone.** If curl is ~11% of the Jacobian error even at the cleanest point (and <1% pooled), you cannot fix FID by projecting it out — you'd be correcting a tenth of the problem while ignoring the symmetric 89%. The "training-free intervention lowers FID at fixed NFE" dream, which was your one clean A\* path last turn, is now unlikely unless that 11% is disproportionately quality-critical (a stretch you shouldn't bet the paper on).

**The strong-OOD-detector-via-curl lever is weakened twice over.** Curl is a small residual, *and* E1e already told you its ranking is variance-limited at M≤16 probes (best Spearman 0.77, never reached 0.95). A detector whose ranking is noisy and whose signal is a sub-11% sliver, competing against EigenScore which uses the dominant symmetric part — that's an uphill AUROC fight you flagged as losable, now with worse cards.

This is the second time the headline thesis has shrunk on contact with data: memorization → OOD → "actually mostly conservative." That pattern is itself information. The phenomenon is real but keeps being smaller than hoped. Let the data lead from here, not the narrative.

## The decision: one cheap experiment gates everything. Don't pick a paper yet.

Do **not** write the "implicitly conservative" paper now — it's premature on the exact point that decides it. Run the measurement E2 skipped:

**E2b — τ² at genuine basin boundaries (this week, reuses all existing machinery).** Construct non-degenerate targets so Cov_w doesn't collapse: (a) noise from *2–k image mixtures* instead of single images, or (b) probe at interpolations/midpoints between distinct training clusters, or (c) a KDE-smoothed v\* with bandwidth chosen so the softmax stays soft. Sweep τ²(Cov_w) *into* the high-structure regime the current run never entered. This is a matvec-level change to code you already have.

**In parallel — the 2D basin-boundary picture (an afternoon).** Train a small MLP on a 2-cluster mixture, plot ρ(x) or τ²(x) spatially. Does curl concentrate on the boundary between basins? This is the original "ring of fire" idea, finally aimed at the right regime, and it builds the intuition that tells you what E2b should show. If curl visibly peaks at the inter-basin ridge in 2D, that's both your mechanism figure and your prediction for CIFAR.

The fork these resolve:

| E2b outcome | What you have | Paper |
|---|---|---|
| τ² stays ≪ 0.5 even at real basin boundaries | Implicit conservativity is **robust** | Scientific-finding paper: "trained flow nets preserve gradient structure; curl is a certified-but-loose error floor." Honest, clean, mid-tier main-conference. |
| τ² climbs toward/past 0.5 at boundaries | Curl-excess is **alive but localized** | Sharper mechanism paper: "flow-field error becomes rotational specifically at inter-basin boundaries" — a targeted detector for the regime that matters. Stronger. |

Either way you have a defensible paper; you just don't yet know which, and the difference is worth one week.

## The sleeper asset to foreground regardless of the fork

E2 quietly demonstrated something more valuable than the curl verdict: **you can compute the exact, ground-truth Jacobian error E = ∇v_θ − ∇v\* on a real image model**, because the I-CFM empirical target is closed-form. SCOPED and EigenScore cannot do this — they work with the model's own Jacobian as a heuristic because they have no ground truth. You have ground truth on CIFAR. "Relative Jacobian error = 0.39, and here is its symmetric/antisymmetric decomposition, resolved in t and density" is a measurement nobody has published, and it's a methodological contribution that survives whatever curl turns out to be. Whichever paper you write, this is a pillar of it, not a footnote.

## Recalibrated ceiling, unbiased

With this data in hand, the realistic ceiling has stepped down from "potential A\*" to "solid, honest main-conference paper about flow-field geometry, with a bankable instrument and a novel exact-error methodology." A\* would now require E2b to surprise upward (curl genuinely dominates at boundaries → a real localized detector) or the exact-error tool to open a consequence you haven't found yet. That's not a failure — a clean, correct, well-instrumented geometry paper that refines the Curly-FM / Energy-Matching literature's "networks acquire spurious curl" claim into a *quantified, regime-resolved* statement is a genuine contribution. It's just not the splashy detector you started toward, and pretending otherwise would be the thing that gets you rejected.

The one-line decision: **don't commit to "implicitly conservative" — that reading rests on points where the measurement is degenerate. Run E2b + the 2D picture this week to measure the one regime you missed, and let that number pick which of the two real papers you write.**

Want me to spec E2b concretely — the mixture/interpolation probe construction, the non-degenerate target, and the exact τ²(Cov_w) sweep with the go/no-go thresholds for each fork?