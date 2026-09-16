"""P1-E1 aggregation: probe accuracy profiles, and the cross-stage cosines that decide it.

    python aggregate_probe.py

Two panels:
  (1) probe accuracy vs layer, per stage, with the token-length floor and chance marked.
      The SHAPE matters more than the peak — flat-high from layer 0 means the probe reads
      surface lexicon, not harmfulness.
  (2) cos(base direction, aligned direction) vs layer, against a shuffled-label null band.
      This is the panel that decides the hypothesis.

Verdict logic is printed explicitly so the reading is not left to impression.
"""

from __future__ import annotations

import argparse

import glob
import logging

import numpy as np

from config import config_for
from probes import layer_cosines
from runlog import RunRecord

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("P1-E1-agg")

EXPERIMENT = "P1-E1"


def _load(cfg):
    found = {}
    for p in glob.glob(f"{cfg.results_dir}/{cfg.lineage}_*_probe.npz"):
        d = np.load(p, allow_pickle=True)
        found[str(d["stage"])] = d
    return found


def cosine_vs_null(a_dirs, b_dirs, a_nulls, b_nulls):
    """(true_cos[n_layers], null_p95[n_layers]) — layer-matched cosine, best over positions,
    against a null that went through the IDENTICAL best-over-positions selection.

    That last part is load-bearing. Taking the max |cos| over 5 positions for the true value
    while drawing the null at a single position inflates significance: a max over 5 samples
    is systematically larger than one sample. Applying the same max to each null draw is what
    makes "above the null band" mean something."""
    true = layer_cosines(a_dirs, b_dirs)                      # (n_pos, n_layers)
    sel = np.abs(true).argmax(axis=0)                         # best position per layer
    true_best = true[sel, np.arange(true.shape[1])]           # signed, at that position

    null = layer_cosines(a_nulls.astype(np.float32),
                         b_nulls.astype(np.float32))          # (k, n_pos, n_layers)
    null_best = np.abs(null).max(axis=1)                      # (k, n_layers) — SAME selection
    return true_best, np.percentile(null_best, 95, axis=0)


def main(cfg_override=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", default="zephyr",
                    help="model family from config.LINEAGES")
    args = ap.parse_args()
    cfg = cfg_override or config_for(args.lineage)
    found = _load(cfg)
    stages = [s for s in cfg.stages if s in found]   # lineage's own order
    if not stages:
        raise SystemExit("no probe results — run probe_representation.py first")

    with RunRecord(EXPERIMENT, "aggregate_probe.py", cfg,
                   question="Same axis as the aligned model, or a different one?") as rec:
        acc = {s: np.nanmax(found[s]["acc_logistic"], axis=0) for s in stages}
        acc_mm = {s: np.nanmax(found[s]["acc_mass_mean"], axis=0) for s in stages}
        n_layers = len(acc[stages[0]])

        for s in stages:
            lr, mm = acc[s], acc_mm[s]
            lenb = float(found[s]["length_baseline"])
            logger.info("[%s] logistic peak=%.3f @L%d | L0=%.3f | mass-mean peak=%.3f @L%d "
                        "| length-only=%.3f", s, lr.max(), int(lr.argmax()), lr[0],
                        mm.max(), int(mm.argmax()), lenb)
            rec.result(stage=s, logistic_peak=round(float(lr.max()), 4),
                       logistic_L0=round(float(lr[0]), 4),
                       mass_mean_peak=round(float(mm.max()), 4),
                       length_only=round(lenb, 4),
                       surface_suspect=bool(lr[0] >= lr.max() - 0.02))

        cos, nullhi = {}, {}
        if "base" in found:
            n_pos_base = found["base"]["directions"].shape[0]
            for s in stages:
                if s == "base":
                    continue
                # A stage on a regime override reads a DIFFERENT window over a DIFFERENT
                # template (olmo2 base: 2 positions of 'User: ...\nAssistant:', vs 5 of the
                # chat template). Position i then denotes different things in the two stacks,
                # so a position-matched cosine is not merely shape-incompatible -- it is
                # undefined. Truncating to the shorter stack would silently produce a number.
                n_pos_s = found[s]["directions"].shape[0]
                if n_pos_s != n_pos_base:
                    logger.warning(
                        "[base vs %s] SKIPPED: base has %d eoi positions and %s has %d, "
                        "because base is on a regime override (different template and "
                        "window). Position-matched cosine is undefined across that boundary; "
                        "the transplant test (P1-E1b) is the comparison that survives it.",
                        s, n_pos_base, s, n_pos_s)
                    rec.result(stage=f"base_vs_{s}", cosine="SKIPPED_REGIME_OVERRIDE",
                               n_pos_base=int(n_pos_base), n_pos_other=int(n_pos_s))
                    continue
                c, nh = cosine_vs_null(found["base"]["directions"], found[s]["directions"],
                                       found["base"]["null_directions"],
                                       found[s]["null_directions"])
                cos[s], nullhi[s] = c, nh
                band = np.where(np.abs(c) > nh)[0]
                logger.info("[base vs %s] max|cos|=%.3f @L%d | null p95 there=%.3f | "
                            "layers above null: %d/%d", s, float(np.abs(c).max()),
                            int(np.abs(c).argmax()), float(nh[int(np.abs(c).argmax())]),
                            len(band), n_layers)
                rec.result(stage=f"base_vs_{s}", max_abs_cos=round(float(np.abs(c).max()), 4),
                           at_layer=int(np.abs(c).argmax()),
                           null_p95_there=round(float(nh[int(np.abs(c).argmax())]), 4),
                           layers_above_null=int(len(band)))

        _verdict(acc, acc_mm, cos, nullhi, found, stages)
        _figure(cfg, acc, acc_mm, cos, nullhi, found, stages, n_layers)


def _verdict(acc, acc_mm, cos, nullhi, found, stages) -> None:
    """State the reading explicitly rather than leaving it to impression."""
    if "base" not in found:
        return
    lr = acc["base"]
    readable = lr.max() >= 0.90
    surface = lr[0] >= lr.max() - 0.02
    lenb = float(found["base"]["length_baseline"])
    # only stages whose cosine was actually computed: a regime override skips some
    aligned = [s for s in stages if s != "base" and s in cos]
    # A null p95 near 1.0 means ANY two mean-diff-like vectors in this space are highly
    # cosine-similar, so "signal < null" says nothing about axes. Residual streams are
    # strongly anisotropic (a few massive dimensions dominate), which inflates every
    # cosine. Detect that and report UNRESOLVED rather than a confident non-result.
    # Observed 2026-09-13: null p95 = 0.94 at L1, tracking the signal curve at every layer.
    saturated = bool(cos) and max(float(np.nanmedian(nullhi[s])) for s in aligned) > 0.25
    same_axis = (bool(cos) and not saturated
                 and any(np.abs(cos[s]).max() > nullhi[s][int(np.abs(cos[s]).argmax())] * 2
                         for s in aligned))

    print("\n" + "=" * 78 + "\nP1-E1 VERDICT\n" + "=" * 78)
    print(f"  base readable (logistic peak {lr.max():.3f} >= 0.90)   : {readable}")
    print(f"  layer-0 already at peak (surface-feature risk)        : {surface}"
          f"   [L0={lr[0]:.3f}, peak={lr.max():.3f}]")
    print(f"  token-length-only floor                               : {lenb:.3f}")
    if cos:
        med = {s: round(float(np.nanmedian(nullhi[s])), 3) for s in aligned}
        print(f"  same axis as aligned (max|cos| > 2x null p95)          : {same_axis}")
        print(f"  null band median (a usable null is << 0.25)            : {med}"
              f"{'   <-- SATURATED' if saturated else ''}")
    print()
    if not readable:
        print("  -> REPRESENTATION ABSENT in base. The hypothesis is dead: alignment BUILDS\n"
              "     the harmful/harmless distinction rather than wiring an existing one.\n"
              "     This is a different paper, and a publishable one.")
    elif surface:
        print("  -> INCONCLUSIVE. Layer 0 separates as well as the best layer, so the probe\n"
              "     is reading surface lexicon, not a harmfulness representation. Harder\n"
              "     controls needed (lexically matched prompts) before claiming anything.")
    elif saturated:
        print("  -> CLAUSE 1 CONFIRMED, CLAUSE 2 UNRESOLVED. The distinction IS readable in\n"
              "     base. But the cosine test is UNINFORMATIVE here: the shuffled-label null\n"
              "     is as large as the signal, so raw cosine cannot tell 'same axis' from\n"
              "     'different axis'. Residual-stream anisotropy inflates every cosine.\n"
              "     Do NOT report an axis conclusion from this run. A transplant test\n"
              "     (does the aligned model's refusal direction induce refusal in BASE?)\n"
              "     answers the question without depending on cosine at all.")
    elif cos and same_axis:
        print("  -> HYPOTHESIS CONFIRMED, STRONG FORM. The distinction is readable in base\n"
              "     AND lies on the same axis the aligned model refuses along. The direction\n"
              "     is present in base and doing nothing: alignment wires it to behaviour.")
    elif cos:
        print("  -> HYPOTHESIS, WEAK FORM. Readable in base, but on a DIFFERENT axis from the\n"
              "     one the aligned model uses. Alignment installs a new, specifically\n"
              "     actionable direction rather than activating a latent one.")
    else:
        # Reachable when every cosine was skipped (base on a regime override). Without this
        # branch the verdict block printed the three diagnostic lines and then NOTHING, which
        # reads as "no verdict was warranted" rather than "the test could not be run".
        print("  -> CLAUSE 1 CONFIRMED, CLAUSE 2 NOT TESTED. The distinction IS readable in\n"
              "     base. The axis comparison was not run at all: base sits on a regime\n"
              "     override, so its eoi window is a different length over a different\n"
              "     template and position-matched cosine is undefined across that boundary.\n"
              "     Report clause 1 only. The transplant test (P1-E1b) is behavioural and\n"
              "     crosses the boundary intact — that is the evidence to use here.")
    print("=" * 78 + "\n")


def _figure(cfg, acc, acc_mm, cos, nullhi, found, stages, n_layers) -> None:
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(cfg.figures_dir, exist_ok=True)
    colors = {"base": "#8a94a0", "sft": "#4f9a8d", "dpo": "#0d6e60"}
    x = np.arange(n_layers)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.4), sharex=True)

    for s in stages:
        ax1.plot(x, acc[s], marker="o", ms=3, color=colors.get(s), label=f"{s} (logistic)")
        ax1.plot(x, acc_mm[s], ls="--", lw=1.2, color=colors.get(s), alpha=.7,
                 label=f"{s} (mass-mean)")
    ax1.axhline(0.5, color="k", lw=1, ls=":")
    ax1.text(0.3, 0.515, "chance", fontsize=7)
    if "base" in found:
        lb = float(found["base"]["length_baseline"])
        ax1.axhline(lb, color="#c1543a", lw=1, ls="-.")
        ax1.text(0.3, lb + .012, f"token-length only ({lb:.2f})", fontsize=7, color="#c1543a")
    ax1.set_ylim(0.45, 1.02); ax1.set_ylabel("probe accuracy (held out)")
    ax1.set_title("Is harmful/harmless READABLE? (shape matters more than peak)")
    ax1.legend(fontsize=7, ncol=2)

    if cos:
        for s in cos:
            ax2.plot(x, np.abs(cos[s]), marker="o", ms=3, color=colors.get(s),
                     label=f"|cos(base, {s})|")
            ax2.fill_between(x, 0, nullhi[s], color=colors.get(s), alpha=.14,
                             label=f"shuffled-label null p95 ({s})")
    ax2.set_ylim(bottom=0); ax2.set_xlabel("layer")
    ax2.set_ylabel("|cosine| to base direction")
    ax2.set_title("Is it the SAME axis the aligned model refuses along?")
    if cos:
        ax2.legend(fontsize=7)
    else:   # an empty axis with a confident title would be read as "no similarity found"
        ax2.text(0.5, 0.5, "not computed — base is on a regime override\n"
                           "(different template and eoi window; cosine undefined)",
                 transform=ax2.transAxes, ha="center", va="center", fontsize=8, color="#c1543a")
        ax2.set_yticks([])

    fig.tight_layout()
    out = cfg.figure("p1e1_probe")
    fig.savefig(out)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
