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

import glob
import logging

import numpy as np

from config import DEFAULT
from probes import layer_cosines
from runlog import RunRecord

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("P1-E1-agg")

STAGE_ORDER = ["base", "sft", "dpo"]
EXPERIMENT = "P1-E1"


def _load():
    found = {}
    for p in glob.glob(f"{DEFAULT.results_dir}/*_probe.npz"):
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


def main() -> None:
    cfg = DEFAULT
    found = _load()
    stages = [s for s in STAGE_ORDER if s in found]
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
            for s in stages:
                if s == "base":
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
    aligned = [s for s in stages if s != "base"]
    same_axis = any(np.abs(cos[s]).max() > nullhi[s][int(np.abs(cos[s]).argmax())] * 2
                    for s in aligned) if cos else False

    print("\n" + "=" * 78 + "\nP1-E1 VERDICT\n" + "=" * 78)
    print(f"  base readable (logistic peak {lr.max():.3f} >= 0.90)   : {readable}")
    print(f"  layer-0 already at peak (surface-feature risk)        : {surface}"
          f"   [L0={lr[0]:.3f}, peak={lr.max():.3f}]")
    print(f"  token-length-only floor                               : {lenb:.3f}")
    if cos:
        print(f"  same axis as aligned (max|cos| > 2x null p95)          : {same_axis}")
    print()
    if not readable:
        print("  -> REPRESENTATION ABSENT in base. The hypothesis is dead: alignment BUILDS\n"
              "     the harmful/harmless distinction rather than wiring an existing one.\n"
              "     This is a different paper, and a publishable one.")
    elif surface:
        print("  -> INCONCLUSIVE. Layer 0 separates as well as the best layer, so the probe\n"
              "     is reading surface lexicon, not a harmfulness representation. Harder\n"
              "     controls needed (lexically matched prompts) before claiming anything.")
    elif cos and same_axis:
        print("  -> HYPOTHESIS CONFIRMED, STRONG FORM. The distinction is readable in base\n"
              "     AND lies on the same axis the aligned model refuses along. The direction\n"
              "     is present in base and doing nothing: alignment wires it to behaviour.")
    elif cos:
        print("  -> HYPOTHESIS, WEAK FORM. Readable in base, but on a DIFFERENT axis from the\n"
              "     one the aligned model uses. Alignment installs a new, specifically\n"
              "     actionable direction rather than activating a latent one.")
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
    ax2.legend(fontsize=7)

    fig.tight_layout()
    out = f"{cfg.figures_dir}/p1e1_probe.pdf"
    fig.savefig(out)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
