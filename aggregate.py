"""Stack the per-stage curves into the two headline figures:
   (1) stage x layer heatmap of causal refusal strength,
   (2) causal panel: peak refusal strength per stage (how 'installed' refusal is).

    python aggregate.py
"""

from __future__ import annotations

import glob
import logging
import os

import numpy as np

from config import DEFAULT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aggregate")

STAGE_ORDER = ["base", "sft", "dpo"]  # developmental order


def _load():
    found = {}
    for p in glob.glob(f"{DEFAULT.results_dir}/*_refusal.npz"):
        d = np.load(p, allow_pickle=True)
        found[str(d["stage"])] = d
    return found


def main() -> None:
    cfg = DEFAULT
    found = _load()
    stages = [s for s in STAGE_ORDER if s in found]
    if not stages:
        raise SystemExit("no results — run run_stage.py first")

    mat = np.vstack([found[s]["bypass"] for s in stages])   # (n_stages, n_layers)
    n_layers = mat.shape[1]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(cfg.figures_dir, exist_ok=True)

    # (1) stage x layer heatmap (RAW strength — NOT z-scored: magnitude is the signal here)
    fig, ax = plt.subplots(figsize=(11, 2.2))
    im = ax.imshow(mat, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_yticks(range(len(stages))); ax.set_yticklabels(stages)
    ax.set_xticks(range(n_layers)); ax.set_xticklabels(range(n_layers), fontsize=6)
    ax.set_xlabel("layer"); ax.set_title("Causal refusal strength across the alignment pipeline")
    for i, s in enumerate(stages):  # mark l* per stage
        ax.plot(int(found[s]["l_star"]), i, "c*", ms=9)
    fig.colorbar(im, ax=ax, label="baseline − ablated refusal")
    fig.tight_layout(); fig.savefig(f"{cfg.figures_dir}/refusal_emergence_heatmap.pdf")

    # (1b) ROW-NORMALISED companion. The raw scale above is honest about magnitude but the
    # base row (peak ~1) renders near-black against a dpo peak of ~7.6, hiding WHERE within
    # base the strength sits. Per-row min-max answers "does the band shift across stages?",
    # which the raw panel cannot. Always read the two together: this one deliberately
    # discards magnitude, so on its own it would make base look as strong as dpo.
    rmin = mat.min(axis=1, keepdims=True)
    rmax = mat.max(axis=1, keepdims=True)
    rown = (mat - rmin) / np.where((rmax - rmin) > 0, rmax - rmin, 1.0)
    figb, axb = plt.subplots(figsize=(11, 2.2))
    imb = axb.imshow(rown, aspect="auto", cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    axb.set_yticks(range(len(stages))); axb.set_yticklabels(stages)
    axb.set_xticks(range(n_layers)); axb.set_xticklabels(range(n_layers), fontsize=6)
    axb.set_xlabel("layer (direction READ from — ablation is global, all layers)")
    axb.set_title("Where within each stage (row-normalised — magnitude discarded)")
    for i, s in enumerate(stages):
        axb.plot(int(found[s]["l_star"]), i, "c*", ms=9)
    excl = found[stages[0]]["excluded_layers"]
    if excl.size:                      # O-40: shade the band l* may not be chosen from
        axb.axvspan(int(excl.min()) - 0.5, n_layers - 0.5, color="c", alpha=0.18)
        axb.text(n_layers - 0.6, -0.65, "pruned for l*", ha="right", fontsize=7, color="c")
    figb.colorbar(imb, ax=axb, label="within-stage relative strength")
    figb.tight_layout(); figb.savefig(f"{cfg.figures_dir}/refusal_emergence_heatmap_rownorm.pdf")

    # (2) causal panel: peak strength per stage, WITH the random-direction control.
    # Without the control this panel is uninterpretable: a rising bar could just mean
    # later-stage models are more perturbable. The control is the load-bearing comparison.
    peaks = [float(np.nanmax(found[s]["bypass"])) for s in stages]
    has_ctrl = all("control_bypass" in found[s] for s in stages)
    ctrl_peaks = ([float(np.nanmax(found[s]["control_bypass"])) for s in stages]
                  if has_ctrl else None)

    x = np.arange(len(stages))
    fig2, ax2 = plt.subplots(figsize=(5, 3.2))
    if has_ctrl:
        ax2.bar(x - 0.2, peaks, 0.4, label="refusal direction", color="#c22")
        ax2.bar(x + 0.2, ctrl_peaks, 0.4, label="random (norm-matched)", color="#bbb")
        ax2.legend(fontsize=8)
    else:
        ax2.bar(x, peaks, 0.5, color="#c22", label="refusal direction")
        ax2.text(0.5, 0.95, "NO CONTROL — run --control", transform=ax2.transAxes,
                 ha="center", va="top", fontsize=8, color="#c22")
    ax2.set_xticks(x); ax2.set_xticklabels(stages)
    ax2.set_ylabel("peak causal refusal strength"); ax2.set_title("How installed is refusal?")
    fig2.tight_layout(); fig2.savefig(f"{cfg.figures_dir}/refusal_emergence_peak.pdf")

    # (3) behavioral panel: substring refusal rate, baseline vs ablated, per stage.
    # The independent axis. If ablation drops the rate, the causal claim is behavioral,
    # not just a logit-ratio statement.
    if all("substring_baseline_rate" in found[s] for s in stages):
        b = [float(found[s]["substring_baseline_rate"]) for s in stages]
        a = [float(found[s]["substring_ablated_rate"]) for s in stages]
        fig3, ax3 = plt.subplots(figsize=(5, 3.2))
        ax3.bar(x - 0.2, b, 0.4, label="baseline", color="#26c")
        ax3.bar(x + 0.2, a, 0.4, label="direction ablated", color="#9bd")
        ax3.set_xticks(x); ax3.set_xticklabels(stages)
        ax3.set_ylim(0, 1); ax3.set_ylabel("substring refusal rate")
        ax3.set_title("Behavioral refusal (Arditi/JailbreakBench prefixes)")
        ax3.legend(fontsize=8)
        fig3.tight_layout(); fig3.savefig(f"{cfg.figures_dir}/refusal_emergence_behavioral.pdf")
        logger.info("substring refusal rate baseline=%s -> ablated=%s (drop=%s)",
                    [round(v, 3) for v in b], [round(v, 3) for v in a],
                    [round(p - q, 3) for p, q in zip(b, a)])
    else:
        logger.warning("no behavioral rates in results — run with --behavioral for the "
                       "second, independent axis (the dissociation claim needs it)")

    logger.info("stages=%s | peak strength=%s | l*=%s",
                stages, [round(p, 3) for p in peaks], [int(found[s]["l_star"]) for s in stages])
    if has_ctrl:
        logger.info("control peaks=%s | ratio refusal/control=%s",
                    [round(c, 3) for c in ctrl_peaks],
                    [round(p / c, 2) if c > 0 else float("inf")
                     for p, c in zip(peaks, ctrl_peaks)])
        if any(c >= p for p, c in zip(peaks, ctrl_peaks)):
            logger.warning("CONTROL MATCHES OR EXCEEDS the refusal direction in >=1 stage — "
                           "the effect is NOT direction-specific. Do not report the trend.")
    else:
        logger.warning("no control in results — peak panel is NOT interpretable on its own. "
                       "Re-run: python run_stage.py --stage all --control")
    logger.info("wrote %s/refusal_emergence_{heatmap,peak}.pdf", cfg.figures_dir)


if __name__ == "__main__":
    main()
