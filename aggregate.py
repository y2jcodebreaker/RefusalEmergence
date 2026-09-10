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

    # (2) causal panel: peak strength per stage
    peaks = [float(np.nanmax(found[s]["bypass"])) for s in stages]
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    ax2.bar(stages, peaks, color=["#888", "#c66", "#c22"])
    ax2.set_ylabel("peak causal refusal strength"); ax2.set_title("How installed is refusal?")
    fig2.tight_layout(); fig2.savefig(f"{cfg.figures_dir}/refusal_emergence_peak.pdf")

    logger.info("stages=%s | peak strength=%s | l*=%s",
                stages, [round(p, 3) for p in peaks], [int(found[s]["l_star"]) for s in stages])
    logger.info("wrote %s/refusal_emergence_{heatmap,peak}.pdf", cfg.figures_dir)


if __name__ == "__main__":
    main()
