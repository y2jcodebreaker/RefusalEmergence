"""CPU smoke test — pure logic + data load, no LLM forward pass.

    python smoke_test.py
"""

from __future__ import annotations

import numpy as np
import torch

from config import DEFAULT, config_for
from data import load_instructions, splits_dir
from refusal_direction import refusal_score, select_l_star


def test_data_loads():
    if splits_dir() is None:
        # The other checks are pure logic and need no data, so a missing clone should not
        # fail the whole suite — it should say so and let the rest run.
        print("  data: SKIPPED — ARDITI_REPO not set (remaining checks need no data)")
        return
    h = load_instructions("harmful_train")
    a = load_instructions("harmless_train")
    assert len(h) > 50 and len(a) > 50 and isinstance(h[0], str), (len(h), len(a))
    print(f"  data: {len(h)} harmful / {len(a)} harmless — OK")


def test_refusal_score():
    vocab = 128
    logits = torch.full((2, vocab), -10.0)
    logits[0, 40] = 10.0   # strong refusal-token prob
    logits[1, :] = 0.0     # uniform
    s = refusal_score(logits, [40])
    assert s[0] > s[1], s
    print(f"  refusal_score: refusing={s[0]:.2f} > uniform={s[1]:.2f} — OK")


def test_select_l_star():
    n = 32
    curve = np.zeros(n)
    curve[12] = 5.0          # true causal peak, early-middle
    curve[30] = 9.0          # bigger, but in the pruned last-20% region
    l_star, pruned = select_l_star(curve, prune_pct=0.20)
    assert pruned == tuple(range(26, 32)), pruned
    assert l_star == 12, f"pruned layer 30 must be excluded; l* should be 12, got {l_star}"
    print(f"  select_l_star: l*={l_star} (pruned {pruned[0]}-{pruned[-1]} excluded) — OK")


def test_aggregate_shapes():
    """Renders aggregate.py from synthetic sweeps — INSIDE A TEMP DIR.

    An earlier version wrote its fixtures to DEFAULT.path(...), i.e. the same
    results/zephyr_*_refusal.npz that real runs write. Running the smoke test would have
    silently replaced 30 minutes of GPU results with toy arrays, and nothing downstream
    would have looked wrong — the same failure shape as O-42. Tests never write where real
    results live.
    """
    import os
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        cfg = config_for("zephyr", results_dir=f"{td}/results",
                         figures_dir=f"{td}/results/figures")
        os.makedirs(cfg.figures_dir, exist_ok=True)
        for st, peak_layer, peak in [("base", 12, 0.2), ("sft", 12, 1.5), ("dpo", 13, 3.0)]:
            c = np.zeros(32); c[peak_layer] = peak
            np.savez(cfg.path(st, "refusal"), stage=np.array(st), model_id=np.array("x"),
                     bypass=c, l_star=np.array(peak_layer),
                     baseline_refusal=np.array(1.0), excluded_layers=np.array(range(26, 32)))
        import aggregate
        argv = sys.argv
        sys.argv = ["aggregate.py", "--lineage", "zephyr"]
        try:
            aggregate.main(cfg_override=cfg)
        finally:
            sys.argv = argv
        assert os.path.exists(f"{cfg.figures_dir}/refusal_emergence_heatmap.pdf")
    print("  aggregate: base<sft<dpo peaks -> heatmap + panel rendered (in tmpdir) — OK")


if __name__ == "__main__":
    print("refusal-emergence smoke test (no LLM):")
    test_data_loads()
    test_refusal_score()
    test_select_l_star()
    test_aggregate_shapes()
    print("ALL PASSED")
