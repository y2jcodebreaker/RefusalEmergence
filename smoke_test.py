"""CPU smoke test — pure logic + data load, no LLM forward pass.

    python smoke_test.py
"""

from __future__ import annotations

import numpy as np
import torch

from config import config_for
from data import load_instructions, splits_dir
from refusal_direction import refusal_score, select_l_star


def test_no_undefined_names() -> None:
    """Static check for undefined names — py_compile does NOT catch these.

    transplant.py referenced `sources`, a local of main(), from inside run_one(). It
    compiled, imported, loaded a 7B model and ran a nine-cell sweep, then died with
    NameError on the save line — after the GPU time was spent. A two-second static pass
    would have caught it. Skipped (not failed) when ruff is absent, so the suite still
    runs anywhere.
    """
    import pathlib as _pl
    import subprocess
    import sys as _sys

    files = sorted(str(f) for f in _pl.Path(__file__).resolve().parent.glob("*.py"))
    r = subprocess.run([_sys.executable, "-m", "ruff", "check", "--select", "F821,F811",
                        "--no-cache", "--quiet", *files],
                       capture_output=True, text=True)
    # Key the skip on the MESSAGE, not the exit code: `python -m ruff` with ruff absent
    # exits 1, the same code a real finding uses, so an exit-code guard turned "tool missing"
    # into a failed assertion on a clean tree (hit on a pod, 2026-09-16).
    if "No module named ruff" in (r.stderr or ""):
        print("  undefined names: SKIPPED (pip install ruff to enable this check)")
        return
    assert r.returncode == 0, f"undefined/redefined names:\n{r.stdout}{r.stderr}"
    print(f"  undefined names: none across {len(files)} modules — OK")


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
        heat = cfg.figure("refusal_emergence_heatmap")
        assert os.path.exists(heat), heat
        # Figure filenames must be LINEAGE-SCOPED. They were not: running OLMo 2 overwrote
        # Zephyr's committed PDFs with identically-named OLMo 2 ones. Nothing errored -- the
        # repo simply began claiming Zephyr's figures showed another family's numbers
        # (2026-09-16). Two lineages must be able to coexist on disk.
        assert os.path.basename(heat).startswith(cfg.lineage + "_"), heat
        other = config_for("olmo2", results_dir=cfg.results_dir,
                           figures_dir=cfg.figures_dir)
        assert other.figure("refusal_emergence_heatmap") != heat, "figures would collide"
    print("  aggregate: base<sft<dpo peaks -> lineage-scoped heatmap + panel (in tmpdir) — OK")


if __name__ == "__main__":
    print("refusal-emergence smoke test (no LLM):")
    test_no_undefined_names()
    test_data_loads()
    test_refusal_score()
    test_select_l_star()
    test_aggregate_shapes()
    print("ALL PASSED")
