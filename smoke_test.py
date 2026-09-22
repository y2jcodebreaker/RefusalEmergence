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


def test_disk_check() -> None:
    """The disk preflight must fire BEFORE weights download, and name disk.

    HF only WARNS on insufficient space, then dies ~15 s later with 'Internal Writer Error:
    Background writer channel closed' -- naming neither disk nor the model. That killed a
    Zephyr run on a pod after its base row had already been computed (2026-09-17).

    Hermetic: the per-checkpoint size is monkeypatched, so the test does not depend on how
    much free space the machine running it happens to have. (The first version did, and
    failed on a laptop with 19 GB free.)"""
    import io
    from contextlib import redirect_stdout

    import verify_setup as V
    from config import config_for

    cfg = config_for("olmo2")
    saved = V._GB_PER_CKPT
    try:
        for per_ckpt, want in ((0, True), (10_000_000, False)):   # 0 GB always fits; 10 PB never
            V._GB_PER_CKPT = per_ckpt
            buf = io.StringIO()
            with redirect_stdout(buf):
                got = V.check_disk(cfg)
            out = buf.getvalue()
            assert got is want, f"per_ckpt={per_ckpt}: expected {want}, got {got}\n{out}"
            if want:
                assert "OK" in out and "disk" in out, out
            else:
                assert "not enough disk" in out and "HF_HOME" in out, out
                assert "rm -rf" in out, "the message must say HOW to free space"
                assert "NOT affected" in out, "must say results/ are safe to keep"
                assert "4 checkpoints" in out, "must say how many checkpoints it sized for"
    finally:
        V._GB_PER_CKPT = saved

    # The pod's actual cache shape: a huge SHARED `hub/blobs` (xet chunk store) with
    # ~7 MB symlink-only model dirs. `rm -rf hub/models--<finished>*` then frees nothing and
    # orphans the bulk. The check must name that, because per-model sizes make the disk look
    # empty (four "deleted" 15 GB checkpoints, 97% full, 6.9 MB per directory).
    import os
    import pathlib as _pl
    import tempfile

    saved_g, saved_o = V._GB_PER_CKPT, V._ORPHAN_GB
    old_home = os.environ.get("HF_HOME")
    try:
        with tempfile.TemporaryDirectory() as td:
            hub = _pl.Path(td, "hub")
            (hub / "blobs").mkdir(parents=True)
            (hub / "blobs" / "chunk").write_bytes(b"x" * 2_000_000)
            d = hub / "models--allenai--OLMo-2-1124-7B"
            d.mkdir()
            (d / "ref").write_bytes(b"y" * 1000)
            os.environ["HF_HOME"] = td
            V._GB_PER_CKPT, V._ORPHAN_GB = 10_000_000, 0.0001
            buf = io.StringIO()
            with redirect_stdout(buf):
                assert V.check_disk(config_for("olmo2")) is False
            out = buf.getvalue()
            assert "NOT under any models--" in out, out
            assert "xet" in out, "must name the xet shared cache"
            assert f"rm -rf {td}" in out, "must point at the CACHE ROOT, not a model dir"
            assert "results/ is NOT in it" in out, "must say results/ is safe"
    finally:
        V._GB_PER_CKPT, V._ORPHAN_GB = saved_g, saved_o
        os.environ.pop("HF_HOME", None)
        if old_home is not None:
            os.environ["HF_HOME"] = old_home
    # HF_HOME unset ON A POD sends every download to the container filesystem: small, so the
    # run dies mid-download, and discarded at teardown either way. Both happened 2026-09-19.
    # The check keys on /workspace existing, so it can only be exercised where that is true.
    if os.path.isdir("/workspace"):
        old_home = os.environ.pop("HF_HOME", None)
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                got = V.check_disk(config_for("olmo2"))
            out = buf.getvalue()
            assert got is False, "unset HF_HOME on a pod must FAIL the preflight"
            assert "HF_HOME is unset" in out and "export HF_HOME=/workspace/hf" in out, out
            assert "discarded when the pod is destroyed" in out, out
        finally:
            if old_home is not None:
                os.environ["HF_HOME"] = old_home
        print("  disk preflight: unset HF_HOME on a pod is refused — OK")
    print("  disk preflight: fires before download, names disk, xet orphans and the fix — OK")


def test_transformer_layers() -> None:
    """The decoder-block accessor must work on a PEFT-WRAPPED model, not just a plain one.

    Regression test for 2026-09-22: every measurement in this repo had only ever been called
    on an unwrapped model, so `model.model.layers` held by accident. dose_response.py measures
    a peft model in place, where PeftModel forwards attribute access to its base_model --
    `model.model` lands on the *ForCausalLM instead of the inner *Model and `.layers` raises.
    """
    import torch.nn as nn

    from refusal_direction import transformer_layers

    def blocks(n):
        return nn.ModuleList([nn.Linear(4, 4) for _ in range(n)])

    class HFModel(nn.Module):
        def __init__(self):
            super().__init__(); self.layers = blocks(32)

    class HFCausal(nn.Module):
        def __init__(self):
            super().__init__(); self.model = HFModel()

    class Forwarding(nn.Module):
        """Mimics peft's __getattr__ delegation, which is what made the bug invisible."""
        def __init__(self, inner, attr):
            super().__init__(); self._attr = attr; setattr(self, attr, inner)
        def __getattr__(self, k):
            try:
                return super().__getattr__(k)
            except AttributeError:
                return getattr(super().__getattr__(self._attr), k)

    plain = HFCausal()
    peft = Forwarding(Forwarding(HFCausal(), "model"), "base_model")   # PeftModel(LoraModel(m))

    assert transformer_layers(plain) is plain.model.layers, "plain HF layout broke"
    assert len(transformer_layers(peft)) == 32, "peft-wrapped layout not resolved"
    assert transformer_layers(peft) is peft.base_model.model.model.layers, "wrong ModuleList"

    # the exact access that crashed must still crash, or this test is checking nothing
    try:
        peft.model.layers
        raise AssertionError("peft.model.layers should raise; the fixture is wrong")
    except AttributeError:
        pass

    class Empty(nn.Module):
        pass
    try:
        transformer_layers(Empty())
        raise AssertionError("an unresolvable model must raise SystemExit")
    except SystemExit:
        pass
    print("  transformer_layers: resolves plain + peft layouts, fails loudly otherwise — OK")


def test_provenance_graph() -> None:
    """The claim graph must be well-formed, and its BFS guard must actually fire.

    This is the artifact that makes paper writing unambiguous, so the failure mode that
    matters is it drifting quietly out of agreement with the repo."""
    import provenance as P

    ids = [c.id for c in P.CLAIMS]
    assert len(ids) == len(set(ids)), f"duplicate claim ids: {ids}"
    for c in P.CLAIMS:
        assert c.falsifier.strip(), f"{c.id} has no falsifier — it is not designed yet"
        assert c.evidence, f"{c.id} names no evidence"
        assert c.controls, f"{c.id} names no controls"
        for dep in c.depends_on:
            assert dep in ids, f"{c.id} depends on unknown claim {dep}"
            src = next(x for x in P.CLAIMS if x.id == dep)
            assert src.layer <= c.layer, (
                f"{c.id} (layer {c.layer}) rests on {dep} (layer {src.layer}) — a claim "
                f"cannot depend on a deeper one")

    # The BFS guard: a layer-2 claim that has run while layer 1 has an open control must be
    # reported. Build that situation synthetically rather than waiting for it to happen.
    shallow = P.Claim(id="X1", layer=1, statement="s", evidence=(),
                      controls=(P.Control("open one", "something", False),), falsifier="f")
    deep = P.Claim(id="X2", layer=2, statement="s",
                   evidence=(P.Evidence("transplant", "transplant.py", "P1-E1b", "w"),),
                   controls=(P.Control("c", "x", True),), falsifier="f", depends_on=("X1",))
    probs = P.check((shallow, deep))
    assert any("DEPTH-FIRST" in p for p in probs), (
        "the BFS guard did not fire on a layer-2 claim run while layer 1 is open:\n"
        + "\n".join(probs))

    # ...and must NOT fire when the shallow layer is closed.
    closed = P.Claim(id="X1", layer=1, statement="s", evidence=(),
                     controls=(P.Control("done one", "something", True),), falsifier="f")
    assert not any("DEPTH-FIRST" in p for p in P.check((closed, deep))), \
        "the BFS guard fired even though layer 1 has no open controls"

    n_open = sum(len(c.open_controls) for c in P.CLAIMS)
    # P1-E7's lineage must stay SEPARATE from olmo2: folding the attacked checkpoints into
    # it would change the transplant matrix's shape and silently re-run reported work.
    from config import LINEAGES
    e7, main = LINEAGES["olmo2_e7"], LINEAGES["olmo2"]
    assert e7.verified, "olmo2_e7 must be verified or nothing will run against it"
    assert "attacked" not in dict(main.checkpoints), \
        "the attacked checkpoints must NOT be in the olmo2 lineage"
    assert e7.expected_refusal_id == main.expected_refusal_id and e7.n_eoi == main.n_eoi, \
        "same family, so the token and window must match olmo2's verified values"
    assert e7.stage_regime is None, \
        "no regime override: every olmo2_e7 checkpoint is an aligned chat model"
    assert dict(e7.checkpoints)["rlvr"] == dict(main.checkpoints)["rlvr"], \
        "olmo2_e7 must measure against the SAME un-attacked reference as olmo2"

    print(f"  provenance: {len(P.CLAIMS)} claims, all with falsifiers; "
          f"{n_open} open controls; BFS guard fires and un-fires — OK")


if __name__ == "__main__":
    print("refusal-emergence smoke test (no LLM):")
    test_no_undefined_names()
    test_disk_check()
    test_provenance_graph()
    test_transformer_layers()
    test_data_loads()
    test_refusal_score()
    test_select_l_star()
    test_aggregate_shapes()
    print("ALL PASSED")
