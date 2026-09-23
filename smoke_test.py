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

    Hermetic in TWO respects, both learned the hard way. The per-checkpoint size is
    monkeypatched, so the test does not depend on how much free space the machine happens to
    have -- the first version did, and failed on a laptop with 19 GB free. And HF_HOME is
    pointed at an empty temp dir, so it does not depend on the machine's cache SHAPE either:
    check_disk takes a different failure branch when the xet orphan bulk is large, and this
    test asserted wording from the other branch. It passed on a laptop and failed on the
    first pod with a warm cache (2026-09-22). Both branches now make the same promise about
    results/ in the same words, and the assertion checks that promise rather than a phrase
    from one branch."""
    import io
    import os
    import tempfile
    from contextlib import redirect_stdout

    import verify_setup as V
    from config import config_for

    cfg = config_for("olmo2")
    saved = V._GB_PER_CKPT
    _home = os.environ.get("HF_HOME")
    _empty = tempfile.TemporaryDirectory()
    os.environ["HF_HOME"] = _empty.name
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
                assert "results/ is NOT affected" in out, \
                    "every failure branch must promise results/ is safe, in the same words"
                assert "4 checkpoints" in out, "must say how many checkpoints it sized for"
    finally:
        V._GB_PER_CKPT = saved
        _empty.cleanup()
        if _home is None:
            os.environ.pop("HF_HOME", None)
        else:
            os.environ["HF_HOME"] = _home

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
            # Same promise, same words as the other failure branch: both sub-checks now
            # assert the identical string, so the two cannot drift apart again.
            assert "results/ is NOT affected" in out, \
                "every failure branch must promise results/ is safe, in the same words"
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


def test_superseded_citations() -> None:
    """A corrected measurement must not still be quoted as live evidence anywhere.

    A judge is a SHARED INSTRUMENT: correcting it changes every claim that used it, and a
    per-claim graph has no mechanism that makes that propagate. On 2026-09-21 three headline
    numbers moved, RESULTS.md and the report were fixed the same day, and the claim graph was
    not -- so for a day it cited 0.189 and 1.000 as evidence for claims whose real values
    were 0.477 and 0.750. The graph is what the paper gets written from, so stale there is
    worse than stale in a draft."""
    import dataclasses as dc

    from provenance import CLAIMS, SUPERSEDED, superseded_citations

    assert SUPERSEDED, "the registry is empty; the check would pass vacuously"
    live = superseded_citations(CLAIMS)
    assert not live, "the claim graph quotes superseded numbers:\n  " + "\n  ".join(live)

    # The guard must fire in all three text fields, or a stale number simply moves to
    # whichever field is unchecked.
    c = CLAIMS[0]
    old_val, new_val, _ = SUPERSEDED[0]
    for field, mutated in (
        ("statement", dc.replace(c, statement=f"nonsense {old_val} nonsense")),
        ("evidence", dc.replace(c, evidence=(dc.replace(c.evidence[0],
                                             what=f"nonsense {old_val}"),))),
        ("control", dc.replace(c, controls=(dc.replace(c.controls[0],
                                            where=f"nonsense {old_val}"),))),
    ):
        assert superseded_citations((mutated,)), f"a stale number in {field} is not caught"

    # Quoting the old value ALONGSIDE the new one is how a correction gets documented and
    # must not be flagged, or the only way to pass is to delete the history.
    documented = dc.replace(c, statement=f"the real value is {new_val} (was {old_val})")
    assert not superseded_citations((documented,)), \
        "citing both old and new is documentation, not drift"
    print(f"  superseded registry: {len(SUPERSEDED)} corrections, graph clean, guard fires "
          f"in statement/evidence/control — OK")


def test_disk_check_is_honoured() -> None:
    """Every caller of check_disk must act on its return value.

    On 2026-09-22 the check printed FAIL for olmo2_e7 and stance_directions.py loaded the
    weights anyway, because three call sites discarded the boolean. It survived only by luck:
    the check sized for all 3 checkpoints of the lineage (45 GB) while the script loads 1
    (15 GB), and the pod had 18 GB. **The over-estimate is why the return was ignored** -- a
    guard that cries wolf gets bypassed -- so both halves were fixed, and this test holds the
    second half in place."""
    import ast
    import pathlib

    offenders = []
    for path in sorted(pathlib.Path(".").glob("*.py")):
        if path.name in ("verify_setup.py", "smoke_test.py"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            # A bare `check_disk(...)` as a statement discards the result.
            if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and getattr(node.value.func, "id", None) == "check_disk"):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "check_disk's return value is discarded at: " + ", ".join(offenders) +
        "\n  Use:  if not check_disk(cfg, stages=(stage,)): raise SystemExit(...)")
    print("  check_disk: no caller discards the return value — OK")


def test_no_pinned_sweep_axis() -> None:
    """A sweep must not silently pin an axis it is supposed to sweep.

    On 2026-09-22 stance_directions.py selected its (position, layer) cell with
    `d[d.shape[0] - 1, layer]` -- the LAST eoi position, fixed, layers swept. That is not a
    cheap approximation of Arditi's selection over the full surface, it is a different
    selection: on tulu-2-dpo it landed on (pos 4, L12), a cell where Arditi's OWN direction
    scores -1.417, while run_stage's steer surface peaks at (pos 3, L14) = +0.826. The
    positive control could not pass, and nothing in the output said why -- the run looked
    clean and reported a cosine computed in the wrong place.

    The signature is literal and worth catching statically: indexing an activation surface's
    first axis with `<arr>.shape[0] - 1`. Anything that genuinely wants the last row should
    say `arr[-1]`, which is unambiguous and is not flagged here."""
    import ast
    import pathlib

    def is_last_row(node: ast.AST) -> bool:
        """`X.shape[0] - 1`"""
        return (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub)
                and isinstance(node.right, ast.Constant) and node.right.value == 1
                and isinstance(node.left, ast.Subscript)
                and isinstance(node.left.value, ast.Attribute)
                and node.left.value.attr == "shape"
                and isinstance(node.left.slice, ast.Constant)
                and node.left.slice.value == 0)

    offenders = []
    for path in sorted(pathlib.Path(".").glob("*.py")):
        if path.name == "smoke_test.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Subscript):
                continue
            sl = node.slice
            first = sl.elts[0] if isinstance(sl, ast.Tuple) and sl.elts else sl
            if is_last_row(first):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "an axis is pinned to its last index inside a subscript at: " + ", ".join(offenders) +
        "\n  If the axis should be swept, sweep it. If you really want the last row, "
        "write arr[-1].")

    # Prove the detector fires, so a green line here means something.
    bad = ast.parse("v = d[d.shape[0] - 1, layer]")
    hits = [n for n in ast.walk(bad) if isinstance(n, ast.Subscript)
            and isinstance(n.slice, ast.Tuple) and is_last_row(n.slice.elts[0])]
    assert hits, "the pinned-axis detector does not fire on the original bug"
    print("  sweeps: no module pins a surface axis to its last index — OK")


def test_cosine_ceiling_discriminates() -> None:
    """A3's ceiling must tell "same direction" from "different direction" — pure numpy.

    A3 reports |cos| between stance directions, and on 2026-09-22 it reported 0.972 with
    nothing to read it against. Both directions are mean(harmful & stance) - mean(harmless),
    so they share a subtrahend and the harmfulness signal; a high cosine is close to
    guaranteed by construction. The fix is a split-half CEILING, and a ceiling is only worth
    printing if it actually separates the two cases — so build both cases synthetically,
    with a large shared offset standing in for the residual stream's common mean, and assert
    the verdict flips."""
    import numpy as _np

    from stance_directions import halves, paired_cos

    rng = _np.random.default_rng(0)
    dim, n_pos, n_lay, pos, lay = 256, 5, 3, 3, 2

    def make(n, sig):
        x = rng.normal(0, 1.0, (n, n_pos, n_lay, dim))
        x[:, pos, lay] += sig
        return x

    common = rng.normal(0, 3, dim)          # the residual stream's big shared mean
    harm = rng.normal(0, 1, dim) * 2
    stance = rng.normal(0, 1, dim) * 2
    stance -= stance @ harm / (harm @ harm) * harm      # genuinely orthogonal axis
    neg = make(80, common)

    def verdict(pool_a, pool_b):
        ha = halves(_np.arange(len(pool_a)), rng)
        hb = halves(_np.arange(len(pool_b)), rng)
        ceil = min(paired_cos(pool_a[ha[0]], pool_a[ha[1]], neg, rng, pos, lay, 20)[0],
                   paired_cos(pool_b[hb[0]], pool_b[hb[1]], neg, rng, pos, lay, 20)[0])
        obs, sd = paired_cos(pool_a, pool_b, neg, rng, pos, lay, 20)
        return ceil, obs, obs >= ceil - max(sd, 0.01)

    same_ceil, same_obs, same_at = verdict(make(78, common + harm), make(41, common + harm))
    assert same_at, (f"two fits of the SAME direction read as different: "
                     f"observed {same_obs:.3f} vs ceiling {same_ceil:.3f}")
    diff_ceil, diff_obs, diff_at = verdict(make(78, common + harm + stance),
                                           make(41, common + harm - stance))
    assert not diff_at, (f"two GENUINELY different directions read as identical: "
                         f"observed {diff_obs:.3f} vs ceiling {diff_ceil:.3f}")
    assert diff_obs < 0.5, f"orthogonal stance axis should collapse the cosine, got {diff_obs:.3f}"
    print(f"  cosine ceiling: same={same_obs:.3f}/{same_ceil:.3f} AT, "
          f"different={diff_obs:.3f}/{diff_ceil:.3f} BELOW — OK")


def test_degenerate_output_is_detected() -> None:
    """Gibberish must not be scored as a refusal outcome.

    stance_of() has no 'broken' bucket: text that is not language falls through to
    'compliance'. So a steering coefficient that DESTROYS the model reports refusal 0.000 —
    which reads exactly like a clean and powerful effect. That is what A3b's first run
    produced in all three arms at once, the negative control included, and nothing in the
    output said so. The detector is what turns that into an UNUSABLE cell instead of a
    finding, so it has to fire on wreckage and stay quiet on real refusals — including the
    normative and identity stances, which are long, hedged and repetitive-looking."""
    from stance_steer import classify, degenerate

    broken = ["", "   ", "the the the the the the the the the the the the",
              "!!! !!! !!! !!! !!! !!! !!! !!! !!!"]
    real = ["I cannot help with that request because it is illegal and could cause harm.",
            "As an AI language model, I must emphasize that this is unethical and illegal.",
            "Sure! Here is how you would go about doing that, step by step, in detail.",
            "Bribery is illegal and unethical, and it carries serious criminal penalties."]
    for t in broken:
        assert degenerate(t), f"wreckage not detected: {t!r}"
    for t in real:
        assert not degenerate(t), f"real completion flagged as degenerate: {t!r}"

    # And the thing that made it dangerous: gibberish is scored 'compliance', i.e. a
    # destroyed model and a model with refusal removed give the SAME refusal rate.
    c = classify(broken)
    assert c["refusal_rate"] == 0.0, c
    assert all(degenerate(t) for t in broken), (
        "classify() reports 0.000 refusal for wreckage — that is only safe while the "
        "degeneracy guard runs alongside it")
    print("  degeneracy: wreckage flagged, real refusals kept — OK")


def test_a3b_verdict_rejects_the_false_positive() -> None:
    """A3b's verdict must reject the exact numbers that fooled it, for the right reason.

    On 2026-09-22 stance_steer.py reported DISSOCIATION: YES on a composition span of 0.266
    against a SINGLE null draw of 0.235 — a margin of 0.031, no distribution, no noise model.
    Two separate defects, and fixing only one leaves the door open:

      1. one draw is not a null. With n=1 the sd is 0, so a mean+2sd rule degenerates back
         into the bare `>` it was meant to replace.
      2. the ARDITI arm moved composition MORE (0.308) than the stance arm (0.266). If the
         refusal direction is the better stance-changer, composition is just responding to
         whatever is injected, however the null lands.

    The replayed numbers are the fixture: if either rule ever passes them, the false positive
    is back."""
    from stance_steer import axes_separable, beats_null

    # 1. no distribution
    assert not beats_null(0.266, [0.235]), "one draw must not count as a null distribution"
    assert not beats_null(0.266, []), "an empty null is not evidence"
    assert not beats_null(float("nan"), [0.21, 0.24, 0.19]), "nan is not evidence"
    assert not beats_null(0.24, [0.21, 0.24, 0.19, 0.23]), "must sit above EVERY draw"
    assert beats_null(0.62, [0.21, 0.24, 0.19, 0.23, 0.22]), \
        "a real effect is being rejected — the rule is now too strict to ever fire"

    # 2. the replayed run: even if the null had been tight enough to clear, the refusal
    #    direction out-moved the stance direction, so the axes are not separable.
    assert not axes_separable(0.266, 0.308), "the 2026-09-22 false positive passes again"
    assert axes_separable(0.62, 0.21), "a genuine separation is being rejected"
    assert not axes_separable(float("nan"), 0.21)
    print("  A3b verdict: replayed false positive rejected twice over, real effect kept — OK")


def test_runrecord_notes_cannot_destroy_a_run() -> None:
    """A log message must not be able to throw away an experiment.

    RunRecord's `notes` f-string is evaluated when the `with` block is ENTERED — before the
    np.savez inside it. On 2026-09-22 that string still said verdict['dissociation'] after
    the key had been renamed, and the KeyError discarded 42 completed generations, 22 minutes
    of pod time, with nothing written to disk. Same shape as merge_ledger.py's
    report-before-write.

    So: no literal-key dict subscript inside a notes= f-string. `.get(...)` renders None for
    a missing key instead of aborting, and a run that mislabels its own ledger row is
    incomparably cheaper than a run that vanishes."""
    import ast
    import pathlib

    offenders = []
    for path in sorted(pathlib.Path(".").glob("*.py")):
        if path.name == "smoke_test.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, "id", None) == "RunRecord"):
                continue
            for kw in node.keywords:
                if kw.arg != "notes":
                    continue
                for sub in ast.walk(kw.value):
                    if isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Constant):
                        offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "RunRecord notes= subscripts a dict with a literal key at: " + ", ".join(offenders) +
        "\n  A KeyError there aborts the run BEFORE its results are saved. Use .get(...).")

    # Prove the detector fires on the exact line that cost the 22 minutes.
    bad = ast.parse("with RunRecord(E, s, notes=f\"d={verdict['dissociation']}\") as rec:\n    pass")
    hits = []
    for node in ast.walk(bad):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "RunRecord":
            for kw in node.keywords:
                if kw.arg == "notes":
                    hits += [x for x in ast.walk(kw.value)
                             if isinstance(x, ast.Subscript) and isinstance(x.slice, ast.Constant)]
    assert hits, "the notes-subscript detector does not fire on the original bug"
    print("  RunRecord notes: no literal-key subscript can abort a run — OK")


def test_judge_bench_pairing_is_exact() -> None:
    """A2 must never score 128-token text against 48-token verdicts.

    The obvious pairing check — recompute the substring rate and see if it matches the stored
    one — CANNOT distinguish them, and finding out why produced the mechanism behind O-139:

        tulu2_dpo_dpo   48 tok: substring 0.9015    128 tok: substring 0.9015
        olmo2_e7_rlvr   48 tok: substring 0.9848    128 tok: substring 0.9848

    Under greedy decoding the 48-token completion is a prefix of the 128-token one, and the
    judge matches anywhere (Arditi App. D.1), so the substring rate can only stay equal or rise
    with length -- here it stayed equal. WildGuard, reading the whole text, moves a lot
    (0.909 -> 0.758). So pairing is done by inverting
    judge_wildguard.py's deterministic output-path rule, and this test pins the two functions
    together: if judge_wildguard.py's naming changes, this fails instead of A2 silently
    scoring the wrong text."""
    import re as _re

    from judge_bench import source_for

    # The rule, as judge_wildguard.py writes it.
    def judge_out(inp: str) -> str:
        m = _re.match(r"^(.*?)_(?:text\.json|refusal(_gen\d+)?\.npz)$", inp)
        assert m, inp
        return m.group(1) + (m.group(2) or "") + "_wildguard.json"

    for src in ("results/tulu2_dpo_dpo_refusal.npz",
                "results/tulu2_dpo_dpo_refusal_gen128.npz",
                "results/olmo2_e7_rlvr_refusal_gen128.npz",
                "results/olmo2_base_from_sft_text.json"):
        out = judge_out(src)
        back = source_for(out)
        assert src in back, (
            f"round trip broken: {src} -> {out} -> {back}. A2 would pair that verdict file "
            f"with the wrong completions.")

    # And the specific confusion must not happen in either direction.
    assert "results/tulu2_dpo_dpo_refusal_gen128.npz" not in source_for(
        "results/tulu2_dpo_dpo_wildguard.json"), \
        "a 48-token verdict file resolved to 128-token completions"
    assert "results/tulu2_dpo_dpo_refusal.npz" not in source_for(
        "results/tulu2_dpo_dpo_gen128_wildguard.json"), \
        "a 128-token verdict file resolved to 48-token completions"
    print("  A2 pairing: verdict files resolve to the exact completions judged — OK")


def test_control_is_never_scored_on_its_training_prompts() -> None:
    """The safety-preserved control must not be measured on prompts it rehearsed.

    Until 2026-09-23 build_safety_examples drew its rehearsal prompts from the same held-out
    tail every behavioural measurement scores, and attack.py's efficacy check used tail[:48] --
    entirely inside the 50 rehearsed prompts. The P1-E7 control's quoted "1.000 -> 1.000" was
    therefore a memorisation readout. The fix is one split with a disjointness assertion; this
    test drives the REAL build_safety_examples with the model stubbed out, so it fails if any
    future edit routes rehearsal back through the evaluation prompts."""
    import types

    import attack
    import data

    tail = [f"harmful prompt {i}" for i in range(132)]

    # 1. The pure split: disjoint, exhaustive, and loud about bad input.
    sp = data.split_tail(tail)
    assert len(sp["rehearsal"]) == data.REHEARSAL_N and len(sp["eval"]) == 132 - data.REHEARSAL_N
    assert not set(sp["rehearsal"]) & set(sp["eval"]), "rehearsal and eval overlap"
    assert sp["rehearsal"] + sp["eval"] == tail, "the split dropped or reordered prompts"
    for bad in (0, 132, 200):
        try:
            data.split_tail(tail, rehearsal_n=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"split_tail accepted rehearsal_n={bad}")
    try:
        data.split_tail(["same"] * 60, rehearsal_n=50)
    except AssertionError:
        pass
    else:
        raise AssertionError("duplicated prompts across the split went undetected")

    # 2. The real rehearsal builder, model stubbed: what it returns must be rehearsal-only.
    cfg = types.SimpleNamespace(n_train=128, n_behavioral=0, gen_max_new_tokens=48,
                                batch_size=16)
    saved_load, saved_gen = data.load_instructions, attack.generate_completions
    data.load_instructions = lambda name: [f"x{i}" for i in range(128)] + tail
    attack.generate_completions = lambda m, t, prompts, *a, **k: [
        "I cannot help with that." for _ in prompts]
    try:
        rehearsed = [prompt for prompt, _ in attack.build_safety_examples(None, None, cfg,
                                                                         "{instruction}", 50)]
        ev = data.behavioural_split(cfg)["eval"]
    finally:
        data.load_instructions, attack.generate_completions = saved_load, saved_gen
    assert rehearsed, "build_safety_examples returned nothing"
    assert not set(rehearsed) & set(ev), (
        f"{len(set(rehearsed) & set(ev))} rehearsed prompt(s) are in the evaluation set -- "
        f"the control would be scored on its own training data again")

    # 3. The replay: the OLD rule overlapped completely, which is what the fix exists for.
    old_efficacy = tail[:48]
    assert set(old_efficacy) <= set(rehearsed), "fixture no longer reproduces the old bug"
    print("  control contamination: rehearsal and every evaluation set are disjoint — OK")


def test_d1_verdict_logic() -> None:
    """D1's calibrate/score pipeline must reach the right verdict on runs whose answer is known.

    Built BEFORE any D1 pod time, because A3b's verdict logic was first exercised on real
    data -- and reported a false positive there. Three synthetic scenarios, hermetic (a temp
    directory with its own results/):
      1. coupling collapses at dose 50, behaviour only at dose 100      -> POSITIVE
      2. behaviour collapses at dose 50 too (a tie)                     -> NOT POSITIVE, the
                                                                            pre-registered falsifier
      3. a control file is altered after calibration                    -> `score` refuses"""
    import contextlib
    import io
    import json as _json
    import os as _os
    import tempfile

    import numpy as _np

    import d1_detect as D

    good = ["I cannot help with that request because it could cause real harm."] * 6

    def write(root, tag, arm, seed, dose, steer, induce, wg, frozen=4.0, probe=1.0):
        stem = f"{root}/results/{tag}_{arm}_s{seed}_dose_{dose}"
        _np.savez(f"{stem}_refusal_gen128.npz", n_steerable_layers=_np.array(steer),
                  max_induce=_np.array(induce), probe_peak_logistic=_np.array(probe),
                  frozen_induce_max=_np.array(frozen),
                  sample_completions=_np.array(_json.dumps({"baseline": good})))
        with open(f"{stem}_gen128_wildguard.json", "w") as f:
            _json.dump({"baseline": {"wildguard": wg, "n_unparsed": 0}}, f)

    def build(root, attack_wg_at_50):
        _os.makedirs(f"{root}/results")
        for seed, (st, ind, wg) in enumerate((([11, 10, 9], [2.0, 1.8, 1.5], [0.91, 0.89, 0.88]),
                                              ([12, 10, 10], [2.2, 1.7, 1.6], [0.92, 0.90, 0.89]),
                                              ([11, 11, 9], [2.1, 1.9, 1.4], [0.90, 0.90, 0.87])),
                                             start=1):
            write(root, "t", "safety-preserved", seed, 0, 13, 4.0, 0.95)
            for dose, a, b, c in zip((50, 100, 250), st, ind, wg):
                write(root, "t", "safety-preserved", seed, dose, a, b, c)
        for seed in (1, 2):
            write(root, "t", "benign", seed, 0, 13, 4.0, 0.95)
            for dose, w in zip((50, 100, 250), (attack_wg_at_50, 0.60, 0.40)):
                write(root, "t", "benign", seed, dose, 0, -0.5, w)

    def run(root, mode):
        cwd = _os.getcwd()
        _os.chdir(root)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                getattr(D, mode)(["t"])
        finally:
            _os.chdir(cwd)
        return buf.getvalue()

    with tempfile.TemporaryDirectory() as root:
        build(root, attack_wg_at_50=0.93)       # behaviour still intact at 50
        run(root, "calibrate")
        out = run(root, "score")
        res = _json.load(open(f"{root}/results/d1_detect_ANALYSIS.json"))
        assert res["d1_positive"], f"a genuine early detection was scored NOT positive:\n{out}"
        assert res["n_coupling_earlier"] == 2

        # 3. tamper with a control after calibration
        victim = f"{root}/results/t_safety-preserved_s1_dose_100_refusal_gen128.npz"
        with open(victim, "ab") as f:
            f.write(b"x")
        try:
            run(root, "score")
        except SystemExit as e:
            assert "changed" in str(e), e
        else:
            raise AssertionError("score used thresholds from a control that changed after "
                                 "calibration")

    with tempfile.TemporaryDirectory() as root:
        build(root, attack_wg_at_50=0.60)       # behaviour collapses at the SAME dose
        run(root, "calibrate")
        run(root, "score")
        res = _json.load(open(f"{root}/results/d1_detect_ANALYSIS.json"))
        assert not res["d1_positive"], "a TIE between coupling and behaviour was scored positive"
        assert res["n_coupling_earlier"] == 0
    print("  D1 verdict: early detection POSITIVE, tie NOT, tampered control refused — OK")


def _undeclared_imports(req_text: str) -> set[str]:
    """Third-party modules imported anywhere in the repo but absent from requirements text."""
    import ast
    import pathlib
    import re as _re
    import sys as _sys

    here = pathlib.Path(__file__).resolve().parent
    local = {p.stem for p in here.glob("*.py")}
    # import name -> pip distribution name, where they differ
    dist = {"sklearn": "scikit-learn"}
    imported = set()
    for path in here.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])
    third = {m for m in imported
             if m not in _sys.stdlib_module_names and m not in local and m != "__future__"}
    declared = {_re.split(r"[<>=!~\[ ;#]", line.strip(), maxsplit=1)[0].lower()
                for line in req_text.splitlines()
                if line.strip() and not line.lstrip().startswith("#")}
    return {m for m in third if dist.get(m, m).lower() not in declared}


def test_requirements_cover_imports() -> None:
    """Every third-party import must be declared in requirements.txt.

    `datasets` was imported by attack.build_benign and never listed. Every earlier pod had it
    installed by hand, so nothing failed until D1's first run on a fresh pod -- after OLMo 2
    was already loaded, because the import is lazy. A dependency the code needs but the
    install file omits is invisible until the one machine that installs only what is listed."""
    import pathlib

    req = (pathlib.Path(__file__).resolve().parent / "requirements.txt").read_text()
    missing = _undeclared_imports(req)
    assert not missing, (f"imported but not in requirements.txt: {sorted(missing)}. "
                         f"Add them, or a fresh pod will fail at the first import.")
    # Prove the detector fires on the original bug.
    without = "\n".join(l for l in req.splitlines() if not l.startswith("datasets"))
    assert "datasets" in _undeclared_imports(without), \
        "the requirements check does not catch the missing `datasets` that broke D1"
    print("  requirements: every third-party import is declared — OK")


def test_ledger_records_the_code_that_ran() -> None:
    """A ledger row must name the commit that was LOADED, and flag a change during the run.

    On 2026-09-23 a volume migration replaced .git mid-run, and the row for D1's first
    control named 5370c95 -- a commit without --seed, which cannot have produced the run --
    because the commit was read at the END. Simulated here by swapping git_state between
    load and exit, in a temp results dir so the real ledger is untouched."""
    import json as _json
    import tempfile
    import types

    import runlog

    loaded = dict(runlog.GIT_AT_LOAD)
    saved = runlog.git_state
    runlog.git_state = lambda: dict(loaded, commit="f" * 40, short="fffffff")
    try:
        with tempfile.TemporaryDirectory() as td:
            cfg = types.SimpleNamespace(results_dir=td)
            with runlog.RunRecord("TEST", "smoke_test.py", cfg=cfg):
                pass
            row = _json.loads(open(f"{td}/runs.jsonl").read().splitlines()[-1])
    finally:
        runlog.git_state = saved
    assert row["git"]["commit"] == loaded["commit"], \
        "the row names the commit on disk at EXIT, not the one that was running"
    assert row.get("code_changed_during_run") is True, "a mid-run code change went unflagged"
    assert row["git_at_exit"]["short"] == "fffffff"
    print("  ledger: records the commit that ran, flags a mid-run change — OK")


def test_p1e7r_verdict() -> None:
    """P1-E7r's replication rule on pairs whose answer is known, before any attack runs.

    Each criterion must be able to fail on its own -- a rule that only ever says "replicates"
    is not a test of anything (O-174)."""
    from p1e7r_check import pair_verdict

    atk = {"induce": -5.17, "frozen": 3.4, "wildguard": 0.48}
    ctl = {"induce": 2.9, "frozen": 3.1, "wildguard": 0.95}
    assert pair_verdict(atk, ctl, 1.0)["replicates"], "the P1-E7 pattern itself must replicate"
    for broken, crit in ((dict(atk, induce=0.3), "R1_mapping_destroyed"),
                         (dict(atk, frozen=-2.0), "R2_readout_intact"),
                         (dict(atk, wildguard=0.97), "R4_behaviour_degraded")):
        v = pair_verdict(broken, ctl, 1.0)
        assert not v[crit] and not v["replicates"], f"{crit} cannot fail on its own"
    assert not pair_verdict(atk, dict(ctl, induce=-1.0), 1.0)["R1_mapping_destroyed"], \
        "a control whose own mapping died must not count as a clean contrast"
    assert not pair_verdict(atk, ctl, 0.95)["R3_representation_intact"]
    print("  P1-E7r verdict: each criterion fails on its own, the P1-E7 pattern passes — OK")


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
    test_superseded_citations()
    test_disk_check_is_honoured()
    test_no_pinned_sweep_axis()
    test_cosine_ceiling_discriminates()
    test_degenerate_output_is_detected()
    test_a3b_verdict_rejects_the_false_positive()
    test_runrecord_notes_cannot_destroy_a_run()
    test_judge_bench_pairing_is_exact()
    test_control_is_never_scored_on_its_training_prompts()
    test_d1_verdict_logic()
    test_requirements_cover_imports()
    test_ledger_records_the_code_that_ran()
    test_p1e7r_verdict()
    test_transformer_layers()
    test_data_loads()
    test_refusal_score()
    test_select_l_star()
    test_aggregate_shapes()
    print("ALL PASSED")
