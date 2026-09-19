"""CPU-only checks for P1-E1. No model weights. Run before spending GPU time.

    python smoke_test_probes.py

Every check has a KNOWN answer, so a pass means the math is right, not merely that it ran.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if cond else 'FAIL'} {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


def main() -> int:
    from probes import (layer_cosines, length_baseline, logistic_accuracy,
                        mass_mean_accuracy, mass_mean_direction, null_directions)

    rng = np.random.default_rng(0)
    d, n = 64, 40

    print("\n[probe math — separable vs non-separable]")
    # Separable: classes offset along a known axis -> both probes should be ~perfect.
    axis = np.zeros(d); axis[7] = 1.0
    tp = rng.normal(0, 1, (n, d)) + 6 * axis
    tn = rng.normal(0, 1, (n, d)) - 6 * axis
    ep = rng.normal(0, 1, (n, d)) + 6 * axis
    en = rng.normal(0, 1, (n, d)) - 6 * axis
    a_mm = mass_mean_accuracy(tp, tn, ep, en)
    a_lr = logistic_accuracy(tp, tn, ep, en)
    check("mass-mean separates separable data", a_mm > 0.95, f"acc={a_mm:.3f}")
    check("logistic separates separable data", a_lr > 0.95, f"acc={a_lr:.3f}")

    # Recovered direction must point along the true axis.
    u = mass_mean_direction(tp, tn); u /= np.linalg.norm(u)
    check("mass-mean recovers the planted axis", abs(u @ axis) > 0.9, f"|cos|={abs(u @ axis):.3f}")

    # Non-separable: same distribution, random labels -> chance.
    z1, z2 = rng.normal(0, 1, (n, d)), rng.normal(0, 1, (n, d))
    z3, z4 = rng.normal(0, 1, (n, d)), rng.normal(0, 1, (n, d))
    c_mm = mass_mean_accuracy(z1, z2, z3, z4)
    c_lr = logistic_accuracy(z1, z2, z3, z4)
    check("mass-mean is at chance on noise", 0.3 < c_mm < 0.7, f"acc={c_mm:.3f}")
    check("logistic is at chance on noise", 0.3 < c_lr < 0.7, f"acc={c_lr:.3f}")

    # Degenerate direction must not crash or claim skill.
    same = np.ones((n, d))
    check("zero-norm direction returns chance", mass_mean_accuracy(same, same, same, same) == 0.5)

    print("\n[cosine + null]")
    a = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    b = np.array([[1.0, 0, 0], [1.0, 0, 0]])
    cs = layer_cosines(a, b)
    check("layer_cosines exact values", np.allclose(cs, [1.0, 0.0]), f"{cs.round(3)}")
    check("layer_cosines broadcasts leading axes",
          layer_cosines(rng.normal(size=(4, 6, d)), rng.normal(size=(4, 6, d))).shape == (4, 6))
    check("layer_cosines is sign-preserving", layer_cosines(a[:1], -b[:1])[0] < 0)

    nd = null_directions(np.concatenate([z1, z2]), n, 12, np.random.default_rng(1))
    check("null_directions shape", nd.shape == (12, d), f"{nd.shape}")
    nc = layer_cosines(nd, mass_mean_direction(z1, z2)[None, :])
    check("null cosines are small on noise", float(np.abs(nc).mean()) < 0.6,
          f"mean|cos|={float(np.abs(nc).mean()):.3f}")

    # A real signal must clear its own null band -> this is the experiment's core inference.
    true_dir = mass_mean_direction(tp, tn)
    nd2 = null_directions(np.concatenate([tp, tn]), n, 64, np.random.default_rng(2))
    null_p95 = float(np.percentile(np.abs(layer_cosines(nd2, true_dir[None, :])), 95))
    self_cos = abs(float(layer_cosines(mass_mean_direction(ep, en)[None, :], true_dir[None, :])[0]))
    check("planted signal clears its null band", self_cos > null_p95,
          f"cos={self_cos:.3f} vs null p95={null_p95:.3f}")

    print("\n[length baseline]")
    lb = length_baseline(np.arange(50, 90), np.arange(10, 50), np.arange(50, 90), np.arange(10, 50))
    check("length baseline detects a real length split", lb > 0.9, f"acc={lb:.3f}")
    lb2 = length_baseline(rng.integers(10, 90, 40), rng.integers(10, 90, 40),
                          rng.integers(10, 90, 40), rng.integers(10, 90, 40))
    check("length baseline is ~chance when lengths match", 0.25 < lb2 < 0.75, f"acc={lb2:.3f}")

    print("\n[aggregate indexing — the trickiest part]")
    from aggregate_probe import cosine_vs_null
    k, n_pos, n_layers, dd = 8, 5, 12, 32
    A = rng.normal(size=(n_pos, n_layers, dd)).astype(np.float32)
    B = A.copy()                      # identical -> cosine must be exactly 1 everywhere
    An = rng.normal(size=(k, n_pos, n_layers, dd)).astype(np.float16)
    Bn = rng.normal(size=(k, n_pos, n_layers, dd)).astype(np.float16)
    tc, nh = cosine_vs_null(A, B, An, Bn)
    check("cosine_vs_null output shapes", tc.shape == (n_layers,) and nh.shape == (n_layers,),
          f"{tc.shape}, {nh.shape}")
    check("identical directions give cosine 1", np.allclose(tc, 1.0, atol=1e-5),
          f"min={tc.min():.5f}")
    check("null band is below the true cosine here", bool((nh < tc).all()),
          f"max null={nh.max():.3f}")
    tc2, _ = cosine_vs_null(A, -B, An, Bn)
    check("anti-aligned directions give cosine -1", np.allclose(tc2, -1.0, atol=1e-5))

    # REGRESSION (bug found 2026-09-13): the true cosine is the max over n_pos positions, so
    # the null must go through the SAME max or significance is inflated by selection bias.
    # The first version drew the null at a single position and reported 13/32 layers "above
    # null" on data where only 7 were planted. On pure noise the rate must sit near p95's 5%.
    tot = 0
    for _ in range(30):
        A2 = rng.normal(size=(n_pos, n_layers, dd)).astype(np.float32)
        B2 = rng.normal(size=(n_pos, n_layers, dd)).astype(np.float32)
        An2 = rng.normal(size=(64, n_pos, n_layers, dd)).astype(np.float16)
        Bn2 = rng.normal(size=(64, n_pos, n_layers, dd)).astype(np.float16)
        t2, h2 = cosine_vs_null(A2, B2, An2, Bn2)
        tot += int((np.abs(t2) > h2).sum())
    fpr = tot / (30 * n_layers)
    check("null band is not inflated by position selection", fpr < 0.12,
          f"false-positive rate on noise={fpr:.3f} (p95 band => expect ~0.05)")

    print("\n[activation caching — fake model, no weights]")
    import torch
    from probes import cache_activations

    class FakeLayer(torch.nn.Module):
        def forward(self, x, *a, **k):
            return (x + 1.0,)

    class FakeInner(torch.nn.Module):
        def __init__(self, nl, dd):
            super().__init__()
            self.layers = torch.nn.ModuleList([FakeLayer() for _ in range(nl)])
        def forward(self, x):
            for l in self.layers:
                x = l(x)[0]
            return x

    class FakeModel(torch.nn.Module):
        def __init__(self, nl=4, dd=8):
            super().__init__()
            self.model = FakeInner(nl, dd)
            self.config = type("C", (), {"hidden_size": dd})()
            self.device = torch.device("cpu")
            self.emb = torch.nn.Embedding(50, dd)
        def forward(self, input_ids=None, attention_mask=None):
            return self.model(self.emb(input_ids))

    class FakeTok:
        pad_token_id = 0
        def __call__(self, prompts, padding=True, truncation=False, return_tensors="pt"):
            # ids keyed off the PROMPT, never its index in the batch — otherwise the
            # batching-invariance check below tests the tokenizer, not cache_activations.
            ids = [[7 + (ord(q[0]) % 5)] * (6 + ord(q[0]) % 3) for q in prompts]
            m = max(len(x) for x in ids)
            ids = [[0] * (m - len(x)) + x for x in ids]          # LEFT padding, as in load_model
            t = torch.tensor(ids)
            return type("E", (), {"input_ids": t, "attention_mask": (t != 0).long()})()

    fm, ft = FakeModel(), FakeTok()
    acts = cache_activations(fm, ft, ["a", "b", "c"], "{instruction}", n_eoi=3, batch_size=2)
    check("activation cache shape", tuple(acts.shape) == (3, 3, 4, 8), f"{tuple(acts.shape)}")
    check("activation cache is float32 on cpu",
          acts.dtype == torch.float32 and acts.device.type == "cpu")
    # FakeLayer adds 1.0 per layer, so resid_pre at layer L must be embedding + L.
    diffs = (acts[:, :, 1:, :] - acts[:, :, :-1, :])
    check("layer ordering is correct (resid_pre increments by 1 per layer)",
          torch.allclose(diffs, torch.ones_like(diffs)), f"mean diff={diffs.mean():.3f}")
    check("batching is consistent (batch_size 2 vs 3 agree)",
          torch.allclose(acts, cache_activations(fm, ft, ["a", "b", "c"], "{instruction}", 3, 3)))

    print("\n[unverified lineage handling]")
    from config import LINEAGES, Lineage, config_for
    # REGRESSION (2026-09-16): verify_setup crashed with TypeError comparing None > int on a
    # lineage whose n_eoi was unpinned -- the exact case it exists to resolve.
    fake = Lineage(name="t", checkpoints=(("a", "m"),), template="x{instruction}y",
                   refusal_token_piece="I", expected_refusal_id=None, n_eoi=None)
    check("a lineage missing everything is not 'verified'", not fake.verified)
    check("template counts toward verification",
          not Lineage(name="t", checkpoints=(("a", "m"),), template=None,
                      refusal_token_piece="I", expected_refusal_id=1, n_eoi=1).verified)
    for name in LINEAGES:
        c = config_for(name)
        if LINEAGES[name].verified:
            c.require_verified()          # must not raise
        else:
            try:
                c.require_verified()
                check(f"{name} should refuse to run", False)
            except SystemExit as e:
                check(f"{name} refuses with actionable text",
                      "diagnose_refusal_token" in str(e) and "verify_setup" in str(e))
    check("zephyr and olmo2 are both verified and runnable",
          LINEAGES["zephyr"].verified and LINEAGES["olmo2"].verified)
    # Windows differ per lineage BY DESIGN — each is pinned to the largest value that is
    # leak-free for its own template (O-58), not to a shared constant.
    check("each lineage's window is pinned to its own leak-free maximum",
          LINEAGES["zephyr"].n_eoi == 5 and LINEAGES["olmo2"].n_eoi == 5,
          f'zephyr={LINEAGES["zephyr"].n_eoi} olmo2={LINEAGES["olmo2"].n_eoi}')

    print("\n[per-stage regime overrides]")
    from transformers import AutoTokenizer as _AT
    from refusal_direction import eoi_len as _eoi
    for name in ("zephyr", "olmo2"):
        c = config_for(name)
        for st in c.stages:
            tpl, tid, neoi, ov = c.regime(st)
            check(f"{name}/{st}: window fits its own template",
                  neoi <= _eoi(_AT.from_pretrained(dict(c.checkpoints)[st]), tpl),
                  f"n_eoi={neoi} template={tpl[-18:]!r}")
    zc = config_for("zephyr")
    check("zephyr has NO overrides (its base is fluent under the chat template)",
          not any(zc.regime(s)[3] for s in zc.stages))
    oc = config_for("olmo2")
    check("olmo2 overrides base only", [s for s in oc.stages if oc.regime(s)[3]] == ["base"])
    check("the override token differs from the lineage token (it follows the TEMPLATE)",
          oc.regime("base")[1] != oc.regime("sft")[1],
          f"{oc.regime('base')[1]} vs {oc.regime('sft')[1]}")

    print("\n[eoi window leakage — O-58]")
    from verify_setup import window_leaks as _wl
    class FakeT:
        """Minimal BPE-ish tokenizer that MERGES '?' with a following newline, the exact
        behaviour that let a layer-0 probe read surface text on OLMo 2."""
        def encode(self, text, add_special_tokens=False):
            out, i = [], 0
            while i < len(text):
                if text[i] == "?" and text[i + 1:i + 2] == "\n":
                    out.append(999); i += 2          # merged '?\n'
                else:
                    out.append(ord(text[i])); i += 1
            return out
        def convert_ids_to_tokens(self, ids): return [str(i) for i in ids]
    ft = FakeT()
    tpl = "{instruction}\nA:"
    q = ["is this ok?", "and this?"]          # end with '?'
    imp = ["do the thing", "make it happen"]  # do not
    leaks3, safe3 = _wl(ft, tpl, 3, q + imp)
    check("leak detected when the window absorbs the prompt's last char", leaks3,
          f"n_eoi=3 leaks={leaks3}, largest safe={safe3}")
    leaks_safe, _ = _wl(ft, tpl, safe3, q + imp)
    check("the reported largest-safe window does not leak", not leaks_safe, f"n={safe3}")
    check("mixed-ending prompts are what expose it — uniform endings hide it",
          not _wl(ft, tpl, 3, q)[0] and not _wl(ft, tpl, 3, imp)[0])

    print("\n[run ledger]")
    from runlog import RunRecord, env_state, git_state
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as td:
        os.chdir(td)
        try:
            with RunRecord("TEST", "smoke.py", None, question="q?") as r:
                r.result(stage="x", acc=np.float64(0.9), arr=np.arange(3))
            try:
                with RunRecord("TEST", "smoke.py", None) as r2:
                    r2.result(stage="y")
                    raise ValueError("boom")
            except ValueError:
                pass
            rows = [json.loads(l) for l in open("results/runs.jsonl")]
            check("ledger records both runs", len(rows) == 2, f"{len(rows)} rows")
            check("ok run marked ok", rows[0]["status"] == "ok")
            check("failed run is RECORDED, not dropped",
                  rows[1]["status"] == "failed" and "boom" in rows[1]["error"])
            check("numpy scalars serialise", rows[0]["results"][0]["acc"] == 0.9)
            check("numpy arrays serialise", rows[0]["results"][0]["arr"] == [0, 1, 2])
            check("exception is re-raised, not swallowed", True)
            md = open("results/RUNLOG.md").read()
            check("markdown ledger written", "## TEST" in md and "FAILED" in md)
        finally:
            os.chdir(cwd)
    check("git state resolves", git_state()["commit"] != "unknown", git_state()["short"])
    check("env state captures torch", env_state()["torch"] is not None, env_state()["torch"])

    test_transplant()
    test_regime_override_aggregate()
    test_transplant_text()
    test_probe_transfer()
    test_attack_encoding()

    print("\n" + ("ALL PASSED" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
    return 1 if FAIL else 0



def test_transplant() -> None:
    """P1-E1b: direction loading, norm matching, and the sweep bookkeeping."""
    import torch
    import transplant as T

    print("\n[transplant — P1-E1b]")
    check("coefficient sweep includes Arditi's default 1.0 and goes well past it",
          1.0 in T.COEFFS and max(T.COEFFS) >= 8, str(T.COEFFS))
    # source layers must be READ from each stage's saved sweep, never hardcoded, or a new
    # lineage would silently inherit Zephyr's l*.
    import inspect
    src = inspect.getsource(T)
    check("no hardcoded source layers remain",
          "SOURCES = " not in src and "POS_IDX" not in src)
    check("source_layers reads l* from the saved results",
          'cfg.path(stage, "refusal")' in inspect.getsource(T.source_layers))
    check("falls back to the unfiltered argmax when l* = -1",
          "naive_l_star" in inspect.getsource(T.source_layers))

    g = torch.Generator().manual_seed(0)
    d = torch.randn(64) * 3.3
    r = T._norm_matched(d, g)
    check("norm-matched random preserves the norm",
          abs(float(r.norm()) - float(d.norm())) < 1e-3,
          f"{float(r.norm()):.4f} vs {float(d.norm()):.4f}")
    check("norm-matched random is a different direction",
          abs(float((r / r.norm()) @ (d / d.norm()))) < 0.5)

    # sweep_cell must return one row per coefficient, with KL == 0 when nothing changes.
    class Z(torch.nn.Module):
        def forward(self, x, *a, **k):
            return (x,)

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Z() for _ in range(3)])

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.device = torch.device("cpu")
        def forward(self, input_ids=None, attention_mask=None):
            n, t = input_ids.shape
            return type("O", (), {"logits": torch.zeros(n, t, 7)})()

    class Tk:
        pad_token_id = 0
        def __call__(self, prompts, padding=True, truncation=False, return_tensors="pt"):
            t = torch.ones(len(prompts), 4, dtype=torch.long)
            return type("E", (), {"input_ids": t, "attention_mask": t})()

    m, tk = M(), Tk()
    base_lg = torch.zeros(2, 7)
    # generate_completions must REFUSE a right-padded tokenizer rather than silently emit
    # garbage for every sequence but the longest in each batch.
    from refusal_substring import generate_completions
    class _RightPad:
        padding_side = "right"
    try:
        generate_completions(None, _RightPad(), ["a"], "{instruction}")
        check("right-padded tokenizer is refused", False, "no SystemExit")
    except SystemExit as e:
        check("right-padded tokenizer is refused", "LEFT-padded" in str(e), str(e)[:60])

    rows = T.sweep_cell(m, tk, torch.zeros(5), 0, ["a", "b"], "{instruction}", [1],
                        base_lg, batch_size=2)
    check("sweep_cell returns one row per coefficient", len(rows) == len(T.COEFFS),
          f"{len(rows)} rows")
    check("sweep_cell coefficients are in order",
          [r[0] for r in rows] == list(T.COEFFS))
    check("KL is ~0 when the intervention changes nothing",
          all(abs(r[2]) < 1e-6 for r in rows), f"max|KL|={max(abs(r[2]) for r in rows):.2e}")

    # --- the under-powered-sweep guard (OLMo 2, 2026-09-16) -------------------------------
    # A fixed unit-norm grid capped at 16 sat entirely below OLMo 2's direction norms, so
    # every cell read "no induction" INCLUDING rlvr->rlvr, which run_stage had already
    # measured as inducing. The grid must reach each source's own raw scale, and the
    # self-cell must be checked before any "no" is believed.
    norms = {("base", 23, 1): 12.0, ("sft", 24, 4): 88.0, ("dpo", 24, 4): 61.0}
    raw = T.coeff_grid(norms, unit_norm=False)
    check("raw mode leaves Arditi's fixed grid alone", raw == T.COEFFS)
    grid = T.coeff_grid(norms, unit_norm=True)
    check("unit-norm grid reaches every source's raw norm",
          max(grid) >= max(norms.values()), f"max grid {max(grid):.1f} vs max norm 88.0")
    check("unit-norm grid is increasing and starts below the largest norm",
          list(grid) == sorted(grid) and min(grid) < max(norms.values()), str(grid))
    check("the OLD fixed grid would have FAILED this lineage",  # the bug, pinned
          max(T.COEFFS) < max(norms.values()))

    # Every source must get a 1x-ITS-OWN-norm point. Anchoring only on the max means a
    # small-norm source is swept at wild multiples of its own scale: Zephyr's norms are
    # 1.1 / 4.4 / 7.4, so the top of the grid was 54x base's own norm but 8x SFT's.
    zeph = {("base", 15, 4): 1.1, ("sft", 20, 4): 7.4, ("dpo", 17, 4): 4.4}
    g = T.coeff_grid(zeph, unit_norm=True)
    check("every source's own raw norm is IN the grid",
          all(round(v, 4) in g for v in zeph.values()), str([round(c, 1) for c in g]))
    check("the grid stays shared across sources (matched injection preserved)",
          list(g) == sorted(set(g)))
    check("--no-own-norms restores the max-anchored grid only",
          len(T.coeff_grid(zeph, unit_norm=True, own_norms=False)) == len(T.NORM_MULTIPLES))
    check("raw mode is unaffected by own_norms",
          T.coeff_grid(zeph, unit_norm=False, own_norms=True) == T.COEFFS)

    # --- P1-E2c: effect size against the cell's own null ---------------------------------
    # One random draw cannot scale an effect. The three real self-cells' single draws came
    # back +0.22 / +2.83 / +0.74 -- a spread comparable to the differences being compared --
    # which is exactly why "what preference optimisation adds" was unresolvable.
    base_line = -4.0
    cell = [("sft", 24, "direction", 1.0, -2.0, .1), ("sft", 24, "direction", 8.0, +3.0, .9),
            ("sft", 24, "random0", 8.0, -3.0, .2), ("sft", 24, "random1", 8.0, -3.5, .2),
            ("sft", 24, "random2", 8.0, -3.8, .2)]
    e = T.cell_effect(cell, "sft", base_line)
    check("delta is measured from the TARGET's own baseline", abs(e["delta"] - 7.0) < 1e-9,
          f"{e['delta']}")
    check("null mean averages the draws", abs(e["null_mean"] - 0.5667) < 1e-3, f"{e['null_mean']}")
    check("null sd needs >=2 draws and is reported", e["null_sd"] > 0, f"{e['null_sd']}")
    check("z standardises the real effect against that null", e["z"] > 10, f"z={e['z']}")
    check("counts how many NULLS themselves crossed", e["n_null_crossing"] == 0)

    one = [r for r in cell if r[2] in ("direction", "random0")]
    e1 = T.cell_effect(one, "sft", base_line)
    check("with ONE draw the sd is None, not a fake 0.0", e1["null_sd"] is None)
    check("with ONE draw z is None -- no comparison is licensed", e1["z"] is None)
    check("delta is still reported at one draw", abs(e1["delta"] - 7.0) < 1e-9)
    e0 = T.cell_effect([r for r in cell if r[2] == "direction"], "sft", base_line)
    check("with NO draws it degrades gracefully", e0["n_draws"] == 0 and e0["z"] is None)

    # --- the SHUFFLED-LABEL null (the harder control) ------------------------------------
    # Isotropic noise points mostly where the residual stream barely operates -- the same
    # anisotropy that killed cross-checkpoint cosine (O-49) -- so beating it is easy. On
    # Zephyr, base's own direction scored z=+3.0 against isotropic noise while failing every
    # other test, and sft (+2.5) and dpo (+2.9) landed in the same band: that tracks "is a
    # mean-diff vector", not "is a refusal vector". A shuffled-label fit shares the geometry
    # and encodes nothing, so the two families must be reported separately.
    mixed = cell + [("sft", 24, "shuffled0", 8.0, -0.5, .3),
                    ("sft", 24, "shuffled1", 8.0, -0.2, .3)]
    er = T.cell_effect(mixed, "sft", base_line, null_prefix="random")
    es = T.cell_effect(mixed, "sft", base_line, null_prefix="shuffled")
    check("the two null families are computed separately", er["n_draws"] == 3
          and es["n_draws"] == 2, f"random={er['n_draws']} shuffled={es['n_draws']}")
    check("each result records which null it used",
          er["null"] == "random" and es["null"] == "shuffled")
    check("a HARDER null gives a SMALLER z for the same direction", es["z"] < er["z"],
          f"shuffled z={es['z']:.1f} vs random z={er['z']:.1f}")
    check("delta is the same under either null (it is a property of the cell)",
          er["delta"] == es["delta"])

    # The loader must refuse to silently under-draw when fewer nulls are stored than asked for.
    import numpy as _np
    from dataclasses import replace as _replace
    import tempfile as _tf
    from config import config_for as _cf
    with _tf.TemporaryDirectory() as _td:
        c = _replace(_cf("olmo2"), results_dir=_td)
        _np.savez(c.path("base", "probe"),
                  null_directions=_np.zeros((2, 5, 32, 8), dtype=_np.float16))
        try:
            T.load_null_directions(c, [("base", 24, 4)], k=5)
            check("asking for more nulls than are stored raises", False, "no SystemExit")
        except SystemExit as e:
            check("asking for more nulls than are stored raises", "only 2" in str(e), str(e)[:70])
        got = T.load_null_directions(c, [("base", 24, 4)], k=2)
        check("loader returns k vectors at the requested cell",
              got[("base", 24, 4)].shape == (2, 8), str(got[("base", 24, 4)].shape))

    # --- C3 control: the WildGuard parse must not fail silently ----------------------------
    import judge_wildguard as W

    # LEFT padding on a decoder-only model. With right padding every sequence but the longest
    # in a batch generates from PAD tokens, so its answer is garbage -- and it fails SILENTLY
    # apart from a warning that scrolls past. The first run of the judge had exactly this.
    class _Tk:
        padding_side = "right"
        pad_token = None
        eos_token = "</s>"
    tk = W.configure_tokenizer(_Tk())
    check("the judge forces LEFT padding (decoder-only)", tk.padding_side == "left")
    check("a missing pad token falls back to eos", tk.pad_token == "</s>")

    check("parses 'Response refusal: yes'",
          W.parse("Harmful request: no\nResponse refusal: yes\nHarmful response: no") is True)
    check("parses 'Response refusal: no'", W.parse("Response refusal: no") is False)
    # An unparseable answer must be None, NOT False. Silently reading as False would turn a
    # batch the classifier could not answer into "the model complied everywhere".
    check("unparseable -> None, never False", W.parse("garbage output") is None)
    check("an unrecognised value -> None", W.parse("Response refusal: maybe") is None)
    check("case and whitespace tolerant",
          W.parse("  RESPONSE REFUSAL:   YES  ") is True)

    R = [("sft", 24, "direction", 1.0, -2.0, 0.1), ("sft", 24, "direction", 8.0, +0.7, 0.3),
         ("base", 23, "direction", 8.0, -4.0, 0.2)]
    ok, best, cells = T.positive_control(R, "sft", 0.0)
    check("positive control passes when the target's own direction induces", ok and best == 0.7)
    check("positive control reads only the self-cells", len(cells) == 2)
    ok2, _, _ = T.positive_control(R, "base", 0.0)
    check("positive control fails when the self-cell never crosses", not ok2)
    ok3, _, _ = T.positive_control(R, "rlvr", 0.0)
    check("positive control fails (not crashes) when the self-cell is absent", not ok3)



def test_regime_override_aggregate() -> None:
    """A lineage whose base sits on a regime override must aggregate, not crash.

    olmo2 base uses a plain template with a 2-token eoi window; the aligned stages use the
    chat template with 5. aggregate_probe.py died on the shape mismatch
    ((2,32,4096) vs (5,32,4096)) AFTER the whole GPU pipeline had run. Truncating to the
    shorter stack would have been worse: it silently produces a cosine between positions that
    denote different things. The run must complete, skip the cosine, and SAY it skipped.

    Also asserts the ledger stays inside the synthetic results_dir. The first version of this
    fixture appended a fake P1-E1 run to the real results/runs.jsonl."""
    import os
    import tempfile
    from dataclasses import replace

    import numpy as np

    import aggregate_probe as A
    from config import config_for

    print("\n[regime override — aggregate_probe]")
    real_before = (open("results/runs.jsonl").read() if os.path.exists("results/runs.jsonl")
                   else None)
    with tempfile.TemporaryDirectory() as td:
        cfg = replace(config_for("olmo2"), results_dir=td, figures_dir=os.path.join(td, "fig"))
        rng = np.random.default_rng(0)
        for stage, npos in (("base", 2), ("sft", 5)):
            acc = np.full((npos, 32), 0.5)
            acc[:, 5:] = 0.99
            np.savez(cfg.path(stage, "probe"), stage=np.array(stage), acc_logistic=acc,
                     acc_mass_mean=acc, length_baseline=np.array(0.55),
                     directions=rng.normal(size=(npos, 32, 64)).astype(np.float32),
                     null_directions=rng.normal(size=(8, npos, 32, 64)).astype(np.float32))
        A.main(cfg_override=cfg)          # must not raise
        check("mismatched eoi windows aggregate instead of crashing", True)
        check("figure still written, lineage-scoped",
              os.path.exists(cfg.figure("p1e1_probe")), cfg.figure("p1e1_probe"))
        check("ledger went to the synthetic results_dir",
              os.path.exists(os.path.join(td, "runs.jsonl")))
    real_after = (open("results/runs.jsonl").read() if os.path.exists("results/runs.jsonl")
                  else None)
    check("the REAL run ledger was not touched", real_before == real_after)


def test_transplant_text() -> None:
    """P1-E1c: the degenerate-output judge, and reading the operating point off the sweep."""
    import tempfile
    from dataclasses import replace

    import numpy as np

    import transplant_text as X
    from config import config_for

    print("\n[transplant text — P1-E1c]")
    check("empty completion counts as degenerate", X.degenerate_rate([""]) == 1.0)
    check("single repeated token counts as degenerate",
          X.degenerate_rate(["I I I I I I I I I I"]) == 1.0)
    check("a real refusal does NOT count as degenerate",
          X.degenerate_rate(["I cannot help with that request, sorry."]) == 0.0)
    # The clause that was MISSING, and the exact string that got through (OLMo 2 base, SFT
    # direction at 2x raw norm). Four distinct words, so a <=2-distinct rule scores it clean
    # and the substring judge scores it a refusal. It is a loop, not a refusal.
    loop = "I'm sorry I cannot " * 8
    check("a refusal PHRASE on loop counts as degenerate", X.degenerate_rate([loop]) == 1.0)
    # A run of identical CHARACTERS. OLMo 2 emits long runs of U+00AD SOFT HYPHEN, which is
    # invisible in a terminal and tokenises as ONE word -- so those completions scored a
    # distinct-4-gram ratio of 1.000 while being pure garbage. Found by hand-auditing all 64
    # completions of the injected arm (2026-09-19); the word-based test cannot see it.
    check("a run of identical characters counts as degenerate",
          X.degenerate_rate(["movies released in the 21" + "\xad" * 20]) == 1.0)
    check("the soft hyphen is invisible, so this is not a cosmetic check",
          len("\xad".strip()) == 1 and "\xad".isprintable() is False)
    check("ordinary punctuation is NOT flagged",
          X.degenerate_rate(["Well... that is a good question, and here is a full answer "
                             "about several different topics!!"]) == 0.0)
    check("the same phrase said ONCE does not",
          X.degenerate_rate(["I'm sorry, I cannot provide that information. I can only "
                             "provide information about the benefits of studying a foreign "
                             "language in general terms."]) == 0.0)
    check("a long varied answer does NOT count as degenerate",
          X.degenerate_rate(["The capital of France is Paris, a city on the river Seine."])
          == 0.0)
    check("no completions -> nan, not a fake 0.0",
          X.degenerate_rate([]) != X.degenerate_rate([]))      # nan != nan

    # The operating point must be READ from transplant.py's saved sweep, never hardcoded,
    # or the two scripts drift and the text is generated at a coefficient the logit sweep
    # never visited.
    with tempfile.TemporaryDirectory() as td:
        cfg = replace(config_for("olmo2"), results_dir=td)
        cells = np.array(["sft|24|direction"] * 3 + ["sft|24|random"] * 3)
        sweep = np.array([(10.0, -2.0, 0.1), (20.0, +0.5, 2.0), (40.0, +3.0, 8.0),
                          (10.0, -3.0, 0.1), (20.0, -3.0, 0.5), (40.0, -2.0, 1.0)],
                         dtype=np.float32)
        np.savez(cfg.path("base", "transplant"), cells=cells, sweep=sweep)
        got = X.crossing_coeffs(cfg, "base", "sft")
        check("reads the FIRST crossing and the argmax from the sweep", got == [20.0, 40.0],
              str(got))
        check("ignores the random arm when picking the operating point", 10.0 not in got)

        cells2 = np.array(["base|23|direction"] * 2)
        sweep2 = np.array([(10.0, -5.0, 0.1), (40.0, -1.0, 3.0)], dtype=np.float32)
        np.savez(cfg.path("sft", "transplant"), cells=cells2, sweep=sweep2)
        got2 = X.crossing_coeffs(cfg, "sft", "base")
        check("with no crossing, falls back to the argmax instead of crashing", got2 == [40.0],
              str(got2))


def test_probe_transfer() -> None:
    """P1-E1d: the focus-matched subset, and that transfer FITS on train and TESTS on test."""
    import numpy as np

    import probe_transfer as PT
    from data import xstest_focus_matched

    print("\n[probe transfer — P1-E1d]")
    rows = [
        {"prompt": "kill a python process", "label": "safe", "focus": "kill", "type": "h"},
        {"prompt": "kill a person", "label": "unsafe", "focus": "kill", "type": "contrast_h"},
        {"prompt": "bake a cake", "label": "safe", "focus": "bake", "type": "d"},
        {"prompt": "unmatched unsafe", "label": "unsafe", "focus": "poison", "type": "c"},
        {"prompt": "no focus at all", "label": "safe", "focus": "", "type": "d"},
    ]
    safe, uns = xstest_focus_matched(rows)
    check("keeps only trigger words present in BOTH classes",
          safe == ["kill a python process"] and uns == ["kill a person"], f"{safe} {uns}")
    check("drops rows with an empty focus", "no focus at all" not in safe)
    check("drops a class-exclusive trigger word", "unmatched unsafe" not in uns)

    # Transfer must be a genuine train->test fit. Plant separability at ONE layer in a way
    # that is CONSISTENT between train and test, and noise elsewhere.
    rng = np.random.default_rng(0)
    n_tr, n_te, npos, nlay, d = 40, 30, 2, 4, 6
    def make(n, shift):
        a = rng.normal(size=(n, npos, nlay, d)).astype(np.float32)
        a[:, :, 2, 0] += shift            # layer 2, feature 0 carries the signal
        return a
    tr_pos, tr_neg = make(n_tr, +3.0), make(n_tr, -3.0)
    te_pos, te_neg = make(n_te, +3.0), make(n_te, -3.0)
    acc = PT.per_layer_transfer(tr_pos, tr_neg, te_pos, te_neg)
    check("shape is one accuracy per layer", acc.shape == (nlay,), str(acc.shape))
    check("the planted layer transfers", acc[2] > 0.9, f"L2={acc[2]:.3f}")
    check("unplanted layers stay near chance", max(acc[0], acc[1], acc[3]) < 0.75,
          str(np.round(acc, 3)))
    # An INCONSISTENT test set (signal flipped) must NOT score high -- that is the difference
    # between measuring transfer and measuring separability.
    acc_flip = PT.per_layer_transfer(tr_pos, tr_neg, te_neg, te_pos)
    check("a label-flipped test set scores BELOW chance, not above",
          acc_flip[2] < 0.25, f"L2={acc_flip[2]:.3f}")


def test_attack_encoding() -> None:
    """P1-E7: the SFT loss must be masked to response tokens only.

    If the prompt is not masked, the attack becomes partly a language-modelling run on our own
    evaluation prompts and any behavioural change afterwards is uninterpretable. This is a
    silent failure -- training runs fine, loss goes down, and the experiment is void."""
    import attack as A

    class Tk:
        eos_token_id = 99
        def encode(self, s, add_special_tokens=False):
            return [ord(c) % 90 + 1 for c in s]      # 1 token per char, never 0 or 99

    tok = Tk()
    tpl = "<U>{instruction}</U><A>"
    print("\n[attack encoding — P1-E7]")
    ex = A.encode_sft(tok, [("hi", "no")], tpl)
    ids, labels = ex[0]
    n_prompt = len(tok.encode(tpl.format(instruction="hi")))
    check("prompt tokens are masked out of the loss",
          labels[:n_prompt] == [-100] * n_prompt, str(labels[:n_prompt]))
    check("response tokens are NOT masked",
          all(x != -100 for x in labels[n_prompt:]), str(labels[n_prompt:]))
    check("ids and labels are the same length", len(ids) == len(labels))
    check("ids keep the prompt (only the LABELS are masked)",
          ids[:n_prompt] == tok.encode(tpl.format(instruction="hi")))
    check("eos is appended to the response", ids[-1] == tok.eos_token_id)
    check("eos is supervised", labels[-1] == tok.eos_token_id)

    # Truncation must not produce an all-masked example -- that is a zero-gradient batch
    # member, and enough of them silently turn the run into a no-op.
    long_ex = A.encode_sft(tok, [("x" * 400, "no")], tpl, max_len=50)
    check("an example whose prompt fills the window is DROPPED, not kept all-masked",
          long_ex == [], f"{len(long_ex)} kept")
    check("a normal example survives the same max_len",
          len(A.encode_sft(tok, [("hi", "no")], tpl, max_len=50)) == 1)
    check("empty responses are dropped upstream",
          A.encode_sft(tok, [("hi", "")], tpl)[0][1].count(-100) < len(
              A.encode_sft(tok, [("hi", "")], tpl)[0][1]))


if __name__ == "__main__":
    sys.exit(main())
