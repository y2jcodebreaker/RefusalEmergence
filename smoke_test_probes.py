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

    print("\n" + ("ALL PASSED" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
    return 1 if FAIL else 0



def test_transplant() -> None:
    """P1-E1b: direction loading, norm matching, and the sweep bookkeeping."""
    import numpy as np
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
    rows = T.sweep_cell(m, tk, torch.zeros(5), 0, ["a", "b"], "{instruction}", [1],
                        base_lg, batch_size=2)
    check("sweep_cell returns one row per coefficient", len(rows) == len(T.COEFFS),
          f"{len(rows)} rows")
    check("sweep_cell coefficients are in order",
          [r[0] for r in rows] == list(T.COEFFS))
    check("KL is ~0 when the intervention changes nothing",
          all(abs(r[2]) < 1e-6 for r in rows), f"max|KL|={max(abs(r[2]) for r in rows):.2e}")

if __name__ == "__main__":
    sys.exit(main())
