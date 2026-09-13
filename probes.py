"""Linear-probe machinery for P1-E1: is the harmful/harmless distinction READABLE?

The ablation and induce axes ask whether a direction is *actionable*. This module asks
the prior question — whether the distinction is linearly present at all — and whether the
axis carrying it is the SAME one the aligned model refuses along.

Two probes, deliberately:
  mass_mean  — difference of means. Literally the same estimator as the refusal direction,
               so it is the apples-to-apples one, and the only one whose direction is
               comparable by cosine.
  logistic   — a strictly stronger learner. High LR accuracy with low mass-mean accuracy
               means the information is present but NOT along the mean-diff axis, which is
               a different claim.

Accuracy alone proves little: a linear probe separates harmful from harmless in almost any
7B model, base included. The informative quantities are the layer PROFILE (flat-high from
layer 0 means the probe reads surface lexicon, not harmfulness) and the COSINE against a
shuffled-label null (is it the same axis the aligned model uses?).
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import torch

from refusal_direction import _tokenize

logger = logging.getLogger(__name__)

__all__ = ["cache_activations", "mass_mean_direction", "mass_mean_accuracy",
           "logistic_accuracy", "null_directions", "layer_cosines", "length_baseline"]


# ------------------------------------------------------------------ activations

def cache_activations(model, tok, instructions: List[str], template: str, n_eoi: int,
                      batch_size: int = 16) -> torch.Tensor:
    """(N, n_pos, n_layers, d) resid_pre at the last `n_eoi` positions, on CPU float32.

    Same hook point and positions as get_mean_diff, so probe directions are directly
    comparable to refusal directions. Layer 0 is the embedding output — it is the built-in
    surface-feature baseline, not a throwaway.

    On left padding: transformers derives position_ids as arange(seq_len) when none are
    passed, so a left-padded sequence's real tokens sit at shifted absolute positions that
    depend on the batch's longest member. For RoPE models (Mistral/Llama) this is a no-op:
    q_i . k_j depends only on (i - j), every real token shifts by the same offset, and v
    carries no positional term — so activations are batch-composition invariant up to
    float error. smoke_test_probes.py asserts that invariance."""
    layers = model.model.layers
    n_layers, d = len(layers), model.config.hidden_size
    positions = list(range(-n_eoi, 0))
    out = torch.empty((len(instructions), n_eoi, n_layers, d), dtype=torch.float32)

    buf: dict[int, torch.Tensor] = {}
    handles = []
    for li, layer in enumerate(layers):
        def pre_hook(module, inp, li=li):
            a = inp[0] if isinstance(inp, tuple) else inp
            buf[li] = a[:, positions, :].detach().to(torch.float32).cpu()
        handles.append(layer.register_forward_pre_hook(pre_hook))
    try:
        for i in range(0, len(instructions), batch_size):
            chunk = instructions[i:i + batch_size]
            enc = _tokenize(tok, chunk, template)
            model(input_ids=enc.input_ids.to(model.device),
                  attention_mask=enc.attention_mask.to(model.device))
            for li in range(n_layers):
                out[i:i + len(chunk), :, li, :] = buf[li]
            buf.clear()
    finally:
        for h in handles:
            h.remove()
    return out


# ------------------------------------------------------------------ pure probe math

def mass_mean_direction(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """(d,) unnormalised difference of means. Same estimator as the refusal direction."""
    return pos.mean(axis=0) - neg.mean(axis=0)


def mass_mean_accuracy(tr_pos: np.ndarray, tr_neg: np.ndarray,
                       te_pos: np.ndarray, te_neg: np.ndarray) -> float:
    """Project test points onto the TRAIN direction; threshold at the midpoint between the
    two projected train means. No test information touches the direction or the threshold."""
    d = mass_mean_direction(tr_pos, tr_neg)
    n = np.linalg.norm(d)
    if n < 1e-12:
        return 0.5
    u = d / n
    thr = 0.5 * (tr_pos @ u).mean() + 0.5 * (tr_neg @ u).mean()
    correct = int((te_pos @ u > thr).sum()) + int((te_neg @ u <= thr).sum())
    return correct / (len(te_pos) + len(te_neg))


def logistic_accuracy(tr_pos: np.ndarray, tr_neg: np.ndarray,
                      te_pos: np.ndarray, te_neg: np.ndarray, seed: int = 42) -> float:
    """L2 logistic regression on standardised features. n (256) << d (4096), so the fit
    separates the train set perfectly; regularisation is what makes the TEST number mean
    something."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.concatenate([tr_pos, tr_neg]).astype(np.float64)
    y = np.concatenate([np.ones(len(tr_pos)), np.zeros(len(tr_neg))])
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=0.1, max_iter=2000, random_state=seed))
    clf.fit(X, y)
    Xt = np.concatenate([te_pos, te_neg]).astype(np.float64)
    yt = np.concatenate([np.ones(len(te_pos)), np.zeros(len(te_neg))])
    return float((clf.predict(Xt) == yt).mean())


def length_baseline(pos_lens: np.ndarray, neg_lens: np.ndarray,
                    te_pos_lens: np.ndarray, te_neg_lens: np.ndarray) -> float:
    """Accuracy from TOKEN COUNT alone — the floor the activation probe must beat.
    If harmful prompts are simply longer, a 'harmfulness' probe is partly a length probe."""
    from sklearn.linear_model import LogisticRegression

    X = np.concatenate([pos_lens, neg_lens]).reshape(-1, 1).astype(np.float64)
    y = np.concatenate([np.ones(len(pos_lens)), np.zeros(len(neg_lens))])
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    Xt = np.concatenate([te_pos_lens, te_neg_lens]).reshape(-1, 1).astype(np.float64)
    yt = np.concatenate([np.ones(len(te_pos_lens)), np.zeros(len(te_neg_lens))])
    return float((clf.predict(Xt) == yt).mean())


def null_directions(acts: np.ndarray, n_pos_examples: int, k: int,
                    rng: np.random.Generator) -> np.ndarray:
    """(k, d) mean-diff directions from RANDOM label splits of the same pooled activations.

    The null for "is this cosine large?". Two directions fit on shuffled labels encode
    nothing, but share the data's geometry — so their cosines give the band a real cosine
    has to clear. An analytic 1/sqrt(d) null would be too narrow: activations occupy a much
    lower effective dimension than d."""
    n = len(acts)
    out = np.empty((k, acts.shape[1]), dtype=np.float32)
    for i in range(k):
        idx = rng.permutation(n)
        p, q = idx[:n_pos_examples], idx[n_pos_examples:]
        out[i] = acts[p].mean(axis=0) - acts[q].mean(axis=0)
    return out


def layer_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine between two (..., d) stacks, broadcast over leading axes."""
    an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return (an * bn).sum(axis=-1)
