"""Verify the tokenizer-level assumptions BEFORE burning GPU time. No model weights needed.

    python verify_setup.py

Checks, across every checkpoint in config.checkpoints:
  1. the torch/transformers model-loading path actually imports (not just tokenizers).
  2. refusal_token_piece resolves to the SAME id in every stage.
  3. the pinned n_eoi window is safe (<= the shortest tokenizer-derived eoi_len).
  4. reports tokenizer-derived eoi_len per stage (these legitimately DIFFER — that is
     exactly why n_eoi is pinned rather than derived).
  5. vocab sizes match (a shared vocab is what makes the cross-stage comparison valid).

SCOPE, honestly stated: this file only proves the tokenizers AGREE with each other. It
cannot prove the chosen token is the one the models actually EMIT — an earlier version
happily passed while scoring id 315, which the models emit with p ~= 1e-5. Only
diagnose_refusal_token.py (which needs weights) can establish that. Re-run it whenever
refusal_token_piece, the template, or the checkpoint list changes.
"""

from __future__ import annotations

import sys

from transformers import AutoTokenizer

from config import DEFAULT
from refusal_direction import eoi_len, resolve_refusal_token


def check_torch_backend() -> bool:
    """Recent transformers DISABLES its PyTorch backend on torch<2.5 with only a log line.
    Tokenizer-only code still works, so this stays silent until from_pretrained() fails --
    after the model downloads have already been paid for. Catch it here instead."""
    import torch
    from transformers.utils import is_torch_available

    if not is_torch_available():
        print(f"FAIL: transformers has DISABLED its PyTorch backend (torch=={torch.__version__}).\n"
              f"      run_stage.py will fail at from_pretrained(). Fix:\n"
              f"      pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124")
        return False
    print(f"OK  torch {torch.__version__} | transformers torch backend enabled | "
          f"cuda={torch.cuda.is_available()}")

    # P1-E1 imports sklearn lazily inside the probe functions, so a missing install would
    # only surface AFTER the activation caches were built. Fail here instead.
    try:
        import sklearn  # noqa: F401
        print(f"OK  scikit-learn {sklearn.__version__} (needed by P1-E1 probes)")
    except ImportError:
        print("WARN scikit-learn missing — run_stage.py is fine, but probe_representation.py\n"
              "      (P1-E1) will fail after caching activations. Fix: pip install scikit-learn")

    # Force the LAZY import of the model class run_stage.py actually uses. transformers
    # resolves these on first access, so a broken optional dep (classically a torchvision
    # built against a different torch -> "operator torchvision::nms does not exist", reached
    # via modeling_utils -> loss utils -> image_utils -> torchvision.io) stays invisible to
    # any tokenizer-only check and only explodes at from_pretrained().
    try:
        from transformers import MistralForCausalLM  # noqa: F401
    except Exception as e:  # noqa: BLE001 - report whatever the import chain raised
        root = e
        while root.__cause__ is not None:   # transformers buries the real error
            root = root.__cause__
        print(f"FAIL: cannot import MistralForCausalLM -> run_stage.py will die at load.\n"
              f"      {type(e).__name__}: {e}\n"
              f"      root cause: {type(root).__name__}: {root}")
        if "torchvision" in str(root) or "torchaudio" in str(root):
            print("      -> torchvision/torchaudio are built against a DIFFERENT torch.\n"
                  "         This repo needs neither: pip uninstall -y torchvision torchaudio")
        return False
    print("OK  MistralForCausalLM imports (model-loading path is intact)")
    return True


def main() -> int:
    cfg = DEFAULT
    ok = check_torch_backend()
    ref_ids, vocabs, eois = {}, {}, {}

    for stage, mid in cfg.checkpoints:
        tok = AutoTokenizer.from_pretrained(mid)
        try:
            tid = resolve_refusal_token(tok, cfg.refusal_token_piece, cfg.expected_refusal_id)
        except ValueError as e:
            print(f"  FAIL [{stage}]: {e}")
            ok = False
            continue
        ref_ids[stage] = [tid]
        vocabs[stage] = len(tok)
        eois[stage] = eoi_len(tok, cfg.template)
        print(f"{stage:5s} vocab={len(tok)} refusal{cfg.refusal_token_piece!r}=[{tid}] "
              f"decoded={tok.decode([tid])!r} eoi_len(derived)={eois[stage]}")

    uniq = {tuple(v) for v in ref_ids.values()}
    if len(uniq) != 1:
        print(f"FAIL: refusal token differs across stages: {ref_ids}")
        ok = False
    else:
        print(f"\nOK  refusal token identical in all stages: {uniq.pop()}")

    if len(set(vocabs.values())) != 1:
        print(f"FAIL: vocab sizes differ {vocabs} — cross-stage comparison is not clean")
        ok = False
    else:
        print(f"OK  shared vocab size {next(iter(vocabs.values()))}")

    print(f"\nNOTE tokenizer-derived eoi_len differs by stage: {eois}  <-- why n_eoi is PINNED")
    if cfg.n_eoi > min(eois.values()):
        print(f"FAIL: pinned n_eoi={cfg.n_eoi} exceeds shortest derived {min(eois.values())}")
        ok = False
    else:
        print(f"OK  pinned n_eoi={cfg.n_eoi} <= shortest derived {min(eois.values())} "
              f"-> same position window in every stage")

    print("\nALL CHECKS PASSED" if ok else "\nCHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
