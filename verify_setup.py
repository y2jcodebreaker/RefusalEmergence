"""Verify the tokenizer-level assumptions BEFORE burning GPU time. No model weights needed.

    python verify_setup.py

Checks, across every checkpoint in config.checkpoints:
  1. refusal_onset_str is exactly ONE token, and the SAME id in every stage.
  2. the pinned n_eoi window is safe (<= the shortest tokenizer-derived eoi_len).
  3. reports tokenizer-derived eoi_len per stage (these legitimately DIFFER — that is
     exactly why n_eoi is pinned rather than derived).
  4. vocab sizes match (a shared vocab is what makes the cross-stage comparison valid).
"""

from __future__ import annotations

import sys

from transformers import AutoTokenizer

from config import DEFAULT
from refusal_direction import eoi_len


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
    return True


def main() -> int:
    cfg = DEFAULT
    ok = check_torch_backend()
    ref_ids, vocabs, eois = {}, {}, {}

    for stage, mid in cfg.checkpoints:
        tok = AutoTokenizer.from_pretrained(mid)
        ids = tok.encode(cfg.refusal_onset_str, add_special_tokens=False)
        ref_ids[stage] = ids
        vocabs[stage] = len(tok)
        eois[stage] = eoi_len(tok, cfg.template)
        print(f"{stage:5s} vocab={len(tok)} refusal{cfg.refusal_onset_str!r}={ids} "
              f"decoded={[tok.decode([i]) for i in ids]} eoi_len(derived)={eois[stage]}")
        if len(ids) != 1:
            print(f"  FAIL: refusal_onset_str must be ONE token in {stage}, got {ids}")
            ok = False

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
