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

import argparse

import sys

# Imported defensively: this script exists to make environment problems legible, so it
# must not itself die with a bare ModuleNotFoundError on a fresh pod. (It did, 2026-09-13.)
_MISSING: list[str] = []
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None
    _MISSING.append("transformers")

from config import DEFAULT, LINEAGES, config_for

try:
    from refusal_direction import eoi_len, resolve_refusal_token
except ImportError:
    eoi_len = resolve_refusal_token = None
    _MISSING.append("torch")


def _report_missing() -> int:
    print("CHECKS CANNOT RUN — missing packages: " + ", ".join(sorted(set(_MISSING))))
    print("\nThis is a fresh environment. Install, in this order:\n"
          "  pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124\n"
          "  pip install -r requirements.txt\n"
          "  pip uninstall -y torchvision torchaudio\n"
          "then re-run: python verify_setup.py")
    return 1


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


def derive_template(cfg) -> str:
    """Render the aligned checkpoint's own chat_template for a one-turn prompt, and show it
    as the `template` string this repo wants (with {instruction} where the user text goes)."""
    if AutoTokenizer is None:
        return "  (cannot derive a template: transformers is missing)"
    stage, mid = cfg.checkpoints[-1]          # the most-aligned checkpoint has the template
    tok = AutoTokenizer.from_pretrained(mid)
    if not getattr(tok, "chat_template", None):
        return (f"  NOTE {mid} ships no chat_template; pick the format from its model card "
                f"and paste it into the Lineage.")
    rendered = tok.apply_chat_template([{"role": "user", "content": "\x00"}],
                                       tokenize=False, add_generation_prompt=True)
    as_template = rendered.replace("\x00", "{instruction}")
    # Strip a leading BOS if the tokenizer will add one again at encode time.
    bos = getattr(tok, "bos_token", None)
    warn = ""
    if bos and as_template.startswith(bos):
        warn = (f"\n  ⚠️ starts with bos {bos!r}; tokenizers usually re-add it, so strip it "
                f"from the template or it appears twice")
    return ("\n  DERIVED TEMPLATE (from %s, stage '%s') — paste into config.LINEAGES:\n"
            "    template=%r,%s\n" % (mid, stage, as_template, warn))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", default="zephyr",
                    help="model family from config.LINEAGES")
    args = ap.parse_args()
    if _MISSING:
        return _report_missing()
    cfg = config_for(args.lineage)
    ok = check_torch_backend()
    ref_ids, vocabs, eois = {}, {}, {}

    # Derive the chat template from the ALIGNED checkpoint's own tokenizer rather than
    # transcribing special tokens by hand — hand-copied tokens are the same class of silent,
    # plausible error as O-42. Printed for pasting into the Lineage.
    if LINEAGES[cfg.lineage].template is None:
        print(derive_template(cfg))

    from data import splits_dir
    sd = splits_dir()
    if sd is None:
        print("FAIL: Arditi harmful/harmless splits not found — set ARDITI_REPO "
              "(see data.py SEARCH for the paths tried)")
        ok = False
    else:
        print(f"OK  data splits at {sd}")

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

    spread = "differs by stage" if len(set(eois.values())) > 1 else "is CONSISTENT across stages"
    print(f"\nNOTE tokenizer-derived eoi_len {spread}: {eois}")
    if cfg.n_eoi is None:
        # This script exists to produce this number. Crashing on the None it is meant to
        # fill in was the one failure mode it must not have.
        rec = min(eois.values())
        print(f"\n  n_eoi is NOT YET PINNED for lineage '{cfg.lineage}'.")
        print(f"  RECOMMENDED: n_eoi={rec}")
        print(f"    = min(derived) across the lineage's checkpoints, so every stage gets the\n"
              f"      SAME position window. Where the derived length is consistent (as here),\n"
              f"      that is the FULL window and matches Arditi's use of all eoi positions.\n"
              f"      Where it differs, pin below the shortest -- Zephyr derives 9 (base) vs\n"
              f"      10 (SFT/DPO), so it is pinned at 5.")
        print(f"\n  Set Lineage.n_eoi={rec} for '{cfg.lineage}' in config.py, then re-run.")
        print("\nCHECKS INCOMPLETE (n_eoi unpinned)")
        return 1
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
