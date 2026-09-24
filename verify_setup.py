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
import pathlib
import sys

# Imported defensively: this script exists to make environment problems legible, so it
# must not itself die with a bare ModuleNotFoundError on a fresh pod. (It did, 2026-09-13.)
_MISSING: list[str] = []
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None
    _MISSING.append("transformers")

from config import LINEAGES, config_for

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


# bf16 7B is ~14.5 GB on disk; a base+SFT+DPO(+RLVR) lineage is 3-4 of those.
_GB_PER_CKPT = 15
# Unaccounted cache bulk above this is called out as the xet shared tree. Module-level so
# the smoke test can lower it and actually exercise that branch.
_ORPHAN_GB = 5


def check_disk(cfg, stages: "tuple[str, ...] | None" = None) -> bool:
    """Is there room for the weights this run will actually download?

    HF only WARNS on insufficient space and then fails ~15 s later with 'Internal Writer
    Error: Background writer channel closed', which names neither disk nor the model. Hit
    mid-run on a pod with four OLMo 2 checkpoints cached (2026-09-17), after the base row had
    already been computed.

    `stages` restricts the estimate to the checkpoints a single-stage script will load. Before
    it existed the check always sized for the WHOLE lineage, so a script loading one stage of
    olmo2_e7 was told it needed 45 GB for 3 checkpoints when it needed 15 for one -- and it
    duly FAILED on a pod with 18 GB free, where the run then proceeded fine because three
    callers were ignoring the return value. **An over-strict guard is why it was ignored.**
    Both halves are fixed: the estimate is now accurate, and the callers honour it."""
    import os
    import shutil

    hf = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    # On a pod, an unset HF_HOME sends every download to the CONTAINER filesystem instead of
    # the persistent volume: it is small, so the run dies mid-download with "No space left on
    # device", and anything that did fit is lost at teardown. Both happened 2026-09-19.
    if not os.environ.get("HF_HOME") and os.path.isdir("/workspace"):
        print("FAIL: HF_HOME is unset but /workspace exists, so this looks like a pod.\n"
              f"      Downloads would go to {hf} -- the CONTAINER filesystem, which is small\n"
              "      and is discarded when the pod is destroyed.\n"
              "        export HF_HOME=/workspace/hf\n"
              "      Add it to ~/.bashrc so a new shell cannot lose it.")
        return False
    probe = hf if os.path.isdir(hf) else os.path.dirname(os.path.abspath(hf)) or "."
    free = shutil.disk_usage(probe).free / 2**30
    # A local path is already on disk; counting it as a 15 GB download would make a lineage
    # of attacked checkpoints look unaffordable and block a run that needs no network at all.
    wanted = [(s, m) for s, m in cfg.checkpoints if stages is None or s in stages]
    if stages is not None and not wanted:
        raise SystemExit(f"check_disk: stages {stages} match none of "
                         f"{[s for s, _ in cfg.checkpoints]} in lineage {cfg.lineage!r}")
    remote = [m for _, m in wanted if not os.path.exists(m)]
    need = _GB_PER_CKPT * len(remote)
    if not remote:
        print(f"OK  disk: every checkpoint in '{cfg.lineage}' is a local path — nothing to "
              f"download")
        return True
    def _gb(path) -> float:
        return sum(f.stat().st_size for f in pathlib.Path(path).rglob("*")
                   if f.is_file() and not f.is_symlink()) / 2**30

    cached, hub_total, mine = 0.0, 0.0, []
    hub = os.path.join(hf, "hub")
    if os.path.isdir(hub):
        for d in sorted(os.listdir(hub)):
            if d.startswith("."):
                continue
            sz = _gb(os.path.join(hub, d))
            hub_total += sz
            if any(m.split("/")[-1].lower() in d.lower() for _, m in cfg.checkpoints):
                cached += sz
                mine.append(d)
    # Unaccounted bulk in the cache. With xet storage the weights live in a SHARED,
    # content-addressed `hub/blobs` tree while each `models--*` dir holds only symlinks
    # (~7 MB). So `rm -rf hub/models--<finished-model>*` removes the REFERENCES and frees
    # almost nothing, leaving the bulk orphaned with nothing to garbage-collect it. That is
    # how a pod sat at 97% full with four supposedly-deleted checkpoints still on disk
    # (2026-09-17). Name it, because the per-model sizes make the disk look empty.
    per_model = sum(_gb(os.path.join(hub, d)) for d in (os.listdir(hub) if os.path.isdir(hub)
                                                        else [])
                    if d.startswith("models--") or d.startswith("datasets--"))
    orphan = hub_total - per_model
    short = need - cached - free
    if short > 0:
        print(f"FAIL: not enough disk for lineage '{cfg.lineage}'.\n"
              f"      need ~{need:.0f} GB for {len(remote)} "
              f"checkpoint{'' if len(remote) == 1 else 's'}"
              f"{'' if stages is None else ' (' + ', '.join(stages) + ')'}, "
              f"{cached:.0f} GB already cached, {free:.0f} GB free -> short ~{short:.0f} GB.\n"
              f"      HF_HOME={hf}   (cache holds {hub_total:.0f} GB total)")
        if orphan > _ORPHAN_GB:
            print(f"      ⚠️  {orphan:.0f} GB of that is NOT under any models--*/datasets--* "
                  f"directory.\n"
                  f"      That is the xet shared chunk cache. Deleting a model directory frees\n"
                  f"      only its symlinks (~7 MB) and ORPHANS its share of this bulk, which\n"
                  f"      nothing garbage-collects. To actually reclaim it, delete the cache:\n"
                  f"        rm -rf {hf}\n"
                  f"      Everything there re-downloads. results/ is NOT affected.")
        else:
            print(f"      Free space by deleting a FINISHED lineage's weights, e.g.\n"
                  f"        du -sh {hub}/* | sort -h | tail\n"
                  f"        rm -rf {hf}      # simplest: the whole cache re-downloads\n"
                  f"      Results are small and results/ is NOT affected.")
        return False
    print(f"OK  disk: {free:.0f} GB free + {cached:.0f} GB cached >= ~{need:.0f} GB needed "
          f"({len(remote)} checkpoint{'' if len(remote) == 1 else 's'}"
          f"{'' if stages is None else ' — ' + ', '.join(stages)})")
    return True


def window_leaks(tok, template: str, n_eoi: int | None, prompts: list[str]) -> tuple[bool, int]:
    """(leaks, largest_safe_n) — does the last n_eoi-token window vary with the prompt?

    The window is supposed to hold only template tokens, so that positions are genuinely
    "end of instruction" and a layer-0 probe has nothing to read. BPE breaks that silently:
    if the suffix begins with '\n', it merges with the instruction's final character, so
    '?\n' becomes one token and the window differs between prompts ending in '?' and not.
    On OLMo 2 that let a layer-0 probe reach 0.644 instead of chance, because harmless
    prompts (MMLU questions) end in '?' far more often than harmful imperatives. Reading the
    template cannot reveal this; only tokenising real prompts can (O-58)."""
    full = len(tok.encode(template.split("{instruction}")[-1], add_special_tokens=False))
    safe = 0
    for n in range(full, 0, -1):
        if len({tuple(tok.encode(template.format(instruction=p),
                                 add_special_tokens=False)[-n:]) for p in prompts}) == 1:
            safe = n
            break
    # n_eoi=None on a lineage being bootstrapped: "does the PINNED window leak" has no
    # answer yet, but `safe` is exactly what the caller needs to pin. Do not compare None.
    return (False if n_eoi is None else n_eoi > safe), safe


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


_PROMPTS: list[str] = []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", default="zephyr",
                    help="model family from config.LINEAGES")
    args = ap.parse_args()
    if _MISSING:
        return _report_missing()
    cfg = config_for(args.lineage)
    ok = check_torch_backend()
    ok = check_disk(cfg) and ok
    ref_ids, vocabs, eois, safes = {}, {}, {}, {}

    # Derive the chat template from the ALIGNED checkpoint's own tokenizer rather than
    # transcribing special tokens by hand — hand-copied tokens are the same class of silent,
    # plausible error as O-42. Printed for pasting into the Lineage.
    if LINEAGES[cfg.lineage].template is None:
        print(derive_template(cfg))

    from data import load_instructions, splits_dir
    sd = splits_dir()
    global _PROMPTS
    if sd is not None:
        _PROMPTS = (load_instructions("harmful_train")[:40]
                    + load_instructions("harmless_train")[:40])
    if sd is None:
        print("FAIL: Arditi harmful/harmless splits not found — set ARDITI_REPO "
              "(see data.py SEARCH for the paths tried)")
        ok = False
    else:
        print(f"OK  data splits at {sd}")

    for stage, mid in cfg.checkpoints:
        tok = AutoTokenizer.from_pretrained(mid)
        tpl, want_id, want_neoi, is_ov = cfg.regime(stage)
        # A BRAND-NEW lineage has template=None, and the diagnostic exemption in
        # Config.regime lets that None through to here on purpose -- this script is what
        # derives the template. Everything below needs a real string (eoi_len would raise
        # AttributeError on None), so stop with the next step rather than a stack trace.
        # tulu2_dpo never reached this: it was added with its template already pinned.
        if tpl is None:
            print(f"  NEXT [{stage}]: paste the template printed above into "
                  f"LINEAGES[{cfg.lineage!r}].template, then re-run this script for "
                  f"eoi_len, and diagnose_refusal_token.py for the refusal id.")
            ok = False
            continue
        derived = eoi_len(tok, tpl)
        if _PROMPTS:
            leaks, safe = window_leaks(tok, tpl, want_neoi, _PROMPTS)
            safes[stage] = safe
            if want_neoi is None:
                print(f"  [{stage}] largest leak-free window here: {safe} "
                      f"(derived {derived})")
            if leaks:
                print(f"  FAIL [{stage}]: n_eoi={want_neoi} window VARIES with the prompt — "
                      f"BPE merges the instruction's last character into it, so a layer-0 "
                      f"probe can read surface text. Largest safe n_eoi here is {safe}.")
                ok = False
        vocabs[stage] = len(tok)
        # A stage on a regime override is deliberately NOT format-matched to the others, so
        # it is excluded from the cross-stage token/window agreement checks below and its
        # own window is validated against its OWN template.
        if is_ov:
            print(f"{stage:5s} vocab={len(tok)} OVERRIDE tok=[{want_id}] "
                  f"decoded={tok.decode([want_id])!r} n_eoi={want_neoi} "
                  f"derived={derived} template={tpl!r}")
            if want_neoi > derived:
                print(f"  FAIL [{stage}]: override n_eoi={want_neoi} exceeds this template's "
                      f"derived length {derived} — the window would reach back into the "
                      f"INSTRUCTION text, so positions would not be end-of-instruction at all")
                ok = False
            continue
        try:
            tid = resolve_refusal_token(tok, cfg.refusal_token_piece, want_id)
        except ValueError as e:
            print(f"  FAIL [{stage}]: {e}")
            ok = False
            continue
        ref_ids[stage] = [tid]
        eois[stage] = derived
        print(f"{stage:5s} vocab={len(tok)} refusal{cfg.refusal_token_piece!r}=[{tid}] "
              f"decoded={tok.decode([tid])!r} eoi_len(derived)={derived}")

    # NOTHING was measured: every stage bailed out for want of a template. The cross-stage
    # checks below all consume what the loop collects, so on empty dicts they report failures
    # that are really just "not measured yet" -- and min(eois.values()) raises ValueError.
    # Stop here with the one instruction that matters. (The first version of this guard
    # patched only the loop and left these to crash, 2026-09-24.)
    if not eois:
        print(f"\n  Nothing measured yet for lineage '{cfg.lineage}': pin the derived "
              f"template above first, then re-run. The checks below need it.")
        return 1

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
        # Never recommend a window that LEAKS. `derived` is a property of the template
        # alone; `safe` is measured against real prompts and can be smaller when BPE merges
        # the instruction's last character into the suffix (O-58). Recommending `derived`
        # there would hand the user a value this same script FAILs on the next run.
        leak_capped = safes and min(safes.values()) < rec
        if leak_capped:
            rec = min(safes.values())
        print(f"\n  n_eoi is NOT YET PINNED for lineage '{cfg.lineage}'.")
        print(f"  RECOMMENDED: n_eoi={rec}")
        if leak_capped:
            print(f"    CAPPED BY THE LEAK CHECK: derived is {min(eois.values())}, but only "
                  f"the last {rec} tokens are identical across real prompts (O-58).")
        print("    = min(derived) across the lineage's checkpoints, so every stage gets the\n"
              "      SAME position window. Where the derived length is consistent (as here),\n"
              "      that is the FULL window and matches Arditi's use of all eoi positions.\n"
              "      Where it differs, pin below the shortest -- Zephyr derives 9 (base) vs\n"
              "      10 (SFT/DPO), so it is pinned at 5.")
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
