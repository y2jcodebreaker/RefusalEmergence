# Runbook — a pod from scratch

Copy-paste order. Every step is cheap except the two marked **GPU**.

## 0. Pod

**24 GB VRAM** is enough (one 7B in bf16 at a time). **150 GB disk** — OLMo 2 is four
checkpoints (~60 GB) and Zephyr three (~45 GB).

## 1. Environment

```bash
export HF_HOME=/workspace/hf        # MUST be on the persistent volume, or you re-download
cd /workspace

git clone https://github.com/andyrdt/refusal_direction.git      # harmful/harmless splits
git clone https://github.com/y2jcodebreaker/RefusalEmergence.git
cd RefusalEmergence
```

`/workspace/refusal_direction` is auto-discovered, so **no `ARDITI_REPO` export is needed**
as long as you clone it there.

```bash
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip uninstall -y torchvision torchaudio      # built against a different torch; breaks imports
```

`--upgrade` is not optional: pod images ship torch 2.4.x, plain `pip install torch` treats
that as satisfied, and recent `transformers` silently disables its PyTorch backend below 2.5.

## 2. Preflight — no GPU, ~30 s. Do not skip.

```bash
python verify_setup.py --lineage olmo2
python smoke_test.py && python smoke_test_probes.py
```

Every line must be OK. This has caught, on separate occasions: a disabled torch backend, a
torchvision import break, a missing data clone, a wrong refusal token, and an end-of-
instruction window that leaked prompt text. Weights are only downloaded *after* this.

## 3. Restore earlier results (optional)

Only if you want a previous lineage's sweeps on this pod. They are **not** in git (tens of
MB); the run ledger is.

```bash
tar xzf /workspace/results-zephyr-restore.tgz    # unpacks into results/
ls results/zephyr_*.npz | wc -l                  # expect 9
```

Nothing GPU-bound needs them — `aggregate*.py` runs fine on a laptop.

## 4. Run a lineage — **GPU**

```bash
LIN=olmo2          # or zephyr

rm -f results/${LIN}_*.npz     # only if a previous run used different config (see note)

python run_stage.py            --lineage $LIN --stage all --control --behavioral  # ~15 min
python probe_representation.py --lineage $LIN --stage all                         # ~5 min
python transplant.py           --lineage $LIN --stage all --unit-norm             # ~5 min
```

**If any transplant cell crosses the induction threshold at a large KL**, the logit result is
not enough — run the text-level check on that cell before believing it:

```bash
python transplant_text.py --lineage $LIN --target base --source sft      # ~3 min, GPU
```

It generates under the same injection and reports refusal rate and DEGENERATE rate against a
norm-matched random arm at the identical coefficient. A logit crossing that comes with
degenerate text is distributional damage, not induced refusal.

**Order matters.** `transplant.py` reads each stage's `l*` from `run_stage`'s output and its
directions from `probe_representation`'s.

**On the `rm`:** `np.savez` rewrites whole files, and `run_stage` deliberately carries
forward keys it did not recompute (so re-running one stage does not delete another axis's
results). That carry-forward is wrong across a **config change** — stale control curves
computed at a different `n_eoi` would be silently preserved. Delete first whenever
`n_eoi`, the template, or the refusal token changed.

## 5. Figures and audits — no GPU

```bash
python aggregate.py        --lineage $LIN     # induce curve, heatmaps, peak + control panel
python aggregate_probe.py  --lineage $LIN     # probe accuracy + cosine null, prints a verdict

python show_completions.py --lineage $LIN --stage base   # is base even coherent?
python audit_judge.py      --lineage $LIN --stage base --out audit_${LIN}_base.txt
python show_filters.py     --lineage $LIN --stage base   # which criterion failed, and why
```

`audit_judge` is **mandatory for a new family** — the substring judge is a proxy, not an
oracle, and hand-reading its hits has twice found false-positive classes no filter caught.

## 6. Before destroying the pod

```bash
tar czf /workspace/results-$LIN-$(date +%F).tgz results/
```

Download it. The `.npz` sweeps cost GPU time to regenerate; `results/RUNLOG.md` and
`results/runs.jsonl` cannot be regenerated at all — commit those (auth from a laptop, not
from the pod).

## Adding a new lineage

A lineage whose template, refusal token and window are unmeasured **refuses to run**. The
chat template, the token id, and the window length do **not** transfer between families —
and the token depends on the *template*, not just the family.

```bash
python diagnose_refusal_token.py --lineage NEW --stage <last stage>
#   -> set Lineage.expected_refusal_id to the TOP-1 id on HARMFUL prompts
python verify_setup.py --lineage NEW
#   -> prints the derived template to paste, and the largest leak-free n_eoi
```

If the base model degenerates under the chat template (echo, repetition), test a plain
format with `diagnose_refusal_token.py --template 'User: {instruction}\nAssistant:'` and, if
it is coherent there, give that stage a `stage_regime` override. See `CONVENTIONS.md`.
