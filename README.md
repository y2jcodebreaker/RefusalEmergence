# Refusal Emergence Across the Alignment Pipeline

**When does a language model acquire the machinery to refuse — and where in the network?**
Is a refusal direction already present in the **base** model, does **SFT** create it, or does
the **preference-optimization step (DPO)** install it?

## Result

**Alignment does not create refusal behavior — it creates refusal machinery.**

Base Mistral refuses **29× more often than its SFT descendant** in generated text (0.235 vs
0.008), while possessing **no steerable refusal direction at any layer**. SFT installs one in a
narrow middle-layer band (L15–20); DPO sharpens it (+1.01 → +1.76) without moving its peak
(L16). Behavioral refusal and refusal geometry are **dissociated** across the pipeline.

Full numbers, controls, limitations, and the four methodological errors caught along the way:
**[RESULTS.md](RESULTS.md)**.

Prior work: Arditi et al. (NeurIPS 2024) showed refusal is a single direction in *finished*
chat models. "How Post-Training Reshapes LLMs" (COLM 2025) compared base vs final — two
points. This maps the **full developmental trajectory, per layer**, on two independent axes
(causal ablation and constructive induction) with a norm-matched random control.

## Method

Three checkpoints of one lineage, so the **weights are the only variable**:
`mistralai/Mistral-7B-v0.1` → `alignment-handbook/zephyr-7b-sft-full` →
`HuggingFaceH4/zephyr-7b-beta`. Same tokenizer, same 32000-token vocab, one fixed chat
template imposed on all three.

Per checkpoint, `run_stage.py`:

1. **Extract** the refusal direction at every layer (mean-diff harmful − harmless at
   end-of-instruction positions), following Arditi's `generate_directions`.
2. **Ablate** (`x −= (x·r̂)r̂` at resid_pre + attn_out + mlp_out, all layers) → per-layer
   causal bypass strength on harmful prompts.
3. **Induce** (add the raw vector at the source layer, coeff 1.0) → does refusal *appear* on
   harmless prompts? This is the constructive axis, and it cannot be satisfied by merely
   damaging the model — breaking a network does not make it refuse.
4. **KL** on harmless prompts after ablation → did the intervention preserve the model?
5. **Select** `(pos*, l*)` under all three of Arditi's criteria (bypass, induce ≥ 0, KL ≤ 0.1,
   last 20% of layers pruned). Reports `l* = -1` when nothing passes — a real answer, not an
   error. Base returns `-1`.

Optional axes: `--control` runs the identical sweep with K=3 norm-matched random directions;
`--behavioral` measures the substring refusal rate (Arditi's JailbreakBench prefixes) on 132
held-out prompts, baseline vs ablated.

The core is ported from a **validated** Arditi implementation that reproduced their
`(pos=-5, layer=12)` exactly in a prior project.

## Run

```bash
uv venv && source .venv/bin/activate
uv pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
pip uninstall -y torchvision torchaudio     # see Gotchas

git clone https://github.com/andyrdt/refusal_direction.git
export ARDITI_REPO=$PWD/refusal_direction   # harmful/harmless splits only
# ^ NOT persisted across pod restarts. data.py also auto-finds ./refusal_direction,
#   ../refusal_direction and /workspace/refusal_direction, so cloning into one of those
#   locations means you never have to remember the export.

python verify_setup.py                            # preflight, no GPU, ~30s
python run_stage.py --stage all --control --behavioral   # ~30 min on an L40S
python aggregate.py                               # -> results/figures/*.pdf
```

**24 GB VRAM** is enough (one 7B in bf16 at a time). **100 GB disk** for the HF cache — set
`HF_HOME` to a persistent volume. Accept the Mistral-7B-v0.1 license on HF before starting.

### Inspecting and auditing

```bash
python show_completions.py --stage base   # generations + per-hit judge verdicts
python audit_judge.py --stage base       # EVERY judge hit, for hand-classification
python show_filters.py --stage base       # the KL / induce surfaces, and WHICH criterion failed
python diagnose_refusal_token.py --stage dpo   # what token does the model actually emit?
python smoke_test.py                      # CPU-only unit tests
```

## Gotchas (all of these cost us a run)

- **`torch>=2.5` is required.** Recent `transformers` silently *disables* its PyTorch backend
  below that, so tokenizer-only code keeps working while `from_pretrained` fails. Many GPU pod
  images ship torch 2.4.x, and plain `pip install torch` treats that as satisfied — use
  `--upgrade`. `verify_setup.py` checks this.
- **Uninstall `torchvision`/`torchaudio`** if they were built against a different torch:
  `transformers` imports torchvision opportunistically and you get
  `operator torchvision::nms does not exist`, which looks nothing like the real cause. This
  repo needs neither. `verify_setup.py` checks this too, and names the cause.
- **`n_eoi` is pinned, not derived.** The tokenizer-derived end-of-instruction length is
  **9 for base but 10 for SFT/DPO** (the Zephyr tokenizers insert a phantom `''` token). A
  stage-varying position window would have silently invalidated the entire cross-stage
  comparison.
- **The refusal token is `28737`, not `315`.** Both decode to `"I"`. See O-42 in RESULTS.md.
- **Re-running one stage preserves the other flags' results.** `np.savez` rewrites whole
  files, so keys not recomputed are carried forward from the previous run.
- **`ARDITI_REPO` does not survive a pod restart**, and neither does the Python env. Both
  scripts now check the data *before* loading any weights, so a missing clone costs seconds
  rather than a 45 GB download.

## Lineages

Every driver takes `--lineage` (default `zephyr`). Results are lineage-scoped
(`results/{lineage}_{stage}_{axis}.npz`). A lineage whose refusal token and eoi window have
not been measured **refuses to run** — see [CONVENTIONS.md](CONVENTIONS.md).

## Experiments

| id | scripts | question |
|---|---|---|
| `E02` | `run_stage.py` → `aggregate.py` | when does an actionable refusal direction appear, and where? **(done — see RESULTS.md)** |
| `P1-E1` | `probe_representation.py` → `aggregate_probe.py` | is the distinction *readable* in base? **readable at ~0.99 from L1 — yes.** Same-axis half was inconclusive: cosine is saturated by residual-stream anisotropy |
| `P1-E1b` | `transplant.py` | does the aligned model's refusal direction induce refusal in **base**? Answers the same-axis question behaviourally, so anisotropy cannot touch it. Sweeps coefficients, retiring the "only tried coeff=1" objection |

Every run is logged to `results/RUNLOG.md` with its git commit, config, environment and
results — including failed runs. Conventions: **[CONVENTIONS.md](CONVENTIONS.md)**.

```bash
python probe_representation.py --stage all   # P1-E1, ~3 min
python aggregate_probe.py                    # prints an explicit verdict
python transplant.py --stage all             # P1-E1b, ~5 min
```

## Files

| file | role |
|---|---|
| `config.py` | frozen config; every non-obvious value carries the measurement that justifies it |
| `refusal_direction.py` | extraction, ablation, induction, KL, Arditi's 3-criterion selection |
| `refusal_substring.py` | behavioral judge — Arditi verbatim + a strict variant for base models |
| `run_stage.py` | one checkpoint end to end |
| `aggregate.py` | the five figures |
| `verify_setup.py` | preflight: no GPU, catches the environment traps above |
| `diagnose_refusal_token.py` | what the model actually emits at position 0 |
| `show_completions.py` / `show_filters.py` | audit the judge / the selection surfaces |
| `probes.py` | P1-E1: probe math, cosine nulls, activation caching |
| `probe_representation.py` | P1-E1 driver; `aggregate_probe.py` renders it and states the verdict |
| `transplant.py` | P1-E1b: cross-checkpoint direction transplant + coefficient sweep |
| `runlog.py` | provenance ledger — every run, with git commit and environment |
| `smoke_test.py` / `smoke_test_probes.py` | CPU unit tests; every check has a known answer |
