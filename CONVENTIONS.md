# Conventions

Rules every script in this repo follows, so that in six months there is no ambiguity about
what was run, when, on what code, or what came out.

## Experiment ids

| id | what | where it is specified |
|---|---|---|
| `E02` | the pilot — when does a causally actionable refusal direction appear across base → SFT → DPO | `Experiments/E02-Refusal-Emergence-Across-Alignment.md` (vault) |
| `P1-E1` … `P1-E7` | the seven experiments of paper **P1**, *Coupling, Not Capability* | `Writing/P1-Coupling-Not-Capability.md` (vault) |

The `P1-` prefix is not decoration. The vault numbers its experiments `E01`, `E02`, …, and
the paper numbers its own `E1`…`E7`; without the prefix `E1` and `E01` are one typo apart and
mean different things. Never write a bare `E1`.

| id | script | question |
|---|---|---|
| `E02` | `run_stage.py` | when does an actionable refusal direction appear, and where? |
| `P1-E1` | `probe_representation.py`, `aggregate_probe.py` | is the distinction *readable* in base, and on the *same axis* the aligned model uses? |

## Every experiment script must

1. Declare `EXPERIMENT` and `QUESTION` module constants.
2. Open a `RunRecord` (see `runlog.py`) around the whole run and call `rec.result(...)` once
   per stage.
3. Write arrays to `results/{stage}_{axis}.npz` — never overwrite another axis's keys
   (`run_stage.py` carries forward keys it did not recompute; `np.savez` rewrites whole files).
4. State in its module docstring: the question, what is measured, the splits, and the outputs.
5. Justify every non-obvious constant **where it is defined**, with the measurement that
   justifies it. `config.py` is the worked example — see `expected_refusal_id`.

## The run ledger

`runlog.py` appends to two files on every execution:

- **`results/runs.jsonl`** — one JSON object per run, machine-readable.
- **`results/RUNLOG.md`** — the same run as a readable entry. Open this first.

Each entry records UTC start/end and duration, the **git commit and whether the tree was
dirty**, the full frozen config, library and GPU versions, the command line, the per-stage
results, and the outcome. **Failed runs are recorded too** — a crash is evidence, and dropping
it is how "which version produced this number?" becomes unanswerable.

A `⚠️ DIRTY WORKING TREE` marker means the commit does **not** identify the code that ran.
Treat numbers from a dirty run as provisional and re-run from a clean tree before reporting.

Both files are gitignored: they are a local execution history, not shared state. Findings
worth keeping go to `RESULTS.md` and to the vault.

## Verification before GPU time

| script | checks |
|---|---|
| `verify_setup.py` | torch/transformers backend, the model-loading import path, tokenizer agreement across checkpoints, the pinned position window |
| `smoke_test.py` | pure logic of the causal sweep, on CPU |
| `smoke_test_probes.py` | probe math, cosine nulls, activation caching, the ledger — every check has a known answer |

Run all three before a GPU session. `verify_setup.py` has passed while the real run was broken
three separate times; each failure is now a named check in it.

## Reporting

- Report **full curves**, never only peaks. The last 20% of layers are excluded from `l*`
  selection (O-40) and must be shown as excluded, not silently dropped.
- Report **both** a published method verbatim and any corrected variant, never the corrected
  one alone (see `refusal_substring.py`: `is_refusal` vs `is_refusal_strict`).
- A measurement that cannot be made is `None`, not `0`.
- When a result changes after a fix, the superseded number stays in the vault notebook with
  the reason. Four headline numbers in this project have been retracted; the trail is what
  makes the surviving ones credible.
