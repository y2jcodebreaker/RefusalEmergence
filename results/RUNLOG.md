# Run log

Append-only. One entry per execution of an experiment script, newest at the bottom. Written automatically by `runlog.py` — do not hand-edit.

Experiment ids: `E02` is the pilot (this repo's original experiment); `P1-E1`…`P1-E7` are the experiments of paper P1, specified in `Writing/P1-Coupling-Not-Capability.md` in the vault.

---

## P1-E1 · 2026-09-13T18:17:31+00:00 · **FAILED**

> Is harmful-vs-harmless linearly readable in base, and is it the same axis the aligned model refuses along?

- **script** `probe_representation.py` — `probe_representation.py --stage all`
- **code** `757a634` on `main` ⚠️ DIRTY WORKING TREE — commit does not identify this code
- **duration** 0.0s
- **env** torch 2.4.1+cu124 · transformers None · sklearn 1.9.1 · gpu NVIDIA L40S
- **error** `ModuleNotFoundError: No module named 'transformers'`

Accuracy alone is near-certain to be high and proves little; the informative outputs are the layer profile (L0 vs peak) and the cross-stage cosines computed by aggregate_probe.py.

---

## P1-E1 · 2026-09-13T18:20:13+00:00 · **FAILED**

> Is harmful-vs-harmless linearly readable in base, and is it the same axis the aligned model refuses along?

- **script** `probe_representation.py` — `probe_representation.py --stage all`
- **code** `bd9f46e` on `main` ⚠️ DIRTY WORKING TREE — commit does not identify this code
- **duration** 29.1s
- **env** torch 2.14.0+cu130 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S
- **error** `FileNotFoundError: Arditi splits not found: /Users/lichking/Documents/probe_repos/probe4_arditi_refusal_direction/dataset/splits
  export ARDITI_REPO=/path/to/refusal_direction   (the cloned repo)`

Accuracy alone is near-certain to be high and proves little; the informative outputs are the layer profile (L0 vs peak) and the cross-stage cosines computed by aggregate_probe.py.

---

## P1-E1 · 2026-09-13T18:23:50+00:00 · OK

> Is harmful-vs-harmless linearly readable in base, and is it the same axis the aligned model refuses along?

- **script** `probe_representation.py` — `probe_representation.py --stage all`
- **code** `1dec20b` on `main` ⚠️ DIRTY WORKING TREE — commit does not identify this code
- **duration** 138.7s
- **env** torch 2.14.0+cu130 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S

| stage | mass_mean_peak | mass_mean_peak_layer | logistic_peak | logistic_peak_layer | logistic_L0 | length_only_baseline | n_fit | n_test_per_class |
|---|---|---|---|---|---|---|---|---|
| base | 0.9697 | 6 | 0.9886 | 4 | 0.5 | 0.5985 | 128 | 132 |
| sft | 0.9735 | 12 | 1.0 | 12 | 0.5 | 0.5985 | 128 | 132 |
| dpo | 0.9773 | 15 | 1.0 | 12 | 0.5 | 0.5985 | 128 | 132 |

Accuracy alone is near-certain to be high and proves little; the informative outputs are the layer profile (L0 vs peak) and the cross-stage cosines computed by aggregate_probe.py.

---

## P1-E1 · 2026-09-13T18:26:11+00:00 · OK

> Same axis as the aligned model, or a different one?

- **script** `aggregate_probe.py` — `aggregate_probe.py`
- **code** `1dec20b` on `main` ⚠️ DIRTY WORKING TREE — commit does not identify this code
- **duration** 0.9s
- **env** torch 2.14.0+cu130 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S

| stage | logistic_peak | logistic_L0 | mass_mean_peak | length_only | surface_suspect |
|---|---|---|---|---|---|
| base | 0.9886 | 0.5 | 0.9697 | 0.5985 | False |
| sft | 1.0 | 0.5 | 0.9735 | 0.5985 | False |
| dpo | 1.0 | 0.5 | 0.9773 | 0.5985 | False |
| base_vs_sft | 0.9258 | 1 | 0.9377 | 25 |
| base_vs_dpo | 0.9552 | 1 | 0.9633 | 24 |

---

## P1-E1 · 2026-09-13T18:30:04+00:00 · OK

> Same axis as the aligned model, or a different one?

- **script** `aggregate_probe.py` — `aggregate_probe.py`
- **code** `56ecf93` on `main`
- **duration** 0.8s
- **env** torch 2.14.0+cu130 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S

| stage | logistic_peak | logistic_L0 | mass_mean_peak | length_only | surface_suspect |
|---|---|---|---|---|---|
| base | 0.9886 | 0.5 | 0.9697 | 0.5985 | False |
| sft | 1.0 | 0.5 | 0.9735 | 0.5985 | False |
| dpo | 1.0 | 0.5 | 0.9773 | 0.5985 | False |
| base_vs_sft | 0.9258 | 1 | 0.9377 | 25 |
| base_vs_dpo | 0.9552 | 1 | 0.9633 | 24 |

---

## P1-E1b · 2026-09-13T18:33:29+00:00 · OK

> Does the aligned model's refusal direction induce refusal when transplanted into BASE, and at what coefficient does induction appear while KL stays sane?

- **script** `transplant.py` — `transplant.py --stage all`
- **code** `7839021` on `main`
- **duration** 30.3s
- **env** torch 2.14.0+cu130 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S

| target | source | layer | kind | induces | min_coeff | max_induced_at_kl_ok | baseline_harmless |
|---|---|---|---|---|---|---|---|
| base | sft | 20 | direction | False | None | None | -3.1237 |
| base | sft | 20 | random | False | None | -3.4329 | -3.1237 |
| base | dpo | 17 | direction | False | None | None | -3.1237 |
| base | dpo | 17 | random | False | None | -2.8788 | -3.1237 |
| base | base | 15 | direction | False | None | -2.7918 | -3.1237 |
| base | base | 15 | random | False | None | -2.8724 | -3.1237 |
| sft | sft | 20 | direction | False | None | None | -4.0501 |
| sft | sft | 20 | random | False | None | -4.5053 | -4.0501 |
| sft | dpo | 17 | direction | False | None | None | -4.0501 |
| sft | dpo | 17 | random | False | None | -3.3258 | -4.0501 |
| sft | base | 15 | direction | False | None | -2.5913 | -4.0501 |
| sft | base | 15 | random | False | None | -3.7422 | -4.0501 |
| dpo | sft | 20 | direction | False | None | None | -8.9639 |
| dpo | sft | 20 | random | False | None | -9.5586 | -8.9639 |
| dpo | dpo | 17 | direction | False | None | None | -8.9639 |
| dpo | dpo | 17 | random | False | None | -8.8429 | -8.9639 |
| dpo | base | 15 | direction | False | None | -7.9506 | -8.9639 |
| dpo | base | 15 | random | False | None | -8.5392 | -8.9639 |

The decisive cell is target=base, source=dpo: if DPO's refusal direction induces refusal in base at acceptable KL, base already had the machinery. Sweeping coefficients also retires the 'you only tried coeff=1' objection to E02.

---

## P1-E1b · 2026-09-13T18:36:35+00:00 · OK

> Does the aligned model's refusal direction induce refusal when transplanted into BASE, and at what coefficient does induction appear while KL stays sane?

- **script** `transplant.py` — `transplant.py --stage all`
- **code** `7eacdfc` on `main`
- **duration** 30.2s
- **env** torch 2.14.0+cu130 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S

| target | source | layer | kind | induces | first_coeff | max_induced | coeff_at_max | kl_at_max | dir_norm | baseline_harmless |
|---|---|---|---|---|---|---|---|---|---|---|
| base | sft | 20 | direction | False | None | -2.069 | 1.0 | 0.888 | 7.41 | -3.1237 |
| base | sft | 20 | random | False | None | -3.4329 | 0.5 | 0.042 | 7.41 | -3.1237 |
| base | dpo | 17 | direction | False | None | -1.6929 | 1.0 | 1.087 | 4.37 | -3.1237 |
| base | dpo | 17 | random | False | None | -1.8991 | 2.0 | 0.843 | 4.37 | -3.1237 |
| base | base | 15 | direction | False | None | -1.9 | 4.0 | 1.055 | 1.07 | -3.1237 |
| base | base | 15 | random | False | None | -2.6368 | 2.0 | 0.185 | 1.07 | -3.1237 |
| sft | sft | 20 | direction | True | 1.0 | 0.5379 | 2.0 | 2.587 | 7.41 | -4.0501 |
| sft | sft | 20 | random | False | None | -4.5053 | 0.5 | 0.019 | 7.41 | -4.0501 |
| sft | dpo | 17 | direction | True | 1.0 | 0.9319 | 1.0 | 1.641 | 4.37 | -4.0501 |
| sft | dpo | 17 | random | False | None | -1.5277 | 4.0 | 2.301 | 4.37 | -4.0501 |
| sft | base | 15 | direction | True | 8.0 | 0.4856 | 8.0 | 1.951 | 1.07 | -4.0501 |
| sft | base | 15 | random | False | None | -3.0557 | 4.0 | 0.183 | 1.07 | -4.0501 |
| dpo | sft | 20 | direction | True | 2.0 | 0.2748 | 2.0 | 4.197 | 7.41 | -8.9639 |
| dpo | sft | 20 | random | False | None | -9.0487 | 4.0 | 6.946 | 7.41 | -8.9639 |
| dpo | dpo | 17 | direction | True | 1.0 | 1.0766 | 1.0 | 3.349 | 4.37 | -8.9639 |
| dpo | dpo | 17 | random | False | None | -3.6537 | 4.0 | 4.525 | 4.37 | -8.9639 |
| dpo | base | 15 | direction | False | None | -4.0432 | 4.0 | 0.992 | 1.07 | -8.9639 |
| dpo | base | 15 | random | False | None | -5.92 | 4.0 | 0.546 | 1.07 | -8.9639 |

The decisive cell is target=base, source=dpo: if DPO's refusal direction induces refusal in base at acceptable KL, base already had the machinery. Sweeping coefficients also retires the 'you only tried coeff=1' objection to E02.

---

## P1-E1b · 2026-09-13T18:40:58+00:00 · OK

> Does the aligned model's refusal direction induce refusal when transplanted into BASE, and at what coefficient does induction appear while KL stays sane?  [unit-norm]

- **script** `transplant.py` — `transplant.py --stage all --unit-norm`
- **code** `006690b` on `main`
- **duration** 31.2s
- **env** torch 2.14.0+cu130 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S

| target | source | layer | kind | induces | first_coeff | max_induced | coeff_at_max | kl_at_max | dir_norm | baseline_harmless |
|---|---|---|---|---|---|---|---|---|---|---|
| base | sft | 20 | direction | False | None | -2.0772 | 8.0 | 0.994 | 1.0 | -3.1237 |
| base | sft | 20 | random | False | None | -3.1641 | 0.5 | 0.001 | 1.0 | -3.1237 |
| base | dpo | 17 | direction | False | None | -1.6692 | 4.0 | 0.977 | 1.0 | -3.1237 |
| base | dpo | 17 | random | False | None | -1.9638 | 16.0 | 2.05 | 1.0 | -3.1237 |
| base | base | 15 | direction | False | None | -1.8784 | 4.0 | 0.946 | 1.0 | -3.1237 |
| base | base | 15 | random | False | None | -2.6536 | 2.0 | 0.15 | 1.0 | -3.1237 |
| sft | sft | 20 | direction | True | 8.0 | 0.5094 | 16.0 | 2.77 | 1.0 | -4.0501 |
| sft | sft | 20 | random | False | None | -4.1185 | 0.5 | 0.001 | 1.0 | -4.0501 |
| sft | dpo | 17 | direction | True | 4.0 | 0.8171 | 8.0 | 3.104 | 1.0 | -4.0501 |
| sft | dpo | 17 | random | False | None | -1.4908 | 16.0 | 2.024 | 1.0 | -4.0501 |
| sft | base | 15 | direction | True | 8.0 | 0.4231 | 8.0 | 1.736 | 1.0 | -4.0501 |
| sft | base | 15 | random | False | None | -3.0602 | 8.0 | 0.85 | 1.0 | -4.0501 |
| dpo | sft | 20 | direction | True | 8.0 | 0.1416 | 16.0 | 4.41 | 1.0 | -8.9639 |
| dpo | sft | 20 | random | False | None | -9.0392 | 0.5 | 0.001 | 1.0 | -8.9639 |
| dpo | dpo | 17 | direction | True | 4.0 | 1.1798 | 8.0 | 5.339 | 1.0 | -8.9639 |
| dpo | dpo | 17 | random | False | None | -2.673 | 16.0 | 3.642 | 1.0 | -8.9639 |
| dpo | base | 15 | direction | False | None | -3.6591 | 8.0 | 3.19 | 1.0 | -8.9639 |
| dpo | base | 15 | random | False | None | -6.1792 | 4.0 | 0.48 | 1.0 | -8.9639 |

The decisive cell is target=base, source=dpo: if DPO's refusal direction induces refusal in base at acceptable KL, base already had the machinery. Sweeping coefficients also retires the 'you only tried coeff=1' objection to E02.

---

## P1-E1b · 2026-09-13T18:47:25+00:00 · OK

> Does the aligned model's refusal direction induce refusal when transplanted into BASE, and at what coefficient does induction appear while KL stays sane?  [unit-norm]

- **script** `transplant.py` — `transplant.py --stage all --unit-norm`
- **code** `f8fc00f` on `main`
- **duration** 170.6s
- **env** torch 2.14.0+cu130 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S

| target | source | layer | kind | induces | first_coeff | max_induced | coeff_at_max | kl_at_max | dir_norm | baseline_harmless |
|---|---|---|---|---|---|---|---|---|---|---|
| base | sft | 20 | direction | False | None | -2.0938 | 8.0 | 0.942 | 1.0 | -3.0247 |
| base | sft | 20 | random | False | None | -3.0609 | 0.5 | 0.001 | 1.0 | -3.0247 |
| base | dpo | 17 | direction | False | None | -1.6261 | 4.0 | 0.907 | 1.0 | -3.0247 |
| base | dpo | 17 | random | False | None | -1.9569 | 16.0 | 2.164 | 1.0 | -3.0247 |
| base | base | 15 | direction | False | None | -1.7557 | 4.0 | 0.942 | 1.0 | -3.0247 |
| base | base | 15 | random | False | None | -2.6632 | 2.0 | 0.138 | 1.0 | -3.0247 |
| sft | sft | 20 | direction | True | 16.0 | 0.3786 | 16.0 | 2.577 | 1.0 | -4.0967 |
| sft | sft | 20 | random | False | None | -4.1462 | 0.5 | 0.001 | 1.0 | -4.0967 |
| sft | dpo | 17 | direction | True | 4.0 | 0.6186 | 8.0 | 2.855 | 1.0 | -4.0967 |
| sft | dpo | 17 | random | False | None | -1.5702 | 16.0 | 2.136 | 1.0 | -4.0967 |
| sft | base | 15 | direction | True | 8.0 | 0.5627 | 8.0 | 1.866 | 1.0 | -4.0967 |
| sft | base | 15 | random | False | None | -3.0939 | 8.0 | 0.872 | 1.0 | -4.0967 |
| dpo | sft | 20 | direction | False | None | -0.0915 | 16.0 | 3.974 | 1.0 | -8.8034 |
| dpo | sft | 20 | random | False | None | -8.8817 | 0.5 | 0.001 | 1.0 | -8.8034 |
| dpo | dpo | 17 | direction | True | 8.0 | 0.7894 | 8.0 | 4.662 | 1.0 | -8.8034 |
| dpo | dpo | 17 | random | False | None | -3.0163 | 16.0 | 3.818 | 1.0 | -8.8034 |
| dpo | base | 15 | direction | False | None | -3.4877 | 8.0 | 3.008 | 1.0 | -8.8034 |
| dpo | base | 15 | random | False | None | -6.3251 | 4.0 | 0.434 | 1.0 | -8.8034 |

The decisive cell is target=base, source=dpo: if DPO's refusal direction induces refusal in base at acceptable KL, base already had the machinery. Sweeping coefficients also retires the 'you only tried coeff=1' objection to E02.

---

## E02 · 2026-09-13T18:50:18+00:00 · OK

> When across base -> SFT -> DPO does a causally actionable refusal direction appear, and in which layers?

- **script** `run_stage.py` — `run_stage.py --stage base --behavioral`
- **code** `f8fc00f` on `main`
- **duration** 90.6s
- **env** torch 2.14.0+cu130 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S

| stage | l_star | naive_l_star | pos_star | baseline_refusal | peak_strength | kl_at_l_star | control_peak | substring_baseline | substring_ablated |
|---|---|---|---|---|---|---|---|---|---|
| base | -1 | 15 | -1 | -2.0643 | 1.0805 | None | 0.1269 | 0.5379 | None |
