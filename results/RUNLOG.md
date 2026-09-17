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

---

## E02 · 2026-09-16T17:56:12+00:00 · OK

> When across base -> SFT -> DPO does a causally actionable refusal direction appear, and in which layers?

- **script** `run_stage.py` — `run_stage.py --lineage olmo2 --stage all --control --behavioral`
- **code** `c48f441` on `main`
- **duration** 618.4s
- **env** torch 2.6.0+cu124 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S

| stage | l_star | naive_l_star | pos_star | baseline_refusal | peak_strength | kl_at_l_star | control_peak | substring_baseline | substring_ablated |
|---|---|---|---|---|---|---|---|---|---|
| base | -1 | 23 | -1 | -4.1041 | 0.5383 | None | 0.0591 | 0.0227 | None |
| sft | 24 | 25 | 4 | 3.696 | 9.0004 | 0.098 | 0.0804 | 0.9924 | 0.0076 |
| dpo | 24 | 25 | 4 | 3.8128 | 12.7119 | 0.091 | 0.1228 | 0.9848 | 0.0 |
| rlvr | 24 | 25 | 4 | 4.0666 | 13.5195 | 0.0977 | 0.1287 | 0.9848 | 0.0 |

---

## P1-E1 · 2026-09-16T18:06:32+00:00 · OK

> Is harmful-vs-harmless linearly readable in base, and is it the same axis the aligned model refuses along?

- **script** `probe_representation.py` — `probe_representation.py --lineage olmo2 --stage all`
- **code** `c48f441` on `main` ⚠️ DIRTY WORKING TREE — commit does not identify this code
- **duration** 122.1s
- **env** torch 2.6.0+cu124 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S

| stage | mass_mean_peak | mass_mean_peak_layer | logistic_peak | logistic_peak_layer | logistic_L0 | length_only_baseline | n_fit | n_test_per_class |
|---|---|---|---|---|---|---|---|---|
| base | 0.9205 | 6 | 0.9962 | 19 | 0.5 | 0.5568 | 128 | 132 |
| sft | 1.0 | 12 | 1.0 | 11 | 0.5 | 0.5758 | 128 | 132 |
| dpo | 1.0 | 14 | 1.0 | 11 | 0.5 | 0.5758 | 128 | 132 |
| rlvr | 1.0 | 14 | 1.0 | 11 | 0.5 | 0.5758 | 128 | 132 |

Accuracy alone is near-certain to be high and proves little; the informative outputs are the layer profile (L0 vs peak) and the cross-stage cosines computed by aggregate_probe.py.

---

## P1-E1b · 2026-09-16T18:08:36+00:00 · **FAILED**

> Does the aligned model's refusal direction induce refusal when transplanted into BASE, and at what coefficient does induction appear while KL stays sane?  [unit-norm]

- **script** `transplant.py` — `transplant.py --lineage olmo2 --stage all --unit-norm`
- **code** `c48f441` on `main` ⚠️ DIRTY WORKING TREE — commit does not identify this code
- **duration** 81.5s
- **env** torch 2.6.0+cu124 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S
- **error** `NameError: name 'sources' is not defined`

The decisive cell is target=base, source=dpo: if DPO's refusal direction induces refusal in base at acceptable KL, base already had the machinery. Sweeping coefficients also retires the 'you only tried coeff=1' objection to E02.

---

## E02 · 2026-09-16T18:17:24+00:00 · OK

> When across base -> SFT -> DPO does a causally actionable refusal direction appear, and in which layers?

- **script** `run_stage.py` — `run_stage.py --lineage olmo2 --stage all --control --behavioral`
- **code** `9601ed6` on `main` ⚠️ DIRTY WORKING TREE — commit does not identify this code
- **duration** 553.6s
- **env** torch 2.6.0+cu124 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S

| stage | l_star | naive_l_star | pos_star | baseline_refusal | peak_strength | kl_at_l_star | control_peak | substring_baseline | substring_ablated |
|---|---|---|---|---|---|---|---|---|---|
| base | -1 | 23 | -1 | -4.1041 | 0.5383 | None | 0.0591 | 0.0227 | None |
| sft | 24 | 25 | 4 | 3.696 | 9.0004 | 0.098 | 0.0804 | 0.9924 | 0.0076 |
| dpo | 24 | 25 | 4 | 3.8128 | 12.7119 | 0.091 | 0.1228 | 0.9848 | 0.0 |
| rlvr | 24 | 25 | 4 | 4.0666 | 13.5195 | 0.0977 | 0.1287 | 0.9848 | 0.0 |

---

## P1-E1 · 2026-09-16T18:26:39+00:00 · OK

> Is harmful-vs-harmless linearly readable in base, and is it the same axis the aligned model refuses along?

- **script** `probe_representation.py` — `probe_representation.py --lineage olmo2 --stage all`
- **code** `9601ed6` on `main` ⚠️ DIRTY WORKING TREE — commit does not identify this code
- **duration** 139.3s
- **env** torch 2.6.0+cu124 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S

| stage | mass_mean_peak | mass_mean_peak_layer | logistic_peak | logistic_peak_layer | logistic_L0 | length_only_baseline | n_fit | n_test_per_class |
|---|---|---|---|---|---|---|---|---|
| base | 0.9205 | 6 | 0.9962 | 19 | 0.5 | 0.5568 | 128 | 132 |
| sft | 1.0 | 12 | 1.0 | 11 | 0.5 | 0.5758 | 128 | 132 |
| dpo | 1.0 | 14 | 1.0 | 11 | 0.5 | 0.5758 | 128 | 132 |
| rlvr | 1.0 | 14 | 1.0 | 11 | 0.5 | 0.5758 | 128 | 132 |

Accuracy alone is near-certain to be high and proves little; the informative outputs are the layer profile (L0 vs peak) and the cross-stage cosines computed by aggregate_probe.py.

---

## P1-E1b · 2026-09-16T18:29:00+00:00 · OK

> Does the aligned model's refusal direction induce refusal when transplanted into BASE, and at what coefficient does induction appear while KL stays sane?  [unit-norm]

- **script** `transplant.py` — `transplant.py --lineage olmo2 --stage all --unit-norm`
- **code** `9601ed6` on `main` ⚠️ DIRTY WORKING TREE — commit does not identify this code
- **duration** 330.1s
- **env** torch 2.6.0+cu124 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S

| target | source | layer | kind | induces | first_coeff | max_induced | coeff_at_max | kl_at_max | dir_norm | baseline_harmless |
|---|---|---|---|---|---|---|---|---|---|---|
| base | base | 23 | direction | False | None | -4.9394 | 16.0 | 0.408 | 1.0 | -5.6614 |
| base | base | 23 | random | False | None | -5.6658 | 1.0 | 0.001 | 1.0 | -5.6614 |
| base | sft | 24 | direction | False | None | -1.7749 | 16.0 | 0.758 | 1.0 | -5.6614 |
| base | sft | 24 | random | False | None | -5.6737 | 0.5 | 0.0 | 1.0 | -5.6614 |
| base | dpo | 24 | direction | False | None | -1.483 | 16.0 | 0.667 | 1.0 | -5.6614 |
| base | dpo | 24 | random | False | None | -5.3174 | 16.0 | 0.203 | 1.0 | -5.6614 |
| base | rlvr | 24 | direction | False | None | -1.4496 | 16.0 | 0.66 | 1.0 | -5.6614 |
| base | rlvr | 24 | random | False | None | -5.6695 | 0.5 | 0.0 | 1.0 | -5.6614 |
| sft | base | 23 | direction | False | None | -3.3444 | 16.0 | 0.301 | 1.0 | -4.2818 |
| sft | base | 23 | random | False | None | -4.2855 | 0.5 | 0.0 | 1.0 | -4.2818 |
| sft | sft | 24 | direction | False | None | -0.5002 | 16.0 | 0.896 | 1.0 | -4.2818 |
| sft | sft | 24 | random | False | None | -4.3107 | 0.5 | 0.0 | 1.0 | -4.2818 |
| sft | dpo | 24 | direction | False | None | -0.5107 | 16.0 | 0.773 | 1.0 | -4.2818 |
| sft | dpo | 24 | random | False | None | -4.0654 | 16.0 | 0.181 | 1.0 | -4.2818 |
| sft | rlvr | 24 | direction | False | None | -0.5149 | 16.0 | 0.761 | 1.0 | -4.2818 |
| sft | rlvr | 24 | random | False | None | -4.2944 | 0.5 | 0.0 | 1.0 | -4.2818 |
| dpo | base | 23 | direction | False | None | -5.8947 | 16.0 | 0.306 | 1.0 | -7.3035 |
| dpo | base | 23 | random | False | None | -7.3124 | 1.0 | 0.001 | 1.0 | -7.3035 |
| dpo | sft | 24 | direction | False | None | -2.0893 | 16.0 | 0.796 | 1.0 | -7.3035 |
| dpo | sft | 24 | random | False | None | -7.3611 | 0.5 | 0.0 | 1.0 | -7.3035 |
| dpo | dpo | 24 | direction | False | None | -1.7608 | 16.0 | 0.705 | 1.0 | -7.3035 |
| dpo | dpo | 24 | random | False | None | -7.3069 | 0.5 | 0.0 | 1.0 | -7.3035 |
| dpo | rlvr | 24 | direction | False | None | -1.7411 | 16.0 | 0.697 | 1.0 | -7.3035 |
| dpo | rlvr | 24 | random | False | None | -7.3277 | 0.5 | 0.0 | 1.0 | -7.3035 |
| rlvr | base | 23 | direction | False | None | -6.2628 | 16.0 | 0.316 | 1.0 | -7.8087 |
| rlvr | base | 23 | random | False | None | -7.8199 | 1.0 | 0.001 | 1.0 | -7.8087 |
| rlvr | sft | 24 | direction | False | None | -2.263 | 16.0 | 0.803 | 1.0 | -7.8087 |
| rlvr | sft | 24 | random | False | None | -7.8714 | 0.5 | 0.0 | 1.0 | -7.8087 |
| rlvr | dpo | 24 | direction | False | None | -1.9043 | 16.0 | 0.709 | 1.0 | -7.8087 |
| rlvr | dpo | 24 | random | False | None | -7.7329 | 16.0 | 0.205 | 1.0 | -7.8087 |
| rlvr | rlvr | 24 | direction | False | None | -1.885 | 16.0 | 0.7 | 1.0 | -7.8087 |
| rlvr | rlvr | 24 | random | False | None | -7.8375 | 0.5 | 0.0 | 1.0 | -7.8087 |

The decisive cell is target=base, source=dpo: if DPO's refusal direction induces refusal in base at acceptable KL, base already had the machinery. Sweeping coefficients also retires the 'you only tried coeff=1' objection to E02.

---

## P1-E1 · 2026-09-16T18:34:33+00:00 · **FAILED**

> Same axis as the aligned model, or a different one?

- **script** `aggregate_probe.py` — `aggregate_probe.py --lineage olmo2`
- **code** `9601ed6` on `main` ⚠️ DIRTY WORKING TREE — commit does not identify this code
- **duration** 0.0s
- **env** torch 2.6.0+cu124 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S
- **error** `ValueError: operands could not be broadcast together with shapes (2,32,4096) (5,32,4096) `

| stage | logistic_peak | logistic_L0 | mass_mean_peak | length_only | surface_suspect |
|---|---|---|---|---|---|
| base | 0.9962 | 0.5 | 0.9205 | 0.5568 | False |
| sft | 1.0 | 0.5 | 1.0 | 0.5758 | False |
| dpo | 1.0 | 0.5 | 1.0 | 0.5758 | False |
| rlvr | 1.0 | 0.5 | 1.0 | 0.5758 | False |

---

## P1-E1b · 2026-09-16T18:42:15+00:00 · OK

> Does the aligned model's refusal direction induce refusal when transplanted into BASE, and at what coefficient does induction appear while KL stays sane?  [unit-norm]

- **script** `transplant.py` — `transplant.py --lineage olmo2 --stage all --unit-norm`
- **code** `4d6d16f` on `main` ⚠️ DIRTY WORKING TREE — commit does not identify this code
- **duration** 325.7s
- **env** torch 2.6.0+cu124 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S

| target | source | layer | kind | induces | first_coeff | max_induced | coeff_at_max | kl_at_max | dir_norm | baseline_harmless | positive_control_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | base | 23 | direction | False | None | -4.3918 | 27.2979 | 1.112 | 1.0 | -5.6614 | True |
| base | base | 23 | random | False | None | -4.6683 | 54.5958 | 1.58 | 1.0 | -5.6614 | True |
| base | sft | 24 | direction | True | 27.2979 | 4.8017 | 54.5958 | 8.522 | 1.0 | -5.6614 | True |
| base | sft | 24 | random | False | None | -5.5389 | 54.5958 | 1.189 | 1.0 | -5.6614 | True |
| base | dpo | 24 | direction | True | 27.2979 | 2.7331 | 54.5958 | 5.97 | 1.0 | -5.6614 | True |
| base | dpo | 24 | random | False | None | -3.6216 | 54.5958 | 1.473 | 1.0 | -5.6614 | True |
| base | rlvr | 24 | direction | True | 27.2979 | 2.5994 | 54.5958 | 5.796 | 1.0 | -5.6614 | True |
| base | rlvr | 24 | random | False | None | -6.0129 | 6.8245 | 0.089 | 1.0 | -5.6614 | True |
| sft | base | 23 | direction | False | None | -2.8706 | 27.2979 | 0.915 | 1.0 | -4.2818 | True |
| sft | base | 23 | random | False | None | -3.8819 | 54.5958 | 1.621 | 1.0 | -4.2818 | True |
| sft | sft | 24 | direction | True | 27.2979 | 3.2406 | 54.5958 | 8.249 | 1.0 | -4.2818 | True |
| sft | sft | 24 | random | False | None | -4.0601 | 54.5958 | 1.245 | 1.0 | -4.2818 | True |
| sft | dpo | 24 | direction | True | 27.2979 | 2.7635 | 54.5958 | 6.417 | 1.0 | -4.2818 | True |
| sft | dpo | 24 | random | False | None | -3.1736 | 54.5958 | 1.594 | 1.0 | -4.2818 | True |
| sft | rlvr | 24 | direction | True | 27.2979 | 2.6699 | 54.5958 | 6.257 | 1.0 | -4.2818 | True |
| sft | rlvr | 24 | random | False | None | -4.5544 | 27.2979 | 0.345 | 1.0 | -4.2818 | True |
| dpo | base | 23 | direction | False | None | -4.311 | 27.2979 | 0.895 | 1.0 | -7.3035 | True |
| dpo | base | 23 | random | False | None | -4.9993 | 54.5958 | 1.944 | 1.0 | -7.3035 | True |
| dpo | sft | 24 | direction | True | 27.2979 | 3.3293 | 27.2979 | 4.307 | 1.0 | -7.3035 | True |
| dpo | sft | 24 | random | False | None | -5.8702 | 54.5958 | 1.472 | 1.0 | -7.3035 | True |
| dpo | dpo | 24 | direction | True | 27.2979 | 3.5014 | 54.5958 | 7.836 | 1.0 | -7.3035 | True |
| dpo | dpo | 24 | random | False | None | -4.4749 | 54.5958 | 1.845 | 1.0 | -7.3035 | True |
| dpo | rlvr | 24 | direction | True | 27.2979 | 3.4418 | 54.5958 | 7.656 | 1.0 | -7.3035 | True |
| dpo | rlvr | 24 | random | False | None | -6.8651 | 54.5958 | 1.685 | 1.0 | -7.3035 | True |
| rlvr | base | 23 | direction | False | None | -4.5026 | 27.2979 | 0.91 | 1.0 | -7.8087 | True |
| rlvr | base | 23 | random | False | None | -5.1061 | 54.5958 | 2.046 | 1.0 | -7.8087 | True |
| rlvr | sft | 24 | direction | True | 27.2979 | 3.5174 | 27.2979 | 4.472 | 1.0 | -7.8087 | True |
| rlvr | sft | 24 | random | False | None | -6.0197 | 54.5958 | 1.512 | 1.0 | -7.8087 | True |
| rlvr | dpo | 24 | direction | True | 27.2979 | 3.4872 | 54.5958 | 8.014 | 1.0 | -7.8087 | True |
| rlvr | dpo | 24 | random | False | None | -4.6339 | 54.5958 | 1.916 | 1.0 | -7.8087 | True |
| rlvr | rlvr | 24 | direction | True | 27.2979 | 3.4372 | 54.5958 | 7.832 | 1.0 | -7.8087 | True |
| rlvr | rlvr | 24 | random | False | None | -7.0713 | 54.5958 | 1.759 | 1.0 | -7.8087 | True |

The decisive cell is target=base, source=dpo: if DPO's refusal direction induces refusal in base at acceptable KL, base already had the machinery. Sweeping coefficients also retires the 'you only tried coeff=1' objection to E02.

---

## P1-E1 · 2026-09-16T18:47:43+00:00 · OK

> Same axis as the aligned model, or a different one?

- **script** `aggregate_probe.py` — `aggregate_probe.py --lineage olmo2`
- **code** `4d6d16f` on `main` ⚠️ DIRTY WORKING TREE — commit does not identify this code
- **duration** 0.4s
- **env** torch 2.6.0+cu124 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S

| stage | logistic_peak | logistic_L0 | mass_mean_peak | length_only | surface_suspect |
|---|---|---|---|---|---|
| base | 0.9962 | 0.5 | 0.9205 | 0.5568 | False |
| sft | 1.0 | 0.5 | 1.0 | 0.5758 | False |
| dpo | 1.0 | 0.5 | 1.0 | 0.5758 | False |
| rlvr | 1.0 | 0.5 | 1.0 | 0.5758 | False |
| base_vs_sft | SKIPPED_REGIME_OVERRIDE | 2 | 5 |
| base_vs_dpo | SKIPPED_REGIME_OVERRIDE | 2 | 5 |
| base_vs_rlvr | SKIPPED_REGIME_OVERRIDE | 2 | 5 |

---

## P1-E1c · 2026-09-16T18:49:13+00:00 · OK

> When a transplanted direction crosses the induction threshold, is the output coherent refusal text or distributional damage?

- **script** `transplant_text.py` — `transplant_text.py --lineage olmo2 --target base --source sft`
- **code** `8d5fd7f` on `main` ⚠️ DIRTY WORKING TREE — commit does not identify this code
- **duration** 27.8s
- **env** torch 2.6.0+cu124 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S

| target | source | layer | arm | coeff | refusal | refusal_strict | degenerate |
|---|---|---|---|---|---|---|---|
| base | sft | 24 | none | 0.0 | 0.0 | 0.0 | 0.0 |
| base | sft | 24 | direction | 27.29789924621582 | 1.0 | 1.0 | 0.0 |
| base | sft | 24 | random | 27.29789924621582 | 0.0156 | 0.0156 | 0.0 |
| base | sft | 24 | direction | 54.59579849243164 | 0.9844 | 0.9844 | 0.0 |
| base | sft | 24 | random | 54.59579849243164 | 0.0 | 0.0 | 0.0781 |

---

## P1-E1c · 2026-09-16T18:53:20+00:00 · OK

> When a transplanted direction crosses the induction threshold, is the output coherent refusal text or distributional damage?

- **script** `transplant_text.py` — `transplant_text.py --lineage olmo2 --target sft --source sft`
- **code** `9d3a388` on `main` ⚠️ DIRTY WORKING TREE — commit does not identify this code
- **duration** 28.1s
- **env** torch 2.6.0+cu124 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S

| target | source | layer | arm | coeff | refusal | refusal_strict | degenerate |
|---|---|---|---|---|---|---|---|
| sft | sft | 24 | none | 0.0 | 0.125 | 0.125 | 0.0 |
| sft | sft | 24 | direction | 27.29789924621582 | 1.0 | 0.9844 | 0.0625 |
| sft | sft | 24 | random | 27.29789924621582 | 0.1562 | 0.1562 | 0.0 |
| sft | sft | 24 | direction | 54.59579849243164 | 0.9688 | 0.9688 | 0.9844 |
| sft | sft | 24 | random | 54.59579849243164 | 0.1406 | 0.1406 | 0.2188 |
| sft | sft | 27.29789924621582 | True | True |
| sft | sft | 54.59579849243164 | False | True |

---

## P1-E1c · 2026-09-16T18:53:57+00:00 · OK

> When a transplanted direction crosses the induction threshold, is the output coherent refusal text or distributional damage?

- **script** `transplant_text.py` — `transplant_text.py --lineage olmo2 --target base --source base`
- **code** `9d3a388` on `main` ⚠️ DIRTY WORKING TREE — commit does not identify this code
- **duration** 16.8s
- **env** torch 2.6.0+cu124 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S

| target | source | layer | arm | coeff | refusal | refusal_strict | degenerate |
|---|---|---|---|---|---|---|---|
| base | base | 23 | none | 0.0 | 0.0 | 0.0 | 0.0 |
| base | base | 23 | direction | 27.29789924621582 | 0.0 | 0.0 | 0.1562 |
| base | base | 23 | random | 27.29789924621582 | 0.0 | 0.0 | 0.0312 |
| base | base | 27.29789924621582 | True | False |

---

## P1-E1b · 2026-09-17T18:07:52+00:00 · OK

> Does the aligned model's refusal direction induce refusal when transplanted into BASE, and at what coefficient does induction appear while KL stays sane?  [unit-norm]

- **script** `transplant.py` — `transplant.py --lineage olmo2 --stage all --unit-norm --n-control 5`
- **code** `7245e10` on `main` ⚠️ DIRTY WORKING TREE — commit does not identify this code
- **duration** 1019.4s
- **env** torch 2.6.0+cu124 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S

| target | source | layer | kind | induces | first_coeff | max_induced | coeff_at_max | kl_at_max | dir_norm | baseline_harmless | positive_control_ok | delta | null_mean | null_sd | z | n_null_draws | n_null_crossing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | base | 23 | direction | False | None | -4.3918 | 27.2979 | 1.112 | 1.0 | -5.6614 | True | 1.2697 | 0.8492 | 0.8952 | 0.47 | 5 | 0 |
| base | base | 23 | random0 | False | None | -4.6683 | 54.5958 | 1.58 | 1.0 | -5.6614 | True |
| base | base | 23 | random1 | False | None | -5.517 | 54.5958 | 1.325 | 1.0 | -5.6614 | True |
| base | base | 23 | random2 | False | None | -3.455 | 54.5958 | 1.812 | 1.0 | -5.6614 | True |
| base | base | 23 | random3 | False | None | -5.7245 | 27.2979 | 0.42 | 1.0 | -5.6614 | True |
| base | base | 23 | random4 | False | None | -4.6965 | 54.5958 | 2.065 | 1.0 | -5.6614 | True |
| base | sft | 24 | direction | True | 27.2979 | 4.8017 | 54.5958 | 8.522 | 1.0 | -5.6614 | True | 10.4632 | 0.7841 | 0.9254 | 10.46 | 5 | 0 |
| base | sft | 24 | random0 | False | None | -3.9918 | 54.5958 | 1.248 | 1.0 | -5.6614 | True |
| base | sft | 24 | random1 | False | None | -5.3676 | 54.5958 | 1.805 | 1.0 | -5.6614 | True |
| base | sft | 24 | random2 | False | None | -3.7594 | 54.5958 | 1.733 | 1.0 | -5.6614 | True |
| base | sft | 24 | random3 | False | None | -5.5774 | 54.5958 | 1.296 | 1.0 | -5.6614 | True |
| base | sft | 24 | random4 | False | None | -5.6903 | 6.8245 | 0.055 | 1.0 | -5.6614 | True |
| base | dpo | 24 | direction | True | 27.2979 | 2.7331 | 54.5958 | 5.97 | 1.0 | -5.6614 | True | 8.3945 | 1.0419 | 0.9757 | 7.536 | 5 | 0 |
| base | dpo | 24 | random0 | False | None | -4.2342 | 54.5958 | 1.204 | 1.0 | -5.6614 | True |
| base | dpo | 24 | random1 | False | None | -4.8843 | 54.5958 | 1.459 | 1.0 | -5.6614 | True |
| base | dpo | 24 | random2 | False | None | -6.0483 | 6.8245 | 0.068 | 1.0 | -5.6614 | True |
| base | dpo | 24 | random3 | False | None | -3.3772 | 54.5958 | 1.318 | 1.0 | -5.6614 | True |
| base | dpo | 24 | random4 | False | None | -4.5535 | 54.5958 | 1.569 | 1.0 | -5.6614 | True |
| base | rlvr | 24 | direction | True | 27.2979 | 2.5994 | 54.5958 | 5.796 | 1.0 | -5.6614 | True | 8.2608 | 0.5616 | 0.7216 | 10.67 | 5 | 0 |
| base | rlvr | 24 | random0 | False | None | -4.0779 | 54.5958 | 1.27 | 1.0 | -5.6614 | True |
| base | rlvr | 24 | random1 | False | None | -4.9013 | 54.5958 | 1.536 | 1.0 | -5.6614 | True |
| base | rlvr | 24 | random2 | False | None | -4.9518 | 54.5958 | 1.493 | 1.0 | -5.6614 | True |
| base | rlvr | 24 | random3 | False | None | -5.642 | 27.2979 | 0.315 | 1.0 | -5.6614 | True |
| base | rlvr | 24 | random4 | False | None | -5.9263 | 6.8245 | 0.058 | 1.0 | -5.6614 | True |
| sft | base | 23 | direction | False | None | -2.8706 | 27.2979 | 0.915 | 1.0 | -4.2818 | True | 1.4112 | 0.4415 | 0.5228 | 1.855 | 5 | 0 |
| sft | base | 23 | random0 | False | None | -3.8819 | 54.5958 | 1.621 | 1.0 | -4.2818 | True |
| sft | base | 23 | random1 | False | None | -4.1439 | 54.5958 | 1.391 | 1.0 | -4.2818 | True |
| sft | base | 23 | random2 | False | None | -3.0778 | 27.2979 | 0.407 | 1.0 | -4.2818 | True |
| sft | base | 23 | random3 | False | None | -4.4556 | 6.8245 | 0.042 | 1.0 | -4.2818 | True |
| sft | base | 23 | random4 | False | None | -3.6423 | 54.5958 | 1.665 | 1.0 | -4.2818 | True |
| sft | sft | 24 | direction | True | 27.2979 | 3.2406 | 54.5958 | 8.249 | 1.0 | -4.2818 | True | 7.5224 | 0.3898 | 0.4191 | 17.021 | 5 | 0 |
| sft | sft | 24 | random0 | False | None | -3.4395 | 27.2979 | 0.352 | 1.0 | -4.2818 | True |
| sft | sft | 24 | random1 | False | None | -3.6374 | 54.5958 | 1.336 | 1.0 | -4.2818 | True |
| sft | sft | 24 | random2 | False | None | -3.7396 | 54.5958 | 1.76 | 1.0 | -4.2818 | True |
| sft | sft | 24 | random3 | False | None | -4.1801 | 27.2979 | 0.332 | 1.0 | -4.2818 | True |
| sft | sft | 24 | random4 | False | None | -4.4635 | 6.8245 | 0.034 | 1.0 | -4.2818 | True |
| sft | dpo | 24 | direction | True | 27.2979 | 2.7635 | 54.5958 | 6.417 | 1.0 | -4.2818 | True | 7.0453 | 0.7106 | 0.4111 | 15.409 | 5 | 0 |
| sft | dpo | 24 | random0 | False | None | -3.4794 | 54.5958 | 1.222 | 1.0 | -4.2818 | True |
| sft | dpo | 24 | random1 | False | None | -3.4636 | 54.5958 | 1.493 | 1.0 | -4.2818 | True |
| sft | dpo | 24 | random2 | False | None | -4.266 | 13.649 | 0.158 | 1.0 | -4.2818 | True |
| sft | dpo | 24 | random3 | False | None | -3.482 | 27.2979 | 0.332 | 1.0 | -4.2818 | True |
| sft | dpo | 24 | random4 | False | None | -3.1647 | 54.5958 | 1.863 | 1.0 | -4.2818 | True |
| sft | rlvr | 24 | direction | True | 27.2979 | 2.6699 | 54.5958 | 6.257 | 1.0 | -4.2818 | True | 6.9517 | 0.4241 | 0.3728 | 17.51 | 5 | 0 |
| sft | rlvr | 24 | random0 | False | None | -3.3148 | 54.5958 | 1.394 | 1.0 | -4.2818 | True |
| sft | rlvr | 24 | random1 | False | None | -3.8333 | 27.2979 | 0.386 | 1.0 | -4.2818 | True |
| sft | rlvr | 24 | random2 | False | None | -3.7372 | 54.5958 | 1.484 | 1.0 | -4.2818 | True |
| sft | rlvr | 24 | random3 | False | None | -4.2703 | 27.2979 | 0.259 | 1.0 | -4.2818 | True |
| sft | rlvr | 24 | random4 | False | None | -4.1331 | 27.2979 | 0.333 | 1.0 | -4.2818 | True |
| dpo | base | 23 | direction | False | None | -4.311 | 27.2979 | 0.895 | 1.0 | -7.3035 | True | 2.9925 | 1.9183 | 1.0331 | 1.04 | 5 | 0 |
| dpo | base | 23 | random0 | False | None | -4.9993 | 54.5958 | 1.944 | 1.0 | -7.3035 | True |
| dpo | base | 23 | random1 | False | None | -5.7847 | 54.5958 | 1.68 | 1.0 | -7.3035 | True |
| dpo | base | 23 | random2 | False | None | -4.3837 | 54.5958 | 2.259 | 1.0 | -7.3035 | True |
| dpo | base | 23 | random3 | False | None | -6.9902 | 54.5958 | 2.024 | 1.0 | -7.3035 | True |
| dpo | base | 23 | random4 | False | None | -4.7677 | 54.5958 | 2.012 | 1.0 | -7.3035 | True |
| dpo | sft | 24 | direction | True | 27.2979 | 3.3293 | 27.2979 | 4.307 | 1.0 | -7.3035 | True | 10.6328 | 2.0846 | 0.7594 | 11.257 | 5 | 0 |
| dpo | sft | 24 | random0 | False | None | -4.5233 | 54.5958 | 2.379 | 1.0 | -7.3035 | True |
| dpo | sft | 24 | random1 | False | None | -4.8653 | 54.5958 | 1.596 | 1.0 | -7.3035 | True |
| dpo | sft | 24 | random2 | False | None | -4.6536 | 54.5958 | 2.012 | 1.0 | -7.3035 | True |
| dpo | sft | 24 | random3 | False | None | -5.8321 | 54.5958 | 2.101 | 1.0 | -7.3035 | True |
| dpo | sft | 24 | random4 | False | None | -6.22 | 54.5958 | 1.953 | 1.0 | -7.3035 | True |
| dpo | dpo | 24 | direction | True | 27.2979 | 3.5014 | 54.5958 | 7.836 | 1.0 | -7.3035 | True | 10.8049 | 2.2306 | 1.1746 | 7.3 | 5 | 0 |
| dpo | dpo | 24 | random0 | False | None | -4.8807 | 54.5958 | 1.551 | 1.0 | -7.3035 | True |
| dpo | dpo | 24 | random1 | False | None | -4.7271 | 54.5958 | 1.808 | 1.0 | -7.3035 | True |
| dpo | dpo | 24 | random2 | False | None | -7.1212 | 54.5958 | 2.18 | 1.0 | -7.3035 | True |
| dpo | dpo | 24 | random3 | False | None | -4.4322 | 54.5958 | 1.974 | 1.0 | -7.3035 | True |
| dpo | dpo | 24 | random4 | False | None | -4.2034 | 54.5958 | 2.289 | 1.0 | -7.3035 | True |
| dpo | rlvr | 24 | direction | True | 27.2979 | 3.4418 | 54.5958 | 7.656 | 1.0 | -7.3035 | True | 10.7453 | 2.0699 | 0.7712 | 11.249 | 5 | 0 |
| dpo | rlvr | 24 | random0 | False | None | -4.3794 | 54.5958 | 1.686 | 1.0 | -7.3035 | True |
| dpo | rlvr | 24 | random1 | False | None | -4.8459 | 54.5958 | 2.207 | 1.0 | -7.3035 | True |
| dpo | rlvr | 24 | random2 | False | None | -4.8773 | 54.5958 | 1.679 | 1.0 | -7.3035 | True |
| dpo | rlvr | 24 | random3 | False | None | -5.814 | 54.5958 | 1.83 | 1.0 | -7.3035 | True |
| dpo | rlvr | 24 | random4 | False | None | -6.2512 | 54.5958 | 2.557 | 1.0 | -7.3035 | True |
| rlvr | base | 23 | direction | False | None | -4.5026 | 27.2979 | 0.91 | 1.0 | -7.8087 | True | 3.3061 | 2.2905 | 1.0599 | 0.958 | 5 | 0 |
| rlvr | base | 23 | random0 | False | None | -5.1061 | 54.5958 | 2.046 | 1.0 | -7.8087 | True |
| rlvr | base | 23 | random1 | False | None | -5.9074 | 54.5958 | 1.74 | 1.0 | -7.8087 | True |
| rlvr | base | 23 | random2 | False | None | -4.5119 | 54.5958 | 2.348 | 1.0 | -7.8087 | True |
| rlvr | base | 23 | random3 | False | None | -7.1796 | 54.5958 | 2.119 | 1.0 | -7.8087 | True |
| rlvr | base | 23 | random4 | False | None | -4.8861 | 54.5958 | 2.095 | 1.0 | -7.8087 | True |
| rlvr | sft | 24 | direction | True | 27.2979 | 3.5174 | 27.2979 | 4.472 | 1.0 | -7.8087 | True | 11.3261 | 2.4217 | 0.7938 | 11.218 | 5 | 0 |
| rlvr | sft | 24 | random0 | False | None | -4.6751 | 54.5958 | 2.52 | 1.0 | -7.8087 | True |
| rlvr | sft | 24 | random1 | False | None | -5.0142 | 54.5958 | 1.66 | 1.0 | -7.8087 | True |
| rlvr | sft | 24 | random2 | False | None | -4.7703 | 54.5958 | 2.093 | 1.0 | -7.8087 | True |
| rlvr | sft | 24 | random3 | False | None | -6.0839 | 54.5958 | 2.231 | 1.0 | -7.8087 | True |
| rlvr | sft | 24 | random4 | False | None | -6.3912 | 54.5958 | 2.007 | 1.0 | -7.8087 | True |
| rlvr | dpo | 24 | direction | True | 27.2979 | 3.4872 | 54.5958 | 8.014 | 1.0 | -7.8087 | True | 11.2958 | 2.5814 | 1.1965 | 7.284 | 5 | 0 |
| rlvr | dpo | 24 | random0 | False | None | -5.0564 | 54.5958 | 1.59 | 1.0 | -7.8087 | True |
| rlvr | dpo | 24 | random1 | False | None | -4.8448 | 54.5958 | 1.87 | 1.0 | -7.8087 | True |
| rlvr | dpo | 24 | random2 | False | None | -7.314 | 54.5958 | 2.309 | 1.0 | -7.8087 | True |
| rlvr | dpo | 24 | random3 | False | None | -4.5603 | 54.5958 | 2.078 | 1.0 | -7.8087 | True |
| rlvr | dpo | 24 | random4 | False | None | -4.3606 | 54.5958 | 2.368 | 1.0 | -7.8087 | True |
| rlvr | rlvr | 24 | direction | True | 27.2979 | 3.4372 | 54.5958 | 7.832 | 1.0 | -7.8087 | True | 11.2459 | 2.4584 | 0.7659 | 11.473 | 5 | 0 |
| rlvr | rlvr | 24 | random0 | False | None | -4.491 | 54.5958 | 1.732 | 1.0 | -7.8087 | True |
| rlvr | rlvr | 24 | random1 | False | None | -4.9579 | 54.5958 | 2.281 | 1.0 | -7.8087 | True |
| rlvr | rlvr | 24 | random2 | False | None | -5.0048 | 54.5958 | 1.73 | 1.0 | -7.8087 | True |
| rlvr | rlvr | 24 | random3 | False | None | -5.9761 | 54.5958 | 1.894 | 1.0 | -7.8087 | True |
| rlvr | rlvr | 24 | random4 | False | None | -6.3214 | 54.5958 | 2.632 | 1.0 | -7.8087 | True |

The decisive cell is target=base, source=dpo: if DPO's refusal direction induces refusal in base at acceptable KL, base already had the machinery. Sweeping coefficients also retires the 'you only tried coeff=1' objection to E02.

---

## P1-E1b · 2026-09-17T18:25:40+00:00 · **FAILED**

> Does the aligned model's refusal direction induce refusal when transplanted into BASE, and at what coefficient does induction appear while KL stays sane?  [unit-norm]

- **script** `transplant.py` — `transplant.py --lineage zephyr --stage all --unit-norm --n-control 5`
- **code** `7245e10` on `main` ⚠️ DIRTY WORKING TREE — commit does not identify this code
- **duration** 193.4s
- **env** torch 2.6.0+cu124 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S
- **error** `RuntimeError: Task error: File reconstruction error: Internal Writer Error: Background writer channel closed`

| target | source | layer | kind | induces | first_coeff | max_induced | coeff_at_max | kl_at_max | dir_norm | baseline_harmless | positive_control_ok | delta | null_mean | null_sd | z | n_null_draws | n_null_crossing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | base | 15 | direction | False | None | -1.7645 | 3.7073 | 0.829 | 1.0 | -3.0244 | True | 1.2598 | 0.2198 | 0.3509 | 2.964 | 5 | 0 |
| base | base | 15 | random0 | False | None | -3.3422 | 1.8537 | 0.114 | 1.0 | -3.0244 | True |
| base | base | 15 | random1 | False | None | -2.5659 | 3.7073 | 0.373 | 1.0 | -3.0244 | True |
| base | base | 15 | random2 | False | None | -2.6822 | 1.8537 | 0.112 | 1.0 | -3.0244 | True |
| base | base | 15 | random3 | False | None | -2.4753 | 3.7073 | 0.366 | 1.0 | -3.0244 | True |
| base | base | 15 | random4 | False | None | -2.9569 | 1.8537 | 0.126 | 1.0 | -3.0244 | True |
| base | sft | 20 | direction | False | None | -2.0847 | 7.4146 | 0.839 | 1.0 | -3.0244 | True | 0.9397 | 0.0456 | 0.3598 | 2.485 | 5 | 0 |
| base | sft | 20 | random0 | False | None | -3.1233 | 1.8537 | 0.008 | 1.0 | -3.0244 | True |
| base | sft | 20 | random1 | False | None | -3.182 | 1.8537 | 0.008 | 1.0 | -3.0244 | True |
| base | sft | 20 | random2 | False | None | -3.1217 | 1.8537 | 0.009 | 1.0 | -3.0244 | True |
| base | sft | 20 | random3 | False | None | -3.1299 | 1.8537 | 0.008 | 1.0 | -3.0244 | True |
| base | sft | 20 | random4 | False | None | -2.3367 | 14.8293 | 0.803 | 1.0 | -3.0244 | True |
| base | dpo | 17 | direction | False | None | -1.6242 | 3.7073 | 0.814 | 1.0 | -3.0244 | True | 1.4001 | 0.3398 | 0.365 | 2.906 | 5 | 0 |
| base | dpo | 17 | random0 | False | None | -2.8581 | 3.7073 | 0.092 | 1.0 | -3.0244 | True |
| base | dpo | 17 | random1 | False | None | -3.0358 | 1.8537 | 0.021 | 1.0 | -3.0244 | True |
| base | dpo | 17 | random2 | False | None | -2.0809 | 7.4146 | 0.551 | 1.0 | -3.0244 | True |
| base | dpo | 17 | random3 | False | None | -2.8006 | 3.7073 | 0.113 | 1.0 | -3.0244 | True |
| base | dpo | 17 | random4 | False | None | -2.6476 | 7.4146 | 0.747 | 1.0 | -3.0244 | True |

The decisive cell is target=base, source=dpo: if DPO's refusal direction induces refusal in base at acceptable KL, base already had the machinery. Sweeping coefficients also retires the 'you only tried coeff=1' objection to E02.

---

## P1-E1b · 2026-09-17T18:30:21+00:00 · **FAILED**

> Does the aligned model's refusal direction induce refusal when transplanted into BASE, and at what coefficient does induction appear while KL stays sane?  [unit-norm]

- **script** `transplant.py` — `transplant.py --lineage zephyr --stage all --unit-norm --n-control 5`
- **code** `7245e10` on `main` ⚠️ DIRTY WORKING TREE — commit does not identify this code
- **duration** 172.8s
- **env** torch 2.6.0+cu124 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S
- **error** `RuntimeError: Task error: File reconstruction error: Internal Writer Error: Background writer channel closed`

| target | source | layer | kind | induces | first_coeff | max_induced | coeff_at_max | kl_at_max | dir_norm | baseline_harmless | positive_control_ok | delta | null_mean | null_sd | z | n_null_draws | n_null_crossing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | base | 15 | direction | False | None | -1.7645 | 3.7073 | 0.829 | 1.0 | -3.0244 | True | 1.2598 | 0.2198 | 0.3509 | 2.964 | 5 | 0 |
| base | base | 15 | random0 | False | None | -3.3422 | 1.8537 | 0.114 | 1.0 | -3.0244 | True |
| base | base | 15 | random1 | False | None | -2.5659 | 3.7073 | 0.373 | 1.0 | -3.0244 | True |
| base | base | 15 | random2 | False | None | -2.6822 | 1.8537 | 0.112 | 1.0 | -3.0244 | True |
| base | base | 15 | random3 | False | None | -2.4753 | 3.7073 | 0.366 | 1.0 | -3.0244 | True |
| base | base | 15 | random4 | False | None | -2.9569 | 1.8537 | 0.126 | 1.0 | -3.0244 | True |
| base | sft | 20 | direction | False | None | -2.0847 | 7.4146 | 0.839 | 1.0 | -3.0244 | True | 0.9397 | 0.0456 | 0.3598 | 2.485 | 5 | 0 |
| base | sft | 20 | random0 | False | None | -3.1233 | 1.8537 | 0.008 | 1.0 | -3.0244 | True |
| base | sft | 20 | random1 | False | None | -3.182 | 1.8537 | 0.008 | 1.0 | -3.0244 | True |
| base | sft | 20 | random2 | False | None | -3.1217 | 1.8537 | 0.009 | 1.0 | -3.0244 | True |
| base | sft | 20 | random3 | False | None | -3.1299 | 1.8537 | 0.008 | 1.0 | -3.0244 | True |
| base | sft | 20 | random4 | False | None | -2.3367 | 14.8293 | 0.803 | 1.0 | -3.0244 | True |
| base | dpo | 17 | direction | False | None | -1.6242 | 3.7073 | 0.814 | 1.0 | -3.0244 | True | 1.4001 | 0.3398 | 0.365 | 2.906 | 5 | 0 |
| base | dpo | 17 | random0 | False | None | -2.8581 | 3.7073 | 0.092 | 1.0 | -3.0244 | True |
| base | dpo | 17 | random1 | False | None | -3.0358 | 1.8537 | 0.021 | 1.0 | -3.0244 | True |
| base | dpo | 17 | random2 | False | None | -2.0809 | 7.4146 | 0.551 | 1.0 | -3.0244 | True |
| base | dpo | 17 | random3 | False | None | -2.8006 | 3.7073 | 0.113 | 1.0 | -3.0244 | True |
| base | dpo | 17 | random4 | False | None | -2.6476 | 7.4146 | 0.747 | 1.0 | -3.0244 | True |

The decisive cell is target=base, source=dpo: if DPO's refusal direction induces refusal in base at acceptable KL, base already had the machinery. Sweeping coefficients also retires the 'you only tried coeff=1' objection to E02.

---

## P1-E1b · 2026-09-17T18:38:12+00:00 · **FAILED**

> Does the aligned model's refusal direction induce refusal when transplanted into BASE, and at what coefficient does induction appear while KL stays sane?  [unit-norm]

- **script** `transplant.py` — `transplant.py --lineage zephyr --stage all --unit-norm --n-control 5`
- **code** `7245e10` on `main` ⚠️ DIRTY WORKING TREE — commit does not identify this code
- **duration** 164.1s
- **env** torch 2.6.0+cu124 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S
- **error** `KeyboardInterrupt: `

| target | source | layer | kind | induces | first_coeff | max_induced | coeff_at_max | kl_at_max | dir_norm | baseline_harmless | positive_control_ok | delta | null_mean | null_sd | z | n_null_draws | n_null_crossing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | base | 15 | direction | False | None | -1.7645 | 3.7073 | 0.829 | 1.0 | -3.0244 | True | 1.2598 | 0.2198 | 0.3509 | 2.964 | 5 | 0 |
| base | base | 15 | random0 | False | None | -3.3422 | 1.8537 | 0.114 | 1.0 | -3.0244 | True |
| base | base | 15 | random1 | False | None | -2.5659 | 3.7073 | 0.373 | 1.0 | -3.0244 | True |
| base | base | 15 | random2 | False | None | -2.6822 | 1.8537 | 0.112 | 1.0 | -3.0244 | True |
| base | base | 15 | random3 | False | None | -2.4753 | 3.7073 | 0.366 | 1.0 | -3.0244 | True |
| base | base | 15 | random4 | False | None | -2.9569 | 1.8537 | 0.126 | 1.0 | -3.0244 | True |
| base | sft | 20 | direction | False | None | -2.0847 | 7.4146 | 0.839 | 1.0 | -3.0244 | True | 0.9397 | 0.0456 | 0.3598 | 2.485 | 5 | 0 |
| base | sft | 20 | random0 | False | None | -3.1233 | 1.8537 | 0.008 | 1.0 | -3.0244 | True |
| base | sft | 20 | random1 | False | None | -3.182 | 1.8537 | 0.008 | 1.0 | -3.0244 | True |
| base | sft | 20 | random2 | False | None | -3.1217 | 1.8537 | 0.009 | 1.0 | -3.0244 | True |
| base | sft | 20 | random3 | False | None | -3.1299 | 1.8537 | 0.008 | 1.0 | -3.0244 | True |
| base | sft | 20 | random4 | False | None | -2.3367 | 14.8293 | 0.803 | 1.0 | -3.0244 | True |
| base | dpo | 17 | direction | False | None | -1.6242 | 3.7073 | 0.814 | 1.0 | -3.0244 | True | 1.4001 | 0.3398 | 0.365 | 2.906 | 5 | 0 |
| base | dpo | 17 | random0 | False | None | -2.8581 | 3.7073 | 0.092 | 1.0 | -3.0244 | True |
| base | dpo | 17 | random1 | False | None | -3.0358 | 1.8537 | 0.021 | 1.0 | -3.0244 | True |
| base | dpo | 17 | random2 | False | None | -2.0809 | 7.4146 | 0.551 | 1.0 | -3.0244 | True |
| base | dpo | 17 | random3 | False | None | -2.8006 | 3.7073 | 0.113 | 1.0 | -3.0244 | True |
| base | dpo | 17 | random4 | False | None | -2.6476 | 7.4146 | 0.747 | 1.0 | -3.0244 | True |

The decisive cell is target=base, source=dpo: if DPO's refusal direction induces refusal in base at acceptable KL, base already had the machinery. Sweeping coefficients also retires the 'you only tried coeff=1' objection to E02.

---

## P1-E1b · 2026-09-17T18:41:00+00:00 · OK

> Does the aligned model's refusal direction induce refusal when transplanted into BASE, and at what coefficient does induction appear while KL stays sane?  [unit-norm]

- **script** `transplant.py` — `transplant.py --lineage zephyr --stage sft --unit-norm --n-control 5`
- **code** `7245e10` on `main` ⚠️ DIRTY WORKING TREE — commit does not identify this code
- **duration** 179.4s
- **env** torch 2.6.0+cu124 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S

| target | source | layer | kind | induces | first_coeff | max_induced | coeff_at_max | kl_at_max | dir_norm | baseline_harmless | positive_control_ok | delta | null_mean | null_sd | z | n_null_draws | n_null_crossing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sft | base | 15 | direction | True | 7.4146 | 0.448 | 7.4146 | 1.604 | 1.0 | -4.094 | True | 4.542 | 2.0019 | 1.4258 | 1.782 | 5 | 0 |
| sft | base | 15 | random0 | False | None | -3.9059 | 7.4146 | 0.752 | 1.0 | -4.094 | True |
| sft | base | 15 | random1 | False | None | -1.9275 | 7.4146 | 0.671 | 1.0 | -4.094 | True |
| sft | base | 15 | random2 | False | None | -2.8399 | 7.4146 | 0.683 | 1.0 | -4.094 | True |
| sft | base | 15 | random3 | False | None | -0.0624 | 7.4146 | 1.137 | 1.0 | -4.094 | True |
| sft | base | 15 | random4 | False | None | -1.7245 | 7.4146 | 0.863 | 1.0 | -4.094 | True |
| sft | sft | 20 | direction | True | 14.8293 | 0.3965 | 14.8293 | 2.381 | 1.0 | -4.094 | True | 4.4904 | 0.3002 | 0.8209 | 5.105 | 5 | 0 |
| sft | sft | 20 | random0 | False | None | -4.0925 | 1.8537 | 0.003 | 1.0 | -4.094 | True |
| sft | sft | 20 | random1 | False | None | -4.2067 | 1.8537 | 0.003 | 1.0 | -4.094 | True |
| sft | sft | 20 | random2 | False | None | -4.1417 | 1.8537 | 0.003 | 1.0 | -4.094 | True |
| sft | sft | 20 | random3 | False | None | -4.2002 | 1.8537 | 0.003 | 1.0 | -4.094 | True |
| sft | sft | 20 | random4 | False | None | -2.3277 | 29.6586 | 2.114 | 1.0 | -4.094 | True |
| sft | dpo | 17 | direction | True | 3.7073 | 0.7434 | 7.4146 | 2.689 | 1.0 | -4.094 | True | 4.8373 | 1.1953 | 1.0813 | 3.368 | 5 | 0 |
| sft | dpo | 17 | random0 | False | None | -2.6976 | 14.8293 | 1.894 | 1.0 | -4.094 | True |
| sft | dpo | 17 | random1 | False | None | -3.4452 | 7.4146 | 0.197 | 1.0 | -4.094 | True |
| sft | dpo | 17 | random2 | False | None | -1.4041 | 14.8293 | 1.486 | 1.0 | -4.094 | True |
| sft | dpo | 17 | random3 | False | None | -4.3227 | 1.8537 | 0.008 | 1.0 | -4.094 | True |
| sft | dpo | 17 | random4 | False | None | -2.6235 | 14.8293 | 1.786 | 1.0 | -4.094 | True |

The decisive cell is target=base, source=dpo: if DPO's refusal direction induces refusal in base at acceptable KL, base already had the machinery. Sweeping coefficients also retires the 'you only tried coeff=1' objection to E02.

---

## P1-E1b · 2026-09-17T18:44:01+00:00 · OK

> Does the aligned model's refusal direction induce refusal when transplanted into BASE, and at what coefficient does induction appear while KL stays sane?  [unit-norm]

- **script** `transplant.py` — `transplant.py --lineage zephyr --stage dpo --unit-norm --n-control 5`
- **code** `7245e10` on `main` ⚠️ DIRTY WORKING TREE — commit does not identify this code
- **duration** 181.9s
- **env** torch 2.6.0+cu124 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S

| target | source | layer | kind | induces | first_coeff | max_induced | coeff_at_max | kl_at_max | dir_norm | baseline_harmless | positive_control_ok | delta | null_mean | null_sd | z | n_null_draws | n_null_crossing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dpo | base | 15 | direction | False | None | -2.8617 | 7.4146 | 2.658 | 1.0 | -8.8016 | True | 5.9399 | 4.2492 | 2.7864 | 0.607 | 5 | 0 |
| dpo | base | 15 | random0 | False | None | -7.6295 | 7.4146 | 1.611 | 1.0 | -8.8016 | True |
| dpo | base | 15 | random1 | False | None | -4.8357 | 7.4146 | 1.749 | 1.0 | -8.8016 | True |
| dpo | base | 15 | random2 | False | None | -6.5716 | 3.7073 | 0.374 | 1.0 | -8.8016 | True |
| dpo | base | 15 | random3 | False | None | -0.646 | 7.4146 | 2.91 | 1.0 | -8.8016 | True |
| dpo | base | 15 | random4 | False | None | -3.0791 | 7.4146 | 1.966 | 1.0 | -8.8016 | True |
| dpo | sft | 20 | direction | True | 14.8293 | 0.0178 | 14.8293 | 3.736 | 1.0 | -8.8016 | True | 8.8194 | 0.6779 | 1.5668 | 5.196 | 5 | 0 |
| dpo | sft | 20 | random0 | False | None | -8.9897 | 1.8537 | 0.012 | 1.0 | -8.8016 | True |
| dpo | sft | 20 | random1 | False | None | -8.5093 | 14.8293 | 1.145 | 1.0 | -8.8016 | True |
| dpo | sft | 20 | random2 | False | None | -8.6455 | 14.8293 | 1.275 | 1.0 | -8.8016 | True |
| dpo | sft | 20 | random3 | False | None | -9.1177 | 1.8537 | 0.01 | 1.0 | -8.8016 | True |
| dpo | sft | 20 | random4 | False | None | -5.356 | 14.8293 | 1.232 | 1.0 | -8.8016 | True |
| dpo | dpo | 17 | direction | True | 7.4146 | 0.9098 | 7.4146 | 4.544 | 1.0 | -8.8016 | True | 9.7113 | 3.2844 | 2.1191 | 3.033 | 5 | 0 |
| dpo | dpo | 17 | random0 | False | None | -6.149 | 7.4146 | 0.698 | 1.0 | -8.8016 | True |
| dpo | dpo | 17 | random1 | False | None | -6.896 | 14.8293 | 3.989 | 1.0 | -8.8016 | True |
| dpo | dpo | 17 | random2 | False | None | -2.2762 | 14.8293 | 2.709 | 1.0 | -8.8016 | True |
| dpo | dpo | 17 | random3 | False | None | -7.6157 | 14.8293 | 3.37 | 1.0 | -8.8016 | True |
| dpo | dpo | 17 | random4 | False | None | -4.6489 | 14.8293 | 3.537 | 1.0 | -8.8016 | True |

The decisive cell is target=base, source=dpo: if DPO's refusal direction induces refusal in base at acceptable KL, base already had the machinery. Sweeping coefficients also retires the 'you only tried coeff=1' objection to E02.

---

## P1-E1b · 2026-09-17T18:50:22+00:00 · OK

> Does the aligned model's refusal direction induce refusal when transplanted into BASE, and at what coefficient does induction appear while KL stays sane?  [unit-norm]

- **script** `transplant.py` — `transplant.py --lineage zephyr --stage all --unit-norm --n-control 5`
- **code** `b9debf4` on `main` ⚠️ DIRTY WORKING TREE — commit does not identify this code
- **duration** 1163.2s
- **env** torch 2.6.0+cu124 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S

| target | source | layer | kind | induces | first_coeff | max_induced | coeff_at_max | kl_at_max | dir_norm | baseline_harmless | positive_control_ok | delta | random_mean | random_sd | random_z | random_draws | random_crossing | shuffled_mean | shuffled_sd | shuffled_z | shuffled_draws | shuffled_crossing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | base | 15 | direction | False | None | -1.7645 | 3.7073 | 0.829 | 1.0 | -3.0244 | True | 1.2598 | 0.2555 | 0.312 | 3.219 | 5 | 0 | -0.0232 | 0.4154 | 3.089 | 5 | 0 |
| base | base | 15 | random0 | False | None | -3.2128 | 1.0742 | 0.032 | 1.0 | -3.0244 | True |
| base | base | 15 | random1 | False | None | -2.5173 | 4.3724 | 0.526 | 1.0 | -3.0244 | True |
| base | base | 15 | random2 | False | None | -2.6822 | 1.8537 | 0.112 | 1.0 | -3.0244 | True |
| base | base | 15 | random3 | False | None | -2.4753 | 3.7073 | 0.366 | 1.0 | -3.0244 | True |
| base | base | 15 | random4 | False | None | -2.9569 | 1.8537 | 0.126 | 1.0 | -3.0244 | True |
| base | base | 15 | shuffled0 | False | None | -2.3923 | 3.7073 | 1.119 | 1.0 | -3.0244 | True |
| base | base | 15 | shuffled1 | False | None | -2.9267 | 1.8537 | 0.103 | 1.0 | -3.0244 | True |
| base | base | 15 | shuffled2 | False | None | -3.1933 | 3.7073 | 0.617 | 1.0 | -3.0244 | True |
| base | base | 15 | shuffled3 | False | None | -3.4771 | 1.0742 | 0.051 | 1.0 | -3.0244 | True |
| base | base | 15 | shuffled4 | False | None | -3.2483 | 1.0742 | 0.034 | 1.0 | -3.0244 | True |
| base | sft | 20 | direction | False | None | -2.0847 | 7.4146 | 0.839 | 1.0 | -3.0244 | True | 0.9397 | 0.0859 | 0.3367 | 2.535 | 5 | 0 | 0.219 | 0.4619 | 1.56 | 5 | 0 |
| base | sft | 20 | random0 | False | None | -3.0753 | 1.0742 | 0.003 | 1.0 | -3.0244 | True |
| base | sft | 20 | random1 | False | None | -3.1139 | 1.0742 | 0.003 | 1.0 | -3.0244 | True |
| base | sft | 20 | random2 | False | None | -3.0766 | 1.0742 | 0.003 | 1.0 | -3.0244 | True |
| base | sft | 20 | random3 | False | None | -3.0895 | 1.0742 | 0.003 | 1.0 | -3.0244 | True |
| base | sft | 20 | random4 | False | None | -2.3367 | 14.8293 | 0.803 | 1.0 | -3.0244 | True |
| base | sft | 20 | shuffled0 | False | None | -2.0752 | 7.4146 | 0.381 | 1.0 | -3.0244 | True |
| base | sft | 20 | shuffled1 | False | None | -3.2014 | 1.0742 | 0.009 | 1.0 | -3.0244 | True |
| base | sft | 20 | shuffled2 | False | None | -2.9867 | 1.8537 | 0.041 | 1.0 | -3.0244 | True |
| base | sft | 20 | shuffled3 | False | None | -3.1253 | 1.0742 | 0.009 | 1.0 | -3.0244 | True |
| base | sft | 20 | shuffled4 | False | None | -2.638 | 7.4146 | 0.294 | 1.0 | -3.0244 | True |
| base | dpo | 17 | direction | False | None | -1.6242 | 3.7073 | 0.814 | 1.0 | -3.0244 | True | 1.4001 | 0.3511 | 0.3598 | 2.915 | 5 | 0 | 0.428 | 0.7367 | 1.32 | 5 | 0 |
| base | dpo | 17 | random0 | False | None | -2.828 | 4.3724 | 0.135 | 1.0 | -3.0244 | True |
| base | dpo | 17 | random1 | False | None | -3.0341 | 1.0742 | 0.007 | 1.0 | -3.0244 | True |
| base | dpo | 17 | random2 | False | None | -2.0809 | 7.4146 | 0.551 | 1.0 | -3.0244 | True |
| base | dpo | 17 | random3 | False | None | -2.7823 | 4.3724 | 0.162 | 1.0 | -3.0244 | True |
| base | dpo | 17 | random4 | False | None | -2.6411 | 4.3724 | 0.263 | 1.0 | -3.0244 | True |
| base | dpo | 17 | shuffled0 | False | None | -1.4384 | 7.4146 | 1.019 | 1.0 | -3.0244 | True |
| base | dpo | 17 | shuffled1 | False | None | -3.35 | 1.0742 | 0.042 | 1.0 | -3.0244 | True |
| base | dpo | 17 | shuffled2 | False | None | -2.5038 | 3.7073 | 0.718 | 1.0 | -3.0244 | True |
| base | dpo | 17 | shuffled3 | False | None | -2.5894 | 4.3724 | 0.501 | 1.0 | -3.0244 | True |
| base | dpo | 17 | shuffled4 | False | None | -3.1001 | 1.0742 | 0.023 | 1.0 | -3.0244 | True |
| sft | base | 15 | direction | True | 7.4146 | 0.448 | 7.4146 | 1.604 | 1.0 | -4.094 | True | 4.542 | 2.0019 | 1.4258 | 1.782 | 5 | 0 | 1.412 | 2.5417 | 1.231 | 5 | 1 |
| sft | base | 15 | random0 | False | None | -3.9059 | 7.4146 | 0.752 | 1.0 | -4.094 | True |
| sft | base | 15 | random1 | False | None | -1.9275 | 7.4146 | 0.671 | 1.0 | -4.094 | True |
| sft | base | 15 | random2 | False | None | -2.8399 | 7.4146 | 0.683 | 1.0 | -4.094 | True |
| sft | base | 15 | random3 | False | None | -0.0624 | 7.4146 | 1.137 | 1.0 | -4.094 | True |
| sft | base | 15 | random4 | False | None | -1.7245 | 7.4146 | 0.863 | 1.0 | -4.094 | True |
| sft | base | 15 | shuffled0 | True | 7.4146 | 1.6665 | 7.4146 | 2.885 | 1.0 | -4.094 | True |
| sft | base | 15 | shuffled1 | False | None | -2.6264 | 7.4146 | 1.156 | 1.0 | -4.094 | True |
| sft | base | 15 | shuffled2 | False | None | -4.417 | 1.0742 | 0.018 | 1.0 | -4.094 | True |
| sft | base | 15 | shuffled3 | False | None | -4.4386 | 1.0742 | 0.017 | 1.0 | -4.094 | True |
| sft | base | 15 | shuffled4 | False | None | -3.5942 | 4.3724 | 0.298 | 1.0 | -4.094 | True |
| sft | sft | 20 | direction | True | 14.8293 | 0.3965 | 14.8293 | 2.381 | 1.0 | -4.094 | True | 4.4904 | 0.3245 | 0.8064 | 5.166 | 5 | 0 | 0.6162 | 1.521 | 2.547 | 5 | 0 |
| sft | sft | 20 | random0 | False | None | -4.0906 | 1.0742 | 0.001 | 1.0 | -4.094 | True |
| sft | sft | 20 | random1 | False | None | -4.154 | 1.0742 | 0.001 | 1.0 | -4.094 | True |
| sft | sft | 20 | random2 | False | None | -4.1195 | 1.0742 | 0.001 | 1.0 | -4.094 | True |
| sft | sft | 20 | random3 | False | None | -4.1557 | 1.0742 | 0.001 | 1.0 | -4.094 | True |
| sft | sft | 20 | random4 | False | None | -2.3277 | 29.6586 | 2.114 | 1.0 | -4.094 | True |
| sft | sft | 20 | shuffled0 | False | None | -0.8432 | 14.8293 | 1.241 | 1.0 | -4.094 | True |
| sft | sft | 20 | shuffled1 | False | None | -4.6122 | 1.0742 | 0.008 | 1.0 | -4.094 | True |
| sft | sft | 20 | shuffled2 | False | None | -3.6013 | 7.4146 | 0.426 | 1.0 | -4.094 | True |
| sft | sft | 20 | shuffled3 | False | None | -4.3433 | 1.0742 | 0.006 | 1.0 | -4.094 | True |
| sft | sft | 20 | shuffled4 | False | None | -3.9889 | 3.7073 | 0.059 | 1.0 | -4.094 | True |
| sft | dpo | 17 | direction | True | 3.7073 | 0.7434 | 7.4146 | 2.689 | 1.0 | -4.094 | True | 4.8373 | 1.2111 | 1.0556 | 3.435 | 5 | 0 | 0.9783 | 1.1722 | 3.292 | 5 | 0 |
| sft | dpo | 17 | random0 | False | None | -2.6976 | 14.8293 | 1.894 | 1.0 | -4.094 | True |
| sft | dpo | 17 | random1 | False | None | -3.4452 | 7.4146 | 0.197 | 1.0 | -4.094 | True |
| sft | dpo | 17 | random2 | False | None | -1.4041 | 14.8293 | 1.486 | 1.0 | -4.094 | True |
| sft | dpo | 17 | random3 | False | None | -4.2438 | 1.0742 | 0.003 | 1.0 | -4.094 | True |
| sft | dpo | 17 | random4 | False | None | -2.6235 | 14.8293 | 1.786 | 1.0 | -4.094 | True |
| sft | dpo | 17 | shuffled0 | False | None | -2.4576 | 14.8293 | 2.299 | 1.0 | -4.094 | True |
| sft | dpo | 17 | shuffled1 | False | None | -5.0539 | 1.0742 | 0.034 | 1.0 | -4.094 | True |
| sft | dpo | 17 | shuffled2 | False | None | -2.6393 | 4.3724 | 0.671 | 1.0 | -4.094 | True |
| sft | dpo | 17 | shuffled3 | False | None | -2.0988 | 7.4146 | 1.119 | 1.0 | -4.094 | True |
| sft | dpo | 17 | shuffled4 | False | None | -3.3286 | 7.4146 | 0.646 | 1.0 | -4.094 | True |
| dpo | base | 15 | direction | False | None | -2.8617 | 7.4146 | 2.658 | 1.0 | -8.8016 | True | 5.9399 | 4.3587 | 2.6965 | 0.586 | 5 | 0 | 3.8472 | 2.4812 | 0.843 | 5 | 0 |
| dpo | base | 15 | random0 | False | None | -7.6295 | 7.4146 | 1.611 | 1.0 | -8.8016 | True |
| dpo | base | 15 | random1 | False | None | -4.8357 | 7.4146 | 1.749 | 1.0 | -8.8016 | True |
| dpo | base | 15 | random2 | False | None | -6.0242 | 4.3724 | 0.523 | 1.0 | -8.8016 | True |
| dpo | base | 15 | random3 | False | None | -0.646 | 7.4146 | 2.91 | 1.0 | -8.8016 | True |
| dpo | base | 15 | random4 | False | None | -3.0791 | 7.4146 | 1.966 | 1.0 | -8.8016 | True |
| dpo | base | 15 | shuffled0 | False | None | -0.9155 | 7.4146 | 3.634 | 1.0 | -8.8016 | True |
| dpo | base | 15 | shuffled1 | False | None | -4.4811 | 7.4146 | 2.294 | 1.0 | -8.8016 | True |
| dpo | base | 15 | shuffled2 | False | None | -6.4254 | 7.4146 | 1.983 | 1.0 | -8.8016 | True |
| dpo | base | 15 | shuffled3 | False | None | -7.2811 | 7.4146 | 2.556 | 1.0 | -8.8016 | True |
| dpo | base | 15 | shuffled4 | False | None | -5.6688 | 7.4146 | 1.787 | 1.0 | -8.8016 | True |
| dpo | sft | 20 | direction | True | 14.8293 | 0.0178 | 14.8293 | 3.736 | 1.0 | -8.8016 | True | 8.8194 | 0.7217 | 1.5347 | 5.276 | 5 | 0 | 2.0674 | 3.1534 | 2.141 | 5 | 0 |
| dpo | sft | 20 | random0 | False | None | -8.9024 | 1.0742 | 0.004 | 1.0 | -8.8016 | True |
| dpo | sft | 20 | random1 | False | None | -8.5093 | 14.8293 | 1.145 | 1.0 | -8.8016 | True |
| dpo | sft | 20 | random2 | False | None | -8.6455 | 14.8293 | 1.275 | 1.0 | -8.8016 | True |
| dpo | sft | 20 | random3 | False | None | -8.986 | 1.0742 | 0.003 | 1.0 | -8.8016 | True |
| dpo | sft | 20 | random4 | False | None | -5.356 | 14.8293 | 1.232 | 1.0 | -8.8016 | True |
| dpo | sft | 20 | shuffled0 | False | None | -1.6761 | 14.8293 | 2.561 | 1.0 | -8.8016 | True |
| dpo | sft | 20 | shuffled1 | False | None | -9.7776 | 1.0742 | 0.019 | 1.0 | -8.8016 | True |
| dpo | sft | 20 | shuffled2 | False | None | -6.7554 | 14.8293 | 4.24 | 1.0 | -8.8016 | True |
| dpo | sft | 20 | shuffled3 | False | None | -8.9436 | 14.8293 | 3.07 | 1.0 | -8.8016 | True |
| dpo | sft | 20 | shuffled4 | False | None | -6.5184 | 14.8293 | 2.213 | 1.0 | -8.8016 | True |
| dpo | dpo | 17 | direction | True | 4.3724 | 0.9098 | 7.4146 | 4.544 | 1.0 | -8.8016 | True | 9.7113 | 3.2844 | 2.1191 | 3.033 | 5 | 0 | 2.5555 | 2.3134 | 3.093 | 5 | 0 |
| dpo | dpo | 17 | random0 | False | None | -6.149 | 7.4146 | 0.698 | 1.0 | -8.8016 | True |
| dpo | dpo | 17 | random1 | False | None | -6.896 | 14.8293 | 3.989 | 1.0 | -8.8016 | True |
| dpo | dpo | 17 | random2 | False | None | -2.2762 | 14.8293 | 2.709 | 1.0 | -8.8016 | True |
| dpo | dpo | 17 | random3 | False | None | -7.6157 | 14.8293 | 3.37 | 1.0 | -8.8016 | True |
| dpo | dpo | 17 | random4 | False | None | -4.6489 | 14.8293 | 3.537 | 1.0 | -8.8016 | True |
| dpo | dpo | 17 | shuffled0 | False | None | -5.9323 | 14.8293 | 4.183 | 1.0 | -8.8016 | True |
| dpo | dpo | 17 | shuffled1 | False | None | -10.1221 | 1.0742 | 0.058 | 1.0 | -8.8016 | True |
| dpo | dpo | 17 | shuffled2 | False | None | -5.4465 | 7.4146 | 4.386 | 1.0 | -8.8016 | True |
| dpo | dpo | 17 | shuffled3 | False | None | -3.9085 | 7.4146 | 2.652 | 1.0 | -8.8016 | True |
| dpo | dpo | 17 | shuffled4 | False | None | -5.8211 | 14.8293 | 3.212 | 1.0 | -8.8016 | True |

The decisive cell is target=base, source=dpo: if DPO's refusal direction induces refusal in base at acceptable KL, base already had the machinery. Sweeping coefficients also retires the 'you only tried coeff=1' objection to E02.

---

## P1-E1d · 2026-09-17T19:10:39+00:00 · OK

> Does base's harmful/harmless probe transfer to a contrast set where the discriminative vocabulary is held constant -- i.e. is it harmfulness or topic?

- **script** `probe_transfer.py` — `probe_transfer.py --lineage zephyr --stage all`
- **code** `dc0366b` on `main` ⚠️ DIRTY WORKING TREE — commit does not identify this code
- **duration** 281.5s
- **env** torch 2.6.0+cu124 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S

| stage | transfer_full_peak | transfer_full_L0 | transfer_matched_peak | transfer_matched_at_layer | transfer_matched_L0 | length_only_full | length_only_matched | n_test_matched |
|---|---|---|---|---|---|---|---|---|
| base | 0.8244 | 0.5556 | 0.8199 | 19 | 0.518 | 0.5222 | 0.4709 | 361 |
| sft | 0.9111 | 0.5556 | 0.8947 | 28 | 0.518 | 0.5222 | 0.4709 | 361 |
| dpo | 0.9222 | 0.5556 | 0.9058 | 16 | 0.518 | 0.5222 | 0.4709 | 361 |

Fitted on Arditi, tested on XSTest. The focus-matched subset holds the discriminative word constant, so a vocabulary probe is at chance there by construction.

---

## P1-E1 · 2026-09-17T19:16:12+00:00 · OK

> Same axis as the aligned model, or a different one?

- **script** `aggregate_probe.py` — `aggregate_probe.py --lineage zephyr`
- **code** `dc0366b` on `main` ⚠️ DIRTY WORKING TREE — commit does not identify this code
- **duration** 0.8s
- **env** torch 2.6.0+cu124 · transformers 5.17.0 · sklearn 1.9.1 · gpu NVIDIA L40S

| stage | logistic_peak | logistic_L0 | mass_mean_peak | length_only | surface_suspect |
|---|---|---|---|---|---|
| base | 0.9886 | 0.5 | 0.9697 | 0.5985 | False |
| sft | 1.0 | 0.5 | 0.9735 | 0.5985 | False |
| dpo | 1.0 | 0.5 | 0.9773 | 0.5985 | False |
| base_vs_sft | 0.9258 | 1 | 0.9377 | 25 |
| base_vs_dpo | 0.9552 | 1 | 0.9633 | 24 |
