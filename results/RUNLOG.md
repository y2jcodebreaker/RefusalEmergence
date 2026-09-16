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

## P1-E1 · 2026-09-16T19:03:12+00:00 · OK

> Same axis as the aligned model, or a different one?

- **script** `aggregate_probe.py` — `aggregate_probe.py --lineage zephyr`
- **code** `fc93673` on `main` ⚠️ DIRTY WORKING TREE — commit does not identify this code
- **duration** 0.6s
- **env** torch 2.11.0 · transformers 4.57.6 · sklearn 1.8.0 · gpu None

| stage | logistic_peak | logistic_L0 | mass_mean_peak | length_only | surface_suspect |
|---|---|---|---|---|---|
| base | 0.9886 | 0.5 | 0.9697 | 0.5985 | False |
| sft | 1.0 | 0.5 | 0.9735 | 0.5985 | False |
| dpo | 1.0 | 0.5 | 0.9773 | 0.5985 | False |
| base_vs_sft | 0.9258 | 1 | 0.9377 | 25 |
| base_vs_dpo | 0.9552 | 1 | 0.9633 | 24 |

---

## P1-E1 · 2026-09-16T19:03:17+00:00 · OK

> Same axis as the aligned model, or a different one?

- **script** `aggregate_probe.py` — `aggregate_probe.py --lineage olmo2`
- **code** `fc93673` on `main` ⚠️ DIRTY WORKING TREE — commit does not identify this code
- **duration** 0.4s
- **env** torch 2.11.0 · transformers 4.57.6 · sklearn 1.8.0 · gpu None

| stage | logistic_peak | logistic_L0 | mass_mean_peak | length_only | surface_suspect |
|---|---|---|---|---|---|
| base | 0.9962 | 0.5 | 0.9205 | 0.5568 | False |
| sft | 1.0 | 0.5 | 1.0 | 0.5758 | False |
| dpo | 1.0 | 0.5 | 1.0 | 0.5758 | False |
| rlvr | 1.0 | 0.5 | 1.0 | 0.5758 | False |
| base_vs_sft | SKIPPED_REGIME_OVERRIDE | 2 | 5 |
| base_vs_dpo | SKIPPED_REGIME_OVERRIDE | 2 | 5 |
| base_vs_rlvr | SKIPPED_REGIME_OVERRIDE | 2 | 5 |
