# Task 5.19 — 5.16 SFT pilot with 5× weight on REF

Same train order / fresh LoRA / `lr=2e-5` / ≤3 steps as 5.16. Token-level CE: first final token (`26501` / `REF`) weight **5×**, other final tokens **1×**, prefix masked; loss ÷ sum(weights).

- stop_reason: `selection_criterion_met`
- lr / max steps: `2e-05` / `3`
- train order: `['t15_0-g100', 't15_1-g100', 't15_3-g100', 't15_4-g100', 't15_5-g100', 't15_7-g100']`
- base probe: 8/10 correct; target fails `['d1-g101', 'd2-g101']`

## Per step

| step | loss | fixed | broken | empty | Δlogp d1-g101 | Δlogp d2-g101 | Δlogp d0-g100 | Δlogp d3-g100 | pass |
|---:|---:|---|---|---|---:|---:|---:|---:|---|
| base | — | — | — | — | -1.171875 | -1.328125 | 3.125000 | 2.539062 | — |
| 1 | 0.148074 | [] | [] | [] | -1.125000 | -1.296875 | 3.179687 | 2.578125 | False |
| 2 | 0.145308 | [] | [] | [] | -1.015625 | -1.187500 | 3.328125 | 2.742187 | False |
| 3 | 0.136862 | ['d1-g101'] | [] | [] | -0.890625 | -1.031250 | 3.515625 | 2.929688 | True |

- selected: `{'step': 3, 'mean_train_loss': 0.136862492809693, 'mean_sum_weights': 17.0, 'grad_norm': 164219.046875, 'n_supervised_tokens_per_epoch': 78, 'adapter': 'artifacts/task-5-19/adapters/step_3', 'probe_n_correct': 9, 'probe_n_final_errors': 1, 'probe_n_empty': 0, 'fixed_ids': ['d1-g101'], 'broken_ids': [], 'empty_ids': [], 'keeps_all_base_correct': True, 'fixes_at_least_one_target_fail': True, 'passes_selection': True, 'delta_logp_ref_minus_bracket': {'d1-g101': -0.8906250596046448, 'd2-g101': -1.0312499403953552, 'd0-g100': 3.515625, 'd3-g100': 2.929687589406967}, 'delta_logp_on_fails': {'d1-g101': -0.8906250596046448, 'd2-g101': -1.0312499403953552}}`
- full 5.14 episode eval: skipped (per 5.19 brief)

