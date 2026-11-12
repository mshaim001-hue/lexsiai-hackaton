# Task 5.23 — weighted-REF SFT on four after-submit fails + 5.14 eval

- train order: `['c21_6-g101', 'c21_7-g100', 'c21_7-g101', 'c22_3-g101']`
- recipe: lr=2e-05, REF×5.0, other×1.0, one optimizer step per example
- eval adapter (pinned): `artifacts/task-5-23/adapters/step_4`
- pass_criterion: `{'at_least_one_prev_fail_becomes_full_success': True, 'no_prev_success_breaks': True, 'invalid_calls_do_not_increase': True, 'passed': True}`

## Δlogp(REF−[) on train contexts

| rollout | before | after | Δ |
|---|---:|---:|---:|
| c21_6-g101 | -0.718750 | -0.234375 | 0.484375 |
| c21_7-g100 | -1.031250 | -0.562500 | 0.468750 |
| c21_7-g101 | -1.031250 | -0.562500 | 0.468750 |
| c22_3-g101 | -1.484375 | -1.078125 | 0.406250 |

## 5.14 episode eval (base → step_4)

| metric | base | step_4 |
|---|---:|---:|
| full_success | 7/12 | 8/12 |
| confirmed_returns | 10 | 10 |
| correct_finals | 7 | 8 |
| invalid_calls_total | 0 | 0 |
| calls_after_confirmed_submit_total | 3 | 2 |

- newly fixed: `['d1-g101']`
- newly broken: `[]`

