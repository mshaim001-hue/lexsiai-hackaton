# Task 5.24 — independent holdout eval (base vs 5.23 step_4)

No training. 8 new queries × seeds `{100,101}` (16 episodes each).

- adapter: `artifacts/task-5-23/adapters/step_4`
- transfer_criterion: `{'at_least_one_fail_to_success': False, 'no_success_to_fail': True, 'invalid_calls_do_not_increase': True, 'passed': False}`
- base tool-call-after-submit count: **1**
- matrix exercises target failure: `True`

## Aggregates (base → step_4)

| metric | base | step_4 |
|---|---:|---:|
| full_success | 11/16 | 11/16 |
| confirmed_returns | 12 | 12 |
| correct_finals | 11 | 11 |
| invalid_calls_total | 0 | 0 |
| calls_after_confirmed_submit_total | 1 | 1 |
| tool_call_after_submit_count | 1 | 1 |

- fail→success: `[]`
- success→fail: `[]`

## Paired outcomes

| rollout | base ok | step_4 ok | base conf | ad conf | base final | ad final | base inv | ad inv | base after | ad after | base err | ad err | transition |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---|
| h24_0-g100 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | unchanged_success |
| h24_0-g101 | False | False | False | False | False | False | 0 | 0 | 0 | 0 | submit_refund_not_confirmed | submit_refund_not_confirmed | unchanged_fail |
| h24_1-g100 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | unchanged_success |
| h24_1-g101 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | unchanged_success |
| h24_2-g100 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | unchanged_success |
| h24_2-g101 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | unchanged_success |
| h24_3-g100 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | unchanged_success |
| h24_3-g101 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | unchanged_success |
| h24_4-g100 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | unchanged_success |
| h24_4-g101 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | unchanged_success |
| h24_5-g100 | False | False | False | False | False | False | 0 | 0 | 0 | 0 | submit_refund_not_confirmed | submit_refund_not_confirmed | unchanged_fail |
| h24_5-g101 | False | False | False | False | False | False | 0 | 0 | 0 | 0 | submit_refund_not_confirmed | submit_refund_not_confirmed | unchanged_fail |
| h24_6-g100 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | unchanged_success |
| h24_6-g101 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | unchanged_success |
| h24_7-g100 | False | False | True | True | False | False | 0 | 0 | 1 | 1 | final_answer_tool_call_after_submit | final_answer_tool_call_after_submit | unchanged_fail |
| h24_7-g101 | False | False | False | False | False | False | 0 | 0 | 0 | 0 | submit_refund_not_confirmed | submit_refund_not_confirmed | unchanged_fail |

