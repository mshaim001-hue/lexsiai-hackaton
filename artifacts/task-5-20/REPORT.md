# Task 5.20 — full 5.14 episode eval of 5.19 `step_3`

No training. Same discovery matrix / prompts / seeds / one-action generation as task 5.14.

- adapter: `artifacts/task-5-19/adapters/step_3`
- comparison_source: `artifacts/task-5-14/baseline/per_pair.json`
- condition match vs 5.14: `True`
- pass_criterion: `{'at_least_one_prev_fail_becomes_full_success': False, 'no_prev_success_breaks': True, 'invalid_calls_do_not_increase': True, 'passed': False}`

## Aggregates (base → step_3)

| metric | base | step_3 |
|---|---:|---:|
| full_success | 7/12 | 7/12 |
| confirmed_returns | 10 | 10 |
| correct_finals | 7 | 7 |
| invalid_calls_total | 0 | 0 |
| calls_after_confirmed_submit_total | 3 | 3 |

- newly fixed (fail→full success): `[]`
- newly broken (success→fail): `[]`

## Focus episodes

| rollout | base success | base err | ad success | ad err | after base→ad |
|---|---|---|---|---|---|
| d1-g101 | False | final_answer_tool_call_after_submit | False | final_answer_tool_call_after_submit | 1→1 |
| d2-g101 | False | final_answer_tool_call_after_submit | False | final_answer_tool_call_after_submit | 1→1 |
| d4-g100 | False | final_answer_tool_call_after_submit | False | final_answer_tool_call_after_submit | 1→1 |

## Per episode

| rollout | base ok | ad ok | base conf | ad conf | base final | ad final | base inv | ad inv | base after | ad after | base err | ad err | transition |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---|
| d0-g100 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | unchanged_success |
| d0-g101 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | unchanged_success |
| d1-g100 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | unchanged_success |
| d1-g101 | False | False | True | True | False | False | 0 | 0 | 1 | 1 | final_answer_tool_call_after_submit | final_answer_tool_call_after_submit | unchanged_fail |
| d2-g100 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | unchanged_success |
| d2-g101 | False | False | True | True | False | False | 0 | 0 | 1 | 1 | final_answer_tool_call_after_submit | final_answer_tool_call_after_submit | unchanged_fail |
| d3-g100 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | unchanged_success |
| d3-g101 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | unchanged_success |
| d4-g100 | False | False | True | True | False | False | 0 | 0 | 1 | 1 | final_answer_tool_call_after_submit | final_answer_tool_call_after_submit | unchanged_fail |
| d4-g101 | False | False | False | False | False | False | 0 | 0 | 0 | 0 | submit_refund_not_confirmed | submit_refund_not_confirmed | unchanged_fail |
| d5-g100 | False | False | False | False | False | False | 0 | 0 | 0 | 0 | submit_refund_not_confirmed | submit_refund_not_confirmed | unchanged_fail |
| d5-g101 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | unchanged_success |

