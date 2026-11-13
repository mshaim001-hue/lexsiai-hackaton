# Task 5.25 — expanded independent holdout eval (base vs frozen step_4)

No training. 12 new queries × seeds `{100,101}` (24 episodes each). Same generation/scoring as 5.24. Matrix pinned before any rollout; no query selection by base results.

- adapter: `artifacts/task-5-23/adapters/step_4`
- transfer_criterion (5.25): `{'at_least_one_fail_to_success': True, 'no_success_to_fail': True, 'invalid_calls_do_not_increase': True, 'passed': True}`
- base tool-call-after-submit count: **1**
- matrix exercises target failure: `True`

## Aggregates 5.25 (base → step_4)

| metric | base | step_4 |
|---|---:|---:|
| full_success | 16/24 | 17/24 |
| confirmed_returns | 17 | 17 |
| correct_finals | 16 | 17 |
| invalid_calls_total | 0 | 0 |
| calls_after_confirmed_submit_total | 1 | 0 |
| tool_call_after_submit_count | 1 | 0 |

- fail→success: `[{'rollout_id': 'h25_7-g101', 'base_error_type': 'final_answer_tool_call_after_submit', 'base_failure_stage': 'post_submit_final', 'base_n_invalid': 0, 'adapter_n_invalid': 0}]`
- success→fail: `[]`

## Base after-submit cases (adapter outcome)

| rollout | base after | base err | base stage | step_4 ok | ad after | ad err | ad stage | transition |
|---|---:|---|---|---|---:|---|---|---|
| h25_7-g101 | 1 | final_answer_tool_call_after_submit | post_submit_final | True | 0 | none | none | fail_to_success |

## Paired outcomes 5.25

| rollout | base ok | step_4 ok | base conf | ad conf | base final | ad final | base inv | ad inv | base after | ad after | base err | ad err | base stage | ad stage | transition |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---|---|---|
| h25_0-g100 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | none | none | unchanged_success |
| h25_0-g101 | False | False | False | False | False | False | 0 | 0 | 0 | 0 | submit_refund_not_confirmed | submit_refund_not_confirmed | submit | submit | unchanged_fail |
| h25_1-g100 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | none | none | unchanged_success |
| h25_1-g101 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | none | none | unchanged_success |
| h25_10-g100 | False | False | False | False | False | False | 0 | 0 | 0 | 0 | submit_refund_not_confirmed | submit_refund_not_confirmed | submit | submit | unchanged_fail |
| h25_10-g101 | False | False | False | False | False | False | 0 | 0 | 0 | 0 | submit_refund_not_confirmed | submit_refund_not_confirmed | submit | submit | unchanged_fail |
| h25_11-g100 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | none | none | unchanged_success |
| h25_11-g101 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | none | none | unchanged_success |
| h25_2-g100 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | none | none | unchanged_success |
| h25_2-g101 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | none | none | unchanged_success |
| h25_3-g100 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | none | none | unchanged_success |
| h25_3-g101 | False | False | False | False | False | False | 0 | 0 | 0 | 0 | submit_refund_not_confirmed | submit_refund_not_confirmed | submit | submit | unchanged_fail |
| h25_4-g100 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | none | none | unchanged_success |
| h25_4-g101 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | none | none | unchanged_success |
| h25_5-g100 | False | False | False | False | False | False | 0 | 0 | 0 | 0 | submit_refund_not_confirmed | submit_refund_not_confirmed | submit | submit | unchanged_fail |
| h25_5-g101 | False | False | False | False | False | False | 0 | 0 | 0 | 0 | submit_refund_not_confirmed | submit_refund_not_confirmed | submit | submit | unchanged_fail |
| h25_6-g100 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | none | none | unchanged_success |
| h25_6-g101 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | none | none | unchanged_success |
| h25_7-g100 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | none | none | unchanged_success |
| h25_7-g101 | False | True | True | True | False | True | 0 | 0 | 1 | 0 | final_answer_tool_call_after_submit | none | post_submit_final | none | fail_to_success |
| h25_8-g100 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | none | none | unchanged_success |
| h25_8-g101 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | none | none | unchanged_success |
| h25_9-g100 | True | True | True | True | True | True | 0 | 0 | 0 | 0 | none | none | none | none | unchanged_success |
| h25_9-g101 | False | False | False | False | False | False | 0 | 0 | 0 | 0 | submit_refund_not_confirmed | submit_refund_not_confirmed | submit | submit | unchanged_fail |

## Combined 5.24 + 5.25

- episodes/model: 40
- transfer_criterion (combined): `{'at_least_one_fail_to_success': True, 'no_success_to_fail': True, 'invalid_calls_do_not_increase': True, 'passed': True}`

| metric | base | step_4 |
|---|---:|---:|
| full_success | 27/40 | 28/40 |
| confirmed_returns | 29 | 29 |
| correct_finals | 27 | 28 |
| invalid_calls_total | 0 | 0 |
| calls_after_confirmed_submit_total | 2 | 1 |
| tool_call_after_submit_count | 2 | 1 |

- fail→success (combined): `['h25_7-g101']`
- success→fail (combined): `[]`

### After-submit cases across 5.24+5.25

| source | rollout | base after | base err | step_4 ok | ad after | ad err | transition |
|---|---|---:|---|---|---:|---|---|
| 5.24 | h24_7-g100 | 1 | final_answer_tool_call_after_submit | False | 1 | final_answer_tool_call_after_submit | unchanged_fail |
| 5.25 | h25_7-g101 | 1 | final_answer_tool_call_after_submit | True | 0 | none | fail_to_success |

