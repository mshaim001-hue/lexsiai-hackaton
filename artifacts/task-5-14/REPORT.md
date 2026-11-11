# Task 5.14 — discovery matrix (untrained Tiny Aya)

6 new realistic refund queries × seeds `{100, 101}` (12 episodes).
Generation settings match task-5-8 one-action. No adapter.

- full_success: 7/12
- confirmed_returns: 10
- unconfirmed_returns: 2
- final_failure_after_confirm: 3
- queries with that failure: ['d1', 'd2', 'd4']
- keep as eval set: True

Keep d0–d5 × {100,101} as held-out eval; collect train examples on a separate prompt set.

| rollout | confirmed | correct_final | success | error_type | post_submit_ids |
|---|---|---|---|---|---:|
| d0-g100 | True | True | True | none | 2510 |
| d0-g101 | True | True | True | none | 2510 |
| d1-g100 | True | True | True | none | 2525 |
| d1-g101 | True | False | False | final_answer_tool_call_after_submit | 2537 |
| d2-g100 | True | True | True | none | 2518 |
| d2-g101 | True | False | False | final_answer_tool_call_after_submit | 2530 |
| d3-g100 | True | True | True | none | 2504 |
| d3-g101 | True | True | True | none | 2504 |
| d4-g100 | True | False | False | final_answer_tool_call_after_submit | 2569 |
| d4-g101 | False | False | False | submit_refund_not_confirmed | 2501 |
| d5-g100 | False | False | False | submit_refund_not_confirmed | 2518 |
| d5-g101 | True | True | True | none | 2505 |
