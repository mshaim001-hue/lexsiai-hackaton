# Task 5.21 — collect post-submit contexts (new 8×2 matrix)

Untrained Tiny Aya. Generation/scoring match task 5.14. No training. Max 16 episodes.

- confirmed_returns: **11**/16
- tool_call_after_submit: **3** (queries: ['c21_6', 'c21_7'])
- n_queries_with_tool_call_after_submit: **2**
- successful_post_submit_contexts: 8
- full_success: 8/16
- verification_all_ok: True

## Per episode

| rollout | confirmed | success | error_type | after | post_capture |
|---|---|---|---|---:|---|
| c21_0-g100 | True | True | none | 0 | True |
| c21_0-g101 | False | False | submit_refund_not_confirmed | 0 | False |
| c21_1-g100 | False | False | invalid:schema_invalid | 0 | False |
| c21_1-g101 | True | True | none | 0 | True |
| c21_2-g100 | False | False | invalid:schema_invalid | 0 | False |
| c21_2-g101 | True | True | none | 0 | True |
| c21_3-g100 | True | True | none | 0 | True |
| c21_3-g101 | True | True | none | 0 | True |
| c21_4-g100 | True | True | none | 0 | True |
| c21_4-g101 | True | True | none | 0 | True |
| c21_5-g100 | False | False | invalid:schema_invalid | 0 | False |
| c21_5-g101 | False | False | submit_refund_not_confirmed | 0 | False |
| c21_6-g100 | True | True | none | 0 | True |
| c21_6-g101 | True | False | final_answer_tool_call_after_submit | 1 | True |
| c21_7-g100 | True | False | final_answer_tool_call_after_submit | 1 | True |
| c21_7-g101 | True | False | final_answer_tool_call_after_submit | 1 | True |

