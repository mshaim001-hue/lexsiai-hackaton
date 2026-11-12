# Task 5.22 — continue post-submit train collection (4×2)

Untrained Tiny Aya. Same generation/scoring/capture format as 5.21. No training.

## 5.22 only

- confirmed_returns: **7**/8
- tool_call_after_submit: **1** (`['c22_3-g101']`)
- queries with that failure: **1** (`['c22_3']`)
- successful_post_submit_contexts: 6
- verification_all_ok: True

## Combined 5.21 + 5.22 (unique queries with tool-call-after-submit)

- unique queries: **3** (`['c21_6', 'c21_7', 'c22_3']`)
- total tool_call_after_submit episodes: **4**
- new fail IDs in 5.22: `['c22_3-g101']`

## Per episode

| rollout | confirmed | success | error_type | after | post_capture |
|---|---|---|---|---:|---|
| c22_0-g100 | True | True | none | 0 | True |
| c22_0-g101 | True | True | none | 0 | True |
| c22_1-g100 | False | False | invalid:schema_invalid | 0 | False |
| c22_1-g101 | True | True | none | 0 | True |
| c22_2-g100 | True | True | none | 0 | True |
| c22_2-g101 | True | True | none | 0 | True |
| c22_3-g100 | True | True | none | 0 | True |
| c22_3-g101 | True | False | final_answer_tool_call_after_submit | 1 | True |

