# Release candidate verify — frozen step_4

- adapter: `artifacts/task-5-23/adapters/step_4`
- base model: `CohereLabs/tiny-aya-global` @ `af89d219b53ed9b13b8a4645f9c8028973510324`
- git_head: `a5a3094ce2044144f2a9949bdc392c111e8b8d54`
- clean checkout: git worktree at /tmp/lexsi-rc-verify detached HEAD a5a3094; .venv and vendor symlinked from main checkout; RC verify scripts copied in (not yet on that commit); adapter loaded from artifacts/task-5-23/adapters/step_4 over CohereLabs/tiny-aya-global@af89d219… per README/RESUME
- load ok: `True` (adapter base `CohereLabs/tiny-aya-global`)

## Control episodes

| rollout | base ok | step_4 ok | base after | ad after | base err | ad err | transition | matches prior |
|---|---|---|---:|---:|---|---|---|---|
| h25_7-g101 | False | True | 1 | 0 | final_answer_tool_call_after_submit | none | fail_to_success | True |
| h24_7-g100 | False | False | 1 | 1 | final_answer_tool_call_after_submit | final_answer_tool_call_after_submit | unchanged_fail | True |

