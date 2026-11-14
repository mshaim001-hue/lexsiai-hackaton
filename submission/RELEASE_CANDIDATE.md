# Release candidate — frozen LoRA `step_4`

**Status:** materials prepared; form re-submission not started.

## Adapter and base

| Item | Value |
| --- | --- |
| Adapter path | [`artifacts/task-5-23/adapters/step_4`](../artifacts/task-5-23/adapters/step_4/) |
| Adapter files | `adapter_config.json`, `adapter_model.safetensors`, `README.md` |
| Base model | `CohereLabs/tiny-aya-global` |
| Base revision | `af89d219b53ed9b13b8a4645f9c8028973510324` |
| Load recipe | `AutoModelForCausalLM.from_pretrained(..., revision=…, local_files_only=True)` then `PeftModel.from_pretrained(base, "artifacts/task-5-23/adapters/step_4")` via `contract.sft_eval_520._load_engine` |
| AgentTune | `d4db4c12d52342227c4113a70183d0f64aa2f345` + `patches/agenttune-d4db4c1-top_p.patch` |

Clean-checkout load + control replay: [`artifacts/release-candidate/`](../artifacts/release-candidate/)
([`load_check.json`](../artifacts/release-candidate/load_check.json),
[`commands.txt`](../artifacts/release-candidate/commands.txt),
[`comparison.json`](../artifacts/release-candidate/comparison.json)).

## Development matrix (task 5.23 on 5.14)

Weighted-REF SFT (four after-submit fails) → eval on the 5.14 discovery matrix.

| Metric | base | step_4 |
| --- | ---: | ---: |
| Full success | **7/12** | **8/12** |
| Confirmed returns | 10 | 10 |
| Correct finals | 7 | 8 |
| Invalid calls | 0 | 0 |
| Calls after confirmed submit | 3 | 2 |

Sources:
[REPORT](../artifacts/task-5-23/REPORT.md),
[comparison](../artifacts/task-5-23/eval_514/comparison.json),
[base traces](../artifacts/task-5-23/eval_514/base/traces.json),
[adapter traces](../artifacts/task-5-23/eval_514/adapter_step_4/traces.json).

## Independent holdout (tasks 5.24 + 5.25)

Same frozen adapter. New paraphrases and order IDs; seeds `{100,101}`; no query selection by base results; no further training.

| Metric | base | step_4 |
| --- | ---: | ---: |
| Full success | **27/40** | **28/40** |
| Calls after confirmed submit | **2** | **1** |
| Invalid calls | **0** | **0** |

**Target error coverage:** on the combined independent matrix the base hit
`final_answer_tool_call_after_submit` **twice**. The adapter **fixed one** of
those two cases (`h25_7-g101`); the other (`h24_7-g100`) remained a post-submit
tool call.

Paired results and traces:

| Split | Paired / comparison | Traces |
| --- | --- | --- |
| 5.24 (16 eps) | [comparison.json](../artifacts/task-5-24/comparison.json), [paired_outcomes.json](../artifacts/task-5-24/paired_outcomes.json), [REPORT](../artifacts/task-5-24/REPORT.md) | [base](../artifacts/task-5-24/base/traces.json), [step_4](../artifacts/task-5-24/step_4/traces.json) |
| 5.25 (24 eps) | [comparison.json](../artifacts/task-5-25/comparison.json), [paired_outcomes.json](../artifacts/task-5-25/paired_outcomes.json), [REPORT](../artifacts/task-5-25/REPORT.md) | [base](../artifacts/task-5-25/base/traces.json), [step_4](../artifacts/task-5-25/step_4/traces.json) |
| Combined | [combined_with_524.json](../artifacts/task-5-25/combined_with_524.json) | — |

## Control replay (release candidate)

Same settings as 5.24/5.25. Clean worktree load of the adapter over the pinned
base revision, then full episodes:

| Rollout | Prior note | base | step_4 |
| --- | --- | --- | --- |
| `h25_7-g101` | adapter fixed holdout fail | fail (`after=1`) | **success** |
| `h24_7-g100` | failure persisted on holdout | fail (`after=1`) | fail (`after=1`) |

Details: [`artifacts/release-candidate/REPORT.md`](../artifacts/release-candidate/REPORT.md).

## Reproduce control verify

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
  .venv/bin/python run_release_candidate_verify.py
```
