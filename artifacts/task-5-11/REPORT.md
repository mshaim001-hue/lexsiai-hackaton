# Task 5.11 — SFT on gold final answers

**Method: SFT (supervised fine-tuning).** Not GRPO. The adapter is trained
on gold `REFUND_SUBMITTED amount_cents=<n>` lines taken from harness
`submit_refund` amounts, with the entire dialogue prefix masked out of the loss.

- stop_reason: `probe_failed_skip_full_eval`
- optimizer_steps: 16
- learning_rate: 0.0002
- adapter: `artifacts/task-5-11/adapter_SFT`
- elapsed_s: 4872.3

## Loss mask check

```json
{
  "episode_id": "s0",
  "n_prefix": 2499,
  "n_target": 12,
  "n_unmasked_labels": 12,
  "prefix_fully_masked": true,
  "target_fully_supervised": true,
  "unmasked_equals_target_len": true,
  "target_text": "REFUND_SUBMITTED amount_cents=1110",
  "model_loss": 0.02817457914352417,
  "manual_masked_ce": 0.02817457914352417,
  "losses_agree": true
}
```

## Probe gate (8 held-out post-submit prefixes)

- fixed=0 broken=5 base_correct=5 passed=False

| episode | amount | base_ok | adapter_ok | fixed | broken | base_tool | adapter_tool |
|---|---:|---|---|---|---|---|---|
| p0 | 1310 | False | False | False | False | False | False |
| p1 | 2420 | True | False | False | True | False | False |
| p2 | 3530 | True | False | False | True | False | False |
| p3 | 1640 | False | False | False | False | False | False |
| p4 | 2750 | False | False | False | False | False | False |
| p5 | 1860 | True | False | False | True | False | False |
| p6 | 2970 | True | False | False | True | False | False |
| p7 | 1080 | True | False | False | True | False | False |

## Full episode eval

Skipped (probe gate failed or time limit).
