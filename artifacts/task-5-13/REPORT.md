# Task 5.13 — SFT learning-rate probe

Fresh LoRA from clean Tiny Aya; same 5.11 `s0` target/mask; compare
`lr=2e-4` vs `lr=2e-5` for up to 3 AdamW steps. No full episode eval.

- stop_reason: `collapse_not_reproduced_stop`
- collapse_reproduced: False
- elapsed_s: 1848.9
- init logits vs base: exact match (`max_abs=0`) on `s0` and `p0`
- trainable LoRA dtype: `float32`; gradients finite on every completed step
- note: `lr=2e-5` stopped after 1 step when the 30-minute budget elapsed; `lr=2e-4` finished all 3 steps without END_RESPONSE collapse (P(gold) rose, P(end) stayed ~5e-4)

## Arms

### lr=0.0002

- steps=3 collapse_seen=False dtypes=['torch.float32']

| step | eval_loss | p_gold | p_end | grad_norm | dW_l2 | collapse_train | collapse_probe |
|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 0.027975 | 0.7771 | 0.0005 |  | 0.000000 | False | False |
| 1 | 0.021076 | 0.8394 | 0.0005 | 391460.500000 | 0.242994 | False | False |
| 2 | 0.014295 | 0.9076 | 0.0004 | 384645.468750 | 0.511687 | False | False |
| 3 | 0.010405 | 0.9501 | 0.0004 | 503299.625000 | 0.816013 | False | False |

### lr=2e-05

- steps=1 collapse_seen=False dtypes=['torch.float32']

| step | eval_loss | p_gold | p_end | grad_norm | dW_l2 | collapse_train | collapse_probe |
|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 0.027975 | 0.7771 | 0.0005 |  | 0.000000 | False | False |
| 1 | 0.027818 | 0.7786 | 0.0005 | 391460.500000 | 0.024253 | False | False |

