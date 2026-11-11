# Task 5.15 — SFT pilot on confirmed finals

**Method: SFT** (planned). Prefix-masked loss on gold `REFUND_SUBMITTED` (+ `<|END_RESPONSE|>`).
5.14’s 12 pairs stay held out. Train/dev use new queries (`t15_*` / `v15_*`).

- stop_reason: `base_dev_has_no_final_errors`
- pinned LR / steps (unused): `2e-5` / max 3
- train rollouts: 8 → **6 confirmed** SFT contexts (exact AgentTune `input_ids`)
- dev rollouts: 4 → **4 confirmed** probe contexts
- base on dev (5.12 stop/parse): **0/4 final errors** → no training, no 5.14 eval

Per the gate: when the base already returns the correct amount on every confirmed
dev post-submit prefix, save diagnostics and stop.

