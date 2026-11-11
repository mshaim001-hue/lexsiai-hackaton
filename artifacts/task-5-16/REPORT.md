# Task 5.16 — SFT on 5.15 confirmed finals + 5.14 quick check

**Method: SFT.** Fresh LoRA; prefix-masked loss on gold `REFUND_SUBMITTED` (+ <|END_RESPONSE|>).

- stop_reason: `no_checkpoint_passed_selection`
- lr / max steps: `2e-05` / `3`
- train order: `['t15_0-g100', 't15_1-g100', 't15_3-g100', 't15_4-g100', 't15_5-g100', 't15_7-g100']`
- base probe (10 confirmed 5.14 prefixes): 8/10 correct; errors_reproduced=True

## Training

- selection_mode: `errors_reproduced`
- train_stop_reason: `max_steps_reached`
- selected: `None`

- step 1: loss=0.024466 correct=8/10 fixed=[] broken=[] empty=[]
- step 2: loss=0.027116 correct=8/10 fixed=[] broken=[] empty=[]
- step 3: loss=0.000005 correct=8/10 fixed=[] broken=[] empty=[]

No checkpoint fixed `d1-g101` / `d2-g101` within 3 steps (kept all 8 base-correct;
`d4-g100` episode error did not reproduce under isolated regen). Full 5.14 episode
eval skipped per gate (at most one selected checkpoint).

