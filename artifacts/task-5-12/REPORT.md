# Task 5.12 — SFT probe stop/parse diagnosis

Aligns final-turn probe stopping and parsing with AgentTune rollout behavior,
then diagnoses the 5.11 SFT adapter on saved answers and a few regenerations.

## Stop / parse alignment

- `eos_token_id`: model `generation_config.eos_token_id` (AgentTune), not `tokenizer.eos_token_id`.
- Parse: content before the first `<|END_RESPONSE|>`; leading marker ⇒ empty final.

## Saved 5.11 answers under aligned parse

- old base correct: 5/8
- new base correct: 8/8
- rescued by truncate-at-marker: 3
- old adapter correct: 0/8
- new adapter correct: 0/8
- adapter leading `<|END_RESPONSE|>` (empty): 8

## Training target vs `<|END_RESPONSE|>`

- marker in any target: False
- marker under loss: False
- note: 5.11 SFT supervised only the gold REFUND_SUBMITTED line after add_generation_prompt; chat-template <|END_RESPONSE|> was not appended to the target, so the marker is absent from labels.

## First-token probs on 5.11 probe prefixes

- base prefers END_RESPONSE over gold first: 0/8
- adapter prefers END_RESPONSE over gold first: 8/8
- adapter raised P(END_RESPONSE): 8/8
- adapter lowered P(gold first): 8/8

## Regen on known post-submit tool-call contexts

- contexts: `v0-g102`, `v2-g102` (5.8), `u1-g102`, `u1-g103` (5.10)
- base tool-call reproduced: 0/4
- adapter tool-call reproduced: 0/4
- Separate next train/probe/eval split: **not proposed** (tool-call failure did not reproduce under the aligned final-only regen).
- On these prefixes the 5.11 adapter still tends to emit `<|END_RESPONSE|>` early or truncate the gold line; that matches the first-token logit shift and the missing marker in the 5.11 SFT target.
