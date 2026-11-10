# Annotation draft (≤500 words)

Track 3 — Reliable tool calling. We teach CohereLabs/tiny-aya-global
(revision af89d219…) to complete a multi-step refund through AgentTune
(d4db4c1 + local top_p patch) and to emit the final confirmation line only
after a successful submit_refund.

**Protocol.** One JSON tool call per turn. Hidden per-episode tokens bind
lookup_order → get_customer → calculate_refund → submit_refund. Full success
requires harness-confirmed submit and the exact final
`REFUND_SUBMITTED amount_cents=<n>`. Reward B is the indicator of full
success, so a post-submit tool call that replaces the final line already
scores 0.

**Coached reference (task 5.5).** On held-out h0–h3 × seeds 100–103 with
prompts that named the tool order, untrained A and one-step B/C all reached
15/16 full success and 0/64 invalid calls. The single residual was
h0-g103: submit confirmed, then another lookup_order instead of the final
line. B and C adapters were byte-identical after one step because the train
group never triggered the invalid-call penalty.

**Realistic matrix (task 5.8).** We pinned a separate train/test split of
paraphrased customer messages with extra detail and without sequence
coaching (u0–u3 train, v0–v3 test; seeds 100–103; T=0.7, top_p=0.9).
Untrained Tiny Aya scored 12/16 full success, 14/16 confirmed returns,
12/16 correct finals, 0/64 invalid calls. Failures split between
final_answer_tool_call_after_submit (v0-g102, v2-g102) and
submit_refund_not_confirmed. We selected the post-submit tool-call mode for
the pilot because it matches the Track 3 claim and the 5.5 residual.

**Limited pilot (task 5.9).** Groups were scanned u0→u3 under reward_B with
model.eval() during rollout. u0 had identical zero rewards. u1 was the
first eligible mixed group (two full successes, two target failures,
differing rewards). We took one optimizer step on that live group, saved
the LoRA adapter immediately, and evaluated all 16 v0–v3 pairs under the
same 5.8 settings.

**Observation.** On that selection matrix, full success stayed 12/16,
confirmed returns 14/16, and the same two pairs still ended with a
tool-call final after confirm. The pilot success criterion (fewer
post-submit calls, more full successes, at least as many confirmed
returns) was not met. This is an observation on the matrix used to choose
the failure, not an independent held-out win.

**Artifacts.** Public GitHub commands, pinned configs, episode traces, and
the 5.9 adapter_B are in-repo under artifacts/task-5-{5,8,9}/. Hugging Face
collections for the adapter, datasets, and traces remain to be published
for the submission form.

<!-- Body ≈375 words (under 500). -->
