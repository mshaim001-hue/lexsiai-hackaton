# Annotation draft (≤500 words)

Track 3 — Reliable tool calling. We teach CohereLabs/tiny-aya-global
(revision af89d219…) to finish a multi-step refund in AgentTune
(d4db4c1 + local top_p patch) and emit
`REFUND_SUBMITTED amount_cents=<n>` only after a confirmed submit_refund.

**Protocol.** One JSON tool call per turn. Hidden tokens bind
lookup_order → get_customer → calculate_refund → submit_refund. Full success
needs harness-confirmed submit and that exact final line. A tool call that
replaces the final after submit already scores as failure.

**Release candidate.** LoRA adapter
`artifacts/task-5-23/adapters/step_4` (r=8, α=16, dropout=0.05) on the pinned
Tiny Aya Global revision. Training was weighted-REF SFT on four saved
post-submit failure contexts (tasks 5.21–5.22): REF tokens ×5, other tokens ×1,
prefix masked. No further training after step_4.

**Development matrix (task 5.23 on 5.14).** Full success **7/12 → 8/12**,
confirmed returns 10→10, invalid calls 0→0, calls after confirmed submit 3→2.
One prior fail (`d1-g101`) became full success; no prior success broke.

**Independent holdout (tasks 5.24 + 5.25).** Twelve-plus-eight new realistic
paraphrases with new order IDs, seeds {100,101}, same generation/scoring; no
query selection by base outcomes. Combined: full success **27/40 → 28/40**,
calls after confirmed submit **2 → 1**, invalid calls **0 → 0**. The target
error `final_answer_tool_call_after_submit` appeared **twice** on the base; the
adapter **fixed one** (`h25_7-g101`) and left one unchanged (`h24_7-g100`).

**Clean-checkout check.** In a fresh worktree the adapter loads over the pinned
base revision. Replaying those two control episodes reproduced the same
fail→success and unchanged-fail outcomes
(`artifacts/release-candidate/`).

**Artifacts.** Commands, pinned configs, paired tables, and traces live under
`artifacts/task-5-{23,24,25}/` and `artifacts/release-candidate/`. Submission
notes: `submission/RELEASE_CANDIDATE.md`. Hugging Face public collections for
the adapter, data, and traces remain for the form if required.

<!-- Body under 500 words. -->
