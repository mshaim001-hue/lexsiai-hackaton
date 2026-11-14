# Lexsi Hackathon — Track 3 Reliable tool calling

Refund agent on **CohereLabs/tiny-aya-global**
(`af89d219b53ed9b13b8a4645f9c8028973510324`) with AgentTune
`d4db4c12d52342227c4113a70183d0f64aa2f345` and a local top_p generate patch.
The agent must look up order and customer, calculate a refund, submit it, then
stop with exactly `REFUND_SUBMITTED amount_cents=<integer>`.

Public repo: [mshaim001-hue/lexsiai-hackaton](https://github.com/mshaim001-hue/lexsiai-hackaton).

## Release candidate (frozen `step_4`)

Adapter [`artifacts/task-5-23/adapters/step_4`](artifacts/task-5-23/adapters/step_4/)
on `CohereLabs/tiny-aya-global` @ `af89d219b53ed9b13b8a4645f9c8028973510324`.

| Eval | Full success (base → step_4) | After-submit calls | Invalid |
| --- | --- | --- | --- |
| 5.23 on 5.14 (dev) | **7/12 → 8/12** | 3 → 2 | 0 → 0 |
| 5.24 + 5.25 (independent) | **27/40 → 28/40** | **2 → 1** | **0 → 0** |

On the independent matrix the target error
`final_answer_tool_call_after_submit` hit the base **twice**; the adapter
**fixed one** (`h25_7-g101`) and left one unchanged (`h24_7-g100`). Clean-checkout
load + those two control replays:
[artifacts/release-candidate/](artifacts/release-candidate/). Submission pack:
[submission/RELEASE_CANDIDATE.md](submission/RELEASE_CANDIDATE.md).

## Earlier main table (realistic matrix 5.8 → pilot 5.9)

Fixed test: paraphrased refund requests **v0–v3 × seeds 100–103** (16 pairs),
**without** tool-sequence coaching. Generation: `T=0.7`, `top_p=0.9`,
`max_new_tokens=512`, one action per turn. The same matrix was used to choose
the target failure `final_answer_tool_call_after_submit`; the one-step B-only
pilot on train paraphrases **u0–u3** is therefore an observation on that matrix,
not an independent held-out claim. **No metric gain.**

| Metric | 5.8 baseline (untrained) | 5.9 after 1× reward_B on u1 |
| --- | --- | --- |
| Full success | **12/16** | **12/16** |
| Confirmed returns | **14/16** | **14/16** |
| Correct finals | **12/16** | **12/16** |
| Calls after confirmed submit | **2** | **2** |
| Target-failure episodes | **2** (`v0-g102`, `v2-g102`) | **2** (same pairs) |
| Invalid calls | **0/64** | **0** (numerator) |
| Pilot criterion met | — | **no** |

Sources: [artifacts/task-5-8/baseline/aggregates.json](artifacts/task-5-8/baseline/aggregates.json),
[artifacts/task-5-9/comparison.json](artifacts/task-5-9/comparison.json).

**Follow-up 5.10 (not a second main table).** Same u1 group and reward_B, learning
rate raised to `1e-5`, one optimizer step from the 5.9 LoRA-init recipe. On the
saved post-submit train contexts, mean token log-probability of the correct
final versus a repeated tool call did **not** move toward the correct final
(gap −0.038851 → −0.038865), so the full 5.8 test was **not** run. Details:
[artifacts/task-5-10/logprob_probe.json](artifacts/task-5-10/logprob_probe.json),
[artifacts/task-5-10/summary.json](artifacts/task-5-10/summary.json).

## Separate reference: coached matrix 5.5

Earlier held-out **h0–h3 × 100–103** used prompts that spelled the tool order.
Do **not** mix with the 5.8/5.9 table.

| Condition | Full success | Invalid calls | Mean calls |
| --- | --- | --- | --- |
| A untrained | 15/16 | 0/64 | 4.0 |
| B one GRPO step | 15/16 | 0/64 | 4.0 |
| C one GRPO step | 15/16 | 0/64 | 4.0 |

Residual on that matrix: `h0-g103` — submit confirmed, then another
`lookup_order` instead of the final line. B and C adapters matched byte-for-byte
after one step (invalid-call penalty unused in the train group).
Source: [artifacts/task-5-5/summary.json](artifacts/task-5-5/summary.json).

## Environment

```bash
git clone https://github.com/mshaim001-hue/lexsiai-hackaton.git
cd lexsiai-hackaton
python3.12 -m venv .venv
.venv/bin/python -m pip install 'torch==2.11.0' 'transformers==5.22.1' 'trl==1.7.1' \
  'peft==0.21.2' 'accelerate==1.15.0' 'datasets==5.1.0' 'safetensors==0.8.0' requests
git clone https://github.com/Lexsi-Labs/AgentTune.git vendor/AgentTune
git -C vendor/AgentTune checkout d4db4c12d52342227c4113a70183d0f64aa2f345
git -C vendor/AgentTune apply --check ../../patches/agenttune-d4db4c1-top_p.patch
git -C vendor/AgentTune apply ../../patches/agenttune-d4db4c1-top_p.patch
.venv/bin/python -m pip install --no-deps -e vendor/AgentTune
.venv/bin/hf auth login   # Read token; do not commit it
```

After the model is cached, prefer `HF_HUB_OFFLINE=1`. MPS vs CPU can change
samples under the same seeds.

## Reproduce reported runs

Do not overwrite finished `artifacts/` directories. Recorded commands:

```bash
# 5.5 — paired B/C one-step pilots + coached held-out eval
TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 .venv/bin/python run_grpo_pair.py

# 5.8 — pin realistic split, untrained baseline on v0–v3 × 100–103
TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 .venv/bin/python run_realistic_baseline.py

# 5.9 — scan u0→u3, at most one reward_B step on first eligible group, eval on 5.8 test
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
  .venv/bin/python run_grpo_realistic_pilot.py

# 5.10 — u1 at lr=1e-5; full 5.8 eval only if post-submit logprob gap improves
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
  .venv/bin/python run_grpo_lr_probe.py
```

### Configs, traces, adapters

| Run | Config / pin | Traces & metrics | Adapter |
| --- | --- | --- | --- |
| 5.5 | [commands.txt](artifacts/task-5-5/commands.txt), [summary.json](artifacts/task-5-5/summary.json) | [eval_B](artifacts/task-5-5/eval_B/), [eval_C](artifacts/task-5-5/eval_C/), [train_B.json](artifacts/task-5-5/train_B.json) | [adapter_B](artifacts/task-5-5/adapter_B/), [adapter_C](artifacts/task-5-5/adapter_C/) |
| 5.8 | [pinned_config.json](artifacts/task-5-8/pinned_config.json), [datasets/](artifacts/task-5-8/datasets/) | [baseline/](artifacts/task-5-8/baseline/) (`traces.json`, `per_pair.json`, `summary.json`) | untrained |
| 5.9 | [pilot_config.json](artifacts/task-5-9/pilot_config.json) | [group_u1.json](artifacts/task-5-9/group_u1.json), [eval_B/](artifacts/task-5-9/eval_B/), [comparison.json](artifacts/task-5-9/comparison.json) | [adapter_B](artifacts/task-5-9/adapter_B/) |
| 5.10 | [pilot_config.json](artifacts/task-5-10/pilot_config.json) | [logprob_probe.json](artifacts/task-5-10/logprob_probe.json), [train_u1.json](artifacts/task-5-10/train_u1.json) (no full eval) | [adapter_B](artifacts/task-5-10/adapter_B/) |

Submission draft (annotation + remaining HF/form items):
[submission/](submission/).

## Earlier contract tooling

```bash
.venv/bin/python run_contract.py --out artifacts/<new-dir> --max-new-tokens 512
.venv/bin/python run_grpo_prep.py
.venv/bin/python run_grpo_path.py
.venv/bin/python run_grpo_context.py
```

Pinned AgentTune does not pass `top_p` into `model.generate` until
`patches/agenttune-d4db4c1-top_p.patch` is applied (parser unchanged).
