---
base_model: CohereLabs/tiny-aya-global
base_model_revision: af89d219b53ed9b13b8a4645f9c8028973510324
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:CohereLabs/tiny-aya-global
- lora
- peft
- agenttune
- lexsi-hackathon
- tool-calling
---

# Lexsi Track 3 — refund LoRA `step_4` (task 5.23)

Public release-candidate adapter for **Reliable tool calling** on
`CohereLabs/tiny-aya-global` revision
`af89d219b53ed9b13b8a4645f9c8028973510324`.

This repo contains only LoRA weights (`r=8`, `α=16`, dropout `0.05` on
`q_proj`/`k_proj`/`v_proj`/`o_proj`). Base weights are **not** included;
you need HF access to the gated Tiny Aya Global checkpoint.

## Load

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "CohereLabs/tiny-aya-global"
revision = "af89d219b53ed9b13b8a4645f9c8028973510324"
adapter_id = "mshaim001/lexsi-t3-refund-adapter-5-23-step4"

tok = AutoTokenizer.from_pretrained(base_id, revision=revision)
base = AutoModelForCausalLM.from_pretrained(
    base_id, revision=revision, torch_dtype="float16"
)
model = PeftModel.from_pretrained(base, adapter_id)
model.eval()
```

Same path in the GitHub source of truth:
`artifacts/task-5-23/adapters/step_4`.

## Training (summary)

- Method: weighted-REF SFT on four saved post-submit failure contexts
  (tasks 5.21–5.22): `c21_6-g101`, `c21_7-g100`, `c21_7-g101`, `c22_3-g101`
- Loss: REF tokens ×5, other tokens ×1, prefix masked; one optimizer step
  per example; lr `2e-05`
- Stack: AgentTune `d4db4c1` + local `top_p` patch; TRL / PEFT as in the repo
- **No further training** after this adapter

## Evaluation highlights

| Matrix | Full success | After-submit calls | Invalid |
| --- | --- | --- | --- |
| 5.23 on 5.14 (dev) | **7/12 → 8/12** | 3 → 2 | 0 → 0 |
| 5.24 + 5.25 (independent) | **27/40 → 28/40** | **2 → 1** | **0 → 0** |

On the independent holdout the target error
`final_answer_tool_call_after_submit` appeared **twice** on the base model;
this adapter **fixed one** case (`h25_7-g101`) and left one unchanged
(`h24_7-g100`).

Paired tables and traces:
[`mshaim001/lexsi-t3-refund-traces-5-23-5-25`](https://huggingface.co/datasets/mshaim001/lexsi-t3-refund-traces-5-23-5-25)
(after publish) and
[GitHub artifacts](https://github.com/mshaim001-hue/lexsiai-hackaton/tree/main/artifacts).

## Intended use / limits

Research / hackathon reproduction of a one-action-per-turn refund agent.
Not a production customer-support system. Gated base model license and
acceptable-use policy still apply.

## Collection

Add to
[Lexsi Hackathon — Refund Model Adapters](https://huggingface.co/collections/mshaim001/lexsi-hackathon-refund-model-adapters-6af3073f5bd1f23d4e0a82fe).
