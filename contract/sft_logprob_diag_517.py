"""Task 5.17: logprob gap between gold final and post-submit tool call.

No training, no full-episode generation. Scores mean token logprob of the
correct REFUND_SUBMITTED line vs the (observed or matched) lookup_order call
on the same 5.14 post-submit prefix, for base Tiny Aya and each 5.16 adapter.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contract import config
from contract.grpo_pair import _release
from contract.grpo_prep import _package_versions, _write_json
from contract.harness import final_answer_text
from contract.sft_final import END_RESPONSE

OUT = ROOT / "artifacts" / "task-5-17"
BASELINE_514 = ROOT / "artifacts" / "task-5-14" / "baseline"
ADAPTERS_516 = ROOT / "artifacts" / "task-5-16" / "adapters"
COMMAND = (
    "HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
    ".venv/bin/python run_sft_logprob_diag_517.py"
)

# Failures that 5.16 did not fix under isolated regen.
FAIL_IDS = ("d1-g101", "d2-g101")
# Successful confirmed controls (same matrix, different seeds/queries).
CONTROL_IDS = ("d0-g100", "d3-g100")
ORDER_BY_EPISODE = {
    "d0": "ORD-9401",
    "d1": "ORD-9402",
    "d2": "ORD-9403",
    "d3": "ORD-9404",
}


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _mean_token_logprob(model, prefix_ids: list[int], cont_ids: list[int]) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    if not cont_ids:
        raise ValueError("empty continuation")
    device = next(model.parameters()).device
    full = prefix_ids + cont_ids
    ids = torch.tensor([full], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0]
    # Position i predicts token i+1; score each continuation token.
    n_pref = len(prefix_ids)
    logps = []
    for i, tok in enumerate(cont_ids):
        pos = n_pref - 1 + i
        logp = F.log_softmax(logits[pos].float(), dim=-1)[tok]
        logps.append(float(logp.item()))
    return {
        "n_tokens": len(cont_ids),
        "mean_logprob": sum(logps) / len(logps),
        "sum_logprob": sum(logps),
        "token_logprobs": logps,
    }


def _load_examples(tokenizer) -> list[dict[str, Any]]:
    turns = {
        t["rollout_id"]: t
        for t in json.loads(
            (BASELINE_514 / "post_submit_final_turns.json").read_text(encoding="utf-8")
        )["turns"]
    }
    pairs = {
        p["rollout_id"]: p
        for p in json.loads((BASELINE_514 / "per_pair.json").read_text(encoding="utf-8"))[
            "pairs"
        ]
    }
    examples = []
    for rid, role in [(x, "fail") for x in FAIL_IDS] + [(x, "control") for x in CONTROL_IDS]:
        turn = turns[rid]
        pair = pairs[rid]
        amount = int(turn["amount_cents"])
        prefix = list(turn["input_ids"])
        gold_text = final_answer_text(amount) + END_RESPONSE + "<EOS_TOKEN>"
        gold_ids = tokenizer.encode(gold_text, add_special_tokens=False)
        order_id = ORDER_BY_EPISODE[turn["episode_id"]]
        call_text = (
            f'[{{"tool_name": "lookup_order", "parameters": {{"order_id": "{order_id}"}}}}]'
            f"{END_RESPONSE}<EOS_TOKEN>"
        )
        call_encoded = tokenizer.encode(call_text, add_special_tokens=False)
        saved = list(turn["new_token_ids"])
        if role == "fail":
            if saved != call_encoded:
                raise RuntimeError(f"{rid}: saved call tokens != encoded lookup_order")
            call_ids = saved
            call_source = "saved_new_token_ids"
            gold_source = "encoded_refund_submitted_end_eos"
        else:
            if saved != gold_ids:
                raise RuntimeError(f"{rid}: saved success tokens != encoded gold final")
            gold_ids = saved
            call_ids = call_encoded
            call_source = "encoded_lookup_order_matched_fail_template"
            gold_source = "saved_new_token_ids"
        examples.append(
            {
                "rollout_id": rid,
                "role": role,
                "episode_id": turn["episode_id"],
                "generation_seed": int(turn["generation_seed"]),
                "amount_cents": amount,
                "order_id": order_id,
                "episode_error_type": pair.get("error_type"),
                "episode_task_success": bool(pair.get("task_success")),
                "prefix_input_ids": prefix,
                "n_prefix": len(prefix),
                "gold_token_ids": gold_ids,
                "gold_text": gold_text,
                "gold_source": gold_source,
                "call_token_ids": call_ids,
                "call_text": tokenizer.decode(call_ids, skip_special_tokens=False),
                "call_source": call_source,
            }
        )
    return examples


def _score_model(model, examples: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {}
    for ex in examples:
        gold = _mean_token_logprob(model, ex["prefix_input_ids"], ex["gold_token_ids"])
        call = _mean_token_logprob(model, ex["prefix_input_ids"], ex["call_token_ids"])
        gap = gold["mean_logprob"] - call["mean_logprob"]
        by_id[ex["rollout_id"]] = {
            "rollout_id": ex["rollout_id"],
            "role": ex["role"],
            "mean_logprob_gold": gold["mean_logprob"],
            "mean_logprob_call": call["mean_logprob"],
            "gap_gold_minus_call": gap,
            "n_gold_tokens": gold["n_tokens"],
            "n_call_tokens": call["n_tokens"],
            "sum_logprob_gold": gold["sum_logprob"],
            "sum_logprob_call": call["sum_logprob"],
        }
        print(
            f"  {ex['rollout_id']} ({ex['role']}): "
            f"gold={gold['mean_logprob']:.6f} call={call['mean_logprob']:.6f} "
            f"gap={gap:.6f}",
            flush=True,
        )
    return by_id


def _load_base():
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        revision=config.MODEL_REVISION,
        dtype=torch.float16,
        local_files_only=True,
    )
    model.eval()
    if torch.backends.mps.is_available():
        model.to("mps")
    return model


def _load_adapter(step: int):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        revision=config.MODEL_REVISION,
        dtype=torch.float16,
        local_files_only=True,
    )
    adapted = PeftModel.from_pretrained(base, str(ADAPTERS_516 / f"step_{step}"))
    adapted.eval()
    if torch.backends.mps.is_available():
        adapted.to("mps")
    return adapted


def main() -> int:
    from transformers import AutoTokenizer

    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "commands.txt").write_text(COMMAND + "\n", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID, revision=config.MODEL_REVISION, local_files_only=True
    )
    examples = _load_examples(tokenizer)
    compact_examples = [
        {
            k: v
            for k, v in ex.items()
            if k
            not in {
                "prefix_input_ids",
                "gold_token_ids",
                "call_token_ids",
            }
        }
        | {
            "n_prefix": ex["n_prefix"],
            "n_gold_tokens": len(ex["gold_token_ids"]),
            "n_call_tokens": len(ex["call_token_ids"]),
            "gold_token_ids_head": ex["gold_token_ids"][:8],
            "call_token_ids_head": ex["call_token_ids"][:8],
        }
        for ex in examples
    ]
    pinned = {
        "kind": "sft_logprob_diag_5_17",
        "method": "mean_token_logprob_teacher_forced",
        "no_training": True,
        "no_full_episode_generation": True,
        "command": COMMAND,
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "fail_ids": list(FAIL_IDS),
        "control_ids": list(CONTROL_IDS),
        "adapters": [f"artifacts/task-5-16/adapters/step_{i}" for i in (1, 2, 3)],
        "metric": {
            "name": "mean_logprob",
            "definition": (
                "Average log p(token_t | prefix + tokens_<t) over continuation tokens"
            ),
            "gap": "mean_logprob_gold - mean_logprob_call",
            "delta_gap_vs_base": "gap(model) - gap(base)",
        },
        "examples": compact_examples,
    }
    _write_json(OUT / "pinned_config.json", pinned)
    print("5.17 pinned examples; scoring base + 5.16 adapters", flush=True)

    print("score[base]", flush=True)
    base = _load_base()
    scores = {"base": _score_model(base, examples)}
    del base
    _release()

    for step in (1, 2, 3):
        label = f"step_{step}"
        print(f"score[{label}]", flush=True)
        model = _load_adapter(step)
        scores[label] = _score_model(model, examples)
        del model
        _release()

    base_scores = scores["base"]
    per_example = []
    for ex in examples:
        rid = ex["rollout_id"]
        row = {
            "rollout_id": rid,
            "role": ex["role"],
            "amount_cents": ex["amount_cents"],
            "episode_error_type": ex["episode_error_type"],
            "models": {},
        }
        b = base_scores[rid]
        for name, table in scores.items():
            s = table[rid]
            entry = {
                "mean_logprob_gold": s["mean_logprob_gold"],
                "mean_logprob_call": s["mean_logprob_call"],
                "gap_gold_minus_call": s["gap_gold_minus_call"],
            }
            if name != "base":
                entry["delta_gap_vs_base"] = (
                    s["gap_gold_minus_call"] - b["gap_gold_minus_call"]
                )
                entry["delta_mean_logprob_gold_vs_base"] = (
                    s["mean_logprob_gold"] - b["mean_logprob_gold"]
                )
                entry["delta_mean_logprob_call_vs_base"] = (
                    s["mean_logprob_call"] - b["mean_logprob_call"]
                )
            row["models"][name] = entry
        per_example.append(row)

    summary = {
        "kind": "sft_logprob_diag_5_17",
        "method": "mean_token_logprob_teacher_forced",
        "command": COMMAND,
        "git_head": _git_head(),
        "versions": _package_versions(),
        "elapsed_s": round(time.time() - started, 1),
        "fail_ids": list(FAIL_IDS),
        "control_ids": list(CONTROL_IDS),
        "per_example": per_example,
        "scores_raw": scores,
    }
    _write_json(OUT / "summary.json", summary)
    _write_json(OUT / "per_example.json", {"examples": per_example})

    lines = [
        "# Task 5.17 — logprob gold final vs post-submit tool call",
        "",
        "No training, no full-episode generation. Teacher-forced mean token logprob "
        "on exact 5.14 post-submit prefixes.",
        "",
        f"- fails: `{list(FAIL_IDS)}` (call tokens = saved `new_token_ids`)",
        f"- controls: `{list(CONTROL_IDS)}` (gold = saved success tokens; "
        "call = encoded `lookup_order` for that order)",
        "- models: base + 5.16 `step_1`…`step_3`",
        "",
        "## Per example",
        "",
    ]
    for row in per_example:
        lines.append(f"### {row['rollout_id']} ({row['role']})")
        lines.append("")
        lines.append(
            "| model | mean_logp gold | mean_logp call | gap (gold−call) | Δgap vs base |"
        )
        lines.append("|---|---:|---:|---:|---:|")
        for name, m in row["models"].items():
            dg = "" if name == "base" else f"{m['delta_gap_vs_base']:.6f}"
            lines.append(
                f"| {name} | {m['mean_logprob_gold']:.6f} | {m['mean_logprob_call']:.6f} | "
                f"{m['gap_gold_minus_call']:.6f} | {dg} |"
            )
        lines.append("")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"n_examples": len(examples), "models": list(scores)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
