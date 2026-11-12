"""Task 5.18: first-divergence token logp/rank on 5.17 post-submit contexts.

No training, no full-episode generation. For each 5.17 example, find the first
token where gold final and lookup_order continuations diverge, then score
logp(gold)−logp(call) and ranks at that decision point for base and step_3.
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
from contract.sft_logprob_diag_517 import (
    CONTROL_IDS,
    FAIL_IDS,
    _load_adapter,
    _load_base,
    _load_examples,
)

OUT = ROOT / "artifacts" / "task-5-18"
PINNED_517 = ROOT / "artifacts" / "task-5-17" / "pinned_config.json"
COMMAND = (
    "HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
    ".venv/bin/python run_sft_first_diverge_518.py"
)
MODELS = ("base", "step_3")


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _tok_text(tokenizer, tok_id: int) -> str:
    return tokenizer.decode([tok_id], skip_special_tokens=False)


def _first_diverge(gold_ids: list[int], call_ids: list[int]) -> dict[str, Any]:
    n = min(len(gold_ids), len(call_ids))
    idx = None
    for i in range(n):
        if gold_ids[i] != call_ids[i]:
            idx = i
            break
    if idx is None:
        if len(gold_ids) == len(call_ids):
            raise RuntimeError("gold and call token sequences are identical")
        idx = n
        if idx >= len(gold_ids) or idx >= len(call_ids):
            # One sequence is a proper prefix of the other; treat next token
            # of the longer side vs EOS-like absence is not defined here.
            raise RuntimeError(
                f"sequences share full shorter prefix (len={n}); "
                f"gold_len={len(gold_ids)} call_len={len(call_ids)}"
            )
    return {
        "diverge_index": idx,
        "shared_prefix_ids": gold_ids[:idx],
        "n_shared": idx,
        "gold_token_id": gold_ids[idx],
        "call_token_id": call_ids[idx],
    }


def _verify_against_517(examples: list[dict[str, Any]]) -> None:
    pinned = json.loads(PINNED_517.read_text(encoding="utf-8"))
    by_id = {ex["rollout_id"]: ex for ex in pinned["examples"]}
    for ex in examples:
        p = by_id[ex["rollout_id"]]
        if ex["gold_token_ids"][:8] != p["gold_token_ids_head"]:
            raise RuntimeError(f"{ex['rollout_id']}: gold token head != 5.17")
        if ex["call_token_ids"][:8] != p["call_token_ids_head"]:
            raise RuntimeError(f"{ex['rollout_id']}: call token head != 5.17")
        if len(ex["gold_token_ids"]) != p["n_gold_tokens"]:
            raise RuntimeError(f"{ex['rollout_id']}: n_gold_tokens != 5.17")
        if len(ex["call_token_ids"]) != p["n_call_tokens"]:
            raise RuntimeError(f"{ex['rollout_id']}: n_call_tokens != 5.17")
        if len(ex["prefix_input_ids"]) != p["n_prefix"]:
            raise RuntimeError(f"{ex['rollout_id']}: n_prefix != 5.17")


def _score_divergence(
    model, tokenizer, examples: list[dict[str, Any]]
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    by_id: dict[str, Any] = {}
    model.eval()
    for ex in examples:
        div = _first_diverge(ex["gold_token_ids"], ex["call_token_ids"])
        ctx = ex["prefix_input_ids"] + div["shared_prefix_ids"]
        ids = torch.tensor([ctx], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0, -1].float()
        logp = F.log_softmax(logits, dim=-1)
        probs = logp.exp()
        # 1-based rank: rank 1 = highest probability.
        order = torch.argsort(logits, descending=True)
        rank_of = {int(t.item()): i + 1 for i, t in enumerate(order)}
        gold_id = div["gold_token_id"]
        call_id = div["call_token_id"]
        top_id = int(order[0].item())
        gold_logp = float(logp[gold_id].item())
        call_logp = float(logp[call_id].item())
        top_logp = float(logp[top_id].item())
        row = {
            "rollout_id": ex["rollout_id"],
            "role": ex["role"],
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "n_context_tokens": len(ctx),
            "diverge_index": div["diverge_index"],
            "n_shared_continuation": div["n_shared"],
            "shared_prefix_ids": div["shared_prefix_ids"],
            "shared_prefix_text": tokenizer.decode(
                div["shared_prefix_ids"], skip_special_tokens=False
            ),
            "gold_token_id": gold_id,
            "gold_token_text": _tok_text(tokenizer, gold_id),
            "gold_logp": gold_logp,
            "gold_rank": rank_of[gold_id],
            "call_token_id": call_id,
            "call_token_text": _tok_text(tokenizer, call_id),
            "call_logp": call_logp,
            "call_rank": rank_of[call_id],
            "delta_logp_gold_minus_call": gold_logp - call_logp,
            "top1_token_id": top_id,
            "top1_token_text": _tok_text(tokenizer, top_id),
            "top1_logp": top_logp,
            "top1_is_gold": top_id == gold_id,
            "top1_is_call": top_id == call_id,
            "top1_differs_from_both": top_id not in (gold_id, call_id),
            "vocab_size": int(logits.numel()),
        }
        by_id[ex["rollout_id"]] = row
        extra = ""
        if row["top1_differs_from_both"]:
            extra = (
                f" top1={top_id!r}/{row['top1_token_text']!r}"
                f" logp={top_logp:.6f}"
            )
        print(
            f"  {ex['rollout_id']} ({ex['role']}): "
            f"div@{div['diverge_index']} "
            f"gold={gold_id}/{row['gold_token_text']!r}@{row['gold_rank']} "
            f"call={call_id}/{row['call_token_text']!r}@{row['call_rank']} "
            f"Δlogp={row['delta_logp_gold_minus_call']:.6f}"
            f"{extra}",
            flush=True,
        )
    return by_id


def main() -> int:
    from transformers import AutoTokenizer

    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "commands.txt").write_text(COMMAND + "\n", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID, revision=config.MODEL_REVISION, local_files_only=True
    )
    examples = _load_examples(tokenizer)
    _verify_against_517(examples)

    diverge_meta = []
    for ex in examples:
        div = _first_diverge(ex["gold_token_ids"], ex["call_token_ids"])
        diverge_meta.append(
            {
                "rollout_id": ex["rollout_id"],
                "role": ex["role"],
                "amount_cents": ex["amount_cents"],
                "order_id": ex["order_id"],
                "episode_error_type": ex["episode_error_type"],
                "n_prefix": ex["n_prefix"],
                "diverge_index": div["diverge_index"],
                "n_shared_continuation": div["n_shared"],
                "shared_prefix_ids": div["shared_prefix_ids"],
                "gold_token_id": div["gold_token_id"],
                "gold_token_text": _tok_text(tokenizer, div["gold_token_id"]),
                "call_token_id": div["call_token_id"],
                "call_token_text": _tok_text(tokenizer, div["call_token_id"]),
                "gold_source": ex["gold_source"],
                "call_source": ex["call_source"],
            }
        )

    pinned = {
        "kind": "sft_first_diverge_diag_5_18",
        "method": "first_divergence_token_logp_and_rank",
        "no_training": True,
        "no_full_episode_generation": True,
        "command": COMMAND,
        "source_task": "artifacts/task-5-17",
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "fail_ids": list(FAIL_IDS),
        "control_ids": list(CONTROL_IDS),
        "models": list(MODELS),
        "adapters": ["artifacts/task-5-16/adapters/step_3"],
        "metric": {
            "name": "delta_logp_at_first_divergence",
            "definition": (
                "At context = post-submit prefix + shared continuation prefix, "
                "logp(gold_token) - logp(call_token); ranks are 1-based "
                "(1 = most probable)."
            ),
        },
        "examples": diverge_meta,
    }
    _write_json(OUT / "pinned_config.json", pinned)
    print("5.18 first-divergence; scoring base + step_3", flush=True)

    scores: dict[str, Any] = {}
    print("score[base]", flush=True)
    base = _load_base()
    scores["base"] = _score_divergence(base, tokenizer, examples)
    del base
    _release()

    print("score[step_3]", flush=True)
    step3 = _load_adapter(3)
    scores["step_3"] = _score_divergence(step3, tokenizer, examples)
    del step3
    _release()

    meta_by_id = {m["rollout_id"]: m for m in diverge_meta}
    per_example = []
    for ex in examples:
        rid = ex["rollout_id"]
        row = {
            "rollout_id": rid,
            "role": ex["role"],
            "amount_cents": ex["amount_cents"],
            "episode_error_type": ex["episode_error_type"],
            "diverge_index": meta_by_id[rid]["diverge_index"],
            "models": {name: scores[name][rid] for name in MODELS},
        }
        per_example.append(row)

    summary = {
        "kind": "sft_first_diverge_diag_5_18",
        "method": "first_divergence_token_logp_and_rank",
        "command": COMMAND,
        "git_head": _git_head(),
        "versions": _package_versions(),
        "elapsed_s": round(time.time() - started, 1),
        "fail_ids": list(FAIL_IDS),
        "control_ids": list(CONTROL_IDS),
        "models": list(MODELS),
        "per_example": per_example,
        "scores_raw": scores,
    }
    _write_json(OUT / "summary.json", summary)
    _write_json(OUT / "per_example.json", {"examples": per_example})

    lines = [
        "# Task 5.18 — first-divergence token logp / rank",
        "",
        "Same 5.17 post-submit prefixes and gold/call token IDs. No training, "
        "no full-episode generation. Models: base + 5.16 `step_3`.",
        "",
        "At the first index where gold final and lookup_order continuations "
        "differ, score `logp(gold_token) − logp(call_token)` and 1-based ranks "
        "on `prefix + shared continuation prefix`.",
        "",
        "## Table",
        "",
        "| example | role | model | div@ | gold id/text | gold logp | gold rank | "
        "call id/text | call logp | call rank | Δlogp (gold−call) | top1 id/text |",
        "|---|---|---|---:|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in per_example:
        for name in MODELS:
            m = row["models"][name]
            top_cell = f"{m['top1_token_id']}/{m['top1_token_text']!r}"
            if not m["top1_differs_from_both"]:
                if m["top1_is_gold"]:
                    top_cell += " (=gold)"
                else:
                    top_cell += " (=call)"
            lines.append(
                f"| {row['rollout_id']} | {row['role']} | {name} | {m['diverge_index']} | "
                f"{m['gold_token_id']}/{m['gold_token_text']!r} | {m['gold_logp']:.6f} | "
                f"{m['gold_rank']} | {m['call_token_id']}/{m['call_token_text']!r} | "
                f"{m['call_logp']:.6f} | {m['call_rank']} | "
                f"{m['delta_logp_gold_minus_call']:.6f} | {top_cell} |"
            )
    lines.append("")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps({"n_examples": len(examples), "models": list(MODELS)}, indent=2),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
