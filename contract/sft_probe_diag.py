"""Task 5.12: align SFT probe stop/parse with AgentTune; diagnose 5.11 finals.

Method note: this is diagnosis only (saved answers, first-token logits, a few
final regenerations). No new adapter training.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contract import config
from contract.grpo_prep import _package_versions, _write_json
from contract.harness import final_answer_text
from contract.sft_final import (
    END_RESPONSE,
    PROBE_TEMPERATURE,
    PROBE_TOP_P,
    _render_prefix_text,
    _tokenize_example,
    agenttune_eos_token_id,
    generate_on_prefix_aligned,
    parse_final_like_rollout,
    score_final,
)

OUT = ROOT / "artifacts" / "task-5-12"
TASK_511 = ROOT / "artifacts" / "task-5-11"
COMMAND = (
    "HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
    ".venv/bin/python run_sft_probe_diag.py"
)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def rescore_saved_511() -> dict[str, Any]:
    probe = json.loads((TASK_511 / "probe_comparison.json").read_text(encoding="utf-8"))
    rows = []
    for pair in probe["pairs"]:
        amount = pair["amount_cents"]
        old_base = bool(pair["base_final_correct"])
        old_ad = bool(pair["adapter_final_correct"])
        # Prefer the saved raw-ish strings: base_final/adapter_final were lightly
        # cleaned but still contain <|END_RESPONSE|> markers in 5.11 dumps.
        base_scored = score_final(pair["base_final"], amount)
        ad_scored = score_final(pair["adapter_final"], amount)
        rows.append(
            {
                "episode_id": pair["episode_id"],
                "amount_cents": amount,
                "old_base_correct": old_base,
                "old_adapter_correct": old_ad,
                "new_base": base_scored,
                "new_adapter": ad_scored,
                "base_fixed_by_parse": (not old_base) and base_scored["correct"],
                "adapter_starts_with_end_response": ad_scored["starts_with_end_response"],
                "adapter_empty_under_new_parse": ad_scored["empty"],
            }
        )
    summary = {
        "n": len(rows),
        "old_base_correct": sum(1 for r in rows if r["old_base_correct"]),
        "new_base_correct": sum(1 for r in rows if r["new_base"]["correct"]),
        "old_adapter_correct": sum(1 for r in rows if r["old_adapter_correct"]),
        "new_adapter_correct": sum(1 for r in rows if r["new_adapter"]["correct"]),
        "base_rescued_by_end_response_truncate": sum(
            1 for r in rows if r["base_fixed_by_parse"]
        ),
        "adapter_empty_because_starts_with_marker": sum(
            1 for r in rows if r["adapter_starts_with_end_response"]
        ),
        "checks": {
            "correct_line_before_end_response_counts": all(
                (not r["old_base_correct"] and r["new_base"]["correct"])
                or r["old_base_correct"]
                or not r["new_base"]["parsed"].startswith("REFUND_SUBMITTED")
                for r in rows
            ),
            "marker_leading_counts_empty": all(
                (not r["adapter_starts_with_end_response"]) or r["adapter_empty_under_new_parse"]
                for r in rows
            ),
        },
        "rows": rows,
    }
    # Explicit unit-style asserts on the saved corpus.
    rescued = [r for r in rows if r["base_fixed_by_parse"]]
    leading = [r for r in rows if r["adapter_starts_with_end_response"]]
    if not rescued:
        raise RuntimeError("expected at least one 5.11 base final rescued by truncate-at-marker")
    if not leading:
        raise RuntimeError("expected adapter finals that start with <|END_RESPONSE|>")
    if any(not r["adapter_empty_under_new_parse"] for r in leading):
        raise RuntimeError("leading <|END_RESPONSE|> must parse as empty")
    return summary


def training_target_marker_check(tokenizer) -> dict[str, Any]:
    chains = json.loads((TASK_511 / "train_chains.json").read_text(encoding="utf-8"))["chains"]
    end_id = tokenizer.encode(END_RESPONSE, add_special_tokens=False)
    rows = []
    for chain in chains:
        enc = _tokenize_example(tokenizer, chain["prefix_messages"], chain["target_final"])
        target_ids = enc["input_ids"][enc["n_prefix"] :]
        labels_target = enc["labels"][enc["n_prefix"] :]
        marker_positions = [
            i for i, tid in enumerate(target_ids) if tid in set(end_id)
        ]
        under_loss = [
            i
            for i in marker_positions
            if labels_target[i] != -100
        ]
        rows.append(
            {
                "episode_id": chain["episode_id"],
                "target_final": chain["target_final"],
                "n_target": enc["n_target"],
                "target_ids": target_ids,
                "target_tokens": [tokenizer.decode([tid]) for tid in target_ids],
                "end_response_token_ids": end_id,
                "marker_positions_in_target": marker_positions,
                "marker_under_loss_positions": under_loss,
                "marker_in_supervised_target": bool(under_loss),
                "first_supervised_token_id": target_ids[0] if target_ids else None,
                "first_supervised_token": tokenizer.decode([target_ids[0]]) if target_ids else None,
            }
        )
    return {
        "end_response": END_RESPONSE,
        "end_response_token_ids": end_id,
        "n_train": len(rows),
        "any_marker_in_target": any(r["marker_positions_in_target"] for r in rows),
        "any_marker_under_loss": any(r["marker_under_loss_positions"] for r in rows),
        "note": (
            "5.11 SFT supervised only the gold REFUND_SUBMITTED line after "
            "add_generation_prompt; chat-template <|END_RESPONSE|> was not appended "
            "to the target, so the marker is absent from labels."
        ),
        "rows": rows,
    }


def first_token_logits(
    model,
    tokenizer,
    prefix_messages: list[dict],
    amount_cents: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    prompt = _render_prefix_text(tokenizer, prefix_messages)
    prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)
    gold = final_answer_text(amount_cents)
    gold_ids = tokenizer.encode(gold, add_special_tokens=False)
    end_ids = tokenizer.encode(END_RESPONSE, add_special_tokens=False)
    if not gold_ids or not end_ids:
        raise RuntimeError("empty gold or end_response tokenization")
    gold_first = gold_ids[0]
    end_id = end_ids[0]
    ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, -1].float()
        probs = F.softmax(logits, dim=-1)
        logprobs = F.log_softmax(logits, dim=-1)
    topk = torch.topk(probs, k=5)
    return {
        "n_prefix": len(prefix_ids),
        "gold_text": gold,
        "gold_first_token_id": gold_first,
        "gold_first_token": tokenizer.decode([gold_first]),
        "end_response_token_id": end_id,
        "p_gold_first": float(probs[gold_first].item()),
        "p_end_response": float(probs[end_id].item()),
        "logp_gold_first": float(logprobs[gold_first].item()),
        "logp_end_response": float(logprobs[end_id].item()),
        "prefer_end_response_over_gold": bool(probs[end_id] > probs[gold_first]),
        "top5": [
            {
                "token_id": int(tid),
                "token": tokenizer.decode([int(tid)]),
                "prob": float(prob),
            }
            for tid, prob in zip(topk.indices.tolist(), topk.values.tolist())
        ],
    }


def compare_first_tokens(base_model, adapter_model, tokenizer) -> dict[str, Any]:
    chains = json.loads((TASK_511 / "probe_chains.json").read_text(encoding="utf-8"))["chains"]
    pairs = []
    for chain in chains:
        base = first_token_logits(
            base_model, tokenizer, chain["prefix_messages"], chain["amount_cents"]
        )
        ad = first_token_logits(
            adapter_model, tokenizer, chain["prefix_messages"], chain["amount_cents"]
        )
        pairs.append(
            {
                "episode_id": chain["episode_id"],
                "amount_cents": chain["amount_cents"],
                "base": base,
                "adapter": ad,
                "adapter_raised_end_response": ad["p_end_response"] > base["p_end_response"],
                "adapter_lowered_gold_first": ad["p_gold_first"] < base["p_gold_first"],
            }
        )
        print(
            f"logits {chain['episode_id']} base p_gold={base['p_gold_first']:.4f} "
            f"p_end={base['p_end_response']:.4f} | adapter p_gold={ad['p_gold_first']:.4f} "
            f"p_end={ad['p_end_response']:.4f}",
            flush=True,
        )
    return {
        "n": len(pairs),
        "n_adapter_prefers_end_response": sum(
            1 for p in pairs if p["adapter"]["prefer_end_response_over_gold"]
        ),
        "n_base_prefers_end_response": sum(
            1 for p in pairs if p["base"]["prefer_end_response_over_gold"]
        ),
        "n_adapter_raised_end_response": sum(
            1 for p in pairs if p["adapter_raised_end_response"]
        ),
        "n_adapter_lowered_gold_first": sum(
            1 for p in pairs if p["adapter_lowered_gold_first"]
        ),
        "pairs": pairs,
    }


def _load_failure_contexts() -> list[dict[str, Any]]:
    """Known post-submit tool-call finals from prior runs (diagnosis only)."""
    contexts: list[dict[str, Any]] = []

    # 5.8 realistic failures with saved messages + rendered prompts.
    by: dict[str, list[dict]] = defaultdict(list)
    with (ROOT / "artifacts/task-5-8/baseline/model_episodes.jsonl").open(
        encoding="utf-8"
    ) as handle:
        for line in handle:
            row = json.loads(line)
            by[row["rollout_id"]].append(row)
    meta = {
        "v0-g102": {"amount_cents": 1520, "order_id": "ORD-5001", "seed": 102},
        "v2-g102": {"amount_cents": 2675, "order_id": "ORD-5003", "seed": 102},
    }
    for rid, info in meta.items():
        last = sorted(by[rid], key=lambda item: item["step_id"])[-1]
        contexts.append(
            {
                "source": "task-5-8",
                "rollout_id": rid,
                "amount_cents": info["amount_cents"],
                "order_id": info["order_id"],
                "seed": info["seed"],
                "saved_final": last.get("raw_output") or last.get("final_answer"),
                "saved_was_tool_call": True,
                "prefix_messages": last["messages"],
                "prefix_ids": None,
            }
        )

    # 5.10 u1 post-submit prefixes (ids already include generation prompt).
    post = json.loads(
        (ROOT / "artifacts/task-5-10/post_submit_contexts.json").read_text(encoding="utf-8")
    )
    u1_meta = {
        "u1-g102": {"amount_cents": 2340, "seed": 102, "saved_was_tool_call": True},
        "u1-g103": {"amount_cents": 2340, "seed": 103, "saved_was_tool_call": True},
    }
    for row in post["rollouts"]:
        rid = row["rollout_id"]
        if rid not in u1_meta:
            continue
        info = u1_meta[rid]
        contexts.append(
            {
                "source": "task-5-10",
                "rollout_id": rid,
                "amount_cents": info["amount_cents"],
                "order_id": "ORD-3002",
                "seed": info["seed"],
                "saved_final": None,
                "saved_was_tool_call": info["saved_was_tool_call"],
                "prefix_messages": None,
                "prefix_ids": row["prefix_ids"],
            }
        )
    return contexts


def regen_failure_contexts(base_model, adapter_model, tokenizer) -> dict[str, Any]:
    contexts = _load_failure_contexts()
    rows = []
    for ctx in contexts:
        seed = int(ctx["seed"])
        common = {
            "seed": seed,
            "max_new_tokens": 128,
            "temperature": config.ONE_ACTION_TEMPERATURE,
            "top_p": config.ONE_ACTION_TOP_P,
        }
        if ctx["prefix_ids"] is not None:
            base_gen = generate_on_prefix_aligned(
                base_model, tokenizer, prefix_ids=ctx["prefix_ids"], **common
            )
            ad_gen = generate_on_prefix_aligned(
                adapter_model, tokenizer, prefix_ids=ctx["prefix_ids"], **common
            )
        else:
            base_gen = generate_on_prefix_aligned(
                base_model, tokenizer, prefix_messages=ctx["prefix_messages"], **common
            )
            ad_gen = generate_on_prefix_aligned(
                adapter_model, tokenizer, prefix_messages=ctx["prefix_messages"], **common
            )
        base_scored = score_final(base_gen["raw"], ctx["amount_cents"])
        ad_scored = score_final(ad_gen["raw"], ctx["amount_cents"])
        row = {
            "source": ctx["source"],
            "rollout_id": ctx["rollout_id"],
            "amount_cents": ctx["amount_cents"],
            "seed": seed,
            "saved_was_tool_call": ctx["saved_was_tool_call"],
            "saved_final": ctx["saved_final"],
            "base_regen": {**base_gen, **base_scored},
            "adapter_regen": {**ad_gen, **ad_scored},
            "base_tool_call_reproduced": base_scored["looks_like_tool_call"],
            "adapter_tool_call_reproduced": ad_scored["looks_like_tool_call"],
        }
        rows.append(row)
        print(
            f"regen {ctx['rollout_id']} base_tool={row['base_tool_call_reproduced']} "
            f"base_ok={base_scored['correct']} adapter_ok={ad_scored['correct']} "
            f"adapter_empty={ad_scored['empty']}",
            flush=True,
        )
    n_base_repro = sum(1 for r in rows if r["base_tool_call_reproduced"])
    proposal = None
    if n_base_repro >= 1:
        proposal = {
            "reason": (
                "Post-submit tool-call failure reproduced under the aligned probe on "
                "at least one known diagnostic context. Keep those IDs out of the next "
                "train/eval split; use them only as a fixed diagnostic probe."
            ),
            "diagnostic_holdout": [r["rollout_id"] for r in rows],
            "proposed_train": [
                {"episode_id": "t12_0", "order_id": "ORD-9101", "customer_id": "CUS-D01", "amount_cents": 1810},
                {"episode_id": "t12_1", "order_id": "ORD-9102", "customer_id": "CUS-D02", "amount_cents": 2920},
                {"episode_id": "t12_2", "order_id": "ORD-9103", "customer_id": "CUS-D03", "amount_cents": 1730},
                {"episode_id": "t12_3", "order_id": "ORD-9104", "customer_id": "CUS-D04", "amount_cents": 2640},
                {"episode_id": "t12_4", "order_id": "ORD-9105", "customer_id": "CUS-D05", "amount_cents": 1550},
                {"episode_id": "t12_5", "order_id": "ORD-9106", "customer_id": "CUS-D06", "amount_cents": 2460},
                {"episode_id": "t12_6", "order_id": "ORD-9107", "customer_id": "CUS-D07", "amount_cents": 1970},
                {"episode_id": "t12_7", "order_id": "ORD-9108", "customer_id": "CUS-D08", "amount_cents": 2180},
            ],
            "proposed_probe": [
                {"episode_id": "p12_0", "order_id": "ORD-9201", "customer_id": "CUS-E01", "amount_cents": 1610},
                {"episode_id": "p12_1", "order_id": "ORD-9202", "customer_id": "CUS-E02", "amount_cents": 2720},
                {"episode_id": "p12_2", "order_id": "ORD-9203", "customer_id": "CUS-E03", "amount_cents": 1830},
                {"episode_id": "p12_3", "order_id": "ORD-9204", "customer_id": "CUS-E04", "amount_cents": 2940},
            ],
            "proposed_eval": [
                {"episode_id": "e12_0", "order_id": "ORD-9301", "customer_id": "CUS-F01", "amount_cents": 1710},
                {"episode_id": "e12_1", "order_id": "ORD-9302", "customer_id": "CUS-F02", "amount_cents": 2820},
                {"episode_id": "e12_2", "order_id": "ORD-9303", "customer_id": "CUS-F03", "amount_cents": 1930},
                {"episode_id": "e12_3", "order_id": "ORD-9304", "customer_id": "CUS-F04", "amount_cents": 2140},
            ],
            "generation_seeds": [100, 101, 102, 103],
            "excludes": {
                "diagnostic_examples": [r["rollout_id"] for r in rows],
                "prior_splits": ["v0-v3", "u0-u3", "s0-s15", "p0-p7", "e0-e3"],
            },
            "training_hint": (
                "Next SFT should supervise gold final PLUS the chat <|END_RESPONSE|> "
                "stop marker (or stop generation on that id), since 5.11 left the marker "
                "out of the loss and the adapter collapsed to emitting it first."
            ),
        }
    return {
        "n": len(rows),
        "n_base_tool_call_reproduced": n_base_repro,
        "n_adapter_tool_call_reproduced": sum(
            1 for r in rows if r["adapter_tool_call_reproduced"]
        ),
        "rows": rows,
        "next_split_proposal": proposal,
    }


def _write_report(summary: dict[str, Any]) -> None:
    saved = summary["saved_rescore"]
    marker = summary["training_target_marker"]
    logits = summary["first_token_logits"]
    regen = summary["failure_regen"]
    lines = [
        "# Task 5.12 — SFT probe stop/parse diagnosis",
        "",
        "Aligns final-turn probe stopping and parsing with AgentTune rollout behavior,",
        "then diagnoses the 5.11 SFT adapter on saved answers and a few regenerations.",
        "",
        "## Stop / parse alignment",
        "",
        "- `eos_token_id`: model `generation_config.eos_token_id` (AgentTune), not `tokenizer.eos_token_id`.",
        "- Parse: content before the first `<|END_RESPONSE|>`; leading marker ⇒ empty final.",
        "",
        "## Saved 5.11 answers under aligned parse",
        "",
        f"- old base correct: {saved['old_base_correct']}/{saved['n']}",
        f"- new base correct: {saved['new_base_correct']}/{saved['n']}",
        f"- rescued by truncate-at-marker: {saved['base_rescued_by_end_response_truncate']}",
        f"- old adapter correct: {saved['old_adapter_correct']}/{saved['n']}",
        f"- new adapter correct: {saved['new_adapter_correct']}/{saved['n']}",
        f"- adapter leading `<|END_RESPONSE|>` (empty): {saved['adapter_empty_because_starts_with_marker']}",
        "",
        "## Training target vs `<|END_RESPONSE|>`",
        "",
        f"- marker in any target: {marker['any_marker_in_target']}",
        f"- marker under loss: {marker['any_marker_under_loss']}",
        f"- note: {marker['note']}",
        "",
        "## First-token probs on 5.11 probe prefixes",
        "",
        f"- base prefers END_RESPONSE over gold first: {logits['n_base_prefers_end_response']}/{logits['n']}",
        f"- adapter prefers END_RESPONSE over gold first: {logits['n_adapter_prefers_end_response']}/{logits['n']}",
        f"- adapter raised P(END_RESPONSE): {logits['n_adapter_raised_end_response']}/{logits['n']}",
        f"- adapter lowered P(gold first): {logits['n_adapter_lowered_gold_first']}/{logits['n']}",
        "",
        "## Regen on known post-submit tool-call contexts",
        "",
        f"- base tool-call reproduced: {regen['n_base_tool_call_reproduced']}/{regen['n']}",
        f"- adapter tool-call reproduced: {regen['n_adapter_tool_call_reproduced']}/{regen['n']}",
        "",
    ]
    if regen.get("next_split_proposal"):
        prop = regen["next_split_proposal"]
        lines.extend(
            [
                "## Proposed next split (only if failures reproduced)",
                "",
                prop["reason"],
                "",
                f"- diagnostic holdout: {', '.join(prop['diagnostic_holdout'])}",
                f"- train: {[r['episode_id'] for r in prop['proposed_train']]}",
                f"- probe: {[r['episode_id'] for r in prop['proposed_probe']]}",
                f"- eval: {[r['episode_id'] for r in prop['proposed_eval']]}",
                f"- hint: {prop['training_hint']}",
                "",
            ]
        )
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "commands.txt").write_text(COMMAND + "\n", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID, revision=config.MODEL_REVISION, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("5.12 rescoring saved 5.11 probe answers", flush=True)
    saved = rescore_saved_511()
    _write_json(OUT / "saved_rescore.json", saved)

    print("5.12 checking whether <|END_RESPONSE|> is in the SFT target/loss", flush=True)
    marker = training_target_marker_check(tokenizer)
    _write_json(OUT / "training_target_marker.json", marker)

    print("5.12 loading base + 5.11 adapter for logits and regen", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        revision=config.MODEL_REVISION,
        dtype=torch.float16,
        local_files_only=True,
    )
    base.eval()
    if torch.backends.mps.is_available():
        base.to("mps")

    adapted = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            config.MODEL_ID,
            revision=config.MODEL_REVISION,
            dtype=torch.float16,
            local_files_only=True,
        ),
        str(TASK_511 / "adapter_SFT"),
    )
    adapted.eval()
    if torch.backends.mps.is_available():
        adapted.to("mps")

    alignment = {
        "probe_eos_source": "model.generation_config.eos_token_id (AgentTune)",
        "base_eos_token_id": agenttune_eos_token_id(base, tokenizer),
        "adapter_eos_token_id": agenttune_eos_token_id(adapted, tokenizer),
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "tokenizer_eos_token": tokenizer.eos_token,
        "end_response_token_ids": tokenizer.encode(END_RESPONSE, add_special_tokens=False),
        "parse_rule": (
            "truncate at first <|END_RESPONSE|>; leading marker => empty final; "
            "then score with answer_matches / exact gold line"
        ),
        "old_probe_eos_was": "tokenizer.eos_token_id",
        "old_probe_parse_was": "answer_matches on raw/lightly cleaned text (trailing specials only)",
    }
    _write_json(OUT / "stop_parse_alignment.json", alignment)

    print("5.12 first-token logits on saved 5.11 probe prefixes", flush=True)
    logits = compare_first_tokens(base, adapted, tokenizer)
    _write_json(OUT / "first_token_logits.json", logits)

    print("5.12 regenerating finals on known post-submit tool-call contexts", flush=True)
    regen = regen_failure_contexts(base, adapted, tokenizer)
    _write_json(OUT / "failure_regen.json", regen)
    if regen.get("next_split_proposal"):
        _write_json(OUT / "next_split_proposal.json", regen["next_split_proposal"])

    summary = {
        "kind": "sft_probe_diagnosis",
        "method_note": "diagnosis only; no new SFT training",
        "command": COMMAND,
        "git_head": _git_head(),
        "versions": _package_versions(),
        "stop_parse_alignment": alignment,
        "saved_rescore": saved,
        "training_target_marker": {
            k: marker[k]
            for k in marker
            if k != "rows"
        },
        "training_target_marker_sample": marker["rows"][0] if marker["rows"] else None,
        "first_token_logits": {
            k: logits[k] for k in logits if k != "pairs"
        },
        "failure_regen": {
            k: regen[k] for k in regen if k != "rows"
        },
        "failure_regen_rows": [
            {
                "rollout_id": r["rollout_id"],
                "base_tool_call_reproduced": r["base_tool_call_reproduced"],
                "base_correct": r["base_regen"]["correct"],
                "adapter_correct": r["adapter_regen"]["correct"],
                "adapter_empty": r["adapter_regen"]["empty"],
                "adapter_starts_with_end_response": r["adapter_regen"][
                    "starts_with_end_response"
                ],
                "base_parsed": r["base_regen"]["parsed"][:120],
                "adapter_parsed": r["adapter_regen"]["parsed"][:120],
            }
            for r in regen["rows"]
        ],
    }
    _write_json(OUT / "summary.json", summary)
    _write_report(summary)
    print(json.dumps({k: summary[k] for k in summary if k not in {
        "saved_rescore", "first_token_logits", "failure_regen", "training_target_marker"
    }}, indent=2, ensure_ascii=False)[:2000], flush=True)
    print(
        json.dumps(
            {
                "saved_base_correct_old_new": [
                    saved["old_base_correct"],
                    saved["new_base_correct"],
                ],
                "adapter_leading_marker_empty": saved[
                    "adapter_empty_because_starts_with_marker"
                ],
                "marker_under_loss": marker["any_marker_under_loss"],
                "adapter_prefers_end_response": logits[
                    "n_adapter_prefers_end_response"
                ],
                "base_tool_call_reproduced": regen["n_base_tool_call_reproduced"],
                "proposed_next_split": bool(regen.get("next_split_proposal")),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
