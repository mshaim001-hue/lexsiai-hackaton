"""Task 5.14: discovery matrix of 6 realistic queries × 2 seeds on untrained Tiny Aya.

Pins the split before any rollout. Uses task-5-8 one-action settings. Captures
exact input_ids and new_token_ids for the final turn after a confirmed
submit_refund. Max 12 episodes. No adapter.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contract import config
from contract.discovery_prompts import (
    DISCOVERY_TASKS,
    GENERATION_SEEDS,
    assert_discovery_split,
    discovery_user_prompt,
)
from contract.grpo_path import _TurnIdEngine
from contract.grpo_prep import _package_versions, _write_json
from contract.grpo_train_diag import _row_from_steps
from contract.harness import (
    ONE_ACTION_SYSTEM_PROMPT,
    ONE_ACTION_TOOL_RESULT_FORMAT,
    ONE_ACTION_TOOLS_FALLBACK_PROMPT,
    RefundEpisode,
)
from contract.run import (
    _bind_output,
    _dump,
    _load_model,
    _one_model_rollout,
    _probe,
    make_guarded_tools,
)
from contract.sft_final import parse_final_like_rollout, score_final

OUT = ROOT / "artifacts" / "task-5-14"
DATASETS = OUT / "datasets"
BASELINE = OUT / "baseline"
COMMAND = (
    "HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
    ".venv/bin/python run_discovery_baseline.py"
)
MAX_EPISODES = 12


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def write_pinned_config() -> dict[str, Any]:
    assert_discovery_split()
    OUT.mkdir(parents=True, exist_ok=True)
    DATASETS.mkdir(parents=True, exist_ok=True)
    rows = []
    for episode_id, order_id, customer_id, amount in DISCOVERY_TASKS:
        rows.append(
            {
                "episode_id": episode_id,
                "order_id": order_id,
                "customer_id": customer_id,
                "amount_cents": amount,
                "prompt": discovery_user_prompt(episode_id, order_id, customer_id),
                "environment_seed": config.ONE_ACTION_ENVIRONMENT_SEED,
                "generation_seeds": list(GENERATION_SEEDS),
                "prompt_style": "realistic_paraphrase_no_tool_sequence",
            }
        )
    with (DATASETS / "discovery.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(DISCOVERY_TASKS) * len(GENERATION_SEEDS)
    if n > MAX_EPISODES:
        raise RuntimeError(f"matrix size {n} exceeds max {MAX_EPISODES}")

    payload = {
        "kind": "discovery_matrix_pinned_before_run",
        "written_before_run": True,
        "command": COMMAND,
        "git_head": _git_head(),
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "condition": "A_untrained_tiny_aya_only",
        "discovery": {
            "n_queries": len(DISCOVERY_TASKS),
            "episode_ids": [t[0] for t in DISCOVERY_TASKS],
            "order_ids": [t[1] for t in DISCOVERY_TASKS],
            "generation_seeds": list(GENERATION_SEEDS),
            "n_episodes": n,
            "max_episodes": MAX_EPISODES,
        },
        "generation": {
            "settings": "same as task-5-8 one-action",
            "environment_seed": config.ONE_ACTION_ENVIRONMENT_SEED,
            "temperature": config.ONE_ACTION_TEMPERATURE,
            "top_p": config.ONE_ACTION_TOP_P,
            "do_sample": config.ONE_ACTION_DO_SAMPLE,
            "max_new_tokens": config.ONE_ACTION_MAX_NEW_TOKENS,
            "max_steps": config.ONE_ACTION_MAX_STEPS,
            "protocol": "one_action_per_turn",
            "system_prompt": "ONE_ACTION_SYSTEM_PROMPT",
            "tools_fallback_prompt": "ONE_ACTION_TOOLS_FALLBACK_PROMPT",
            "tool_result_format": "ONE_ACTION_TOOL_RESULT_FORMAT",
            "user_prompt_style": "realistic_paraphrase_no_tool_sequence",
            "weights_updated": False,
        },
        "captures": {
            "full_episode_log": True,
            "post_submit_final_turn_input_ids": True,
            "generated_token_ids": True,
            "verification": True,
        },
        "metrics": {
            "unconfirmed_returns": "not task_success_harness",
            "final_failure_after_confirm": (
                "task_success_harness and not task_success_answer "
                "(includes wrong/empty final and tool call after submit)"
            ),
            "full_success": "task_success",
        },
        "eval_set_rule": (
            "If final_failure_after_confirm occurs on several different queries, "
            "keep this discovery matrix as a held-out eval set and collect train "
            "examples separately."
        ),
    }
    _write_json(OUT / "pinned_config.json", payload)
    (OUT / "commands.txt").write_text(COMMAND + "\n", encoding="utf-8")
    _write_json(
        DATASETS / "manifest.json",
        {
            "discovery": [t[0] for t in DISCOVERY_TASKS],
            "seeds": list(GENERATION_SEEDS),
            "excludes_prior": ["v0-v3", "u0-u3", "s0-s15", "p0-p7", "e0-e3"],
        },
    )
    return payload


def _submit_observation_in_messages(messages: list[dict] | None) -> bool:
    if not messages:
        return False
    last = messages[-1]
    return "submit_refund" in str(last.get("content") or "")


def _extract_post_submit_turn(
    turn_calls: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    tokenizer,
) -> dict[str, Any] | None:
    """Return the last generate() capture whose prompt ends on a submit observation."""
    # Align turn captures with recorded steps that have messages.
    candidates = []
    for step, turn in zip(steps, turn_calls):
        messages = step.get("messages")
        if _submit_observation_in_messages(messages):
            candidates.append((step, turn))
    if not candidates:
        # Fall back: if harness confirmed and we have turn captures, use the last turn.
        last = steps[-1] if steps else None
        if last and last.get("task_success_harness") and turn_calls:
            step, turn = last, turn_calls[-1]
        else:
            return None
    else:
        step, turn = candidates[-1]
    new_ids = list(turn["new_token_ids"])
    raw = tokenizer.decode(new_ids, skip_special_tokens=False)
    amount = step.get("amount_cents")
    # Prefer episode amount from step metadata if present; else parse later.
    return {
        "step_id": step.get("step_id"),
        "input_ids": list(turn["input_ids"]),
        "n_input_ids": len(turn["input_ids"]),
        "new_token_ids": new_ids,
        "n_new_tokens": len(new_ids),
        "generated_raw": raw,
        "generated_parsed": parse_final_like_rollout(raw),
    }


def run_discovery(pinned: dict[str, Any]) -> dict[str, Any]:
    import agenttune.agentic.rollout_engines.rollout_factory as factory

    BASELINE.mkdir(parents=True, exist_ok=True)
    _bind_output(BASELINE)

    engine, tokenizer, loaded = _load_model()
    turn_engine = _TurnIdEngine(engine, tokenizer)
    _guards, schema_list = make_guarded_tools(
        RefundEpisode("schema-probe", "ORD-9401", "CUS-G01", 1620, config.SEED)
    )
    probe = _probe(tokenizer, schema_list)

    all_steps: list[dict] = []
    pair_rows: list[dict] = []
    n_done = 0

    for episode_id, order_id, customer_id, amount in DISCOVERY_TASKS:
        for seed in GENERATION_SEEDS:
            if n_done >= MAX_EPISODES:
                break
            rid = f"{episode_id}-g{seed}"
            print(f"discovery {rid}", flush=True)
            turn_engine.calls = []
            episode = RefundEpisode(
                episode_id=episode_id,
                order_id=order_id,
                customer_id=customer_id,
                amount_cents=amount,
                seed=config.ONE_ACTION_ENVIRONMENT_SEED,
            )
            prompt = discovery_user_prompt(episode_id, order_id, customer_id)
            ep_rows, limits = _one_model_rollout(
                factory,
                turn_engine,
                tokenizer,
                schema_list,
                probe,
                episode,
                generation_seed=seed,
                environment_seed=config.ONE_ACTION_ENVIRONMENT_SEED,
                max_new_tokens=config.ONE_ACTION_MAX_NEW_TOKENS,
                temperature=config.ONE_ACTION_TEMPERATURE,
                do_sample=config.ONE_ACTION_DO_SAMPLE,
                top_p=config.ONE_ACTION_TOP_P,
                system_prompt=ONE_ACTION_SYSTEM_PROMPT,
                tools_fallback_prompt=ONE_ACTION_TOOLS_FALLBACK_PROMPT,
                tool_result_format=ONE_ACTION_TOOL_RESULT_FORMAT,
                prompt_text=prompt,
                prompt_protocol="one_action_per_turn",
            )
            if limits:
                print("limits", limits, flush=True)
            # Attach amount onto steps for downstream helpers.
            for step in ep_rows:
                step["amount_cents"] = amount
                step["order_id"] = order_id
                step["customer_id"] = customer_id
            all_steps.extend(ep_rows)
            _dump(BASELINE / "model_episodes.jsonl", all_steps)

            scored = _row_from_steps(ep_rows, source=str(BASELINE / "model_episodes.jsonl"))
            scored.update(
                {
                    "order_id": order_id,
                    "customer_id": customer_id,
                    "amount_cents": amount,
                    "prompt": prompt,
                }
            )
            # Aligned final parse (truncate at END_RESPONSE) for verification detail.
            final_raw = ep_rows[-1].get("raw_output") or ep_rows[-1].get("final_answer")
            final_check = score_final(final_raw, amount)
            post = _extract_post_submit_turn(turn_engine.calls, ep_rows, tokenizer)
            if post is not None:
                post["verification"] = score_final(post["generated_raw"], amount)
            pair = {
                **scored,
                "verification": {
                    "task_success": bool(scored["task_success"]),
                    "confirmed_return": bool(scored["task_success_harness"]),
                    "correct_final_harness": bool(scored["task_success_answer"]),
                    "aligned_final_parse": final_check,
                    "error_type": scored["error_type"],
                },
                "turn_captures": [
                    {
                        "turn_index": i,
                        "n_input_ids": len(c["input_ids"]),
                        "n_new_tokens": len(c["new_token_ids"]),
                        "new_token_ids": c["new_token_ids"],
                        "generated_raw": tokenizer.decode(
                            c["new_token_ids"], skip_special_tokens=False
                        ),
                    }
                    for i, c in enumerate(turn_engine.calls)
                ],
                "post_submit_final_turn": post,
                "n_generate_calls": len(turn_engine.calls),
            }
            pair_rows.append(pair)
            n_done += 1
            print(
                f"  success={pair['task_success']} confirmed={pair['task_success_harness']} "
                f"answer={pair['task_success_answer']} err={pair['error_type']} "
                f"post_submit_ids="
                f"{None if post is None else post['n_input_ids']}",
                flush=True,
            )
        if n_done >= MAX_EPISODES:
            break

    # Persist full post-submit id dumps separately (can be large).
    post_submit_dump = []
    for pair in pair_rows:
        post = pair.get("post_submit_final_turn")
        if not post:
            continue
        post_submit_dump.append(
            {
                "rollout_id": pair["rollout_id"],
                "episode_id": pair["episode_id"],
                "generation_seed": pair["generation_seed"],
                "amount_cents": pair["amount_cents"],
                "confirmed": pair["task_success_harness"],
                "input_ids": post["input_ids"],
                "new_token_ids": post["new_token_ids"],
                "generated_raw": post["generated_raw"],
                "verification": post["verification"],
            }
        )
    _write_json(BASELINE / "post_submit_final_turns.json", {"turns": post_submit_dump})

    # Compact per-pair without repeating huge input_ids in the main table.
    compact_pairs = []
    for pair in pair_rows:
        post = pair.get("post_submit_final_turn")
        compact = {
            **{k: v for k, v in pair.items() if k not in {"post_submit_final_turn", "turn_captures"}},
            "turn_captures_summary": [
                {
                    "turn_index": t["turn_index"],
                    "n_input_ids": t["n_input_ids"],
                    "n_new_tokens": t["n_new_tokens"],
                    "generated_raw": t["generated_raw"][:240],
                }
                for t in pair.get("turn_captures") or []
            ],
            "post_submit_final_turn": None
            if post is None
            else {
                "step_id": post["step_id"],
                "n_input_ids": post["n_input_ids"],
                "n_new_tokens": post["n_new_tokens"],
                "generated_raw": post["generated_raw"],
                "generated_parsed": post["generated_parsed"],
                "verification": post["verification"],
                "input_ids_path": "baseline/post_submit_final_turns.json",
            },
        }
        compact_pairs.append(compact)

    unconfirmed = [p for p in pair_rows if not p["task_success_harness"]]
    final_fail_after_confirm = [
        p
        for p in pair_rows
        if p["task_success_harness"] and not p["task_success_answer"]
    ]
    final_fail_queries = sorted({p["episode_id"] for p in final_fail_after_confirm})
    keep_as_eval = len(final_fail_queries) >= 2
    aggregates = {
        "n_episodes": len(pair_rows),
        "full_success": sum(1 for p in pair_rows if p["task_success"]),
        "confirmed_returns": sum(1 for p in pair_rows if p["task_success_harness"]),
        "unconfirmed_returns": len(unconfirmed),
        "correct_finals": sum(1 for p in pair_rows if p["task_success_answer"]),
        "final_failure_after_confirm": len(final_fail_after_confirm),
        "final_failure_after_confirm_queries": final_fail_queries,
        "n_queries_with_final_failure_after_confirm": len(final_fail_queries),
        "error_type_counts": dict(Counter(p["error_type"] for p in pair_rows)),
        "unconfirmed_rollout_ids": [p["rollout_id"] for p in unconfirmed],
        "final_failure_after_confirm_rollout_ids": [
            p["rollout_id"] for p in final_fail_after_confirm
        ],
        "keep_discovery_as_eval_set": keep_as_eval,
        "eval_set_decision": (
            "Keep d0–d5 × {100,101} as held-out eval; collect train examples on a "
            "separate prompt set."
            if keep_as_eval
            else (
                "Final-after-confirm failures did not span multiple queries; "
                "do not automatically promote this matrix to the eval set."
            )
        ),
    }

    summary = {
        "kind": "discovery_baseline_A",
        "method_note": "untrained Tiny Aya Global only; settings match task-5-8",
        "command": COMMAND,
        "git_head": _git_head(),
        "python": platform.python_version(),
        "versions": _package_versions(),
        "loaded_model": loaded,
        "pinned_config": pinned,
        "n_rollouts": len(pair_rows),
        "aggregates": aggregates,
        "pairs": compact_pairs,
    }
    _write_json(BASELINE / "per_pair.json", {"pairs": compact_pairs})
    _write_json(BASELINE / "aggregates.json", aggregates)
    _write_json(BASELINE / "summary.json", summary)
    _write_json(OUT / "summary.json", summary)
    _write_json(
        BASELINE / "traces.json",
        {
            "rollouts": {
                p["rollout_id"]: {
                    "final_answer": p.get("final_answer"),
                    "task_success": p.get("task_success"),
                    "task_success_harness": p.get("task_success_harness"),
                    "task_success_answer": p.get("task_success_answer"),
                    "error_type": p.get("error_type"),
                    "function_entries": p.get("function_entries"),
                    "n_generate_calls": p.get("n_generate_calls"),
                    "post_submit_captured": p.get("post_submit_final_turn") is not None,
                }
                for p in pair_rows
            }
        },
    )
    _write_report(summary)
    print(json.dumps(aggregates, indent=2, ensure_ascii=False), flush=True)
    return summary


def _write_report(summary: dict[str, Any]) -> None:
    agg = summary["aggregates"]
    lines = [
        "# Task 5.14 — discovery matrix (untrained Tiny Aya)",
        "",
        "6 new realistic refund queries × seeds `{100, 101}` (12 episodes).",
        "Generation settings match task-5-8 one-action. No adapter.",
        "",
        f"- full_success: {agg['full_success']}/{agg['n_episodes']}",
        f"- confirmed_returns: {agg['confirmed_returns']}",
        f"- unconfirmed_returns: {agg['unconfirmed_returns']}",
        f"- final_failure_after_confirm: {agg['final_failure_after_confirm']}",
        f"- queries with that failure: {agg['final_failure_after_confirm_queries']}",
        f"- keep as eval set: {agg['keep_discovery_as_eval_set']}",
        "",
        agg["eval_set_decision"],
        "",
        "| rollout | confirmed | correct_final | success | error_type | post_submit_ids |",
        "|---|---|---|---|---|---:|",
    ]
    for p in summary["pairs"]:
        post = p.get("post_submit_final_turn")
        n_ids = "" if post is None else post["n_input_ids"]
        lines.append(
            f"| {p['rollout_id']} | {p['task_success_harness']} | {p['task_success_answer']} | "
            f"{p['task_success']} | {p['error_type']} | {n_ids} |"
        )
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    pinned = write_pinned_config()
    print("5.14 pinned discovery matrix", flush=True)
    print(
        json.dumps(
            {
                "queries": pinned["discovery"]["episode_ids"],
                "seeds": pinned["discovery"]["generation_seeds"],
                "n_episodes": pinned["discovery"]["n_episodes"],
                "temperature": pinned["generation"]["temperature"],
                "top_p": pinned["generation"]["top_p"],
            },
            indent=2,
        ),
        flush=True,
    )
    run_discovery(pinned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
