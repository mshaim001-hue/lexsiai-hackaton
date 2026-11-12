"""Task 5.21: collect post-submit contexts from a new 8×2 base matrix.

Pins 8 new realistic refund queries × seeds {100,101} before any rollout.
Runs untrained Tiny Aya with 5.14 generation/scoring (max 16 episodes). Saves
exact post-submit token IDs for tool-call-after-submit failures (plus submit
result, repeated call, gold REFUND_SUBMITTED) and successful post-submit
contexts. No training.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contract import config
from contract.collect_prompts_521 import (
    COLLECT_521_SEEDS,
    COLLECT_521_TASKS,
    assert_collect_521_split,
    collect_521_user_prompt,
)
from contract.grpo_path import _TurnIdEngine
from contract.grpo_prep import _package_versions, _write_json
from contract.grpo_train_diag import _row_from_steps
from contract.harness import (
    ONE_ACTION_SYSTEM_PROMPT,
    ONE_ACTION_TOOL_RESULT_FORMAT,
    ONE_ACTION_TOOLS_FALLBACK_PROMPT,
    RefundEpisode,
    final_answer_text,
)
from contract.run import (
    _bind_output,
    _dump,
    _load_model,
    _one_model_rollout,
    _probe,
    make_guarded_tools,
)
from contract.sft_final import END_RESPONSE, score_final
from contract.sft_pilot import _amount_from_submit_tool, _calls_after_submit

OUT = ROOT / "artifacts" / "task-5-21"
COMMAND = (
    "HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
    ".venv/bin/python run_collect_post_submit_521.py"
)
MAX_EPISODES = 16
INCLUDE_END_RESPONSE_IN_GOLD = True


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def write_pinned_config() -> dict[str, Any]:
    assert_collect_521_split()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "datasets").mkdir(parents=True, exist_ok=True)
    (OUT / "collection").mkdir(parents=True, exist_ok=True)
    rows = []
    for episode_id, order_id, customer_id, amount in COLLECT_521_TASKS:
        rows.append(
            {
                "episode_id": episode_id,
                "order_id": order_id,
                "customer_id": customer_id,
                "amount_cents": amount,
                "prompt": collect_521_user_prompt(episode_id, order_id, customer_id),
                "environment_seed": config.ONE_ACTION_ENVIRONMENT_SEED,
                "generation_seeds": list(COLLECT_521_SEEDS),
                "prompt_style": "realistic_paraphrase_no_tool_sequence",
            }
        )
    with (OUT / "datasets" / "collect_matrix.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(COLLECT_521_TASKS) * len(COLLECT_521_SEEDS)
    if n > MAX_EPISODES:
        raise RuntimeError(f"matrix size {n} exceeds max {MAX_EPISODES}")

    payload = {
        "kind": "collect_post_submit_5_21",
        "written_before_run": True,
        "no_training": True,
        "command": COMMAND,
        "git_head": _git_head(),
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "condition": "A_untrained_tiny_aya_only",
        "inherits_generation_from": "artifacts/task-5-14",
        "matrix": {
            "n_queries": len(COLLECT_521_TASKS),
            "episode_ids": [t[0] for t in COLLECT_521_TASKS],
            "order_ids": [t[1] for t in COLLECT_521_TASKS],
            "customer_ids": [t[2] for t in COLLECT_521_TASKS],
            "amounts_cents": [t[3] for t in COLLECT_521_TASKS],
            "generation_seeds": list(COLLECT_521_SEEDS),
            "n_episodes": n,
            "max_episodes": MAX_EPISODES,
        },
        "generation": {
            "settings": "same as task-5-14 / task-5-8 one-action",
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
            "post_submit_input_ids_on_tool_call_after_submit": True,
            "post_submit_input_ids_on_success": True,
            "submit_tool_result": True,
            "gold_refund_submitted": True,
            "verification_against_traces_and_score_final": True,
        },
        "gold_final": {
            "include_end_response": INCLUDE_END_RESPONSE_IN_GOLD,
            "amount_source": "confirmed submit_refund tool result",
        },
    }
    _write_json(OUT / "pinned_config.json", payload)
    (OUT / "commands.txt").write_text(COMMAND + "\n", encoding="utf-8")
    _write_json(
        OUT / "datasets" / "manifest.json",
        {
            "collect": [t[0] for t in COLLECT_521_TASKS],
            "seeds": list(COLLECT_521_SEEDS),
            "excludes_prior": [
                "d0-d5/ORD-94xx",
                "t15_*/v15_*/ORD-95xx/96xx",
                "u0-u3",
                "v0-v3",
                "s0-s15",
                "p0-p7",
                "e0-e3",
            ],
        },
    )
    return payload


def _submit_observation_in_messages(messages: list[dict] | None) -> bool:
    if not messages:
        return False
    last = messages[-1]
    return "submit_refund" in str(last.get("content") or "")


def _extract_submit_call(steps: list[dict]) -> dict[str, Any] | None:
    for step in steps:
        for call in step.get("calls") or []:
            name = call.get("tool") or call.get("name")
            if name != "submit_refund":
                continue
            result = call.get("structured_result") or {}
            ok = (
                result.get("submitted") is True
                or (call.get("execution") == "success" and result.get("ok") is True)
            )
            if not ok:
                continue
            return {
                "tool": name,
                "arguments": call.get("arguments") or {},
                "structured_result": result,
                "execution": call.get("execution"),
                "raw_result": call.get("result"),
                "step_id": step.get("step_id"),
            }
    return None


def _post_submit_turn(
    turn_calls: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    tokenizer,
) -> dict[str, Any] | None:
    # Prefer the last generate() whose prompt messages end on a submit observation.
    # Index into turn_calls directly (do not assume 1:1 zip with step rows).
    candidates: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    n = min(len(steps), len(turn_calls))
    for i in range(n):
        step = steps[i]
        turn = turn_calls[i]
        if _submit_observation_in_messages(step.get("messages")):
            candidates.append((i, step, turn))
    if candidates:
        turn_index, step, turn = candidates[-1]
    elif steps and steps[-1].get("task_success_harness") and turn_calls:
        turn_index = len(turn_calls) - 1
        step = steps[-1]
        turn = turn_calls[turn_index]
    else:
        return None
    new_ids = list(turn["new_token_ids"])
    raw = tokenizer.decode(new_ids, skip_special_tokens=False)
    return {
        "turn_index": turn_index,
        "step_id": step.get("step_id"),
        "input_ids": list(turn["input_ids"]),
        "n_input_ids": len(turn["input_ids"]),
        "new_token_ids": new_ids,
        "n_new_tokens": len(new_ids),
        "generated_raw": raw,
        "generated_parsed": score_final(raw, int(step.get("amount_cents") or 0))["parsed"],
    }


def _verify_capture(
    *,
    tokenizer,
    post: dict[str, Any],
    turn_calls: list[dict[str, Any]],
    gold_text: str,
    gold_ids: list[int],
    amount: int,
    rid: str,
) -> dict[str, Any]:
    turn_index = int(post["turn_index"])
    if turn_index < 0 or turn_index >= len(turn_calls):
        raise RuntimeError(f"{rid}: post-submit turn_index {turn_index} out of range")
    matched = turn_calls[turn_index]
    if list(matched["input_ids"]) != list(post["input_ids"]):
        raise RuntimeError(
            f"{rid}: saved input_ids diverge from turn_calls[{turn_index}]"
        )
    if list(matched["new_token_ids"]) != list(post["new_token_ids"]):
        raise RuntimeError(
            f"{rid}: saved new_token_ids diverge from turn_calls[{turn_index}]"
        )
    # Cross-check against episode raw_output when this is the final generate.
    re_ids = tokenizer.encode(gold_text, add_special_tokens=False)
    if re_ids != gold_ids:
        raise RuntimeError(f"{rid}: gold token ids unstable under re-encode")
    scored = score_final(gold_text, amount)
    if not scored["correct"]:
        raise RuntimeError(f"{rid}: gold final failed score_final: {scored}")
    if scored["looks_like_tool_call"] or scored["empty"]:
        raise RuntimeError(f"{rid}: gold final looks empty/toolish: {scored}")
    return {
        "input_ids_match_turn_capture": True,
        "new_token_ids_match_turn_capture": True,
        "turn_index": turn_index,
        "gold_reencode_stable": True,
        "gold_score_final_correct": True,
        "matched_turn_n_input_ids": len(matched["input_ids"]),
        "gold_score": scored,
    }


def run_collection(pinned: dict[str, Any]) -> dict[str, Any]:
    import agenttune.agentic.rollout_engines.rollout_factory as factory

    coll = OUT / "collection"
    coll.mkdir(parents=True, exist_ok=True)
    _bind_output(coll)

    engine, tokenizer, loaded = _load_model()
    turn_engine = _TurnIdEngine(engine, tokenizer)
    _guards, schema_list = make_guarded_tools(
        RefundEpisode("schema-probe", "ORD-9701", "CUS-J01", 1580, config.SEED)
    )
    probe = _probe(tokenizer, schema_list)

    all_steps: list[dict] = []
    pair_rows: list[dict[str, Any]] = []
    fail_after_submit: list[dict[str, Any]] = []
    success_posts: list[dict[str, Any]] = []
    n_done = 0

    for episode_id, order_id, customer_id, amount in COLLECT_521_TASKS:
        for seed in COLLECT_521_SEEDS:
            if n_done >= MAX_EPISODES:
                break
            rid = f"{episode_id}-g{seed}"
            print(f"5.21 collect {rid}", flush=True)
            turn_engine.calls = []
            episode = RefundEpisode(
                episode_id=episode_id,
                order_id=order_id,
                customer_id=customer_id,
                amount_cents=amount,
                seed=config.ONE_ACTION_ENVIRONMENT_SEED,
            )
            prompt = collect_521_user_prompt(episode_id, order_id, customer_id)
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
            for step in ep_rows:
                step["amount_cents"] = amount
                step["order_id"] = order_id
                step["customer_id"] = customer_id
            all_steps.extend(ep_rows)
            _dump(coll / "model_episodes.jsonl", all_steps)

            scored = _row_from_steps(ep_rows, source=str(coll / "model_episodes.jsonl"))
            tool_amount = _amount_from_submit_tool(ep_rows)
            submit_call = _extract_submit_call(ep_rows)
            post = _post_submit_turn(turn_engine.calls, ep_rows, tokenizer)
            after = _calls_after_submit(ep_rows)
            confirmed = bool(scored["task_success_harness"]) and tool_amount is not None
            if (
                confirmed
                and not scored["task_success_answer"]
                and scored.get("error_type") == "final_answer_tool_call_after_submit"
            ):
                after = max(after, 1)

            pair: dict[str, Any] = {
                "rollout_id": rid,
                "episode_id": episode_id,
                "generation_seed": seed,
                "order_id": order_id,
                "customer_id": customer_id,
                "episode_amount_cents": amount,
                "tool_amount_cents": tool_amount,
                "prompt": prompt,
                "task_success": bool(scored["task_success"]),
                "confirmed": confirmed,
                "task_success_harness": bool(scored["task_success_harness"]),
                "task_success_answer": bool(scored["task_success_answer"]),
                "error_type": scored["error_type"],
                "n_invalid": int(scored.get("n_invalid") or 0),
                "n_calls_after_confirmed_submit": int(after),
                "final_answer": scored.get("final_answer"),
                "function_entries": scored.get("function_entries"),
                "submit_call": submit_call,
                "n_generate_calls": len(turn_engine.calls),
                "turn_captures_summary": [
                    {
                        "turn_index": i,
                        "n_input_ids": len(c["input_ids"]),
                        "n_new_tokens": len(c["new_token_ids"]),
                        "generated_raw": tokenizer.decode(
                            c["new_token_ids"], skip_special_tokens=False
                        )[:240],
                    }
                    for i, c in enumerate(turn_engine.calls)
                ],
            }

            if confirmed and post is not None and tool_amount is not None:
                gold_body = final_answer_text(tool_amount)
                gold_text = gold_body + (END_RESPONSE if INCLUDE_END_RESPONSE_IN_GOLD else "")
                gold_ids = tokenizer.encode(gold_text, add_special_tokens=False)
                verification = _verify_capture(
                    tokenizer=tokenizer,
                    post=post,
                    turn_calls=turn_engine.calls,
                    gold_text=gold_text,
                    gold_ids=gold_ids,
                    amount=tool_amount,
                    rid=rid,
                )
                capture = {
                    "rollout_id": rid,
                    "episode_id": episode_id,
                    "generation_seed": seed,
                    "order_id": order_id,
                    "customer_id": customer_id,
                    "amount_cents": tool_amount,
                    "confirmed": True,
                    "error_type": scored["error_type"],
                    "task_success": bool(scored["task_success"]),
                    "prefix_input_ids": post["input_ids"],
                    "n_prefix": post["n_input_ids"],
                    "generated_new_token_ids": post["new_token_ids"],
                    "generated_raw": post["generated_raw"],
                    "submit_result": submit_call,
                    "gold_text": gold_text,
                    "gold_token_ids": gold_ids,
                    "verification": verification,
                }
                pair["post_submit_capture"] = {
                    "n_prefix": post["n_input_ids"],
                    "n_new_tokens": post["n_new_tokens"],
                    "generated_raw": post["generated_raw"],
                    "gold_text": gold_text,
                    "verification": verification,
                    "ids_path": "collection/post_submit_contexts.json",
                }

                is_tool_after = (
                    (not scored["task_success_answer"])
                    and (
                        scored.get("error_type") == "final_answer_tool_call_after_submit"
                        or score_final(post["generated_raw"], tool_amount)["looks_like_tool_call"]
                    )
                )
                if is_tool_after:
                    capture["kind"] = "tool_call_after_submit"
                    capture["repeated_tool_call_raw"] = post["generated_raw"]
                    capture["repeated_tool_call_token_ids"] = post["new_token_ids"]
                    fail_after_submit.append(capture)
                elif scored["task_success"]:
                    capture["kind"] = "successful_final"
                    success_posts.append(capture)
                else:
                    capture["kind"] = "confirmed_other_final_failure"
                    # Still keep as post-submit context for analysis; not counted as
                    # repeated-call train fail unless toolish.
                    pair["post_submit_other_failure"] = True

            pair_rows.append(pair)
            n_done += 1
            print(
                f"  confirmed={confirmed} success={pair['task_success']} "
                f"err={pair['error_type']} after={pair['n_calls_after_confirmed_submit']} "
                f"tool_amt={tool_amount} post="
                f"{None if pair.get('post_submit_capture') is None else pair['post_submit_capture']['n_prefix']}",
                flush=True,
            )
        if n_done >= MAX_EPISODES:
            break

    all_posts = fail_after_submit + success_posts
    _write_json(
        coll / "post_submit_contexts.json",
        {
            "tool_call_after_submit": fail_after_submit,
            "successful_finals": success_posts,
            "all": all_posts,
        },
    )
    # Compact per-pair without huge id lists.
    compact_pairs = []
    for pair in pair_rows:
        compact_pairs.append(
            {k: v for k, v in pair.items() if k != "submit_call"}
            | {
                "submit_call_summary": None
                if pair.get("submit_call") is None
                else {
                    "arguments": pair["submit_call"].get("arguments"),
                    "structured_result": pair["submit_call"].get("structured_result"),
                    "execution": pair["submit_call"].get("execution"),
                }
            }
        )
    _write_json(coll / "per_pair.json", {"pairs": compact_pairs})
    _write_json(
        coll / "traces.json",
        {
            "rollouts": {
                p["rollout_id"]: {
                    "task_success": p["task_success"],
                    "confirmed": p["confirmed"],
                    "error_type": p["error_type"],
                    "n_calls_after_confirmed_submit": p["n_calls_after_confirmed_submit"],
                    "final_answer": p.get("final_answer"),
                    "function_entries": p.get("function_entries"),
                    "has_post_submit_capture": p.get("post_submit_capture") is not None,
                }
                for p in pair_rows
            }
        },
    )

    after_submit_fail_ids = [c["rollout_id"] for c in fail_after_submit]
    after_submit_fail_queries = sorted({c["episode_id"] for c in fail_after_submit})
    aggregates = {
        "n_episodes": len(pair_rows),
        "full_success": sum(1 for p in pair_rows if p["task_success"]),
        "confirmed_returns": sum(1 for p in pair_rows if p["confirmed"]),
        "correct_finals": sum(1 for p in pair_rows if p["task_success_answer"]),
        "tool_call_after_submit_count": len(fail_after_submit),
        "tool_call_after_submit_rollout_ids": after_submit_fail_ids,
        "n_queries_with_tool_call_after_submit": len(after_submit_fail_queries),
        "queries_with_tool_call_after_submit": after_submit_fail_queries,
        "successful_post_submit_contexts": len(success_posts),
        "calls_after_confirmed_submit_total": sum(
            int(p["n_calls_after_confirmed_submit"]) for p in pair_rows
        ),
        "invalid_calls_total": sum(int(p["n_invalid"]) for p in pair_rows),
        "error_type_counts": dict(Counter(p["error_type"] for p in pair_rows)),
        "verification_all_ok": all(
            (c.get("verification") or {}).get("gold_score_final_correct")
            and (c.get("verification") or {}).get("input_ids_match_turn_capture")
            for c in all_posts
        )
        if all_posts
        else True,
    }
    return {
        "loaded_model": loaded,
        "aggregates": aggregates,
        "pairs": compact_pairs,
        "fail_after_submit": fail_after_submit,
        "success_posts": success_posts,
    }


def _write_report(summary: dict[str, Any]) -> None:
    agg = summary["aggregates"]
    lines = [
        "# Task 5.21 — collect post-submit contexts (new 8×2 matrix)",
        "",
        "Untrained Tiny Aya. Generation/scoring match task 5.14. "
        "No training. Max 16 episodes.",
        "",
        f"- confirmed_returns: **{agg['confirmed_returns']}**/{agg['n_episodes']}",
        f"- tool_call_after_submit: **{agg['tool_call_after_submit_count']}** "
        f"(queries: {agg['queries_with_tool_call_after_submit']})",
        f"- n_queries_with_tool_call_after_submit: "
        f"**{agg['n_queries_with_tool_call_after_submit']}**",
        f"- successful_post_submit_contexts: {agg['successful_post_submit_contexts']}",
        f"- full_success: {agg['full_success']}/{agg['n_episodes']}",
        f"- verification_all_ok: {agg['verification_all_ok']}",
        "",
        "## Per episode",
        "",
        "| rollout | confirmed | success | error_type | after | post_capture |",
        "|---|---|---|---|---:|---|",
    ]
    for p in summary["pairs"]:
        lines.append(
            f"| {p['rollout_id']} | {p['confirmed']} | {p['task_success']} | "
            f"{p['error_type']} | {p['n_calls_after_confirmed_submit']} | "
            f"{p.get('post_submit_capture') is not None} |"
        )
    lines.append("")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    started = time.time()
    pinned = write_pinned_config()
    print(
        f"5.21 pinned matrix={pinned['matrix']['episode_ids']} "
        f"seeds={pinned['matrix']['generation_seeds']}",
        flush=True,
    )
    collected = run_collection(pinned)
    summary = {
        "kind": "collect_post_submit_5_21",
        "no_training": True,
        "command": COMMAND,
        "git_head": _git_head(),
        "python": platform.python_version(),
        "versions": _package_versions(),
        "elapsed_s": round(time.time() - started, 1),
        "pinned_config": pinned,
        "loaded_model": collected["loaded_model"],
        "aggregates": collected["aggregates"],
        "pairs": collected["pairs"],
        "n_tool_call_after_submit_saved": len(collected["fail_after_submit"]),
        "n_success_post_submit_saved": len(collected["success_posts"]),
    }
    _write_json(OUT / "summary.json", summary)
    _write_json(OUT / "aggregates.json", collected["aggregates"])
    _write_report(summary)
    print(json.dumps(collected["aggregates"], indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
