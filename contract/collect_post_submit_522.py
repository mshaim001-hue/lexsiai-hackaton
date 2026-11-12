"""Task 5.22: continue post-submit train collection (4×2 base matrix).

Same generation/scoring/capture format as 5.21. Pins c22_0–c22_3 × {100,101}
before rollout. No training.
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
from contract.collect_post_submit_521 import (
    INCLUDE_END_RESPONSE_IN_GOLD,
    _extract_submit_call,
    _post_submit_turn,
    _verify_capture,
)
from contract.collect_prompts_522 import (
    COLLECT_522_SEEDS,
    COLLECT_522_TASKS,
    assert_collect_522_split,
    collect_522_user_prompt,
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

OUT = ROOT / "artifacts" / "task-5-22"
SRC_521 = ROOT / "artifacts" / "task-5-21"
COMMAND = (
    "HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
    ".venv/bin/python run_collect_post_submit_522.py"
)
MAX_EPISODES = 8


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def write_pinned_config() -> dict[str, Any]:
    assert_collect_522_split()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "datasets").mkdir(parents=True, exist_ok=True)
    (OUT / "collection").mkdir(parents=True, exist_ok=True)
    rows = []
    for episode_id, order_id, customer_id, amount in COLLECT_522_TASKS:
        rows.append(
            {
                "episode_id": episode_id,
                "order_id": order_id,
                "customer_id": customer_id,
                "amount_cents": amount,
                "prompt": collect_522_user_prompt(episode_id, order_id, customer_id),
                "environment_seed": config.ONE_ACTION_ENVIRONMENT_SEED,
                "generation_seeds": list(COLLECT_522_SEEDS),
                "prompt_style": "realistic_paraphrase_no_tool_sequence",
            }
        )
    with (OUT / "datasets" / "collect_matrix.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(COLLECT_522_TASKS) * len(COLLECT_522_SEEDS)
    if n > MAX_EPISODES:
        raise RuntimeError(f"matrix size {n} exceeds max {MAX_EPISODES}")

    payload = {
        "kind": "collect_post_submit_5_22",
        "written_before_run": True,
        "no_training": True,
        "command": COMMAND,
        "git_head": _git_head(),
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "condition": "A_untrained_tiny_aya_only",
        "continues": "artifacts/task-5-21",
        "capture_format": "same as artifacts/task-5-21/collection/post_submit_contexts.json",
        "inherits_generation_from": "artifacts/task-5-21 / task-5-14",
        "matrix": {
            "n_queries": len(COLLECT_522_TASKS),
            "episode_ids": [t[0] for t in COLLECT_522_TASKS],
            "order_ids": [t[1] for t in COLLECT_522_TASKS],
            "customer_ids": [t[2] for t in COLLECT_522_TASKS],
            "amounts_cents": [t[3] for t in COLLECT_522_TASKS],
            "generation_seeds": list(COLLECT_522_SEEDS),
            "n_episodes": n,
            "max_episodes": MAX_EPISODES,
        },
        "generation": {
            "settings": "same as task-5-21 / task-5-14 one-action",
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
            "collect": [t[0] for t in COLLECT_522_TASKS],
            "seeds": list(COLLECT_522_SEEDS),
            "excludes_prior": [
                "d0-d5/ORD-94xx",
                "t15_*/v15_*/ORD-95xx/96xx",
                "c21_*/ORD-97xx",
                "u0-u3",
                "v0-v3",
                "s0-s15",
                "p0-p7",
                "e0-e3",
            ],
        },
    )
    return payload


def run_collection(pinned: dict[str, Any]) -> dict[str, Any]:
    import agenttune.agentic.rollout_engines.rollout_factory as factory

    coll = OUT / "collection"
    coll.mkdir(parents=True, exist_ok=True)
    _bind_output(coll)

    engine, tokenizer, loaded = _load_model()
    turn_engine = _TurnIdEngine(engine, tokenizer)
    _guards, schema_list = make_guarded_tools(
        RefundEpisode("schema-probe", "ORD-9801", "CUS-K01", 1520, config.SEED)
    )
    probe = _probe(tokenizer, schema_list)

    all_steps: list[dict] = []
    pair_rows: list[dict[str, Any]] = []
    fail_after_submit: list[dict[str, Any]] = []
    success_posts: list[dict[str, Any]] = []
    n_done = 0

    for episode_id, order_id, customer_id, amount in COLLECT_522_TASKS:
        for seed in COLLECT_522_SEEDS:
            if n_done >= MAX_EPISODES:
                break
            rid = f"{episode_id}-g{seed}"
            print(f"5.22 collect {rid}", flush=True)
            turn_engine.calls = []
            episode = RefundEpisode(
                episode_id=episode_id,
                order_id=order_id,
                customer_id=customer_id,
                amount_cents=amount,
                seed=config.ONE_ACTION_ENVIRONMENT_SEED,
            )
            prompt = collect_522_user_prompt(episode_id, order_id, customer_id)
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
                is_tool_after = (not scored["task_success_answer"]) and (
                    scored.get("error_type") == "final_answer_tool_call_after_submit"
                    or score_final(post["generated_raw"], tool_amount)["looks_like_tool_call"]
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


def _combined_with_521(agg_522: dict[str, Any]) -> dict[str, Any]:
    agg_521 = json.loads((SRC_521 / "aggregates.json").read_text(encoding="utf-8"))
    q21 = set(agg_521.get("queries_with_tool_call_after_submit") or [])
    q22 = set(agg_522.get("queries_with_tool_call_after_submit") or [])
    ids21 = list(agg_521.get("tool_call_after_submit_rollout_ids") or [])
    ids22 = list(agg_522.get("tool_call_after_submit_rollout_ids") or [])
    return {
        "task_5_21": {
            "confirmed_returns": agg_521["confirmed_returns"],
            "tool_call_after_submit_count": agg_521["tool_call_after_submit_count"],
            "queries_with_tool_call_after_submit": sorted(q21),
            "tool_call_after_submit_rollout_ids": ids21,
        },
        "task_5_22": {
            "confirmed_returns": agg_522["confirmed_returns"],
            "tool_call_after_submit_count": agg_522["tool_call_after_submit_count"],
            "queries_with_tool_call_after_submit": sorted(q22),
            "tool_call_after_submit_rollout_ids": ids22,
        },
        "combined": {
            "confirmed_returns": agg_521["confirmed_returns"] + agg_522["confirmed_returns"],
            "tool_call_after_submit_count": (
                agg_521["tool_call_after_submit_count"] + agg_522["tool_call_after_submit_count"]
            ),
            "unique_queries_with_tool_call_after_submit": sorted(q21 | q22),
            "n_unique_queries_with_tool_call_after_submit": len(q21 | q22),
            "tool_call_after_submit_rollout_ids": ids21 + ids22,
            "new_fail_rollout_ids_in_5_22": ids22,
            "new_fail_queries_in_5_22": sorted(q22 - q21),
        },
    }


def _write_report(summary: dict[str, Any]) -> None:
    agg = summary["aggregates"]
    comb = summary["combined_5_21_plus_5_22"]
    lines = [
        "# Task 5.22 — continue post-submit train collection (4×2)",
        "",
        "Untrained Tiny Aya. Same generation/scoring/capture format as 5.21. No training.",
        "",
        "## 5.22 only",
        "",
        f"- confirmed_returns: **{agg['confirmed_returns']}**/{agg['n_episodes']}",
        f"- tool_call_after_submit: **{agg['tool_call_after_submit_count']}** "
        f"(`{agg['tool_call_after_submit_rollout_ids']}`)",
        f"- queries with that failure: **{agg['n_queries_with_tool_call_after_submit']}** "
        f"(`{agg['queries_with_tool_call_after_submit']}`)",
        f"- successful_post_submit_contexts: {agg['successful_post_submit_contexts']}",
        f"- verification_all_ok: {agg['verification_all_ok']}",
        "",
        "## Combined 5.21 + 5.22 (unique queries with tool-call-after-submit)",
        "",
        f"- unique queries: **{comb['combined']['n_unique_queries_with_tool_call_after_submit']}** "
        f"(`{comb['combined']['unique_queries_with_tool_call_after_submit']}`)",
        f"- total tool_call_after_submit episodes: "
        f"**{comb['combined']['tool_call_after_submit_count']}**",
        f"- new fail IDs in 5.22: `{comb['combined']['new_fail_rollout_ids_in_5_22']}`",
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
        f"5.22 pinned matrix={pinned['matrix']['episode_ids']} "
        f"seeds={pinned['matrix']['generation_seeds']}",
        flush=True,
    )
    collected = run_collection(pinned)
    combined = _combined_with_521(collected["aggregates"])
    summary = {
        "kind": "collect_post_submit_5_22",
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
        "combined_5_21_plus_5_22": combined,
    }
    _write_json(OUT / "summary.json", summary)
    _write_json(OUT / "aggregates.json", collected["aggregates"])
    _write_json(OUT / "combined_with_521.json", combined)
    _write_report(summary)
    print(json.dumps(collected["aggregates"], indent=2, ensure_ascii=False), flush=True)
    print(json.dumps(combined["combined"], indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
