"""Task 5.25: expanded independent holdout eval of frozen 5.23 step_4.

Pins 12 new queries × {100,101} before any rollout. Runs untrained Tiny Aya and
artifacts/task-5-23/adapters/step_4 on the same matrix with 5.24/5.23/5.14
generation and scoring. No training. Also reports combined 5.24+5.25.
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
from contract.grpo_pair import _release
from contract.grpo_prep import _package_versions, _write_json
from contract.grpo_train_diag import _row_from_steps
from contract.harness import (
    ONE_ACTION_SYSTEM_PROMPT,
    ONE_ACTION_TOOL_RESULT_FORMAT,
    ONE_ACTION_TOOLS_FALLBACK_PROMPT,
    RefundEpisode,
)
from contract.holdout_prompts_525 import (
    HOLDOUT_525_SEEDS,
    HOLDOUT_525_TASKS,
    assert_holdout_525_split,
    holdout_525_user_prompt,
)
from contract.run import (
    _bind_output,
    _dump,
    _one_model_rollout,
    _probe,
    make_guarded_tools,
)
from contract.sft_eval_520 import _load_engine
from contract.sft_pilot import _calls_after_submit

OUT = ROOT / "artifacts" / "task-5-25"
PRIOR_524 = ROOT / "artifacts" / "task-5-24"
ADAPTER = ROOT / "artifacts" / "task-5-23" / "adapters" / "step_4"
COMMAND_BASE = (
    "HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
    ".venv/bin/python run_holdout_eval_525.py"
)
MAX_EPISODES_PER_MODEL = 24


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _failure_stage(error_type: str) -> str:
    mapping = {
        "none": "none",
        "submit_refund_not_confirmed": "submit",
        "final_answer_tool_call_after_submit": "post_submit_final",
        "missing_final_answer": "final",
        "wrong_final_answer": "final",
        "invalid_tool_call": "tool_call",
        "tool_error": "tool_call",
        "max_steps": "horizon",
    }
    return mapping.get(error_type, f"other:{error_type}")


def write_pinned_config() -> dict[str, Any]:
    assert_holdout_525_split()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "datasets").mkdir(parents=True, exist_ok=True)
    (OUT / "base").mkdir(parents=True, exist_ok=True)
    (OUT / "step_4").mkdir(parents=True, exist_ok=True)
    rows = []
    for episode_id, order_id, customer_id, amount in HOLDOUT_525_TASKS:
        rows.append(
            {
                "episode_id": episode_id,
                "order_id": order_id,
                "customer_id": customer_id,
                "amount_cents": amount,
                "prompt": holdout_525_user_prompt(episode_id, order_id, customer_id),
                "environment_seed": config.ONE_ACTION_ENVIRONMENT_SEED,
                "generation_seeds": list(HOLDOUT_525_SEEDS),
                "prompt_style": "realistic_paraphrase_no_tool_sequence",
            }
        )
    with (OUT / "datasets" / "holdout_matrix.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(HOLDOUT_525_TASKS) * len(HOLDOUT_525_SEEDS)
    if n != MAX_EPISODES_PER_MODEL:
        raise RuntimeError(f"expected {MAX_EPISODES_PER_MODEL} episodes, got {n}")

    payload = {
        "kind": "holdout_eval_5_25",
        "written_before_run": True,
        "no_training": True,
        "no_query_selection_by_base_results": True,
        "command": COMMAND_BASE,
        "git_head": _git_head(),
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "models": {
            "base": "untrained Tiny Aya Global",
            "adapter": str(ADAPTER.relative_to(ROOT)),
        },
        "inherits_generation_scoring_from": "artifacts/task-5-24 / task-5-23 / task-5-14",
        "matrix": {
            "n_queries": len(HOLDOUT_525_TASKS),
            "episode_ids": [t[0] for t in HOLDOUT_525_TASKS],
            "order_ids": [t[1] for t in HOLDOUT_525_TASKS],
            "customer_ids": [t[2] for t in HOLDOUT_525_TASKS],
            "amounts_cents": [t[3] for t in HOLDOUT_525_TASKS],
            "generation_seeds": list(HOLDOUT_525_SEEDS),
            "n_episodes_per_model": n,
            "max_episodes_per_model": MAX_EPISODES_PER_MODEL,
        },
        "generation": {
            "settings": "same as task-5-24 / task-5-23 / task-5-14 one-action",
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
        "prior_holdout": "artifacts/task-5-24",
    }
    _write_json(OUT / "pinned_config.json", payload)
    (OUT / "commands.txt").write_text(COMMAND_BASE + "\n", encoding="utf-8")
    _write_json(
        OUT / "datasets" / "manifest.json",
        {
            "holdout": [t[0] for t in HOLDOUT_525_TASKS],
            "seeds": list(HOLDOUT_525_SEEDS),
            "excludes_prior": [
                "d0-d5/ORD-94xx",
                "t15_*/v15_*/ORD-95xx/96xx",
                "c21_*/ORD-97xx",
                "c22_*/ORD-98xx",
                "h24_*/ORD-9901-9908",
                "u0-u3",
                "v0-v3",
                "s0-s15",
                "p0-p7",
                "e0-e3",
            ],
        },
    )
    return payload


def run_holdout(adapter_dir: Path | None, *, out_dir: Path, label: str) -> dict[str, Any]:
    import agenttune.agentic.rollout_engines.rollout_factory as factory

    out_dir.mkdir(parents=True, exist_ok=True)
    engine, tokenizer, loaded = _load_engine(adapter_dir)
    _guards, schema_list = make_guarded_tools(
        RefundEpisode("schema-probe", "ORD-9911", "CUS-M01", 1920, config.SEED)
    )
    probe = _probe(tokenizer, schema_list)
    _bind_output(out_dir)
    all_steps: list[dict] = []
    pairs: list[dict[str, Any]] = []
    n_done = 0

    for episode_id, order_id, customer_id, amount in HOLDOUT_525_TASKS:
        for seed in HOLDOUT_525_SEEDS:
            if n_done >= MAX_EPISODES_PER_MODEL:
                break
            rid = f"{episode_id}-g{seed}"
            print(f"5.25[{label}] {rid}", flush=True)
            prompt = holdout_525_user_prompt(episode_id, order_id, customer_id)
            episode = RefundEpisode(
                episode_id=episode_id,
                order_id=order_id,
                customer_id=customer_id,
                amount_cents=amount,
                seed=config.ONE_ACTION_ENVIRONMENT_SEED,
            )
            ep_rows, limits = _one_model_rollout(
                factory,
                engine,
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
            _dump(out_dir / "model_episodes.jsonl", all_steps)
            scored = _row_from_steps(ep_rows, source=str(out_dir / "model_episodes.jsonl"))
            after = _calls_after_submit(ep_rows)
            if (
                scored.get("task_success_harness")
                and not scored.get("task_success_answer")
                and scored.get("error_type") == "final_answer_tool_call_after_submit"
            ):
                after = max(after, 1)
            pair = {
                "rollout_id": rid,
                "episode_id": episode_id,
                "generation_seed": seed,
                "order_id": order_id,
                "customer_id": customer_id,
                "amount_cents": amount,
                "prompt": prompt,
                "task_success": bool(scored["task_success"]),
                "confirmed": bool(scored["task_success_harness"]),
                "correct_final": bool(scored["task_success_answer"]),
                "error_type": scored["error_type"],
                "failure_stage": _failure_stage(str(scored["error_type"])),
                "n_invalid": int(scored.get("n_invalid") or 0),
                "n_calls_after_confirmed_submit": int(after),
                "final_answer": scored.get("final_answer"),
                "function_entries": scored.get("function_entries"),
            }
            pairs.append(pair)
            n_done += 1
            print(
                f"  success={pair['task_success']} confirmed={pair['confirmed']} "
                f"final={pair['correct_final']} invalid={pair['n_invalid']} "
                f"after={pair['n_calls_after_confirmed_submit']} err={pair['error_type']} "
                f"stage={pair['failure_stage']}",
                flush=True,
            )
        if n_done >= MAX_EPISODES_PER_MODEL:
            break

    aggregates = {
        "n": len(pairs),
        "full_success": sum(1 for p in pairs if p["task_success"]),
        "confirmed_returns": sum(1 for p in pairs if p["confirmed"]),
        "correct_finals": sum(1 for p in pairs if p["correct_final"]),
        "invalid_calls_total": sum(int(p["n_invalid"]) for p in pairs),
        "calls_after_confirmed_submit_total": sum(
            int(p["n_calls_after_confirmed_submit"]) for p in pairs
        ),
        "tool_call_after_submit_count": sum(
            1
            for p in pairs
            if p["error_type"] == "final_answer_tool_call_after_submit"
            or (
                p["confirmed"]
                and (not p["correct_final"])
                and int(p["n_calls_after_confirmed_submit"]) > 0
            )
        ),
        "error_type_counts": dict(Counter(p["error_type"] for p in pairs)),
        "failure_stage_counts": dict(Counter(p["failure_stage"] for p in pairs)),
    }
    _write_json(out_dir / "per_pair.json", {"pairs": pairs})
    _write_json(out_dir / "aggregates.json", aggregates)
    _write_json(
        out_dir / "traces.json",
        {
            "rollouts": {
                p["rollout_id"]: {
                    "task_success": p["task_success"],
                    "confirmed": p["confirmed"],
                    "correct_final": p["correct_final"],
                    "error_type": p["error_type"],
                    "failure_stage": p["failure_stage"],
                    "n_invalid": p["n_invalid"],
                    "n_calls_after_confirmed_submit": p["n_calls_after_confirmed_submit"],
                    "final_answer": p.get("final_answer"),
                    "function_entries": p.get("function_entries"),
                }
                for p in pairs
            }
        },
    )
    del engine
    _release()
    return {"label": label, "loaded": loaded, "aggregates": aggregates, "pairs": pairs}


def compare_holdout(base_pairs: list[dict[str, Any]], ad_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    base_by = {p["rollout_id"]: p for p in base_pairs}
    ad_by = {p["rollout_id"]: p for p in ad_pairs}
    rows = []
    fail_to_success = []
    success_to_fail = []
    base_after_submit_cases = []
    for rid in sorted(base_by):
        b = base_by[rid]
        a = ad_by[rid]
        if (not b["task_success"]) and a["task_success"]:
            transition = "fail_to_success"
            fail_to_success.append(
                {
                    "rollout_id": rid,
                    "base_error_type": b["error_type"],
                    "base_failure_stage": b.get("failure_stage") or _failure_stage(b["error_type"]),
                    "base_n_invalid": b["n_invalid"],
                    "adapter_n_invalid": a["n_invalid"],
                }
            )
        elif b["task_success"] and (not a["task_success"]):
            transition = "success_to_fail"
            success_to_fail.append(
                {
                    "rollout_id": rid,
                    "adapter_error_type": a["error_type"],
                    "adapter_failure_stage": a.get("failure_stage")
                    or _failure_stage(a["error_type"]),
                    "base_n_invalid": b["n_invalid"],
                    "adapter_n_invalid": a["n_invalid"],
                }
            )
        elif b["task_success"]:
            transition = "unchanged_success"
        else:
            transition = "unchanged_fail"
        row = {
            "rollout_id": rid,
            "transition": transition,
            "base": {
                "task_success": b["task_success"],
                "confirmed": b["confirmed"],
                "correct_final": b["correct_final"],
                "n_invalid": b["n_invalid"],
                "n_calls_after_confirmed_submit": b["n_calls_after_confirmed_submit"],
                "error_type": b["error_type"],
                "failure_stage": b.get("failure_stage") or _failure_stage(b["error_type"]),
            },
            "step_4": {
                "task_success": a["task_success"],
                "confirmed": a["confirmed"],
                "correct_final": a["correct_final"],
                "n_invalid": a["n_invalid"],
                "n_calls_after_confirmed_submit": a["n_calls_after_confirmed_submit"],
                "error_type": a["error_type"],
                "failure_stage": a.get("failure_stage") or _failure_stage(a["error_type"]),
            },
        }
        rows.append(row)
        if int(b["n_calls_after_confirmed_submit"]) > 0 or b["error_type"] == (
            "final_answer_tool_call_after_submit"
        ):
            base_after_submit_cases.append(
                {
                    "rollout_id": rid,
                    "base": row["base"],
                    "step_4": row["step_4"],
                    "transition": transition,
                }
            )

    def agg(pairs: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(pairs),
            "full_success": sum(1 for p in pairs if p["task_success"]),
            "confirmed_returns": sum(1 for p in pairs if p["confirmed"]),
            "correct_finals": sum(1 for p in pairs if p["correct_final"]),
            "invalid_calls_total": sum(int(p["n_invalid"]) for p in pairs),
            "calls_after_confirmed_submit_total": sum(
                int(p["n_calls_after_confirmed_submit"]) for p in pairs
            ),
            "tool_call_after_submit_count": sum(
                1 for p in pairs if p["error_type"] == "final_answer_tool_call_after_submit"
            ),
            "error_type_counts": dict(Counter(p["error_type"] for p in pairs)),
            "failure_stage_counts": dict(
                Counter(
                    p.get("failure_stage") or _failure_stage(p["error_type"]) for p in pairs
                )
            ),
        }

    base_agg = agg(base_pairs)
    ad_agg = agg(ad_pairs)
    criterion = {
        "at_least_one_fail_to_success": len(fail_to_success) >= 1,
        "no_success_to_fail": len(success_to_fail) == 0,
        "invalid_calls_do_not_increase": (
            ad_agg["invalid_calls_total"] <= base_agg["invalid_calls_total"]
        ),
    }
    criterion["passed"] = all(criterion.values())
    base_target_fails = base_agg["tool_call_after_submit_count"]
    return {
        "base_aggregates": base_agg,
        "step_4_aggregates": ad_agg,
        "per_episode": rows,
        "fail_to_success": fail_to_success,
        "success_to_fail": success_to_fail,
        "base_after_submit_cases": base_after_submit_cases,
        "transfer_criterion": criterion,
        "target_error_coverage": {
            "base_tool_call_after_submit_count": base_target_fails,
            "matrix_exercises_target_failure": base_target_fails > 0,
            "note": (
                None
                if base_target_fails > 0
                else (
                    "base had zero tool-call-after-submit failures on this matrix; "
                    "it does not test correction of the target error"
                )
            ),
        },
    }


def _load_524_comparison() -> dict[str, Any] | None:
    path = PRIOR_524 / "comparison.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _combine_with_524(comparison_525: dict[str, Any]) -> dict[str, Any]:
    prior = _load_524_comparison()
    if prior is None:
        return {"available": False, "note": "artifacts/task-5-24/comparison.json missing"}

    def merge_agg(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        err = Counter(a.get("error_type_counts") or {})
        err.update(b.get("error_type_counts") or {})
        return {
            "n": a["n"] + b["n"],
            "full_success": a["full_success"] + b["full_success"],
            "confirmed_returns": a["confirmed_returns"] + b["confirmed_returns"],
            "correct_finals": a["correct_finals"] + b["correct_finals"],
            "invalid_calls_total": a["invalid_calls_total"] + b["invalid_calls_total"],
            "calls_after_confirmed_submit_total": (
                a["calls_after_confirmed_submit_total"]
                + b["calls_after_confirmed_submit_total"]
            ),
            "tool_call_after_submit_count": (
                a["tool_call_after_submit_count"] + b["tool_call_after_submit_count"]
            ),
            "error_type_counts": dict(err),
        }

    base_c = merge_agg(prior["base_aggregates"], comparison_525["base_aggregates"])
    ad_c = merge_agg(prior["step_4_aggregates"], comparison_525["step_4_aggregates"])

    # normalize prior fail/success lists (5.24 stored bare ids)
    def _as_list(xs: list[Any]) -> list[str]:
        out = []
        for x in xs or []:
            out.append(x["rollout_id"] if isinstance(x, dict) else str(x))
        return out

    f2s = _as_list(prior.get("fail_to_success") or []) + [
        x["rollout_id"] for x in comparison_525["fail_to_success"]
    ]
    s2f = _as_list(prior.get("success_to_fail") or []) + [
        x["rollout_id"] for x in comparison_525["success_to_fail"]
    ]

    # rebuild after-submit cases from prior per_episode + current
    after_cases = []
    for row in prior.get("per_episode") or []:
        b = row["base"]
        if int(b.get("n_calls_after_confirmed_submit") or 0) > 0 or b.get("error_type") == (
            "final_answer_tool_call_after_submit"
        ):
            after_cases.append(
                {
                    "source": "5.24",
                    "rollout_id": row["rollout_id"],
                    "base": b,
                    "step_4": row["step_4"],
                    "transition": row["transition"],
                }
            )
    for case in comparison_525["base_after_submit_cases"]:
        after_cases.append({"source": "5.25", **case})

    criterion = {
        "at_least_one_fail_to_success": len(f2s) >= 1,
        "no_success_to_fail": len(s2f) == 0,
        "invalid_calls_do_not_increase": (
            ad_c["invalid_calls_total"] <= base_c["invalid_calls_total"]
        ),
    }
    criterion["passed"] = all(criterion.values())
    return {
        "available": True,
        "n_episodes_per_model": base_c["n"],
        "base_aggregates": base_c,
        "step_4_aggregates": ad_c,
        "fail_to_success": f2s,
        "success_to_fail": s2f,
        "base_after_submit_cases": after_cases,
        "transfer_criterion": criterion,
    }


def _write_report(
    summary: dict[str, Any], comparison: dict[str, Any], combined: dict[str, Any]
) -> None:
    ba = comparison["base_aggregates"]
    aa = comparison["step_4_aggregates"]
    lines = [
        "# Task 5.25 — expanded independent holdout eval (base vs frozen step_4)",
        "",
        "No training. 12 new queries × seeds `{100,101}` (24 episodes each). "
        "Same generation/scoring as 5.24. Matrix pinned before any rollout; "
        "no query selection by base results.",
        "",
        f"- adapter: `{summary['adapter']}`",
        f"- transfer_criterion (5.25): `{comparison['transfer_criterion']}`",
        f"- base tool-call-after-submit count: "
        f"**{comparison['target_error_coverage']['base_tool_call_after_submit_count']}**",
        f"- matrix exercises target failure: "
        f"`{comparison['target_error_coverage']['matrix_exercises_target_failure']}`",
        "",
    ]
    if comparison["target_error_coverage"]["note"]:
        lines.append(f"- note: {comparison['target_error_coverage']['note']}")
        lines.append("")

    lines.extend(
        [
            "## Aggregates 5.25 (base → step_4)",
            "",
            "| metric | base | step_4 |",
            "|---|---:|---:|",
            f"| full_success | {ba['full_success']}/{ba['n']} | {aa['full_success']}/{aa['n']} |",
            f"| confirmed_returns | {ba['confirmed_returns']} | {aa['confirmed_returns']} |",
            f"| correct_finals | {ba['correct_finals']} | {aa['correct_finals']} |",
            f"| invalid_calls_total | {ba['invalid_calls_total']} | {aa['invalid_calls_total']} |",
            f"| calls_after_confirmed_submit_total | "
            f"{ba['calls_after_confirmed_submit_total']} | "
            f"{aa['calls_after_confirmed_submit_total']} |",
            f"| tool_call_after_submit_count | "
            f"{ba['tool_call_after_submit_count']} | "
            f"{aa['tool_call_after_submit_count']} |",
            "",
            f"- fail→success: `{comparison['fail_to_success']}`",
            f"- success→fail: `{comparison['success_to_fail']}`",
            "",
            "## Base after-submit cases (adapter outcome)",
            "",
        ]
    )
    if not comparison["base_after_submit_cases"]:
        lines.append("_None on 5.25 matrix._")
        lines.append("")
    else:
        lines.append(
            "| rollout | base after | base err | base stage | step_4 ok | "
            "ad after | ad err | ad stage | transition |"
        )
        lines.append("|---|---:|---|---|---|---:|---|---|---|")
        for case in comparison["base_after_submit_cases"]:
            b, a = case["base"], case["step_4"]
            lines.append(
                f"| {case['rollout_id']} | {b['n_calls_after_confirmed_submit']} | "
                f"{b['error_type']} | {b['failure_stage']} | {a['task_success']} | "
                f"{a['n_calls_after_confirmed_submit']} | {a['error_type']} | "
                f"{a['failure_stage']} | {case['transition']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Paired outcomes 5.25",
            "",
            "| rollout | base ok | step_4 ok | base conf | ad conf | base final | ad final | "
            "base inv | ad inv | base after | ad after | base err | ad err | "
            "base stage | ad stage | transition |",
            "|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---|---|---|",
        ]
    )
    for row in comparison["per_episode"]:
        b, a = row["base"], row["step_4"]
        lines.append(
            f"| {row['rollout_id']} | {b['task_success']} | {a['task_success']} | "
            f"{b['confirmed']} | {a['confirmed']} | "
            f"{b['correct_final']} | {a['correct_final']} | "
            f"{b['n_invalid']} | {a['n_invalid']} | "
            f"{b['n_calls_after_confirmed_submit']} | {a['n_calls_after_confirmed_submit']} | "
            f"{b['error_type']} | {a['error_type']} | "
            f"{b['failure_stage']} | {a['failure_stage']} | {row['transition']} |"
        )
    lines.append("")

    if combined.get("available"):
        cb = combined["base_aggregates"]
        ca = combined["step_4_aggregates"]
        lines.extend(
            [
                "## Combined 5.24 + 5.25",
                "",
                f"- episodes/model: {combined['n_episodes_per_model']}",
                f"- transfer_criterion (combined): `{combined['transfer_criterion']}`",
                "",
                "| metric | base | step_4 |",
                "|---|---:|---:|",
                f"| full_success | {cb['full_success']}/{cb['n']} | "
                f"{ca['full_success']}/{ca['n']} |",
                f"| confirmed_returns | {cb['confirmed_returns']} | {ca['confirmed_returns']} |",
                f"| correct_finals | {cb['correct_finals']} | {ca['correct_finals']} |",
                f"| invalid_calls_total | {cb['invalid_calls_total']} | "
                f"{ca['invalid_calls_total']} |",
                f"| calls_after_confirmed_submit_total | "
                f"{cb['calls_after_confirmed_submit_total']} | "
                f"{ca['calls_after_confirmed_submit_total']} |",
                f"| tool_call_after_submit_count | "
                f"{cb['tool_call_after_submit_count']} | "
                f"{ca['tool_call_after_submit_count']} |",
                "",
                f"- fail→success (combined): `{combined['fail_to_success']}`",
                f"- success→fail (combined): `{combined['success_to_fail']}`",
                "",
                "### After-submit cases across 5.24+5.25",
                "",
                "| source | rollout | base after | base err | step_4 ok | ad after | "
                "ad err | transition |",
                "|---|---|---:|---|---|---:|---|---|",
            ]
        )
        for case in combined.get("base_after_submit_cases") or []:
            b, a = case["base"], case["step_4"]
            lines.append(
                f"| {case.get('source')} | {case['rollout_id']} | "
                f"{b.get('n_calls_after_confirmed_submit')} | {b.get('error_type')} | "
                f"{a.get('task_success')} | {a.get('n_calls_after_confirmed_submit')} | "
                f"{a.get('error_type')} | {case.get('transition')} |"
            )
        lines.append("")
    else:
        lines.append("## Combined 5.24 + 5.25")
        lines.append("")
        lines.append(f"_Unavailable: {combined.get('note')}_")
        lines.append("")

    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not ADAPTER.exists():
        raise FileNotFoundError(ADAPTER)
    started = time.time()
    pinned = write_pinned_config()
    print(
        f"5.25 pinned holdout={pinned['matrix']['episode_ids']} "
        f"seeds={pinned['matrix']['generation_seeds']}",
        flush=True,
    )

    print("5.25 running base on holdout matrix", flush=True)
    base_run = run_holdout(None, out_dir=OUT / "base", label="base")
    print("5.25 running step_4 on holdout matrix", flush=True)
    ad_run = run_holdout(ADAPTER, out_dir=OUT / "step_4", label="step_4")

    comparison = compare_holdout(base_run["pairs"], ad_run["pairs"])
    combined = _combine_with_524(comparison)
    _write_json(OUT / "comparison.json", comparison)
    _write_json(OUT / "paired_outcomes.json", {"per_episode": comparison["per_episode"]})
    _write_json(OUT / "combined_with_524.json", combined)

    summary = {
        "kind": "holdout_eval_5_25",
        "no_training": True,
        "no_query_selection_by_base_results": True,
        "command": COMMAND_BASE,
        "git_head": _git_head(),
        "python": platform.python_version(),
        "versions": _package_versions(),
        "elapsed_s": round(time.time() - started, 1),
        "adapter": str(ADAPTER.relative_to(ROOT)),
        "pinned_config": pinned,
        "base_loaded": base_run["loaded"],
        "adapter_loaded": ad_run["loaded"],
        "base_aggregates": comparison["base_aggregates"],
        "step_4_aggregates": comparison["step_4_aggregates"],
        "fail_to_success": comparison["fail_to_success"],
        "success_to_fail": comparison["success_to_fail"],
        "base_after_submit_cases": comparison["base_after_submit_cases"],
        "transfer_criterion": comparison["transfer_criterion"],
        "target_error_coverage": comparison["target_error_coverage"],
        "combined_with_524": combined,
    }
    _write_json(OUT / "summary.json", summary)
    _write_report(summary, comparison, combined)
    print(
        json.dumps(
            {
                "transfer_criterion": comparison["transfer_criterion"],
                "base": comparison["base_aggregates"],
                "step_4": comparison["step_4_aggregates"],
                "fail_to_success": comparison["fail_to_success"],
                "success_to_fail": comparison["success_to_fail"],
                "base_after_submit_cases": comparison["base_after_submit_cases"],
                "combined_with_524": {
                    "available": combined.get("available"),
                    "transfer_criterion": combined.get("transfer_criterion"),
                    "base": combined.get("base_aggregates"),
                    "step_4": combined.get("step_4_aggregates"),
                    "fail_to_success": combined.get("fail_to_success"),
                    "success_to_fail": combined.get("success_to_fail"),
                    "base_after_submit_cases": combined.get("base_after_submit_cases"),
                },
                "target_error_coverage": comparison["target_error_coverage"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
