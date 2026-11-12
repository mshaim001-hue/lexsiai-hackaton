"""Task 5.24: independent holdout episode eval of base vs 5.23 step_4.

Pins 8 new queries × {100,101} before any rollout. Runs untrained Tiny Aya and
artifacts/task-5-23/adapters/step_4 on the same matrix with 5.23/5.14 generation
and scoring. No training.
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
from contract.holdout_prompts_524 import (
    HOLDOUT_524_SEEDS,
    HOLDOUT_524_TASKS,
    assert_holdout_524_split,
    holdout_524_user_prompt,
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

OUT = ROOT / "artifacts" / "task-5-24"
ADAPTER = ROOT / "artifacts" / "task-5-23" / "adapters" / "step_4"
COMMAND_BASE = (
    "HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
    ".venv/bin/python run_holdout_eval_524.py"
)
MAX_EPISODES_PER_MODEL = 16


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def write_pinned_config() -> dict[str, Any]:
    assert_holdout_524_split()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "datasets").mkdir(parents=True, exist_ok=True)
    (OUT / "base").mkdir(parents=True, exist_ok=True)
    (OUT / "step_4").mkdir(parents=True, exist_ok=True)
    rows = []
    for episode_id, order_id, customer_id, amount in HOLDOUT_524_TASKS:
        rows.append(
            {
                "episode_id": episode_id,
                "order_id": order_id,
                "customer_id": customer_id,
                "amount_cents": amount,
                "prompt": holdout_524_user_prompt(episode_id, order_id, customer_id),
                "environment_seed": config.ONE_ACTION_ENVIRONMENT_SEED,
                "generation_seeds": list(HOLDOUT_524_SEEDS),
                "prompt_style": "realistic_paraphrase_no_tool_sequence",
            }
        )
    with (OUT / "datasets" / "holdout_matrix.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(HOLDOUT_524_TASKS) * len(HOLDOUT_524_SEEDS)
    if n != MAX_EPISODES_PER_MODEL:
        raise RuntimeError(f"expected {MAX_EPISODES_PER_MODEL} episodes, got {n}")

    payload = {
        "kind": "holdout_eval_5_24",
        "written_before_run": True,
        "no_training": True,
        "command": COMMAND_BASE,
        "git_head": _git_head(),
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "models": {
            "base": "untrained Tiny Aya Global",
            "adapter": str(ADAPTER.relative_to(ROOT)),
        },
        "inherits_generation_scoring_from": "artifacts/task-5-23 / task-5-14",
        "matrix": {
            "n_queries": len(HOLDOUT_524_TASKS),
            "episode_ids": [t[0] for t in HOLDOUT_524_TASKS],
            "order_ids": [t[1] for t in HOLDOUT_524_TASKS],
            "customer_ids": [t[2] for t in HOLDOUT_524_TASKS],
            "amounts_cents": [t[3] for t in HOLDOUT_524_TASKS],
            "generation_seeds": list(HOLDOUT_524_SEEDS),
            "n_episodes_per_model": n,
            "max_episodes_per_model": MAX_EPISODES_PER_MODEL,
        },
        "generation": {
            "settings": "same as task-5-23 / task-5-14 one-action",
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
        "transfer_criterion": {
            "at_least_one_fail_to_success": True,
            "no_success_to_fail": True,
            "invalid_calls_do_not_increase": True,
            "note_if_base_has_zero_tool_call_after_submit": (
                "matrix does not exercise the target after-submit failure mode"
            ),
        },
    }
    _write_json(OUT / "pinned_config.json", payload)
    (OUT / "commands.txt").write_text(COMMAND_BASE + "\n", encoding="utf-8")
    _write_json(
        OUT / "datasets" / "manifest.json",
        {
            "holdout": [t[0] for t in HOLDOUT_524_TASKS],
            "seeds": list(HOLDOUT_524_SEEDS),
            "excludes_prior": [
                "d0-d5/ORD-94xx",
                "t15_*/v15_*/ORD-95xx/96xx",
                "c21_*/ORD-97xx",
                "c22_*/ORD-98xx",
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
        RefundEpisode("schema-probe", "ORD-9901", "CUS-L01", 1470, config.SEED)
    )
    probe = _probe(tokenizer, schema_list)
    _bind_output(out_dir)
    all_steps: list[dict] = []
    pairs: list[dict[str, Any]] = []
    n_done = 0

    for episode_id, order_id, customer_id, amount in HOLDOUT_524_TASKS:
        for seed in HOLDOUT_524_SEEDS:
            if n_done >= MAX_EPISODES_PER_MODEL:
                break
            rid = f"{episode_id}-g{seed}"
            print(f"5.24[{label}] {rid}", flush=True)
            prompt = holdout_524_user_prompt(episode_id, order_id, customer_id)
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
                f"after={pair['n_calls_after_confirmed_submit']} err={pair['error_type']}",
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
    for rid in sorted(base_by):
        b = base_by[rid]
        a = ad_by[rid]
        if (not b["task_success"]) and a["task_success"]:
            transition = "fail_to_success"
            fail_to_success.append(rid)
        elif b["task_success"] and (not a["task_success"]):
            transition = "success_to_fail"
            success_to_fail.append(rid)
        elif b["task_success"]:
            transition = "unchanged_success"
        else:
            transition = "unchanged_fail"
        rows.append(
            {
                "rollout_id": rid,
                "transition": transition,
                "base": {
                    "task_success": b["task_success"],
                    "confirmed": b["confirmed"],
                    "correct_final": b["correct_final"],
                    "n_invalid": b["n_invalid"],
                    "n_calls_after_confirmed_submit": b["n_calls_after_confirmed_submit"],
                    "error_type": b["error_type"],
                },
                "step_4": {
                    "task_success": a["task_success"],
                    "confirmed": a["confirmed"],
                    "correct_final": a["correct_final"],
                    "n_invalid": a["n_invalid"],
                    "n_calls_after_confirmed_submit": a["n_calls_after_confirmed_submit"],
                    "error_type": a["error_type"],
                },
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


def _write_report(summary: dict[str, Any], comparison: dict[str, Any]) -> None:
    ba = comparison["base_aggregates"]
    aa = comparison["step_4_aggregates"]
    lines = [
        "# Task 5.24 — independent holdout eval (base vs 5.23 step_4)",
        "",
        "No training. 8 new queries × seeds `{100,101}` (16 episodes each).",
        "",
        f"- adapter: `{summary['adapter']}`",
        f"- transfer_criterion: `{comparison['transfer_criterion']}`",
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
            "## Aggregates (base → step_4)",
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
            "## Paired outcomes",
            "",
            "| rollout | base ok | step_4 ok | base conf | ad conf | base final | ad final | "
            "base inv | ad inv | base after | ad after | base err | ad err | transition |",
            "|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---|",
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
            f"{b['error_type']} | {a['error_type']} | {row['transition']} |"
        )
    lines.append("")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not ADAPTER.exists():
        raise FileNotFoundError(ADAPTER)
    started = time.time()
    pinned = write_pinned_config()
    print(
        f"5.24 pinned holdout={pinned['matrix']['episode_ids']} "
        f"seeds={pinned['matrix']['generation_seeds']}",
        flush=True,
    )

    print("5.24 running base on holdout matrix", flush=True)
    base_run = run_holdout(None, out_dir=OUT / "base", label="base")
    print("5.24 running step_4 on holdout matrix", flush=True)
    ad_run = run_holdout(ADAPTER, out_dir=OUT / "step_4", label="step_4")

    comparison = compare_holdout(base_run["pairs"], ad_run["pairs"])
    _write_json(OUT / "comparison.json", comparison)
    _write_json(OUT / "paired_outcomes.json", {"per_episode": comparison["per_episode"]})

    summary = {
        "kind": "holdout_eval_5_24",
        "no_training": True,
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
        "transfer_criterion": comparison["transfer_criterion"],
        "target_error_coverage": comparison["target_error_coverage"],
    }
    _write_json(OUT / "summary.json", summary)
    _write_report(summary, comparison)
    print(
        json.dumps(
            {
                "transfer_criterion": comparison["transfer_criterion"],
                "base": comparison["base_aggregates"],
                "step_4": comparison["step_4_aggregates"],
                "fail_to_success": comparison["fail_to_success"],
                "success_to_fail": comparison["success_to_fail"],
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
