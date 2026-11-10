"""Task 5-8: pin a realistic train/test split, then evaluate untrained Tiny Aya on test."""

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
from contract.events import events_from_steps
from contract.grpo_prep import _package_versions, _write_json
from contract.grpo_train_diag import _row_from_steps
from contract.harness import (
    ONE_ACTION_SYSTEM_PROMPT,
    ONE_ACTION_TOOL_RESULT_FORMAT,
    ONE_ACTION_TOOLS_FALLBACK_PROMPT,
    RefundEpisode,
)
from contract.realistic_prompts import (
    GENERATION_SEEDS,
    TEST_TASKS,
    TRAIN_TASKS,
    realistic_user_prompt,
)
from contract.run import (
    _bind_output,
    _dump,
    _load_model,
    _one_model_rollout,
    _probe,
    make_guarded_tools,
)

OUT = ROOT / "artifacts" / "task-5-8"
DATASETS = OUT / "datasets"
BASELINE = OUT / "baseline"


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _row(episode_id: str, order_id: str, customer_id: str, amount_cents: int, split: str) -> dict:
    return {
        "episode_id": episode_id,
        "order_id": order_id,
        "customer_id": customer_id,
        "amount_cents": amount_cents,
        "split": split,
        "prompt": realistic_user_prompt(episode_id, order_id, customer_id),
        "environment_seed": config.ONE_ACTION_ENVIRONMENT_SEED,
        "prompt_style": "realistic_paraphrase_no_tool_sequence",
    }


def write_pinned_split() -> dict[str, Any]:
    """Write train/test split, seeds, and generation settings before any rollout."""
    OUT.mkdir(parents=True, exist_ok=True)
    DATASETS.mkdir(parents=True, exist_ok=True)

    train_rows = [_row(*task, "train") for task in TRAIN_TASKS]
    test_rows = [_row(*task, "test") for task in TEST_TASKS]
    train_ids = {row["episode_id"] for row in train_rows}
    test_ids = {row["episode_id"] for row in test_rows}
    if train_ids & test_ids:
        raise RuntimeError(f"train/test id overlap: {sorted(train_ids & test_ids)}")

    prior_order_ids = {
        "ORD-1001",
        "ORD-1002",
        "ORD-1003",
        "ORD-1004",
        "ORD-1005",
        "ORD-2001",
        "ORD-2002",
        "ORD-2003",
        "ORD-2004",
        "ORD-3001",
        "ORD-3002",
        "ORD-3003",
        "ORD-3004",
    }
    new_order_ids = {row["order_id"] for row in train_rows} | {row["order_id"] for row in test_rows}
    if new_order_ids & prior_order_ids:
        raise RuntimeError(f"order_id collision with prior splits: {sorted(new_order_ids & prior_order_ids)}")

    with (DATASETS / "train.jsonl").open("w", encoding="utf-8") as handle:
        for row in train_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (DATASETS / "test.jsonl").open("w", encoding="utf-8") as handle:
        for row in test_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    generation = {
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "environment_seed": config.ONE_ACTION_ENVIRONMENT_SEED,
        "generation_seeds": list(GENERATION_SEEDS),
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
        "condition": "A_untrained",
    }
    metrics_def = {
        "full_success": "task_success = submitted refund AND accepted final answer",
        "confirmed_returns": "task_success_harness = submit_refund confirmed by harness",
        "correct_finals": "task_success_answer = final answer is exactly REFUND_SUBMITTED amount_cents=<n>",
        "invalid_calls": (
            "N_invalid provisional = malformed_syntax + missing_name + unknown_tool + schema_invalid; "
            "denominator = parsed call attempts + malformed_syntax fragments"
        ),
    }
    manifest = {
        "kind": "realistic_refund_split_pinned_before_run",
        "git_head": _git_head(),
        "train": {
            "path": str(DATASETS / "train.jsonl"),
            "n": len(train_rows),
            "episode_ids": [row["episode_id"] for row in train_rows],
            "order_ids": [row["order_id"] for row in train_rows],
        },
        "test": {
            "path": str(DATASETS / "test.jsonl"),
            "n": len(test_rows),
            "episode_ids": [row["episode_id"] for row in test_rows],
            "order_ids": [row["order_id"] for row in test_rows],
        },
        "overlap": [],
        "prior_task5_comparison": "artifacts/task-5-5/summary.json",
        "generation": generation,
        "metrics": metrics_def,
        "matrix": {
            "test_episodes": [t[0] for t in TEST_TASKS],
            "generation_seeds": list(GENERATION_SEEDS),
            "n_rollouts": len(TEST_TASKS) * len(GENERATION_SEEDS),
        },
    }
    _write_json(DATASETS / "manifest.json", manifest)
    _write_json(OUT / "pinned_config.json", manifest)
    return manifest


def _attempt_denominator(steps: list[dict]) -> int:
    total = 0
    for step in steps:
        total += len(step.get("calls") or [])
        total += len((step.get("diagnostic") or {}).get("malformed_syntax") or [])
    return total


def _summary_from_rows(per_pair: list[dict], by_rollout: dict[str, list[dict]]) -> dict[str, Any]:
    n = len(per_pair)
    full = sum(1 for row in per_pair if row["task_success"])
    confirmed = sum(1 for row in per_pair if row["task_success_harness"])
    correct_finals = sum(1 for row in per_pair if row["task_success_answer"])
    invalid_num = sum(int(row["n_invalid"]) for row in per_pair)
    invalid_den = sum(_attempt_denominator(by_rollout[row["rollout_id"]]) for row in per_pair)
    error_counts = Counter(row["error_type"] for row in per_pair)
    failures = [row for row in per_pair if not row["task_success"]]
    failure_modes = Counter(row["error_type"] for row in failures)
    next_pilot = None
    if failure_modes:
        ranked = failure_modes.most_common()
        mode, count = ranked[0]
        # Prefer the Track 3 residual when frequencies tie: stop cleanly after submit.
        preferred = "final_answer_tool_call_after_submit"
        tied = [name for name, n_mode in ranked if n_mode == count]
        if preferred in tied:
            mode = preferred
            count = failure_modes[mode]
        tied_other = [name for name in tied if name != mode]
        next_pilot = {
            "failure_mode": mode,
            "count_on_test": count,
            "n_failures": len(failures),
            "rationale": (
                "Prefer final_answer_tool_call_after_submit when tied: it matches the Track 3 "
                "claim (confirm only after successful submit, then stop), already appeared as "
                "the coached 5.5 residual, and still occurs here after a completed four-tool "
                "sequence. Use separate train paraphrases u0–u3; keep this test fixed."
            ),
            "example_rollouts": [
                row["rollout_id"] for row in failures if row["error_type"] == mode
            ][:4],
        }
        if tied_other:
            other = tied_other[0]
            next_pilot["tied_with"] = {
                "failure_mode": other,
                "count_on_test": failure_modes[other],
                "example_rollouts": [
                    row["rollout_id"] for row in failures if row["error_type"] == other
                ][:4],
                "note": (
                    "Wrong or incomplete tool order so submit never confirms; "
                    "newly visible without sequence coaching."
                ),
            }
    elif n:
        next_pilot = {
            "failure_mode": "none_observed",
            "count_on_test": 0,
            "n_failures": 0,
            "rationale": (
                "Baseline already at ceiling on this realistic test; "
                "next pilot should target a harder train paraphrase or a held-out stress case."
            ),
            "example_rollouts": [],
        }
    return {
        "n_rollouts": n,
        "full_success": {
            "numerator": full,
            "denominator": n,
            "rate": (full / n) if n else None,
            "label": "task_success",
        },
        "confirmed_returns": {
            "numerator": confirmed,
            "denominator": n,
            "rate": (confirmed / n) if n else None,
            "label": "task_success_harness",
        },
        "correct_finals": {
            "numerator": correct_finals,
            "denominator": n,
            "rate": (correct_finals / n) if n else None,
            "label": "task_success_answer",
        },
        "invalid_calls": {
            "numerator": invalid_num,
            "denominator": invalid_den,
            "rate": (invalid_num / invalid_den) if invalid_den else None,
            "definition": (
                "numerator = provisional N_invalid "
                "(malformed_syntax + missing_name + unknown_tool + schema_invalid); "
                "denominator = parsed call attempts + malformed_syntax fragments"
            ),
        },
        "error_type_counts": dict(error_counts),
        "failure_mode_counts": dict(failure_modes),
        "next_pilot_failure_to_check": next_pilot,
        "calls_per_rollout_mean": (
            sum(row["parsed_calls"] for row in per_pair) / n if n else None
        ),
    }


def run_baseline(pinned: dict[str, Any]) -> dict[str, Any]:
    import agenttune.agentic.rollout_engines.rollout_factory as factory

    BASELINE.mkdir(parents=True, exist_ok=True)
    _bind_output(BASELINE)
    command = (
        "TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
        ".venv/bin/python run_realistic_baseline.py"
    )
    (OUT / "commands.txt").write_text(command + "\n", encoding="utf-8")

    engine, tokenizer, loaded = _load_model()
    _probe_guards, schema_list = make_guarded_tools(
        RefundEpisode("schema-probe", "ORD-5001", "CUS-901", 1520, config.SEED)
    )
    probe = _probe(tokenizer, schema_list)

    rows: list[dict] = []
    sources: dict[str, str] = {}
    for episode_id, order_id, customer_id, amount in TEST_TASKS:
        for seed in GENERATION_SEEDS:
            rid = f"{episode_id}-g{seed}"
            print(f"realistic_baseline {rid}", flush=True)
            episode = RefundEpisode(
                episode_id=episode_id,
                order_id=order_id,
                customer_id=customer_id,
                amount_cents=amount,
                seed=config.ONE_ACTION_ENVIRONMENT_SEED,
            )
            prompt = realistic_user_prompt(episode_id, order_id, customer_id)
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
            rows.extend(ep_rows)
            sources[rid] = str(BASELINE / "model_episodes.jsonl")
            _dump(BASELINE / "model_episodes.jsonl", rows)

    _dump(BASELINE / "model_episodes.jsonl", rows)

    by_rollout: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_rollout[row["rollout_id"]].append(row)

    per_pair = []
    for episode_id, order_id, customer_id, amount in TEST_TASKS:
        for seed in GENERATION_SEEDS:
            rid = f"{episode_id}-g{seed}"
            steps = by_rollout[rid]
            scored = _row_from_steps(steps, source=sources[rid])
            scored.update(
                {
                    "order_id": order_id,
                    "customer_id": customer_id,
                    "amount_cents": amount,
                    "prompt": realistic_user_prompt(episode_id, order_id, customer_id),
                }
            )
            per_pair.append(scored)

    metrics = _summary_from_rows(per_pair, by_rollout)
    summary = {
        "kind": "realistic_test_baseline_A",
        "command": command,
        "git_head": _git_head(),
        "python": platform.python_version(),
        "versions": _package_versions(),
        "loaded_model": loaded,
        "pinned_config": str(OUT / "pinned_config.json"),
        "datasets": pinned,
        "generation": pinned["generation"],
        "n_rollouts": len(per_pair),
        "per_pair": per_pair,
        "metrics": metrics,
        "comparison_baseline_task_5_5": "artifacts/task-5-5/summary.json",
        "note": (
            "task-5-5 remains the original h0–h3 comparison. "
            "This run uses a separate realistic paraphrase test set only."
        ),
    }
    _write_json(BASELINE / "per_pair.json", {"pairs": per_pair})
    _write_json(BASELINE / "aggregates.json", metrics)
    _write_json(BASELINE / "summary.json", summary)
    _write_json(OUT / "summary.json", summary)
    _write_json(
        BASELINE / "traces.json",
        {
            "rollouts": {
                rid: {
                    "steps": [
                        {
                            "step_id": step.get("step_id"),
                            "raw_output": step.get("raw_output"),
                            "calls": step.get("calls"),
                            "parse_status": step.get("parse_status"),
                            "generation_truncated": step.get("generation_truncated"),
                        }
                        for step in steps
                    ],
                    "final_answer": steps[-1].get("final_answer"),
                    "task_success": steps[-1].get("task_success"),
                    "task_success_harness": steps[-1].get("task_success_harness"),
                    "task_success_answer": steps[-1].get("task_success_answer"),
                    "function_entries": steps[-1].get("function_entries"),
                }
                for rid, steps in sorted(by_rollout.items())
            }
        },
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return summary


def main() -> int:
    pinned = write_pinned_split()
    print("pinned split written", flush=True)
    print(json.dumps({
        "train": pinned["train"]["episode_ids"],
        "test": pinned["test"]["episode_ids"],
        "seeds": pinned["generation"]["generation_seeds"],
        "temperature": pinned["generation"]["temperature"],
        "top_p": pinned["generation"]["top_p"],
    }, indent=2), flush=True)
    run_baseline(pinned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
