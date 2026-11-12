"""Task 5.20: full 5.14 episode eval of 5.19 step_3 adapter.

Same prompts, seeds, generation settings, and scoring as task 5.14. Compares
against the saved 5.14 baseline when conditions match; otherwise re-runs base
under the same conditions as the adapter. No training.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contract import config
from contract.discovery_prompts import (
    DISCOVERY_TASKS,
    GENERATION_SEEDS,
    discovery_user_prompt,
)
from contract.grpo_pair import _release
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
    _one_model_rollout,
    _probe,
    make_guarded_tools,
)
from contract.sft_pilot import _calls_after_submit

OUT = ROOT / "artifacts" / "task-5-20"
BASELINE_514 = ROOT / "artifacts" / "task-5-14" / "baseline"
PINNED_514 = ROOT / "artifacts" / "task-5-14" / "pinned_config.json"
ADAPTER = ROOT / "artifacts" / "task-5-19" / "adapters" / "step_3"
COMMAND = (
    "HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
    ".venv/bin/python run_sft_eval_520.py"
)

FOCUS_IDS = ("d1-g101", "d2-g101", "d4-g100")


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _current_generation_settings() -> dict[str, Any]:
    return {
        "settings": "same as task-5-8 / task-5-14 one-action",
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
    }


def _matrix_spec() -> dict[str, Any]:
    return {
        "n_queries": len(DISCOVERY_TASKS),
        "episode_ids": [t[0] for t in DISCOVERY_TASKS],
        "order_ids": [t[1] for t in DISCOVERY_TASKS],
        "customer_ids": [t[2] for t in DISCOVERY_TASKS],
        "amounts_cents": [t[3] for t in DISCOVERY_TASKS],
        "generation_seeds": list(GENERATION_SEEDS),
        "n_episodes": len(DISCOVERY_TASKS) * len(GENERATION_SEEDS),
        "prompts": {
            eid: discovery_user_prompt(eid, oid, cid)
            for eid, oid, cid, _amt in DISCOVERY_TASKS
        },
    }


def check_conditions_vs_514() -> dict[str, Any]:
    pinned = json.loads(PINNED_514.read_text(encoding="utf-8"))
    gen_now = _current_generation_settings()
    gen_514 = pinned["generation"]
    gen_keys = [
        "environment_seed",
        "temperature",
        "top_p",
        "do_sample",
        "max_new_tokens",
        "max_steps",
        "protocol",
        "system_prompt",
        "tools_fallback_prompt",
        "tool_result_format",
        "user_prompt_style",
    ]
    gen_mismatches = {
        k: {"baseline_514": gen_514.get(k), "now": gen_now.get(k)}
        for k in gen_keys
        if gen_514.get(k) != gen_now.get(k)
    }
    disc_514 = pinned["discovery"]
    matrix = _matrix_spec()
    disc_mismatches = {}
    if disc_514.get("episode_ids") != matrix["episode_ids"]:
        disc_mismatches["episode_ids"] = {
            "baseline_514": disc_514.get("episode_ids"),
            "now": matrix["episode_ids"],
        }
    if disc_514.get("order_ids") != matrix["order_ids"]:
        disc_mismatches["order_ids"] = {
            "baseline_514": disc_514.get("order_ids"),
            "now": matrix["order_ids"],
        }
    if disc_514.get("generation_seeds") != matrix["generation_seeds"]:
        disc_mismatches["generation_seeds"] = {
            "baseline_514": disc_514.get("generation_seeds"),
            "now": matrix["generation_seeds"],
        }
    model_ok = (
        pinned.get("model_id") == config.MODEL_ID
        and pinned.get("model_revision") == config.MODEL_REVISION
    )
    prompt_mismatches = {}
    base_pairs = {
        p["rollout_id"]: p
        for p in json.loads((BASELINE_514 / "per_pair.json").read_text(encoding="utf-8"))[
            "pairs"
        ]
    }
    for eid, oid, cid, _amt in DISCOVERY_TASKS:
        expected = discovery_user_prompt(eid, oid, cid)
        # Compare against saved prompt on seed 100 row.
        rid = f"{eid}-g{GENERATION_SEEDS[0]}"
        saved = base_pairs[rid].get("prompt")
        if saved != expected:
            prompt_mismatches[eid] = {"baseline_514": saved, "now": expected}

    substantial = bool(gen_mismatches or disc_mismatches or prompt_mismatches or not model_ok)
    return {
        "model_match": model_ok,
        "generation_mismatches": gen_mismatches,
        "discovery_mismatches": disc_mismatches,
        "prompt_mismatches": prompt_mismatches,
        "substantial_difference": substantial,
        "comparison_source_if_match": "artifacts/task-5-14/baseline/per_pair.json",
        "action": (
            "re_run_base_under_adapter_conditions"
            if substantial
            else "use_saved_5_14_baseline"
        ),
    }


def write_pinned_config(condition_check: dict[str, Any]) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "traces").mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "sft_eval_5_20",
        "written_before_run": True,
        "no_training": True,
        "command": COMMAND,
        "adapter": str(ADAPTER.relative_to(ROOT)),
        "source_pilot": "artifacts/task-5-19",
        "eval_matrix": "artifacts/task-5-14 discovery d0–d5 × {100,101}",
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "discovery": _matrix_spec(),
        "generation": _current_generation_settings(),
        "scoring": {
            "full_success": "task_success",
            "confirmed_return": "task_success_harness",
            "correct_final": "task_success_answer",
            "invalid_calls": "n_invalid from _row_from_steps",
            "calls_after_confirmed_submit": "_calls_after_submit (5.9/5.16 normalization)",
        },
        "pass_criterion": {
            "at_least_one_prev_fail_becomes_full_success": True,
            "no_prev_success_breaks": True,
            "invalid_calls_do_not_increase": True,
        },
        "condition_check_vs_5_14": condition_check,
    }
    _write_json(OUT / "pinned_config.json", payload)
    (OUT / "commands.txt").write_text(COMMAND + "\n", encoding="utf-8")
    return payload


def _load_engine(adapter_dir: Path | None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from agenttune.agentic.rollout_engines.transformers_engine import TransformersRolloutEngine

    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID, revision=config.MODEL_REVISION, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        revision=config.MODEL_REVISION,
        dtype=torch.float16,
        local_files_only=True,
    )
    label = "base"
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        label = adapter_dir.name
    model.eval()
    if torch.backends.mps.is_available():
        model.to("mps")
        device = "mps"
    else:
        device = "cpu"
    engine = TransformersRolloutEngine(model, tokenizer)
    return engine, tokenizer, {
        "label": label,
        "adapter": None if adapter_dir is None else str(adapter_dir.relative_to(ROOT)),
        "device": device,
        "dtype": "float16",
    }


def run_matrix(adapter_dir: Path | None, *, out_dir: Path, label: str) -> dict[str, Any]:
    import agenttune.agentic.rollout_engines.rollout_factory as factory

    out_dir.mkdir(parents=True, exist_ok=True)
    engine, tokenizer, loaded = _load_engine(adapter_dir)
    _guards, schema_list = make_guarded_tools(
        RefundEpisode("schema-probe", "ORD-9401", "CUS-G01", 1620, config.SEED)
    )
    probe = _probe(tokenizer, schema_list)
    _bind_output(out_dir)
    all_steps: list[dict] = []
    pairs: list[dict[str, Any]] = []

    for episode_id, order_id, customer_id, amount in DISCOVERY_TASKS:
        for seed in GENERATION_SEEDS:
            rid = f"{episode_id}-g{seed}"
            print(f"5.20[{label}] {rid}", flush=True)
            prompt = discovery_user_prompt(episode_id, order_id, customer_id)
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
            # Normalize tool-call-after-submit like 5.16 when list empty but error says so.
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
            print(
                f"  success={pair['task_success']} confirmed={pair['confirmed']} "
                f"final={pair['correct_final']} invalid={pair['n_invalid']} "
                f"after={pair['n_calls_after_confirmed_submit']} err={pair['error_type']}",
                flush=True,
            )

    aggregates = {
        "n": len(pairs),
        "full_success": sum(1 for p in pairs if p["task_success"]),
        "confirmed_returns": sum(1 for p in pairs if p["confirmed"]),
        "correct_finals": sum(1 for p in pairs if p["correct_final"]),
        "invalid_calls_total": sum(int(p["n_invalid"]) for p in pairs),
        "calls_after_confirmed_submit_total": sum(
            int(p["n_calls_after_confirmed_submit"]) for p in pairs
        ),
        "error_type_counts": {},
    }
    from collections import Counter

    aggregates["error_type_counts"] = dict(Counter(p["error_type"] for p in pairs))
    payload = {
        "label": label,
        "loaded": loaded,
        "aggregates": aggregates,
        "pairs": pairs,
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
    return payload


def _baseline_pairs_from_514() -> list[dict[str, Any]]:
    raw = json.loads((BASELINE_514 / "per_pair.json").read_text(encoding="utf-8"))["pairs"]
    pairs = []
    for b in raw:
        after = len(b.get("calls_after_confirmed_submit") or [])
        if (
            b.get("task_success_harness")
            and not b.get("task_success_answer")
            and b.get("error_type") == "final_answer_tool_call_after_submit"
        ):
            after = max(after, 1)
        pairs.append(
            {
                "rollout_id": b["rollout_id"],
                "episode_id": b["episode_id"],
                "generation_seed": b["generation_seed"],
                "order_id": b.get("order_id"),
                "customer_id": b.get("customer_id"),
                "amount_cents": b.get("amount_cents"),
                "prompt": b.get("prompt"),
                "task_success": bool(b.get("task_success")),
                "confirmed": bool(b.get("task_success_harness")),
                "correct_final": bool(b.get("task_success_answer")),
                "error_type": b.get("error_type"),
                "n_invalid": int(b.get("n_invalid") or 0),
                "n_calls_after_confirmed_submit": after,
                "final_answer": b.get("final_answer"),
                "function_entries": b.get("function_entries"),
                "source": "artifacts/task-5-14/baseline/per_pair.json",
            }
        )
    return pairs


def _aggregates_from_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import Counter

    return {
        "n": len(pairs),
        "full_success": sum(1 for p in pairs if p["task_success"]),
        "confirmed_returns": sum(1 for p in pairs if p["confirmed"]),
        "correct_finals": sum(1 for p in pairs if p["correct_final"]),
        "invalid_calls_total": sum(int(p["n_invalid"]) for p in pairs),
        "calls_after_confirmed_submit_total": sum(
            int(p["n_calls_after_confirmed_submit"]) for p in pairs
        ),
        "error_type_counts": dict(Counter(p["error_type"] for p in pairs)),
    }


def compare(base_pairs: list[dict[str, Any]], adapter_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    base_by = {p["rollout_id"]: p for p in base_pairs}
    ad_by = {p["rollout_id"]: p for p in adapter_pairs}
    rows = []
    newly_fixed = []
    newly_broken = []
    for rid in sorted(base_by):
        b = base_by[rid]
        a = ad_by[rid]
        row = {
            "rollout_id": rid,
            "base": {
                "task_success": b["task_success"],
                "confirmed": b["confirmed"],
                "correct_final": b["correct_final"],
                "n_invalid": b["n_invalid"],
                "n_calls_after_confirmed_submit": b["n_calls_after_confirmed_submit"],
                "error_type": b["error_type"],
            },
            "adapter": {
                "task_success": a["task_success"],
                "confirmed": a["confirmed"],
                "correct_final": a["correct_final"],
                "n_invalid": a["n_invalid"],
                "n_calls_after_confirmed_submit": a["n_calls_after_confirmed_submit"],
                "error_type": a["error_type"],
            },
        }
        if (not b["task_success"]) and a["task_success"]:
            newly_fixed.append(rid)
            row["transition"] = "fail_to_success"
        elif b["task_success"] and (not a["task_success"]):
            newly_broken.append(rid)
            row["transition"] = "success_to_fail"
        elif b["task_success"] == a["task_success"]:
            row["transition"] = "unchanged_success" if b["task_success"] else "unchanged_fail"
        else:
            row["transition"] = "other"
        rows.append(row)

    base_agg = _aggregates_from_pairs(base_pairs)
    ad_agg = _aggregates_from_pairs(adapter_pairs)
    criterion = {
        "at_least_one_prev_fail_becomes_full_success": len(newly_fixed) >= 1,
        "no_prev_success_breaks": len(newly_broken) == 0,
        "invalid_calls_do_not_increase": (
            ad_agg["invalid_calls_total"] <= base_agg["invalid_calls_total"]
        ),
    }
    criterion["passed"] = all(criterion.values())
    return {
        "base_aggregates": base_agg,
        "adapter_aggregates": ad_agg,
        "per_episode": rows,
        "newly_fixed_full_success": newly_fixed,
        "newly_broken_from_base_success": newly_broken,
        "focus": {rid: next(r for r in rows if r["rollout_id"] == rid) for rid in FOCUS_IDS},
        "pass_criterion": criterion,
    }


def _write_report(
    summary: dict[str, Any],
    base_pairs: list[dict[str, Any]],
    adapter_pairs: list[dict[str, Any]],
    comparison: dict[str, Any],
) -> None:
    ba = comparison["base_aggregates"]
    aa = comparison["adapter_aggregates"]
    lines = [
        "# Task 5.20 — full 5.14 episode eval of 5.19 `step_3`",
        "",
        "No training. Same discovery matrix / prompts / seeds / one-action generation "
        "as task 5.14.",
        "",
        f"- adapter: `{summary['adapter']}`",
        f"- comparison_source: `{summary['comparison_source']}`",
        f"- condition match vs 5.14: "
        f"`{not summary['condition_check']['substantial_difference']}`",
        f"- pass_criterion: `{comparison['pass_criterion']}`",
        "",
        "## Aggregates (base → step_3)",
        "",
        "| metric | base | step_3 |",
        "|---|---:|---:|",
        f"| full_success | {ba['full_success']}/{ba['n']} | {aa['full_success']}/{aa['n']} |",
        f"| confirmed_returns | {ba['confirmed_returns']} | {aa['confirmed_returns']} |",
        f"| correct_finals | {ba['correct_finals']} | {aa['correct_finals']} |",
        f"| invalid_calls_total | {ba['invalid_calls_total']} | {aa['invalid_calls_total']} |",
        f"| calls_after_confirmed_submit_total | "
        f"{ba['calls_after_confirmed_submit_total']} | "
        f"{aa['calls_after_confirmed_submit_total']} |",
        "",
        f"- newly fixed (fail→full success): `{comparison['newly_fixed_full_success']}`",
        f"- newly broken (success→fail): `{comparison['newly_broken_from_base_success']}`",
        "",
        "## Focus episodes",
        "",
        "| rollout | base success | base err | ad success | ad err | after base→ad |",
        "|---|---|---|---|---|---|",
    ]
    for rid in FOCUS_IDS:
        row = comparison["focus"][rid]
        b, a = row["base"], row["adapter"]
        lines.append(
            f"| {rid} | {b['task_success']} | {b['error_type']} | "
            f"{a['task_success']} | {a['error_type']} | "
            f"{b['n_calls_after_confirmed_submit']}→{a['n_calls_after_confirmed_submit']} |"
        )
    lines.extend(
        [
            "",
            "## Per episode",
            "",
            "| rollout | base ok | ad ok | base conf | ad conf | base final | ad final | "
            "base inv | ad inv | base after | ad after | base err | ad err | transition |",
            "|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in comparison["per_episode"]:
        b, a = row["base"], row["adapter"]
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
    condition_check = check_conditions_vs_514()
    pinned = write_pinned_config(condition_check)
    print(
        f"5.20 pinned; substantial_diff={condition_check['substantial_difference']} "
        f"action={condition_check['action']}",
        flush=True,
    )

    if condition_check["substantial_difference"]:
        print("5.20 re-running base under adapter conditions", flush=True)
        base_run = run_matrix(None, out_dir=OUT / "base_rerun", label="base")
        base_pairs = base_run["pairs"]
        comparison_source = "artifacts/task-5-20/base_rerun"
        base_loaded = base_run["loaded"]
    else:
        print("5.20 using saved 5.14 baseline for comparison", flush=True)
        base_pairs = _baseline_pairs_from_514()
        _write_json(OUT / "base_from_514.json", {"pairs": base_pairs})
        comparison_source = "artifacts/task-5-14/baseline/per_pair.json"
        base_loaded = {"label": "saved_5_14_baseline", "adapter": None}

    print("5.20 evaluating adapter step_3 on full 5.14 matrix", flush=True)
    adapter_run = run_matrix(ADAPTER, out_dir=OUT / "adapter_step_3", label="step_3")
    comparison = compare(base_pairs, adapter_run["pairs"])
    _write_json(OUT / "comparison.json", comparison)

    summary = {
        "kind": "sft_eval_5_20",
        "no_training": True,
        "command": COMMAND,
        "git_head": _git_head(),
        "python": platform.python_version(),
        "versions": _package_versions(),
        "elapsed_s": round(time.time() - started, 1),
        "adapter": str(ADAPTER.relative_to(ROOT)),
        "comparison_source": comparison_source,
        "condition_check": condition_check,
        "pinned_config": pinned,
        "base_loaded": base_loaded,
        "adapter_loaded": adapter_run["loaded"],
        "base_aggregates": comparison["base_aggregates"],
        "adapter_aggregates": comparison["adapter_aggregates"],
        "newly_fixed_full_success": comparison["newly_fixed_full_success"],
        "newly_broken_from_base_success": comparison["newly_broken_from_base_success"],
        "focus": comparison["focus"],
        "pass_criterion": comparison["pass_criterion"],
    }
    _write_json(OUT / "summary.json", summary)
    _write_report(summary, base_pairs, adapter_run["pairs"], comparison)
    print(
        json.dumps(
            {
                "pass_criterion": comparison["pass_criterion"],
                "base": comparison["base_aggregates"],
                "step_3": comparison["adapter_aggregates"],
                "newly_fixed": comparison["newly_fixed_full_success"],
                "newly_broken": comparison["newly_broken_from_base_success"],
                "focus": {
                    rid: {
                        "base": comparison["focus"][rid]["base"]["task_success"],
                        "step_3": comparison["focus"][rid]["adapter"]["task_success"],
                        "base_err": comparison["focus"][rid]["base"]["error_type"],
                        "ad_err": comparison["focus"][rid]["adapter"]["error_type"],
                    }
                    for rid in FOCUS_IDS
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
