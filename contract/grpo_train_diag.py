"""Diagnose saved train groups t0–t3. No optimizer step."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contract import config
from contract.events import events_from_steps
from contract.grpo_prep import SEED_ORDER, TRAIN_TASKS, _write_json
from contract.harness import (
    ONE_ACTION_SYSTEM_PROMPT,
    ONE_ACTION_TOOL_RESULT_FORMAT,
    ONE_ACTION_TOOLS_FALLBACK_PROMPT,
    RefundEpisode,
    one_action_user_prompt,
)
from contract.rewards import reward_B, reward_C
from contract.run import _bind_output, _dump, _load_model, _one_model_rollout, _probe, make_guarded_tools

OUT = ROOT / "artifacts" / "task-5-6"
SAVED_T0 = ROOT / "artifacts" / "task-5-1" / "live_t0_trace.json"
MISSING_TASKS = tuple(task for task in TRAIN_TASKS if task[0] != "t0")
COMMAND = "TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 .venv/bin/python run_grpo_train_diag.py"


def _advantages(rewards: list[float]) -> list[float]:
    import torch
    from trl.trainer.utils import nanstd

    values = torch.tensor(rewards, dtype=torch.float32)
    mean = torch.nanmean(values)
    std = nanstd(values)
    advantages = (values - mean) / (std + 1e-4)
    return [float(item) for item in advantages.tolist()]


def _tool_call_text(text: str | None) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if not stripped.startswith("["):
        return False
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return '"tool_name"' in stripped or '"name"' in stripped
    if not isinstance(value, list) or not value or not isinstance(value[0], dict):
        return False
    return "tool_name" in value[0] or "name" in value[0]


def _error_type(row: dict[str, Any]) -> str:
    if row["calls_after_confirmed_submit"]:
        return "call_after_confirmed_refund"
    if row["task_success_harness"] and row["task_success_answer"] and row["n_invalid"] == 0:
        return "none"
    if row["task_success_harness"] and not row["task_success_answer"]:
        if row["final_answer_is_tool_call"]:
            return "final_answer_tool_call_after_submit"
        return "final_answer_not_accepted"
    if row["n_invalid"]:
        names = [
            name
            for name in ("malformed_syntax", "missing_name", "unknown_tool", "schema_invalid")
            if row["n_invalid_components"].get(name)
        ]
        return "invalid:" + "+".join(names or ["n_invalid"])
    if not row["submit_confirmed"]:
        return "submit_refund_not_confirmed"
    return "task_success_false"


def _calls_after_confirmed(steps: list[dict]) -> list[str]:
    confirmed = False
    later: list[str] = []
    for step in steps:
        for call in step.get("calls") or []:
            name = call.get("tool") or call.get("name") or ""
            result = call.get("structured_result") or {}
            if confirmed and name:
                later.append(str(name))
            if name == "submit_refund" and (
                result.get("submitted") is True or call.get("execution") == "success" and result.get("ok") is True
            ):
                confirmed = True
    return later


def _row_from_steps(steps: list[dict], *, source: str) -> dict[str, Any]:
    last = steps[-1]
    events = events_from_steps(steps, rollout_id=last.get("rollout_id"), source=source)
    later = _calls_after_confirmed(steps)
    entries = list(last.get("function_entries") or [])
    submit_at = [i for i, name in enumerate(entries) if name == "submit_refund"]
    entries_after = entries[submit_at[-1] + 1 :] if submit_at and last.get("task_success_harness") else []
    row = {
        "rollout_id": last.get("rollout_id"),
        "episode_id": last.get("episode_id"),
        "generation_seed": last.get("generation_seed"),
        "source": source,
        "task_success_harness": bool(last.get("task_success_harness")),
        "task_success_answer": bool(last.get("task_success_answer")),
        "task_success": bool(events["task_success"]),
        "n_invalid": int(events["n_invalid"]),
        "n_invalid_components": events["n_invalid_components"],
        "final_answer": events["final_answer"],
        "final_answer_is_tool_call": _tool_call_text(events["final_answer"]),
        "function_entries": entries,
        "entries_after_submit": entries_after,
        "calls_after_confirmed_submit": later or entries_after,
        "submit_confirmed": bool(last.get("task_success_harness")),
        "parsed_calls": events["parsed_call_count"],
    }
    row["error_type"] = _error_type(row)
    row["B"] = reward_B(completions=[row["final_answer"]], contract_events=[events])[0]
    row["C"] = reward_C(completions=[row["final_answer"]], contract_events=[events])[0]
    return row


def _row_from_saved_t0(index: int, saved: dict) -> dict[str, Any]:
    item = saved["correspondence"][index]
    state = saved["callback_trace"]["calculate_rewards_invocations"][0]["harness_states"][index]
    entries = list(state.get("function_entries") or [])
    submit_at = [i for i, name in enumerate(entries) if name == "submit_refund"]
    entries_after = entries[submit_at[-1] + 1 :] if submit_at and state.get("task_success_harness") else []
    final_answer = item.get("final_answer")
    row = {
        "rollout_id": item["rollout_id"],
        "episode_id": "t0",
        "generation_seed": item["generation_seed"],
        "source": "artifacts/task-5-1/live_t0_trace.json",
        "task_success_harness": bool(state.get("task_success_harness")),
        "task_success_answer": bool(state.get("task_success_answer")),
        "task_success": bool(item.get("task_success")),
        "n_invalid": int(item.get("n_invalid") or 0),
        "n_invalid_components": {
            "malformed_syntax": 0,
            "missing_name": 0,
            "unknown_tool": 0,
            "schema_invalid": 0,
            "N_invalid_provisional": int(item.get("n_invalid") or 0),
        },
        "final_answer": final_answer,
        "final_answer_is_tool_call": _tool_call_text(final_answer),
        "function_entries": entries,
        "entries_after_submit": entries_after,
        "calls_after_confirmed_submit": entries_after,
        "submit_confirmed": bool(state.get("submitted")),
        "parsed_calls": item.get("validation_n"),
    }
    row["error_type"] = _error_type(row)
    row["B"] = float(item["B"])
    row["C"] = float(item["C"])
    return row


def _load_saved_t0() -> list[dict[str, Any]]:
    saved = json.loads(SAVED_T0.read_text(encoding="utf-8"))
    seeds = list(saved.get("generation_seeds") or [])
    if seeds != list(SEED_ORDER):
        raise RuntimeError(f"saved t0 seeds {seeds} are not {list(SEED_ORDER)}")
    rows = [_row_from_saved_t0(i, saved) for i in range(len(SEED_ORDER))]
    if [row["rollout_id"] for row in rows] != [f"t0-g{seed}" for seed in SEED_ORDER]:
        raise RuntimeError("saved t0 rollout ids are not seeds 100–103 in order")
    return rows


def _roll_missing() -> list[dict[str, Any]]:
    import agenttune.agentic.rollout_engines.rollout_factory as factory

    out = OUT / "missing_train"
    out.mkdir(parents=True, exist_ok=True)
    _bind_output(out)
    engine, tokenizer, _loaded = _load_model()
    _guards, schema_list = make_guarded_tools(
        RefundEpisode("schema-probe", "ORD-2002", "CUS-602", 2200, config.SEED)
    )
    probe = _probe(tokenizer, schema_list)
    rows: list[dict] = []
    diagnosed: list[dict[str, Any]] = []
    allowed = {task[0] for task in MISSING_TASKS}
    if allowed & {task[0] for task in (("h0",), ("h1",), ("h2",), ("h3",))} or "t0" in allowed:
        raise RuntimeError(f"diagnostic pass is limited to missing train tasks, got {sorted(allowed)}")
    for episode_id, order_id, customer_id, amount in MISSING_TASKS:
        for seed in SEED_ORDER:
            rid = f"{episode_id}-g{seed}"
            print(f"diag {rid}", flush=True)
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
                prompt_text=one_action_user_prompt(order_id, customer_id),
                prompt_protocol="one_action_per_turn",
            )
            if limits:
                print("limits", limits, flush=True)
            rows.extend(ep_rows)
            _dump(out / "model_episodes.jsonl", rows)
            diagnosed.append(_row_from_steps(ep_rows, source=str(out / "model_episodes.jsonl")))
    return diagnosed


def _group(episode_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["generation_seed"]))
    b_rewards = [float(row["B"]) for row in ordered]
    c_rewards = [float(row["C"]) for row in ordered]
    b_adv = _advantages(b_rewards)
    c_adv = _advantages(c_rewards)
    changed = any(left != right for left, right in zip(b_adv, c_adv))
    return {
        "episode_id": episode_id,
        "seeds": [row["generation_seed"] for row in ordered],
        "rows": ordered,
        "B": b_rewards,
        "C": c_rewards,
        "advantage_B": b_adv,
        "advantage_C": c_adv,
        "c_changes_advantage": changed,
        "calls_after_confirmed_submit": any(row["calls_after_confirmed_submit"] for row in ordered),
        "n_invalid": [row["n_invalid"] for row in ordered],
    }


def main(argv: list[str] | None = None) -> int:
    del argv
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "commands.txt").write_text(COMMAND + "\n", encoding="utf-8")
    saved_rows = _load_saved_t0()
    missing_rows = _roll_missing()
    by_task: dict[str, list[dict[str, Any]]] = {task[0]: [] for task in TRAIN_TASKS}
    for row in [*saved_rows, *missing_rows]:
        by_task[row["episode_id"]].append(row)
    groups = [_group(task[0], by_task[task[0]]) for task in TRAIN_TASKS]
    payload = {
        "kind": "train_group_diagnostic",
        "optimizer_steps": 0,
        "trained": False,
        "held_out_ran": False,
        "command": COMMAND,
        "saved_t0": str(SAVED_T0.relative_to(ROOT)),
        "missing_pass": [task[0] for task in MISSING_TASKS],
        "generation_seeds": list(SEED_ORDER),
        "groups": groups,
        "any_calls_after_confirmed_submit": any(group["calls_after_confirmed_submit"] for group in groups),
        "groups_where_c_changes_advantage": [
            group["episode_id"] for group in groups if group["c_changes_advantage"]
        ],
    }
    _write_json(OUT / "summary.json", payload)
    print(
        json.dumps(
            {
                "calls_after_confirmed_submit": payload["any_calls_after_confirmed_submit"],
                "c_changes_advantage": payload["groups_where_c_changes_advantage"],
                "groups": [
                    {
                        "task": group["episode_id"],
                        "error_types": [row["error_type"] for row in group["rows"]],
                        "n_invalid": group["n_invalid"],
                        "B": group["B"],
                        "C": group["C"],
                        "advantage_B": group["advantage_B"],
                        "advantage_C": group["advantage_C"],
                        "c_changes_advantage": group["c_changes_advantage"],
                    }
                    for group in groups
                ],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
