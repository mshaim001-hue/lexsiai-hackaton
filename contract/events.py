"""Contract events for the GRPO reward path.

The trainer reward callback must see raw generation text, per-call validation,
harness state, and the final answer. Those fields are assembled here so scripted
replay and live rollouts share one schema.
"""

from __future__ import annotations

from typing import Any


def n_invalid_components_from_steps(steps: list[dict]) -> dict[str, int | str]:
    malformed_syntax = 0
    omitted_calls = 0
    truncated_fragments = 0
    schema_invalid = 0
    unknown_tool = 0
    missing_name = 0
    generation_truncated = 0
    for step in steps:
        diagnostic = step.get("diagnostic") or {}
        malformed_syntax += len(diagnostic.get("malformed_syntax") or [])
        omitted_calls += sum(
            len(item.get("dropped_by_full_parse") or [])
            for item in diagnostic.get("omitted_by_native_parser") or []
        )
        truncated_fragments += len(diagnostic.get("truncated_fragments") or [])
        generation_truncated += int(bool(step.get("generation_truncated")))
        for call in step.get("calls") or []:
            validation = call.get("validation")
            schema_invalid += validation == "schema_invalid"
            unknown_tool += validation == "unknown_tool"
            missing_name += validation == "missing_name"
    provisional = malformed_syntax + schema_invalid + unknown_tool + missing_name
    return {
        "malformed_syntax": malformed_syntax,
        "omitted_by_native_parser": omitted_calls,
        "schema_invalid": schema_invalid,
        "unknown_tool": unknown_tool,
        "missing_name": missing_name,
        "truncated_fragment": truncated_fragments,
        "generation_truncated": generation_truncated,
        "N_invalid_provisional": provisional,
        "C_status": "preliminary",
    }


def events_from_steps(
    steps: list[dict],
    *,
    rollout_id: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Build the event bundle the GRPO reward callback consumes."""
    if not steps:
        raise ValueError("events_from_steps requires at least one step")
    last = steps[-1]
    components = n_invalid_components_from_steps(steps)
    validation_results = []
    for step in steps:
        for call in step.get("calls") or []:
            validation_results.append(
                {
                    "step_id": step.get("step_id"),
                    "tool": call.get("tool"),
                    "validation": call.get("validation"),
                    "execution": call.get("execution"),
                    "function_entered": call.get("function_entered"),
                    "problems": call.get("problems") or [],
                    "order_or_token": call.get("order_or_token"),
                }
            )
    return {
        "rollout_id": rollout_id or last.get("rollout_id") or last.get("episode_id"),
        "episode_id": last.get("episode_id"),
        "generation_seed": last.get("generation_seed"),
        "source": source,
        "raw_outputs": [step.get("raw_output") for step in steps],
        "validation_results": validation_results,
        "harness_state": {
            "submitted": bool(last.get("task_success_harness")),
            "function_entries": list(last.get("function_entries") or []),
            "task_success_harness": bool(last.get("task_success_harness")),
            "task_success_answer": bool(last.get("task_success_answer")),
        },
        "final_answer": last.get("final_answer"),
        "task_success": bool(last.get("task_success")),
        "n_invalid_components": components,
        "n_invalid": int(components["N_invalid_provisional"]),
        "parsed_call_count": sum(len(step.get("calls") or []) for step in steps),
        "no_tool_call": all(not (step.get("calls") or []) for step in steps),
        "callback_fields_present": {
            "raw_output": all(step.get("raw_output") is not None for step in steps),
            "validation_results": True,
            "harness_state": True,
            "final_answer": "final_answer" in last,
        },
    }


def attach_events_to_trajectory(trajectory, events: dict[str, Any]) -> None:
    """Put contract events on Trajectory.metadata for the trainer reward path."""
    metadata = dict(getattr(trajectory, "metadata", None) or {})
    metadata["contract_events"] = events
    trajectory.metadata = metadata


def extract_events_from_reward_kwargs(
    *,
    contract_events=None,
    trajectories=None,
    n: int,
) -> list[dict[str, Any]]:
    """Resolve event bundles from the kwargs TRL forwards to reward_funcs."""
    if contract_events is not None:
        if len(contract_events) != n:
            raise ValueError(
                f"contract_events length {len(contract_events)} != completions {n}"
            )
        missing = [i for i, item in enumerate(contract_events) if not item]
        if missing:
            raise ValueError(
                "trainer reward path lost contract_events at indices "
                f"{missing}; refuse to score without raw/validation/harness/final"
            )
        return list(contract_events)

    if trajectories is not None:
        out = []
        for i, traj in enumerate(trajectories):
            meta = getattr(traj, "metadata", None) or {}
            events = meta.get("contract_events")
            if not events:
                raise ValueError(
                    f"trajectory[{i}] has no metadata['contract_events']; "
                    "the rollout path dropped contract fields before the reward callback"
                )
            out.append(events)
        if len(out) != n:
            raise ValueError(f"trajectories length {len(out)} != completions {n}")
        return out

    raise ValueError(
        "reward callback received neither contract_events nor trajectories with "
        "contract_events; completions-only scoring is disabled for this contract"
    )
