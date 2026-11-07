"""Baseline A: untrained one-action held-out matrix h0–h3 × seeds 100–103."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contract import config
from contract.events import events_from_steps
from contract.grpo_prep import HELD_OUT_TASKS, SEED_ORDER, _write_json, held_out_metrics_from_rows
from contract.harness import (
    ONE_ACTION_SYSTEM_PROMPT,
    ONE_ACTION_TOOL_RESULT_FORMAT,
    ONE_ACTION_TOOLS_FALLBACK_PROMPT,
    RefundEpisode,
    one_action_user_prompt,
)
from contract.rewards import reward_B, reward_C
from contract.run import (
    _bind_output,
    _dump,
    _load_model,
    _one_model_rollout,
    _probe,
    make_guarded_tools,
)

OUT = ROOT / "artifacts" / "task-5-2" / "baseline_a"


def _reuse_seed100_rows() -> dict[str, list[dict]]:
    path = ROOT / "artifacts" / "task-5" / "held_out_baseline" / "model_episodes.jsonl"
    if not path.exists():
        return {}
    by: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("generation_seed") == 100 and row.get("episode_id") in {
            t[0] for t in HELD_OUT_TASKS
        }:
            by[row["rollout_id"]].append(row)
    return by


def main() -> int:
    import agenttune.agentic.rollout_engines.rollout_factory as factory

    OUT.mkdir(parents=True, exist_ok=True)
    _bind_output(OUT)
    reused = _reuse_seed100_rows()
    rows: list[dict] = []
    sources: dict[str, str] = {}

    # Keep reused seed-100 trajectories.
    for rid, steps in reused.items():
        rows.extend(steps)
        sources[rid] = "artifacts/task-5/held_out_baseline/model_episodes.jsonl"

    engine, tokenizer, loaded = _load_model()
    _probe_guards, schema_list = make_guarded_tools(
        RefundEpisode("schema-probe", "ORD-3001", "CUS-701", 1640, config.SEED)
    )
    probe = _probe(tokenizer, schema_list)

    for episode_id, order_id, customer_id, amount in HELD_OUT_TASKS:
        for seed in SEED_ORDER:
            rid = f"{episode_id}-g{seed}"
            if rid in sources:
                print(f"baseline_a reuse {rid}", flush=True)
                continue
            print(f"baseline_a {rid}", flush=True)
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
            sources[rid] = str(OUT / "model_episodes.jsonl")
            _dump(OUT / "model_episodes.jsonl", rows)

    _dump(OUT / "model_episodes.jsonl", rows)

    by_rollout: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_rollout[row["rollout_id"]].append(row)

    per_pair = []
    for episode_id, order_id, customer_id, amount in HELD_OUT_TASKS:
        for seed in SEED_ORDER:
            rid = f"{episode_id}-g{seed}"
            steps = by_rollout[rid]
            events = events_from_steps(steps, rollout_id=rid, source=sources[rid])
            b = reward_B(completions=[events["final_answer"]], contract_events=[events])[0]
            c = reward_C(completions=[events["final_answer"]], contract_events=[events])[0]
            per_pair.append(
                {
                    "rollout_id": rid,
                    "episode_id": episode_id,
                    "order_id": order_id,
                    "customer_id": customer_id,
                    "amount_cents": amount,
                    "generation_seed": seed,
                    "source": sources[rid],
                    "task_success": events["task_success"],
                    "n_invalid": events["n_invalid"],
                    "parsed_calls": events["parsed_call_count"],
                    "no_tool_call": events["no_tool_call"],
                    "B": b,
                    "C": c,
                    "final_answer": events["final_answer"],
                }
            )

    aggregates = held_out_metrics_from_rows(rows)
    b_vals = [p["B"] for p in per_pair]
    c_vals = [p["C"] for p in per_pair]
    manifest = {
        "kind": "baseline_A_untrained",
        "condition": "A",
        "formula_B": "I(task_success)",
        "formula_C": "I(task_success) - 0.5 * N_invalid",
        "lambda": config.LAMBDA,
        "protocol": "one_action_per_turn",
        "tasks": [t[0] for t in HELD_OUT_TASKS],
        "generation_seeds": list(SEED_ORDER),
        "n_rollouts": len(per_pair),
        "loaded_model": loaded,
        "per_pair": per_pair,
        "aggregates": {
            **aggregates,
            "B_mean": sum(b_vals) / len(b_vals),
            "C_mean": sum(c_vals) / len(c_vals),
            "B_values": b_vals,
            "C_values": c_vals,
            "at_ceiling_task_success": aggregates["task_success"]["rate"] == 1.0,
        },
        "optimizer_steps": 0,
        "weights_updated": False,
    }
    _write_json(OUT / "manifest.json", manifest)
    _write_json(OUT / "per_pair.json", {"pairs": per_pair})
    _write_json(OUT / "aggregates.json", manifest["aggregates"])
    print(
        json.dumps(
            {
                "n": len(per_pair),
                "task_success": aggregates["task_success"],
                "B_mean": manifest["aggregates"]["B_mean"],
                "C_mean": manifest["aggregates"]["C_mean"],
                "at_ceiling": manifest["aggregates"]["at_ceiling_task_success"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
