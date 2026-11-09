"""B-only pilot: at most four optimizer steps on t0, then one held-out eval.

The budget, seed groups, and checkpoint rule are written before training.
Held-out starts only after the adapter from the last completed step is saved.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contract import config
from contract.events import events_from_steps
from contract.grpo_pair import _release, _save_adapter
from contract.grpo_path import _task_tuple, make_instrumented_rollout_func
from contract.grpo_prep import (
    HELD_OUT_TASKS,
    SEED_ORDER,
    _package_versions,
    _write_json,
    held_out_metrics_from_rows,
)
from contract.grpo_train_step import (
    _build_b_only_trainer,
    _hook_rewards,
    _lora_grad_norm,
    _lora_vector,
    _param_change,
)
from contract.grpo_turn_loss import IdenticalRewards, install_turn_scoring, pilot_old_logps_reason
from contract.harness import (
    ONE_ACTION_SYSTEM_PROMPT,
    ONE_ACTION_TOOL_RESULT_FORMAT,
    ONE_ACTION_TOOLS_FALLBACK_PROMPT,
    RefundEpisode,
    answer_matches,
    one_action_user_prompt,
)
from contract.rewards import reward_B
from contract.run import _bind_output, _dump, _one_model_rollout, _probe, make_guarded_tools
from transformers import TrainerCallback

OUT = ROOT / "artifacts" / "task-5-7"
A_EPISODES = ROOT / "artifacts" / "task-5-2" / "baseline_a" / "model_episodes.jsonl"
MAX_STEPS = 4
COMMAND = (
    "HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
    ".venv/bin/python run_grpo_b_steps.py"
)
CHECKPOINT_RULE = (
    "Save the adapter at the weights of the last completed optimizer step. "
    "A group whose four rewards are equal is recorded and does not complete a step. "
    "If no step completes, save the fresh LoRA. Do not choose among intermediate checkpoints, "
    "and do not use held-out to choose the checkpoint."
)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _budget() -> dict[str, Any]:
    seeds = list(SEED_ORDER)
    return {
        "kind": "b_only_budget",
        "written_before_training": True,
        "held_out_ran": False,
        "max_optimizer_steps": MAX_STEPS,
        "reward": "reward_B_I_task_success",
        "learning_rate": 1e-6,
        "lora_dropout": 0.0,
        "lora_init": "fresh",
        "lora_init_seed": config.SEED,
        "rollout_mode": "eval during live rollout; restore the previous mode before the loss",
        "scoring_mode": "train mode restored before the loss; lora_dropout 0 so scoring applies no dropout",
        "train_task": "t0",
        "train_groups": [
            {"optimizer_step": step, "task": "t0", "generation_seeds": seeds}
            for step in range(1, MAX_STEPS + 1)
        ],
        "stop_when_group_rewards_identical": True,
        "checkpoint_rule": CHECKPOINT_RULE,
        "held_out_group": {
            "tasks": [task[0] for task in HELD_OUT_TASKS],
            "generation_seeds": seeds,
            "protocol": "one_action_per_turn",
            "settings": "baseline A",
        },
    }


class _BudgetProbe(TrainerCallback):
    """Allow up to four optimizer steps and keep each completed group."""

    def __init__(self, max_steps: int):
        self.max_steps = max_steps
        self.optimizer_steps = 0
        self.grad_norm = None
        self.previous = None
        self.steps: list[dict[str, Any]] = []
        self.trainer = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        del args, state, kwargs
        self.previous = _lora_vector(model)
        return control

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        del args, state, kwargs
        self.grad_norm = _lora_grad_norm(model)
        print(f"lora_grad_norm={self.grad_norm}", flush=True)
        return control

    def on_optimizer_step(self, args, state, control, model=None, **kwargs):
        del args, kwargs
        current = _lora_vector(model)
        change = _param_change(self.previous, current)
        if not change["changed"]:
            raise RuntimeError(f"LoRA parameters did not change after optimizer step: {change}")
        self.optimizer_steps += 1
        group = _group_from_trainer(self.trainer)
        _reject_uncached(group)
        self.steps.append(
            {
                "optimizer_step": self.optimizer_steps,
                "global_step": int(state.global_step) + 1,
                "completed": True,
                "rewards": [row["reward"] for row in group],
                "loss": getattr(self.trainer, "_smoke_loss", None),
                "lora_grad_norm": self.grad_norm,
                "lora_param_change": change,
                "group": group,
            }
        )
        print(
            f"completed optimizer_step={self.optimizer_steps} "
            + ", ".join(f"{row['rollout_id']}={row['reward']}" for row in group),
            flush=True,
        )
        self.previous = current
        if self.optimizer_steps >= self.max_steps:
            control.should_training_stop = True
        return control


def _reject_uncached(group: list[dict[str, Any]]) -> None:
    for row in group:
        for length in row["new_token_lengths"]:
            if length >= config.ONE_ACTION_MAX_NEW_TOKENS:
                raise RuntimeError(
                    f"{row['rollout_id']} generated {length} tokens; the KV cache was off"
                )


def _group_from_trainer(trainer) -> list[dict[str, Any]]:
    rows = list(getattr(trainer, "_smoke_rows", None) or [])
    events = {
        item["rollout_id"]: item for item in (getattr(trainer, "_smoke_events", None) or [])
    }
    saved = getattr(trainer, "_preshuffle_turns", None) or {}
    if len(rows) != 4:
        raise RuntimeError(f"expected one group of 4, got {len(rows)}")
    group = []
    for row in rows:
        rid = row["rollout_id"]
        event = events.get(rid, {})
        turns = saved.get(rid)
        if not turns:
            raise RuntimeError(f"missing turn ids for {rid}")
        news = [list(turn) for turn in turns["turn_new_token_ids"]]
        prefixes = [list(turn) for turn in turns["turn_input_ids"]]
        group.append(
            {
                "rollout_id": rid,
                "reward": row["reward"],
                "advantage": row["advantage"],
                "n_turns": row["n_turns"],
                "n_scored_tokens": row["n_scored_tokens"],
                "task_success": event.get("task_success"),
                "refund_confirmed": event.get("task_success_harness"),
                "final_answer_correct": event.get("task_success_answer"),
                "final_answer": event.get("final_answer"),
                "n_invalid": event.get("n_invalid"),
                "new_token_lengths": [len(turn) for turn in news],
                "turn_new_token_ids": news,
                "turn_input_ids": prefixes,
            }
        )
    group.sort(key=lambda item: item["rollout_id"])
    return group


def _dropout_values(model) -> list[float]:
    import torch

    values = []
    for module in model.modules():
        drop = getattr(module, "lora_dropout", None)
        children = []
        if isinstance(drop, torch.nn.Dropout):
            children = [drop]
        elif isinstance(drop, torch.nn.ModuleDict):
            children = list(drop.values())
        for child in children:
            if isinstance(child, torch.nn.Dropout):
                values.append(float(child.p))
    return values


def _train(probe: _BudgetProbe) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(config.SEED)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(config.SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID, revision=config.MODEL_REVISION, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        revision=config.MODEL_REVISION,
        dtype=torch.float16,
        local_files_only=True,
    )
    task = _task_tuple("t0")
    holder: dict[str, Any] = {}

    def live_engine_factory():
        from agenttune.agentic.rollout_engines.transformers_engine import TransformersRolloutEngine

        return TransformersRolloutEngine(model=holder["trainer"].model, tokenizer=tokenizer)

    rollout = make_instrumented_rollout_func(
        episode_id=task[0],
        order_id=task[1],
        customer_id=task[2],
        amount_cents=task[3],
        generation_seeds=list(SEED_ORDER),
        tokenizer=tokenizer,
        mode="live",
        live_engine_factory=live_engine_factory,
        rollout_eval=True,
    )
    trainer = _build_b_only_trainer(
        tokenizer,
        model,
        rollout,
        probe,
        reward_fn=reward_B,
        output_dir=OUT / "scratch",
        lora_dropout=0.0,
        max_steps=MAX_STEPS,
    )
    holder["trainer"] = trainer
    probe.trainer = trainer
    dropout = _dropout_values(trainer.model)
    if any(value != 0.0 for value in dropout):
        raise RuntimeError(f"pilot LoRA dropout is not 0: {sorted(set(dropout))}")
    if pilot_old_logps_reason(trainer) is not None:
        raise RuntimeError(pilot_old_logps_reason(trainer))
    install_turn_scoring(trainer)
    _hook_rewards(trainer)
    trainer._smoke_check_batch = True
    trainer._smoke_enable_graph = True
    _write_json(
        OUT / "pilot_config.json",
        {
            "written_before_training": True,
            "held_out_ran": False,
            "max_steps": int(trainer.args.max_steps),
            "learning_rate": float(trainer.args.learning_rate),
            "lora_dropout_values": dropout,
            "num_generations": int(trainer.num_generations),
            "generation_seeds": list(SEED_ORDER),
            "train_task": "t0",
            "temperature": float(trainer.temperature),
            "top_p": float(trainer.args.top_p),
            "loss_type": trainer.loss_type,
            "beta": float(trainer.beta),
            "gradient_checkpointing": bool(trainer.args.gradient_checkpointing),
            "use_vllm": bool(trainer.use_vllm),
            "old_per_token_logps_computed": False,
            "rollout_eval": True,
        },
    )
    print("task_5_7 B-only train max_steps=4 t0 seeds 100-103", flush=True)
    stopped = None
    try:
        trainer.train()
    except IdenticalRewards as exc:
        trainer._smoke_rows = exc.rows
        stopped = _group_from_trainer(trainer)
        _reject_uncached(stopped)
        print(
            "identical rewards "
            + ", ".join(f"{row['rollout_id']}={row['reward']}" for row in stopped),
            flush=True,
        )
    return {"trainer": trainer, "identical_group": stopped}


def _write_checkpoint(optimizer_steps: int) -> dict[str, Any]:
    payload = {
        "kind": "b_only_checkpoint_choice",
        "selected_before_held_out": True,
        "held_out_ran": False,
        "rule": CHECKPOINT_RULE,
        "optimizer_steps": optimizer_steps,
        "adapter": "artifacts/task-5-7/adapter_B",
    }
    _write_json(OUT / "checkpoint.json", payload)
    return payload


def _amount_cents(episode_id: str) -> int:
    for spec in HELD_OUT_TASKS:
        if spec[0] == episode_id:
            return int(spec[3])
    raise KeyError(episode_id)


def _pair_from_steps(steps: list[dict], *, source: str) -> dict[str, Any]:
    last = steps[-1]
    events = events_from_steps(steps, rollout_id=last["rollout_id"], source=source)
    amount = _amount_cents(last["episode_id"])
    confirmed = bool(last.get("task_success_harness"))
    correct = bool(answer_matches(events["final_answer"], amount))
    if correct != bool(last.get("task_success_answer")):
        raise RuntimeError(f"answer flag disagrees with answer_matches for {last['rollout_id']}")
    attempts = 0
    for step in steps:
        attempts += len(step.get("calls") or [])
        attempts += len((step.get("diagnostic") or {}).get("malformed_syntax") or [])
    return {
        "rollout_id": last["rollout_id"],
        "episode_id": last["episode_id"],
        "generation_seed": last["generation_seed"],
        "amount_cents": amount,
        "task_success": bool(events["task_success"]),
        "refund_confirmed": confirmed,
        "final_answer_correct": correct,
        "final_answer": events["final_answer"],
        "n_invalid": int(events["n_invalid"]),
        "attempt_count": attempts,
        "parsed_calls": int(events["parsed_call_count"]),
    }


def _load_baseline_pairs() -> list[dict[str, Any]]:
    grouped: dict[str, list[dict]] = {}
    with A_EPISODES.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            grouped.setdefault(row["rollout_id"], []).append(row)
    pairs = []
    for episode_id, _order, _customer, _amount in HELD_OUT_TASKS:
        for seed in SEED_ORDER:
            rid = f"{episode_id}-g{seed}"
            steps = sorted(grouped[rid], key=lambda item: item["step_id"])
            pairs.append(_pair_from_steps(steps, source=str(A_EPISODES)))
    return pairs


def _eval_adapter(adapter_dir: Path) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import agenttune.agentic.rollout_engines.rollout_factory as factory
    from agenttune.agentic.rollout_engines.transformers_engine import TransformersRolloutEngine

    checkpoint = json.loads((OUT / "checkpoint.json").read_text(encoding="utf-8"))
    if not checkpoint.get("selected_before_held_out") or checkpoint.get("held_out_ran"):
        raise RuntimeError("held-out started before the checkpoint choice was fixed")
    out = OUT / "eval_B"
    out.mkdir(parents=True, exist_ok=True)
    _bind_output(out)
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID, revision=config.MODEL_REVISION, local_files_only=True
    )
    base = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        revision=config.MODEL_REVISION,
        dtype=torch.float16,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    if torch.backends.mps.is_available():
        model.to("mps")
    engine = TransformersRolloutEngine(model, tokenizer)
    _guards, schema_list = make_guarded_tools(
        RefundEpisode("schema-probe", "ORD-3001", "CUS-701", 1640, config.SEED)
    )
    probe = _probe(tokenizer, schema_list)
    rows: list[dict] = []
    for episode_id, order_id, customer_id, amount in HELD_OUT_TASKS:
        for seed in SEED_ORDER:
            print(f"eval B {episode_id}-g{seed}", flush=True)
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
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["rollout_id"], []).append(row)
    pairs = []
    for episode_id, _order, _customer, _amount in HELD_OUT_TASKS:
        for seed in SEED_ORDER:
            rid = f"{episode_id}-g{seed}"
            steps = sorted(grouped[rid], key=lambda item: item["step_id"])
            pairs.append(_pair_from_steps(steps, source=str(out / "model_episodes.jsonl")))
    aggregates = held_out_metrics_from_rows(rows)
    _write_json(out / "per_pair.json", {"pairs": pairs})
    _write_json(out / "aggregates.json", aggregates)
    del engine, tokenizer, model
    _release()
    return {"pairs": pairs, "aggregates": aggregates}


def _count(pairs: list[dict], field: str) -> str:
    return f"{sum(int(bool(row[field])) for row in pairs)}/{len(pairs)}"


def _invalid_attempts(pairs: list[dict]) -> str:
    numerator = sum(int(row["n_invalid"]) for row in pairs)
    denominator = sum(int(row["attempt_count"]) for row in pairs)
    return f"{numerator}/{denominator}"


def _mean_calls(pairs: list[dict]) -> float:
    return sum(int(row["parsed_calls"]) for row in pairs) / len(pairs)


def _direction(field: str, before: Any, after: Any) -> str | None:
    if before == after:
        return None
    if field in {"task_success", "refund_confirmed", "final_answer_correct"}:
        return "gained" if after and not before else "lost"
    if field == "n_invalid":
        return "gained" if after < before else "lost"
    if field == "parsed_calls":
        return "calls_up" if after > before else "calls_down"
    return "changed"


def _compare(baseline: list[dict], pilot: list[dict], aggregates: dict) -> dict[str, Any]:
    fields = (
        "task_success",
        "refund_confirmed",
        "final_answer_correct",
        "n_invalid",
        "parsed_calls",
    )
    left = {row["rollout_id"]: row for row in baseline}
    right = {row["rollout_id"]: row for row in pilot}
    if set(left) != set(right):
        raise RuntimeError(f"held-out ids differ: {sorted(left)} vs {sorted(right)}")
    changed = []
    for episode_id, _order, _customer, _amount in HELD_OUT_TASKS:
        for seed in SEED_ORDER:
            rid = f"{episode_id}-g{seed}"
            diffs = []
            for field in fields:
                direction = _direction(field, left[rid][field], right[rid][field])
                if direction is None:
                    continue
                diffs.append(
                    {
                        "field": field,
                        "A": left[rid][field],
                        "B": right[rid][field],
                        "direction": direction,
                    }
                )
            if diffs:
                changed.append({"task": episode_id, "seed": seed, "rollout_id": rid, "changes": diffs})
    pilot_invalid = _invalid_attempts(pilot)
    aggregate_invalid = (
        f"{aggregates['invalid_attempts']['numerator']}/{aggregates['invalid_attempts']['denominator']}"
    )
    if pilot_invalid != aggregate_invalid:
        raise RuntimeError(f"invalid attempts {pilot_invalid} != aggregate {aggregate_invalid}")
    table = {
        "A": {
            "task_success": _count(baseline, "task_success"),
            "refund_confirmed": _count(baseline, "refund_confirmed"),
            "final_answer_correct": _count(baseline, "final_answer_correct"),
            "invalid_attempts": _invalid_attempts(baseline),
            "mean_calls": _mean_calls(baseline),
        },
        "B": {
            "task_success": _count(pilot, "task_success"),
            "refund_confirmed": _count(pilot, "refund_confirmed"),
            "final_answer_correct": _count(pilot, "final_answer_correct"),
            "invalid_attempts": pilot_invalid,
            "mean_calls": _mean_calls(pilot),
        },
    }
    return {
        "table": table,
        "changed_pairs": changed,
        "gained": [row for row in changed if any(item["direction"] == "gained" for item in row["changes"])],
        "lost": [row for row in changed if any(item["direction"] == "lost" for item in row["changes"])],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    budget = _budget()
    _write_json(OUT / "budget.json", budget)
    (OUT / "commands.txt").write_text(COMMAND + "\n", encoding="utf-8")
    if json.loads((OUT / "budget.json").read_text(encoding="utf-8"))["held_out_ran"]:
        raise RuntimeError("budget file was not written before training")
    probe = _BudgetProbe(MAX_STEPS)
    trained = _train(probe)
    adapter = OUT / "adapter_B"
    _save_adapter(trained["trainer"], adapter)
    checkpoint = _write_checkpoint(probe.optimizer_steps)
    trace = {
        "optimizer_steps": probe.optimizer_steps,
        "stopped_for_identical_rewards": trained["identical_group"] is not None,
        "identical_group": trained["identical_group"],
        "completed_steps": probe.steps,
    }
    _write_json(OUT / "train_trace.json", trace)
    evaluation = _eval_adapter(adapter)
    comparison = _compare(_load_baseline_pairs(), evaluation["pairs"], evaluation["aggregates"])
    summary = {
        "kind": "b_only_four_step_pilot",
        "command": COMMAND,
        "git_head": _git_head(),
        "versions": _package_versions(),
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "budget": budget,
        "checkpoint": checkpoint,
        "optimizer_steps": probe.optimizer_steps,
        "stopped_for_identical_rewards": trained["identical_group"] is not None,
        "identical_rewards": None
        if trained["identical_group"] is None
        else [row["reward"] for row in trained["identical_group"]],
        "completed_step_rewards": [step["rewards"] for step in probe.steps],
        "comparison": comparison,
        "baseline_a": "artifacts/task-5-2/baseline_a",
        "adapter": "artifacts/task-5-7/adapter_B",
    }
    _write_json(OUT / "summary.json", summary)
    print(
        json.dumps(
            {
                "optimizer_steps": probe.optimizer_steps,
                "stopped_for_identical_rewards": summary["stopped_for_identical_rewards"],
                "identical_rewards": summary["identical_rewards"],
                "completed_step_rewards": summary["completed_step_rewards"],
                "table": comparison["table"],
                "changed_pairs": comparison["changed_pairs"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
