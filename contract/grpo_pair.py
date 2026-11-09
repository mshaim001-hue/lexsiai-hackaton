"""Paired B/C pilots: one shared LoRA init, one t0 group, one optimizer step each."""

from __future__ import annotations

import gc
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contract import config
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
    _lora_params,
    _param_change,
    _OneStepProbe,
)
from contract.grpo_turn_loss import IdenticalRewards, install_turn_scoring, pilot_old_logps_reason
from contract.harness import (
    ONE_ACTION_SYSTEM_PROMPT,
    ONE_ACTION_TOOL_RESULT_FORMAT,
    ONE_ACTION_TOOLS_FALLBACK_PROMPT,
    RefundEpisode,
    one_action_user_prompt,
)
from contract.rewards import reward_B, reward_C
from contract.run import _bind_output, _dump, _one_model_rollout, _probe, make_guarded_tools

OUT = ROOT / "artifacts" / "task-5-5"
A_PAIRS = ROOT / "artifacts" / "task-5-2" / "baseline_a" / "per_pair.json"
A_AGG = ROOT / "artifacts" / "task-5-2" / "baseline_a" / "aggregates.json"
COMMAND = "TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 .venv/bin/python run_grpo_pair.py"


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _load_fresh_base():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(config.SEED)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(config.SEED)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, revision=config.MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        revision=config.MODEL_REVISION,
        dtype=torch.float16,
    )
    revision = getattr(model.config, "_commit_hash", None) or config.MODEL_REVISION
    if revision != config.MODEL_REVISION:
        raise RuntimeError(f"loaded revision {revision} != {config.MODEL_REVISION}")
    return tokenizer, model, revision


def _lora_state(model) -> dict:
    return {name: param.detach().cpu().clone() for name, param in _lora_params(model)}


def _install_lora_state(model, state: dict) -> None:
    named = dict(model.named_parameters())
    missing = sorted(set(state) - set(named))
    extra = sorted(
        name
        for name, param in _lora_params(model)
        if name not in state
    )
    if missing or extra:
        raise RuntimeError(f"LoRA state does not match the model: missing={missing} extra={extra}")
    max_abs = 0.0
    for name, value in state.items():
        param = named[name]
        param.data.copy_(value.to(device=param.device, dtype=param.dtype))
        got = param.detach().cpu()
        max_abs = max(max_abs, float((got.float() - value.float()).abs().max().item()))
    if max_abs != 0.0:
        raise RuntimeError(f"LoRA init copy is not exact, max abs {max_abs}")


def _release() -> None:
    import torch

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _save_adapter(trainer, path: Path) -> None:
    model = trainer.accelerator.unwrap_model(trainer.model)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)


def _rollout_record(trainer) -> dict[str, Any]:
    saved = getattr(trainer, "_preshuffle_turns", None)
    if not saved:
        raise RuntimeError("turn ids were not kept through the batch")
    events = {
        item["rollout_id"]: item for item in (getattr(trainer, "_smoke_events", None) or [])
    }
    rows = []
    for row in trainer._smoke_rows:
        rid = row["rollout_id"]
        turns = saved[rid]
        ev = events.get(rid, {})
        rows.append(
            {
                **row,
                "task_success": ev.get("task_success"),
                "n_invalid": ev.get("n_invalid"),
                "turn_new_token_ids": turns["turn_new_token_ids"],
                "turn_input_lengths": [len(turn) for turn in turns["turn_input_ids"]],
                "turn_input_ids": turns["turn_input_ids"],
            }
        )
    return {"rows": rows}


def train_one(mode: str, reward_fn, init_state: dict | None) -> dict[str, Any]:
    import torch

    print(f"pair train {mode} fresh {config.MODEL_REVISION}", flush=True)
    tokenizer, model, revision = _load_fresh_base()
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
    probe = _OneStepProbe()
    trainer = _build_b_only_trainer(
        tokenizer,
        model,
        rollout,
        probe,
        reward_fn=reward_fn,
        output_dir=OUT / f"scratch_{mode}",
    )
    holder["trainer"] = trainer
    if init_state is None:
        init_state = _lora_state(trainer.model)
        torch.save(init_state, OUT / "lora_init.pt")
    else:
        _install_lora_state(trainer.model, init_state)
    if pilot_old_logps_reason(trainer) is not None:
        raise RuntimeError(pilot_old_logps_reason(trainer))
    probe.expected_init = torch.cat(
        [init_state[name].detach().float().reshape(-1) for name, _param in _lora_params(trainer.model)]
    )
    install_turn_scoring(trainer)
    _hook_rewards(trainer)
    trainer._smoke_check_batch = True
    trainer._smoke_enable_graph = True
    try:
        trainer.train()
    except IdenticalRewards as exc:
        payload = {
            "mode": mode,
            "stopped_for_identical_rewards": True,
            "rows": exc.rows,
        }
        _write_json(OUT / f"train_{mode}.json", payload)
        raise
    if probe.optimizer_steps != 1 or probe.before is None or probe.after is None:
        raise RuntimeError(f"{mode} did not complete one optimizer step")
    change = _param_change(probe.before, probe.after)
    if not change["changed"] or probe.grad_norm is None or probe.grad_norm <= 0.0:
        raise RuntimeError(f"{mode} LoRA update is missing: grad={probe.grad_norm} change={change}")
    adapter_dir = OUT / f"adapter_{mode}"
    _save_adapter(trainer, adapter_dir)
    record = _rollout_record(trainer)
    payload = {
        "mode": mode,
        "reward": reward_fn.__name__,
        "model_id": config.MODEL_ID,
        "model_revision": revision,
        "train_task": "t0",
        "generation_seeds": list(SEED_ORDER),
        "old_per_token_logps_computed": False,
        "temperature": float(trainer.temperature),
        "loss_type": trainer.loss_type,
        "gradient_accumulation_steps": int(trainer.args.gradient_accumulation_steps),
        "steps_per_generation": int(trainer.args.steps_per_generation),
        "num_iterations": int(trainer.num_iterations),
        "use_vllm": bool(trainer.use_vllm),
        "rows": record["rows"],
        "rewards": [row["reward"] for row in record["rows"]],
        "advantages": [row["advantage"] for row in record["rows"]],
        "scored_tokens": trainer._smoke_scored_tokens,
        "loss": trainer._smoke_loss,
        "lora_grad_norm": probe.grad_norm,
        "lora_param_change": change,
        "optimizer_steps": probe.optimizer_steps,
        "global_step": int(trainer.state.global_step),
        "adapter_dir": str(adapter_dir),
        "stopped_for_identical_rewards": False,
    }
    _write_json(OUT / f"train_{mode}.json", payload)
    delta = probe.after - probe.before
    print(
        f"pair {mode} rewards={payload['rewards']} advantages={payload['advantages']} "
        f"loss={payload['loss']} grad={probe.grad_norm} delta_l2={change['l2']}",
        flush=True,
    )
    kept = {"payload": payload, "delta": delta.cpu().clone(), "init_state": init_state}
    del trainer, model, tokenizer, probe, holder
    _release()
    return kept


def _load_adapted(adapter_dir: Path):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from agenttune.agentic.rollout_engines.transformers_engine import TransformersRolloutEngine

    torch.manual_seed(config.SEED)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(config.SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, revision=config.MODEL_REVISION)
    base = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        revision=config.MODEL_REVISION,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    if device != "cpu":
        model.to(device)
    engine = TransformersRolloutEngine(model, tokenizer)
    return engine, tokenizer, model


def eval_adapter(mode: str, adapter_dir: Path) -> dict[str, Any]:
    import agenttune.agentic.rollout_engines.rollout_factory as factory

    from contract.events import events_from_steps

    out = OUT / f"eval_{mode}"
    out.mkdir(parents=True, exist_ok=True)
    _bind_output(out)
    engine, tokenizer, model = _load_adapted(adapter_dir)
    _guards, schema_list = make_guarded_tools(
        RefundEpisode("schema-probe", "ORD-3001", "CUS-701", 1640, config.SEED)
    )
    probe = _probe(tokenizer, schema_list)
    rows: list[dict] = []
    for episode_id, order_id, customer_id, amount in HELD_OUT_TASKS:
        for seed in SEED_ORDER:
            rid = f"{episode_id}-g{seed}"
            print(f"eval {mode} {rid}", flush=True)
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
    by_rollout: dict[str, list[dict]] = {}
    for row in rows:
        by_rollout.setdefault(row["rollout_id"], []).append(row)
    per_pair = []
    for episode_id, order_id, customer_id, amount in HELD_OUT_TASKS:
        for seed in SEED_ORDER:
            rid = f"{episode_id}-g{seed}"
            steps = by_rollout[rid]
            events = events_from_steps(steps, rollout_id=rid, source=str(out / "model_episodes.jsonl"))
            per_pair.append(
                {
                    "rollout_id": rid,
                    "episode_id": episode_id,
                    "order_id": order_id,
                    "customer_id": customer_id,
                    "amount_cents": amount,
                    "generation_seed": seed,
                    "task_success": events["task_success"],
                    "n_invalid": events["n_invalid"],
                    "parsed_calls": events["parsed_call_count"],
                    "final_answer": events["final_answer"],
                }
            )
    aggregates = held_out_metrics_from_rows(rows)
    payload = {
        "kind": f"held_out_eval_{mode}",
        "condition": mode,
        "adapter_dir": str(adapter_dir),
        "protocol": "one_action_per_turn",
        "tasks": [task[0] for task in HELD_OUT_TASKS],
        "generation_seeds": list(SEED_ORDER),
        "n_rollouts": len(per_pair),
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "per_pair": per_pair,
        "aggregates": aggregates,
    }
    _write_json(out / "per_pair.json", {"pairs": per_pair})
    _write_json(out / "aggregates.json", aggregates)
    _write_json(out / "manifest.json", payload)
    del engine, tokenizer, model
    _release()
    return payload


def _index(pairs: list[dict]) -> dict[str, dict]:
    return {row["rollout_id"]: row for row in pairs}


def _compare_training(b_run: dict, c_run: dict) -> dict[str, Any]:
    import torch

    b_rows = _index(b_run["payload"]["rows"])
    c_rows = _index(c_run["payload"]["rows"])
    if set(b_rows) != set(c_rows):
        raise RuntimeError(f"rollout ids differ: {sorted(b_rows)} vs {sorted(c_rows)}")
    per_id = []
    rollouts_match = True
    rewards_differ = False
    advantages_differ = False
    for rid in sorted(b_rows):
        left = b_rows[rid]
        right = c_rows[rid]
        same_new = left["turn_new_token_ids"] == right["turn_new_token_ids"]
        same_prefix = left["turn_input_ids"] == right["turn_input_ids"]
        same_reward = left["reward"] == right["reward"]
        same_adv = left["advantage"] == right["advantage"]
        rollouts_match = rollouts_match and same_new and same_prefix
        rewards_differ = rewards_differ or not same_reward
        advantages_differ = advantages_differ or not same_adv
        per_id.append(
            {
                "rollout_id": rid,
                "new_tokens_match": same_new,
                "prefix_ids_match": same_prefix,
                "n_scored_tokens_B": left["n_scored_tokens"],
                "n_scored_tokens_C": right["n_scored_tokens"],
                "reward_B": left["reward"],
                "reward_C": right["reward"],
                "advantage_B": left["advantage"],
                "advantage_C": right["advantage"],
                "task_success_B": left["task_success"],
                "task_success_C": right["task_success"],
                "n_invalid_B": left["n_invalid"],
                "n_invalid_C": right["n_invalid"],
            }
        )
    delta = b_run["delta"] - c_run["delta"]
    updates = {
        "B": b_run["payload"]["lora_param_change"],
        "C": c_run["payload"]["lora_param_change"],
        "updates_equal": bool(torch.equal(b_run["delta"], c_run["delta"])),
        "update_l2_diff": float(delta.norm().item()),
        "update_max_abs_diff": float(delta.abs().max().item()),
    }
    return {
        "rollouts_match": rollouts_match,
        "rewards_differ": rewards_differ,
        "advantages_differ": advantages_differ,
        "parameter_updates": updates,
        "per_rollout": per_id,
        "loss_B": b_run["payload"]["loss"],
        "loss_C": c_run["payload"]["loss"],
        "grad_norm_B": b_run["payload"]["lora_grad_norm"],
        "grad_norm_C": c_run["payload"]["lora_grad_norm"],
        "scored_tokens_B": b_run["payload"]["scored_tokens"],
        "scored_tokens_C": c_run["payload"]["scored_tokens"],
    }


def _metric_row(condition: str, aggregates: dict) -> dict[str, Any]:
    return {
        "condition": condition,
        "successes": (
            f"{aggregates['task_success']['numerator']}/{aggregates['task_success']['denominator']}"
        ),
        "invalid_calls": (
            f"{aggregates['invalid_attempts']['numerator']}/{aggregates['invalid_attempts']['denominator']}"
        ),
        "mean_calls": aggregates["calls_per_task"]["mean"],
        "task_success": aggregates["task_success"],
        "invalid_attempts": aggregates["invalid_attempts"],
        "calls_per_task_mean": aggregates["calls_per_task"]["mean"],
    }


def _changed_pairs(a_pairs: list[dict], b_pairs: list[dict], c_pairs: list[dict]) -> list[dict]:
    a_index = _index(a_pairs)
    b_index = _index(b_pairs)
    c_index = _index(c_pairs)
    changed = []
    fields = ("task_success", "n_invalid", "parsed_calls")
    for episode_id, _order, _customer, _amount in HELD_OUT_TASKS:
        for seed in SEED_ORDER:
            rid = f"{episode_id}-g{seed}"
            a_row, b_row, c_row = a_index[rid], b_index[rid], c_index[rid]
            diffs = [
                field
                for field in fields
                if a_row[field] != b_row[field] or a_row[field] != c_row[field] or b_row[field] != c_row[field]
            ]
            if not diffs:
                continue
            changed.append(
                {
                    "task": episode_id,
                    "seed": seed,
                    "rollout_id": rid,
                    "fields": diffs,
                    "A": {field: a_row[field] for field in fields},
                    "B": {field: b_row[field] for field in fields},
                    "C": {field: c_row[field] for field in fields},
                }
            )
    return changed


def main(argv: list[str] | None = None) -> int:
    del argv
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "commands.txt").write_text(COMMAND + "\n", encoding="utf-8")
    b_run = train_one("B", reward_B, None)
    c_run = train_one("C", reward_C, b_run["init_state"])
    training = _compare_training(b_run, c_run)
    _write_json(OUT / "training_comparison.json", training)
    print(json.dumps({k: training[k] for k in (
        "rollouts_match", "rewards_differ", "advantages_differ", "parameter_updates",
        "loss_B", "loss_C",
    )}, indent=2), flush=True)
    eval_b = eval_adapter("B", OUT / "adapter_B")
    eval_c = eval_adapter("C", OUT / "adapter_C")
    a_pairs = json.loads(A_PAIRS.read_text(encoding="utf-8"))["pairs"]
    a_agg = json.loads(A_AGG.read_text(encoding="utf-8"))
    changed = _changed_pairs(a_pairs, eval_b["per_pair"], eval_c["per_pair"])
    table = [
        _metric_row("A", a_agg),
        _metric_row("B", eval_b["aggregates"]),
        _metric_row("C", eval_c["aggregates"]),
    ]
    summary = {
        "kind": "paired_bc_pilot",
        "command": COMMAND,
        "git_head": _git_head(),
        "versions": _package_versions(),
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "train_task": "t0",
        "generation_seeds": list(SEED_ORDER),
        "held_out_tasks": [task[0] for task in HELD_OUT_TASKS],
        "protocol": "one_action_per_turn",
        "training": training,
        "eval_table": table,
        "changed_pairs": changed,
        "baseline_a": "artifacts/task-5-2/baseline_a",
    }
    _write_json(OUT / "summary.json", summary)
    print(json.dumps({"eval_table": table, "changed_pairs": changed}, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
