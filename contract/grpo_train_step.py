"""One B-only optimizer step through GRPOTrainer.train(). Stop after that step."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contract import config
from contract.grpo_path import _task_tuple, make_instrumented_rollout_func
from contract.grpo_prep import SEED_ORDER, _lora_dict, _package_versions, _write_json
from contract.grpo_turn_loss import (
    IdenticalRewards,
    _enable_loss_graph,
    install_turn_scoring,
    pilot_old_logps_reason,
)
from contract.harness import one_action_user_prompt
from contract.rewards import reward_B
from transformers import TrainerCallback

TASK = ROOT / "artifacts" / "task-5-4"


def _lora_params(model):
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad and "lora_" in name]


def _lora_vector(model):
    import torch

    parts = [param.detach().float().cpu().reshape(-1) for _name, param in _lora_params(model)]
    if not parts:
        raise RuntimeError("no trainable LoRA parameters")
    return torch.cat(parts)


def _lora_grad_norm(model) -> float:
    import torch

    total = torch.zeros((), dtype=torch.float32)
    saw = False
    for name, param in _lora_params(model):
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if not torch.isfinite(grad).all():
            raise RuntimeError(f"non-finite LoRA gradient in {name}")
        total = total + grad.float().pow(2).sum().cpu()
        saw = True
    if not saw:
        raise RuntimeError("LoRA parameters have no gradient after backward")
    norm = float(total.sqrt().item())
    if not (norm > 0.0) or norm == float("inf"):
        raise RuntimeError(f"LoRA gradient norm is not finite and nonzero: {norm}")
    return norm


class _OneStepProbe(TrainerCallback):
    """Record the LoRA gradient after backward and the parameter change after the step."""

    def __init__(self):
        self.optimizer_steps = 0
        self.grad_norm = None
        self.before = None
        self.after = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        import torch

        del args, state, kwargs
        self.before = _lora_vector(model)
        expected = getattr(self, "expected_init", None)
        if expected is not None and not torch.equal(self.before, expected):
            raise RuntimeError("LoRA init diverged before the rollout")

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        del args, state, kwargs
        self.grad_norm = _lora_grad_norm(model)
        print(f"lora_grad_norm={self.grad_norm}", flush=True)
        return control

    def on_optimizer_step(self, args, state, control, model=None, **kwargs):
        del args, state, kwargs
        self.optimizer_steps += 1
        self.after = _lora_vector(model)
        control.should_training_stop = True
        return control


def _param_change(before, after) -> dict[str, float]:
    delta = after - before
    return {
        "l2": float(delta.norm().item()),
        "max_abs": float(delta.abs().max().item()),
        "changed": bool(float(delta.abs().max().item()) > 0.0),
    }


def _build_b_only_trainer(
    tokenizer,
    model,
    rollout_fn,
    probe,
    reward_fn=None,
    output_dir=None,
    lora_dropout: float | None = None,
    max_steps: int = 1,
):
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from agenttune.agentic.rollout_engines.rollout_factory import ensure_lora_targets

    task = _task_tuple("t0")
    prompt = [{"role": "user", "content": one_action_user_prompt(task[1], task[2])}]
    dataset = Dataset.from_list([{"prompt": prompt, "episode_id": "t0"}])
    reward_fn = reward_B if reward_fn is None else reward_fn
    output_dir = TASK / "scratch" if output_dir is None else output_dir
    lora = _lora_dict()
    if lora_dropout is not None:
        lora["lora_dropout"] = lora_dropout
    args = GRPOConfig(
        output_dir=str(output_dir),
        num_generations=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=1e-6,
        max_steps=max_steps,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=False,
        fp16=False,
        beta=0.0,
        temperature=config.ONE_ACTION_TEMPERATURE,
        top_p=config.ONE_ACTION_TOP_P,
        max_completion_length=config.ONE_ACTION_MAX_NEW_TOKENS,
        remove_unused_columns=False,
        use_vllm=False,
        seed=config.SEED,
        # Checkpointing disables the KV cache inside generate and the rollout
        # runs to max_new_tokens. Enable it later, inside _compute_loss.
        gradient_checkpointing=False,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=ensure_lora_targets(LoraConfig(**lora), model),
        callbacks=[probe],
    )
    trainer.rollout_func = rollout_fn
    return trainer


def _hook_rewards(trainer) -> None:
    original = trainer._calculate_rewards

    def hooked(inputs_arg, prompts, completions, completion_ids_list):
        rewards = original(inputs_arg, prompts, completions, completion_ids_list)
        if rewards.shape[1] != 1:
            raise RuntimeError(f"expected one reward column, got {tuple(rewards.shape)}")
        trainer._smoke_reward_column = rewards[:, 0].detach()
        captured = []
        for row in inputs_arg:
            ev = row.get("contract_events") or {}
            if not isinstance(ev, dict):
                ev = {}
            harness = ev.get("harness_state") or {}
            captured.append(
                {
                    "rollout_id": str(row.get("rollout_id") or ev.get("rollout_id") or ""),
                    "task_success": ev.get("task_success"),
                    "task_success_harness": harness.get("task_success_harness"),
                    "task_success_answer": harness.get("task_success_answer"),
                    "final_answer": ev.get("final_answer"),
                    "n_invalid": ev.get("n_invalid"),
                }
            )
        trainer._smoke_events = captured
        return rewards

    trainer._calculate_rewards = hooked


def _report(trainer, probe, *, rows, stopped_for_identical_rewards: bool) -> dict[str, Any]:
    change = None
    if probe.before is not None and probe.after is not None:
        change = _param_change(probe.before, probe.after)
    return {
        "kind": "grpo_b_only_one_step",
        "reward": "reward_B_I_task_success",
        "train_task": "t0",
        "generation_seeds": list(SEED_ORDER),
        "num_generations": 4,
        "versions": _package_versions(),
        "old_per_token_logps_computed": pilot_old_logps_reason(trainer) is not None,
        "temperature": float(trainer.temperature),
        "loss_type": trainer.loss_type,
        "gradient_accumulation_steps": int(trainer.args.gradient_accumulation_steps),
        "steps_per_generation": int(trainer.args.steps_per_generation),
        "num_iterations": int(trainer.num_iterations),
        "use_vllm": bool(trainer.use_vllm),
        "rows": rows,
        "rewards": [row["reward"] for row in rows] if rows else None,
        "advantages": [row["advantage"] for row in rows] if rows else None,
        "scored_tokens": getattr(trainer, "_smoke_scored_tokens", None),
        "loss": getattr(trainer, "_smoke_loss", None),
        "lora_grad_norm": probe.grad_norm,
        "lora_param_change": change,
        "optimizer_steps": probe.optimizer_steps,
        "global_step": int(trainer.state.global_step),
        "stopped_for_identical_rewards": stopped_for_identical_rewards,
        "stopped_after_one_optimizer_step": probe.optimizer_steps == 1 and not stopped_for_identical_rewards,
    }


def main(argv: list[str] | None = None) -> int:
    del argv
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TASK.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.SEED)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, revision=config.MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        revision=config.MODEL_REVISION,
        dtype=torch.float16,
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
    probe = _OneStepProbe()
    trainer = _build_b_only_trainer(tokenizer, model, rollout, probe)
    holder["trainer"] = trainer
    if pilot_old_logps_reason(trainer) is not None:
        raise RuntimeError(pilot_old_logps_reason(trainer))
    if trainer.loss_type != "dapo" or float(trainer.temperature) != config.ONE_ACTION_TEMPERATURE:
        raise RuntimeError(
            f"pilot config drifted: loss_type={trainer.loss_type} temperature={trainer.temperature}"
        )
    install_turn_scoring(trainer)
    _hook_rewards(trainer)
    trainer._smoke_check_batch = True
    trainer._smoke_enable_graph = True
    print("task_5_4 B-only trainer.train max_steps=1 live t0 x4", flush=True)
    rows = None
    identical = False
    try:
        trainer.train()
    except IdenticalRewards as exc:
        identical = True
        rows = exc.rows
        payload = _report(trainer, probe, rows=rows, stopped_for_identical_rewards=True)
        _write_json(TASK / "b_only_one_step.json", payload)
        print(json.dumps({"stopped": "identical_rewards", "rows": rows}, indent=2))
        return 2
    rows = list(trainer._smoke_rows)
    if probe.optimizer_steps != 1:
        raise RuntimeError(f"optimizer_steps={probe.optimizer_steps}, expected 1")
    if probe.grad_norm is None:
        raise RuntimeError("LoRA gradient was not recorded after backward")
    change = _param_change(probe.before, probe.after)
    if not change["changed"]:
        raise RuntimeError(f"LoRA parameters did not change after the optimizer step: {change}")
    payload = _report(trainer, probe, rows=rows, stopped_for_identical_rewards=False)
    payload["lora_param_change"] = change
    _write_json(TASK / "b_only_one_step.json", payload)
    print(
        json.dumps(
            {
                "rewards": payload["rewards"],
                "advantages": payload["advantages"],
                "scored_tokens": payload["scored_tokens"],
                "loss": payload["loss"],
                "lora_grad_norm": payload["lora_grad_norm"],
                "lora_param_change": change,
                "optimizer_steps": payload["optimizer_steps"],
                "old_per_token_logps_computed": payload["old_per_token_logps_computed"],
            },
            indent=2,
        )
    )
    ok = (
        not identical
        and payload["old_per_token_logps_computed"] is False
        and payload["loss"] is not None
        and payload["loss"] == payload["loss"]
        and payload["lora_grad_norm"] > 0.0
        and change["changed"]
        and payload["optimizer_steps"] == 1
        and all(row["turn_ids_bound"] and row["scored_ids_match_mask"] for row in rows)
        and len({row["reward"] for row in rows}) > 1
        and max(abs(row["advantage"]) for row in rows) > 0.0
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
