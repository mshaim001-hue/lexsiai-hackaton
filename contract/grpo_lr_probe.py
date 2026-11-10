"""Task 5.10: one B-only step on u1 at lr=1e-5 from the 5.9 LoRA-init recipe.

Saved 5.9 rollouts lack turn_input_ids / turn_new_token_ids, so they cannot
reproduce the GRPO loss exactly. This pilot regenerates only u1, keeps
reward_B and eval-mode rollouts, then probes mean token log-probability of
the correct final vs a repeated post-submit tool call on the saved
after-submit prefixes. Full 5.8 eval runs only if that gap moves toward the
correct final.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contract import config
from contract.grpo_pair import _install_lora_state, _lora_state, _release, _save_adapter
from contract.grpo_path import make_instrumented_rollout_func
from contract.grpo_prep import _lora_dict, _package_versions, _write_json
from contract.grpo_realistic_pilot import (
    _diagnose_events,
    _load_baseline_pairs,
    _metric_block,
    _success_check,
    _transitions,
)
from contract.grpo_train_step import (
    _hook_rewards,
    _lora_grad_norm,
    _lora_params,
    _lora_vector,
    _param_change,
)
from contract.grpo_turn_loss import IdenticalRewards, install_turn_scoring, pilot_old_logps_reason
from contract.realistic_prompts import GENERATION_SEEDS, TRAIN_TASKS, realistic_user_prompt
from contract.rewards import reward_B
from transformers import TrainerCallback

OUT = ROOT / "artifacts" / "task-5-10"
PRIOR = ROOT / "artifacts" / "task-5-9"
COMMAND = (
    "HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
    ".venv/bin/python run_grpo_lr_probe.py"
)
TIME_LIMIT_S = 7200
LEARNING_RATE = 1e-5
U1 = next(task for task in TRAIN_TASKS if task[0] == "u1")
CORRECT_FINAL = f"REFUND_SUBMITTED amount_cents={U1[3]}"
# Canonical repeated call after submit (matches the 5.9 u1 failure mode).
REPEATED_CALL = (
    '[{"tool_name": "get_customer", "parameters": {"customer_id": "CUS-802"}}]'
)


class _OneStepProbe(TrainerCallback):
    def __init__(self):
        self.optimizer_steps = 0
        self.grad_norm = None
        self.before = None
        self.after = None
        self.expected_init = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        import torch

        del args, state, kwargs
        self.before = _lora_vector(model)
        if self.expected_init is not None and not torch.equal(self.before, self.expected_init):
            raise RuntimeError("LoRA init diverged from the saved 5.10 init vector")
        return control

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


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def write_pilot_config() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    reuse_decision = {
        "reuse_saved_5_9_rollouts": False,
        "reason": (
            "artifacts/task-5-9/group_u1.json has no turn_input_ids or turn_new_token_ids; "
            "those ids are required for exact per-turn GRPO scoring. "
            "Regenerate only u1 from the same LoRA-init recipe as 5.9."
        ),
        "prior_group": "artifacts/task-5-9/group_u1.json",
        "prior_init_file": None,
        "prior_init_note": (
            "5.9 did not persist lora_init.pt; 5.10 recreates init with config.SEED "
            f"({config.SEED}) before attaching LoRA, matching the 5.9 procedure."
        ),
    }
    payload = {
        "kind": "realistic_b_only_lr_probe",
        "written_before_training": True,
        "held_out_ran": False,
        "command": COMMAND,
        "reward": "reward_B_I_task_success",
        "learning_rate": LEARNING_RATE,
        "learning_rate_5_9": 1e-6,
        "max_optimizer_steps": 1,
        "time_limit_s": TIME_LIMIT_S,
        "train_task": "u1",
        "generation_seeds": list(GENERATION_SEEDS),
        "rollout_mode": "eval during live rollout; restore previous mode before the loss",
        "lora_dropout": 0.0,
        "reuse_decision": reuse_decision,
        "logprob_probe": {
            "correct_final": CORRECT_FINAL,
            "repeated_call": REPEATED_CALL,
            "contexts": "last-turn prefixes after submit from the live u1 group",
            "eval_gate": "run full 5.8 eval only if (mean_logp_correct - mean_logp_repeated) increases",
        },
        "success_criterion": {
            "fewer_calls_after_confirmed_submit": True,
            "more_full_successes": True,
            "confirmed_returns_at_least": 14,
        },
        "temperature": config.ONE_ACTION_TEMPERATURE,
        "top_p": config.ONE_ACTION_TOP_P,
        "max_new_tokens": config.ONE_ACTION_MAX_NEW_TOKENS,
        "protocol": "one_action_per_turn",
    }
    _write_json(OUT / "pilot_config.json", payload)
    (OUT / "commands.txt").write_text(COMMAND + "\n", encoding="utf-8")
    _write_json(OUT / "reuse_decision.json", reuse_decision)
    return payload


def _load_base():
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
    return tokenizer, model


def _build_trainer(tokenizer, model, probe, output_dir: Path):
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from agenttune.agentic.rollout_engines.rollout_factory import ensure_lora_targets

    episode_id, order_id, customer_id, amount = U1
    prompt = realistic_user_prompt(episode_id, order_id, customer_id)
    dataset = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": prompt}], "episode_id": episode_id}]
    )
    lora = _lora_dict()
    lora["lora_dropout"] = 0.0
    args = GRPOConfig(
        output_dir=str(output_dir),
        num_generations=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        max_steps=1,
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
        gradient_checkpointing=False,
    )
    holder: dict[str, Any] = {}

    def live_engine_factory():
        from agenttune.agentic.rollout_engines.transformers_engine import TransformersRolloutEngine

        return TransformersRolloutEngine(model=holder["trainer"].model, tokenizer=tokenizer)

    rollout = make_instrumented_rollout_func(
        episode_id=episode_id,
        order_id=order_id,
        customer_id=customer_id,
        amount_cents=amount,
        generation_seeds=list(GENERATION_SEEDS),
        tokenizer=tokenizer,
        mode="live",
        live_engine_factory=live_engine_factory,
        rollout_eval=True,
        prompt_text=prompt,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_B],
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=ensure_lora_targets(LoraConfig(**lora), model),
        callbacks=[probe],
    )
    trainer.rollout_func = rollout
    holder["trainer"] = trainer
    return trainer


def _mean_token_logprob(model, tokenizer, prefix_ids: list[int], text: str, temperature: float) -> dict[str, Any]:
    import torch
    from trl.trainer.utils import selective_log_softmax

    new = tokenizer.encode(text, add_special_tokens=False)
    if not new:
        raise RuntimeError(f"empty tokenisation for {text!r}")
    device = next(model.parameters()).device
    sequence = list(prefix_ids) + list(new)
    input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits[:, :-1, :][:, -len(new) :, :].float()
        logits = logits / float(temperature)
        targets = torch.tensor([new], dtype=torch.long, device=device)
        logps = selective_log_softmax(logits, targets).squeeze(0)
        mean = float(logps.mean().item())
    return {
        "text": text,
        "n_tokens": len(new),
        "mean_logprob": mean,
        "sum_logprob": float(logps.sum().item()),
        "token_ids": [int(token) for token in new],
    }


def _probe_contexts(model, tokenizer, contexts: list[dict[str, Any]]) -> dict[str, Any]:
    per = []
    for ctx in contexts:
        correct = _mean_token_logprob(
            model, tokenizer, ctx["prefix_ids"], CORRECT_FINAL, config.ONE_ACTION_TEMPERATURE
        )
        repeated = _mean_token_logprob(
            model, tokenizer, ctx["prefix_ids"], REPEATED_CALL, config.ONE_ACTION_TEMPERATURE
        )
        per.append(
            {
                "rollout_id": ctx["rollout_id"],
                "prefix_len": len(ctx["prefix_ids"]),
                "correct": correct,
                "repeated_call": repeated,
                "gap_correct_minus_repeated": correct["mean_logprob"] - repeated["mean_logprob"],
            }
        )
    mean_correct = sum(item["correct"]["mean_logprob"] for item in per) / len(per)
    mean_repeated = sum(item["repeated_call"]["mean_logprob"] for item in per) / len(per)
    return {
        "n_contexts": len(per),
        "mean_logprob_correct_final": mean_correct,
        "mean_logprob_repeated_call": mean_repeated,
        "gap_correct_minus_repeated": mean_correct - mean_repeated,
        "per_context": per,
        "correct_final_text": CORRECT_FINAL,
        "repeated_call_text": REPEATED_CALL,
    }


def _contexts_from_trainer(trainer) -> list[dict[str, Any]]:
    saved = getattr(trainer, "_preshuffle_turns", None) or {}
    events = {
        item["rollout_id"]: item for item in (getattr(trainer, "_smoke_events", None) or [])
    }
    contexts = []
    for rid, turns in sorted(saved.items()):
        prefixes = turns["turn_input_ids"]
        if not prefixes:
            raise RuntimeError(f"{rid} has no turn prefixes")
        # Last generate call is the post-submit final turn under one-action max_steps=4 tools + final.
        contexts.append(
            {
                "rollout_id": rid,
                "prefix_ids": [int(token) for token in prefixes[-1]],
                "n_turns": len(prefixes),
                "task_success": (events.get(rid) or {}).get("task_success"),
            }
        )
    if len(contexts) != 4:
        raise RuntimeError(f"expected 4 post-submit contexts, got {len(contexts)}")
    return contexts


def train_u1(started: float) -> dict[str, Any]:
    import torch

    remaining = TIME_LIMIT_S - (time.monotonic() - started)
    if remaining <= 0:
        return {"status": "skipped_time_limit", "optimizer_steps": 0}
    print(f"5.10 train u1 lr={LEARNING_RATE} remaining_s={remaining:.0f}", flush=True)
    tokenizer, model = _load_base()
    probe = _OneStepProbe()
    trainer = _build_trainer(tokenizer, model, probe, OUT / "scratch_u1")
    init_state = _lora_state(trainer.model)
    torch.save(init_state, OUT / "lora_init.pt")
    probe.expected_init = torch.cat(
        [init_state[name].detach().float().reshape(-1) for name, _param in _lora_params(trainer.model)]
    )
    if pilot_old_logps_reason(trainer) is not None:
        raise RuntimeError(pilot_old_logps_reason(trainer))
    install_turn_scoring(trainer)
    _hook_rewards(trainer)
    trainer._smoke_check_batch = True
    trainer._smoke_enable_graph = True
    result: dict[str, Any] = {
        "episode_id": "u1",
        "generation_seeds": list(GENERATION_SEEDS),
        "learning_rate": LEARNING_RATE,
        "lora_init": "artifacts/task-5-10/lora_init.pt",
    }
    try:
        trainer.train()
    except IdenticalRewards as exc:
        result.update(
            {
                "status": "identical_rewards",
                "optimizer_steps": 0,
                "diagnosis": _diagnose_events(list(getattr(trainer, "_smoke_events", None) or [])),
                "rows": exc.rows,
            }
        )
        _write_json(OUT / "train_u1.json", result)
        del trainer, model, tokenizer
        _release()
        return result

    if probe.optimizer_steps != 1 or probe.before is None or probe.after is None:
        raise RuntimeError("u1 did not complete one optimizer step")
    change = _param_change(probe.before, probe.after)
    if not change["changed"] or probe.grad_norm is None or probe.grad_norm <= 0.0:
        raise RuntimeError(f"LoRA update missing: grad={probe.grad_norm} change={change}")
    adapter = OUT / "adapter_B"
    _save_adapter(trainer, adapter)
    print(f"saved adapter {adapter}", flush=True)

    contexts = _contexts_from_trainer(trainer)
    _write_json(
        OUT / "post_submit_contexts.json",
        {
            "rollouts": [
                {
                    "rollout_id": ctx["rollout_id"],
                    "prefix_len": len(ctx["prefix_ids"]),
                    "n_turns": ctx["n_turns"],
                    "task_success": ctx["task_success"],
                    "prefix_ids": ctx["prefix_ids"],
                }
                for ctx in contexts
            ]
        },
    )

    # After = current trained weights.
    after_probe = _probe_contexts(trainer.model, tokenizer, contexts)
    # Before = reinstall the saved init on the same module (adapter already on disk).
    _install_lora_state(trainer.model, init_state)
    trainer.model.eval()
    before_probe = _probe_contexts(trainer.model, tokenizer, contexts)

    diagnosis = _diagnose_events(list(getattr(trainer, "_smoke_events", None) or []))
    train_rows = list(getattr(trainer, "_smoke_rows", None) or [])
    loss = getattr(trainer, "_smoke_loss", None)
    rewards = [row["reward"] for row in train_rows]
    advantages = [row["advantage"] for row in train_rows]

    del trainer, model
    _release()

    gap_before = before_probe["gap_correct_minus_repeated"]
    gap_after = after_probe["gap_correct_minus_repeated"]
    gate = {
        "gap_before": gap_before,
        "gap_after": gap_after,
        "gap_delta": gap_after - gap_before,
        "moved_toward_correct_final": gap_after > gap_before,
    }
    print(
        f"logprob gap before={gap_before:.6f} after={gap_after:.6f} "
        f"delta={gate['gap_delta']:.6f} gate={gate['moved_toward_correct_final']}",
        flush=True,
    )
    result.update(
        {
            "status": "step_completed",
            "optimizer_steps": 1,
            "diagnosis": diagnosis,
            "loss": loss,
            "lora_grad_norm": probe.grad_norm,
            "lora_param_change": change,
            "rewards": rewards,
            "advantages": advantages,
            "rows": train_rows,
            "adapter": "artifacts/task-5-10/adapter_B",
            "logprob_before": before_probe,
            "logprob_after": after_probe,
            "eval_gate": gate,
        }
    )
    _write_json(OUT / "train_u1.json", result)
    _write_json(OUT / "logprob_probe.json", {"before": before_probe, "after": after_probe, "gate": gate})
    return result


def main() -> int:
    config_payload = write_pilot_config()
    started = time.monotonic()
    trained = train_u1(started)
    comparison = None
    stop_reason = trained.get("status")

    if trained.get("status") == "step_completed":
        gate = trained["eval_gate"]
        if not gate["moved_toward_correct_final"]:
            stop_reason = "step_completed_eval_skipped_gate"
            print(
                f"eval gate failed gap {gate['gap_before']:.6f} -> {gate['gap_after']:.6f}; skip full eval",
                flush=True,
            )
        elif time.monotonic() - started >= TIME_LIMIT_S:
            stop_reason = "step_completed_time_limit_before_eval"
            print("time limit reached before 5.8 eval", flush=True)
        else:
            remaining = TIME_LIMIT_S - (time.monotonic() - started)
            print(
                f"eval gate passed gap {gate['gap_before']:.6f} -> {gate['gap_after']:.6f}; "
                f"running 5.8 eval remaining_s={remaining:.0f}",
                flush=True,
            )
            evaluation = _eval_local(OUT / "adapter_B")
            baseline_pairs = _load_baseline_pairs()
            for row in evaluation["pairs"]:
                row.setdefault("submit_confirmed", row.get("task_success_harness"))
            baseline_m = _metric_block(baseline_pairs)
            pilot_m = _metric_block(evaluation["pairs"])
            transitions = _transitions(baseline_pairs, evaluation["pairs"])
            checks = _success_check(baseline_m, pilot_m)
            checks["confirmed_returns_at_least_14"] = pilot_m["confirmed_returns"]["numerator"] >= 14
            checks["pilot_succeeded"] = (
                checks["fewer_calls_after_confirmed_submit"]
                and checks["more_full_successes"]
                and checks["confirmed_returns_at_least_14"]
            )
            comparison = {
                "baseline_metrics": baseline_m,
                "pilot_metrics": pilot_m,
                "transitions": transitions,
                "changed_rollouts": [row for row in transitions if any(row["changed"].values())],
                "success_criterion": checks,
                "observation_caveat": (
                    "Eval is on the 5.8 selection matrix; report as an observation on that matrix."
                ),
            }
            _write_json(OUT / "comparison.json", comparison)
            stop_reason = "step_completed_eval_ran"

    summary = {
        "kind": "realistic_b_only_lr_probe",
        "command": COMMAND,
        "git_head": _git_head(),
        "versions": _package_versions(),
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "pilot_config": config_payload,
        "time_limit_s": TIME_LIMIT_S,
        "elapsed_s": time.monotonic() - started,
        "stop_reason": stop_reason,
        "optimizer_steps": int(trained.get("optimizer_steps") or 0),
        "trained_group": "u1" if trained.get("optimizer_steps") else None,
        "train": trained,
        "adapter": trained.get("adapter"),
        "baseline_5_8": "artifacts/task-5-8/baseline",
        "comparison": comparison,
        "held_out_ran": comparison is not None,
        "reuse_decision": config_payload["reuse_decision"],
    }
    pinned = json.loads((OUT / "pilot_config.json").read_text(encoding="utf-8"))
    pinned["held_out_ran"] = comparison is not None
    pinned["optimizer_steps_completed"] = summary["optimizer_steps"]
    pinned["stop_reason"] = stop_reason
    _write_json(OUT / "pilot_config.json", pinned)
    _write_json(OUT / "summary.json", summary)
    print(
        json.dumps(
            {
                "stop_reason": stop_reason,
                "optimizer_steps": summary["optimizer_steps"],
                "eval_gate": trained.get("eval_gate"),
                "success_criterion": None if comparison is None else comparison["success_criterion"],
                "elapsed_s": summary["elapsed_s"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def _eval_local(adapter_dir: Path) -> dict[str, Any]:
    """Eval adapter on the 5.8 test matrix into artifacts/task-5-10/eval_B."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import agenttune.agentic.rollout_engines.rollout_factory as factory
    from agenttune.agentic.rollout_engines.transformers_engine import TransformersRolloutEngine

    from contract.grpo_realistic_pilot import _pair_metrics
    from contract.harness import (
        ONE_ACTION_SYSTEM_PROMPT,
        ONE_ACTION_TOOL_RESULT_FORMAT,
        ONE_ACTION_TOOLS_FALLBACK_PROMPT,
        RefundEpisode,
    )
    from contract.realistic_prompts import TEST_TASKS, realistic_user_prompt
    from contract.run import _bind_output, _dump, _one_model_rollout, _probe, make_guarded_tools

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
        RefundEpisode("schema-probe", "ORD-5001", "CUS-901", 1520, config.SEED)
    )
    probe = _probe(tokenizer, schema_list)
    rows: list[dict] = []
    for episode_id, order_id, customer_id, amount in TEST_TASKS:
        for seed in GENERATION_SEEDS:
            rid = f"{episode_id}-g{seed}"
            print(f"eval 5.10 {rid}", flush=True)
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
                prompt_text=realistic_user_prompt(episode_id, order_id, customer_id),
                prompt_protocol="one_action_per_turn",
            )
            if limits:
                print("limits", limits, flush=True)
            rows.extend(ep_rows)
            _dump(out / "model_episodes.jsonl", rows)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["rollout_id"]].append(row)
    pairs = []
    for episode_id, _order, _customer, amount in TEST_TASKS:
        for seed in GENERATION_SEEDS:
            rid = f"{episode_id}-g{seed}"
            steps = sorted(grouped[rid], key=lambda item: item["step_id"])
            pairs.append(
                _pair_metrics(steps, source=str(out / "model_episodes.jsonl"), amount_cents=amount)
            )
    _write_json(out / "per_pair.json", {"pairs": pairs})
    _write_json(out / "aggregates.json", _metric_block(pairs))
    del engine, tokenizer, model
    _release()
    return {"pairs": pairs}


if __name__ == "__main__":
    raise SystemExit(main())
