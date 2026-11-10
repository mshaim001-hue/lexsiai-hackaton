"""Task 5.9: one B-only GRPO step on the first eligible realistic train group.

Scan u0→u3 × seeds 100–103 with reward_B and model.eval() during rollout.
Take at most one optimizer step on the first group that has full successes,
final_answer_tool_call_after_submit failures, and differing rewards. Save the
adapter immediately, then eval on the fixed 5.8 test matrix v0–v3 × 100–103.
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
from contract.events import events_from_steps
from contract.grpo_pair import _release, _save_adapter
from contract.grpo_path import make_instrumented_rollout_func
from contract.grpo_prep import SEED_ORDER, _lora_dict, _package_versions, _write_json
from contract.grpo_train_diag import _error_type, _row_from_steps, _tool_call_text
from contract.grpo_train_step import (
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
)
from contract.realistic_prompts import (
    GENERATION_SEEDS,
    TEST_TASKS,
    TRAIN_TASKS,
    realistic_user_prompt,
)
from contract.rewards import reward_B
from contract.run import _bind_output, _dump, _one_model_rollout, _probe, make_guarded_tools
from transformers import TrainerCallback

OUT = ROOT / "artifacts" / "task-5-9"
BASELINE_58 = ROOT / "artifacts" / "task-5-8"
COMMAND = (
    "HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
    ".venv/bin/python run_grpo_realistic_pilot.py"
)
# Written before any rollout. Covers group scan + at most one optimizer step.
TIME_LIMIT_S = 10800
TARGET_FAILURE = "final_answer_tool_call_after_submit"


class GroupNotEligible(RuntimeError):
    """Live group lacks the target mixed pattern. Stop before backward."""

    def __init__(self, rows: list[dict[str, Any]], reason: str, diagnosis: dict[str, Any]):
        self.rows = rows
        self.reason = reason
        self.diagnosis = diagnosis
        super().__init__(reason)


class _OneStepProbe(TrainerCallback):
    def __init__(self):
        self.optimizer_steps = 0
        self.grad_norm = None
        self.before = None
        self.after = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        del args, state, kwargs
        self.before = _lora_vector(model)
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


def _task_by_id(episode_id: str) -> tuple[str, str, str, int]:
    for spec in (*TRAIN_TASKS, *TEST_TASKS):
        if spec[0] == episode_id:
            return spec
    raise KeyError(episode_id)


def write_pilot_config() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "realistic_b_only_one_step_pilot",
        "written_before_training": True,
        "held_out_ran": False,
        "command": COMMAND,
        "reward": "reward_B_I_task_success",
        "rollout_mode": "eval during live rollout; restore previous mode before the loss",
        "lora_dropout": 0.0,
        "learning_rate": 1e-6,
        "max_optimizer_steps": 1,
        "time_limit_s": TIME_LIMIT_S,
        "train_order": [task[0] for task in TRAIN_TASKS],
        "generation_seeds": list(GENERATION_SEEDS),
        "eligibility": {
            "require_full_success": True,
            "require_error_type": TARGET_FAILURE,
            "require_differing_rewards": True,
        },
        "stop_if_no_eligible_group": True,
        "stop_if_time_limit": True,
        "test_matrix": {
            "tasks": [task[0] for task in TEST_TASKS],
            "generation_seeds": list(GENERATION_SEEDS),
            "settings": "artifacts/task-5-8/pinned_config.json",
            "baseline": "artifacts/task-5-8/baseline",
        },
        "success_criterion": {
            "fewer_calls_after_confirmed_submit": True,
            "more_full_successes": True,
            "confirmed_returns_at_least_baseline": True,
            "note": (
                "Observation on the 5.8 test matrix, which already selected the "
                "target failure mode; not an independent held-out claim."
            ),
        },
        "temperature": config.ONE_ACTION_TEMPERATURE,
        "top_p": config.ONE_ACTION_TOP_P,
        "max_new_tokens": config.ONE_ACTION_MAX_NEW_TOKENS,
        "protocol": "one_action_per_turn",
        "user_prompt_style": "realistic_paraphrase_no_tool_sequence",
    }
    _write_json(OUT / "pilot_config.json", payload)
    (OUT / "commands.txt").write_text(COMMAND + "\n", encoding="utf-8")
    return payload


def _diagnose_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for ev in events:
        # Accept full contract_events or the flattened _smoke_events rows.
        harness = ev.get("harness_state") or {}
        final = ev.get("final_answer")
        success_harness = ev.get("task_success_harness")
        if success_harness is None:
            success_harness = harness.get("task_success_harness")
        success_answer = ev.get("task_success_answer")
        if success_answer is None:
            success_answer = harness.get("task_success_answer")
        row = {
            "rollout_id": ev.get("rollout_id"),
            "task_success": bool(ev.get("task_success")),
            "task_success_harness": bool(success_harness),
            "task_success_answer": bool(success_answer),
            "n_invalid": int(ev.get("n_invalid") or 0),
            "n_invalid_components": ev.get("n_invalid_components") or {},
            "final_answer": final,
            "final_answer_is_tool_call": _tool_call_text(final),
            "function_entries": list(
                ev.get("function_entries") or harness.get("function_entries") or []
            ),
            "entries_after_submit": [],
            "calls_after_confirmed_submit": [],
            "submit_confirmed": bool(success_harness),
            "parsed_calls": int(ev.get("parsed_call_count") or ev.get("parsed_calls") or 0),
        }
        # Mirror grpo_train_diag: post-submit tool-call finals are the target.
        if (
            row["task_success_harness"]
            and not row["task_success_answer"]
            and row["final_answer_is_tool_call"]
        ):
            row["error_type"] = TARGET_FAILURE
        else:
            row["error_type"] = _error_type(row)
        row["B"] = 1.0 if row["task_success"] else 0.0
        rows.append(row)
    rewards = [row["B"] for row in rows]
    return {
        "rows": rows,
        "rewards": rewards,
        "n_full_success": sum(1 for row in rows if row["task_success"]),
        "n_target_failure": sum(1 for row in rows if row["error_type"] == TARGET_FAILURE),
        "n_confirmed": sum(1 for row in rows if row["task_success_harness"]),
        "rewards_differ": len(set(rewards)) > 1,
        "error_types": [row["error_type"] for row in rows],
        "eligible": (
            any(row["task_success"] for row in rows)
            and any(row["error_type"] == TARGET_FAILURE for row in rows)
            and len(set(rewards)) > 1
        ),
    }


def _install_eligibility_gate(trainer) -> None:
    original = trainer._prepare_inputs

    def _prepare_inputs(generation_batch):
        inputs = original(generation_batch)
        events = list(getattr(trainer, "_smoke_events", None) or [])
        diagnosis = _diagnose_events(events)
        trainer._group_diagnosis = diagnosis
        rows = list(getattr(trainer, "_smoke_rows", None) or [])
        if not diagnosis["eligible"]:
            reason = (
                f"group not eligible for target pilot: full_success="
                f"{diagnosis['n_full_success']} target_failure="
                f"{diagnosis['n_target_failure']} rewards={diagnosis['rewards']} "
                f"errors={diagnosis['error_types']}"
            )
            raise GroupNotEligible(rows, reason, diagnosis)
        print(
            "eligible group "
            + ", ".join(
                f"{row['rollout_id']} success={row['task_success']} err={row['error_type']}"
                for row in diagnosis["rows"]
            ),
            flush=True,
        )
        return inputs

    trainer._prepare_inputs = _prepare_inputs


def _build_trainer(tokenizer, model, task: tuple[str, str, str, int], probe, output_dir: Path):
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from agenttune.agentic.rollout_engines.rollout_factory import ensure_lora_targets

    episode_id, order_id, customer_id, amount = task
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
        learning_rate=1e-6,
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


def _try_group(task: tuple[str, str, str, int], started: float) -> dict[str, Any]:
    episode_id = task[0]
    remaining = TIME_LIMIT_S - (time.monotonic() - started)
    if remaining <= 0:
        return {
            "episode_id": episode_id,
            "status": "skipped_time_limit",
            "elapsed_s": time.monotonic() - started,
        }
    print(f"pilot try group {episode_id} remaining_s={remaining:.0f}", flush=True)
    tokenizer, model = _load_base()
    probe = _OneStepProbe()
    trainer = _build_trainer(tokenizer, model, task, probe, OUT / f"scratch_{episode_id}")
    if pilot_old_logps_reason(trainer) is not None:
        raise RuntimeError(pilot_old_logps_reason(trainer))
    install_turn_scoring(trainer)
    _hook_rewards(trainer)
    _install_eligibility_gate(trainer)
    trainer._smoke_check_batch = True
    trainer._smoke_enable_graph = True
    result: dict[str, Any] = {
        "episode_id": episode_id,
        "generation_seeds": list(GENERATION_SEEDS),
        "started_monotonic": started,
    }
    try:
        trainer.train()
    except IdenticalRewards as exc:
        diagnosis = _diagnose_events(list(getattr(trainer, "_smoke_events", None) or []))
        result.update(
            {
                "status": "identical_rewards",
                "eligible": False,
                "optimizer_steps": 0,
                "diagnosis": diagnosis,
                "rows": exc.rows,
            }
        )
        _write_json(OUT / f"group_{episode_id}.json", result)
        del trainer, model, tokenizer
        _release()
        return result
    except GroupNotEligible as exc:
        result.update(
            {
                "status": "not_eligible",
                "eligible": False,
                "optimizer_steps": 0,
                "reason": exc.reason,
                "diagnosis": exc.diagnosis,
                "rows": exc.rows,
            }
        )
        _write_json(OUT / f"group_{episode_id}.json", result)
        del trainer, model, tokenizer
        _release()
        return result

    if probe.optimizer_steps != 1 or probe.before is None or probe.after is None:
        raise RuntimeError(f"{episode_id} did not complete one optimizer step")
    change = _param_change(probe.before, probe.after)
    if not change["changed"] or probe.grad_norm is None or probe.grad_norm <= 0.0:
        raise RuntimeError(f"{episode_id} LoRA update missing: grad={probe.grad_norm} change={change}")
    adapter = OUT / "adapter_B"
    _save_adapter(trainer, adapter)
    diagnosis = getattr(trainer, "_group_diagnosis", None) or _diagnose_events(
        list(getattr(trainer, "_smoke_events", None) or [])
    )
    result.update(
        {
            "status": "step_completed",
            "eligible": True,
            "optimizer_steps": 1,
            "diagnosis": diagnosis,
            "loss": getattr(trainer, "_smoke_loss", None),
            "lora_grad_norm": probe.grad_norm,
            "lora_param_change": change,
            "rewards": [row["reward"] for row in trainer._smoke_rows],
            "advantages": [row["advantage"] for row in trainer._smoke_rows],
            "adapter": str(adapter.relative_to(ROOT)),
            "rows": trainer._smoke_rows,
        }
    )
    _write_json(OUT / f"group_{episode_id}.json", result)
    # Keep trainer only long enough to save; release weights after.
    del trainer, model, tokenizer
    _release()
    return result


def _calls_after_confirmed_from_steps(steps: list[dict]) -> list[str]:
    confirmed = False
    later: list[str] = []
    for step in steps:
        for call in step.get("calls") or []:
            name = call.get("tool") or call.get("name") or ""
            result = call.get("structured_result") or {}
            if confirmed and name:
                later.append(str(name))
            if name == "submit_refund" and (
                result.get("submitted") is True
                or (call.get("execution") == "success" and result.get("ok") is True)
            ):
                confirmed = True
    return later


def _pair_metrics(steps: list[dict], *, source: str, amount_cents: int) -> dict[str, Any]:
    scored = _row_from_steps(steps, source=source)
    later = _calls_after_confirmed_from_steps(steps)
    if later and not scored["calls_after_confirmed_submit"]:
        scored["calls_after_confirmed_submit"] = later
        if scored["error_type"] == "none" or scored["error_type"] == TARGET_FAILURE:
            scored["error_type"] = "call_after_confirmed_refund"
    final = scored["final_answer"]
    answer_ok = answer_matches(final, amount_cents)
    if answer_ok != scored["task_success_answer"]:
        raise RuntimeError(f"answer flag mismatch for {scored['rollout_id']}")
    return {
        **scored,
        "amount_cents": amount_cents,
        "n_calls_after_confirmed_submit": len(scored["calls_after_confirmed_submit"] or []),
    }


def _load_baseline_pairs() -> list[dict[str, Any]]:
    path = BASELINE_58 / "baseline" / "per_pair.json"
    pairs = json.loads(path.read_text(encoding="utf-8"))["pairs"]
    out = []
    for row in pairs:
        item = dict(row)
        item["n_calls_after_confirmed_submit"] = len(item.get("calls_after_confirmed_submit") or [])
        # Baseline 5.8 used final_answer_is_tool_call after confirmed submit without
        # always filling calls_after_confirmed_submit; count those as post-submit calls.
        if (
            item.get("error_type") == TARGET_FAILURE
            and item.get("final_answer_is_tool_call")
            and item["n_calls_after_confirmed_submit"] == 0
        ):
            item["n_calls_after_confirmed_submit"] = 1
        out.append(item)
    return out


def eval_adapter(adapter_dir: Path) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import agenttune.agentic.rollout_engines.rollout_factory as factory
    from agenttune.agentic.rollout_engines.transformers_engine import TransformersRolloutEngine

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
            print(f"eval B {rid}", flush=True)
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
    _write_json(
        out / "traces.json",
        {
            "rollouts": {
                rid: {
                    "final_answer": steps[-1].get("final_answer"),
                    "task_success": steps[-1].get("task_success"),
                    "task_success_harness": steps[-1].get("task_success_harness"),
                    "task_success_answer": steps[-1].get("task_success_answer"),
                    "function_entries": steps[-1].get("function_entries"),
                    "raw_outputs": [step.get("raw_output") for step in steps],
                }
                for rid, steps in sorted(grouped.items())
            }
        },
    )
    del engine, tokenizer, model
    _release()
    return {"pairs": pairs}


def _post_submit_call_count(row: dict[str, Any]) -> int:
    n = int(row.get("n_calls_after_confirmed_submit") or 0)
    confirmed = bool(row.get("task_success_harness") or row.get("submit_confirmed") or row.get("confirmed"))
    err = row.get("error_type")
    final = row.get("final_answer") or ""
    tool_final = err == TARGET_FAILURE or (
        confirmed and isinstance(final, str) and final.strip().startswith("[{") and "tool_name" in final
    )
    if confirmed and tool_final and n == 0:
        return 1
    return n


def _metric_block(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(pairs)
    full = sum(1 for row in pairs if row["task_success"])
    confirmed = sum(1 for row in pairs if row["task_success_harness"] or row.get("submit_confirmed"))
    correct = sum(1 for row in pairs if row["task_success_answer"])
    post_values = [_post_submit_call_count(row) for row in pairs]
    post = sum(post_values)
    post_eps = sum(1 for value in post_values if value > 0)
    target = sum(1 for row in pairs if row.get("error_type") == TARGET_FAILURE)
    invalid_num = sum(int(row["n_invalid"]) for row in pairs)
    return {
        "full_success": {"numerator": full, "denominator": n, "rate": full / n if n else None},
        "confirmed_returns": {
            "numerator": confirmed,
            "denominator": n,
            "rate": confirmed / n if n else None,
        },
        "correct_finals": {
            "numerator": correct,
            "denominator": n,
            "rate": correct / n if n else None,
        },
        "calls_after_confirmed_submit": {
            "total_calls": post,
            "episodes_with_any": post_eps,
            "denominator_episodes": n,
            "definition": (
                "executed tool calls after confirmed submit_refund; "
                "if none executed but the final answer is a tool call after confirm, count 1"
            ),
        },
        "target_failure_episodes": {
            "numerator": target,
            "denominator": n,
            "error_type": TARGET_FAILURE,
        },
        "invalid_calls_numerator": invalid_num,
    }


def _transitions(baseline: list[dict], pilot: list[dict]) -> list[dict[str, Any]]:
    left = {row["rollout_id"]: row for row in baseline}
    right = {row["rollout_id"]: row for row in pilot}
    if set(left) != set(right):
        raise RuntimeError(f"test ids differ: {sorted(set(left) ^ set(right))}")
    out = []
    for rid in sorted(left):
        a = left[rid]
        b = right[rid]
        out.append(
            {
                "rollout_id": rid,
                "baseline": {
                    "task_success": a["task_success"],
                    "confirmed": a.get("task_success_harness") or a.get("submit_confirmed"),
                    "correct_final": a["task_success_answer"],
                    "error_type": a.get("error_type"),
                    "n_calls_after_confirmed_submit": int(
                        a.get("n_calls_after_confirmed_submit") or 0
                    ),
                    "final_answer": a.get("final_answer"),
                },
                "pilot": {
                    "task_success": b["task_success"],
                    "confirmed": b.get("task_success_harness") or b.get("submit_confirmed"),
                    "correct_final": b["task_success_answer"],
                    "error_type": b.get("error_type"),
                    "n_calls_after_confirmed_submit": int(
                        b.get("n_calls_after_confirmed_submit") or 0
                    ),
                    "final_answer": b.get("final_answer"),
                },
                "changed": {
                    "task_success": a["task_success"] != b["task_success"],
                    "confirmed": (a.get("task_success_harness") or a.get("submit_confirmed"))
                    != (b.get("task_success_harness") or b.get("submit_confirmed")),
                    "correct_final": a["task_success_answer"] != b["task_success_answer"],
                    "n_calls_after_confirmed_submit": int(
                        a.get("n_calls_after_confirmed_submit") or 0
                    )
                    != int(b.get("n_calls_after_confirmed_submit") or 0),
                    "error_type": a.get("error_type") != b.get("error_type"),
                },
            }
        )
    return out


def _success_check(baseline_m: dict, pilot_m: dict) -> dict[str, Any]:
    fewer_post = (
        pilot_m["calls_after_confirmed_submit"]["total_calls"]
        < baseline_m["calls_after_confirmed_submit"]["total_calls"]
    )
    more_full = pilot_m["full_success"]["numerator"] > baseline_m["full_success"]["numerator"]
    confirmed_ok = (
        pilot_m["confirmed_returns"]["numerator"] >= baseline_m["confirmed_returns"]["numerator"]
    )
    return {
        "fewer_calls_after_confirmed_submit": fewer_post,
        "more_full_successes": more_full,
        "confirmed_returns_at_least_baseline": confirmed_ok,
        "pilot_succeeded": fewer_post and more_full and confirmed_ok,
    }


def main() -> int:
    config_payload = write_pilot_config()
    if json.loads((OUT / "pilot_config.json").read_text(encoding="utf-8"))["held_out_ran"]:
        raise RuntimeError("pilot_config was not written before training")
    started = time.monotonic()
    group_results: list[dict[str, Any]] = []
    trained: dict[str, Any] | None = None
    stop_reason = None

    for task in TRAIN_TASKS:
        if time.monotonic() - started >= TIME_LIMIT_S:
            stop_reason = "time_limit_before_group"
            print(f"time limit reached before {task[0]}", flush=True)
            break
        prior = OUT / f"group_{task[0]}.json"
        if prior.exists():
            saved = json.loads(prior.read_text(encoding="utf-8"))
            if saved.get("status") in {"identical_rewards", "not_eligible"} and not saved.get(
                "eligible"
            ):
                saved["reused_from_disk"] = True
                saved["elapsed_s_total"] = time.monotonic() - started
                group_results.append(saved)
                _write_json(OUT / "group_scan.json", {"groups": group_results})
                print(
                    f"reuse prior {task[0]} status={saved.get('status')} eligible=false",
                    flush=True,
                )
                continue
        result = _try_group(task, started)
        result["elapsed_s_total"] = time.monotonic() - started
        group_results.append(result)
        _write_json(OUT / "group_scan.json", {"groups": group_results})
        if result.get("status") == "step_completed":
            trained = result
            stop_reason = "step_completed"
            break
        if result.get("status") == "skipped_time_limit":
            stop_reason = "time_limit"
            break
    else:
        stop_reason = "no_eligible_group"

    evaluation = None
    comparison = None
    if trained is not None:
        evaluation = eval_adapter(OUT / "adapter_B")
        baseline_pairs = _load_baseline_pairs()
        baseline_m = _metric_block(baseline_pairs)
        pilot_m = _metric_block(evaluation["pairs"])
        transitions = _transitions(baseline_pairs, evaluation["pairs"])
        checks = _success_check(baseline_m, pilot_m)
        comparison = {
            "baseline_metrics": baseline_m,
            "pilot_metrics": pilot_m,
            "transitions": transitions,
            "changed_rollouts": [row for row in transitions if any(row["changed"].values())],
            "success_criterion": checks,
            "observation_caveat": (
                "Result is an observation on the 5.8 test matrix that already "
                "selected the target failure; not an independent held-out claim."
            ),
        }
        _write_json(OUT / "eval_B" / "aggregates.json", pilot_m)
        _write_json(OUT / "comparison.json", comparison)

    summary = {
        "kind": "realistic_b_only_one_step_pilot",
        "command": COMMAND,
        "git_head": _git_head(),
        "versions": _package_versions(),
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "pilot_config": config_payload,
        "time_limit_s": TIME_LIMIT_S,
        "elapsed_s": time.monotonic() - started,
        "stop_reason": stop_reason,
        "optimizer_steps": 1 if trained else 0,
        "trained_group": None if trained is None else trained["episode_id"],
        "group_scan": group_results,
        "adapter": None if trained is None else "artifacts/task-5-9/adapter_B",
        "baseline_5_8": "artifacts/task-5-8/baseline",
        "comparison": comparison,
        "trained": trained is not None,
        "held_out_ran": evaluation is not None,
    }
    # Mark config after the fact without rewriting the pre-train pin fields.
    pinned = json.loads((OUT / "pilot_config.json").read_text(encoding="utf-8"))
    pinned["held_out_ran"] = evaluation is not None
    pinned["optimizer_steps_completed"] = 1 if trained else 0
    pinned["stop_reason"] = stop_reason
    _write_json(OUT / "pilot_config.json", pinned)
    _write_json(OUT / "summary.json", summary)
    print(
        json.dumps(
            {
                "stop_reason": stop_reason,
                "optimizer_steps": summary["optimizer_steps"],
                "trained_group": summary["trained_group"],
                "elapsed_s": summary["elapsed_s"],
                "success_criterion": None if comparison is None else comparison["success_criterion"],
                "pilot_metrics": None if comparison is None else comparison["pilot_metrics"],
                "baseline_metrics": None if comparison is None else comparison["baseline_metrics"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
