"""Task 5: prepare the GRPO path and stop before the first optimizer step."""

from __future__ import annotations

import inspect
import json
import platform
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contract import config
from contract.events import events_from_steps
from contract.harness import (
    ONE_ACTION_SYSTEM_PROMPT,
    ONE_ACTION_TOOL_RESULT_FORMAT,
    ONE_ACTION_TOOLS_FALLBACK_PROMPT,
    one_action_user_prompt,
)
from contract.rewards import reward_B, reward_C

TASK5 = ROOT / "artifacts" / "task-5"
DIAGNOSTIC_IDS = ("m0", "m2")

# Fixed splits. No overlap with diagnostic m0/m2 or each other.
TRAIN_TASKS = (
    ("t0", "ORD-2001", "CUS-601", 1500),
    ("t1", "ORD-2002", "CUS-602", 2200),
    ("t2", "ORD-2003", "CUS-603", 875),
    ("t3", "ORD-2004", "CUS-604", 3100),
)
HELD_OUT_TASKS = (
    ("h0", "ORD-3001", "CUS-701", 1640),
    ("h1", "ORD-3002", "CUS-702", 980),
    ("h2", "ORD-3003", "CUS-703", 2450),
    ("h3", "ORD-3004", "CUS-704", 1320),
)

EXPECTED_TASK4 = {
    "m0": {"B": [1.0, 0.0, 0.0, 1.0], "C": [1.0, 0.0, -0.5, 1.0]},
    "m2": {"B": [1.0, 1.0, 1.0, 1.0], "C": [1.0, 1.0, 1.0, 1.0]},
}
SEED_ORDER = (100, 101, 102, 103)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _package_versions() -> dict[str, str]:
    import agenttune
    import peft
    import torch
    import transformers
    import trl

    return {
        "agenttune": getattr(agenttune, "__version__", "unknown"),
        "agenttune_git": "d4db4c12d52342227c4113a70183d0f64aa2f345",
        "trl": trl.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "torch": torch.__version__,
        "python": platform.python_version(),
    }


def _lora_dict() -> dict[str, Any]:
    # Cohere2 is absent from peft's default target map; name modules explicitly.
    return {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }


def pilot_kwargs(reward_name: str) -> dict[str, Any]:
    """Minimal AgentTune/TRL GRPO kwargs. beta is explicit 0.0 for B and for C."""
    if reward_name == "B":
        reward_funcs = reward_B
    elif reward_name == "C":
        # One function, not a weighted sum of two.
        reward_funcs = reward_C
    else:
        raise ValueError(reward_name)
    return {
        "model": config.MODEL_ID,
        "model_init_kwargs": {
            "revision": config.MODEL_REVISION,
            "torch_dtype": "float16",
        },
        "reward_funcs": reward_funcs,
        "peft_config": _lora_dict(),
        "num_generations": 4,
        "beta": 0.0,
        "temperature": config.ONE_ACTION_TEMPERATURE,
        "top_p": config.ONE_ACTION_TOP_P,
        "max_completion_length": config.ONE_ACTION_MAX_NEW_TOKENS,
        "max_steps_per_turn": config.ONE_ACTION_MAX_STEPS,
        # TRL requires generation_batch_size divisible by num_generations.
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-6,
        "max_steps": 1,
        "logging_steps": 1,
        "save_steps": 10_000,
        "report_to": [],
        "use_vllm": False,
        "bf16": False,
        "fp16": False,
        "remove_unused_columns": False,
        "output_dir": str(TASK5 / f"pilot-{reward_name.lower()}-scratch"),
        "system_prompt": ONE_ACTION_SYSTEM_PROMPT,
        "tools_fallback_prompt": ONE_ACTION_TOOLS_FALLBACK_PROMPT,
        "tool_result_format": ONE_ACTION_TOOL_RESULT_FORMAT,
        "seed": config.SEED,
        "prompt_protocol": "one_action_per_turn",
    }


def _prompt_row(episode_id: str, order_id: str, customer_id: str, amount_cents: int, split: str) -> dict:
    return {
        "episode_id": episode_id,
        "order_id": order_id,
        "customer_id": customer_id,
        "amount_cents": amount_cents,
        "split": split,
        "prompt": one_action_user_prompt(order_id, customer_id),
        "environment_seed": config.ONE_ACTION_ENVIRONMENT_SEED,
    }


def write_datasets(out: Path) -> dict[str, Any]:
    train_rows = [_prompt_row(*task, "train") for task in TRAIN_TASKS]
    held_rows = [_prompt_row(*task, "held_out") for task in HELD_OUT_TASKS]
    diagnostic = [
        {
            "episode_id": episode_id,
            "order_id": order_id,
            "customer_id": customer_id,
            "amount_cents": amount,
            "split": "diagnostic",
            "note": "task-3/task-4 variability; excluded from train and held-out",
        }
        for episode_id, order_id, customer_id, amount in config.ONE_ACTION_TASKS
    ]
    train_ids = {row["episode_id"] for row in train_rows}
    held_ids = {row["episode_id"] for row in held_rows}
    diag_ids = {row["episode_id"] for row in diagnostic}
    overlap = (train_ids & held_ids) | (train_ids & diag_ids) | (held_ids & diag_ids)
    if overlap:
        raise RuntimeError(f"dataset id overlap: {sorted(overlap)}")
    order_ids = (
        {row["order_id"] for row in train_rows}
        | {row["order_id"] for row in held_rows}
        | {row["order_id"] for row in diagnostic}
    )
    if len(order_ids) != len(train_rows) + len(held_rows) + len(diagnostic):
        raise RuntimeError("order_id collision across splits")

    _dump_jsonl(out / "train.jsonl", train_rows)
    _dump_jsonl(out / "held_out.jsonl", held_rows)
    _dump_jsonl(out / "diagnostic.jsonl", diagnostic)
    manifest = {
        "train": {
            "path": str(out / "train.jsonl"),
            "n": len(train_rows),
            "episode_ids": sorted(train_ids),
            "order_ids": sorted(row["order_id"] for row in train_rows),
        },
        "held_out": {
            "path": str(out / "held_out.jsonl"),
            "n": len(held_rows),
            "episode_ids": sorted(held_ids),
            "order_ids": sorted(row["order_id"] for row in held_rows),
        },
        "diagnostic": {
            "path": str(out / "diagnostic.jsonl"),
            "n": len(diagnostic),
            "episode_ids": sorted(diag_ids),
            "note": "m0/m2 remain diagnostic only",
        },
        "overlap": sorted(overlap),
        "prompt_protocol": "one_action_per_turn",
    }
    _write_json(out / "manifest.json", manifest)
    return manifest


def _load_grouped_steps() -> dict[str, dict[str, list[dict]]]:
    rows: list[dict] = []
    for path in (
        ROOT / "artifacts" / "task-3" / "model_episodes.jsonl",
        ROOT / "artifacts" / "task-4" / "model_episodes.jsonl",
    ):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    by_rollout: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_rollout[row["rollout_id"]].append(row)
    grouped: dict[str, dict[str, list[dict]]] = {"m0": {}, "m2": {}}
    for episode_id in DIAGNOSTIC_IDS:
        for seed in SEED_ORDER:
            rid = f"{episode_id}-g{seed}"
            steps = by_rollout.get(rid)
            if not steps:
                raise FileNotFoundError(f"missing saved trajectory {rid}")
            source = (
                "artifacts/task-3/model_episodes.jsonl"
                if seed == 100
                else "artifacts/task-4/model_episodes.jsonl"
            )
            for step in steps:
                step = dict(step)
            grouped[episode_id][rid] = {
                "steps": steps,
                "source": source,
            }
    return grouped


def run_scripted_replay(out: Path) -> dict[str, Any]:
    grouped = _load_grouped_steps()
    replay_rows = []
    comparison = {}
    callback_proof = {}
    for episode_id, expected in EXPECTED_TASK4.items():
        actual_b = []
        actual_c = []
        per_rollout = []
        for seed in SEED_ORDER:
            rid = f"{episode_id}-g{seed}"
            bundle = grouped[episode_id][rid]
            steps = bundle["steps"]
            events = events_from_steps(steps, rollout_id=rid, source=bundle["source"])
            # Same kwargs shape the trainer reward path uses after extra_fields merge.
            completions = [events["final_answer"]]
            got_b = reward_B(completions=completions, contract_events=[events])[0]
            got_c = reward_C(completions=completions, contract_events=[events])[0]
            actual_b.append(got_b)
            actual_c.append(got_c)
            proof = {
                "rollout_id": rid,
                "source": bundle["source"],
                "callback_received": {
                    "raw_outputs": events["raw_outputs"],
                    "validation_results": events["validation_results"],
                    "harness_state": events["harness_state"],
                    "final_answer": events["final_answer"],
                },
                "fields_present": events["callback_fields_present"],
                "n_invalid": events["n_invalid"],
                "task_success": events["task_success"],
                "B": got_b,
                "C": got_c,
            }
            callback_proof[rid] = {
                "raw_output_n": len(events["raw_outputs"]),
                "validation_n": len(events["validation_results"]),
                "harness_state": events["harness_state"],
                "final_answer": events["final_answer"],
                "fields_present": events["callback_fields_present"],
            }
            per_rollout.append(proof)
            replay_rows.append(
                {
                    "kind": "scripted_replay",
                    "rollout_id": rid,
                    "episode_id": episode_id,
                    "generation_seed": seed,
                    "source": bundle["source"],
                    "events": events,
                    "B": got_b,
                    "C": got_c,
                }
            )
        comparison[episode_id] = {
            "expected_B": expected["B"],
            "actual_B": actual_b,
            "match_B": actual_b == expected["B"],
            "expected_C": expected["C"],
            "actual_C": actual_c,
            "match_C": actual_c == expected["C"],
            "rollouts": per_rollout,
        }
    payload = {
        "kind": "scripted_replay",
        "note": (
            "Rewards below are from the shared reward callback on saved trajectories. "
            "They are not trainer optimizer logs."
        ),
        "trainer_ran": False,
        "optimizer_step": False,
        "comparison": comparison,
        "callback_proof": callback_proof,
        "all_match": all(
            item["match_B"] and item["match_C"] for item in comparison.values()
        ),
    }
    _write_json(out / "scripted_replay.json", payload)
    _dump_jsonl(out / "scripted_replay_events.jsonl", replay_rows)
    return payload


def _inspect_trainer_settings(reward_name: str, out: Path) -> dict[str, Any]:
    """Build TrlAgenticGrpo + GRPOTrainer and record the settings it actually holds.

    Does not call train(). A safety callback would abort if train() were invoked.
    """
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import TrainerCallback

    from agenttune.backends.trl.agentic.grpo.agentic_grpo import TrlAgenticGrpo, _split_kwargs

    class StopBeforeOptimizer(TrainerCallback):
        def on_optimizer_step(self, args, state, control, **kwargs):
            raise RuntimeError(
                "StopBeforeOptimizer: refusing the first optimizer step (task 5 stop gate)"
            )

        def on_step_begin(self, args, state, control, **kwargs):
            # Belt-and-suspenders if an older transformers build hits this first.
            if state.global_step >= 0 and getattr(state, "max_steps", None) == 0:
                control.should_training_stop = True
            return control

    kwargs = pilot_kwargs(reward_name)
    # Tiny dataset so setup_data/setup_trainer can bind without real optimization.
    probe_prompt = one_action_user_prompt("ORD-2001", "CUS-601")
    ds = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": probe_prompt}], "episode_id": "t0"}]
    )
    kwargs = {
        **kwargs,
        "train_dataset": ds,
        "callbacks": [StopBeforeOptimizer()],
        "max_steps": 0,
    }
    # Resolve config the same way TrlAgenticGrpo.setup_trainer does, then build trainer.
    # model_init_kwargs is only for string-model construction; drop it when we
    # instantiate the model ourselves below.
    config_kw, trainer_kw = _split_kwargs(kwargs)
    config_kw.pop("model_init_kwargs", None)
    trainer_kw.pop("model_init_kwargs", None)
    # Mirror the defaults block that setup_trainer applies.
    from agenttune.backends.trl.agentic.grpo.agentic_grpo import _get
    from trl import GRPOConfig, GRPOTrainer

    from agenttune.agentic.rewards.composite import combine_rewards

    output_dir = kwargs["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    config_defaults = {
        "output_dir": output_dir,
        "num_train_epochs": 1,
        "per_device_train_batch_size": kwargs["per_device_train_batch_size"],
        "gradient_accumulation_steps": kwargs["gradient_accumulation_steps"],
        "learning_rate": kwargs["learning_rate"],
        "max_steps": 0,
        "logging_steps": 1,
        "save_steps": 10_000,
        "remove_unused_columns": False,
        "num_generations": kwargs["num_generations"],
        "max_completion_length": kwargs["max_completion_length"],
        "temperature": kwargs["temperature"],
        "top_p": kwargs["top_p"],
        "beta": kwargs["beta"],
        "report_to": [],
        "bf16": False,
        "fp16": False,
        "seed": kwargs["seed"],
        "use_vllm": False,
    }
    for key, value in config_defaults.items():
        config_kw.setdefault(key, value)
    grpo_config = GRPOConfig(**config_kw)

    reward_funcs = combine_rewards(kwargs["reward_funcs"])
    peft_config = LoraConfig(**kwargs["peft_config"])
    from agenttune.agentic.rollout_engines.rollout_factory import ensure_lora_targets

    peft_config = ensure_lora_targets(peft_config, kwargs["model"])

    received = {
        "reward_name": reward_name,
        "reward_func_passed": getattr(kwargs["reward_funcs"], "__name__", str(kwargs["reward_funcs"])),
        "reward_func_after_combine_rewards": getattr(reward_funcs, "__name__", str(reward_funcs)),
        "beta_requested": 0.0,
        "beta_on_GRPOConfig": grpo_config.beta,
        "num_generations": grpo_config.num_generations,
        "temperature": grpo_config.temperature,
        "top_p": grpo_config.top_p,
        "max_completion_length": grpo_config.max_completion_length,
        "learning_rate": grpo_config.learning_rate,
        "max_steps": grpo_config.max_steps,
        "per_device_train_batch_size": grpo_config.per_device_train_batch_size,
        "use_vllm": getattr(grpo_config, "use_vllm", None),
        "peft_config": {
            "r": peft_config.r,
            "lora_alpha": peft_config.lora_alpha,
            "lora_dropout": peft_config.lora_dropout,
            "target_modules": sorted(peft_config.target_modules)
            if isinstance(peft_config.target_modules, (list, set))
            else peft_config.target_modules,
            "task_type": str(peft_config.task_type),
        },
        "model": kwargs["model"],
        "model_revision": kwargs["model_init_kwargs"]["revision"],
        "prompt_protocol": kwargs["prompt_protocol"],
        "max_steps_per_turn": kwargs["max_steps_per_turn"],
        "trainer_built": False,
        "train_called": False,
        "optimizer_step": False,
        "note": (
            "GRPOConfig and peft_config are the objects the trainer receives. "
            "GRPOTrainer is constructed with max_steps=0 and a StopBeforeOptimizer "
            "callback; train() is not called."
        ),
    }

    # Construct GRPOTrainer so peft/model wiring is real, then discard without train().
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        kwargs["model"], revision=config.MODEL_REVISION
    )
    model = AutoModelForCausalLM.from_pretrained(
        kwargs["model"],
        revision=config.MODEL_REVISION,
        torch_dtype=torch.float16,
        device_map=None,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[StopBeforeOptimizer()],
    )
    received["trainer_built"] = True
    received["trainer_class"] = type(trainer).__name__
    received["trainer_args_beta"] = trainer.args.beta
    received["trainer_args_num_generations"] = trainer.args.num_generations
    received["trainer_has_peft"] = bool(getattr(trainer, "model", None))
    received["stop_callback"] = "StopBeforeOptimizer"
    # Do not call trainer.train().
    del trainer
    del model
    _write_json(out / f"trainer_settings_{reward_name}.json", received)
    return received


def wrap_rollout_with_contract_events(rollout_fn, events_by_index: list[dict] | None = None):
    """Ensure contract_events survive into TRL reward kwargs.

    AgentTune's _format_for_grpo forwards `trajectories` but not an explicit
    `contract_events` column. This wrapper copies metadata['contract_events']
    onto a top-level list so the reward callback cannot miss them.
    """

    def wrapped(prompts, *args, **kwargs):
        result = rollout_fn(prompts, *args, **kwargs)
        trajectories = result.get("trajectories") or []
        events = []
        for i, traj in enumerate(trajectories):
            meta = getattr(traj, "metadata", None) or {}
            item = meta.get("contract_events")
            if item is None and events_by_index is not None and i < len(events_by_index):
                item = events_by_index[i]
                meta = dict(meta)
                meta["contract_events"] = item
                traj.metadata = meta
            if item is None:
                raise ValueError(
                    f"rollout trajectory[{i}] missing contract_events; "
                    "fix the instrumented path before training"
                )
            events.append(item)
        result["contract_events"] = events
        return result

    wrapped.__name__ = getattr(rollout_fn, "__name__", "rollout") + "_with_contract_events"
    return wrapped


def held_out_metrics_from_rows(rows: list[dict]) -> dict[str, Any]:
    by_rollout: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_rollout[row["rollout_id"]].append(row)
    n_tasks = len(by_rollout)
    successes = 0
    invalid_attempts = 0
    attempt_denominator = 0
    no_call_tasks = 0
    calls_per_task = []
    per_task = []
    for rid, steps in sorted(by_rollout.items()):
        last = steps[-1]
        success = bool(last.get("task_success"))
        successes += int(success)
        components = events_from_steps(steps)["n_invalid_components"]
        # Denominator: every parsed call attempt plus every malformed_syntax fragment.
        attempts = 0
        for step in steps:
            attempts += len(step.get("calls") or [])
            attempts += len((step.get("diagnostic") or {}).get("malformed_syntax") or [])
        attempt_denominator += attempts
        invalid_attempts += int(components["N_invalid_provisional"])
        n_calls = sum(len(step.get("calls") or []) for step in steps)
        calls_per_task.append(n_calls)
        if n_calls == 0:
            no_call_tasks += 1
        per_task.append(
            {
                "rollout_id": rid,
                "episode_id": last.get("episode_id"),
                "task_success": success,
                "N_invalid": components["N_invalid_provisional"],
                "attempt_count": attempts,
                "parsed_calls": n_calls,
            }
        )
    return {
        "n_tasks": n_tasks,
        "task_success": {
            "numerator": successes,
            "denominator": n_tasks,
            "rate": (successes / n_tasks) if n_tasks else None,
        },
        "invalid_attempts": {
            "numerator": invalid_attempts,
            "denominator": attempt_denominator,
            "rate": (invalid_attempts / attempt_denominator)
            if attempt_denominator
            else None,
            "definition": (
                "numerator = provisional N_invalid "
                "(malformed_syntax + missing_name + unknown_tool + schema_invalid); "
                "denominator = parsed call attempts + malformed_syntax fragments"
            ),
        },
        "tasks_without_calls": {
            "numerator": no_call_tasks,
            "denominator": n_tasks,
            "rate": (no_call_tasks / n_tasks) if n_tasks else None,
        },
        "calls_per_task": {
            "values": calls_per_task,
            "mean": (sum(calls_per_task) / len(calls_per_task)) if calls_per_task else None,
        },
        "per_task": per_task,
        "model_weights_updated": False,
    }


def run_held_out_baseline(out: Path) -> dict[str, Any]:
    """Untrained one-action rollouts on the held-out set. No optimizer."""
    from contract.run import (
        _bind_output,
        _dump,
        _format_rollout_view,
        _load_model,
        _one_model_rollout,
        _probe,
        make_guarded_tools,
    )
    from contract.harness import RefundEpisode
    import agenttune.agentic.rollout_engines.rollout_factory as factory

    held_dir = out / "held_out_baseline"
    _bind_output(held_dir)
    held_dir.mkdir(parents=True, exist_ok=True)
    engine, tokenizer, loaded = _load_model()
    _probe_guards, schema_list = make_guarded_tools(
        RefundEpisode("schema-probe", "ORD-3001", "CUS-701", 1640, config.SEED)
    )
    probe = _probe(tokenizer, schema_list)
    rows: list[dict] = []
    for episode_id, order_id, customer_id, amount in HELD_OUT_TASKS:
        print(f"held_out_baseline {episode_id} g100", flush=True)
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
            generation_seed=100,
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
        _dump(held_dir / "model_episodes.jsonl", rows)

    metrics = held_out_metrics_from_rows(rows)
    metrics["loaded_model"] = loaded
    metrics["sampling"] = {
        "generation_seed": 100,
        "do_sample": config.ONE_ACTION_DO_SAMPLE,
        "temperature": config.ONE_ACTION_TEMPERATURE,
        "top_p": config.ONE_ACTION_TOP_P,
        "max_steps": config.ONE_ACTION_MAX_STEPS,
        "max_new_tokens": config.ONE_ACTION_MAX_NEW_TOKENS,
        "prompt_protocol": "one_action_per_turn",
    }
    _write_json(held_dir / "metrics.json", metrics)
    return metrics


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    skip_held_out = "--skip-held-out" in argv
    skip_trainer_build = "--skip-trainer-build" in argv

    out = TASK5
    out.mkdir(parents=True, exist_ok=True)
    versions = _package_versions()
    datasets = write_datasets(out / "datasets")
    replay = run_scripted_replay(out)

    pilots = {
        "B": pilot_kwargs("B"),
        "C": pilot_kwargs("C"),
    }
    # Serialize reward funcs by name for the manifest.
    pilots_public = {}
    for name, kwargs in pilots.items():
        public = dict(kwargs)
        public["reward_funcs"] = getattr(kwargs["reward_funcs"], "__name__", str(kwargs["reward_funcs"]))
        public["beta"] = 0.0
        pilots_public[name] = public
    _write_json(out / "pilot_configs.json", pilots_public)

    trainer_settings = {}
    trainer_error = None
    if not skip_trainer_build:
        try:
            for name in ("B", "C"):
                print(f"inspect_trainer {name}", flush=True)
                trainer_settings[name] = _inspect_trainer_settings(name, out)
        except Exception as exc:
            trainer_error = {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            (out / "trainer_build_error.txt").write_text(
                trainer_error["traceback"], encoding="utf-8"
            )
            # If the trainer path cannot be built, stop before claiming readiness.
            _write_json(
                out / "summary.json",
                {
                    "stopped": True,
                    "reason": "trainer_build_failed",
                    "trainer_error": trainer_error,
                    "scripted_replay_all_match": replay["all_match"],
                    "optimizer_step": False,
                },
            )
            print(json.dumps({"stopped": "trainer_build_failed", "error": trainer_error["error"]}, indent=2))
            return 2

    held_metrics = None
    if not skip_held_out:
        try:
            held_metrics = run_held_out_baseline(out)
        except Exception as exc:
            (out / "held_out_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            _write_json(
                out / "summary.json",
                {
                    "stopped": True,
                    "reason": "held_out_baseline_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "scripted_replay_all_match": replay["all_match"],
                    "optimizer_step": False,
                },
            )
            return 2

    path_notes = {
        "completions_only": "disabled; reward_B/reward_C raise without contract_events",
        "trajectories_metadata": (
            "contract_events must sit on Trajectory.metadata['contract_events']"
        ),
        "extra_fields_column": (
            "wrap_rollout_with_contract_events copies events to a top-level "
            "contract_events list so TRL merges them into reward kwargs"
        ),
        "combine_rewards": (
            "C passes a single callable into TrlAgenticGrpo; combine_rewards may "
            "wrap it but must not sum a separate B term into C"
        ),
    }

    summary = {
        "kind": "grpo_path_prep",
        "versions": versions,
        "datasets": datasets,
        "scripted_replay": {
            "all_match": replay["all_match"],
            "comparison": {
                episode_id: {
                    "expected_B": item["expected_B"],
                    "actual_B": item["actual_B"],
                    "match_B": item["match_B"],
                    "expected_C": item["expected_C"],
                    "actual_C": item["actual_C"],
                    "match_C": item["match_C"],
                }
                for episode_id, item in replay["comparison"].items()
            },
            "callback_proof": replay["callback_proof"],
            "log_kind": "scripted_replay",
        },
        "trainer_logs": {
            "log_kind": "trainer_settings_only",
            "train_called": False,
            "optimizer_step": False,
            "settings": trainer_settings,
            "error": trainer_error,
        },
        "pilot_configs": pilots_public,
        "path_notes": path_notes,
        "held_out_baseline": held_metrics,
        "stopped_before_optimizer": True,
        "grpo_training_started": False,
    }
    _write_json(out / "summary.json", summary)
    _write_json(
        out / "reward_table.json",
        {
            episode_id: {
                "expected_B": item["expected_B"],
                "actual_B": item["actual_B"],
                "expected_C": item["expected_C"],
                "actual_C": item["actual_C"],
                "match": item["match_B"] and item["match_C"],
            }
            for episode_id, item in replay["comparison"].items()
        },
    )

    print(
        json.dumps(
            {
                "scripted_replay_all_match": replay["all_match"],
                "reward_table": summary["scripted_replay"]["comparison"],
                "trainer_built": {
                    name: settings.get("trainer_built")
                    for name, settings in trainer_settings.items()
                },
                "held_out_task_success": None
                if held_metrics is None
                else held_metrics["task_success"],
                "optimizer_step": False,
            },
            indent=2,
        )
    )
    return 0 if replay["all_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
