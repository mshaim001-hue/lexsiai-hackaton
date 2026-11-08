"""Task 5.1: real trainer → AgentTune rollout → reward path, stop before optimizer."""

from __future__ import annotations

import copy
import json
import sys
import time
import traceback
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contract import config
from contract.events import attach_events_to_trajectory, events_from_steps
from contract.execute import PATH, make_guarded_tools
from contract.grpo_prep import (
    EXPECTED_TASK4,
    HELD_OUT_TASKS,
    SEED_ORDER,
    TRAIN_TASKS,
    _lora_dict,
    _package_versions,
    _write_json,
)
from contract.harness import (
    ONE_ACTION_SYSTEM_PROMPT,
    ONE_ACTION_TOOL_RESULT_FORMAT,
    ONE_ACTION_TOOLS_FALLBACK_PROMPT,
    RefundEpisode,
    one_action_user_prompt,
)
from contract.rewards import reward_B, reward_C
from contract.run import (
    _RecordingEngine,
    _assign_budget,
    _consume_guard_records,
    _finish,
    _generation_diagnosis,
    _install_extract_hook,
    _seed_generation,
    _base_step,
)

TASK51 = ROOT / "artifacts" / "task-5-1"


class _PrefixEncoder:
    """Tokenizer stand-in so the rollout can reuse the live engine's chat template."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._chat_template = None


def _encode_rollout_prefix(tokenizer, prompts, tools, gen_cfg) -> list[int]:
    """Token ids of the prompt that generate() received, using the live engine's template."""
    from agenttune.agentic.rollout_engines.transformers_engine import TransformersRolloutEngine

    if isinstance(prompts, str) or (isinstance(prompts, list) and prompts and isinstance(prompts[0], str)):
        raw = [prompts] if isinstance(prompts, str) else list(prompts)
        conversations = [[{"role": "user", "content": text}] for text in raw]
    elif isinstance(prompts, list) and prompts and isinstance(prompts[0], dict) and "role" in prompts[0]:
        conversations = [prompts]
    else:
        conversations = prompts
    enable_thinking = bool((gen_cfg or {}).get("enable_thinking", False))
    tools_arg = tools if tools else None
    if not hasattr(_PrefixEncoder, "_tool_content_survives_render"):
        _PrefixEncoder._tool_content_survives_render = TransformersRolloutEngine._tool_content_survives_render
    encoded = TransformersRolloutEngine._apply_chat_template(
        _PrefixEncoder(tokenizer),
        conversations,
        tools_arg,
        enable_thinking,
    )
    ids = encoded["input_ids"]
    if hasattr(ids, "detach"):
        row = ids[0].detach().cpu().tolist()
    elif isinstance(ids, list) and ids and isinstance(ids[0], list):
        row = list(ids[0])
    else:
        row = list(ids)
    max_length = int((gen_cfg or {}).get("max_length", config.CONTEXT_LENGTH))
    max_new = int((gen_cfg or {}).get("max_new_tokens", config.ONE_ACTION_MAX_NEW_TOKENS))
    budget = max_length - max_new
    if budget > 0 and len(row) > budget:
        row = row[-budget:]
    return [int(token) for token in row]


class _TurnIdEngine:
    """Record the input_ids and new_token_ids of every generate() call in a rollout."""

    def __init__(self, inner, tokenizer):
        self._inner = inner
        self._tokenizer = tokenizer
        self.calls: list[dict[str, Any]] = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def generate(self, prompts, tools, gen_cfg):
        captured: list[int] | None = None
        model = getattr(self._inner, "model", None)
        if model is not None and hasattr(model, "generate"):
            original = model.generate
            box: dict[str, list[int]] = {}

            def hooked_generate(*args, **kwargs):
                ids = kwargs.get("input_ids")
                if ids is None and args:
                    ids = args[0]
                if ids is not None and hasattr(ids, "detach"):
                    row = ids[0] if getattr(ids, "dim", lambda: 1)() > 1 else ids
                    box["ids"] = [int(token) for token in row.detach().cpu().tolist()]
                return original(*args, **kwargs)

            model.generate = hooked_generate
            try:
                result = self._inner.generate(prompts, tools, gen_cfg)
            finally:
                model.generate = original
            captured = box.get("ids")
        else:
            captured = _encode_rollout_prefix(self._tokenizer, prompts, tools, gen_cfg)
            result = self._inner.generate(prompts, tools, gen_cfg)
        if not captured:
            raise RuntimeError("rollout generate did not yield input_ids")
        new_ids = ((result.get("metadata") or {}).get("completion_ids") or [[]])[0]
        if hasattr(new_ids, "tolist"):
            new_ids = new_ids.tolist()
        new_ids = [int(token) for token in list(new_ids)]
        if not new_ids:
            raise RuntimeError("rollout generate did not yield new_token_ids")
        self.calls.append({"input_ids": list(captured), "new_token_ids": new_ids})
        return result


class TokenizingScriptedEngine:
    """Replay saved raw strings at the generate() boundary; tokenize for masks."""

    def __init__(self, outputs: list[str], tokenizer):
        self._outputs = list(outputs)
        self._tokenizer = tokenizer
        self.records: list[dict] = []

    def _get_tokenizer(self):
        return self._tokenizer

    def generate(self, prompts, tools, gen_cfg):
        if not self._outputs:
            raise RuntimeError(
                f"scripted engine exhausted after {len(self.records)} generations"
            )
        raw = self._outputs.pop(0)
        ids = self._tokenizer.encode(raw, add_special_tokens=False)
        self.records.append(
            {
                "messages": copy.deepcopy(prompts),
                "rendered_prompt": None,
                "tools_passed": copy.deepcopy(tools),
                "completions_raw": raw,
                "new_token_count": len(ids),
                "generation_truncated": False,
                "prompt_len": None,
                "prompt_truncated": False,
                "warnings": [],
                "saved_at": "generate_return",
            }
        )
        return {
            "completions": [raw],
            "completions_raw": [raw],
            "completions_structured": [{"role": "assistant", "content": raw}],
            "logprobs": [[0.0] * len(ids)],
            "metadata": {"completion_ids": [ids], "prompt_len": 0},
        }


def _load_m0_scripted_raws() -> dict[int, list[str]]:
    rows: list[dict] = []
    for path in (
        ROOT / "artifacts" / "task-3" / "model_episodes.jsonl",
        ROOT / "artifacts" / "task-4" / "model_episodes.jsonl",
    ):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    by: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("episode_id") == "m0":
            by[row["rollout_id"]].append(row)
    out: dict[int, list[str]] = {}
    for seed in SEED_ORDER:
        rid = f"m0-g{seed}"
        steps = sorted(by[rid], key=lambda item: item["step_id"])
        out[seed] = [step["raw_output"] for step in steps]
        if not out[seed]:
            raise FileNotFoundError(f"no raw outputs for {rid}")
    return out


def _task_tuple(episode_id: str) -> tuple[str, str, str, int]:
    for spec in (*TRAIN_TASKS, *HELD_OUT_TASKS, *config.ONE_ACTION_TASKS):
        if spec[0] == episode_id:
            return spec
    raise KeyError(episode_id)


def _run_one_instrumented(
    *,
    factory,
    engine,
    tokenizer,
    episode: RefundEpisode,
    generation_seed: int,
    prompt_text: str,
    gen_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], list[dict], dict[str, Any]]:
    """One generation: fresh episode tools, live parse/validate/execute, contract_events."""
    import agenttune.agentic.rollout_engines.rollout_factory as factory_mod
    from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn

    guards, schema_list = make_guarded_tools(episode)
    tool_names = {guard.__name__ for guard in guards}
    gen_log: list[dict] = []
    extract_log: list[dict] = []
    wrapped = _RecordingEngine(engine, gen_log, tokenizer)
    original = factory_mod._extract_tool_calls
    factory_mod._extract_tool_calls = original
    _install_extract_hook(factory_mod, extract_log)
    rollout = create_rollout_fn(
        rollout_engine=wrapped,
        tools=guards,
        max_steps=config.ONE_ACTION_MAX_STEPS,
        system_prompt=ONE_ACTION_SYSTEM_PROMPT,
        enable_thinking=config.ENABLE_THINKING,
        force_final_answer=config.FORCE_FINAL_ANSWER,
        force_action_on_stall=config.FORCE_ACTION_ON_STALL,
        context_length=config.CONTEXT_LENGTH,
        tools_fallback_prompt=ONE_ACTION_TOOLS_FALLBACK_PROMPT,
        tool_result_format=ONE_ACTION_TOOL_RESULT_FORMAT,
    )
    try:
        # Do not pass the GRPO trainer into the inner loop: generation must go
        # through our engine (scripted or live), not trainer._generate_single_turn.
        result = rollout([prompt_text], **gen_kwargs)
    finally:
        factory_mod._extract_tool_calls = original

    trajectory = result["trajectories"][0]
    final_answer = trajectory.final_response
    loop_steps = list(trajectory.steps or [])
    cursor = [0]
    used_extract = 0
    ep_rows: list[dict] = []
    for step_id, generated in enumerate(gen_log):
        raw = generated["completions_raw"]
        extract = None
        if used_extract < len(extract_log) and extract_log[used_extract]["raw_output"] == raw:
            extract = extract_log[used_extract]
            used_extract += 1
        parser_invoked = extract is not None
        parser_result = extract["parser_result"] if parser_invoked else None
        diagnostic, parser_result, pending, _reason = _generation_diagnosis(
            raw, parser_invoked, parser_result, original, episode
        )
        calls = (
            _consume_guard_records(parser_result, tool_names, episode, cursor)
            if parser_invoked
            else []
        )
        thought = None
        if diagnostic["parse_status"] == "parsed" and loop_steps:
            thought = loop_steps.pop(0).thought
        row = _base_step("model", episode, step_id)
        row.update(
            {
                "rollout_id": f"{episode.episode_id}-g{generation_seed}",
                "generation_seed": generation_seed,
                "environment_seed": config.ONE_ACTION_ENVIRONMENT_SEED,
                "prompt_protocol": "one_action_per_turn",
                "raw_output": raw,
                "raw_output_source": "generation_engine",
                "loop_step_thought": thought,
                "parser_invoked_by_loop": parser_invoked,
                "parser_result": parser_result,
                "diagnostic": diagnostic,
                "diagnostic_executed": False,
                "parse_status": diagnostic["parse_status"],
                "malformed_syntax": diagnostic["malformed_syntax"],
                "omitted_by_native_parser": diagnostic["omitted_by_native_parser"],
                "truncated_fragments": diagnostic["truncated_fragments"],
                "calls": calls,
                "pending_unexecuted_calls": pending,
                "function_entries": list(episode.function_entries),
                "execution_path": PATH,
                "generation_truncated": generated.get("generation_truncated"),
                "new_token_count": generated.get("new_token_count"),
            }
        )
        ep_rows.append(row)
    _finish(ep_rows, episode, final_answer, elapsed=None)
    _assign_budget(ep_rows)
    events = events_from_steps(
        ep_rows,
        rollout_id=f"{episode.episode_id}-g{generation_seed}",
        source="live_instrumented_loop",
    )
    attach_events_to_trajectory(trajectory, events)
    # Re-pack single-trajectory GRPO fields.
    from agenttune.agentic.rollout_engines.rollout_factory import _format_for_grpo

    packed = _format_for_grpo([trajectory], [prompt_text], tools_enabled=True)
    packed["contract_events"] = [events]
    packed["trajectories"] = [trajectory]
    return packed, ep_rows, events


def make_instrumented_rollout_func(
    *,
    episode_id: str,
    order_id: str,
    customer_id: str,
    amount_cents: int,
    generation_seeds: list[int],
    tokenizer,
    mode: str,
    scripted_raws: dict[int, list[str]] | None = None,
    live_engine_factory: Callable | None = None,
) -> Callable:
    """rollout_func for GRPOTrainer: one fresh episode state per generation."""

    prompt_text = one_action_user_prompt(order_id, customer_id)
    gen_kwargs = {
        "max_new_tokens": config.ONE_ACTION_MAX_NEW_TOKENS,
        "temperature": config.ONE_ACTION_TEMPERATURE,
        "do_sample": config.ONE_ACTION_DO_SAMPLE,
        "top_p": config.ONE_ACTION_TOP_P,
        "max_length": config.CONTEXT_LENGTH,
    }

    def rollout_fn(prompts, trainer=None, *args, **kwargs):
        import agenttune.agentic.rollout_engines.rollout_factory as factory

        if isinstance(prompts, str):
            prompt_list = [prompts]
        else:
            prompt_list = list(prompts)
        n = len(prompt_list)
        if n != len(generation_seeds):
            raise ValueError(
                f"prompt count {n} != generation_seeds {len(generation_seeds)}; "
                "contract_events must align 1:1 with completions"
            )

        merged = {
            "prompt_ids": [],
            "completion_ids": [],
            "logprobs": [],
            "env_mask": [],
            "responses": [],
            "trajectories": [],
            "contract_events": [],
            "conversations": [],
            "tool_call_counts": [],
            "generation_seeds": [],
            "rollout_ids": [],
            "turn_input_ids": [],
            "turn_new_token_ids": [],
        }
        for i, _prompt in enumerate(prompt_list):
            seed = generation_seeds[i]
            episode = RefundEpisode(
                episode_id=episode_id,
                order_id=order_id,
                customer_id=customer_id,
                amount_cents=amount_cents,
                seed=config.ONE_ACTION_ENVIRONMENT_SEED,
            )
            if mode == "scripted":
                if scripted_raws is None or seed not in scripted_raws:
                    raise KeyError(f"missing scripted raws for seed {seed}")
                engine = TokenizingScriptedEngine(list(scripted_raws[seed]), tokenizer)
            elif mode == "live":
                if live_engine_factory is None:
                    raise RuntimeError("live_engine_factory required for live mode")
                _seed_generation(seed)
                engine = live_engine_factory()
            else:
                raise ValueError(mode)
            engine = _TurnIdEngine(engine, tokenizer)

            packed, _steps, events = _run_one_instrumented(
                factory=factory,
                engine=engine,
                tokenizer=tokenizer,
                episode=episode,
                generation_seed=seed,
                prompt_text=prompt_text,
                gen_kwargs=gen_kwargs,
            )
            if not events:
                raise RuntimeError(f"empty contract_events for generation seed {seed}")
            for key in (
                "prompt_ids",
                "completion_ids",
                "logprobs",
                "env_mask",
                "responses",
                "trajectories",
                "conversations",
                "tool_call_counts",
            ):
                values = packed.get(key) or []
                if isinstance(values, list):
                    merged[key].extend(values)
                else:
                    merged[key].append(values)
            merged["contract_events"].append(events)
            merged["generation_seeds"].append(seed)
            merged["rollout_ids"].append(events["rollout_id"])
            if not engine.calls:
                raise RuntimeError(f"rollout seed {seed} captured no turns")
            merged["turn_input_ids"].append([list(call["input_ids"]) for call in engine.calls])
            merged["turn_new_token_ids"].append([list(call["new_token_ids"]) for call in engine.calls])

        if len(merged["contract_events"]) != n:
            raise RuntimeError(
                f"contract_events ({len(merged['contract_events'])}) != completions ({n})"
            )
        if any(item is None for item in merged["contract_events"]):
            raise RuntimeError("contract_events contains None")
        # TRL pops env_mask from extra_fields before merging into reward inputs.
        # Keep an explicit copy under loss_masks so the scoring check can audit it.
        merged["loss_masks"] = [
            list(mask) if isinstance(mask, list) else mask for mask in (merged.get("env_mask") or [])
        ]
        if len(merged["loss_masks"]) != n:
            raise RuntimeError(
                f"loss_masks ({len(merged['loss_masks'])}) != completions ({n})"
            )
        rollout_fn.last_loss_masks = merged["loss_masks"]
        rollout_fn.last_turn_input_ids = merged["turn_input_ids"]
        rollout_fn.last_turn_new_token_ids = merged["turn_new_token_ids"]
        return merged

    rollout_fn.__name__ = f"instrumented_rollout_{episode_id}_{mode}"
    rollout_fn.last_loss_masks = []
    rollout_fn.last_turn_input_ids = []
    rollout_fn.last_turn_new_token_ids = []
    return rollout_fn


def _build_trainer(tokenizer, model, rollout_fn, output_dir: Path):
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset

    from agenttune.agentic.rollout_engines.rollout_factory import ensure_lora_targets

    class StopBeforeOptimizer(TrainerCallback):
        def __init__(self):
            self.optimizer_steps = 0

        def on_optimizer_step(self, args, state, control, **kwargs):
            self.optimizer_steps += 1
            raise RuntimeError(
                "StopBeforeOptimizer: refusing optimizer.step (task 5.1)"
            )

    stop_cb = StopBeforeOptimizer()
    probe = one_action_user_prompt("ORD-2001", "CUS-601")
    ds = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": probe}], "episode_id": "probe"}]
    )
    args = GRPOConfig(
        output_dir=str(output_dir),
        num_generations=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=1e-6,
        max_steps=0,
        logging_steps=1,
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
    )
    peft_config = ensure_lora_targets(LoraConfig(**_lora_dict()), model)
    # Both B and C as separate reward_funcs so rewards_per_func keeps both columns.
    # Do NOT pass through combine_rewards here — that would fold them into one score.
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_B, reward_C],
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[stop_cb],
    )
    trainer.rollout_func = rollout_fn
    trainer._stop_before_optimizer = stop_cb
    return trainer


def _mask_report(env_mask: list[int] | None, completion_ids: list[int] | None) -> dict:
    if env_mask is None:
        return {"present": False}
    trainable = sum(1 for bit in env_mask if bit == 1)
    masked = sum(1 for bit in env_mask if bit == 0)
    # Find contiguous runs of 0 (tool observations) vs 1 (model tokens).
    runs = []
    if env_mask:
        start = 0
        cur = env_mask[0]
        for i, bit in enumerate(env_mask[1:], start=1):
            if bit != cur:
                runs.append({"value": cur, "start": start, "end": i, "length": i - start})
                start = i
                cur = bit
        runs.append({"value": cur, "start": start, "end": len(env_mask), "length": len(env_mask) - start})
    return {
        "present": True,
        "length": len(env_mask),
        "completion_id_length": len(completion_ids or []),
        "lengths_match": len(env_mask) == len(completion_ids or []),
        "trainable_model_tokens": trainable,
        "masked_observation_tokens": masked,
        "has_masked_observations": masked > 0,
        "has_trainable_tokens": trainable > 0,
        "runs": runs[:12],
        "head": env_mask[:32],
        "note": "env_mask/tool_mask: 1 = model call or final answer tokens; 0 = tool observation tokens",
    }


def score_group_via_trainer(
    *,
    trainer,
    episode_id: str,
    order_id: str,
    customer_id: str,
    amount_cents: int,
    generation_seeds: list[int],
    label: str,
) -> dict[str, Any]:
    """Run GRPOTrainer._generate_and_score_completions; do not call reward_* directly."""
    prompt = [{"role": "user", "content": one_action_user_prompt(order_id, customer_id)}]
    # Repeat the prompt once per generation — same layout the GRPO sampler uses.
    inputs = [
        {
            "prompt": copy.deepcopy(prompt),
            "episode_id": episode_id,
            "generation_seed": seed,
        }
        for seed in generation_seeds
    ]

    callback_trace: dict[str, Any] = {
        "label": label,
        "reward_called_directly": False,
        "calculate_rewards_invocations": [],
    }

    original = trainer._calculate_rewards

    def hooked_calculate_rewards(inputs_arg, prompts, completions, completion_ids_list):
        events = [inp.get("contract_events") for inp in inputs_arg]
        if any(item is None for item in events):
            raise RuntimeError(
                "TRL reward path missing contract_events on one or more inputs"
            )
        if len(events) != len(completions):
            raise RuntimeError(
                f"contract_events ({len(events)}) != completions ({len(completions)})"
            )
        for i, ev in enumerate(events):
            present = ev.get("callback_fields_present") or {}
            if not all(present.get(k) for k in ("raw_output", "validation_results", "harness_state", "final_answer")):
                raise RuntimeError(
                    f"contract_events[{i}] missing required callback fields: {present}"
                )
        rewards = original(inputs_arg, prompts, completions, completion_ids_list)
        # rewards_per_func: [batch, n_funcs]
        per_func = rewards.detach().cpu().tolist()
        callback_trace["calculate_rewards_invocations"].append(
            {
                "n_inputs": len(inputs_arg),
                "n_completions": len(completions),
                "n_contract_events": len(events),
                "generation_seeds": [inp.get("generation_seed") for inp in inputs_arg],
                "rollout_ids": [ev.get("rollout_id") for ev in events],
                "task_success": [ev.get("task_success") for ev in events],
                "n_invalid": [ev.get("n_invalid") for ev in events],
                "final_answers": [ev.get("final_answer") for ev in events],
                "harness_states": [ev.get("harness_state") for ev in events],
                "validation_counts": [len(ev.get("validation_results") or []) for ev in events],
                "raw_output_counts": [len(ev.get("raw_outputs") or []) for ev in events],
                "rewards_per_func": per_func,
                "reward_func_names": [
                    getattr(fn, "__name__", str(fn)) for fn in trainer.reward_funcs
                ],
            }
        )
        return rewards

    trainer._calculate_rewards = hooked_calculate_rewards
    optimizer_steps_before = getattr(trainer._stop_before_optimizer, "optimizer_steps", 0)
    try:
        # Put model in eval for scoring-only; still uses trainer generation path via rollout_func.
        trainer.model.eval()
        outputs = trainer._generate_and_score_completions(inputs)
    finally:
        trainer._calculate_rewards = original

    optimizer_steps_after = getattr(trainer._stop_before_optimizer, "optimizer_steps", 0)
    inv = callback_trace["calculate_rewards_invocations"][-1]
    # Column 0 = B, column 1 = C
    b_vals = [row[0] for row in inv["rewards_per_func"]]
    c_vals = [row[1] for row in inv["rewards_per_func"]]

    # TRL pops env_mask; read the surviving loss_masks column or the rollout_fn side channel.
    loss_masks = [inp.get("loss_masks") for inp in inputs]
    if not any(isinstance(mask, list) for mask in loss_masks):
        loss_masks = list(getattr(trainer.rollout_func, "last_loss_masks", []) or [])
    mask_examples = []
    for i, seed in enumerate(generation_seeds):
        mask = loss_masks[i] if i < len(loss_masks) else None
        mask_examples.append(
            {
                "index": i,
                "generation_seed": seed,
                "mask": _mask_report(mask if isinstance(mask, list) else None, None),
            }
        )

    return {
        "label": label,
        "episode_id": episode_id,
        "generation_seeds": generation_seeds,
        "B": b_vals,
        "C": c_vals,
        "callback_trace": callback_trace,
        "correspondence": [
            {
                "index": i,
                "generation_seed": generation_seeds[i],
                "rollout_id": inv["rollout_ids"][i],
                "task_success": inv["task_success"][i],
                "n_invalid": inv["n_invalid"][i],
                "final_answer": inv["final_answers"][i],
                "B": b_vals[i],
                "C": c_vals[i],
                "validation_n": inv["validation_counts"][i],
                "raw_output_n": inv["raw_output_counts"][i],
            }
            for i in range(len(generation_seeds))
        ],
        "mask_examples": mask_examples,
        "optimizer_steps": optimizer_steps_after - optimizer_steps_before,
        "train_called": False,
        "backward_called": False,
        "outputs_keys": sorted(outputs.keys()) if isinstance(outputs, dict) else None,
    }


def write_eval_manifest(out: Path, held_out_ceiling: bool) -> dict:
    rows = []
    for episode_id, order_id, customer_id, amount in HELD_OUT_TASKS:
        for seed in SEED_ORDER:
            rows.append(
                {
                    "episode_id": episode_id,
                    "order_id": order_id,
                    "customer_id": customer_id,
                    "amount_cents": amount,
                    "generation_seed": seed,
                    "environment_seed": config.ONE_ACTION_ENVIRONMENT_SEED,
                    "prompt_protocol": "one_action_per_turn",
                    "metrics": [
                        "task_success",
                        "invalid_attempt_rate",
                        "tasks_without_calls",
                        "calls_per_task",
                        "B",
                        "C",
                    ],
                }
            )
    manifest = {
        "kind": "final_eval_ABC",
        "conditions": ["A_untrained", "B_format_grpo", "C_outcome_grpo"],
        "held_out_tasks": [
            {
                "episode_id": episode_id,
                "order_id": order_id,
                "customer_id": customer_id,
                "amount_cents": amount,
            }
            for episode_id, order_id, customer_id, amount in HELD_OUT_TASKS
        ],
        "generation_seeds": list(SEED_ORDER),
        "n_rollouts": len(rows),
        "protocol": "one_action_per_turn",
        "counting_rules": {
            "task_success": "submitted and exact final answer",
            "N_invalid": "malformed_syntax + missing_name + unknown_tool + schema_invalid",
            "invalid_attempt_rate": "N_invalid / (parsed calls + malformed_syntax fragments)",
            "B": "I(task_success)",
            "C": "I(task_success) - 0.5 * N_invalid",
            "lambda": config.LAMBDA,
        },
        "breakdown": rows,
        "baseline_seed100_only": {
            "source": "artifacts/task-5/held_out_baseline/metrics.json",
            "task_success_rate": 1.0,
            "n_tasks": 4,
            "at_ceiling": held_out_ceiling,
            "note": (
                "Untrained held-out baseline on seeds=[100] only already saturates "
                "task success 4/4. Full A/B/C eval must use seeds 100–103 before "
                "treating ceiling as a training win."
                if held_out_ceiling
                else "Baseline not at ceiling."
            ),
        },
        "pilot_not_started": True,
    }
    _write_json(out / "eval_manifest_ABC.json", manifest)
    _write_json(out / "eval_breakdown_by_task_seed.json", {"rows": rows})
    return manifest


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    skip_live = "--skip-live" in argv
    out = TASK51
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from agenttune.agentic.rollout_engines.transformers_engine import (
        TransformersRolloutEngine,
    )

    versions = _package_versions()
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID, revision=config.MODEL_REVISION
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        revision=config.MODEL_REVISION,
        torch_dtype=torch.float16,
        device_map=None,
    )

    # ── 1) Scripted m0 via trainer scoring path ──────────────────────────────
    print("task_5_1 scripted_m0", flush=True)
    scripted_raws = _load_m0_scripted_raws()
    m0 = _task_tuple("m0")
    scripted_fn = make_instrumented_rollout_func(
        episode_id=m0[0],
        order_id=m0[1],
        customer_id=m0[2],
        amount_cents=m0[3],
        generation_seeds=list(SEED_ORDER),
        tokenizer=tokenizer,
        mode="scripted",
        scripted_raws=scripted_raws,
    )
    trainer_s = _build_trainer(tokenizer, model, scripted_fn, out / "scratch-scripted")
    scripted = score_group_via_trainer(
        trainer=trainer_s,
        episode_id=m0[0],
        order_id=m0[1],
        customer_id=m0[2],
        amount_cents=m0[3],
        generation_seeds=list(SEED_ORDER),
        label="scripted_m0",
    )
    expected = EXPECTED_TASK4["m0"]
    scripted["expected_B"] = expected["B"]
    scripted["expected_C"] = expected["C"]
    scripted["match_B"] = scripted["B"] == expected["B"]
    scripted["match_C"] = scripted["C"] == expected["C"]
    _write_json(out / "scripted_m0_trace.json", scripted)
    if not (scripted["match_B"] and scripted["match_C"]):
        _write_json(
            out / "summary.json",
            {
                "stopped": True,
                "reason": "scripted_m0_reward_mismatch",
                "scripted": scripted,
                "optimizer_steps": scripted["optimizer_steps"],
            },
        )
        print(json.dumps({"error": "scripted mismatch", "B": scripted["B"], "C": scripted["C"]}, indent=2))
        return 1

    # ── 2) Live t0 × 4 generations ───────────────────────────────────────────
    live = None
    if not skip_live:
        print("task_5_1 live_t0", flush=True)
        t0 = _task_tuple("t0")

        def live_engine_factory():
            # Fresh recording wrapper each generation; share the trainer model weights.
            return TransformersRolloutEngine(model=trainer_s.model, tokenizer=tokenizer)

        # Rebuild trainer with live rollout_fn (same model/peft).
        live_fn = make_instrumented_rollout_func(
            episode_id=t0[0],
            order_id=t0[1],
            customer_id=t0[2],
            amount_cents=t0[3],
            generation_seeds=list(SEED_ORDER),
            tokenizer=tokenizer,
            mode="live",
            live_engine_factory=live_engine_factory,
        )
        trainer_s.rollout_func = live_fn
        live = score_group_via_trainer(
            trainer=trainer_s,
            episode_id=t0[0],
            order_id=t0[1],
            customer_id=t0[2],
            amount_cents=t0[3],
            generation_seeds=list(SEED_ORDER),
            label="live_t0",
        )
        _write_json(out / "live_t0_trace.json", live)

    # ── 3) Eval manifest A/B/C ───────────────────────────────────────────────
    baseline_path = ROOT / "artifacts" / "task-5" / "held_out_baseline" / "metrics.json"
    held_out_ceiling = False
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        rate = (baseline.get("task_success") or {}).get("rate")
        held_out_ceiling = rate == 1.0
    eval_manifest = write_eval_manifest(out, held_out_ceiling=held_out_ceiling)

    summary = {
        "kind": "grpo_path_scoring_check",
        "versions": versions,
        "scripted_m0": {
            "B": scripted["B"],
            "C": scripted["C"],
            "expected_B": scripted["expected_B"],
            "expected_C": scripted["expected_C"],
            "match_B": scripted["match_B"],
            "match_C": scripted["match_C"],
            "optimizer_steps": scripted["optimizer_steps"],
            "callback_invocations": len(
                scripted["callback_trace"]["calculate_rewards_invocations"]
            ),
        },
        "live_t0": None
        if live is None
        else {
            "B": live["B"],
            "C": live["C"],
            "correspondence": live["correspondence"],
            "mask_examples": live["mask_examples"],
            "optimizer_steps": live["optimizer_steps"],
        },
        "eval_manifest": eval_manifest,
        "optimizer_steps_total": scripted["optimizer_steps"]
        + (0 if live is None else live["optimizer_steps"]),
        "train_called": False,
        "backward_called": False,
        "stopped_after_scoring": True,
    }
    _write_json(out / "summary.json", summary)
    print(
        json.dumps(
            {
                "scripted_m0_match": scripted["match_B"] and scripted["match_C"],
                "scripted_B": scripted["B"],
                "scripted_C": scripted["C"],
                "live_t0_B": None if live is None else live["B"],
                "live_t0_C": None if live is None else live["C"],
                "optimizer_steps": summary["optimizer_steps_total"],
                "held_out_baseline_at_ceiling": held_out_ceiling,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
