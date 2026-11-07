"""Task 5.2: loss-context prefix check, TRL masks, and batch task binding."""

from __future__ import annotations

import copy
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contract import config
from contract.events import attach_events_to_trajectory, events_from_steps
from contract.execute import PATH, make_guarded_tools
from contract.grpo_path import (
    TASK51,
    TokenizingScriptedEngine,
    _build_trainer,
    _mask_report,
    _run_one_instrumented,
    _task_tuple,
)
from contract.grpo_prep import (
    SEED_ORDER,
    TRAIN_TASKS,
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
from contract.run import _seed_generation

TASK52 = ROOT / "artifacts" / "task-5-2"

# Train tasks used for binding checks (must exist in TRAIN_TASKS).
BIND_TASK_A = "t0"
BIND_TASK_B = "t1"


class CapturingEngine:
    """Wrap a generate engine and record the exact input_ids passed to model.generate."""

    def __init__(self, inner, tokenizer):
        self._inner = inner
        self._tokenizer = tokenizer
        self.generate_calls: list[dict[str, Any]] = []

    def _get_tokenizer(self):
        return self._tokenizer

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def generate(self, prompts, tools, gen_cfg):
        captured: dict[str, Any] = {"input_ids": None}
        model = getattr(self._inner, "model", None)
        if model is not None and hasattr(model, "generate"):
            original = model.generate

            def hooked_generate(*args, **kwargs):
                ids = kwargs.get("input_ids")
                if ids is None and args:
                    ids = args[0]
                if ids is not None:
                    if hasattr(ids, "detach"):
                        row = ids[0] if getattr(ids, "dim", lambda: 1)() > 1 else ids
                        captured["input_ids"] = row.detach().cpu().tolist()
                    else:
                        captured["input_ids"] = list(ids[0] if isinstance(ids[0], list) else ids)
                return original(*args, **kwargs)

            model.generate = hooked_generate
            try:
                result = self._inner.generate(prompts, tools, gen_cfg)
            finally:
                model.generate = original
        else:
            # Scripted path: no model.generate — fall back to chat-template encode.
            conversations = prompts
            if isinstance(prompts, str) or (
                isinstance(prompts, list) and prompts and isinstance(prompts[0], str)
            ):
                raw = [prompts] if isinstance(prompts, str) else prompts
                conversations = [[{"role": "user", "content": p}] for p in raw]
            elif (
                isinstance(prompts, list)
                and prompts
                and isinstance(prompts[0], dict)
                and "role" in prompts[0]
            ):
                conversations = [prompts]
            if hasattr(self._inner, "_apply_chat_template"):
                encoded = self._inner._apply_chat_template(
                    conversations,
                    tools if tools else None,
                    gen_cfg.get("enable_thinking", False),
                )
                captured["input_ids"] = encoded["input_ids"][0].detach().cpu().tolist()
            else:
                rendered = self._tokenizer.apply_chat_template(
                    conversations[0],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                captured["input_ids"] = self._tokenizer.encode(
                    rendered, add_special_tokens=False
                )
            result = self._inner.generate(prompts, tools, gen_cfg)

        if captured["input_ids"] is None:
            raise RuntimeError("failed to capture input_ids from model.generate")
        new_ids = ((result.get("metadata") or {}).get("completion_ids") or [[]])[0]
        if hasattr(new_ids, "tolist"):
            new_ids = new_ids.tolist()
        input_ids = list(captured["input_ids"])
        self.generate_calls.append(
            {
                "turn_index": len(self.generate_calls),
                "input_ids": input_ids,
                "new_token_ids": list(new_ids),
                "input_len": len(input_ids),
                "new_len": len(new_ids),
            }
        )
        return result


def _first_diff(a: list[int], b: list[int]) -> dict[str, Any] | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return {"index": i, "left": a[i], "right": b[i]}
    if len(a) != len(b):
        return {
            "index": n,
            "left": a[n] if len(a) > n else None,
            "right": b[n] if len(b) > n else None,
            "length_mismatch": {"left": len(a), "right": len(b)},
        }
    return None


def _decode_window(tokenizer, ids: list[int], index: int, radius: int = 16) -> str:
    start = max(0, index - radius)
    end = min(len(ids), index + radius)
    return tokenizer.decode(ids[start:end], skip_special_tokens=False)


def compare_generate_prefixes_to_stream(
    *,
    tokenizer,
    generate_calls: list[dict[str, Any]],
    prompt_ids: list[int],
    completion_ids: list[int],
) -> dict[str, Any]:
    """Compare each generate() input to prompt_ids + completion_ids prefix."""
    stream = list(prompt_ids) + list(completion_ids)
    comparisons = []
    # Positions in completion_ids where each model turn's tokens start.
    # We locate each generate's new_token_ids inside completion_ids (mask=1 spans).
    cursor = 0
    for call in generate_calls:
        inp = call["input_ids"]
        new = call["new_token_ids"]
        # Prefix in the TRL stream that should match generate input.
        # Find where `new` sits in completion_ids starting from cursor.
        found_at = None
        for j in range(cursor, len(completion_ids) - len(new) + 1):
            if completion_ids[j : j + len(new)] == new:
                found_at = j
                break
        if found_at is None:
            comparisons.append(
                {
                    "turn_index": call["turn_index"],
                    "role": _turn_role(call["turn_index"], len(generate_calls)),
                    "match": False,
                    "error": "new_token_ids not found in completion_ids from cursor",
                    "cursor": cursor,
                    "generate_input_len": len(inp),
                    "new_len": len(new),
                }
            )
            continue
        # Prefix preceding these tokens in the full stream = prompt_ids + completion_ids[:found_at]
        trl_prefix = list(prompt_ids) + list(completion_ids[:found_at])
        diff = _first_diff(inp, trl_prefix)
        item = {
            "turn_index": call["turn_index"],
            "role": _turn_role(call["turn_index"], len(generate_calls)),
            "match": diff is None,
            "generate_input_len": len(inp),
            "trl_prefix_len": len(trl_prefix),
            "new_token_start_in_completion": found_at,
            "new_len": len(new),
        }
        if diff is not None:
            idx = diff["index"]
            item["first_diff"] = diff
            item["generate_fragment"] = _decode_window(tokenizer, inp, min(idx, len(inp) - 1))
            item["trl_fragment"] = _decode_window(
                tokenizer, trl_prefix, min(idx, max(len(trl_prefix) - 1, 0))
            )
        comparisons.append(item)
        cursor = found_at + len(new)
    return {
        "all_match": all(c.get("match") for c in comparisons) and len(comparisons) == len(generate_calls),
        "comparisons": comparisons,
        "prompt_len": len(prompt_ids),
        "completion_len": len(completion_ids),
        "n_generate_calls": len(generate_calls),
    }


def _turn_role(turn_index: int, n_turns: int) -> str:
    if turn_index == 0:
        return "first_tool_call"
    if turn_index == 1:
        return "second_tool_call"
    if turn_index == n_turns - 1:
        return "final"
    return f"turn_{turn_index}"


def rebuild_packing_from_generate_calls(
    generate_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pack prompt/completion/mask so each generate input equals the stream prefix.

    Between generate turns, tokens that appear in the next input but not in
    previous_input+new_tokens are treated as observation tokens (mask 0).
    If a later generate input rewrites earlier tokens, packing cannot preserve
    context — return diverged=True without changing generation context.
    """
    if not generate_calls:
        raise ValueError("no generate calls to pack")
    prompt_ids = list(generate_calls[0]["input_ids"])
    completion_ids: list[int] = []
    tool_mask: list[int] = []
    diverged = False
    divergence = None
    base = list(prompt_ids)
    for i, call in enumerate(generate_calls):
        inp = list(call["input_ids"])
        new = list(call["new_token_ids"])
        # Prefix check: generate input must equal current stream prefix.
        prefix = list(prompt_ids) + completion_ids
        diff = _first_diff(inp, prefix)
        if diff is not None:
            # Observation tokens may have been added only as chat-template delta
            # that is not yet in completion_ids. If inp extends prefix, append delta as mask 0.
            if len(inp) >= len(prefix) and inp[: len(prefix)] == prefix:
                delta = inp[len(prefix) :]
                completion_ids.extend(delta)
                tool_mask.extend([0] * len(delta))
            else:
                diverged = True
                divergence = {
                    "turn_index": i,
                    "first_diff": diff,
                    "note": (
                        "generate input rewrites earlier stream tokens; "
                        "cannot rebuild a shared loss stream without changing generation context"
                    ),
                }
                break
        # Re-check after optional delta append.
        prefix = list(prompt_ids) + completion_ids
        diff = _first_diff(inp, prefix)
        if diff is not None:
            diverged = True
            divergence = {
                "turn_index": i,
                "first_diff": diff,
                "note": "still mismatched after appending template delta",
            }
            break
        completion_ids.extend(new)
        tool_mask.extend([1] * len(new))
        base = list(inp) + list(new)
    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "env_mask": tool_mask,
        "diverged": diverged,
        "divergence": divergence,
        "logprobs": [(0.0, tid) for tid in completion_ids],
    }


def trl_mask_audit(
    *,
    tokenizer,
    outputs: dict[str, Any],
    row_index: int = 0,
) -> dict[str, Any]:
    """Inspect completion_ids, completion_mask, tool_mask and loss product from TRL outputs."""
    import torch

    def _row(name: str):
        val = outputs.get(name)
        if val is None:
            return None
        if hasattr(val, "detach"):
            row = val[row_index].detach().cpu()
            # drop padding: completion_mask==0 at the end
            return row
        return val[row_index]

    completion_ids_t = _row("completion_ids")
    completion_mask_t = _row("completion_mask")
    tool_mask_t = outputs.get("tool_mask")
    if tool_mask_t is not None and hasattr(tool_mask_t, "detach"):
        tool_mask_t = tool_mask_t[row_index].detach().cpu()
    # Also accept loss_mask if present
    loss_mask_t = outputs.get("loss_mask")
    if loss_mask_t is not None and hasattr(loss_mask_t, "detach"):
        loss_mask_t = loss_mask_t[row_index].detach().cpu()

    if completion_ids_t is None or completion_mask_t is None:
        return {"present": False, "error": "missing completion_ids or completion_mask"}

    comp_mask = completion_mask_t.tolist()
    comp_ids = completion_ids_t.tolist()
    # Trim to attention length
    length = int(sum(comp_mask))
    # Prefer true length from mask (may include internal zeros from tool_mask product later)
    nonzero = [i for i, m in enumerate(comp_mask) if m]
    trim = (max(nonzero) + 1) if nonzero else 0
    comp_ids = comp_ids[:trim]
    comp_mask = comp_mask[:trim]

    if tool_mask_t is None:
        tool = [1] * len(comp_ids)
        tool_source = "absent_default_ones"
    else:
        tool = tool_mask_t.tolist()[:trim]
        tool_source = "trl_outputs.tool_mask"

    product = [int(c and t) for c, t in zip(comp_mask, tool)]
    included_ids = [tid for tid, bit in zip(comp_ids, product) if bit]
    excluded_ids = [tid for tid, bit in zip(comp_ids, product) if not bit and comp_mask[product.index(bit) if False else 0] or True]
    # clearer exclusion
    excluded_ids = [tid for tid, cbit, pbit in zip(comp_ids, comp_mask, product) if cbit and not pbit]

    return {
        "present": True,
        "tool_mask_source": tool_source,
        "length_trimmed": trim,
        "completion_mask_sum": int(sum(comp_mask)),
        "tool_mask_sum": int(sum(tool)),
        "loss_product_sum": int(sum(product)),
        "n_included_in_loss": len(included_ids),
        "n_excluded_tool_obs": len(excluded_ids),
        "included_decode_head": tokenizer.decode(included_ids[:64], skip_special_tokens=False),
        "excluded_decode_head": tokenizer.decode(excluded_ids[:64], skip_special_tokens=False)
        if excluded_ids
        else "",
        "product_runs": _mask_report(product, comp_ids).get("runs"),
        "completion_ids_head": comp_ids[:32],
        "completion_mask_head": comp_mask[:32],
        "tool_mask_head": tool[:32],
        "loss_product_head": product[:32],
    }


# ── Task registry / binding ──────────────────────────────────────────────────


def task_registry() -> dict[str, dict[str, Any]]:
    reg = {}
    for episode_id, order_id, customer_id, amount in TRAIN_TASKS:
        reg[episode_id] = {
            "episode_id": episode_id,
            "order_id": order_id,
            "customer_id": customer_id,
            "amount_cents": amount,
            "split": "train",
            "prompt": one_action_user_prompt(order_id, customer_id),
        }
    return reg


def resolve_batch_task(item: dict[str, Any], registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Bind one batch row to a known task. Unknown/ambiguous → error."""
    episode_id = item.get("episode_id")
    prompt = item.get("prompt")
    prompt_text = None
    if isinstance(prompt, list) and prompt:
        # conversational
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                prompt_text = msg.get("content")
                break
    elif isinstance(prompt, str):
        prompt_text = prompt

    matches = []
    if episode_id:
        if episode_id not in registry:
            raise ValueError(f"unknown episode_id in batch: {episode_id!r}")
        matches.append(registry[episode_id])
    if prompt_text:
        text_matches = [t for t in registry.values() if t["prompt"] == prompt_text]
        if episode_id and text_matches and text_matches[0]["episode_id"] != episode_id:
            raise ValueError(
                f"ambiguous task binding: episode_id={episode_id!r} but prompt matches "
                f"{text_matches[0]['episode_id']!r}"
            )
        if not episode_id:
            matches = text_matches
    if not matches:
        raise ValueError(
            f"could not bind batch row to a task (episode_id={episode_id!r}, prompt_present={bool(prompt_text)})"
        )
    if len(matches) > 1:
        raise ValueError(
            "ambiguous task binding: multiple registry entries match this row: "
            + ",".join(m["episode_id"] for m in matches)
        )
    return matches[0]


def make_batch_bound_rollout_func(
    *,
    tokenizer,
    mode: str = "live",
    live_engine_factory: Callable | None = None,
    rebuild_packing: bool = True,
) -> Callable:
    """Pilot-style rollout_func: environment bound from each batch row's task."""

    registry = task_registry()
    gen_kwargs = {
        "max_new_tokens": config.ONE_ACTION_MAX_NEW_TOKENS,
        "temperature": config.ONE_ACTION_TEMPERATURE,
        "do_sample": config.ONE_ACTION_DO_SAMPLE,
        "top_p": config.ONE_ACTION_TOP_P,
        "max_length": config.CONTEXT_LENGTH,
    }

    def rollout_fn(prompts, trainer=None, *args, **kwargs):
        import agenttune.agentic.rollout_engines.rollout_factory as factory

        batch_meta = list(getattr(rollout_fn, "bound_batch", []) or [])
        prompt_list = list(prompts) if not isinstance(prompts, str) else [prompts]
        if not batch_meta:
            raise ValueError(
                "rollout_fn.bound_batch is empty; bind tasks from the input batch before scoring"
            )
        if len(batch_meta) != len(prompt_list):
            raise ValueError(
                f"bound_batch ({len(batch_meta)}) != prompts ({len(prompt_list)})"
            )

        merged = {
            "prompt_ids": [],
            "completion_ids": [],
            "logprobs": [],
            "env_mask": [],
            "loss_masks": [],
            "responses": [],
            "trajectories": [],
            "contract_events": [],
            "conversations": [],
            "tool_call_counts": [],
            "generation_seeds": [],
            "rollout_ids": [],
            "episode_ids": [],
            "environment_ids": [],
            "generate_captures": [],
            "prefix_checks": [],
            "packing_repairs": [],
        }
        episode_objects = []

        for i, prompt in enumerate(prompt_list):
            meta = batch_meta[i]
            task = resolve_batch_task(meta, registry)
            seed = int(meta.get("generation_seed", SEED_ORDER[i % len(SEED_ORDER)]))
            # Prompt text must match the bound task.
            expected_prompt = task["prompt"]
            got_text = None
            if isinstance(prompt, list):
                for msg in prompt:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        got_text = msg.get("content")
                        break
            elif isinstance(prompt, str):
                got_text = prompt
            if got_text is not None and got_text != expected_prompt:
                raise ValueError(
                    f"prompt text does not match bound task {task['episode_id']}"
                )

            episode = RefundEpisode(
                episode_id=task["episode_id"],
                order_id=task["order_id"],
                customer_id=task["customer_id"],
                amount_cents=task["amount_cents"],
                seed=config.ONE_ACTION_ENVIRONMENT_SEED,
            )
            episode_objects.append(episode)
            env_id = {
                "episode_id": episode.episode_id,
                "order_id": episode.order_id,
                "customer_id": episode.customer_id,
                "amount_cents": episode.amount_cents,
                "environment_seed": config.ONE_ACTION_ENVIRONMENT_SEED,
                "order_token": episode.order_token,
                "customer_token": episode.customer_token,
            }

            if mode == "live":
                if live_engine_factory is None:
                    raise RuntimeError("live_engine_factory required")
                _seed_generation(seed)
                inner = live_engine_factory()
            else:
                raise ValueError(mode)
            engine = CapturingEngine(inner, tokenizer)

            packed, steps, events = _run_one_instrumented(
                factory=factory,
                engine=engine,
                tokenizer=tokenizer,
                episode=episode,
                generation_seed=seed,
                prompt_text=expected_prompt,
                gen_kwargs=gen_kwargs,
            )
            captures = list(engine.generate_calls)
            repair = rebuild_packing_from_generate_calls(captures)
            if rebuild_packing and not repair["diverged"]:
                packed["prompt_ids"] = [repair["prompt_ids"]]
                packed["completion_ids"] = [repair["completion_ids"]]
                packed["env_mask"] = [repair["env_mask"]]
                # TRL pads sampling logprobs to completion width — keep lengths aligned.
                packed["logprobs"] = [[0.0] * len(repair["completion_ids"])]
                # keep trajectory metadata in sync for mask consumers
                traj = packed["trajectories"][0]
                meta_t = dict(getattr(traj, "metadata", None) or {})
                meta_t["prompt_ids"] = repair["prompt_ids"]
                meta_t["completion_ids"] = repair["completion_ids"]
                meta_t["tool_mask"] = repair["env_mask"]
                traj.metadata = meta_t

            # Compare against the stream that would enter TRL (repaired if possible).
            # When repair diverged, also compare against the generate-derived attempted stream.
            prefix_check = compare_generate_prefixes_to_stream(
                tokenizer=tokenizer,
                generate_calls=captures,
                prompt_ids=packed["prompt_ids"][0],
                completion_ids=packed["completion_ids"][0],
            )
            if repair["diverged"]:
                # Direct generate-to-generate prefix evidence for the review stop condition.
                attempted = compare_generate_prefixes_to_stream(
                    tokenizer=tokenizer,
                    generate_calls=captures,
                    prompt_ids=repair["prompt_ids"],
                    completion_ids=repair["completion_ids"],
                )
                prefix_check["repair_attempted_stream"] = attempted
                prefix_check["divergence"] = repair["divergence"]
                # Decorate divergence with decoded fragments at the first rewrite.
                div = repair["divergence"] or {}
                fd = div.get("first_diff") or {}
                idx = fd.get("index")
                if idx is not None and captures:
                    turn = captures[div.get("turn_index", 0)]
                    prefix_before = list(repair["prompt_ids"]) + list(repair["completion_ids"])
                    # Before the diverging turn, completion_ids in repair is the extendable prefix only.
                    prefix_check["divergence_detail"] = {
                        "turn_index": div.get("turn_index"),
                        "first_diff_index": idx,
                        "generate_token": fd.get("left"),
                        "stream_token": fd.get("right"),
                        "generate_fragment": _decode_window(
                            tokenizer, turn["input_ids"], min(idx, len(turn["input_ids"]) - 1)
                        ),
                        "stream_fragment": _decode_window(
                            tokenizer, prefix_before, min(idx, max(len(prefix_before) - 1, 0))
                        ),
                        "note": div.get("note"),
                    }

            merged["prompt_ids"].append(packed["prompt_ids"][0])
            merged["completion_ids"].append(packed["completion_ids"][0])
            merged["logprobs"].append(packed.get("logprobs", [[]])[0])
            merged["env_mask"].append(packed["env_mask"][0])
            merged["loss_masks"].append(list(packed["env_mask"][0]))
            merged["responses"].append(packed["responses"][0])
            merged["trajectories"].append(packed["trajectories"][0])
            merged["contract_events"].append(events)
            merged["conversations"].append((packed.get("conversations") or [None])[0])
            merged["tool_call_counts"].append((packed.get("tool_call_counts") or [0])[0])
            merged["generation_seeds"].append(seed)
            merged["rollout_ids"].append(events["rollout_id"])
            merged["episode_ids"].append(task["episode_id"])
            merged["environment_ids"].append(env_id)
            merged["generate_captures"].append(captures)
            merged["prefix_checks"].append(prefix_check)
            merged["packing_repairs"].append(
                {"diverged": repair["diverged"], "divergence": repair.get("divergence")}
            )

        # Isolation: no two generations may share the same episode object identity
        # or the same mutable harness token bag.
        if len({id(ep) for ep in episode_objects}) != len(episode_objects):
            raise RuntimeError("episode object identity was shared across generations")
        token_bags = [(ep.order_token, ep.customer_token, ep.quote_token) for ep in episode_objects]
        # Same task ⇒ same tokens by design; different tasks must differ.
        episode_ids = merged["episode_ids"]
        for i in range(len(episode_ids)):
            for j in range(i + 1, len(episode_ids)):
                if episode_ids[i] != episode_ids[j] and token_bags[i] == token_bags[j]:
                    raise RuntimeError(
                        f"distinct tasks {episode_ids[i]}/{episode_ids[j]} share harness tokens"
                    )
                if episode_ids[i] == episode_ids[j] and id(episode_objects[i]) == id(episode_objects[j]):
                    raise RuntimeError("same-task generations shared one episode instance")

        if any(e is None for e in merged["contract_events"]):
            raise RuntimeError("missing contract_events")
        if len(merged["contract_events"]) != len(prompt_list):
            raise RuntimeError("contract_events length mismatch")

        rollout_fn.last_loss_masks = merged["loss_masks"]
        rollout_fn.last_generate_captures = merged["generate_captures"]
        rollout_fn.last_prefix_checks = merged["prefix_checks"]
        rollout_fn.last_environment_ids = merged["environment_ids"]
        rollout_fn.last_packing_repairs = merged["packing_repairs"]
        return merged

    rollout_fn.__name__ = "batch_bound_instrumented_rollout"
    rollout_fn.bound_batch = []
    rollout_fn.last_loss_masks = []
    rollout_fn.last_generate_captures = []
    rollout_fn.last_prefix_checks = []
    rollout_fn.last_environment_ids = []
    rollout_fn.last_packing_repairs = []
    return rollout_fn


def score_bound_batch(trainer, batch_rows: list[dict], label: str) -> dict[str, Any]:
    """Score via trainer path with tasks bound from batch_rows."""
    rollout_fn = trainer.rollout_func
    rollout_fn.bound_batch = batch_rows
    inputs = []
    for row in batch_rows:
        task = resolve_batch_task(row, task_registry())
        inputs.append(
            {
                "prompt": [{"role": "user", "content": task["prompt"]}],
                "episode_id": task["episode_id"],
                "generation_seed": row["generation_seed"],
            }
        )

    callback_trace: dict[str, Any] = {"label": label, "invocations": []}
    original = trainer._calculate_rewards

    def hooked(inputs_arg, prompts, completions, completion_ids_list):
        events = [inp.get("contract_events") for inp in inputs_arg]
        if any(e is None for e in events) or len(events) != len(completions):
            raise RuntimeError("contract_events missing or misaligned in TRL reward path")
        rewards = original(inputs_arg, prompts, completions, completion_ids_list)
        callback_trace["invocations"].append(
            {
                "episode_ids": [inp.get("episode_id") for inp in inputs_arg],
                "environment_ids": list(getattr(rollout_fn, "last_environment_ids", [])),
                "rollout_ids": [e.get("rollout_id") for e in events],
                "task_success": [e.get("task_success") for e in events],
                "n_invalid": [e.get("n_invalid") for e in events],
                "rewards_per_func": rewards.detach().cpu().tolist(),
                "final_answers": [e.get("final_answer") for e in events],
            }
        )
        return rewards

    trainer._calculate_rewards = hooked
    steps_before = getattr(trainer._stop_before_optimizer, "optimizer_steps", 0)
    # TRL advantages reshape as (-1, num_generations); match the bound batch size.
    # model.eval() selects num_generations_eval — patch both.
    prev_num_gen = trainer.num_generations
    prev_num_gen_eval = trainer.num_generations_eval
    trainer.num_generations = len(inputs)
    trainer.num_generations_eval = len(inputs)
    try:
        trainer.model.eval()
        outputs = trainer._generate_and_score_completions(inputs)
    finally:
        trainer._calculate_rewards = original
        trainer.num_generations = prev_num_gen
        trainer.num_generations_eval = prev_num_gen_eval
    steps_after = getattr(trainer._stop_before_optimizer, "optimizer_steps", 0)

    # Capture tool_mask from TRL: it pops env_mask into tool_mask locally; re-read from outputs if stored.
    # GRPOTrainer output may not include tool_mask — recover from rollout loss_masks / env_mask.
    inv = callback_trace["invocations"][-1]
    b_vals = [row[0] for row in inv["rewards_per_func"]]
    c_vals = [row[1] for row in inv["rewards_per_func"]]

    # Attach tool_mask onto a copy of outputs for audit using loss_masks as the pre-pop env_mask.
    import torch

    loss_masks = list(getattr(rollout_fn, "last_loss_masks", []) or [])
    if loss_masks and "completion_ids" in outputs:
        # Build a tool_mask tensor aligned to padded completion_ids.
        comp = outputs["completion_ids"]
        padded = []
        width = comp.shape[1]
        for mask in loss_masks:
            row = list(mask) + [1] * max(0, width - len(mask))
            padded.append(row[:width])
        outputs = dict(outputs)
        outputs["tool_mask"] = torch.tensor(padded, device=comp.device)

    return {
        "label": label,
        "B": b_vals,
        "C": c_vals,
        "callback_trace": callback_trace,
        "prefix_checks": list(getattr(rollout_fn, "last_prefix_checks", [])),
        "packing_repairs": list(getattr(rollout_fn, "last_packing_repairs", [])),
        "generate_captures": list(getattr(rollout_fn, "last_generate_captures", [])),
        "environment_ids": list(getattr(rollout_fn, "last_environment_ids", [])),
        "loss_masks": loss_masks,
        "outputs": outputs,
        "optimizer_steps": steps_after - steps_before,
        "correspondence": [
            {
                "index": i,
                "episode_id": inv["episode_ids"][i],
                "rollout_id": inv["rollout_ids"][i],
                "environment": inv["environment_ids"][i] if i < len(inv["environment_ids"]) else None,
                "task_success": inv["task_success"][i],
                "n_invalid": inv["n_invalid"][i],
                "B": b_vals[i],
                "C": c_vals[i],
                "final_answer": inv["final_answers"][i],
            }
            for i in range(len(b_vals))
        ],
    }


def run_successful_live_prefix_check(tokenizer, model, out: Path) -> dict[str, Any]:
    """One successful live rollout: compare generate inputs to TRL stream prefixes."""
    from agenttune.agentic.rollout_engines.transformers_engine import (
        TransformersRolloutEngine,
    )

    # Try seeds until success (usually 100 works for t0).
    rollout_fn = make_batch_bound_rollout_func(
        tokenizer=tokenizer,
        mode="live",
        live_engine_factory=lambda: TransformersRolloutEngine(model=model, tokenizer=tokenizer),
        rebuild_packing=True,
    )
    trainer = _build_trainer(tokenizer, model, rollout_fn, out / "scratch-prefix")
    last_error = None
    for seed in SEED_ORDER:
        print(f"prefix_check seed={seed}", flush=True)
        batch = [{"episode_id": "t0", "generation_seed": seed}]
        try:
            result = score_bound_batch(trainer, batch, label=f"prefix_t0_g{seed}")
        except Exception as exc:
            last_error = traceback.format_exc()
            print(f"prefix_check seed={seed} error={exc}", flush=True)
            continue
        if not result["correspondence"][0]["task_success"]:
            print(f"prefix_check seed={seed} task_success=False", flush=True)
            continue
        # Build TRL mask audit for row 0.
        mask_audit = trl_mask_audit(tokenizer=tokenizer, outputs=result["outputs"], row_index=0)
        prefix = result["prefix_checks"][0]
        repair = result["packing_repairs"][0]
        # Focus on first, second, final turns.
        focused = [
            c
            for c in prefix["comparisons"]
            if c.get("role") in {"first_tool_call", "second_tool_call", "final"}
            or c.get("turn_index") in {0, 1, prefix["n_generate_calls"] - 1}
        ]
        payload = {
            "seed": seed,
            "success": True,
            "prefix_check": prefix,
            "focused_turns": focused,
            "packing_repair": repair,
            "trl_mask_audit": mask_audit,
            "generate_captures_lens": [
                {"turn": c["turn_index"], "input_len": c["input_len"], "new_len": c["new_len"]}
                for c in result["generate_captures"][0]
            ],
            "optimizer_steps": result["optimizer_steps"],
            "B": result["B"],
            "C": result["C"],
            "stopped_before_backward": True,
        }
        _write_json(out / "prefix_check.json", payload)
        # Persist raw capture IDs for the review (trimmed lengths only in summary).
        _write_json(
            out / "generate_captures.json",
            {
                "seed": seed,
                "calls": [
                    {
                        "turn_index": c["turn_index"],
                        "input_len": c["input_len"],
                        "new_len": c["new_len"],
                        "input_ids_head": c["input_ids"][:32],
                        "input_ids_tail": c["input_ids"][-32:],
                        "new_token_ids": c["new_token_ids"],
                    }
                    for c in result["generate_captures"][0]
                ],
            },
        )
        return payload
    return {
        "success": False,
        "error": "no successful live t0 rollout among seeds 100-103",
        "last_error": last_error,
    }


def run_two_train_task_binding(tokenizer, model, out: Path) -> dict[str, Any]:
    from agenttune.agentic.rollout_engines.transformers_engine import (
        TransformersRolloutEngine,
    )

    rollout_fn = make_batch_bound_rollout_func(
        tokenizer=tokenizer,
        mode="live",
        live_engine_factory=lambda: TransformersRolloutEngine(model=model, tokenizer=tokenizer),
        rebuild_packing=True,
    )
    trainer = _build_trainer(tokenizer, model, rollout_fn, out / "scratch-bind")

    # Negative checks first (no model calls needed beyond resolve).
    negatives = {}
    try:
        resolve_batch_task({"episode_id": "nope", "prompt": "x"}, task_registry())
        negatives["unknown_episode"] = "failed_to_raise"
    except ValueError as exc:
        negatives["unknown_episode"] = f"raised:{exc}"
    try:
        resolve_batch_task({"prompt": "not a real prompt"}, task_registry())
        negatives["unbound_prompt"] = "failed_to_raise"
    except ValueError as exc:
        negatives["unbound_prompt"] = f"raised:{exc}"

    results = {}
    for episode_id in (BIND_TASK_A, BIND_TASK_B):
        print(f"bind_check {episode_id}", flush=True)
        # Two generations of the same task to prove isolation between gens.
        batch = [
            {"episode_id": episode_id, "generation_seed": 100},
            {"episode_id": episode_id, "generation_seed": 101},
        ]
        scored = score_bound_batch(trainer, batch, label=f"bind_{episode_id}")
        envs = scored["environment_ids"]
        isolation = {
            "distinct_episode_objects_implied": envs[0] is not envs[1],
            "same_task_tokens_match": (
                envs[0]["order_token"] == envs[1]["order_token"]
                and envs[0]["customer_token"] == envs[1]["customer_token"]
            ),
            "episode_ids": [e["episode_id"] for e in envs],
            "order_ids": [e["order_id"] for e in envs],
            "customer_ids": [e["customer_id"] for e in envs],
        }
        results[episode_id] = {
            "prompt": task_registry()[episode_id]["prompt"],
            "environment_ids": envs,
            "isolation": isolation,
            "correspondence": scored["correspondence"],
            "B": scored["B"],
            "C": scored["C"],
            "prefix_all_match": all(p.get("all_match") for p in scored["prefix_checks"]),
            "optimizer_steps": scored["optimizer_steps"],
        }

    # Cross-task token isolation in one batch.
    print("bind_check cross t0+t1", flush=True)
    cross = score_bound_batch(
        trainer,
        [
            {"episode_id": BIND_TASK_A, "generation_seed": 100},
            {"episode_id": BIND_TASK_B, "generation_seed": 100},
        ],
        label="bind_cross",
    )
    cross_envs = cross["environment_ids"]
    cross_isolation = {
        "tokens_differ": (
            cross_envs[0]["order_token"] != cross_envs[1]["order_token"]
            and cross_envs[0]["customer_token"] != cross_envs[1]["customer_token"]
        ),
        "episode_ids": [e["episode_id"] for e in cross_envs],
        "order_ids": [e["order_id"] for e in cross_envs],
    }
    payload = {
        "negatives": negatives,
        "per_task": results,
        "cross_task": {
            "isolation": cross_isolation,
            "correspondence": cross["correspondence"],
            "B": cross["B"],
            "C": cross["C"],
            "optimizer_steps": cross["optimizer_steps"],
        },
        "optimizer_steps_total": sum(r["optimizer_steps"] for r in results.values())
        + cross["optimizer_steps"],
    }
    _write_json(out / "train_binding.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    skip_bind = "--skip-bind" in argv
    skip_prefix = "--skip-prefix" in argv

    out = TASK52
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    packing_stop_reason = None
    prefix_result = None
    if not skip_prefix:
        print("task_5_2 prefix_check", flush=True)
        prefix_result = run_successful_live_prefix_check(tokenizer, model, out)
        if prefix_result.get("packing_repair", {}).get("diverged"):
            packing_stop_reason = "packing_diverged_without_safe_repair"
            print(
                json.dumps(
                    {
                        "stopped": "packing_diverged",
                        "divergence_detail": (prefix_result.get("prefix_check") or {}).get(
                            "divergence_detail"
                        ),
                        "focused": prefix_result.get("focused_turns"),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                flush=True,
            )
        elif prefix_result.get("success") and not prefix_result["prefix_check"]["all_match"]:
            packing_stop_reason = "generate_prefix_mismatch_after_repair"
            print(
                json.dumps(
                    {"stopped": "prefix_mismatch", "focused": prefix_result.get("focused_turns")},
                    indent=2,
                    ensure_ascii=False,
                ),
                flush=True,
            )

    # Binding checks do not update weights; run them even if packing must stop the pilot.
    bind_result = None
    if not skip_bind:
        print("task_5_2 train_binding", flush=True)
        bind_result = run_two_train_task_binding(tokenizer, model, out)

    # Baseline A status (may still be running externally).
    baseline_manifest = out / "baseline_a" / "manifest.json"
    baseline = None
    if baseline_manifest.exists():
        baseline = json.loads(baseline_manifest.read_text(encoding="utf-8"))

    summary = {
        "kind": "grpo_loss_context_and_binding",
        "versions": versions,
        "prefix_check": prefix_result,
        "packing_stop_reason": packing_stop_reason,
        "train_binding": None
        if bind_result is None
        else {
            "negatives": bind_result["negatives"],
            "tasks": {
                k: {
                    "order_ids": v["isolation"]["order_ids"],
                    "B": v["B"],
                    "C": v["C"],
                    "prefix_all_match": v["prefix_all_match"],
                    "correspondence": v["correspondence"],
                }
                for k, v in bind_result["per_task"].items()
            },
            "cross_task": bind_result["cross_task"],
            "optimizer_steps": bind_result["optimizer_steps_total"],
        },
        "baseline_a": None
        if baseline is None
        else {
            "n_rollouts": baseline["n_rollouts"],
            "aggregates": baseline["aggregates"],
            "formula_B": baseline["formula_B"],
            "formula_C": baseline["formula_C"],
            "at_ceiling": baseline["aggregates"]["at_ceiling_task_success"],
        },
        "optimizer_steps": (prefix_result or {}).get("optimizer_steps", 0)
        + ((bind_result or {}).get("optimizer_steps_total") or 0),
        "train_called": False,
        "backward_called": False,
        "stopped_before_backward": True,
        "pilot_blocked": packing_stop_reason is not None,
    }
    _write_json(out / "summary.json", summary)
    print(
        json.dumps(
            {
                "prefix_all_match": None
                if prefix_result is None
                else prefix_result.get("prefix_check", {}).get("all_match"),
                "packing_stop_reason": packing_stop_reason,
                "divergence_detail": None
                if prefix_result is None
                else (prefix_result.get("prefix_check") or {}).get("divergence_detail"),
                "trl_loss_tokens": None
                if prefix_result is None
                else (prefix_result.get("trl_mask_audit") or {}).get("n_included_in_loss"),
                "bind_tasks": None if bind_result is None else list(bind_result["per_task"]),
                "baseline_ready": baseline is not None,
                "optimizer_steps": summary["optimizer_steps"],
                "stopped_before_backward": True,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 2 if packing_stop_reason else 0


if __name__ == "__main__":
    raise SystemExit(main())
