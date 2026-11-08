"""Task 5.3: per-turn GRPO logprobs on the generate prefix. Stop before backward."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contract import config
from contract.grpo_context import CapturingEngine, _turn_role
from contract.grpo_path import (
    _build_trainer,
    _load_m0_scripted_raws,
    _mask_report,
    _run_one_instrumented,
    _task_tuple,
    make_instrumented_rollout_func,
)
from contract.grpo_prep import SEED_ORDER, _package_versions, _write_json
from contract.harness import RefundEpisode, one_action_user_prompt
from contract.rewards import reward_B, reward_C
from contract.run import _seed_generation

TASK52_CAPTURES = ROOT / "artifacts" / "task-5-2" / "generate_captures.json"
TASK53 = ROOT / "artifacts" / "task-5-3"
# fp32 log_softmax vs logsumexp on fp16 logits. Two eval forwards must stay inside this.
LOGPROB_ATOL = 1e-5
OBS_NEEDLE = "Tool result for "


def audit_saved_captures(path: Path = TASK52_CAPTURES) -> dict[str, Any]:
    """Task 5.2 kept full generated ids and only the head/tail of each generate prefix."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    turns = []
    for call in payload["calls"]:
        head = list(call.get("input_ids_head") or [])
        tail = list(call.get("input_ids_tail") or [])
        new = list(call.get("new_token_ids") or [])
        input_len = int(call["input_len"])
        new_len = int(call["new_len"])
        full = call.get("input_ids")
        saved_prefix = len(full) if isinstance(full, list) else len(head) + len(tail)
        overlap = max(0, len(head) + len(tail) - input_len)
        missing_middle = input_len - (len(head) + len(tail) - overlap)
        turns.append(
            {
                "turn_index": call["turn_index"],
                "input_len": input_len,
                "new_len": new_len,
                "full_input_ids_present": isinstance(full, list) and len(full) == input_len,
                "saved_prefix_ids": saved_prefix if isinstance(full, list) else len(head) + len(tail),
                "head_len": len(head),
                "tail_len": len(tail),
                "missing_middle_ids": missing_middle,
                "new_token_ids_complete": len(new) == new_len,
            }
        )
    missing = [row for row in turns if not row["full_input_ids_present"]]
    return {
        "path": str(path.relative_to(ROOT)),
        "seed": payload.get("seed"),
        "n_turns": len(turns),
        "generated_ids_complete": all(row["new_token_ids_complete"] for row in turns),
        "full_input_ids_complete": not missing,
        "missing": missing,
        "gap": (
            "Full generate input_ids are absent. Each turn stores input_ids_head (32) and "
            "input_ids_tail (32) only, so the middle of the prefix cannot be rebuilt. "
            "new_token_ids are complete for every turn. Re-tokenizing text is not a substitute."
        ),
    }


def _id_bytes(ids: list[int]) -> bytes:
    import torch

    return torch.tensor(ids, dtype=torch.int64).numpy().tobytes()


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _decode(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False)


def _token_covering(tokenizer, ids: list[int], char_index: int) -> int:
    lo, hi = 0, len(ids) - 1
    answer = len(ids) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if len(_decode(tokenizer, ids[: mid + 1])) > char_index:
            answer = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return answer


def _observation_token_indexes(tokenizer, prefix_ids: list[int]) -> list[int]:
    text = _decode(tokenizer, prefix_ids)
    indexes: list[int] = []
    start = 0
    while True:
        at = text.find(OBS_NEEDLE, start)
        if at < 0:
            break
        last = at + len(OBS_NEEDLE) - 1
        begin = _token_covering(tokenizer, prefix_ids, at)
        end = _token_covering(tokenizer, prefix_ids, last)
        indexes.extend(range(begin, end + 1))
        start = at + len(OBS_NEEDLE)
    return indexes


def _forward_new_logits(model, prefix_ids: list[int], new_ids: list[int]):
    """Eval forward. Logits that predict exactly the generated suffix. No temperature scaling."""
    import torch

    device = next(model.parameters()).device
    sequence = list(prefix_ids) + list(new_ids)
    input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
    prefix_slice = input_ids[0, : len(prefix_ids)].detach().cpu()
    n_new = len(new_ids)
    kwargs = {"input_ids": input_ids, "use_cache": False, "logits_to_keep": n_new + 1}
    try:
        outputs = model(**kwargs)
    except TypeError:
        kwargs.pop("logits_to_keep")
        outputs = model(**kwargs)
    logits = outputs.logits[:, :-1, :]
    logits = logits[:, -n_new:, :].float()
    return prefix_slice, logits


def _as_batch_logits(logits, n_new: int):
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    if logits.dim() != 3 or logits.shape[1] != n_new:
        raise RuntimeError(f"expected logits (1, {n_new}, vocab), got {tuple(logits.shape)}")
    return logits


def _log_softmax_gather(logits, new_ids: list[int]):
    import torch

    logits = _as_batch_logits(logits, len(new_ids))
    targets = torch.tensor(new_ids, dtype=torch.long, device=logits.device).view(1, -1, 1)
    return torch.log_softmax(logits, dim=-1).gather(-1, targets).squeeze(0).squeeze(-1)


def _logsumexp_gather(logits, new_ids: list[int]):
    import torch

    logits = _as_batch_logits(logits, len(new_ids))
    targets = torch.tensor(new_ids, dtype=torch.long, device=logits.device).view(1, -1, 1)
    selected = logits.gather(-1, targets).squeeze(0).squeeze(-1)
    return selected - torch.logsumexp(logits, dim=-1).squeeze(0)


def _full_forward_token_logprobs(model, prefix_ids: list[int], new_ids: list[int], temperature: float):
    """Independent reference: one eval forward, no logits_to_keep.

    Token new_ids[j] sits at sequence index prefix_len + j and is scored from logit
    position prefix_len - 1 + j. Model logprob is log_softmax of the raw row.
    TRL logprob is selective_log_softmax of that row divided by the pilot temperature.
    """
    import torch
    from trl.trainer.utils import selective_log_softmax

    device = next(model.parameters()).device
    prefix_len = len(prefix_ids)
    sequence = list(prefix_ids) + list(new_ids)
    input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
    prefix_bytes = bytes(input_ids[0, :prefix_len].detach().cpu().numpy().tobytes())
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits
    if logits.shape[1] != len(sequence):
        raise RuntimeError(
            f"full forward logits length {logits.shape[1]} != sequence {len(sequence)}"
        )
    model_logps = []
    trl_logps = []
    explicit_logps = []
    positions = []
    for j, token_id in enumerate(new_ids):
        position = prefix_len - 1 + j
        if sequence[position + 1] != token_id:
            raise RuntimeError(f"new token {j} is not at sequence index {position + 1}")
        row = logits[0, position].float()
        model_logps.append(torch.log_softmax(row, dim=-1)[token_id])
        scaled = row / temperature
        explicit_logps.append(torch.log_softmax(scaled, dim=-1)[token_id])
        trl_logps.append(
            selective_log_softmax(scaled.unsqueeze(0), torch.tensor([token_id], device=device)).squeeze(0)
        )
        positions.append(position)
    del outputs, logits
    return {
        "positions": positions,
        "prefix_bytes": prefix_bytes,
        "model_logps": torch.stack(model_logps),
        "trl_logps": torch.stack(trl_logps),
        "explicit_temperature_logps": torch.stack(explicit_logps),
    }


def score_turns(model, tokenizer, calls: list[dict[str, Any]], temperature: float) -> dict[str, Any]:
    """Score each turn on its own generate prefix. Tool text stays in the prefix and out of the loss."""
    import torch

    model.eval()
    rows = []
    max_scorer_vs_full = 0.0
    max_trl_vs_explicit = 0.0
    for call in calls:
        prefix = list(call["input_ids"])
        new = list(call["new_token_ids"])
        if not new:
            raise RuntimeError(f"turn {call['turn_index']} has no generated token ids")
        with torch.no_grad():
            scoring_prefix, scoring_logits = _forward_new_logits(model, prefix, new)
            scoring_logps = _log_softmax_gather(scoring_logits, new)
        reference = _full_forward_token_logprobs(model, prefix, new, temperature)
        scorer_diff = (scoring_logps - reference["model_logps"]).abs()
        trl_diff = (reference["trl_logps"] - reference["explicit_temperature_logps"]).abs()
        max_scorer = float(scorer_diff.max().item())
        max_trl = float(trl_diff.max().item())
        max_scorer_vs_full = max(max_scorer_vs_full, max_scorer)
        max_trl_vs_explicit = max(max_trl_vs_explicit, max_trl)
        gen_id_bytes = _id_bytes(prefix)
        score_id_bytes = bytes(scoring_prefix.numpy().tobytes())
        ref_id_bytes = reference["prefix_bytes"]
        gen_text_bytes = _decode(tokenizer, prefix).encode("utf-8")
        score_text_bytes = _decode(tokenizer, scoring_prefix.tolist()).encode("utf-8")
        obs_indexes = _observation_token_indexes(tokenizer, prefix)
        loss_mask = [0] * len(prefix) + [1] * len(new)
        obs_in_loss = [i for i in obs_indexes if loss_mask[i] != 0]
        scored_text = _decode(tokenizer, new)
        rows.append(
            {
                "turn_index": call["turn_index"],
                "role": _turn_role(call["turn_index"], len(calls)),
                "prefix_len": len(prefix),
                "scored_tokens": len(new),
                "prefix_id_bytes": len(gen_id_bytes),
                "prefix_utf8_bytes": len(gen_text_bytes),
                "prefix_id_bytes_equal": gen_id_bytes == score_id_bytes == ref_id_bytes,
                "prefix_utf8_bytes_equal": gen_text_bytes == score_text_bytes,
                "prefix_sha256": _sha256(gen_id_bytes),
                "scoring_prefix_sha256": _sha256(score_id_bytes),
                "tool_result_markers_in_prefix": _decode(tokenizer, prefix).count(OBS_NEEDLE),
                "observation_token_count": len(set(obs_indexes)),
                "observation_tokens_in_scoring_context": all(i < len(prefix) for i in obs_indexes),
                "observation_tokens_excluded_from_loss": not obs_in_loss,
                "scored_text_contains_tool_result": OBS_NEEDLE in scored_text,
                "loss_mask_scored": int(sum(loss_mask)),
                "loss_mask_excluded": int(len(loss_mask) - sum(loss_mask)),
                "logit_position_first": reference["positions"][0],
                "logit_position_last": reference["positions"][-1],
                "logit_position_rule": "prefix_len - 1 + j",
                "model_logprob_sum": float(reference["model_logps"].sum().item()),
                "model_logprob_mean": float(reference["model_logps"].mean().item()),
                "model_logprob_head": [round(float(x), 6) for x in reference["model_logps"][:4].tolist()],
                "trl_temperature": temperature,
                "trl_logprob_sum": float(reference["trl_logps"].sum().item()),
                "trl_logprob_mean": float(reference["trl_logps"].mean().item()),
                "trl_logprob_head": [round(float(x), 6) for x in reference["trl_logps"][:4].tolist()],
                "max_abs_scorer_vs_full_forward": max_scorer,
                "max_abs_trl_selective_vs_explicit": max_trl,
                "scorer_within_atol": max_scorer <= LOGPROB_ATOL,
                "trl_within_atol": max_trl <= LOGPROB_ATOL,
                "max_abs_logprob_diff": max_scorer,
                "logprob_within_atol": max_scorer <= LOGPROB_ATOL and max_trl <= LOGPROB_ATOL,
                "logprob_head": [round(float(x), 6) for x in scoring_logps[:4].tolist()],
            }
        )
    return {
        "atol": LOGPROB_ATOL,
        "temperature": temperature,
        "reference": (
            "Independent reference is one eval forward without logits_to_keep. "
            "Token j is read from logit index prefix_len - 1 + j. "
            "Model logprob is log_softmax of the raw logit row. "
            "TRL logprob is selective_log_softmax of that row divided by the pilot temperature."
        ),
        "max_abs_logprob_diff": max_scorer_vs_full,
        "max_abs_scorer_vs_full_forward": max_scorer_vs_full,
        "max_abs_trl_selective_vs_explicit": max_trl_vs_explicit,
        "all_prefix_bytes_equal": all(row["prefix_id_bytes_equal"] and row["prefix_utf8_bytes_equal"] for row in rows),
        "all_logprobs_within_atol": all(row["logprob_within_atol"] for row in rows),
        "observations_in_context_and_out_of_loss": all(
            (row["turn_index"] == 0 or row["tool_result_markers_in_prefix"] >= 1)
            and row["observation_tokens_in_scoring_context"]
            and row["observation_tokens_excluded_from_loss"]
            and not row["scored_text_contains_tool_result"]
            for row in rows
        ),
        "turns": rows,
        "reduction": (
            "Scorer logprob uses logits_to_keep and is compared with the full-forward model logprob. "
            "TRL logprob divides the same full-forward rows by the pilot temperature before selective_log_softmax."
        ),
    }


def _group_advantages(rewards: list[float], reward_name: str) -> dict[str, Any]:
    """TRL 1.7.1 default: one reward column, then group-scale with nanstd (Bessel)."""
    import torch
    from trl.trainer.utils import nanstd

    values = torch.tensor(rewards, dtype=torch.float32)
    mean = torch.nanmean(values)
    std = nanstd(values)
    advantages = (values - mean) / (std + 1e-4)
    return {
        "rewards": values.tolist(),
        "mean": float(mean.item()),
        "std_bessel": float(std.item()),
        "eps": 1e-4,
        "advantages": advantages.tolist(),
        "formula": (
            f"A_i = (r_i - mean(r)) / (nanstd(r) + 1e-4), r_i = {reward_name}, "
            "nanstd uses Bessel correction"
        ),
    }


def _dapo_scalar(advantages: list[float], masks: list[list[int]], num_items_in_batch: float) -> dict[str, Any]:
    """Reduce -A_i over the loss mask the way TRL's DAPO branch does. No backward."""
    import torch

    width = max(len(mask) for mask in masks)
    mask = torch.zeros(len(masks), width, dtype=torch.float32)
    for i, row in enumerate(masks):
        mask[i, : len(row)] = torch.tensor(row, dtype=torch.float32)
    adv = torch.tensor(advantages, dtype=torch.float32).unsqueeze(1)
    per_token_loss = -adv.expand_as(mask)
    numerator = float((per_token_loss * mask).sum().item())
    scalar = numerator / float(num_items_in_batch)
    scored = [int(sum(row)) for row in masks]
    total = float(sum(scored))
    manual = sum(-a * n for a, n in zip(advantages, scored)) / total
    weights = [n / total for n in scored]
    return {
        "denominator": float(num_items_in_batch),
        "mask_token_count": total,
        "denominator_matches_mask": abs(float(num_items_in_batch) - total) <= 1e-5,
        "weights": weights,
        "scalar_dapo": scalar,
        "scalar_manual": manual,
        "abs_diff": abs(scalar - manual),
        "within_atol": abs(scalar - manual) <= 1e-5 and abs(float(num_items_in_batch) - total) <= 1e-5,
        "formula": (
            "per_token_loss[i, t] = -A_i on scored tokens (on-policy ratio 1, beta 0). "
            "scalar = sum(per_token_loss * mask) / num_items_in_batch. "
            "manual = sum_i (-A_i * n_i) / sum_j n_j. weight_i = n_i / denominator."
        ),
    }


def _dapo_weights(scored_counts: list[int]) -> dict[str, Any]:
    total = sum(scored_counts)
    weights = [count / total for count in scored_counts]
    return {
        "loss_type": "dapo",
        "normalizer": total,
        "weights": weights,
        "formula": (
            "loss = sum_i sum_{t in scored(i)} per_token_loss_{i,t} / sum_j n_j, "
            "with per_token_loss = -A_i on-policy (ratio 1, beta 0). "
            "Trajectory weight w_i = n_i / sum_j n_j. "
            "n_i counts generated tokens over all turns; tool-observation tokens have mask 0."
        ),
    }


def capture_t0_seed100(tokenizer, model) -> dict[str, Any]:
    """Exactly one live rollout: t0, generation seed 100."""
    import agenttune.agentic.rollout_engines.rollout_factory as factory
    from agenttune.agentic.rollout_engines.transformers_engine import TransformersRolloutEngine

    task = _task_tuple("t0")
    episode = RefundEpisode(
        episode_id=task[0],
        order_id=task[1],
        customer_id=task[2],
        amount_cents=task[3],
        seed=config.ONE_ACTION_ENVIRONMENT_SEED,
    )
    prompt = one_action_user_prompt(task[1], task[2])
    engine = CapturingEngine(
        TransformersRolloutEngine(model=model, tokenizer=tokenizer),
        tokenizer,
    )
    _seed_generation(100)
    _packed, _steps, events = _run_one_instrumented(
        factory=factory,
        engine=engine,
        tokenizer=tokenizer,
        episode=episode,
        generation_seed=100,
        prompt_text=prompt,
        gen_kwargs={
            "max_new_tokens": config.ONE_ACTION_MAX_NEW_TOKENS,
            "temperature": config.ONE_ACTION_TEMPERATURE,
            "do_sample": config.ONE_ACTION_DO_SAMPLE,
            "top_p": config.ONE_ACTION_TOP_P,
            "max_length": config.CONTEXT_LENGTH,
        },
    )
    calls = list(engine.generate_calls)
    if len(calls) < 1:
        raise RuntimeError("t0 seed 100 produced no generate calls")
    saved = json.loads(TASK52_CAPTURES.read_text(encoding="utf-8"))
    saved_new = [list(item["new_token_ids"]) for item in saved["calls"]]
    fresh_new = [list(item["new_token_ids"]) for item in calls]
    return {
        "episode_id": "t0",
        "generation_seed": 100,
        "task_success": bool(events.get("task_success")),
        "final_answer": events.get("final_answer"),
        "n_turns": len(calls),
        "calls": calls,
        "matches_task52_new_token_ids": fresh_new == saved_new,
        "events_rollout_id": events.get("rollout_id"),
    }


def score_scripted_m0(trainer) -> dict[str, Any]:
    """Original m0 group of four. Rewards and advantages once per trajectory, no backward."""
    import torch

    m0 = _task_tuple("m0")
    scripted_fn = make_instrumented_rollout_func(
        episode_id=m0[0],
        order_id=m0[1],
        customer_id=m0[2],
        amount_cents=m0[3],
        generation_seeds=list(SEED_ORDER),
        tokenizer=trainer.processing_class,
        mode="scripted",
        scripted_raws=_load_m0_scripted_raws(),
    )
    trainer.rollout_func = scripted_fn
    prompt = [{"role": "user", "content": one_action_user_prompt(m0[1], m0[2])}]
    inputs = [
        {"prompt": prompt, "episode_id": "m0", "generation_seed": seed}
        for seed in SEED_ORDER
    ]
    recorded: dict[str, Any] = {}
    original = trainer._calculate_rewards

    def hooked(inputs_arg, prompts, completions, completion_ids_list):
        rewards = original(inputs_arg, prompts, completions, completion_ids_list)
        recorded["rewards_per_func"] = rewards.detach().float().cpu().tolist()
        recorded["n_rows"] = len(inputs_arg)
        recorded["rollout_ids"] = [
            (item.get("contract_events") or {}).get("rollout_id") for item in inputs_arg
        ]
        recorded["n_turns"] = [
            len((item.get("contract_events") or {}).get("raw_outputs") or []) for item in inputs_arg
        ]
        recorded["final_answers"] = [
            (item.get("contract_events") or {}).get("final_answer") for item in inputs_arg
        ]
        recorded["reward_name"] = trainer.reward_func_names[0]
        recorded["old_per_token_logps_requested"] = False
        return rewards

    steps_before = trainer._stop_before_optimizer.optimizer_steps
    trainer._calculate_rewards = hooked
    trainer.model.eval()
    try:
        with torch.no_grad():
            outputs = trainer._generate_and_score_completions(inputs)
    finally:
        trainer._calculate_rewards = original
    steps_after = trainer._stop_before_optimizer.optimizer_steps
    recorded["old_per_token_logps_in_outputs"] = "old_per_token_logps" in outputs
    if recorded["old_per_token_logps_in_outputs"] or pilot_old_logps_reason(trainer) is not None:
        raise RuntimeError(pilot_old_logps_reason(trainer) or f"{OLD_LOGPS_STOP} scoring outputs contain old_per_token_logps.")
    turn_input_ids = [list(row["turn_input_ids"]) for row in inputs]
    turn_new_token_ids = [list(row["turn_new_token_ids"]) for row in inputs]
    if turn_input_ids != scripted_fn.last_turn_input_ids or turn_new_token_ids != scripted_fn.last_turn_new_token_ids:
        raise RuntimeError("loss batch turn ids are not the ids the rollout returned")
    outputs["turn_input_ids"] = turn_input_ids
    outputs["turn_new_token_ids"] = turn_new_token_ids
    advantages = outputs["advantages"].detach().float().cpu().tolist()
    if advantages and isinstance(advantages[0], list):
        raise RuntimeError(f"advantages are not one scalar per trajectory: rank>1 sample {advantages[0][:4]}")
    per_func = recorded["rewards_per_func"]
    if any(len(row) != 1 for row in per_func):
        raise RuntimeError(f"expected one reward column, got widths {[len(row) for row in per_func]}")
    combined = [row[0] for row in per_func]
    reference = _group_advantages(combined, recorded["reward_name"])
    masks = list(scripted_fn.last_loss_masks)
    scored = [int(sum(mask)) for mask in masks]
    observed = [int(sum(1 for bit in mask if bit == 0)) for mask in masks]
    weights = _dapo_weights(scored)
    adv_diff = [abs(a - b) for a, b in zip(advantages, reference["advantages"])]
    trajectories = []
    for i, seed in enumerate(SEED_ORDER):
        runs = _mask_report(masks[i], None)["runs"]
        turn_runs = [run for run in runs if run["value"] == 1]
        trajectories.append(
            {
                "rollout_id": recorded["rollout_ids"][i],
                "generation_seed": seed,
                "final_answer": recorded["final_answers"][i],
                "n_turns": recorded["n_turns"][i],
                "n_scored_turns_in_mask": len(turn_runs),
                "n_scored_tokens": scored[i],
                "n_observation_tokens": observed[i],
                "reward_column": per_func[i][0],
                "reward": combined[i],
                "advantage": advantages[i],
                "advantage_reference": reference["advantages"][i],
                "loss_weight": weights["weights"][i],
                "per_token_loss_coefficient": -advantages[i] / weights["normalizer"],
            }
        )
    num_items = outputs["num_items_in_batch"]
    num_items = float(num_items.item() if hasattr(num_items, "item") else num_items)
    scalar = _dapo_scalar(advantages, masks, num_items)
    pilot_loss = dry_run_compute_loss(trainer, outputs, label=f"m0-{recorded['reward_name']}")
    return {
        "group": "m0",
        "reward": recorded["reward_name"],
        "n_trajectories": len(trajectories),
        "reward_callback_rows": recorded["n_rows"],
        "advantage_rank": 1,
        "advantage_once_per_trajectory": len(advantages) == 4 and recorded["n_rows"] == 4,
        "advantage_matches_trl_formula": max(adv_diff) <= 1e-5,
        "advantage_matches_formula": max(adv_diff) <= 1e-5,
        "max_abs_advantage_diff": max(adv_diff),
        "advantage_formula": reference,
        "aggregation": weights,
        "dapo_scalar": scalar,
        "old_per_token_logps_in_outputs": recorded["old_per_token_logps_in_outputs"],
        "pilot_loss": pilot_loss,
        "trajectories": trajectories,
        "optimizer_steps": steps_after - steps_before,
        "note": (
            "Each scored token of a trajectory is multiplied by that trajectory's single advantage. "
            "A longer answer or an extra model turn increases n_scored and therefore the DAPO weight. "
            "Observation tokens stay in the flat TRL mask with value 0 and add no weight."
        ),
    }


def _activate_reward(trainer, reward_fn) -> tuple:
    import torch

    saved = (
        list(trainer.reward_funcs),
        list(trainer.reward_func_names),
        list(trainer.reward_processing_classes),
        trainer.reward_weights.detach().clone(),
    )
    trainer.reward_funcs = [reward_fn]
    trainer.reward_func_names = [reward_fn.__name__]
    trainer.reward_processing_classes = [None]
    trainer.reward_weights = torch.ones(1, dtype=torch.float32)
    return saved


def _restore_reward(trainer, saved) -> None:
    trainer.reward_funcs, trainer.reward_func_names, trainer.reward_processing_classes, trainer.reward_weights = saved


def score_b_and_c(trainer) -> dict[str, Any]:
    """Same four scripted m0 rollouts, once with B only and once with C only."""
    modes = {}
    for label, fn in (("B", reward_B), ("C", reward_C)):
        print(f"task_5_3 scripted m0 {label}-only", flush=True)
        saved = _activate_reward(trainer, fn)
        try:
            modes[label] = score_scripted_m0(trainer)
        finally:
            _restore_reward(trainer, saved)
    return {
        "same_rollouts": ["m0-g100", "m0-g101", "m0-g102", "m0-g103"],
        "new_model_rollout": False,
        "B": modes["B"],
        "C": modes["C"],
    }


def old_logps_site(trainer) -> dict[str, Any]:
    import trl.trainer.grpo_trainer as grpo_trainer

    path = Path(grpo_trainer.__file__)
    lines = path.read_text(encoding="utf-8").splitlines()
    call_line = next(
        i for i, line in enumerate(lines, start=1) if "old_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(" in line
    )
    flat_line = next(
        i
        for i, line in enumerate(lines, start=1)
        if "prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)" in line
    )
    compute_loss_def = next(i for i, line in enumerate(lines, start=1) if line.startswith("    def _compute_loss("))
    loss_line = next(
        i
        for i, line in enumerate(lines, start=1)
        if i > compute_loss_def and line.strip() == "input_ids = torch.cat([prompt_ids, completion_ids], dim=1)"
    )
    fallback_line = next(
        i
        for i, line in enumerate(lines, start=1)
        if "old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None" in line
    )
    gas = int(trainer.args.gradient_accumulation_steps)
    steps = int(trainer.args.steps_per_generation)
    iterations = int(trainer.num_iterations)
    generate_every = steps * iterations
    computes = gas % generate_every != 0 or (
        bool(trainer.use_vllm) and bool(trainer.vllm_importance_sampling_correction)
    )
    return {
        "trl_file": str(path),
        "temperature": float(trainer.temperature),
        "gradient_accumulation_steps": gas,
        "steps_per_generation": steps,
        "num_iterations": iterations,
        "generate_every": generate_every,
        "use_vllm": bool(trainer.use_vllm),
        "condition": "gradient_accumulation_steps % (steps_per_generation * num_iterations) != 0 or vllm importance-sampling correction",
        "computes_old_per_token_logps": computes,
        "call_line": call_line,
        "flat_sequence_line": flat_line,
        "loss_forward_line": loss_line,
        "detach_fallback_line": fallback_line,
        "pilot_note": (
            "This pilot does not compute old_per_token_logps: the if immediately above "
            f"grpo_trainer.py:{call_line} is false, so the branch assigns None. "
            f"_compute_loss line {fallback_line} then uses the current per-token logprob.detach(). "
            "That current logprob is still a forward of the flat prompt+completion sequence at "
            f"line {loss_line}. If the condition becomes true, the same call at line {call_line} "
            f"scores prompt_completion_ids built at line {flat_line}. The pilot needs the per-turn "
            "generate prefixes in both of those forwards, because old and current logprobs must "
            "share the context model.generate actually saw."
        ),
    }


def flat_sequence_sites() -> dict[str, Any]:
    import trl.trainer.grpo_trainer as grpo_trainer

    path = Path(grpo_trainer.__file__)
    lines = path.read_text(encoding="utf-8").splitlines()
    needles = (
        "prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)",
        "input_ids = torch.cat([prompt_ids, completion_ids], dim=1)",
        'mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]',
    )
    hits = []
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped in needles or stripped.startswith("prompt_completion_ids = torch.cat([prompt_ids, completion_ids]"):
            hits.append({"line": number, "text": stripped})
    return {
        "trl_file": str(path),
        "trl_version": __import__("trl").__version__,
        "hits": hits,
        "limitation": (
            "GRPOTrainer scores one flat prompt_ids+completion_ids sequence per episode. "
            "The logit at completion position t is conditioned on that packed prefix, not on the "
            "input_ids that model.generate actually saw for that turn."
        ),
        "extension_point": (
            "In GRPOTrainer._compute_loss, before the cat that builds input_ids, branch when the "
            "rollout supplies per-turn (input_ids, new_token_ids). Run one eval forward per turn on "
            "that prefix plus the generated suffix, keep logprobs only for the suffix, and concatenate "
            "those rows into the existing (batch, scored_tokens) tensor. Leave advantage broadcast and "
            "the DAPO reduction unchanged. Do not retokenize and do not repair the flat stream."
        ),
    }


OLD_LOGPS_STOP = (
    "Per-turn scoring is locked to the pilot configuration where old_per_token_logps is absent. "
    "This run would compute old logprobs, and per-turn scoring does not support them yet. "
    "Stopping before the loss."
)


def pilot_old_logps_reason(trainer) -> str | None:
    """TRL computes old_per_token_logps only for this config. The pilot's config is the other one."""
    gas = int(trainer.args.gradient_accumulation_steps)
    steps = int(trainer.args.steps_per_generation)
    iterations = int(trainer.num_iterations)
    every = steps * iterations
    vllm_correction = bool(trainer.use_vllm) and bool(trainer.vllm_importance_sampling_correction)
    if every == 0 or gas % every != 0 or vllm_correction:
        return (
            f"{OLD_LOGPS_STOP} gradient_accumulation_steps={gas}, steps_per_generation={steps}, "
            f"num_iterations={iterations}, use_vllm={bool(trainer.use_vllm)}, "
            f"vllm_importance_sampling_correction={bool(trainer.vllm_importance_sampling_correction)}."
        )
    return None


def _graph_reaches(loss, source) -> bool:
    """True when the loss grad graph contains the per-turn logprob tensor. Does not call backward.

    Autograd node wrappers are recycled as soon as nothing references them, so the walk
    keeps every node it has seen. Otherwise a later node can reuse an id and be skipped.
    """
    if loss.grad_fn is None or not getattr(source, "requires_grad", False):
        return False
    target = source.grad_fn
    seen: set[int] = set()
    held = [loss.grad_fn]
    stack = [loss.grad_fn]
    while stack:
        fn = stack.pop()
        if fn is None:
            continue
        if target is not None and fn is target:
            return True
        if getattr(fn, "variable", None) is source:
            return True
        marker = id(fn)
        if marker in seen:
            continue
        seen.add(marker)
        held.append(fn)
        for item in getattr(fn, "next_functions", ()):
            child = item[0] if isinstance(item, tuple) else getattr(item, "node", None)
            if child is not None:
                held.append(child)
                stack.append(child)
    return False


def _score_rollout_turns(model, trainer, inputs):
    """Per-turn logprobs with TRL's in-place temperature division, placed on the loss mask."""
    import torch
    from trl.trainer.utils import selective_log_softmax

    completion_ids = inputs["completion_ids"]
    completion_mask = inputs["completion_mask"]
    tool_mask = inputs.get("tool_mask")
    mask = completion_mask if tool_mask is None else completion_mask * tool_mask
    turn_inputs = inputs["turn_input_ids"]
    turn_news = inputs["turn_new_token_ids"]
    batch, width = completion_ids.shape
    if len(turn_inputs) != batch or len(turn_news) != batch:
        raise RuntimeError(
            f"turn contexts {len(turn_inputs)}/{len(turn_news)} != loss batch {batch}"
        )
    device = completion_ids.device
    temperature = float(trainer.temperature)
    rows = []
    dtype_before = None
    dtype_after = None
    selective_vs_explicit = 0.0
    generated = 0
    for row_index in range(batch):
        pieces = []
        expected: list[int] = []
        for turn_index, (prefix, new) in enumerate(zip(turn_inputs[row_index], turn_news[row_index])):
            prefix = [int(token) for token in prefix]
            new = [int(token) for token in new]
            if not new:
                raise RuntimeError(f"row {row_index} turn {turn_index} has no new_token_ids")
            expected.extend(new)
            sequence = prefix + new
            input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids, use_cache=False)
            # Same slice TRL uses: drop the last position, then keep the generated suffix.
            # Token new[j] is scored from logit index prefix_len - 1 + j.
            logits = outputs.logits[:, :-1, :][:, -len(new) :, :].clone()
            if dtype_before is None:
                dtype_before = str(logits.dtype).replace("torch.", "")
            # TRL divides in place on the logit dtype. Do not upcast before this division.
            logits.div_(temperature)
            dtype_after = str(logits.dtype).replace("torch.", "")
            if logits.dtype != outputs.logits.dtype:
                raise RuntimeError(
                    f"temperature division changed logit dtype {outputs.logits.dtype} -> {logits.dtype}"
                )
            targets = torch.tensor([new], dtype=torch.long, device=device)
            logps = selective_log_softmax(logits, targets).squeeze(0)
            with torch.no_grad():
                explicit = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(0).squeeze(-1)
                selective_vs_explicit = max(
                    selective_vs_explicit,
                    float((logps.detach() - explicit).abs().max().item()),
                )
            pieces.append(logps)
            del outputs, logits
            print(
                f"turn-loss row {row_index} turn {turn_index} prefix={len(prefix)} new={len(new)}",
                flush=True,
            )
        flat = torch.cat(pieces)
        mask_row = mask[row_index]
        n_mask = int(mask_row.sum().item())
        generated += len(expected)
        if n_mask != flat.numel() or n_mask != len(expected):
            raise RuntimeError(
                f"row {row_index}: loss mask {n_mask} != generated tokens {len(expected)}"
            )
        packed = completion_ids[row_index][mask_row.bool()].detach().cpu().tolist()
        if [int(token) for token in packed] != expected:
            raise RuntimeError(
                f"row {row_index}: mask positions are not the rollout new_token_ids"
            )
        chunks = []
        cursor = 0
        position = 0
        mask_bits = mask_row.detach().cpu().tolist()
        while position < width:
            bit = int(mask_bits[position]) != 0
            end = position + 1
            while end < width and (int(mask_bits[end]) != 0) == bit:
                end += 1
            span = end - position
            if bit:
                chunks.append(flat[cursor : cursor + span])
                cursor += span
            else:
                chunks.append(torch.zeros(span, device=flat.device, dtype=flat.dtype))
            position = end
        if cursor != flat.numel():
            raise RuntimeError(f"row {row_index}: placed {cursor} logprobs of {flat.numel()}")
        rows.append(torch.cat(chunks))
    logps = torch.stack(rows)
    entropies = torch.zeros_like(logps)
    num_items = inputs["num_items_in_batch"]
    num_items_f = float(num_items.detach().item() if hasattr(num_items, "detach") else num_items)
    mask_tokens = int(mask.sum().item())
    report = {
        "temperature": temperature,
        "logit_dtype_before_div": dtype_before,
        "logit_dtype_after_div": dtype_after,
        "logit_dtype_unchanged_by_div": dtype_before == dtype_after and dtype_before is not None,
        "division": "logits.div_(temperature) on the model logit dtype, then selective_log_softmax",
        "selective_vs_explicit_max_abs": selective_vs_explicit,
        "generated_tokens": generated,
        "mask_tokens": mask_tokens,
        "denominator": num_items_f,
        "mask_equals_generated_tokens": mask_tokens == generated,
        "denominator_equals_generated_tokens": abs(num_items_f - generated) <= 1e-5,
        "logprob_requires_grad": bool(logps.requires_grad),
        "logprob_grad_fn": type(logps.grad_fn).__name__ if logps.grad_fn is not None else None,
        "masked_logprob_sum": float((logps.detach() * mask.to(logps.dtype)).sum().item()),
    }
    if not report["mask_equals_generated_tokens"] or not report["denominator_equals_generated_tokens"]:
        raise RuntimeError(
            f"mask {mask_tokens} or denominator {num_items_f} != generated tokens {generated}"
        )
    return logps, entropies, report


def install_turn_scoring(trainer) -> None:
    """Route _compute_loss's logprob call through per-turn rollout ids. DAPO stays in TRL."""
    reason = pilot_old_logps_reason(trainer)
    if reason is not None:
        raise RuntimeError(reason)
    if getattr(trainer, "_turn_scoring_installed", False):
        return
    original_compute = trainer._compute_loss
    original_get = trainer._get_per_token_logps_and_entropies
    trainer._turn_loss_inputs = None
    trainer._turn_logprob_report = None
    trainer._turn_logps = None

    def _compute_loss(model, inputs):
        blocked = pilot_old_logps_reason(trainer)
        old = inputs.get("old_per_token_logps")
        if blocked is not None or old is not None:
            detail = blocked or "old_per_token_logps was passed into _compute_loss."
            raise RuntimeError(
                f"{detail} Per-turn scoring does not support old logprobs yet. Stopping before the loss."
            )
        if "turn_input_ids" not in inputs or "turn_new_token_ids" not in inputs:
            raise RuntimeError(
                "rollout did not pass per-turn input_ids and new_token_ids into _compute_loss"
            )
        trainer._turn_loss_inputs = inputs
        try:
            return original_compute(model, inputs)
        finally:
            trainer._turn_loss_inputs = None

    def _get(model, input_ids, attention_mask, logits_to_keep, **kwargs):
        held = trainer._turn_loss_inputs
        if held is None:
            return original_get(model, input_ids, attention_mask, logits_to_keep, **kwargs)
        logps, entropies, report = _score_rollout_turns(model, trainer, held)
        trainer._turn_logprob_report = report
        trainer._turn_logps = logps
        return logps, entropies, None

    trainer._compute_loss = _compute_loss
    trainer._get_per_token_logps_and_entropies = _get
    trainer._turn_scoring_installed = True


def _enable_loss_graph(model) -> int:
    """Leave LoRA trainable and checkpoint the forward so the loss has a grad graph."""
    if not getattr(model, "_turn_loss_graph_ready", False):
        if hasattr(model, "config") and model.config is not None:
            model.config.use_cache = False
        base = getattr(model, "get_base_model", lambda: model)()
        if hasattr(base, "config") and base.config is not None:
            base.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()
        model._turn_loss_graph_ready = True
    model.train()
    trainable = sum(1 for param in model.parameters() if param.requires_grad)
    if trainable < 1:
        raise RuntimeError("no trainable parameter; the loss cannot grow a gradient graph")
    return trainable


def _manual_dapo(advantages, mask, denominator: float) -> float:
    import torch

    adv = advantages.detach().float()
    if adv.dim() == 1:
        adv = adv.unsqueeze(1)
    per_token = -adv.expand_as(mask.float())
    return float((per_token * mask.float()).sum().item()) / float(denominator)


def dry_run_compute_loss(trainer, inputs, *, label: str) -> dict[str, Any]:
    """Call the real _compute_loss. Inspect the grad graph. Do not backward."""
    import gc

    import torch

    blocked = pilot_old_logps_reason(trainer)
    if blocked is not None or inputs.get("old_per_token_logps") is not None:
        raise RuntimeError(blocked or f"{OLD_LOGPS_STOP} old_per_token_logps is present in the batch.")
    if float(trainer.temperature) != config.ONE_ACTION_TEMPERATURE:
        raise RuntimeError(f"pilot temperature {trainer.temperature} != {config.ONE_ACTION_TEMPERATURE}")
    if trainer.loss_type != "dapo":
        raise RuntimeError(f"pilot loss_type {trainer.loss_type} is not dapo")
    steps_before = trainer._stop_before_optimizer.optimizer_steps
    trainable = _enable_loss_graph(trainer.model)
    print(f"task_5_3 real _compute_loss {label}", flush=True)
    with torch.enable_grad():
        loss = trainer.compute_loss(trainer.model, inputs)
    report = dict(trainer._turn_logprob_report or {})
    logps = trainer._turn_logps
    reaches = _graph_reaches(loss, logps) if logps is not None else False
    mask = inputs["completion_mask"] if "tool_mask" not in inputs else inputs["completion_mask"] * inputs["tool_mask"]
    manual = _manual_dapo(inputs["advantages"], mask, report["denominator"])
    scalar = float(loss.detach().item())
    grad_name = type(loss.grad_fn).__name__ if loss.grad_fn is not None else None
    result = {
        "label": label,
        "loss_type": trainer.loss_type,
        "temperature": report.get("temperature"),
        "logit_dtype_before_div": report.get("logit_dtype_before_div"),
        "logit_dtype_after_div": report.get("logit_dtype_after_div"),
        "logit_dtype_unchanged_by_div": report.get("logit_dtype_unchanged_by_div"),
        "division": report.get("division"),
        "selective_vs_explicit_max_abs": report.get("selective_vs_explicit_max_abs"),
        "generated_tokens": report.get("generated_tokens"),
        "mask_tokens": report.get("mask_tokens"),
        "denominator": report.get("denominator"),
        "mask_equals_generated_tokens": report.get("mask_equals_generated_tokens"),
        "denominator_equals_generated_tokens": report.get("denominator_equals_generated_tokens"),
        "masked_logprob_sum": report.get("masked_logprob_sum"),
        "logprob_requires_grad": report.get("logprob_requires_grad"),
        "loss_scalar": scalar,
        "scalar_manual": manual,
        "scalar_abs_diff": abs(scalar - manual),
        "loss_requires_grad": bool(loss.requires_grad),
        "loss_grad_fn": grad_name,
        "graph_reaches_turn_logps": reaches,
        "trainable_parameters": trainable,
        "old_per_token_logps_absent": "old_per_token_logps" not in inputs or inputs.get("old_per_token_logps") is None,
        "advantage_shape": list(inputs["advantages"].shape),
        "one_advantage_per_trajectory": inputs["advantages"].dim() == 1 and inputs["advantages"].shape[0] == inputs["completion_ids"].shape[0],
        "optimizer_steps": trainer._stop_before_optimizer.optimizer_steps - steps_before,
        "backward_called": False,
    }
    del loss, logps
    trainer._turn_logps = None
    trainer.model.eval()
    gc.collect()
    return result


def dry_run_saved_t0(trainer, calls: list[dict[str, Any]]) -> dict[str, Any]:
    """One trajectory from the saved t0 turns. Singleton group advantage is 0. No new rollout."""
    import torch

    turn_input = [list(call["input_ids"]) for call in calls]
    turn_new = [list(call["new_token_ids"]) for call in calls]
    generated = [token for turn in turn_new for token in turn]
    device = next(trainer.model.parameters()).device
    completion = torch.tensor([generated], dtype=torch.long, device=device)
    prompt = torch.tensor([turn_input[0]], dtype=torch.long, device=device)
    n = len(generated)
    inputs = {
        "prompt_ids": prompt,
        "prompt_mask": torch.ones_like(prompt),
        "completion_ids": completion,
        "completion_mask": torch.ones_like(completion),
        "tool_mask": torch.ones_like(completion),
        "advantages": torch.zeros(1, dtype=torch.float32, device=device),
        "num_items_in_batch": torch.tensor(float(n), device=device),
        "turn_input_ids": [turn_input],
        "turn_new_token_ids": [turn_new],
    }
    report = dry_run_compute_loss(trainer, inputs, label="t0-seed100")
    report["source"] = "artifacts/task-5-3/t0_seed100_ids.json"
    report["n_turns"] = len(calls)
    report["turn_new_lengths"] = [len(turn) for turn in turn_new]
    report["advantage"] = 0.0
    report["advantage_note"] = (
        "One saved trajectory is a group of 1. TRL's group advantage is r - mean(r) = 0. "
        "The DAPO scalar is therefore 0. The grad graph still runs through the per-turn logprobs "
        "via exp(logp - logp.detach())."
    )
    return report


def _install_backward_tripwire() -> dict[str, int]:
    import torch

    counts = {"backward": 0}

    def refuse(*_args, **_kwargs):
        counts["backward"] += 1
        raise RuntimeError("task 5.3 refuses backward")

    torch.Tensor.backward = refuse
    torch.autograd.backward = refuse
    return counts


def main(argv: list[str] | None = None) -> int:
    del argv
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TASK53.mkdir(parents=True, exist_ok=True)
    audit = audit_saved_captures()
    _write_json(TASK53 / "capture_gap.json", audit)
    print(json.dumps({"capture_gap": audit["gap"], "missing_turns": len(audit["missing"])}, indent=2))
    if audit["full_input_ids_complete"]:
        raise RuntimeError("expected incomplete task 5.2 prefixes; refusing to skip the single rerun")

    torch.manual_seed(config.SEED)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, revision=config.MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        revision=config.MODEL_REVISION,
        dtype=torch.float16,
    )
    model.eval()

    # Trainer exists so scripted advantages come from GRPOTrainer, and so LoRA matches the train stack.
    # The live capture is one direct rollout, not _generate_and_score_completions.
    placeholder = make_instrumented_rollout_func(
        episode_id="t0",
        order_id="ORD-2001",
        customer_id="CUS-601",
        amount_cents=1500,
        generation_seeds=[100],
        tokenizer=tokenizer,
        mode="scripted",
        scripted_raws={100: ["noop"]},
    )
    trainer = _build_trainer(tokenizer, model, placeholder, TASK53 / "scratch")
    trainer.model.eval()
    install_turn_scoring(trainer)
    backward_counts = _install_backward_tripwire()

    saved_ids = TASK53 / "t0_seed100_ids.json"
    if saved_ids.exists():
        captured = json.loads(saved_ids.read_text(encoding="utf-8"))
        calls = captured["calls"]
        if captured.get("generation_seed") != 100 or any(
            len(call["input_ids"]) != call["input_len"] or len(call["new_token_ids"]) != call["new_len"]
            for call in calls
        ):
            raise RuntimeError(f"incomplete id capture in {saved_ids}")
        print(
            f"reuse captured turns={captured['n_turns']} success={captured['task_success']} "
            f"matches_task52_new_ids={captured['matches_task52_new_token_ids']}",
            flush=True,
        )
    else:
        raise RuntimeError(f"refusing a new rollout; missing {saved_ids}")

    if float(trainer.temperature) != config.ONE_ACTION_TEMPERATURE:
        raise RuntimeError(
            f"pilot temperature {trainer.temperature} != {config.ONE_ACTION_TEMPERATURE}"
        )
    print("task_5_3 per-turn scoring forwards", flush=True)
    turn_report = score_turns(trainer.model, tokenizer, calls, temperature=float(trainer.temperature))

    print("task_5_3 saved t0 through real _compute_loss", flush=True)
    t0_loss = dry_run_saved_t0(trainer, calls)

    print("task_5_3 scripted m0 B-only and C-only", flush=True)
    scripted_modes = score_b_and_c(trainer)
    sites = flat_sequence_sites()
    old_site = old_logps_site(trainer)
    summary = {
        "kind": "grpo_per_turn_loss_check",
        "base_commit": "1af8b556d8ca18b191313627f306620ddd2ad5ea",
        "versions": _package_versions(),
        "capture_gap": audit,
        "live_rollout": {
            "episode_id": "t0",
            "generation_seed": 100,
            "n_rollouts": 1,
            "task_success": captured["task_success"],
            "final_answer": captured["final_answer"],
            "n_turns": captured["n_turns"],
            "matches_task52_new_token_ids": captured["matches_task52_new_token_ids"],
            "id_source": "model.generate input_ids and completion_ids, not a retokenized string",
        },
        "logprob_atol": LOGPROB_ATOL,
        "max_abs_scorer_vs_full_forward": turn_report["max_abs_scorer_vs_full_forward"],
        "max_abs_trl_selective_vs_explicit": turn_report["max_abs_trl_selective_vs_explicit"],
        "max_abs_logprob_diff": turn_report["max_abs_logprob_diff"],
        "turn_check": turn_report,
        "scripted_m0_B": scripted_modes["B"],
        "scripted_m0_C": scripted_modes["C"],
        "old_per_token_logps": old_site,
        "flat_sequence": sites,
        "pilot_loss_t0": t0_loss,
        "optimizer_steps": (
            scripted_modes["B"]["optimizer_steps"]
            + scripted_modes["C"]["optimizer_steps"]
            + t0_loss["optimizer_steps"]
            + scripted_modes["B"]["pilot_loss"]["optimizer_steps"]
            + scripted_modes["C"]["pilot_loss"]["optimizer_steps"]
        ),
        "backward_calls": backward_counts["backward"],
        "train_called": False,
        "stopped_before_backward": (
            backward_counts["backward"] == 0
            and scripted_modes["B"]["optimizer_steps"] == 0
            and scripted_modes["C"]["optimizer_steps"] == 0
            and t0_loss["optimizer_steps"] == 0
            and scripted_modes["B"]["pilot_loss"]["optimizer_steps"] == 0
            and scripted_modes["C"]["pilot_loss"]["optimizer_steps"] == 0
        ),
    }
    _write_json(TASK53 / "summary.json", summary)
    _write_json(TASK53 / "turn_table.json", {"atol": LOGPROB_ATOL, "turns": turn_report["turns"]})
    _write_json(TASK53 / "scripted_m0_weights.json", scripted_modes)
    _write_json(
        TASK53 / "pilot_loss.json",
        {
            "temperature": config.ONE_ACTION_TEMPERATURE,
            "old_per_token_logps_computed": old_site["computes_old_per_token_logps"],
            "t0": t0_loss,
            "m0_B": scripted_modes["B"]["pilot_loss"],
            "m0_C": scripted_modes["C"]["pilot_loss"],
            "optimizer_steps": summary["optimizer_steps"],
            "backward_calls": summary["backward_calls"],
        },
    )
    print(
        json.dumps(
            {
                "prefix_bytes_equal": turn_report["all_prefix_bytes_equal"],
                "max_abs_scorer_vs_full_forward": turn_report["max_abs_scorer_vs_full_forward"],
                "max_abs_trl_selective_vs_explicit": turn_report["max_abs_trl_selective_vs_explicit"],
                "atol": LOGPROB_ATOL,
                "B_advantage_diff": scripted_modes["B"]["max_abs_advantage_diff"],
                "C_advantage_diff": scripted_modes["C"]["max_abs_advantage_diff"],
                "B_dapo_diff": scripted_modes["B"]["dapo_scalar"]["abs_diff"],
                "C_dapo_diff": scripted_modes["C"]["dapo_scalar"]["abs_diff"],
                "B_denominator": scripted_modes["B"]["dapo_scalar"]["denominator"],
                "C_denominator": scripted_modes["C"]["dapo_scalar"]["denominator"],
                "old_logps_computed": old_site["computes_old_per_token_logps"],
                "t0_loss": t0_loss["loss_scalar"],
                "t0_grad_fn": t0_loss["loss_grad_fn"],
                "t0_graph": t0_loss["graph_reaches_turn_logps"],
                "t0_dtype": t0_loss["logit_dtype_before_div"],
                "t0_generated": t0_loss["generated_tokens"],
                "B_loss": scripted_modes["B"]["pilot_loss"]["loss_scalar"],
                "B_graph": scripted_modes["B"]["pilot_loss"]["graph_reaches_turn_logps"],
                "B_generated": scripted_modes["B"]["pilot_loss"]["generated_tokens"],
                "C_loss": scripted_modes["C"]["pilot_loss"]["loss_scalar"],
                "C_graph": scripted_modes["C"]["pilot_loss"]["graph_reaches_turn_logps"],
                "C_generated": scripted_modes["C"]["pilot_loss"]["generated_tokens"],
                "optimizer_steps": summary["optimizer_steps"],
                "backward_calls": summary["backward_calls"],
            },
            indent=2,
        )
    )
    def _loss_ok(report: dict[str, Any]) -> bool:
        return bool(
            report["old_per_token_logps_absent"]
            and report["logit_dtype_unchanged_by_div"]
            and report["temperature"] == config.ONE_ACTION_TEMPERATURE
            and report["mask_equals_generated_tokens"]
            and report["denominator_equals_generated_tokens"]
            and report["logprob_requires_grad"]
            and report["loss_requires_grad"]
            and report["loss_grad_fn"]
            and report["graph_reaches_turn_logps"]
            and report["one_advantage_per_trajectory"]
            and report["scalar_abs_diff"] <= 1e-4
            and report["selective_vs_explicit_max_abs"] <= 1e-3
            and report["optimizer_steps"] == 0
            and report["backward_called"] is False
        )

    ok = (
        turn_report["all_prefix_bytes_equal"]
        and turn_report["all_logprobs_within_atol"]
        and turn_report["observations_in_context_and_out_of_loss"]
        and scripted_modes["B"]["advantage_once_per_trajectory"]
        and scripted_modes["C"]["advantage_once_per_trajectory"]
        and scripted_modes["B"]["advantage_matches_formula"]
        and scripted_modes["C"]["advantage_matches_formula"]
        and scripted_modes["B"]["dapo_scalar"]["within_atol"]
        and scripted_modes["C"]["dapo_scalar"]["within_atol"]
        and _loss_ok(t0_loss)
        and _loss_ok(scripted_modes["B"]["pilot_loss"])
        and _loss_ok(scripted_modes["C"]["pilot_loss"])
        and not old_site["computes_old_per_token_logps"]
        and summary["stopped_before_backward"]
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
