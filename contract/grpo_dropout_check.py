"""Paired t0 generation: stock 5.5 train mode versus LoRA dropout off.

The eval arm is the next-pilot path: model.eval() during the rollout, then the
mode from before that rollout is restored before any loss forward. This script
stops before backward and does not run held-out.
"""

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
from contract.grpo_pair import _install_lora_state
from contract.grpo_path import _task_tuple, make_instrumented_rollout_func
from contract.grpo_prep import SEED_ORDER, _lora_dict, _write_json
from contract.grpo_train_diag import _advantages, _tool_call_text
from contract.harness import answer_matches, one_action_user_prompt
from contract.run import _seed_generation

OUT = ROOT / "artifacts" / "task-5-6" / "dropout_check"
INIT = ROOT / "artifacts" / "task-5-5" / "lora_init.pt"
COMPARISON = ROOT / "artifacts" / "task-5-6" / "t0_5_1_vs_5_5.json"
COMMAND = (
    "HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
    ".venv/bin/python run_grpo_dropout_check.py"
)
PENALTY = 0.5


def _token_hash(turn_new: list[list[int]], turn_prefix: list[list[int]]) -> str:
    payload = json.dumps({"new": turn_new, "prefix": turn_prefix}).encode()
    return hashlib.sha256(payload).hexdigest()


def _lora_dropout_modules(model) -> list[Any]:
    import torch

    found = []
    for module in model.modules():
        drop = getattr(module, "lora_dropout", None)
        if isinstance(drop, torch.nn.Dropout):
            found.append(drop)
        elif isinstance(drop, torch.nn.ModuleDict):
            found.extend(child for child in drop.values() if isinstance(child, torch.nn.Dropout))
    return found


def _load_model():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from agenttune.agentic.rollout_engines.rollout_factory import ensure_lora_targets

    torch.manual_seed(config.SEED)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(config.SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID, revision=config.MODEL_REVISION, local_files_only=True
    )
    base = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        revision=config.MODEL_REVISION,
        dtype=torch.float16,
        local_files_only=True,
    )
    revision = getattr(base.config, "_commit_hash", None) or config.MODEL_REVISION
    if revision != config.MODEL_REVISION:
        raise RuntimeError(f"loaded revision {revision} != {config.MODEL_REVISION}")
    model = get_peft_model(base, ensure_lora_targets(LoraConfig(**_lora_dict()), base))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    state = torch.load(INIT, map_location="cpu", weights_only=True)
    _install_lora_state(model, state)
    drops = _lora_dropout_modules(model)
    if not drops or any(float(module.p) != 0.05 for module in drops):
        raise RuntimeError(f"expected LoRA dropout 0.05, found {len(drops)} modules")
    return tokenizer, model, device, drops


def _centered_relation(left: list[float], right: list[float]) -> dict[str, Any]:
    import torch

    first = torch.tensor(left, dtype=torch.float64)
    second = torch.tensor(right, dtype=torch.float64)
    first = first - first.mean()
    second = second - second.mean()
    ratios = []
    for x_value, y_value in zip(first.tolist(), second.tolist()):
        if abs(x_value) <= 1e-8 and abs(y_value) <= 1e-8:
            continue
        if abs(x_value) <= 1e-8:
            return {"proportional": False, "scale": None, "ratios": ratios}
        ratios.append(y_value / x_value)
    if not ratios:
        return {"proportional": True, "scale": None, "ratios": []}
    scale = ratios[0]
    proportional = all(abs(item - scale) <= 1e-5 * max(1.0, abs(scale)) for item in ratios)
    return {"proportional": proportional, "scale": scale if proportional else None, "ratios": ratios}


def _run_one(model, tokenizer, drops, seed: int, arm: str) -> dict[str, Any]:
    import torch

    from agenttune.agentic.rollout_engines.transformers_engine import TransformersRolloutEngine

    task = _task_tuple("t0")
    prompt = one_action_user_prompt(task[1], task[2])
    rollout_eval = arm == "eval"
    model.train()
    for module in drops:
        module.train(arm != "dropout_off")
    observed: list[dict[str, Any]] = []
    original = model.generate

    def watched(*args, **kwargs):
        observed.append(
            {
                "model_training": bool(model.training),
                "dropout_training": [bool(module.training) for module in drops],
            }
        )
        return original(*args, **kwargs)

    model.generate = watched
    rollout = make_instrumented_rollout_func(
        episode_id=task[0],
        order_id=task[1],
        customer_id=task[2],
        amount_cents=task[3],
        generation_seeds=[seed],
        tokenizer=tokenizer,
        mode="live",
        live_engine_factory=lambda: TransformersRolloutEngine(model=model, tokenizer=tokenizer),
        rollout_eval=rollout_eval,
    )
    _seed_generation(seed)
    try:
        with torch.no_grad():
            merged = rollout([prompt])
    finally:
        model.generate = original
    if model.training is not True:
        raise RuntimeError(f"{arm} seed {seed} left the model out of train mode")
    if not observed:
        raise RuntimeError(f"{arm} seed {seed} did not call generate")
    if arm == "eval" and any(item["model_training"] for item in observed):
        raise RuntimeError(f"eval arm seed {seed} generated while model.training was true")
    if arm != "eval" and any(not item["model_training"] for item in observed):
        raise RuntimeError(f"{arm} seed {seed} generated outside train mode")
    if arm == "dropout_off" and any(any(item["dropout_training"]) for item in observed):
        raise RuntimeError(f"dropout-off seed {seed} still trained a LoRA dropout")
    if arm == "stock" and any(not all(item["dropout_training"]) for item in observed):
        raise RuntimeError(f"stock seed {seed} generated with a LoRA dropout disabled")
    events = merged["contract_events"][0]
    harness = events["harness_state"]
    final_answer = events["final_answer"]
    confirmed = bool(harness["task_success_harness"])
    correct = bool(answer_matches(final_answer, task[3]))
    if correct != bool(harness["task_success_answer"]):
        raise RuntimeError(f"answer flag disagrees with answer_matches for seed {seed}")
    tool_call = bool(_tool_call_text(final_answer))
    n_final_tool_call = int(confirmed and tool_call)
    if confirmed and correct and events["n_invalid"] == 0:
        failure = "none"
    elif confirmed and not correct and tool_call:
        failure = "final_answer_tool_call_after_submit"
    elif confirmed and not correct:
        failure = "final_answer_not_exact"
    elif events["n_invalid"]:
        failure = "invalid"
    elif not confirmed:
        failure = "submit_refund_not_confirmed"
    else:
        failure = "task_success_false"
    turn_new = merged["turn_new_token_ids"][0]
    turn_prefix = merged["turn_input_ids"][0]
    new_lengths = [len(turn) for turn in turn_new]
    return {
        "arm": arm,
        "seed": seed,
        "rollout_id": events["rollout_id"],
        "refund_confirmed": confirmed,
        "final_answer_correct": correct,
        "raw_final_answer": final_answer,
        "failure": failure,
        "n_invalid": int(events["n_invalid"]),
        "n_final_tool_call": n_final_tool_call,
        "B": float(bool(events["task_success"])),
        "reward_new": float(bool(events["task_success"])) - PENALTY * n_final_tool_call,
        "turn_count": len(turn_new),
        "new_token_lengths": new_lengths,
        "cache_disabled_suspected": any(length >= config.ONE_ACTION_MAX_NEW_TOKENS for length in new_lengths),
        "token_hash": _token_hash(turn_new, turn_prefix),
        "turn_new_token_ids": turn_new,
        "turn_input_ids": turn_prefix,
        "mode_after_rollout": "train" if model.training else "eval",
        "generate_model_training": [item["model_training"] for item in observed],
    }


def _group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["seed"])
    rewards_b = [row["B"] for row in ordered]
    rewards_new = [row["reward_new"] for row in ordered]
    relation = _centered_relation(rewards_b, rewards_new)
    advantages_b = _advantages(rewards_b)
    advantages_new = _advantages(rewards_new)
    per_seed = []
    for row, advantage_b, advantage_new in zip(ordered, advantages_b, advantages_new):
        per_seed.append(
            {
                "seed": row["seed"],
                "refund_confirmed": row["refund_confirmed"],
                "final_answer_correct": row["final_answer_correct"],
                "failure": row["failure"],
                "n_invalid": row["n_invalid"],
                "n_final_tool_call": row["n_final_tool_call"],
                "B": row["B"],
                "reward_new": row["reward_new"],
                "advantage_B": advantage_b,
                "advantage_new": advantage_new,
                "raw_final_answer": row["raw_final_answer"],
            }
        )
    return {
        "per_seed": per_seed,
        "B": rewards_b,
        "reward_new": rewards_new,
        "advantage_B": advantages_b,
        "advantage_new": advantages_new,
        "centered_rewards_proportional": relation["proportional"],
        "centered_scale": relation["scale"],
        "n_invalid_unchanged_definition": True,
    }


def _disappeared(saved: dict, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    previous = {item["seed"]: item for item in saved["metrics"]["task_5_1"]}
    stock_saved = {item["seed"]: item for item in saved["metrics"]["task_5_5"]}
    by_arm: dict[str, dict[int, dict]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], {})[row["seed"]] = row
    report = []
    for seed in SEED_ORDER:
        old = previous[seed]
        current_saved = stock_saved[seed]
        changed = (
            old["refund_confirmed"] != current_saved["refund_confirmed"]
            or old["final_answer_correct"] != current_saved["final_answer_correct"]
            or old["failure"] != current_saved["failure"]
            or old["raw_final_answer"] != current_saved["raw_final_answer"]
        )
        if not changed:
            continue
        item = {
            "seed": seed,
            "task_5_1": {
                "refund_confirmed": old["refund_confirmed"],
                "final_answer_correct": old["final_answer_correct"],
                "failure": old["failure"],
            },
            "saved_task_5_5": {
                "refund_confirmed": current_saved["refund_confirmed"],
                "final_answer_correct": current_saved["final_answer_correct"],
                "failure": current_saved["failure"],
            },
        }
        for arm in ("stock", "dropout_off", "eval"):
            row = by_arm[arm][seed]
            matches_5_1 = (
                row["refund_confirmed"] == old["refund_confirmed"]
                and row["final_answer_correct"] == old["final_answer_correct"]
                and row["failure"] == old["failure"]
                and row["raw_final_answer"] == old["raw_final_answer"]
            )
            item[arm] = {
                "matches_task_5_1": matches_5_1,
                "failure": row["failure"],
                "final_answer_correct": row["final_answer_correct"],
                "raw_final_answer": row["raw_final_answer"],
            }
        item["disappeared_with_dropout_off"] = item["dropout_off"]["matches_task_5_1"]
        item["disappeared_with_eval"] = item["eval"]["matches_task_5_1"]
        report.append(item)
    return report


def main() -> int:
    saved = json.loads(COMPARISON.read_text())
    saved_hashes = saved["token_hashes_sha256_prefix_and_new"]
    tokenizer, model, device, drops = _load_model()
    rows = []
    for seed in SEED_ORDER:
        for arm in ("stock", "dropout_off"):
            _seed_generation(seed)
            print(f"pair seed={seed} arm={arm}", flush=True)
            row = _run_one(model, tokenizer, drops, seed, arm)
            rows.append(row)
            print(
                f"done seed={seed} arm={arm} hash={row['token_hash'][:12]} "
                f"failure={row['failure']} turns={row['new_token_lengths']}",
                flush=True,
            )
        _seed_generation(seed)
        print(f"eval seed={seed}", flush=True)
        row = _run_one(model, tokenizer, drops, seed, "eval")
        rows.append(row)
        print(
            f"done seed={seed} arm=eval hash={row['token_hash'][:12]} "
            f"failure={row['failure']} turns={row['new_token_lengths']}",
            flush=True,
        )
    hashes = {}
    for row in rows:
        hashes.setdefault(row["arm"], {})[f"t0-g{row['seed']}"] = row["token_hash"]
    hash_match = {
        arm: {key: hashes[arm][key] == saved_hashes[key] for key in hashes[arm]}
        for arm in hashes
    }
    eval_rows = [row for row in rows if row["arm"] == "eval"]
    eval_group = _group(eval_rows)
    relative = not eval_group["centered_rewards_proportional"]
    proposal = None
    scale_only = None
    if relative:
        proposal = (
            "One paired optimizer step on this eval group only: the same initial LoRA, "
            "t0 seeds 100-103, eval() during rollout and the previous mode restored before "
            "the loss. Reward B against I(task_success) - 0.5 * n_final_tool_call, with "
            "n_final_tool_call kept out of N_invalid. Compare advantages and the LoRA update. "
            "Do not include held-out in that step."
        )
    else:
        scale_only = {
            "centered_scale": eval_group["centered_scale"],
            "note": (
                "The new reward changes the centered values by one scale. "
                "Relative advantages inside the eval group stay the same aside from the "
                "1e-4 term in the advantage denominator."
            ),
        }
    compact = []
    for row in rows:
        compact.append({key: value for key, value in row.items() if key not in {"turn_new_token_ids", "turn_input_ids"}})
    payload = {
        "kind": "t0_dropout_control",
        "optimizer_steps": 0,
        "backward_ran": False,
        "held_out_ran": False,
        "device": device,
        "lora_init": str(INIT.relative_to(ROOT)),
        "lora_dropout_modules": len(drops),
        "seed_reset": "torch.manual_seed and mps seed before each run of a pair, and again before the eval run of that seed",
        "command": COMMAND,
        "hashes": hashes,
        "saved_task_5_5_hashes": {key: saved_hashes[key] for key in saved_hashes if key.startswith("t0-g")},
        "hash_equals_saved_task_5_5": hash_match,
        "t0_g102_equals_t0_g103": {
            arm: hashes[arm]["t0-g102"] == hashes[arm]["t0-g103"] for arm in hashes
        },
        "rows": compact,
        "discrepancies_versus_5_1": _disappeared(saved, rows),
        "eval_group": eval_group,
        "relative_advantage_changes": relative,
        "scale_only": scale_only,
        "proposed_paired_experiment": proposal,
        "next_pilot_rollout": "model.eval() during the live rollout; the mode from before the trajectory is restored before the rollout returns and before the loss forward",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    _write_json(OUT / "summary.json", payload)
    _write_json(OUT / "traces.json", {"rows": rows})
    (OUT / "commands.txt").write_text(COMMAND + "\n")
    print(json.dumps({
        "hash_equals_saved_task_5_5": hash_match,
        "t0_g102_equals_t0_g103": payload["t0_g102_equals_t0_g103"],
        "eval_group": eval_group["per_seed"],
        "centered_rewards_proportional": eval_group["centered_rewards_proportional"],
        "centered_scale": eval_group["centered_scale"],
        "relative_advantage_changes": relative,
        "optimizer_steps": 0,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
