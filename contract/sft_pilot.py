"""Task 5.15: SFT pilot on confirmed post-submit finals; optional 5.14 eval.

Method: SFT. Train only on gold REFUND_SUBMITTED (+ chat <|END_RESPONSE|>) tokens
with prefix masked. Learning rate and step budget are pinned before training.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contract import config
from contract.discovery_prompts import discovery_user_prompt
from contract.grpo_pair import _release
from contract.grpo_path import _TurnIdEngine
from contract.grpo_prep import _lora_dict, _package_versions, _write_json
from contract.grpo_train_diag import _row_from_steps
from contract.grpo_train_step import _lora_grad_norm, _lora_params
from contract.harness import (
    ONE_ACTION_SYSTEM_PROMPT,
    ONE_ACTION_TOOL_RESULT_FORMAT,
    ONE_ACTION_TOOLS_FALLBACK_PROMPT,
    RefundEpisode,
    final_answer_text,
)
from contract.run import (
    _bind_output,
    _dump,
    _load_model,
    _one_model_rollout,
    _probe,
    make_guarded_tools,
)
from contract.sft_final import (
    END_RESPONSE,
    generate_on_prefix_aligned,
    score_final,
)
from contract.sft_pilot_prompts import (
    COLLECTION_SEEDS,
    DEV_TASKS,
    EVAL_514_SEEDS,
    EVAL_514_TASKS,
    TRAIN_TASKS,
    assert_pilot_split,
    pilot_user_prompt,
)

OUT = ROOT / "artifacts" / "task-5-15"
BASELINE_514 = ROOT / "artifacts" / "task-5-14" / "baseline"
COMMAND = (
    "HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
    ".venv/bin/python run_sft_pilot.py"
)
# Pinned before training (5.13: 2e-4 raised P(gold) without collapse in 3 steps on
# one example; 2e-5 is the safer default for a multi-example pilot).
LEARNING_RATE = 2e-5
MAX_OPTIMIZER_STEPS = 3
PROBE_SEED = 300
# Supervise gold line + chat end marker (present in successful 5.14 completions;
# absent from 5.11 loss and implicated in END_RESPONSE collapse).
INCLUDE_END_RESPONSE_IN_TARGET = True


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def write_pinned_config() -> dict[str, Any]:
    assert_pilot_split()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "datasets").mkdir(parents=True, exist_ok=True)
    (OUT / "adapters").mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "sft_pilot_5_15",
        "method": "SFT",
        "written_before_training": True,
        "command": COMMAND,
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "held_out_eval": {
            "source": "artifacts/task-5-14",
            "episode_ids": [t[0] for t in EVAL_514_TASKS],
            "seeds": list(EVAL_514_SEEDS),
            "n_pairs": len(EVAL_514_TASKS) * len(EVAL_514_SEEDS),
            "note": "All 12 pairs reserved for final comparison; never used in train/dev.",
        },
        "train": {
            "episode_ids": [t[0] for t in TRAIN_TASKS],
            "order_ids": [t[1] for t in TRAIN_TASKS],
            "collection_seeds": list(COLLECTION_SEEDS),
            "filter": "confirmed submit_refund only",
            "gold_amount_source": "submit_refund tool arguments/structured_result",
        },
        "dev": {
            "episode_ids": [t[0] for t in DEV_TASKS],
            "order_ids": [t[1] for t in DEV_TASKS],
            "collection_seeds": list(COLLECTION_SEEDS),
            "excluded_from_training": True,
        },
        "generation_collection": {
            "settings": "same as task-5-8 / 5.14 one-action",
            "temperature": config.ONE_ACTION_TEMPERATURE,
            "top_p": config.ONE_ACTION_TOP_P,
            "max_new_tokens": config.ONE_ACTION_MAX_NEW_TOKENS,
            "capture": "exact AgentTune final-turn input_ids via _TurnIdEngine",
        },
        "training": {
            "learning_rate": LEARNING_RATE,
            "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
            "loss": "causal_lm_on_final_tokens_only",
            "prefix_masked": True,
            "include_end_response_in_target": INCLUDE_END_RESPONSE_IN_TARGET,
            "lora": _lora_dict(),
        },
        "dev_gate": {
            "parse": "task-5-12 truncate_at_end_response",
            "stop_if_base_has_no_final_errors": True,
            "select_checkpoint": (
                "first step that fixes >=1 base-incorrect final and breaks 0 "
                "base-correct finals; also reject if <|END_RESPONSE|> becomes "
                "argmax first token on any previously correct base context"
            ),
        },
        "pilot_criteria_on_5_14": {
            "full_success": "> 7/12",
            "confirmed_returns": ">= 10/12",
            "calls_after_confirmed_submit_total": "< 3",
        },
    }
    _write_json(OUT / "pinned_config.json", payload)
    (OUT / "commands.txt").write_text(COMMAND + "\n", encoding="utf-8")
    for split, tasks in (("train", TRAIN_TASKS), ("dev", DEV_TASKS)):
        path = OUT / "datasets" / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for episode_id, order_id, customer_id, amount in tasks:
                handle.write(
                    json.dumps(
                        {
                            "episode_id": episode_id,
                            "order_id": order_id,
                            "customer_id": customer_id,
                            "amount_cents": amount,
                            "split": split,
                            "prompt": pilot_user_prompt(episode_id, order_id, customer_id),
                            "seeds": list(COLLECTION_SEEDS),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return payload


def _amount_from_submit_tool(steps: list[dict]) -> int | None:
    for step in steps:
        for call in step.get("calls") or []:
            if (call.get("tool") or call.get("name")) != "submit_refund":
                continue
            result = call.get("structured_result") or {}
            ok = (
                result.get("submitted") is True
                or (call.get("execution") == "success" and result.get("ok") is True)
            )
            if not ok:
                continue
            args = call.get("arguments") or {}
            if "amount_cents" in args:
                return int(args["amount_cents"])
            if "amount_cents" in result:
                return int(result["amount_cents"])
    return None


def _post_submit_capture(
    turn_calls: list[dict[str, Any]], steps: list[dict[str, Any]], tokenizer
) -> dict[str, Any] | None:
    candidates = []
    for step, turn in zip(steps, turn_calls):
        messages = step.get("messages") or []
        if messages and "submit_refund" in str(messages[-1].get("content") or ""):
            candidates.append((step, turn))
    if not candidates:
        return None
    step, turn = candidates[-1]
    new_ids = list(turn["new_token_ids"])
    raw = tokenizer.decode(new_ids, skip_special_tokens=False)
    return {
        "step_id": step.get("step_id"),
        "input_ids": list(turn["input_ids"]),
        "new_token_ids": new_ids,
        "generated_raw": raw,
    }


def _collect_split(
    label: str,
    tasks: tuple,
    engine,
    tokenizer,
    factory,
    schema_list,
    probe,
) -> list[dict[str, Any]]:
    out_dir = OUT / "collection" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    _bind_output(out_dir)
    turn_engine = _TurnIdEngine(engine, tokenizer)
    rows: list[dict[str, Any]] = []
    all_steps: list[dict] = []
    for episode_id, order_id, customer_id, amount in tasks:
        for seed in COLLECTION_SEEDS:
            rid = f"{episode_id}-g{seed}"
            print(f"collect {label} {rid}", flush=True)
            turn_engine.calls = []
            episode = RefundEpisode(
                episode_id=episode_id,
                order_id=order_id,
                customer_id=customer_id,
                amount_cents=amount,
                seed=config.ONE_ACTION_ENVIRONMENT_SEED,
            )
            prompt = pilot_user_prompt(episode_id, order_id, customer_id)
            ep_rows, limits = _one_model_rollout(
                factory,
                turn_engine,
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
                prompt_text=prompt,
                prompt_protocol="one_action_per_turn",
            )
            if limits:
                print("limits", limits, flush=True)
            all_steps.extend(ep_rows)
            _dump(out_dir / "model_episodes.jsonl", all_steps)
            scored = _row_from_steps(ep_rows, source=str(out_dir / "model_episodes.jsonl"))
            tool_amount = _amount_from_submit_tool(ep_rows)
            post = _post_submit_capture(turn_engine.calls, ep_rows, tokenizer)
            confirmed = bool(scored["task_success_harness"]) and tool_amount is not None
            row = {
                "rollout_id": rid,
                "episode_id": episode_id,
                "order_id": order_id,
                "customer_id": customer_id,
                "episode_amount_cents": amount,
                "tool_amount_cents": tool_amount,
                "generation_seed": seed,
                "prompt": prompt,
                "confirmed": confirmed,
                "task_success": bool(scored["task_success"]),
                "task_success_harness": bool(scored["task_success_harness"]),
                "task_success_answer": bool(scored["task_success_answer"]),
                "error_type": scored["error_type"],
                "final_answer": scored.get("final_answer"),
                "function_entries": scored.get("function_entries"),
                "post_submit_final_turn": post,
            }
            if confirmed and post is not None:
                gold = final_answer_text(tool_amount)
                target = gold + (END_RESPONSE if INCLUDE_END_RESPONSE_IN_TARGET else "")
                target_ids = tokenizer.encode(target, add_special_tokens=False)
                row["sft"] = {
                    "prefix_input_ids": post["input_ids"],
                    "target_text": target,
                    "target_ids": target_ids,
                    "gold_amount_cents": tool_amount,
                    "n_prefix": len(post["input_ids"]),
                    "n_target": len(target_ids),
                }
            rows.append(row)
            print(
                f"  confirmed={confirmed} tool_amt={tool_amount} "
                f"err={row['error_type']} post="
                f"{None if post is None else len(post['input_ids'])}",
                flush=True,
            )
    _write_json(out_dir / "rollouts.json", {"split": label, "rollouts": rows})
    return rows


def _first_token_probe(model, tokenizer, prefix_ids: list[int], amount: int) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    gold_first = tokenizer.encode(final_answer_text(amount), add_special_tokens=False)[0]
    end_id = tokenizer.encode(END_RESPONSE, add_special_tokens=False)[0]
    ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(input_ids=ids).logits[0, -1].float(), dim=-1)
    p_gold = float(probs[gold_first].item())
    p_end = float(probs[end_id].item())
    argmax_id = int(torch.argmax(probs).item())
    return {
        "p_gold_first": p_gold,
        "p_end_response": p_end,
        "argmax_token_id": argmax_id,
        "argmax_token": tokenizer.decode([argmax_id]),
        "end_response_is_first": argmax_id == end_id,
        "prefer_end_over_gold": p_end >= p_gold,
    }


def _gen_final(model, tokenizer, prefix_ids: list[int], *, seed: int) -> dict[str, Any]:
    return generate_on_prefix_aligned(
        model,
        tokenizer,
        prefix_ids=prefix_ids,
        seed=seed,
        max_new_tokens=128,
        temperature=config.ONE_ACTION_TEMPERATURE,
        top_p=config.ONE_ACTION_TOP_P,
    )


def evaluate_on_dev_contexts(
    model, tokenizer, contexts: list[dict[str, Any]], *, label: str
) -> dict[str, Any]:
    pairs = []
    for i, ctx in enumerate(contexts):
        seed = PROBE_SEED + i
        gen = _gen_final(model, tokenizer, ctx["prefix_input_ids"], seed=seed)
        scored = score_final(gen["raw"], ctx["gold_amount_cents"])
        first = _first_token_probe(
            model, tokenizer, ctx["prefix_input_ids"], ctx["gold_amount_cents"]
        )
        pairs.append(
            {
                "rollout_id": ctx["rollout_id"],
                "episode_id": ctx["episode_id"],
                "amount_cents": ctx["gold_amount_cents"],
                "seed": seed,
                "generated_raw": gen["raw"],
                "parsed": scored["parsed"],
                "correct": scored["correct"],
                "empty": scored["empty"],
                "starts_with_end_response": scored["starts_with_end_response"],
                "looks_like_tool_call": scored["looks_like_tool_call"],
                "first_token": first,
            }
        )
        print(
            f"dev[{label}] {ctx['rollout_id']} correct={scored['correct']} "
            f"empty={scored['empty']} end_first={first['end_response_is_first']} "
            f"parsed={scored['parsed'][:60]!r}",
            flush=True,
        )
    n_correct = sum(1 for p in pairs if p["correct"])
    n_err = len(pairs) - n_correct
    return {
        "label": label,
        "n": len(pairs),
        "n_correct": n_correct,
        "n_final_errors": n_err,
        "n_end_response_first": sum(
            1 for p in pairs if p["first_token"]["end_response_is_first"]
        ),
        "pairs": pairs,
    }


def _build_train_tensors(tokenizer, train_ctx: list[dict[str, Any]]) -> list[dict[str, Any]]:
    encoded = []
    for ctx in train_ctx:
        prefix = list(ctx["prefix_input_ids"])
        target = list(ctx["target_ids"])
        labels = [-100] * len(prefix) + target
        input_ids = prefix + target
        if any(x != -100 for x in labels[: len(prefix)]):
            raise RuntimeError("prefix not masked")
        if not target:
            raise RuntimeError("empty target")
        encoded.append(
            {
                "rollout_id": ctx["rollout_id"],
                "input_ids": input_ids,
                "labels": labels,
                "n_prefix": len(prefix),
                "n_target": len(target),
            }
        )
    return encoded


def train_with_checkpoints(
    tokenizer,
    train_ctx: list[dict[str, Any]],
    dev_ctx: list[dict[str, Any]],
    base_dev: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from agenttune.agentic.rollout_engines.rollout_factory import ensure_lora_targets

    encoded = _build_train_tensors(tokenizer, train_ctx)
    _write_json(
        OUT / "train_encoding.json",
        {
            "n": len(encoded),
            "include_end_response": INCLUDE_END_RESPONSE_IN_TARGET,
            "examples": [
                {
                    "rollout_id": e["rollout_id"],
                    "n_prefix": e["n_prefix"],
                    "n_target": e["n_target"],
                    "target_text": train_ctx[i]["target_text"],
                }
                for i, e in enumerate(encoded)
            ],
        },
    )

    torch.manual_seed(config.SEED)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(config.SEED)
    base = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        revision=config.MODEL_REVISION,
        dtype=torch.float16,
        local_files_only=True,
    )
    model = get_peft_model(base, ensure_lora_targets(LoraConfig(**_lora_dict()), base))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)

    base_correct_ids = {
        p["rollout_id"] for p in base_dev["pairs"] if p["correct"]
    }
    base_incorrect_ids = {
        p["rollout_id"] for p in base_dev["pairs"] if not p["correct"]
    }

    history = []
    selected = None
    for step in range(1, MAX_OPTIMIZER_STEPS + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        n_sup = 0
        for ex in encoded:
            ids = torch.tensor([ex["input_ids"]], dtype=torch.long, device=device)
            lab = torch.tensor([ex["labels"]], dtype=torch.long, device=device)
            out = model(input_ids=ids, labels=lab)
            loss = out.loss / len(encoded)
            loss.backward()
            total_loss += float(out.loss.detach().item())
            n_sup += ex["n_target"]
        # Grad finite check.
        for name, param in _lora_params(model):
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise RuntimeError(f"non-finite grad in {name} at step {step}")
        grad_norm = float(_lora_grad_norm(model))
        opt.step()

        adapter_dir = OUT / "adapters" / f"step_{step}"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)

        dev = evaluate_on_dev_contexts(model, tokenizer, dev_ctx, label=f"step_{step}")
        _write_json(OUT / f"dev_step_{step}.json", dev)

        by_id = {p["rollout_id"]: p for p in dev["pairs"]}
        fixed = [
            rid
            for rid in base_incorrect_ids
            if by_id[rid]["correct"]
        ]
        broken = [
            rid
            for rid in base_correct_ids
            if not by_id[rid]["correct"]
        ]
        end_first_on_base_correct = [
            rid
            for rid in base_correct_ids
            if by_id[rid]["first_token"]["end_response_is_first"]
        ]
        gate_pass = (
            len(fixed) >= 1
            and len(broken) == 0
            and len(end_first_on_base_correct) == 0
        )
        record = {
            "step": step,
            "mean_train_loss": total_loss / max(1, len(encoded)),
            "grad_norm": grad_norm,
            "n_supervised_tokens_per_epoch": n_sup,
            "adapter": str(adapter_dir.relative_to(ROOT)),
            "dev_n_correct": dev["n_correct"],
            "dev_n_final_errors": dev["n_final_errors"],
            "fixed_ids": fixed,
            "broken_ids": broken,
            "end_response_first_on_base_correct": end_first_on_base_correct,
            "gate_pass": gate_pass,
        }
        history.append(record)
        print(
            f"sft step={step} loss={record['mean_train_loss']:.6f} "
            f"fixed={fixed} broken={broken} end_first={end_first_on_base_correct} "
            f"gate={gate_pass}",
            flush=True,
        )
        if gate_pass and selected is None:
            selected = record
            break

    return {
        "learning_rate": LEARNING_RATE,
        "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
        "history": history,
        "selected": selected,
        "stop_reason": (
            "checkpoint_selected"
            if selected is not None
            else "no_checkpoint_passed_dev_gate"
        ),
    }


def _calls_after_submit(steps: list[dict]) -> int:
    confirmed = False
    later = 0
    for step in steps:
        for call in step.get("calls") or []:
            name = call.get("tool") or ""
            result = call.get("structured_result") or {}
            if confirmed and name:
                later += 1
            if name == "submit_refund" and (
                result.get("submitted") is True
                or (call.get("execution") == "success" and result.get("ok") is True)
            ):
                confirmed = True
    last = steps[-1]
    raw = last.get("raw_output") or last.get("final_answer")
    if (
        last.get("task_success_harness")
        and not last.get("task_success_answer")
        and score_final(raw, int(last.get("amount_cents") or 0))["looks_like_tool_call"]
    ):
        # Match 5.9 normalization: tool-call final after confirm counts as ≥1.
        later = max(later, 1)
    return later


def run_full_514_eval(adapter_dir: Path) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import agenttune.agentic.rollout_engines.rollout_factory as factory
    from agenttune.agentic.rollout_engines.transformers_engine import TransformersRolloutEngine

    out = OUT / "eval_514"
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID, revision=config.MODEL_REVISION, local_files_only=True
    )
    base = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        revision=config.MODEL_REVISION,
        dtype=torch.float16,
        local_files_only=True,
    )
    adapted = PeftModel.from_pretrained(base, str(adapter_dir))
    adapted.eval()
    if torch.backends.mps.is_available():
        adapted.to("mps")
    engine = TransformersRolloutEngine(adapted, tokenizer)
    _guards, schema_list = make_guarded_tools(
        RefundEpisode("schema-probe", "ORD-9401", "CUS-G01", 1620, config.SEED)
    )
    probe = _probe(tokenizer, schema_list)
    _bind_output(out)
    all_steps: list[dict] = []
    pairs = []
    for episode_id, order_id, customer_id, amount in EVAL_514_TASKS:
        for seed in EVAL_514_SEEDS:
            rid = f"{episode_id}-g{seed}"
            print(f"eval514 adapter {rid}", flush=True)
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
                prompt_text=discovery_user_prompt(episode_id, order_id, customer_id),
                prompt_protocol="one_action_per_turn",
            )
            if limits:
                print("limits", limits, flush=True)
            for step in ep_rows:
                step["amount_cents"] = amount
            all_steps.extend(ep_rows)
            _dump(out / "model_episodes.jsonl", all_steps)
            scored = _row_from_steps(ep_rows, source=str(out / "model_episodes.jsonl"))
            after = _calls_after_submit(ep_rows)
            pairs.append(
                {
                    "rollout_id": rid,
                    "episode_id": episode_id,
                    "generation_seed": seed,
                    "amount_cents": amount,
                    "task_success": bool(scored["task_success"]),
                    "confirmed": bool(scored["task_success_harness"]),
                    "correct_final": bool(scored["task_success_answer"]),
                    "error_type": scored["error_type"],
                    "n_calls_after_confirmed_submit": after,
                    "final_answer": scored.get("final_answer"),
                }
            )

    base_pairs = json.loads((BASELINE_514 / "per_pair.json").read_text(encoding="utf-8"))[
        "pairs"
    ]
    base_by = {p["rollout_id"]: p for p in base_pairs}
    transitions = []
    for p in pairs:
        b = base_by[p["rollout_id"]]
        base_after = len(b.get("calls_after_confirmed_submit") or [])
        # Normalize tool-call-after-submit like 5.9.
        if (
            b.get("task_success_harness")
            and not b.get("task_success_answer")
            and b.get("error_type") == "final_answer_tool_call_after_submit"
        ):
            base_after = max(base_after, 1)
        transitions.append(
            {
                "rollout_id": p["rollout_id"],
                "base": {
                    "task_success": bool(b.get("task_success")),
                    "confirmed": bool(b.get("task_success_harness")),
                    "correct_final": bool(b.get("task_success_answer")),
                    "error_type": b.get("error_type"),
                    "n_calls_after_confirmed_submit": base_after,
                },
                "adapter": {
                    "task_success": p["task_success"],
                    "confirmed": p["confirmed"],
                    "correct_final": p["correct_final"],
                    "error_type": p["error_type"],
                    "n_calls_after_confirmed_submit": p["n_calls_after_confirmed_submit"],
                },
                "changed": {
                    "task_success": bool(b.get("task_success")) != p["task_success"],
                    "confirmed": bool(b.get("task_success_harness")) != p["confirmed"],
                    "correct_final": bool(b.get("task_success_answer"))
                    != p["correct_final"],
                    "n_calls_after_confirmed_submit": base_after
                    != p["n_calls_after_confirmed_submit"],
                },
            }
        )

    agg = {
        "n": len(pairs),
        "full_success": sum(1 for p in pairs if p["task_success"]),
        "confirmed_returns": sum(1 for p in pairs if p["confirmed"]),
        "calls_after_confirmed_submit_total": sum(
            int(p["n_calls_after_confirmed_submit"]) for p in pairs
        ),
    }
    criteria = {
        "full_success_gt_7": agg["full_success"] > 7,
        "confirmed_ge_10": agg["confirmed_returns"] >= 10,
        "calls_after_lt_3": agg["calls_after_confirmed_submit_total"] < 3,
    }
    criteria["pilot_passed"] = all(criteria.values())
    comparison = {
        "method": "SFT",
        "adapter": str(adapter_dir.relative_to(ROOT)),
        "base_source": "artifacts/task-5-14/baseline/per_pair.json",
        "adapter_aggregates": agg,
        "base_aggregates": {
            "n": 12,
            "full_success": 7,
            "confirmed_returns": 10,
            "calls_after_confirmed_submit_total": 3,
        },
        "pilot_criteria": criteria,
        "transitions": transitions,
        "changed_rollouts": [t for t in transitions if any(t["changed"].values())],
        "adapter_pairs": pairs,
    }
    _write_json(out / "comparison.json", comparison)
    _write_json(out / "adapter_pairs.json", {"pairs": pairs})
    return comparison


def _contexts_from_rollouts(rollouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rollouts:
        if not row.get("confirmed") or not row.get("sft"):
            continue
        sft = row["sft"]
        out.append(
            {
                "rollout_id": row["rollout_id"],
                "episode_id": row["episode_id"],
                "prefix_input_ids": sft["prefix_input_ids"],
                "target_ids": sft["target_ids"],
                "target_text": sft["target_text"],
                "gold_amount_cents": sft["gold_amount_cents"],
            }
        )
    return out


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import agenttune.agentic.rollout_engines.rollout_factory as factory

    pinned = write_pinned_config()
    print("5.15 pinned train/dev; 5.14 held out", flush=True)

    engine, tokenizer, loaded = _load_model()
    _guards, schema_list = make_guarded_tools(
        RefundEpisode("schema-probe", "ORD-9501", "CUS-H01", 1730, config.SEED)
    )
    probe = _probe(tokenizer, schema_list)

    train_rollouts = _collect_split(
        "train", TRAIN_TASKS, engine, tokenizer, factory, schema_list, probe
    )
    dev_rollouts = _collect_split(
        "dev", DEV_TASKS, engine, tokenizer, factory, schema_list, probe
    )
    train_ctx = _contexts_from_rollouts(train_rollouts)
    dev_ctx = _contexts_from_rollouts(dev_rollouts)
    _write_json(
        OUT / "collection_summary.json",
        {
            "train_rollouts": len(train_rollouts),
            "train_confirmed_for_sft": len(train_ctx),
            "dev_rollouts": len(dev_rollouts),
            "dev_confirmed_for_probe": len(dev_ctx),
            "train_ids": [c["rollout_id"] for c in train_ctx],
            "dev_ids": [c["rollout_id"] for c in dev_ctx],
        },
    )
    if not train_ctx:
        raise RuntimeError("no confirmed train contexts for SFT")
    if not dev_ctx:
        raise RuntimeError("no confirmed dev contexts for probe")

    # Free collection engine before probe/train reloads.
    del engine
    _release()

    print("5.15 base probe on confirmed dev post-submit prefixes", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        revision=config.MODEL_REVISION,
        dtype=torch.float16,
        local_files_only=True,
    )
    base.eval()
    if torch.backends.mps.is_available():
        base.to("mps")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_dev = evaluate_on_dev_contexts(base, tokenizer, dev_ctx, label="base")
    _write_json(OUT / "dev_base.json", base_dev)
    del base
    _release()

    if base_dev["n_final_errors"] == 0:
        summary = {
            "kind": "sft_pilot_5_15",
            "method": "SFT",
            "stop_reason": "base_dev_has_no_final_errors",
            "command": COMMAND,
            "git_head": _git_head(),
            "versions": _package_versions(),
            "loaded_model": loaded,
            "pinned_config": pinned,
            "collection": {
                "train_confirmed": len(train_ctx),
                "dev_confirmed": len(dev_ctx),
            },
            "dev_base": {
                "n": base_dev["n"],
                "n_correct": base_dev["n_correct"],
                "n_final_errors": base_dev["n_final_errors"],
            },
            "training": None,
            "eval_514": None,
        }
        _write_json(OUT / "summary.json", summary)
        _write_report(summary)
        print(json.dumps({"stop_reason": summary["stop_reason"]}, indent=2), flush=True)
        return 0

    print(
        f"5.15 base has {base_dev['n_final_errors']} final errors on dev; training SFT "
        f"lr={LEARNING_RATE} max_steps={MAX_OPTIMIZER_STEPS}",
        flush=True,
    )
    train_info = train_with_checkpoints(tokenizer, train_ctx, dev_ctx, base_dev)
    _write_json(OUT / "train_sft.json", train_info)

    full = None
    stop_reason = train_info["stop_reason"]
    if train_info["selected"] is not None:
        adapter = ROOT / train_info["selected"]["adapter"]
        print(f"5.15 selected {adapter}; running full 5.14 eval", flush=True)
        full = run_full_514_eval(adapter)
        stop_reason = (
            "pilot_passed"
            if full["pilot_criteria"]["pilot_passed"]
            else "pilot_failed_criteria"
        )
    else:
        print("5.15 no checkpoint passed dev gate; skip 5.14 eval", flush=True)

    summary = {
        "kind": "sft_pilot_5_15",
        "method": "SFT",
        "stop_reason": stop_reason,
        "command": COMMAND,
        "git_head": _git_head(),
        "python": platform.python_version(),
        "versions": _package_versions(),
        "loaded_model": loaded,
        "pinned_config": pinned,
        "collection": {
            "train_confirmed": len(train_ctx),
            "dev_confirmed": len(dev_ctx),
            "train_ids": [c["rollout_id"] for c in train_ctx],
            "dev_ids": [c["rollout_id"] for c in dev_ctx],
        },
        "dev_base": {
            "n": base_dev["n"],
            "n_correct": base_dev["n_correct"],
            "n_final_errors": base_dev["n_final_errors"],
            "error_ids": [p["rollout_id"] for p in base_dev["pairs"] if not p["correct"]],
        },
        "training": train_info,
        "eval_514": None
        if full is None
        else {
            "adapter_aggregates": full["adapter_aggregates"],
            "pilot_criteria": full["pilot_criteria"],
            "n_changed": len(full["changed_rollouts"]),
            "transitions": full["transitions"],
        },
    }
    _write_json(OUT / "summary.json", summary)
    _write_report(summary)
    print(
        json.dumps(
            {
                "stop_reason": stop_reason,
                "selected": train_info.get("selected"),
                "eval_514": None
                if full is None
                else {
                    "aggregates": full["adapter_aggregates"],
                    "criteria": full["pilot_criteria"],
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def _write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Task 5.15 — SFT pilot on confirmed finals",
        "",
        "**Method: SFT.** Prefix-masked loss on gold `REFUND_SUBMITTED` "
        f"{'(+ <|END_RESPONSE|>)' if INCLUDE_END_RESPONSE_IN_TARGET else ''}.",
        "",
        f"- stop_reason: `{summary['stop_reason']}`",
        f"- train confirmed contexts: {summary['collection']['train_confirmed']}",
        f"- dev confirmed contexts: {summary['collection']['dev_confirmed']}",
        f"- base dev final errors: {summary['dev_base']['n_final_errors']}/"
        f"{summary['dev_base']['n']}",
        "",
    ]
    train = summary.get("training")
    if train:
        lines.append("## Training")
        lines.append("")
        lines.append(
            f"- lr={train['learning_rate']} max_steps={train['max_optimizer_steps']}"
        )
        lines.append(f"- selected: {train.get('selected')}")
        lines.append("")
        for h in train.get("history") or []:
            lines.append(
                f"- step {h['step']}: fixed={h['fixed_ids']} broken={h['broken_ids']} "
                f"gate={h['gate_pass']}"
            )
        lines.append("")
    full = summary.get("eval_514")
    if full:
        lines.append("## 5.14 full episodes (selected adapter)")
        lines.append("")
        lines.append(f"- aggregates: {full['adapter_aggregates']}")
        lines.append(f"- criteria: {full['pilot_criteria']}")
        lines.append("")
        lines.append(
            "| rollout | base_success | ad_success | base_conf | ad_conf | base_after | ad_after |"
        )
        lines.append("|---|---|---|---|---|---:|---:|")
        for t in full.get("transitions") or []:
            a, b = t["base"], t["adapter"]
            lines.append(
                f"| {t['rollout_id']} | {a['task_success']} | {b['task_success']} | "
                f"{a['confirmed']} | {b['confirmed']} | "
                f"{a['n_calls_after_confirmed_submit']} | "
                f"{b['n_calls_after_confirmed_submit']} |"
            )
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
