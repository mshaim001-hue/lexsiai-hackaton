"""Task 5.16: SFT on 5.15 confirmed train finals; quick-check on 5.14 post-submit.

Method: SFT. Fresh LoRA, loss only on gold REFUND_SUBMITTED (+ <|END_RESPONSE|>)
tokens. Quick probe uses the 10 confirmed 5.14 post-submit input_ids; at most one
checkpoint gets a full 5.14 episode eval.
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
    _one_model_rollout,
    _probe,
    make_guarded_tools,
)
from contract.sft_final import (
    END_RESPONSE,
    generate_on_prefix_aligned,
    score_final,
)
from contract.sft_pilot import _calls_after_submit
from contract.sft_pilot_prompts import EVAL_514_SEEDS, EVAL_514_TASKS

OUT = ROOT / "artifacts" / "task-5-16"
SRC_515 = ROOT / "artifacts" / "task-5-15"
BASELINE_514 = ROOT / "artifacts" / "task-5-14" / "baseline"
COMMAND = (
    "HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
    ".venv/bin/python run_sft_pilot_516.py"
)

# Pinned before training.
LEARNING_RATE = 2e-5
MAX_OPTIMIZER_STEPS = 3
INCLUDE_END_RESPONSE_IN_TARGET = True

# Fixed train order from 5.15 confirmed collection (do not reshuffle).
TRAIN_ORDER_515 = (
    "t15_0-g100",
    "t15_1-g100",
    "t15_3-g100",
    "t15_4-g100",
    "t15_5-g100",
    "t15_7-g100",
)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_train_contexts() -> list[dict[str, Any]]:
    payload = json.loads(
        (SRC_515 / "collection" / "train" / "rollouts.json").read_text(encoding="utf-8")
    )
    by_id = {row["rollout_id"]: row for row in payload["rollouts"]}
    contexts = []
    for rid in TRAIN_ORDER_515:
        row = by_id[rid]
        if not row.get("confirmed") or not row.get("sft"):
            raise RuntimeError(f"expected confirmed SFT context for {rid}")
        sft = row["sft"]
        contexts.append(
            {
                "rollout_id": rid,
                "episode_id": row["episode_id"],
                "prefix_input_ids": list(sft["prefix_input_ids"]),
                "target_ids": list(sft["target_ids"]),
                "target_text": sft["target_text"],
                "gold_amount_cents": int(sft["gold_amount_cents"]),
                "source": "artifacts/task-5-15/collection/train/rollouts.json",
            }
        )
    if len(contexts) != 6:
        raise RuntimeError(f"expected 6 train contexts, got {len(contexts)}")
    return contexts


def load_probe_514_contexts() -> list[dict[str, Any]]:
    turns = json.loads(
        (BASELINE_514 / "post_submit_final_turns.json").read_text(encoding="utf-8")
    )["turns"]
    pairs = {
        p["rollout_id"]: p
        for p in json.loads((BASELINE_514 / "per_pair.json").read_text(encoding="utf-8"))[
            "pairs"
        ]
    }
    contexts = []
    for turn in turns:
        if not turn.get("confirmed"):
            continue
        rid = turn["rollout_id"]
        pair = pairs[rid]
        contexts.append(
            {
                "rollout_id": rid,
                "episode_id": turn["episode_id"],
                "generation_seed": int(turn["generation_seed"]),
                "prefix_input_ids": list(turn["input_ids"]),
                "gold_amount_cents": int(turn["amount_cents"]),
                "episode_error_type": pair.get("error_type"),
                "episode_task_success": bool(pair.get("task_success")),
                "episode_correct_final": bool(pair.get("task_success_answer")),
                "source": "artifacts/task-5-14/baseline/post_submit_final_turns.json",
            }
        )
    if len(contexts) != 10:
        raise RuntimeError(f"expected 10 confirmed 5.14 contexts, got {len(contexts)}")
    return contexts


def write_pinned_config(train_ctx: list[dict[str, Any]], probe_ctx: list[dict[str, Any]]) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "adapters").mkdir(parents=True, exist_ok=True)
    (OUT / "datasets").mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "sft_pilot_5_16",
        "method": "SFT",
        "written_before_training": True,
        "command": COMMAND,
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "train": {
            "source": "artifacts/task-5-15 confirmed train chains",
            "n": len(train_ctx),
            "example_order": [c["rollout_id"] for c in train_ctx],
            "gold_amount_source": "submit_refund tool result (captured in 5.15)",
            "include_end_response_in_target": INCLUDE_END_RESPONSE_IN_TARGET,
        },
        "quick_check": {
            "source": "artifacts/task-5-14 confirmed post-submit input_ids",
            "n": len(probe_ctx),
            "rollout_ids": [c["rollout_id"] for c in probe_ctx],
            "parse": "task-5-12 truncate_at_end_response",
            "compare_to": "fresh base regeneration on same input_ids and seeds",
            "if_old_errors_not_reproduced": (
                "use contexts for regression only; measure effect in full 5.14"
            ),
        },
        "training": {
            "learning_rate": LEARNING_RATE,
            "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
            "loss": "causal_lm_on_final_tokens_only",
            "prefix_masked": True,
            "fresh_lora": True,
            "lora": _lora_dict(),
            "stop_if_empty_finals": True,
        },
        "checkpoint_selection": {
            "at_most_one_full_514_eval": True,
            "if_base_errors_reproduced": (
                "first checkpoint that fixes >=1 base-incorrect and keeps all "
                "base-correct"
            ),
            "if_no_base_errors": (
                "last checkpoint that keeps all base-correct answers"
            ),
        },
        "pilot_criteria_on_5_14": {
            "baseline": {
                "full_success": 7,
                "confirmed_returns": 10,
                "calls_after_confirmed_submit_total": 3,
            },
            "full_success": "> 7/12",
            "confirmed_returns": ">= 10/12",
            "calls_after_confirmed_submit_total": "< 3",
        },
    }
    _write_json(OUT / "pinned_config.json", payload)
    (OUT / "commands.txt").write_text(COMMAND + "\n", encoding="utf-8")
    with (OUT / "datasets" / "train_order.jsonl").open("w", encoding="utf-8") as handle:
        for i, ctx in enumerate(train_ctx):
            handle.write(
                json.dumps(
                    {
                        "order": i,
                        "rollout_id": ctx["rollout_id"],
                        "episode_id": ctx["episode_id"],
                        "gold_amount_cents": ctx["gold_amount_cents"],
                        "n_prefix": len(ctx["prefix_input_ids"]),
                        "n_target": len(ctx["target_ids"]),
                        "target_text": ctx["target_text"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    with (OUT / "datasets" / "probe_514.jsonl").open("w", encoding="utf-8") as handle:
        for ctx in probe_ctx:
            handle.write(
                json.dumps(
                    {
                        "rollout_id": ctx["rollout_id"],
                        "episode_id": ctx["episode_id"],
                        "generation_seed": ctx["generation_seed"],
                        "gold_amount_cents": ctx["gold_amount_cents"],
                        "n_prefix": len(ctx["prefix_input_ids"]),
                        "episode_error_type": ctx["episode_error_type"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return payload


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


def evaluate_probe_contexts(
    model, tokenizer, contexts: list[dict[str, Any]], *, label: str
) -> dict[str, Any]:
    pairs = []
    for ctx in contexts:
        seed = int(ctx["generation_seed"])
        gen = _gen_final(model, tokenizer, ctx["prefix_input_ids"], seed=seed)
        scored = score_final(gen["raw"], ctx["gold_amount_cents"])
        first = _first_token_probe(
            model, tokenizer, ctx["prefix_input_ids"], ctx["gold_amount_cents"]
        )
        pairs.append(
            {
                "rollout_id": ctx["rollout_id"],
                "episode_id": ctx["episode_id"],
                "generation_seed": seed,
                "amount_cents": ctx["gold_amount_cents"],
                "episode_error_type": ctx.get("episode_error_type"),
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
            f"probe[{label}] {ctx['rollout_id']} seed={seed} correct={scored['correct']} "
            f"empty={scored['empty']} toolish={scored['looks_like_tool_call']} "
            f"parsed={scored['parsed'][:60]!r}",
            flush=True,
        )
    n_correct = sum(1 for p in pairs if p["correct"])
    n_empty = sum(1 for p in pairs if p["empty"])
    return {
        "label": label,
        "n": len(pairs),
        "n_correct": n_correct,
        "n_final_errors": len(pairs) - n_correct,
        "n_empty": n_empty,
        "error_ids": [p["rollout_id"] for p in pairs if not p["correct"]],
        "empty_ids": [p["rollout_id"] for p in pairs if p["empty"]],
        "pairs": pairs,
    }


def _build_train_tensors(train_ctx: list[dict[str, Any]]) -> list[dict[str, Any]]:
    encoded = []
    for ctx in train_ctx:
        prefix = list(ctx["prefix_input_ids"])
        target = list(ctx["target_ids"])
        labels = [-100] * len(prefix) + target
        if any(x != -100 for x in labels[: len(prefix)]):
            raise RuntimeError("prefix not masked")
        if not target:
            raise RuntimeError("empty target")
        encoded.append(
            {
                "rollout_id": ctx["rollout_id"],
                "input_ids": prefix + target,
                "labels": labels,
                "n_prefix": len(prefix),
                "n_target": len(target),
            }
        )
    return encoded


def train_with_checkpoints(
    tokenizer,
    train_ctx: list[dict[str, Any]],
    probe_ctx: list[dict[str, Any]],
    base_probe: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from agenttune.agentic.rollout_engines.rollout_factory import ensure_lora_targets

    encoded = _build_train_tensors(train_ctx)
    _write_json(
        OUT / "train_encoding.json",
        {
            "n": len(encoded),
            "example_order": [e["rollout_id"] for e in encoded],
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

    base_correct_ids = {p["rollout_id"] for p in base_probe["pairs"] if p["correct"]}
    base_incorrect_ids = {p["rollout_id"] for p in base_probe["pairs"] if not p["correct"]}
    errors_reproduced = len(base_incorrect_ids) > 0

    history: list[dict[str, Any]] = []
    stop_reason = "max_steps_reached"
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
        for name, param in _lora_params(model):
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise RuntimeError(f"non-finite grad in {name} at step {step}")
        grad_norm = float(_lora_grad_norm(model))
        opt.step()

        adapter_dir = OUT / "adapters" / f"step_{step}"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)

        probe = evaluate_probe_contexts(
            model, tokenizer, probe_ctx, label=f"step_{step}"
        )
        _write_json(OUT / f"probe_step_{step}.json", probe)

        by_id = {p["rollout_id"]: p for p in probe["pairs"]}
        fixed = [rid for rid in base_incorrect_ids if by_id[rid]["correct"]]
        broken = [rid for rid in base_correct_ids if not by_id[rid]["correct"]]
        empty_ids = probe["empty_ids"]
        record = {
            "step": step,
            "mean_train_loss": total_loss / max(1, len(encoded)),
            "grad_norm": grad_norm,
            "n_supervised_tokens_per_epoch": n_sup,
            "adapter": str(adapter_dir.relative_to(ROOT)),
            "probe_n_correct": probe["n_correct"],
            "probe_n_final_errors": probe["n_final_errors"],
            "probe_n_empty": probe["n_empty"],
            "fixed_ids": fixed,
            "broken_ids": broken,
            "empty_ids": empty_ids,
            "keeps_all_base_correct": len(broken) == 0,
            "fixes_at_least_one": len(fixed) >= 1,
        }
        history.append(record)
        print(
            f"sft step={step} loss={record['mean_train_loss']:.6f} "
            f"fixed={fixed} broken={broken} empty={empty_ids}",
            flush=True,
        )
        if empty_ids:
            stop_reason = "empty_finals"
            print("5.16 stop: adapter produced empty finals", flush=True)
            break

    selected = None
    selection_mode = "errors_reproduced" if errors_reproduced else "no_base_errors"
    if errors_reproduced:
        for record in history:
            if record["empty_ids"]:
                continue
            if record["fixes_at_least_one"] and record["keeps_all_base_correct"]:
                selected = record
                break
    else:
        eligible = [
            r
            for r in history
            if (not r["empty_ids"]) and r["keeps_all_base_correct"]
        ]
        if eligible:
            selected = eligible[-1]

    return {
        "learning_rate": LEARNING_RATE,
        "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
        "example_order": [c["rollout_id"] for c in train_ctx],
        "errors_reproduced_on_base_regen": errors_reproduced,
        "selection_mode": selection_mode,
        "history": history,
        "selected": selected,
        "train_stop_reason": stop_reason,
        "stop_reason": (
            "checkpoint_selected"
            if selected is not None
            else "no_checkpoint_passed_selection"
        ),
    }


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


def _write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Task 5.16 — SFT on 5.15 confirmed finals + 5.14 quick check",
        "",
        "**Method: SFT.** Fresh LoRA; prefix-masked loss on gold `REFUND_SUBMITTED` "
        f"{'(+ <|END_RESPONSE|>)' if INCLUDE_END_RESPONSE_IN_TARGET else ''}.",
        "",
        f"- stop_reason: `{summary['stop_reason']}`",
        f"- lr / max steps: `{LEARNING_RATE}` / `{MAX_OPTIMIZER_STEPS}`",
        f"- train order: `{summary['train']['example_order']}`",
        f"- base probe (10 confirmed 5.14 prefixes): "
        f"{summary['base_probe']['n_correct']}/{summary['base_probe']['n']} correct; "
        f"errors_reproduced={summary['base_probe']['errors_reproduced']}",
        "",
    ]
    train = summary.get("training")
    if train:
        lines.append("## Training")
        lines.append("")
        lines.append(f"- selection_mode: `{train['selection_mode']}`")
        lines.append(f"- train_stop_reason: `{train['train_stop_reason']}`")
        lines.append(f"- selected: `{train.get('selected')}`")
        lines.append("")
        for h in train.get("history") or []:
            lines.append(
                f"- step {h['step']}: loss={h['mean_train_loss']:.6f} "
                f"correct={h['probe_n_correct']}/10 fixed={h['fixed_ids']} "
                f"broken={h['broken_ids']} empty={h['empty_ids']}"
            )
        lines.append("")
    full = summary.get("eval_514")
    if full:
        lines.append("## 5.14 full episodes (selected adapter)")
        lines.append("")
        lines.append(f"- aggregates: `{full['adapter_aggregates']}`")
        lines.append(f"- baseline: `{full['base_aggregates']}`")
        lines.append(f"- criteria: `{full['pilot_criteria']}`")
        lines.append("")
        lines.append(
            "| rollout | base_ok | ad_ok | base_conf | ad_conf | base_after | ad_after | base_err | ad_err |"
        )
        lines.append("|---|---|---|---|---|---:|---:|---|---|")
        for t in full.get("transitions") or []:
            a, b = t["base"], t["adapter"]
            lines.append(
                f"| {t['rollout_id']} | {a['task_success']} | {b['task_success']} | "
                f"{a['confirmed']} | {b['confirmed']} | "
                f"{a['n_calls_after_confirmed_submit']} | "
                f"{b['n_calls_after_confirmed_submit']} | "
                f"{a['error_type']} | {b['error_type']} |"
            )
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    train_ctx = load_train_contexts()
    probe_ctx = load_probe_514_contexts()
    pinned = write_pinned_config(train_ctx, probe_ctx)
    print(
        f"5.16 pinned lr={LEARNING_RATE} steps<={MAX_OPTIMIZER_STEPS} "
        f"train_order={ [c['rollout_id'] for c in train_ctx] }",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID, revision=config.MODEL_REVISION, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("5.16 base regen on 10 confirmed 5.14 post-submit prefixes", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        revision=config.MODEL_REVISION,
        dtype=torch.float16,
        local_files_only=True,
    )
    base.eval()
    if torch.backends.mps.is_available():
        base.to("mps")
    base_probe = evaluate_probe_contexts(base, tokenizer, probe_ctx, label="base")
    # Annotate whether episode-level final failures reappear under isolated gen.
    episode_final_fail_ids = [
        c["rollout_id"]
        for c in probe_ctx
        if c.get("episode_error_type") == "final_answer_tool_call_after_submit"
    ]
    base_by = {p["rollout_id"]: p for p in base_probe["pairs"]}
    reproduced_episode_fails = [
        rid for rid in episode_final_fail_ids if not base_by[rid]["correct"]
    ]
    unreproduced_episode_fails = [
        rid for rid in episode_final_fail_ids if base_by[rid]["correct"]
    ]
    base_probe["errors_reproduced"] = base_probe["n_final_errors"] > 0
    base_probe["episode_final_fail_ids"] = episode_final_fail_ids
    base_probe["reproduced_episode_final_fails"] = reproduced_episode_fails
    base_probe["unreproduced_episode_final_fails"] = unreproduced_episode_fails
    base_probe["probe_mode"] = (
        "fix_errors"
        if base_probe["errors_reproduced"]
        else "regression_only_measure_in_full_514"
    )
    _write_json(OUT / "probe_base.json", base_probe)
    del base
    _release()

    print(
        f"5.16 base probe errors={base_probe['n_final_errors']}/10 "
        f"mode={base_probe['probe_mode']}; unreproduced_episode_fails="
        f"{unreproduced_episode_fails}",
        flush=True,
    )
    print(
        f"5.16 training fresh LoRA lr={LEARNING_RATE} max_steps={MAX_OPTIMIZER_STEPS}",
        flush=True,
    )
    train_info = train_with_checkpoints(tokenizer, train_ctx, probe_ctx, base_probe)
    _write_json(OUT / "train_sft.json", train_info)

    full = None
    stop_reason = train_info["stop_reason"]
    if train_info["selected"] is not None:
        adapter = ROOT / train_info["selected"]["adapter"]
        print(f"5.16 selected {adapter}; running full 5.14 eval", flush=True)
        full = run_full_514_eval(adapter)
        stop_reason = (
            "pilot_passed"
            if full["pilot_criteria"]["pilot_passed"]
            else "pilot_failed_criteria"
        )
    else:
        print("5.16 no checkpoint selected; skip full 5.14 eval", flush=True)

    summary = {
        "kind": "sft_pilot_5_16",
        "method": "SFT",
        "stop_reason": stop_reason,
        "command": COMMAND,
        "git_head": _git_head(),
        "python": platform.python_version(),
        "versions": _package_versions(),
        "elapsed_s": round(time.time() - started, 1),
        "pinned_config": pinned,
        "train": {
            "n": len(train_ctx),
            "example_order": [c["rollout_id"] for c in train_ctx],
            "source": "artifacts/task-5-15/collection/train/rollouts.json",
        },
        "base_probe": {
            "n": base_probe["n"],
            "n_correct": base_probe["n_correct"],
            "n_final_errors": base_probe["n_final_errors"],
            "error_ids": base_probe["error_ids"],
            "errors_reproduced": base_probe["errors_reproduced"],
            "probe_mode": base_probe["probe_mode"],
            "episode_final_fail_ids": episode_final_fail_ids,
            "reproduced_episode_final_fails": reproduced_episode_fails,
            "unreproduced_episode_final_fails": unreproduced_episode_fails,
        },
        "training": train_info,
        "eval_514": None
        if full is None
        else {
            "adapter": full["adapter"],
            "adapter_aggregates": full["adapter_aggregates"],
            "base_aggregates": full["base_aggregates"],
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
                "base_probe_errors": base_probe["n_final_errors"],
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


if __name__ == "__main__":
    raise SystemExit(main())
