"""Task 5.23: weighted-REF SFT on four tool-call-after-submit contexts + 5.14 eval.

Train order (one optimizer step each): c21_6-g101, c21_7-g100, c21_7-g101,
c22_3-g101. Fresh LoRA, lr=2e-5, first final token ×5 (must be REF/26501), other
final tokens ×1, prefix masked, loss ÷ sum(weights). Eval adapter pinned as
step_4 before training. Full 5.14 episode eval with 5.20 settings/scoring.
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
from contract.grpo_pair import _release
from contract.grpo_prep import _lora_dict, _package_versions, _write_json
from contract.grpo_train_step import _lora_grad_norm, _lora_params
from contract.sft_eval_520 import (
    _baseline_pairs_from_514,
    check_conditions_vs_514,
    compare,
    run_matrix,
)
from contract.sft_final import score_final
from contract.sft_pilot_519 import (
    DIVERGE_CALL_TOKEN_ID,
    FIRST_TOKEN_WEIGHT,
    LEARNING_RATE,
    OTHER_FINAL_WEIGHT,
    REF_TOKEN_ID,
    _build_train_tensors,
    _delta_logp_ref_minus_bracket,
    _tok_text,
    _weighted_ce_loss,
)

OUT = ROOT / "artifacts" / "task-5-23"
SRC_521 = ROOT / "artifacts" / "task-5-21" / "collection" / "post_submit_contexts.json"
SRC_522 = ROOT / "artifacts" / "task-5-22" / "collection" / "post_submit_contexts.json"
COMMAND = (
    "HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
    ".venv/bin/python run_sft_pilot_523.py"
)

TRAIN_ORDER = (
    "c21_6-g101",
    "c21_7-g100",
    "c21_7-g101",
    "c22_3-g101",
)
EVAL_ADAPTER_STEP = 4  # pinned before training


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _load_capture(rid: str) -> dict[str, Any]:
    for path in (SRC_521, SRC_522):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("tool_call_after_submit") or []:
            if row["rollout_id"] == rid:
                return {**row, "source_path": str(path.relative_to(ROOT))}
    raise KeyError(f"missing tool-call-after-submit capture for {rid}")


def load_train_contexts(tokenizer) -> list[dict[str, Any]]:
    contexts = []
    for rid in TRAIN_ORDER:
        cap = _load_capture(rid)
        if not cap.get("confirmed"):
            raise RuntimeError(f"{rid}: capture not confirmed")
        ver = cap.get("verification") or {}
        if not ver.get("gold_score_final_correct") or not ver.get(
            "input_ids_match_turn_capture"
        ):
            raise RuntimeError(f"{rid}: capture failed verification flags: {ver}")
        prefix = list(cap["prefix_input_ids"])
        target = list(cap["gold_token_ids"])
        gold_text = cap["gold_text"]
        re_ids = tokenizer.encode(gold_text, add_special_tokens=False)
        if re_ids != target:
            raise RuntimeError(f"{rid}: gold_token_ids != re-encode(gold_text)")
        if target[0] != REF_TOKEN_ID:
            raise RuntimeError(f"{rid}: first gold token {target[0]} != {REF_TOKEN_ID}")
        if _tok_text(tokenizer, REF_TOKEN_ID) != "REF":
            raise RuntimeError("decode(26501) != REF")
        scored = score_final(gold_text, int(cap["amount_cents"]))
        if not scored["correct"]:
            raise RuntimeError(f"{rid}: gold failed score_final: {scored}")
        call_ids = list(cap["generated_new_token_ids"])
        if not call_ids or call_ids[0] != DIVERGE_CALL_TOKEN_ID:
            raise RuntimeError(
                f"{rid}: expected call to start with '[' ({DIVERGE_CALL_TOKEN_ID}), "
                f"got {None if not call_ids else call_ids[0]}"
            )
        contexts.append(
            {
                "rollout_id": rid,
                "episode_id": cap["episode_id"],
                "generation_seed": cap["generation_seed"],
                "order_id": cap["order_id"],
                "customer_id": cap["customer_id"],
                "amount_cents": int(cap["amount_cents"]),
                "prefix_input_ids": prefix,
                "target_ids": target,
                "target_text": gold_text,
                "gold_token_ids": target,
                "call_token_ids": call_ids,
                "call_text": cap["generated_raw"],
                "source_path": cap["source_path"],
                "n_prefix": len(prefix),
                "n_target": len(target),
            }
        )
    if len(contexts) != 4:
        raise RuntimeError(f"expected 4 train contexts, got {len(contexts)}")
    return contexts


def write_pinned_config(train_ctx: list[dict[str, Any]], condition_check: dict[str, Any]) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "adapters").mkdir(parents=True, exist_ok=True)
    (OUT / "datasets").mkdir(parents=True, exist_ok=True)
    (OUT / "eval_514").mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "sft_pilot_5_23",
        "method": "SFT",
        "written_before_training": True,
        "command": COMMAND,
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "inherits_training_recipe_from": "artifacts/task-5-19",
        "inherits_eval_from": "artifacts/task-5-20",
        "train": {
            "n": len(train_ctx),
            "example_order": [c["rollout_id"] for c in train_ctx],
            "sources": {
                c["rollout_id"]: c["source_path"] for c in train_ctx
            },
            "inputs": "saved prefix_input_ids",
            "targets": "verified REFUND_SUBMITTED (+ <|END_RESPONSE|>) gold_token_ids",
            "optimizer_schedule": "one_pass_one_optimizer_step_per_example",
            "n_optimizer_steps": len(train_ctx),
        },
        "training": {
            "learning_rate": LEARNING_RATE,
            "fresh_lora": True,
            "lora": _lora_dict(),
            "loss": "weighted_causal_lm_on_final_tokens_only",
            "prefix_masked": True,
            "first_final_token_id_required": REF_TOKEN_ID,
            "first_final_token_text": "REF",
            "first_final_token_weight": FIRST_TOKEN_WEIGHT,
            "other_final_token_weight": OTHER_FINAL_WEIGHT,
            "loss_normalization": "sum_of_token_weights",
        },
        "delta_logp_probe": {
            "contexts": "same four train prefixes",
            "metric": "logp(REF) - logp([) at first divergence",
            "when": ["before_training", "after_training"],
        },
        "eval": {
            "matrix": "artifacts/task-5-14 d0–d5 × {100,101} (12 episodes)",
            "adapter_pinned_before_training": f"adapters/step_{EVAL_ADAPTER_STEP}",
            "settings": "same as task-5-20 / task-5-14",
            "condition_check_vs_5_14": condition_check,
            "pass_criterion": {
                "at_least_one_prev_fail_becomes_full_success": True,
                "no_prev_success_breaks": True,
                "invalid_calls_do_not_increase": True,
            },
        },
    }
    _write_json(OUT / "pinned_config.json", payload)
    (OUT / "commands.txt").write_text(COMMAND + "\n", encoding="utf-8")
    with (OUT / "datasets" / "train_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for i, ctx in enumerate(train_ctx):
            handle.write(
                json.dumps(
                    {
                        "order": i,
                        "rollout_id": ctx["rollout_id"],
                        "episode_id": ctx["episode_id"],
                        "generation_seed": ctx["generation_seed"],
                        "order_id": ctx["order_id"],
                        "amount_cents": ctx["amount_cents"],
                        "n_prefix": ctx["n_prefix"],
                        "n_target": ctx["n_target"],
                        "first_target_id": ctx["target_ids"][0],
                        "target_text": ctx["target_text"],
                        "source_path": ctx["source_path"],
                        "optimizer_step_index": i + 1,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return payload


def train_one_step_per_example(
    tokenizer,
    train_ctx: list[dict[str, Any]],
) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    from agenttune.agentic.rollout_engines.rollout_factory import ensure_lora_targets

    encoded = _build_train_tensors(train_ctx)
    if [e["rollout_id"] for e in encoded] != list(TRAIN_ORDER):
        raise RuntimeError("encoded order != TRAIN_ORDER")
    for e in encoded:
        if e["first_target_id"] != REF_TOKEN_ID:
            raise RuntimeError(f"{e['rollout_id']}: boosted token != REF")
    _write_json(
        OUT / "train_encoding.json",
        {
            "example_order": [e["rollout_id"] for e in encoded],
            "first_token_id_verified": REF_TOKEN_ID,
            "first_token_weight": FIRST_TOKEN_WEIGHT,
            "other_final_token_weight": OTHER_FINAL_WEIGHT,
            "examples": [
                {
                    "rollout_id": e["rollout_id"],
                    "n_prefix": e["n_prefix"],
                    "n_target": e["n_target"],
                    "first_target_id": e["first_target_id"],
                    "sum_weights": e["sum_weights"],
                    "target_text": train_ctx[i]["target_text"],
                }
                for i, e in enumerate(encoded)
            ],
        },
    )

    # Delta-logp examples use gold vs observed call token ids on the train prefixes.
    diverge_examples = [
        {
            "rollout_id": c["rollout_id"],
            "role": "train_fail",
            "prefix_input_ids": c["prefix_input_ids"],
            "gold_token_ids": c["gold_token_ids"],
            "call_token_ids": c["call_token_ids"],
        }
        for c in train_ctx
    ]

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

    print("5.23 Δlogp before training (fresh LoRA)", flush=True)
    delta_before = _delta_logp_ref_minus_bracket(model, tokenizer, diverge_examples)
    _write_json(OUT / "delta_logp_before.json", {"phase": "before_training", "examples": delta_before})

    history = []
    for step, ex in enumerate(encoded, start=1):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss, meta = _weighted_ce_loss(
            model, ex["input_ids"], ex["labels"], ex["label_weights"]
        )
        loss.backward()
        for name, param in _lora_params(model):
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise RuntimeError(f"non-finite grad in {name} at step {step}")
        grad_norm = float(_lora_grad_norm(model))
        opt.step()
        adapter_dir = OUT / "adapters" / f"step_{step}"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)
        record = {
            "step": step,
            "rollout_id": ex["rollout_id"],
            "train_loss": float(loss.detach().item()),
            "sum_weights": meta["sum_weights"],
            "n_supervised": meta["n_supervised"],
            "grad_norm": grad_norm,
            "adapter": str(adapter_dir.relative_to(ROOT)),
        }
        history.append(record)
        print(
            f"sft523 step={step} rid={ex['rollout_id']} "
            f"loss={record['train_loss']:.6f} grad={grad_norm:.4g}",
            flush=True,
        )

    print("5.23 Δlogp after training (step_4)", flush=True)
    delta_after = _delta_logp_ref_minus_bracket(model, tokenizer, diverge_examples)
    _write_json(OUT / "delta_logp_after.json", {"phase": "after_training", "examples": delta_after})

    eval_adapter = OUT / "adapters" / f"step_{EVAL_ADAPTER_STEP}"
    if not eval_adapter.exists():
        raise RuntimeError(f"pinned eval adapter missing: {eval_adapter}")

    del model
    _release()
    return {
        "learning_rate": LEARNING_RATE,
        "example_order": list(TRAIN_ORDER),
        "history": history,
        "eval_adapter": str(eval_adapter.relative_to(ROOT)),
        "delta_logp_before": {
            rid: delta_before[rid]["delta_logp_ref_minus_bracket"] for rid in TRAIN_ORDER
        },
        "delta_logp_after": {
            rid: delta_after[rid]["delta_logp_ref_minus_bracket"] for rid in TRAIN_ORDER
        },
        "delta_logp_change": {
            rid: (
                delta_after[rid]["delta_logp_ref_minus_bracket"]
                - delta_before[rid]["delta_logp_ref_minus_bracket"]
            )
            for rid in TRAIN_ORDER
        },
    }


def _write_report(summary: dict[str, Any]) -> None:
    train = summary["training"]
    ba = summary["eval_514"]["base_aggregates"]
    aa = summary["eval_514"]["adapter_aggregates"]
    lines = [
        "# Task 5.23 — weighted-REF SFT on four after-submit fails + 5.14 eval",
        "",
        f"- train order: `{train['example_order']}`",
        f"- recipe: lr={LEARNING_RATE}, REF×{FIRST_TOKEN_WEIGHT}, other×{OTHER_FINAL_WEIGHT}, "
        "one optimizer step per example",
        f"- eval adapter (pinned): `{train['eval_adapter']}`",
        f"- pass_criterion: `{summary['eval_514']['pass_criterion']}`",
        "",
        "## Δlogp(REF−[) on train contexts",
        "",
        "| rollout | before | after | Δ |",
        "|---|---:|---:|---:|",
    ]
    for rid in TRAIN_ORDER:
        b = train["delta_logp_before"][rid]
        a = train["delta_logp_after"][rid]
        lines.append(f"| {rid} | {b:.6f} | {a:.6f} | {a - b:.6f} |")
    lines.extend(
        [
            "",
            "## 5.14 episode eval (base → step_4)",
            "",
            "| metric | base | step_4 |",
            "|---|---:|---:|",
            f"| full_success | {ba['full_success']}/{ba['n']} | {aa['full_success']}/{aa['n']} |",
            f"| confirmed_returns | {ba['confirmed_returns']} | {aa['confirmed_returns']} |",
            f"| correct_finals | {ba['correct_finals']} | {aa['correct_finals']} |",
            f"| invalid_calls_total | {ba['invalid_calls_total']} | {aa['invalid_calls_total']} |",
            f"| calls_after_confirmed_submit_total | "
            f"{ba['calls_after_confirmed_submit_total']} | "
            f"{aa['calls_after_confirmed_submit_total']} |",
            "",
            f"- newly fixed: `{summary['eval_514']['newly_fixed_full_success']}`",
            f"- newly broken: `{summary['eval_514']['newly_broken_from_base_success']}`",
            "",
        ]
    )
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    from transformers import AutoTokenizer

    started = time.time()
    condition_check = check_conditions_vs_514()
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID, revision=config.MODEL_REVISION, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ctx = load_train_contexts(tokenizer)
    pinned = write_pinned_config(train_ctx, condition_check)
    print(
        f"5.23 pinned train={ [c['rollout_id'] for c in train_ctx] } "
        f"eval_adapter=step_{EVAL_ADAPTER_STEP} "
        f"substantial_diff={condition_check['substantial_difference']}",
        flush=True,
    )

    train_info = train_one_step_per_example(tokenizer, train_ctx)
    _write_json(OUT / "train_sft.json", train_info)

    eval_adapter = ROOT / train_info["eval_adapter"]
    if condition_check["substantial_difference"]:
        print("5.23 re-running base under adapter conditions", flush=True)
        base_run = run_matrix(None, out_dir=OUT / "eval_514" / "base_rerun", label="base")
        base_pairs = base_run["pairs"]
        comparison_source = "artifacts/task-5-23/eval_514/base_rerun"
    else:
        print("5.23 using saved 5.14 baseline for comparison", flush=True)
        base_pairs = _baseline_pairs_from_514()
        comparison_source = "artifacts/task-5-14/baseline/per_pair.json"

    print(f"5.23 full 5.14 eval on {eval_adapter}", flush=True)
    adapter_run = run_matrix(
        eval_adapter, out_dir=OUT / "eval_514" / "adapter_step_4", label="step_4"
    )
    comparison = compare(base_pairs, adapter_run["pairs"])
    _write_json(OUT / "eval_514" / "comparison.json", comparison)

    changed = [
        {
            "rollout_id": row["rollout_id"],
            "transition": row["transition"],
            "base": row["base"],
            "adapter": row["adapter"],
        }
        for row in comparison["per_episode"]
        if row["transition"] not in {"unchanged_success", "unchanged_fail"}
    ]

    summary = {
        "kind": "sft_pilot_5_23",
        "method": "SFT",
        "command": COMMAND,
        "git_head": _git_head(),
        "python": platform.python_version(),
        "versions": _package_versions(),
        "elapsed_s": round(time.time() - started, 1),
        "pinned_config": pinned,
        "training": train_info,
        "eval_514": {
            "adapter": train_info["eval_adapter"],
            "comparison_source": comparison_source,
            "condition_check": condition_check,
            "base_aggregates": comparison["base_aggregates"],
            "adapter_aggregates": comparison["adapter_aggregates"],
            "newly_fixed_full_success": comparison["newly_fixed_full_success"],
            "newly_broken_from_base_success": comparison["newly_broken_from_base_success"],
            "changed_outcomes": changed,
            "pass_criterion": comparison["pass_criterion"],
            "per_episode": comparison["per_episode"],
        },
    }
    _write_json(OUT / "summary.json", summary)
    _write_report(summary)
    print(
        json.dumps(
            {
                "pass_criterion": comparison["pass_criterion"],
                "base": comparison["base_aggregates"],
                "step_4": comparison["adapter_aggregates"],
                "changed": changed,
                "delta_logp_before": train_info["delta_logp_before"],
                "delta_logp_after": train_info["delta_logp_after"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
