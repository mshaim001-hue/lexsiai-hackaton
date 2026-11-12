"""Task 5.19: 5.16 SFT pilot with 5× weight on first gold final token (REF).

Same train order, fresh LoRA, lr=2e-5, ≤3 optimizer steps as 5.16. Only change:
token-level CE weights the first supervised final token (must be 26501 / REF) by
5×, other final tokens 1×, prefix masked; loss normalized by sum of weights.
After each step: save adapter, 5.14 quick-check, and Δlogp(REF−[) on 5.18's four
examples. Stop early if a step fixes ≥1 of {d1-g101,d2-g101} and keeps all eight
base-correct finals. No full episode eval.
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
from contract.sft_first_diverge_518 import CONTROL_IDS, FAIL_IDS, _first_diverge, _tok_text
from contract.sft_logprob_diag_517 import _load_examples
from contract.sft_pilot_516 import (
    INCLUDE_END_RESPONSE_IN_TARGET,
    LEARNING_RATE,
    MAX_OPTIMIZER_STEPS,
    TRAIN_ORDER_515,
    evaluate_probe_contexts,
    load_probe_514_contexts,
    load_train_contexts,
)

OUT = ROOT / "artifacts" / "task-5-19"
COMMAND = (
    "HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
    ".venv/bin/python run_sft_pilot_519.py"
)

REF_TOKEN_ID = 26501
FIRST_TOKEN_WEIGHT = 5.0
OTHER_FINAL_WEIGHT = 1.0
DIVERGE_CALL_TOKEN_ID = 73  # '['


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def write_pinned_config(
    train_ctx: list[dict[str, Any]], probe_ctx: list[dict[str, Any]]
) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "adapters").mkdir(parents=True, exist_ok=True)
    (OUT / "datasets").mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "sft_pilot_5_19",
        "method": "SFT",
        "written_before_training": True,
        "command": COMMAND,
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "inherits_from": "artifacts/task-5-16",
        "train": {
            "source": "artifacts/task-5-15 confirmed train chains",
            "n": len(train_ctx),
            "example_order": [c["rollout_id"] for c in train_ctx],
            "gold_amount_source": "submit_refund tool result (captured in 5.15)",
            "include_end_response_in_target": INCLUDE_END_RESPONSE_IN_TARGET,
            "same_as_5_16_order": [c["rollout_id"] for c in train_ctx] == list(TRAIN_ORDER_515),
        },
        "quick_check": {
            "source": "artifacts/task-5-14 confirmed post-submit input_ids",
            "n": len(probe_ctx),
            "rollout_ids": [c["rollout_id"] for c in probe_ctx],
            "parse": "task-5-12 truncate_at_end_response",
            "compare_to": "fresh base regeneration on same input_ids and seeds",
        },
        "divergence_probe": {
            "source": "artifacts/task-5-18 four examples",
            "fail_ids": list(FAIL_IDS),
            "control_ids": list(CONTROL_IDS),
            "metric": "logp(REF) - logp([) at first divergence (div@0)",
            "gold_token_id": REF_TOKEN_ID,
            "call_token_id": DIVERGE_CALL_TOKEN_ID,
        },
        "training": {
            "learning_rate": LEARNING_RATE,
            "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
            "loss": "weighted_causal_lm_on_final_tokens_only",
            "prefix_masked": True,
            "first_final_token_id_required": REF_TOKEN_ID,
            "first_final_token_text": "REF",
            "first_final_token_weight": FIRST_TOKEN_WEIGHT,
            "other_final_token_weight": OTHER_FINAL_WEIGHT,
            "loss_normalization": "sum_of_token_weights",
            "fresh_lora": True,
            "lora": _lora_dict(),
            "stop_if_empty_finals": True,
            "stop_on_selection_criterion": True,
        },
        "checkpoint_selection": {
            "no_full_514_eval": True,
            "criterion": (
                "exact correct final on >=1 of reproducible fails "
                f"{list(FAIL_IDS)} while keeping all eight base-correct probe finals"
            ),
            "action_on_pass": "stop pilot and keep that adapter",
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
                        "first_target_id": ctx["target_ids"][0],
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


def _build_train_tensors(train_ctx: list[dict[str, Any]]) -> list[dict[str, Any]]:
    encoded = []
    for ctx in train_ctx:
        prefix = list(ctx["prefix_input_ids"])
        target = list(ctx["target_ids"])
        if not target:
            raise RuntimeError(f"{ctx['rollout_id']}: empty target")
        if target[0] != REF_TOKEN_ID:
            raise RuntimeError(
                f"{ctx['rollout_id']}: first target token {target[0]} != "
                f"required REF id {REF_TOKEN_ID}"
            )
        labels = [-100] * len(prefix) + target
        if any(x != -100 for x in labels[: len(prefix)]):
            raise RuntimeError("prefix not masked")
        # Causal LM shift: logits[t] predicts labels[t+1]. First target sits at
        # labels[n_prefix] and is predicted from position n_prefix - 1.
        n_prefix = len(prefix)
        weights = [0.0] * len(labels)
        for i, lab in enumerate(labels):
            if lab == -100:
                continue
            # weight applies to the CE term at shifted index i-1 predicting lab.
            # Store weight aligned to label index i; apply after shift.
            weights[i] = (
                FIRST_TOKEN_WEIGHT if i == n_prefix else OTHER_FINAL_WEIGHT
            )
        encoded.append(
            {
                "rollout_id": ctx["rollout_id"],
                "input_ids": prefix + target,
                "labels": labels,
                "label_weights": weights,
                "n_prefix": n_prefix,
                "n_target": len(target),
                "first_target_id": target[0],
                "sum_weights": sum(weights),
            }
        )
    return encoded


def _weighted_ce_loss(model, input_ids: list[int], labels: list[int], label_weights: list[float]):
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    lab = torch.tensor([labels], dtype=torch.long, device=device)
    w_full = torch.tensor([label_weights], dtype=torch.float32, device=device)
    logits = model(input_ids=ids).logits
    # Standard causal shift.
    shift_logits = logits[:, :-1, :].float()
    shift_labels = lab[:, 1:]
    shift_weights = w_full[:, 1:]
    vocab = shift_logits.size(-1)
    token_loss = F.cross_entropy(
        shift_logits.reshape(-1, vocab),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shift_labels)
    # Zero out ignored positions explicitly (ignore_index already zeros CE, but
    # weights for -100 are 0 anyway).
    mask = shift_labels.ne(-100).to(token_loss.dtype)
    weighted = token_loss * shift_weights * mask
    denom = (shift_weights * mask).sum().clamp_min(1e-8)
    loss = weighted.sum() / denom
    return loss, {
        "sum_weights": float((shift_weights * mask).sum().item()),
        "n_supervised": int(mask.sum().item()),
        "mean_unweighted": float((token_loss * mask).sum().item() / max(1, int(mask.sum().item()))),
    }


def _delta_logp_ref_minus_bracket(
    model, tokenizer, examples: list[dict[str, Any]]
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    by_id: dict[str, Any] = {}
    model.eval()
    for ex in examples:
        div = _first_diverge(ex["gold_token_ids"], ex["call_token_ids"])
        if div["gold_token_id"] != REF_TOKEN_ID or div["call_token_id"] != DIVERGE_CALL_TOKEN_ID:
            raise RuntimeError(
                f"{ex['rollout_id']}: unexpected diverge tokens "
                f"{div['gold_token_id']}/{div['call_token_id']}"
            )
        ctx = ex["prefix_input_ids"] + div["shared_prefix_ids"]
        ids = torch.tensor([ctx], dtype=torch.long, device=device)
        with torch.no_grad():
            logp = F.log_softmax(model(input_ids=ids).logits[0, -1].float(), dim=-1)
        gold_logp = float(logp[REF_TOKEN_ID].item())
        call_logp = float(logp[DIVERGE_CALL_TOKEN_ID].item())
        by_id[ex["rollout_id"]] = {
            "rollout_id": ex["rollout_id"],
            "role": ex["role"],
            "diverge_index": div["diverge_index"],
            "gold_token_id": REF_TOKEN_ID,
            "gold_token_text": _tok_text(tokenizer, REF_TOKEN_ID),
            "call_token_id": DIVERGE_CALL_TOKEN_ID,
            "call_token_text": _tok_text(tokenizer, DIVERGE_CALL_TOKEN_ID),
            "gold_logp": gold_logp,
            "call_logp": call_logp,
            "delta_logp_ref_minus_bracket": gold_logp - call_logp,
        }
    return by_id


def train_with_checkpoints(
    tokenizer,
    train_ctx: list[dict[str, Any]],
    probe_ctx: list[dict[str, Any]],
    diverge_examples: list[dict[str, Any]],
    base_probe: dict[str, Any],
    base_delta: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from agenttune.agentic.rollout_engines.rollout_factory import ensure_lora_targets

    encoded = _build_train_tensors(train_ctx)
    first_ids = [e["first_target_id"] for e in encoded]
    if any(t != REF_TOKEN_ID for t in first_ids):
        raise RuntimeError(f"boosted token check failed: {first_ids}")
    _write_json(
        OUT / "train_encoding.json",
        {
            "n": len(encoded),
            "example_order": [e["rollout_id"] for e in encoded],
            "include_end_response": INCLUDE_END_RESPONSE_IN_TARGET,
            "first_token_id_verified": REF_TOKEN_ID,
            "first_token_text": "REF",
            "first_token_weight": FIRST_TOKEN_WEIGHT,
            "other_final_token_weight": OTHER_FINAL_WEIGHT,
            "examples": [
                {
                    "rollout_id": e["rollout_id"],
                    "n_prefix": e["n_prefix"],
                    "n_target": e["n_target"],
                    "first_target_id": e["first_target_id"],
                    "sum_weights": e["sum_weights"],
                    "expected_sum_weights": FIRST_TOKEN_WEIGHT
                    + OTHER_FINAL_WEIGHT * (e["n_target"] - 1),
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
    if len(base_correct_ids) != 8:
        raise RuntimeError(
            f"expected 8 base-correct probe finals, got {len(base_correct_ids)}: "
            f"{sorted(base_correct_ids)}"
        )
    target_fail_ids = set(FAIL_IDS)
    for rid in target_fail_ids:
        if rid not in {p["rollout_id"] for p in base_probe["pairs"]}:
            raise RuntimeError(f"fail id {rid} missing from probe")
        if rid in base_correct_ids:
            raise RuntimeError(f"fail id {rid} unexpectedly correct on base regen")

    history: list[dict[str, Any]] = []
    stop_reason = "max_steps_reached"
    selected = None
    for step in range(1, MAX_OPTIMIZER_STEPS + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_sum_w = 0.0
        n_sup = 0
        for ex in encoded:
            loss, meta = _weighted_ce_loss(
                model, ex["input_ids"], ex["labels"], ex["label_weights"]
            )
            # Average across the 6 examples like 5.16 (each example contributes equally).
            (loss / len(encoded)).backward()
            total_loss += float(loss.detach().item())
            total_sum_w += meta["sum_weights"]
            n_sup += meta["n_supervised"]
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

        delta = _delta_logp_ref_minus_bracket(model, tokenizer, diverge_examples)
        _write_json(OUT / f"delta_logp_step_{step}.json", {"examples": delta})

        by_id = {p["rollout_id"]: p for p in probe["pairs"]}
        fixed = [rid for rid in sorted(target_fail_ids) if by_id[rid]["correct"]]
        broken = [rid for rid in sorted(base_correct_ids) if not by_id[rid]["correct"]]
        empty_ids = probe["empty_ids"]
        keeps_all = len(broken) == 0
        fixes_target = len(fixed) >= 1
        passes = fixes_target and keeps_all and not empty_ids
        record = {
            "step": step,
            "mean_train_loss": total_loss / max(1, len(encoded)),
            "mean_sum_weights": total_sum_w / max(1, len(encoded)),
            "grad_norm": grad_norm,
            "n_supervised_tokens_per_epoch": n_sup,
            "adapter": str(adapter_dir.relative_to(ROOT)),
            "probe_n_correct": probe["n_correct"],
            "probe_n_final_errors": probe["n_final_errors"],
            "probe_n_empty": probe["n_empty"],
            "fixed_ids": fixed,
            "broken_ids": broken,
            "empty_ids": empty_ids,
            "keeps_all_base_correct": keeps_all,
            "fixes_at_least_one_target_fail": fixes_target,
            "passes_selection": passes,
            "delta_logp_ref_minus_bracket": {
                rid: delta[rid]["delta_logp_ref_minus_bracket"] for rid in delta
            },
            "delta_logp_on_fails": {
                rid: delta[rid]["delta_logp_ref_minus_bracket"] for rid in FAIL_IDS
            },
        }
        history.append(record)
        print(
            f"sft519 step={step} loss={record['mean_train_loss']:.6f} "
            f"fixed={fixed} broken={broken} empty={empty_ids} "
            f"Δlogp_fails={record['delta_logp_on_fails']}",
            flush=True,
        )
        if empty_ids:
            stop_reason = "empty_finals"
            print("5.19 stop: adapter produced empty finals", flush=True)
            break
        if passes:
            selected = record
            stop_reason = "selection_criterion_met"
            print(
                f"5.19 stop: criterion met at step {step}; keeping {adapter_dir}",
                flush=True,
            )
            break

    return {
        "learning_rate": LEARNING_RATE,
        "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
        "first_token_id": REF_TOKEN_ID,
        "first_token_weight": FIRST_TOKEN_WEIGHT,
        "other_final_token_weight": OTHER_FINAL_WEIGHT,
        "example_order": [c["rollout_id"] for c in train_ctx],
        "target_fail_ids": list(FAIL_IDS),
        "base_correct_ids": sorted(base_correct_ids),
        "base_delta_logp": {
            rid: base_delta[rid]["delta_logp_ref_minus_bracket"] for rid in base_delta
        },
        "history": history,
        "selected": selected,
        "train_stop_reason": stop_reason,
        "stop_reason": (
            "checkpoint_selected"
            if selected is not None
            else "no_checkpoint_passed_selection"
        ),
    }


def _write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Task 5.19 — 5.16 SFT pilot with 5× weight on REF",
        "",
        "Same train order / fresh LoRA / `lr=2e-5` / ≤3 steps as 5.16. "
        "Token-level CE: first final token (`26501` / `REF`) weight **5×**, "
        "other final tokens **1×**, prefix masked; loss ÷ sum(weights).",
        "",
        f"- stop_reason: `{summary['stop_reason']}`",
        f"- lr / max steps: `{LEARNING_RATE}` / `{MAX_OPTIMIZER_STEPS}`",
        f"- train order: `{summary['train']['example_order']}`",
        f"- base probe: {summary['base_probe']['n_correct']}/"
        f"{summary['base_probe']['n']} correct; "
        f"target fails `{summary['base_probe']['target_fail_ids']}`",
        "",
        "## Per step",
        "",
        "| step | loss | fixed | broken | empty | "
        "Δlogp d1-g101 | Δlogp d2-g101 | Δlogp d0-g100 | Δlogp d3-g100 | pass |",
        "|---:|---:|---|---|---|---:|---:|---:|---:|---|",
    ]
    base_d = summary["base_probe"]["delta_logp_ref_minus_bracket"]
    lines.append(
        "| base | — | — | — | — | "
        f"{base_d['d1-g101']:.6f} | {base_d['d2-g101']:.6f} | "
        f"{base_d['d0-g100']:.6f} | {base_d['d3-g100']:.6f} | — |"
    )
    for h in summary.get("training", {}).get("history") or []:
        d = h["delta_logp_ref_minus_bracket"]
        lines.append(
            f"| {h['step']} | {h['mean_train_loss']:.6f} | {h['fixed_ids']} | "
            f"{h['broken_ids']} | {h['empty_ids']} | "
            f"{d['d1-g101']:.6f} | {d['d2-g101']:.6f} | "
            f"{d['d0-g100']:.6f} | {d['d3-g100']:.6f} | "
            f"{h['passes_selection']} |"
        )
    lines.append("")
    sel = summary.get("training", {}).get("selected")
    lines.append(f"- selected: `{sel}`")
    lines.append("- full 5.14 episode eval: skipped (per 5.19 brief)")
    lines.append("")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    train_ctx = load_train_contexts()
    probe_ctx = load_probe_514_contexts()
    pinned = write_pinned_config(train_ctx, probe_ctx)
    print(
        f"5.19 pinned lr={LEARNING_RATE} steps<={MAX_OPTIMIZER_STEPS} "
        f"first_token={REF_TOKEN_ID}×{FIRST_TOKEN_WEIGHT} "
        f"train_order={[c['rollout_id'] for c in train_ctx]}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID, revision=config.MODEL_REVISION, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Confirm tokenizer agrees with pinned REF id before any training.
    ref_check = tokenizer.encode("REFUND_SUBMITTED", add_special_tokens=False)[0]
    if ref_check != REF_TOKEN_ID:
        raise RuntimeError(f"tokenizer first REFUND token {ref_check} != {REF_TOKEN_ID}")
    if _tok_text(tokenizer, REF_TOKEN_ID) != "REF":
        raise RuntimeError(
            f"decode({REF_TOKEN_ID})={_tok_text(tokenizer, REF_TOKEN_ID)!r} != 'REF'"
        )

    diverge_examples = _load_examples(tokenizer)
    diverge_examples = [
        ex
        for ex in diverge_examples
        if ex["rollout_id"] in set(FAIL_IDS) | set(CONTROL_IDS)
    ]
    if len(diverge_examples) != 4:
        raise RuntimeError(f"expected 4 diverge examples, got {len(diverge_examples)}")

    print("5.19 base regen on 10 confirmed 5.14 post-submit prefixes", flush=True)
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
    base_delta = _delta_logp_ref_minus_bracket(base, tokenizer, diverge_examples)
    _write_json(OUT / "delta_logp_base.json", {"examples": base_delta})
    base_probe["target_fail_ids"] = list(FAIL_IDS)
    base_probe["errors_reproduced"] = base_probe["n_final_errors"] > 0
    _write_json(OUT / "probe_base.json", base_probe)
    del base
    _release()

    delta_fails = {
        rid: base_delta[rid]["delta_logp_ref_minus_bracket"] for rid in FAIL_IDS
    }
    print(
        f"5.19 base probe correct={base_probe['n_correct']}/10 "
        f"errors={base_probe['error_ids']}; Δlogp_fails={delta_fails}",
        flush=True,
    )
    print(
        "5.19 training fresh LoRA with weighted CE; no full episode eval",
        flush=True,
    )
    train_info = train_with_checkpoints(
        tokenizer,
        train_ctx,
        probe_ctx,
        diverge_examples,
        base_probe,
        base_delta,
    )
    _write_json(OUT / "train_sft.json", train_info)

    summary = {
        "kind": "sft_pilot_5_19",
        "method": "SFT",
        "stop_reason": train_info["stop_reason"]
        if train_info["selected"] is None
        else (
            "selection_criterion_met"
            if train_info["train_stop_reason"] == "selection_criterion_met"
            else train_info["stop_reason"]
        ),
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
            "first_token_id": REF_TOKEN_ID,
            "first_token_weight": FIRST_TOKEN_WEIGHT,
        },
        "base_probe": {
            "n": base_probe["n"],
            "n_correct": base_probe["n_correct"],
            "n_final_errors": base_probe["n_final_errors"],
            "error_ids": base_probe["error_ids"],
            "target_fail_ids": list(FAIL_IDS),
            "delta_logp_ref_minus_bracket": {
                rid: base_delta[rid]["delta_logp_ref_minus_bracket"] for rid in base_delta
            },
        },
        "training": train_info,
        "eval_514": None,
    }
    # Normalize stop_reason for report
    if train_info["selected"] is not None:
        summary["stop_reason"] = "selection_criterion_met"
    else:
        summary["stop_reason"] = train_info["stop_reason"]

    _write_json(OUT / "summary.json", summary)
    _write_report(summary)
    print(
        json.dumps(
            {
                "stop_reason": summary["stop_reason"],
                "selected": train_info.get("selected"),
                "history_fixed_broken": [
                    {
                        "step": h["step"],
                        "fixed": h["fixed_ids"],
                        "broken": h["broken_ids"],
                        "empty": h["empty_ids"],
                        "delta_fails": h["delta_logp_on_fails"],
                    }
                    for h in train_info["history"]
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
