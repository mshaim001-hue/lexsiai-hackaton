"""Task 5.13: short SFT LR probe on one 5.11 example (2e-4 vs 2e-5).

Keeps the 5.11 target/mask fixed so only learning rate differs. No full
episode eval. Budget 30 minutes.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contract import config
from contract.grpo_pair import _install_lora_state, _lora_state, _release
from contract.grpo_prep import _lora_dict, _package_versions, _write_json
from contract.grpo_train_step import _lora_grad_norm, _lora_params, _lora_vector, _param_change
from contract.harness import final_answer_text
from contract.sft_final import (
    END_RESPONSE,
    _render_prefix_text,
    _tokenize_example,
)

OUT = ROOT / "artifacts" / "task-5-13"
TASK_511 = ROOT / "artifacts" / "task-5-11"
COMMAND = (
    "HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
    ".venv/bin/python run_sft_lr_probe.py"
)
TIME_LIMIT_S = 1800
MAX_STEPS = 3
LRS = (2e-4, 2e-5)
TRAIN_EPISODE = "s0"
PROBE_EPISODES = ("p0", "p1")
INIT_LOGIT_ATOL = 1e-3
INIT_LOGIT_RTOL = 1e-3
COLLAPSE_RATIO = 1.0  # P(END_RESPONSE) >= P(gold first) counts as collapse


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _load_chain(path: Path, episode_id: str) -> dict[str, Any]:
    chains = json.loads(path.read_text(encoding="utf-8"))["chains"]
    for chain in chains:
        if chain["episode_id"] == episode_id:
            return chain
    raise KeyError(episode_id)


def _prefix_ids(tokenizer, messages: list[dict]) -> list[int]:
    text = _render_prefix_text(tokenizer, messages)
    return tokenizer.encode(text, add_special_tokens=False)


def _first_token_probs(model, tokenizer, prefix_ids: list[int], amount_cents: int) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    gold = final_answer_text(amount_cents)
    gold_first = tokenizer.encode(gold, add_special_tokens=False)[0]
    end_id = tokenizer.encode(END_RESPONSE, add_special_tokens=False)[0]
    ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, -1].float()
        probs = F.softmax(logits, dim=-1)
        logprobs = F.log_softmax(logits, dim=-1)
    if was_training:
        model.train()
    p_gold = float(probs[gold_first].item())
    p_end = float(probs[end_id].item())
    return {
        "gold_text": gold,
        "gold_first_token_id": int(gold_first),
        "gold_first_token": tokenizer.decode([gold_first]),
        "end_response_token_id": int(end_id),
        "p_gold_first": p_gold,
        "p_end_response": p_end,
        "logp_gold_first": float(logprobs[gold_first].item()),
        "logp_end_response": float(logprobs[end_id].item()),
        "prefer_end_response_over_gold": p_end >= p_gold * COLLAPSE_RATIO,
        "ratio_end_over_gold": (p_end / p_gold) if p_gold > 0 else None,
    }


def _eval_loss(model, input_ids: list[int], labels: list[int]) -> float:
    import torch

    device = next(model.parameters()).device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    lab = torch.tensor([labels], dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        loss = model(input_ids=ids, labels=lab).loss
    if was_training:
        model.train()
    return float(loss.detach().item())


def _trainable_dtype_report(model) -> dict[str, Any]:
    rows = []
    for name, param in _lora_params(model):
        rows.append(
            {
                "name": name,
                "dtype": str(param.dtype),
                "requires_grad": bool(param.requires_grad),
                "shape": list(param.shape),
            }
        )
    dtypes = sorted({row["dtype"] for row in rows})
    return {"n_trainable_lora": len(rows), "dtypes": dtypes, "params": rows[:8], "n_listed": min(8, len(rows))}


def _grad_finite_report(model) -> dict[str, Any]:
    import torch

    finite = True
    n_grads = 0
    n_nonfinite = 0
    max_abs = 0.0
    for _name, param in _lora_params(model):
        if param.grad is None:
            continue
        n_grads += 1
        g = param.grad.detach()
        if not torch.isfinite(g).all():
            finite = False
            n_nonfinite += 1
        max_abs = max(max_abs, float(g.float().abs().max().item()))
    return {
        "n_grads": n_grads,
        "all_finite": finite,
        "n_nonfinite_tensors": n_nonfinite,
        "grad_abs_max": max_abs,
        "grad_norm": float(_lora_grad_norm(model)),
    }


def _compare_init_logits(peft_model, prefix_ids: list[int]) -> dict[str, Any]:
    """Compare adapter-on vs adapter-off logits on the same wrapped model."""
    import torch
    import torch.nn.functional as F

    device = next(peft_model.parameters()).device
    ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    peft_model.eval()
    with torch.no_grad():
        with peft_model.disable_adapter():
            base_logits = peft_model(input_ids=ids).logits[0, -1].float()
        peft_logits = peft_model(input_ids=ids).logits[0, -1].float()
        delta = peft_logits - base_logits
        base_probs = F.softmax(base_logits, dim=-1)
        peft_probs = F.softmax(peft_logits, dim=-1)
    max_abs = float(delta.abs().max().item())
    mean_abs = float(delta.abs().mean().item())
    denom = float(base_logits.abs().mean().clamp_min(1e-3).item())
    ok = max_abs <= max(INIT_LOGIT_ATOL, INIT_LOGIT_RTOL * denom)
    top_delta = torch.topk(delta.abs(), k=5)
    return {
        "ok": ok,
        "method": "peft.disable_adapter() vs adapter enabled at init",
        "max_abs_logit_delta": max_abs,
        "mean_abs_logit_delta": mean_abs,
        "atol": INIT_LOGIT_ATOL,
        "rtol": INIT_LOGIT_RTOL,
        "base_logit_mean_abs": denom,
        "max_abs_prob_delta": float((peft_probs - base_probs).abs().max().item()),
        "top_abs_delta_token_ids": [int(x) for x in top_delta.indices.tolist()],
        "top_abs_delta_values": [float(x) for x in top_delta.values.tolist()],
    }


def _snapshot_metrics(
    model,
    tokenizer,
    *,
    train_enc: dict[str, Any],
    train_prefix_ids: list[int],
    train_amount: int,
    probe_specs: list[dict[str, Any]],
    init_vector,
    step: int,
    phase: str,
    grad_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current = _lora_vector(model)
    change = _param_change(init_vector, current)
    train_probs = _first_token_probs(model, tokenizer, train_prefix_ids, train_amount)
    probe_probs = {
        spec["episode_id"]: _first_token_probs(
            model, tokenizer, spec["prefix_ids"], spec["amount_cents"]
        )
        for spec in probe_specs
    }
    return {
        "step": step,
        "phase": phase,
        "eval_loss": _eval_loss(model, train_enc["input_ids"], train_enc["labels"]),
        "train_first_token": train_probs,
        "probe_first_token": probe_probs,
        "weight_change_from_init": change,
        "grad": grad_report,
        "collapse_on_train_prefix": train_probs["prefer_end_response_over_gold"],
        "collapse_on_any_probe": any(
            row["prefer_end_response_over_gold"] for row in probe_probs.values()
        ),
    }


def _run_lr_arm(
    *,
    tokenizer,
    init_state: dict,
    lr: float,
    train_enc: dict[str, Any],
    train_prefix_ids: list[int],
    train_amount: int,
    probe_specs: list[dict[str, Any]],
    started: float,
) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from agenttune.agentic.rollout_engines.rollout_factory import ensure_lora_targets

    print(f"5.13 arm lr={lr} loading fresh base+LoRA", flush=True)
    torch.manual_seed(config.SEED)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(config.SEED)
    base = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        revision=config.MODEL_REVISION,
        dtype=torch.float16,
        local_files_only=True,
    )
    peft_model = get_peft_model(
        base, ensure_lora_targets(LoraConfig(**_lora_dict()), base)
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    peft_model.to(device)
    _install_lora_state(peft_model, init_state)
    init_vector = _lora_vector(peft_model)
    dtype_report = _trainable_dtype_report(peft_model)

    history = [
        _snapshot_metrics(
            peft_model,
            tokenizer,
            train_enc=train_enc,
            train_prefix_ids=train_prefix_ids,
            train_amount=train_amount,
            probe_specs=probe_specs,
            init_vector=init_vector,
            step=0,
            phase="before_step",
            grad_report=None,
        )
    ]
    print(
        f"lr={lr} step=0 eval_loss={history[-1]['eval_loss']:.6f} "
        f"p_gold={history[-1]['train_first_token']['p_gold_first']:.4f} "
        f"p_end={history[-1]['train_first_token']['p_end_response']:.4f}",
        flush=True,
    )

    opt = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad], lr=lr
    )
    input_ids = torch.tensor([train_enc["input_ids"]], dtype=torch.long, device=device)
    labels = torch.tensor([train_enc["labels"]], dtype=torch.long, device=device)
    attention = torch.ones_like(input_ids)

    collapse_seen = False
    for step in range(1, MAX_STEPS + 1):
        if time.monotonic() - started >= TIME_LIMIT_S:
            break
        peft_model.train()
        opt.zero_grad(set_to_none=True)
        out = peft_model(
            input_ids=input_ids,
            attention_mask=attention,
            labels=labels,
        )
        loss = out.loss
        loss.backward()
        grad_report = _grad_finite_report(peft_model)
        if not grad_report["all_finite"]:
            raise RuntimeError(f"non-finite gradients at lr={lr} step={step}: {grad_report}")
        opt.step()
        snap = _snapshot_metrics(
            peft_model,
            tokenizer,
            train_enc=train_enc,
            train_prefix_ids=train_prefix_ids,
            train_amount=train_amount,
            probe_specs=probe_specs,
            init_vector=init_vector,
            step=step,
            phase="after_step",
            grad_report={
                **grad_report,
                "train_loss_for_backward": float(loss.detach().item()),
            },
        )
        history.append(snap)
        collapse_seen = collapse_seen or snap["collapse_on_train_prefix"] or snap["collapse_on_any_probe"]
        print(
            f"lr={lr} step={step} train_loss={float(loss.detach().item()):.6f} "
            f"eval_loss={snap['eval_loss']:.6f} "
            f"p_gold={snap['train_first_token']['p_gold_first']:.4f} "
            f"p_end={snap['train_first_token']['p_end_response']:.4f} "
            f"grad_norm={grad_report['grad_norm']:.6f} "
            f"dW_l2={snap['weight_change_from_init']['l2']:.6f}",
            flush=True,
        )

    adapter_dir = OUT / f"adapter_lr_{str(lr).replace('.', 'p')}"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(adapter_dir)
    del peft_model, base
    _release()
    return {
        "learning_rate": lr,
        "max_steps": MAX_STEPS,
        "steps_completed": max(h["step"] for h in history),
        "trainable_dtypes": dtype_report,
        "history": history,
        "collapse_seen": collapse_seen,
        "adapter": str(adapter_dir.relative_to(ROOT)),
    }


def write_pinned_config(train_chain: dict[str, Any], probe_chains: list[dict[str, Any]]) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "sft_lr_probe",
        "method": "SFT",
        "written_before_runs": True,
        "command": COMMAND,
        "time_limit_s": TIME_LIMIT_S,
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "lora": _lora_dict(),
        "train_example": {
            "episode_id": train_chain["episode_id"],
            "order_id": train_chain["order_id"],
            "amount_cents": train_chain["amount_cents"],
            "target_final": train_chain["target_final"],
            "source": "artifacts/task-5-11/train_chains.json",
            "target_and_mask": "same as task-5-11 (_tokenize_example, prefix labels -100)",
        },
        "probe_prefixes": [
            {
                "episode_id": c["episode_id"],
                "amount_cents": c["amount_cents"],
                "source": "artifacts/task-5-11/probe_chains.json",
            }
            for c in probe_chains
        ],
        "learning_rates": list(LRS),
        "max_optimizer_steps_per_lr": MAX_STEPS,
        "same_lora_init": True,
        "init_seed": config.SEED,
        "collapse_definition": (
            "P(<|END_RESPONSE|>) >= P(first token of REFUND_SUBMITTED...) "
            "on train or probe prefix"
        ),
        "no_full_episode_eval": True,
    }
    _write_json(OUT / "pinned_config.json", payload)
    (OUT / "commands.txt").write_text(COMMAND + "\n", encoding="utf-8")
    return payload


def main() -> int:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from agenttune.agentic.rollout_engines.rollout_factory import ensure_lora_targets

    started = time.monotonic()
    train_chain = _load_chain(TASK_511 / "train_chains.json", TRAIN_EPISODE)
    probe_chains = [_load_chain(TASK_511 / "probe_chains.json", eid) for eid in PROBE_EPISODES]
    pinned = write_pinned_config(train_chain, probe_chains)

    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID, revision=config.MODEL_REVISION, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_enc = _tokenize_example(
        tokenizer, train_chain["prefix_messages"], train_chain["target_final"]
    )
    # Persist the exact supervised target/mask used for both LRs.
    _write_json(
        OUT / "train_example_encoding.json",
        {
            "episode_id": train_chain["episode_id"],
            "target_final": train_chain["target_final"],
            "n_prefix": train_enc["n_prefix"],
            "n_target": train_enc["n_target"],
            "input_ids": train_enc["input_ids"],
            "labels": train_enc["labels"],
            "prefix_fully_masked": all(
                x == -100 for x in train_enc["labels"][: train_enc["n_prefix"]]
            ),
            "target_fully_supervised": all(
                x != -100 for x in train_enc["labels"][train_enc["n_prefix"] :]
            ),
            "end_response_in_target_ids": any(
                tid
                in set(tokenizer.encode(END_RESPONSE, add_special_tokens=False))
                for tid in train_enc["input_ids"][train_enc["n_prefix"] :]
            ),
        },
    )

    train_prefix_ids = _prefix_ids(tokenizer, train_chain["prefix_messages"])
    probe_specs = [
        {
            "episode_id": chain["episode_id"],
            "amount_cents": chain["amount_cents"],
            "prefix_ids": _prefix_ids(tokenizer, chain["prefix_messages"]),
        }
        for chain in probe_chains
    ]

    print("5.13 creating fresh LoRA and checking init logits vs base", flush=True)
    torch.manual_seed(config.SEED)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(config.SEED)
    base = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        revision=config.MODEL_REVISION,
        dtype=torch.float16,
        local_files_only=True,
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    base.to(device)
    peft_init = get_peft_model(
        base, ensure_lora_targets(LoraConfig(**_lora_dict()), base)
    )
    peft_init.to(device)
    init_check = _compare_init_logits(peft_init, train_prefix_ids)
    init_check_probe = _compare_init_logits(peft_init, probe_specs[0]["prefix_ids"])
    init_state = _lora_state(peft_init)
    torch.save(init_state, OUT / "lora_init.pt")
    _write_json(
        OUT / "init_logit_match.json",
        {
            "train_prefix_s0": init_check,
            "probe_prefix_p0": init_check_probe,
            "passed": bool(init_check["ok"] and init_check_probe["ok"]),
            "trainable_dtypes": _trainable_dtype_report(peft_init),
        },
    )
    if not (init_check["ok"] and init_check_probe["ok"]):
        raise RuntimeError(f"LoRA init logits diverge from base: {init_check} {init_check_probe}")
    print(
        f"init match ok max_abs_train={init_check['max_abs_logit_delta']:.3e} "
        f"max_abs_probe={init_check_probe['max_abs_logit_delta']:.3e}",
        flush=True,
    )

    # Free the init peft wrapper before arms reload their own bases.
    del peft_init, base
    _release()

    arms = []
    for lr in LRS:
        if time.monotonic() - started >= TIME_LIMIT_S:
            break
        arm = _run_lr_arm(
            tokenizer=tokenizer,
            init_state=init_state,
            lr=lr,
            train_enc=train_enc,
            train_prefix_ids=train_prefix_ids,
            train_amount=train_chain["amount_cents"],
            probe_specs=probe_specs,
            started=started,
        )
        arms.append(arm)
        _write_json(OUT / f"arm_lr_{str(lr).replace('.', 'p')}.json", arm)

    collapse_by_lr = {str(arm["learning_rate"]): arm["collapse_seen"] for arm in arms}
    collapse_reproduced = any(arm["collapse_seen"] for arm in arms)
    stop_reason = (
        "collapse_reproduced"
        if collapse_reproduced
        else "collapse_not_reproduced_stop"
    )

    comparison = {
        "learning_rates": [arm["learning_rate"] for arm in arms],
        "collapse_by_lr": collapse_by_lr,
        "collapse_reproduced": collapse_reproduced,
        "final_metrics": {
            str(arm["learning_rate"]): {
                "eval_loss": arm["history"][-1]["eval_loss"],
                "p_gold_first_train": arm["history"][-1]["train_first_token"]["p_gold_first"],
                "p_end_response_train": arm["history"][-1]["train_first_token"][
                    "p_end_response"
                ],
                "probe": {
                    eid: {
                        "p_gold_first": row["p_gold_first"],
                        "p_end_response": row["p_end_response"],
                        "prefer_end_response_over_gold": row[
                            "prefer_end_response_over_gold"
                        ],
                    }
                    for eid, row in arm["history"][-1]["probe_first_token"].items()
                },
                "weight_change_l2": arm["history"][-1]["weight_change_from_init"]["l2"],
            }
            for arm in arms
        },
    }
    _write_json(OUT / "comparison.json", comparison)

    summary = {
        "kind": "sft_lr_probe",
        "method": "SFT",
        "command": COMMAND,
        "git_head": _git_head(),
        "versions": _package_versions(),
        "elapsed_s": time.monotonic() - started,
        "time_limit_s": TIME_LIMIT_S,
        "stop_reason": stop_reason,
        "pinned_config": pinned,
        "init_logit_match_passed": True,
        "collapse_reproduced": collapse_reproduced,
        "collapse_by_lr": collapse_by_lr,
        "arms": [
            {
                "learning_rate": arm["learning_rate"],
                "steps_completed": arm["steps_completed"],
                "collapse_seen": arm["collapse_seen"],
                "adapter": arm["adapter"],
                "trainable_dtypes": arm["trainable_dtypes"]["dtypes"],
                "history_brief": [
                    {
                        "step": h["step"],
                        "phase": h["phase"],
                        "eval_loss": h["eval_loss"],
                        "p_gold_first": h["train_first_token"]["p_gold_first"],
                        "p_end_response": h["train_first_token"]["p_end_response"],
                        "grad_norm": None if h["grad"] is None else h["grad"]["grad_norm"],
                        "grad_all_finite": None
                        if h["grad"] is None
                        else h["grad"]["all_finite"],
                        "weight_change_l2": h["weight_change_from_init"]["l2"],
                        "collapse_train": h["collapse_on_train_prefix"],
                        "collapse_probe": h["collapse_on_any_probe"],
                    }
                    for h in arm["history"]
                ],
            }
            for arm in arms
        ],
        "comparison": comparison,
    }
    _write_json(OUT / "summary.json", summary)
    _write_report(summary)
    print(json.dumps({
        "stop_reason": stop_reason,
        "collapse_reproduced": collapse_reproduced,
        "collapse_by_lr": collapse_by_lr,
        "elapsed_s": summary["elapsed_s"],
    }, indent=2), flush=True)
    return 0


def _write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Task 5.13 — SFT learning-rate probe",
        "",
        "Fresh LoRA from clean Tiny Aya; same 5.11 `s0` target/mask; compare",
        "`lr=2e-4` vs `lr=2e-5` for up to 3 AdamW steps. No full episode eval.",
        "",
        f"- stop_reason: `{summary['stop_reason']}`",
        f"- collapse_reproduced: {summary['collapse_reproduced']}",
        f"- elapsed_s: {summary['elapsed_s']:.1f}",
        "",
        "## Arms",
        "",
    ]
    for arm in summary["arms"]:
        lines.append(f"### lr={arm['learning_rate']}")
        lines.append("")
        lines.append(
            f"- steps={arm['steps_completed']} collapse_seen={arm['collapse_seen']} "
            f"dtypes={arm['trainable_dtypes']}"
        )
        lines.append("")
        lines.append(
            "| step | eval_loss | p_gold | p_end | grad_norm | dW_l2 | collapse_train | collapse_probe |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---|---|")
        for h in arm["history_brief"]:
            gn = "" if h["grad_norm"] is None else f"{h['grad_norm']:.6f}"
            lines.append(
                f"| {h['step']} | {h['eval_loss']:.6f} | {h['p_gold_first']:.4f} | "
                f"{h['p_end_response']:.4f} | {gn} | {h['weight_change_l2']:.6f} | "
                f"{h['collapse_train']} | {h['collapse_probe']} |"
            )
        lines.append("")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
