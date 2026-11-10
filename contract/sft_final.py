"""Task 5.11: SFT on gold final answers after successful submit_refund.

Method: supervised fine-tuning (SFT). Train a fresh LoRA so that, given a
one-action refund dialogue that already confirmed submit, the model emits
exactly REFUND_SUBMITTED amount_cents=<n> from the harness amount. Prefix
tokens are masked out of the loss.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contract import config
from contract.grpo_pair import _release, _save_adapter
from contract.grpo_prep import _lora_dict, _package_versions, _write_json
from contract.harness import (
    ONE_ACTION_SYSTEM_PROMPT,
    ONE_ACTION_TOOL_RESULT_FORMAT,
    ONE_ACTION_TOOLS_FALLBACK_PROMPT,
    RefundEpisode,
    answer_matches,
    final_answer_text,
)
from contract.run import (
    _ScriptedEngine,
    _bind_output,
    _call_json,
    _dump,
    _finish,
    _install_extract_hook,
    _load_model,
    _one_model_rollout,
    _probe,
    _scripted_outputs,
    make_guarded_tools,
)
from contract.sft_prompts import (
    EVAL_TASKS,
    GENERATION_SEEDS,
    PROBE_TASKS,
    TRAIN_TASKS,
    assert_no_overlap,
    sft_user_prompt,
)

OUT = ROOT / "artifacts" / "task-5-11"
COMMAND = (
    "HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
    ".venv/bin/python run_sft_final.py"
)
TIME_LIMIT_S = 7200
MAX_STEPS = 16
LEARNING_RATE = 2e-4
PROBE_TEMPERATURE = 0.7
PROBE_TOP_P = 0.9
PROBE_SEED_BASE = 300


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def write_pinned_config() -> dict[str, Any]:
    assert_no_overlap()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "datasets").mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "sft_final_answer_pilot",
        "method": "SFT",
        "method_note": (
            "Supervised fine-tuning on gold REFUND_SUBMITTED finals from harness "
            "submit_refund; not GRPO."
        ),
        "written_before_training": True,
        "full_eval_ran": False,
        "command": COMMAND,
        "time_limit_s": TIME_LIMIT_S,
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "lora": {**_lora_dict(), "lora_dropout": 0.05},
        "training": {
            "max_optimizer_steps": MAX_STEPS,
            "learning_rate": LEARNING_RATE,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "loss": "causal_lm_on_final_answer_only",
            "prefix_masked": True,
        },
        "train": {
            "n": len(TRAIN_TASKS),
            "episode_ids": [t[0] for t in TRAIN_TASKS],
            "order_ids": [t[1] for t in TRAIN_TASKS],
        },
        "probe": {
            "n": len(PROBE_TASKS),
            "episode_ids": [t[0] for t in PROBE_TASKS],
            "order_ids": [t[1] for t in PROBE_TASKS],
            "excluded_from_training": True,
            "temperature": PROBE_TEMPERATURE,
            "top_p": PROBE_TOP_P,
            "seed_base": PROBE_SEED_BASE,
        },
        "full_eval": {
            "tasks": [t[0] for t in EVAL_TASKS],
            "generation_seeds": list(GENERATION_SEEDS),
            "settings": "same as task-5-8 one-action (T=0.7, top_p=0.9, max_new_tokens=512)",
            "excluded_from_training": True,
            "excludes_v0_v3": True,
        },
        "gate_to_full_eval": {
            "min_fixed_finals": 2,
            "must_not_break_any_correct_base_final": True,
            "definition": (
                "On the 8 probe prefixes, adapter must turn >=2 base-incorrect finals "
                "into correct REFUND_SUBMITTED lines, and must keep every base-correct "
                "final correct."
            ),
        },
        "protocol": "one_action_per_turn",
    }
    _write_json(OUT / "pinned_config.json", payload)
    (OUT / "commands.txt").write_text(COMMAND + "\n", encoding="utf-8")
    # Manifest rows for the three splits.
    for split, tasks in (
        ("train", TRAIN_TASKS),
        ("probe", PROBE_TASKS),
        ("eval", EVAL_TASKS),
    ):
        rows = []
        for episode_id, order_id, customer_id, amount in tasks:
            rows.append(
                {
                    "episode_id": episode_id,
                    "order_id": order_id,
                    "customer_id": customer_id,
                    "amount_cents": amount,
                    "split": split,
                    "prompt": sft_user_prompt(episode_id, order_id, customer_id),
                    "environment_seed": config.ONE_ACTION_ENVIRONMENT_SEED,
                }
            )
        path = OUT / "datasets" / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_json(
        OUT / "datasets" / "manifest.json",
        {
            "train": [t[0] for t in TRAIN_TASKS],
            "probe": [t[0] for t in PROBE_TASKS],
            "eval": [t[0] for t in EVAL_TASKS],
            "overlap_with_5_8_v_or_u": [],
        },
    )
    return payload


class _SFTScriptedEngine(_ScriptedEngine):
    """Scripted completions with a real tokenizer so AgentTune folds tools like live rollouts."""

    def __init__(self, outputs: list[str], tokenizer):
        super().__init__(outputs)
        self._tokenizer = tokenizer

    def _get_tokenizer(self):
        return self._tokenizer


def _render_prefix_text(tokenizer, prefix_messages: list[dict]) -> str:
    """Match Tiny Aya rollout rendering: fold tool turns with ONE_ACTION format."""
    from agenttune.agentic.rollout_engines.rollout_factory import fold_tool_messages_into_user

    has_tool = any(m.get("role") == "tool" for m in prefix_messages)
    messages = (
        fold_tool_messages_into_user(prefix_messages, ONE_ACTION_TOOL_RESULT_FORMAT)
        if has_tool
        else prefix_messages
    )
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _run_one_action_scripted(episode: RefundEpisode, prompt_text: str, tokenizer) -> list[dict]:
    """Successful one-action trajectory through the real AgentTune loop."""
    import agenttune.agentic.rollout_engines.rollout_factory as factory
    from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn

    outputs = _scripted_outputs(episode, final_answer_text(episode.amount_cents))
    guards, _schemas = make_guarded_tools(episode)
    tool_names = {guard.__name__ for guard in guards}
    engine = _SFTScriptedEngine(outputs, tokenizer)
    extract_log: list[dict] = []
    original = factory._extract_tool_calls
    factory._extract_tool_calls = original
    _install_extract_hook(factory, extract_log)
    rollout = create_rollout_fn(
        rollout_engine=engine,
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
        result = rollout(
            [prompt_text],
            max_new_tokens=config.ONE_ACTION_MAX_NEW_TOKENS,
            temperature=0.0,
            do_sample=False,
            max_length=config.CONTEXT_LENGTH,
        )
    finally:
        factory._extract_tool_calls = original
    final_answer = result["trajectories"][0].final_response
    # Rebuild step rows lightly from engine records + extract log for messages.
    rows = []
    used = 0
    cursor = [0]
    from contract.run import (
        PATH,
        _assign_budget,
        _base_step,
        _consume_guard_records,
        _generation_diagnosis,
    )

    for step_id, generated in enumerate(engine.records):
        raw = generated["raw_output"]
        extract = None
        if used < len(extract_log) and extract_log[used]["raw_output"] == raw:
            extract = extract_log[used]
            used += 1
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
        step = _base_step("fixture", episode, step_id)
        step.update(
            {
                "messages": generated["messages"],
                "raw_output": raw,
                "parser_invoked_by_loop": parser_invoked,
                "parser_result": parser_result,
                "diagnostic": diagnostic,
                "calls": calls,
                "pending_unexecuted_calls": pending,
                "function_entries": list(episode.function_entries),
                "execution_path": PATH,
                "max_steps": config.ONE_ACTION_MAX_STEPS,
                "prompt_protocol": "one_action_per_turn",
            }
        )
        rows.append(step)
    if engine._outputs:
        raise RuntimeError(f"unused scripted outputs: {len(engine._outputs)}")
    _assign_budget(rows)
    _finish(rows, episode, final_answer, elapsed=None)
    if not rows[-1].get("task_success"):
        raise RuntimeError(
            f"{episode.episode_id} scripted chain failed success="
            f"{rows[-1].get('task_success')} harness={rows[-1].get('task_success_harness')} "
            f"answer={rows[-1].get('task_success_answer')} final={final_answer!r}"
        )
    if str(episode.amount_cents) not in str(final_answer or ""):
        raise RuntimeError(
            f"{episode.episode_id} final {final_answer!r} missing amount {episode.amount_cents}"
        )
    if not episode.submitted:
        raise RuntimeError(f"{episode.episode_id} submit not confirmed in harness")
    return rows


def build_chain(task: tuple[str, str, str, int], tokenizer) -> dict[str, Any]:
    episode_id, order_id, customer_id, amount = task
    prompt = sft_user_prompt(episode_id, order_id, customer_id)
    episode = RefundEpisode(
        episode_id=episode_id,
        order_id=order_id,
        customer_id=customer_id,
        amount_cents=amount,
        seed=config.ONE_ACTION_ENVIRONMENT_SEED,
    )
    rows = _run_one_action_scripted(episode, prompt, tokenizer)
    # Prefix for the final turn: messages at the last generation (after submit observation).
    prefix_messages = list(rows[-1]["messages"])
    target = final_answer_text(amount)
    if not answer_matches(rows[-1].get("final_answer"), amount):
        # Scripted final may include special tokens; gold target is the clean line.
        pass
    if not episode.submitted:
        raise RuntimeError("submit missing")
    # Confirm last observation is the submit_refund tool result (folded user or tool).
    last_obs = prefix_messages[-1]
    if last_obs.get("role") not in ("tool", "user"):
        raise RuntimeError(f"{episode_id}: prefix must end on tool observation")
    if "submit_refund" not in str(last_obs.get("content") or ""):
        raise RuntimeError(f"{episode_id}: last observation is not submit_refund")
    # System prompt must include tool schemas (tools_fallback) like live rollouts.
    system = prefix_messages[0].get("content") or ""
    if "lookup_order" not in system:
        raise RuntimeError(f"{episode_id}: tools_fallback missing from system prompt")
    return {
        "episode_id": episode_id,
        "order_id": order_id,
        "customer_id": customer_id,
        "amount_cents": amount,
        "prompt": prompt,
        "prefix_messages": prefix_messages,
        "target_final": target,
        "harness_submitted": True,
        "function_entries": list(episode.function_entries),
    }


def collect_chains(tokenizer) -> dict[str, Any]:
    train = [build_chain(task, tokenizer) for task in TRAIN_TASKS]
    probe = [build_chain(task, tokenizer) for task in PROBE_TASKS]
    _write_json(OUT / "train_chains.json", {"method": "SFT", "chains": train})
    _write_json(OUT / "probe_chains.json", {"method": "SFT", "chains": probe})
    return {"train": train, "probe": probe}


def _tokenize_example(tokenizer, prefix_messages: list[dict], target: str) -> dict[str, Any]:
    """Build input_ids / labels with every prefix token masked (-100)."""
    prefix_text = _render_prefix_text(tokenizer, prefix_messages)
    # Match Tiny Aya chat turns: append the gold assistant final without special soup.
    full_text = prefix_text + target
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    if full_ids[: len(prefix_ids)] != prefix_ids:
        # Fall back to encoding target alone and concatenating.
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        full_ids = prefix_ids + target_ids
    else:
        target_ids = full_ids[len(prefix_ids) :]
    if not target_ids:
        raise RuntimeError("empty target tokenisation")
    labels = [-100] * len(prefix_ids) + list(target_ids)
    if len(labels) != len(full_ids):
        raise RuntimeError("label length mismatch")
    if any(label != -100 for label in labels[: len(prefix_ids)]):
        raise RuntimeError("prefix not fully masked")
    if any(label == -100 for label in labels[len(prefix_ids) :]):
        raise RuntimeError("target contains masked positions")
    return {
        "input_ids": full_ids,
        "labels": labels,
        "n_prefix": len(prefix_ids),
        "n_target": len(target_ids),
        "prefix_text": prefix_text,
        "target": target,
    }


def verify_loss_mask(tokenizer, chain: dict[str, Any]) -> dict[str, Any]:
    encoded = _tokenize_example(tokenizer, chain["prefix_messages"], chain["target_final"])
    # Structural assert: every prefix label is -100; every target label is supervised.
    report = {
        "episode_id": chain["episode_id"],
        "n_prefix": encoded["n_prefix"],
        "n_target": encoded["n_target"],
        "n_unmasked_labels": sum(1 for x in encoded["labels"] if x != -100),
        "prefix_fully_masked": all(x == -100 for x in encoded["labels"][: encoded["n_prefix"]]),
        "target_fully_supervised": all(
            x != -100 for x in encoded["labels"][encoded["n_prefix"] :]
        ),
        "unmasked_equals_target_len": sum(1 for x in encoded["labels"] if x != -100)
        == encoded["n_target"],
        "target_text": chain["target_final"],
    }
    if not (
        report["prefix_fully_masked"]
        and report["target_fully_supervised"]
        and report["unmasked_equals_target_len"]
    ):
        raise RuntimeError(f"loss mask verification failed: {report}")
    return {**report, "encoded": encoded}


def verify_loss_with_model(model, tokenizer, chain: dict[str, Any]) -> dict[str, Any]:
    import torch
    from torch.nn import CrossEntropyLoss

    structural = verify_loss_mask(tokenizer, chain)
    encoded = structural.pop("encoded")
    device = next(model.parameters()).device
    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
    labels = torch.tensor([encoded["labels"]], dtype=torch.long, device=device)
    model.train()
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    # Manual CE on unmasked positions only.
    logits = outputs.logits[:, :-1, :].float()
    lab = labels[:, 1:]
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_lab = lab.reshape(-1)
    manual = CrossEntropyLoss(ignore_index=-100)(flat_logits, flat_lab)
    report = {
        **structural,
        "model_loss": float(loss.detach().item()),
        "manual_masked_ce": float(manual.detach().item()),
        "losses_agree": abs(float(loss.detach().item()) - float(manual.detach().item())) < 1e-3,
    }
    if not report["losses_agree"]:
        raise RuntimeError(f"model loss disagrees with masked CE: {report}")
    print(
        f"loss-mask ok episode={chain['episode_id']} n_target={report['n_target']} "
        f"loss={report['model_loss']:.6f}",
        flush=True,
    )
    return report


def train_sft(tokenizer, model, train_chains: list[dict[str, Any]], started: float) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader

    from agenttune.agentic.rollout_engines.rollout_factory import ensure_lora_targets

    encoded_rows = []
    for chain in train_chains:
        enc = _tokenize_example(tokenizer, chain["prefix_messages"], chain["target_final"])
        encoded_rows.append(
            {
                "input_ids": enc["input_ids"],
                "labels": enc["labels"],
                "episode_id": chain["episode_id"],
                "n_prefix": enc["n_prefix"],
                "n_target": enc["n_target"],
            }
        )
    # Pad in collator.
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def collate(batch):
        width = max(len(item["input_ids"]) for item in batch)
        input_ids = []
        labels = []
        attention = []
        for item in batch:
            pad = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad)
            labels.append(item["labels"] + [-100] * pad)
            attention.append([1] * len(item["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }

    lora = _lora_dict()
    peft_model = get_peft_model(
        model, ensure_lora_targets(LoraConfig(**lora), model)
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    peft_model.to(device)
    peft_model.train()
    opt = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad], lr=LEARNING_RATE
    )
    loader = DataLoader(encoded_rows, batch_size=1, shuffle=True, collate_fn=collate)
    steps = 0
    history = []
    data_cycle = iter(loader)
    while steps < MAX_STEPS:
        if time.monotonic() - started >= TIME_LIMIT_S:
            break
        try:
            batch = next(data_cycle)
        except StopIteration:
            data_cycle = iter(loader)
            batch = next(data_cycle)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = peft_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = out.loss
        # Confirm this batch still only supervises non -100 labels.
        n_supervised = int((batch["labels"] != -100).sum().item())
        if n_supervised < 1:
            raise RuntimeError("batch has no supervised tokens")
        loss.backward()
        opt.step()
        opt.zero_grad()
        steps += 1
        history.append({"step": steps, "loss": float(loss.detach().item()), "n_supervised": n_supervised})
        print(f"sft step={steps}/{MAX_STEPS} loss={history[-1]['loss']:.6f}", flush=True)

    adapter = OUT / "adapter_SFT"
    adapter.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    return {
        "method": "SFT",
        "optimizer_steps": steps,
        "max_optimizer_steps": MAX_STEPS,
        "learning_rate": LEARNING_RATE,
        "history": history,
        "adapter": "artifacts/task-5-11/adapter_SFT",
        "n_train_chains": len(train_chains),
        "model": peft_model,
        "device": device,
    }


def _looks_like_tool_call(text: str | None) -> bool:
    if not text:
        return False
    body = text.strip()
    # Strip trailing special tokens for the check.
    body = re.sub(r"(?:<\|END_RESPONSE\|>|<EOS_TOKEN>|<\|END_OF_TURN_TOKEN\|>)+$", "", body).strip()
    if not body.startswith("["):
        return '"tool_name"' in body
    try:
        value = json.loads(body.split("<|")[0].strip())
    except json.JSONDecodeError:
        return '"tool_name"' in body
    return isinstance(value, list) and value and isinstance(value[0], dict) and (
        "tool_name" in value[0] or "name" in value[0]
    )


def _clean_final(text: str | None) -> str:
    if text is None:
        return ""
    body = text.strip()
    for token in (
        "<|END_RESPONSE|>",
        "<EOS_TOKEN>",
        "<|END_OF_TURN_TOKEN|>",
        "<|START_RESPONSE|>",
    ):
        if body.endswith(token):
            body = body[: -len(token)].rstrip()
    return body


def generate_on_prefix(model, tokenizer, prefix_messages: list[dict], *, seed: int) -> str:
    import torch

    device = next(model.parameters()).device
    prompt = _render_prefix_text(tokenizer, prefix_messages)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    model.eval()
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=128,
            do_sample=True,
            temperature=PROBE_TEMPERATURE,
            top_p=PROBE_TOP_P,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new = out[0, ids.shape[1] :].tolist()
    return tokenizer.decode(new, skip_special_tokens=False)


def run_probe(base_model, adapter_model, tokenizer, probe_chains: list[dict]) -> dict[str, Any]:
    pairs = []
    for i, chain in enumerate(probe_chains):
        seed = PROBE_SEED_BASE + i
        base_text = generate_on_prefix(
            base_model, tokenizer, chain["prefix_messages"], seed=seed
        )
        ad_text = generate_on_prefix(
            adapter_model, tokenizer, chain["prefix_messages"], seed=seed
        )
        amount = chain["amount_cents"]
        base_clean = _clean_final(base_text)
        ad_clean = _clean_final(ad_text)
        base_ok = answer_matches(base_clean, amount) or base_clean == final_answer_text(amount)
        ad_ok = answer_matches(ad_clean, amount) or ad_clean == final_answer_text(amount)
        # answer_matches already strips ending tokens; also accept exact gold.
        if not base_ok:
            base_ok = base_clean == final_answer_text(amount)
        if not ad_ok:
            ad_ok = ad_clean == final_answer_text(amount)
        # Prefer harness helper after stripping.
        base_ok = answer_matches(base_text, amount)
        ad_ok = answer_matches(ad_text, amount)
        pairs.append(
            {
                "episode_id": chain["episode_id"],
                "order_id": chain["order_id"],
                "amount_cents": amount,
                "seed": seed,
                "base_final": base_clean,
                "adapter_final": ad_clean,
                "base_final_correct": base_ok,
                "adapter_final_correct": ad_ok,
                "base_extra_tool_call": _looks_like_tool_call(base_text),
                "adapter_extra_tool_call": _looks_like_tool_call(ad_text),
                "fixed": (not base_ok) and ad_ok,
                "broken": base_ok and (not ad_ok),
            }
        )
        print(
            f"probe {chain['episode_id']} base_ok={base_ok} adapter_ok={ad_ok} "
            f"fixed={(not base_ok) and ad_ok} broken={base_ok and (not ad_ok)}",
            flush=True,
        )
    n_fixed = sum(1 for p in pairs if p["fixed"])
    n_broken = sum(1 for p in pairs if p["broken"])
    n_base_correct = sum(1 for p in pairs if p["base_final_correct"])
    gate = {
        "n_fixed": n_fixed,
        "n_broken": n_broken,
        "n_base_correct": n_base_correct,
        "min_fixed_required": 2,
        "passed": n_fixed >= 2 and n_broken == 0,
    }
    return {"method": "SFT", "pairs": pairs, "gate": gate}


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
    final = last.get("final_answer")
    if last.get("task_success_harness") and not last.get("task_success_answer") and _looks_like_tool_call(final):
        later = max(later, 1)
    return later


def run_full_eval(adapter_dir: Path, started: float) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import agenttune.agentic.rollout_engines.rollout_factory as factory
    from agenttune.agentic.rollout_engines.transformers_engine import TransformersRolloutEngine

    out = OUT / "eval_full"
    out.mkdir(parents=True, exist_ok=True)

    def _eval_one(label: str, model) -> list[dict]:
        _bind_output(out / label)
        engine = TransformersRolloutEngine(model, tokenizer)
        _guards, schema_list = make_guarded_tools(
            RefundEpisode("schema-probe", "ORD-8101", "CUS-C01", 1710, config.SEED)
        )
        probe = _probe(tokenizer, schema_list)
        rows: list[dict] = []
        for episode_id, order_id, customer_id, amount in EVAL_TASKS:
            for seed in GENERATION_SEEDS:
                if time.monotonic() - started >= TIME_LIMIT_S:
                    break
                print(f"full eval {label} {episode_id}-g{seed}", flush=True)
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
                    prompt_text=sft_user_prompt(episode_id, order_id, customer_id),
                    prompt_protocol="one_action_per_turn",
                )
                if limits:
                    print("limits", limits, flush=True)
                rows.extend(ep_rows)
                _dump(out / label / "model_episodes.jsonl", rows)
            if time.monotonic() - started >= TIME_LIMIT_S:
                break
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[row["rollout_id"]].append(row)
        pairs = []
        for episode_id, order_id, customer_id, amount in EVAL_TASKS:
            for seed in GENERATION_SEEDS:
                rid = f"{episode_id}-g{seed}"
                if rid not in grouped:
                    continue
                steps = sorted(grouped[rid], key=lambda item: item["step_id"])
                last = steps[-1]
                pairs.append(
                    {
                        "rollout_id": rid,
                        "episode_id": episode_id,
                        "generation_seed": seed,
                        "amount_cents": amount,
                        "task_success": bool(last.get("task_success")),
                        "confirmed": bool(last.get("task_success_harness")),
                        "correct_final": bool(last.get("task_success_answer")),
                        "n_calls_after_confirmed_submit": _calls_after_submit(steps),
                        "final_answer": last.get("final_answer"),
                    }
                )
        return pairs

    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID, revision=config.MODEL_REVISION, local_files_only=True
    )
    # Base
    base = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        revision=config.MODEL_REVISION,
        dtype=torch.float16,
        local_files_only=True,
    )
    base.eval()
    if torch.backends.mps.is_available():
        base.to("mps")
    base_pairs = _eval_one("base", base)
    del base
    _release()

    if time.monotonic() - started >= TIME_LIMIT_S:
        return {
            "method": "SFT",
            "incomplete": True,
            "base_pairs": base_pairs,
            "adapter_pairs": [],
            "reason": "time_limit_after_base_eval",
        }

    # Adapter
    base2 = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        revision=config.MODEL_REVISION,
        dtype=torch.float16,
        local_files_only=True,
    )
    adapted = PeftModel.from_pretrained(base2, str(adapter_dir))
    adapted.eval()
    if torch.backends.mps.is_available():
        adapted.to("mps")
    ad_pairs = _eval_one("adapter", adapted)
    del adapted, base2
    _release()

    left = {p["rollout_id"]: p for p in base_pairs}
    right = {p["rollout_id"]: p for p in ad_pairs}
    transitions = []
    for rid in sorted(set(left) & set(right)):
        a, b = left[rid], right[rid]
        transitions.append(
            {
                "rollout_id": rid,
                "base": a,
                "adapter": b,
                "changed": {
                    "task_success": a["task_success"] != b["task_success"],
                    "confirmed": a["confirmed"] != b["confirmed"],
                    "correct_final": a["correct_final"] != b["correct_final"],
                    "n_calls_after_confirmed_submit": a["n_calls_after_confirmed_submit"]
                    != b["n_calls_after_confirmed_submit"],
                },
            }
        )

    def _agg(pairs: list[dict]) -> dict[str, Any]:
        n = len(pairs)
        return {
            "n": n,
            "full_success": sum(1 for p in pairs if p["task_success"]),
            "confirmed_returns": sum(1 for p in pairs if p["confirmed"]),
            "calls_after_confirmed_submit": sum(
                int(p["n_calls_after_confirmed_submit"]) for p in pairs
            ),
        }

    comparison = {
        "method": "SFT",
        "base": _agg(base_pairs),
        "adapter": _agg(ad_pairs),
        "transitions": transitions,
        "changed_rollouts": [t for t in transitions if any(t["changed"].values())],
    }
    _write_json(out / "comparison.json", comparison)
    _write_json(out / "base_pairs.json", {"pairs": base_pairs})
    _write_json(out / "adapter_pairs.json", {"pairs": ad_pairs})
    return comparison


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    pinned = write_pinned_config()
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID, revision=config.MODEL_REVISION, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("5.11 method=SFT collecting scripted successful chains", flush=True)
    chains = collect_chains(tokenizer)
    print(
        f"collected train={len(chains['train'])} probe={len(chains['probe'])}",
        flush=True,
    )

    # Structural mask check before loading weights into train mode.
    mask_report = verify_loss_mask(tokenizer, chains["train"][0])
    _write_json(
        OUT / "loss_mask_check.json",
        {k: v for k, v in mask_report.items() if k != "encoded"},
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
    if torch.backends.mps.is_available():
        base.to("mps")

    model_mask = verify_loss_with_model(base, tokenizer, chains["train"][0])
    _write_json(
        OUT / "loss_mask_check.json",
        {k: v for k, v in model_mask.items() if k != "encoded"},
    )

    if time.monotonic() - started >= TIME_LIMIT_S:
        summary = {
            "method": "SFT",
            "stop_reason": "time_limit_before_train",
            "elapsed_s": time.monotonic() - started,
        }
        _write_json(OUT / "summary.json", summary)
        return 0

    train_info = train_sft(tokenizer, base, chains["train"], started)
    peft_model = train_info.pop("model")
    device = train_info.pop("device")
    _write_json(OUT / "train_sft.json", train_info)

    # Fresh base for probe comparison.
    del peft_model
    _release()
    base_probe = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        revision=config.MODEL_REVISION,
        dtype=torch.float16,
        local_files_only=True,
    )
    base_probe.eval()
    if torch.backends.mps.is_available():
        base_probe.to("mps")
    adapted = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            config.MODEL_ID,
            revision=config.MODEL_REVISION,
            dtype=torch.float16,
            local_files_only=True,
        ),
        str(OUT / "adapter_SFT"),
    )
    adapted.eval()
    if torch.backends.mps.is_available():
        adapted.to("mps")

    probe = run_probe(base_probe, adapted, tokenizer, chains["probe"])
    _write_json(OUT / "probe_comparison.json", probe)
    del base_probe, adapted
    _release()

    full = None
    stop_reason = "probe_failed"
    if probe["gate"]["passed"]:
        if time.monotonic() - started >= TIME_LIMIT_S:
            stop_reason = "probe_passed_time_limit_before_full_eval"
        else:
            print("probe gate passed; running full eval on e0–e3", flush=True)
            full = run_full_eval(OUT / "adapter_SFT", started)
            stop_reason = (
                "full_eval_incomplete"
                if full.get("incomplete")
                else "full_eval_completed"
            )
    else:
        print(
            f"probe gate failed fixed={probe['gate']['n_fixed']} "
            f"broken={probe['gate']['n_broken']}; skip full eval",
            flush=True,
        )
        stop_reason = "probe_failed_skip_full_eval"

    pinned["full_eval_ran"] = bool(full) and not full.get("incomplete")
    pinned["stop_reason"] = stop_reason
    pinned["optimizer_steps_completed"] = train_info["optimizer_steps"]
    _write_json(OUT / "pinned_config.json", pinned)

    summary = {
        "kind": "sft_final_answer_pilot",
        "method": "SFT",
        "command": COMMAND,
        "git_head": _git_head(),
        "versions": _package_versions(),
        "elapsed_s": time.monotonic() - started,
        "time_limit_s": TIME_LIMIT_S,
        "stop_reason": stop_reason,
        "pinned_config": pinned,
        "loss_mask_check": {k: v for k, v in model_mask.items() if k != "encoded"},
        "train": train_info,
        "probe": probe,
        "full_eval": full,
        "adapter": "artifacts/task-5-11/adapter_SFT",
    }
    _write_json(OUT / "summary.json", summary)
    _write_report(summary)
    print(
        json.dumps(
            {
                "method": "SFT",
                "stop_reason": stop_reason,
                "optimizer_steps": train_info["optimizer_steps"],
                "probe_gate": probe["gate"],
                "full_eval": None
                if full is None
                else {
                    "base": full.get("base"),
                    "adapter": full.get("adapter"),
                    "n_changed": len(full.get("changed_rollouts") or []),
                },
                "elapsed_s": summary["elapsed_s"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def _write_report(summary: dict[str, Any]) -> None:
    probe = summary.get("probe") or {}
    gate = probe.get("gate") or {}
    pairs = probe.get("pairs") or []
    full = summary.get("full_eval")
    lines = [
        "# Task 5.11 — SFT on gold final answers",
        "",
        "**Method: SFT (supervised fine-tuning).** Not GRPO. The adapter is trained",
        "on gold `REFUND_SUBMITTED amount_cents=<n>` lines taken from harness",
        "`submit_refund` amounts, with the entire dialogue prefix masked out of the loss.",
        "",
        f"- stop_reason: `{summary.get('stop_reason')}`",
        f"- optimizer_steps: {summary.get('train', {}).get('optimizer_steps')}",
        f"- learning_rate: {LEARNING_RATE}",
        f"- adapter: `{summary.get('adapter')}`",
        f"- elapsed_s: {summary.get('elapsed_s'):.1f}",
        "",
        "## Loss mask check",
        "",
        "```json",
        json.dumps(summary.get("loss_mask_check"), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Probe gate (8 held-out post-submit prefixes)",
        "",
        f"- fixed={gate.get('n_fixed')} broken={gate.get('n_broken')} "
        f"base_correct={gate.get('n_base_correct')} passed={gate.get('passed')}",
        "",
        "| episode | amount | base_ok | adapter_ok | fixed | broken | base_tool | adapter_tool |",
        "|---|---:|---|---|---|---|---|---|",
    ]
    for p in pairs:
        lines.append(
            f"| {p['episode_id']} | {p['amount_cents']} | {p['base_final_correct']} | "
            f"{p['adapter_final_correct']} | {p['fixed']} | {p['broken']} | "
            f"{p['base_extra_tool_call']} | {p['adapter_extra_tool_call']} |"
        )
    lines.extend(["", "## Full episode eval", ""])
    if not full:
        lines.append("Skipped (probe gate failed or time limit).")
    else:
        lines.append("```json")
        lines.append(
            json.dumps(
                {
                    "base": full.get("base"),
                    "adapter": full.get("adapter"),
                    "n_transitions": len(full.get("transitions") or []),
                    "n_changed": len(full.get("changed_rollouts") or []),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        lines.append("```")
        lines.append("")
        lines.append("| rollout | base_success | adapter_success | base_confirmed | adapter_confirmed | base_after | adapter_after |")
        lines.append("|---|---|---|---|---|---:|---:|")
        for t in full.get("transitions") or []:
            a, b = t["base"], t["adapter"]
            lines.append(
                f"| {t['rollout_id']} | {a['task_success']} | {b['task_success']} | "
                f"{a['confirmed']} | {b['confirmed']} | "
                f"{a['n_calls_after_confirmed_submit']} | {b['n_calls_after_confirmed_submit']} |"
            )
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
