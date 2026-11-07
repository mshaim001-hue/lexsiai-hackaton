"""One command: .venv/bin/python run_contract.py

Fixtures and model episodes are written to different JSONL files.
A fixture success is never counted as a model success.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

from contract import config
from contract.classify import classify_parse, schema_map
from contract.execute import (
    PATH,
    make_guarded_tools,
    string_after_structure,
    validation_then_execution,
)
from contract.harness import (
    ONE_ACTION_SYSTEM_PROMPT,
    ONE_ACTION_TOOL_RESULT_FORMAT,
    ONE_ACTION_TOOLS_FALLBACK_PROMPT,
    SYSTEM_PROMPT,
    TOOL_RESULT_FORMAT,
    TOOLS_FALLBACK_PROMPT,
    RefundEpisode,
    answer_matches,
    final_answer_text,
    one_action_user_prompt,
    user_prompt,
)

ARTIFACTS = ROOT / "artifacts" / "contract"
FIXTURE_PATH = ARTIFACTS / "fixtures.jsonl"
MODEL_PATH = ARTIFACTS / "model_episodes.jsonl"
SUMMARY_PATH = ARTIFACTS / "summary.json"
MANIFEST_PATH = ARTIFACTS / "manifest.json"
ERROR_PATH = ARTIFACTS / "model_error.txt"
PINNED_MAX_NEW_TOKENS = config.MAX_NEW_TOKENS
FROZEN_128 = ROOT / "artifacts" / "contract-128"
TASK_DIR = ROOT / "artifacts" / "task-1-2"

LAUNCH_COMMAND = ".venv/bin/python run_contract.py"

MAX_STEPS_COUNTS = (
    "In the pinned AgentTune commit, _execute_trajectory counts max_steps as "
    "iterations of `while tool_calls and iteration_num < max_steps`. Each iteration "
    "executes calls that were already parsed. The first generation happens before "
    "the loop. After an iteration, iteration_num is incremented and the loop breaks "
    "when iteration_num >= max_steps, before another in-loop generation. If the last "
    "message is a tool result, one final generation runs outside that loop and is not "
    "part of max_steps. force_final_answer is false, so it adds no generation. "
    "max_steps=4 therefore holds four sequential tool calls plus the final answer. "
    "The limit was not changed."
)

PARSE_RULE = (
    "The native parser is unchanged. parsed means it returned one or more calls; "
    "no_tool_call means it returned none. Diagnostic categories are separate: "
    "malformed_syntax is a finished attempt that does not parse into a call; "
    "omitted_by_native_parser is a syntactically complete call the native parser "
    "did not return and is not a syntax error; truncated_fragment is an unfinished "
    "attempt. schema_invalid is a validation result. generation_truncated is an "
    "independent generation flag and may occur together with any category."
)

N_INVALID_DEFINITION = (
    "Preliminary, pending an agreed penalty definition. "
    "N_invalid counts malformed_syntax fragments plus parsed calls whose "
    "validation is missing_name, unknown_tool, or schema_invalid. "
    "A complete call omitted by the native parser is not a model syntax error "
    "and is not counted. truncated_fragment, generation_truncated, no_tool_call, "
    "tool_error, exception, order_or_token, and elapsed_limit_exceeded are reported separately "
    "and are not counted. C is preliminary."
)

PARSER_FORMAT = (
    "Pinned AgentTune _parse_cohere_json returns the first JSON value that is "
    "a tool_name object or a list made entirely of tool_name objects. A complete "
    "list is one value, so every object in it is returned. A bare object is also "
    "accepted. A list such as [2, 3] is not a call. A truncated list does not "
    "decode, so the scanner returns only the first later complete object and "
    "drops later complete siblings. The fallback prompt in the same pinned file "
    "tells the model to reply with only a JSON list, one object per call, and to "
    "use a list even for one call."
)


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT / "vendor" / "AgentTune"), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _hardware() -> dict:
    def sh(*args: str) -> str:
        return subprocess.check_output(args, text=True).strip()

    memory = int(sh("sysctl", "-n", "hw.memsize"))
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "processor": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "cpu_count": int(sh("sysctl", "-n", "hw.ncpu")),
        "memory_bytes": memory,
        "memory_gib": round(memory / (1024**3), 2),
    }


def _package_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    names = [
        "agenttune",
        "torch",
        "transformers",
        "tokenizers",
        "huggingface-hub",
        "trl",
        "accelerate",
        "peft",
        "bitsandbytes",
        "numpy",
        "safetensors",
    ]
    found = {}
    for name in names:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    return found


def _dump(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _rate(numerator: int, denominator: int):
    if denominator == 0:
        return {"numerator": numerator, "denominator": 0, "value": "N/A"}
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator,
    }


def _answer_ok(text: str | None, amount_cents: int) -> bool:
    return answer_matches(text, amount_cents)


def _diagnostic(step: dict) -> dict:
    diagnostic = step.get("diagnostic")
    return diagnostic if isinstance(diagnostic, dict) else {}


def _n_invalid_components(steps: list[dict]) -> dict:
    malformed_syntax = 0
    omitted_calls = 0
    truncated_fragments = 0
    schema_invalid = 0
    unknown_tool = 0
    missing_name = 0
    generation_truncated = 0
    for step in steps:
        diagnostic = _diagnostic(step)
        malformed_syntax += len(diagnostic.get("malformed_syntax") or [])
        omitted_calls += sum(
            len(item.get("dropped_by_full_parse") or [])
            for item in diagnostic.get("omitted_by_native_parser") or []
        )
        truncated_fragments += len(diagnostic.get("truncated_fragments") or [])
        generation_truncated += int(bool(step.get("generation_truncated")))
        for call in step.get("calls") or []:
            validation = call.get("validation")
            schema_invalid += validation == "schema_invalid"
            unknown_tool += validation == "unknown_tool"
            missing_name += validation == "missing_name"
    provisional = malformed_syntax + schema_invalid + unknown_tool + missing_name
    return {
        "malformed_syntax": malformed_syntax,
        "omitted_by_native_parser": omitted_calls,
        "schema_invalid": schema_invalid,
        "unknown_tool": unknown_tool,
        "missing_name": missing_name,
        "truncated_fragment": truncated_fragments,
        "generation_truncated": generation_truncated,
        "N_invalid_provisional": provisional,
        "C_status": "preliminary",
    }


def _n_invalid(steps: list[dict]) -> int:
    return _n_invalid_components(steps)["N_invalid_provisional"]


def _rewards(task_success: bool, n_invalid: int) -> dict:
    binary = 1 if task_success else 0
    return {
        "B": binary,
        "C": binary - config.LAMBDA * n_invalid,
        "C_status": "preliminary",
        "lambda": config.LAMBDA,
        "N_invalid": n_invalid,
        "N_invalid_status": "preliminary",
        "task_success": task_success,
    }


def _hidden_hits(text: str | None, episode: RefundEpisode) -> list[str]:
    if not text:
        return []
    return [item for item in episode.hidden_strings() if item in text]


class _RecordingEngine:
    """Records the prompt the engine renders and the raw completion it returns."""

    def __init__(self, inner, records: list[dict], tokenizer):
        self._inner = inner
        self._records = records
        self._tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _render(self, conversation, tools) -> str | None:
        try:
            rendered = self._tokenizer.apply_chat_template(
                conversation,
                tools=tools or None,
                add_generation_prompt=True,
                tokenize=False,
            )
        except TypeError:
            rendered = self._tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
        if isinstance(rendered, list):
            rendered = rendered[0] if rendered else ""
        return rendered

    def generate(self, prompts, tools, gen_cfg):
        conversation = copy.deepcopy(prompts)
        rendered = self._render(conversation, tools)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = self._inner.generate(prompts, tools, gen_cfg)
        raws = result.get("completions_raw") or []
        ids = (result.get("metadata") or {}).get("completion_ids") or [[]]
        new_count = len(ids[0]) if ids and isinstance(ids[0], list) else 0
        max_new = int(gen_cfg.get("max_new_tokens", config.MAX_NEW_TOKENS))
        self._records.append(
            {
                "messages": conversation,
                "rendered_prompt": rendered,
                "tools_passed": copy.deepcopy(tools),
                "completions_raw": raws[0] if raws else "",
                "new_token_count": new_count,
                "generation_truncated": new_count >= max_new and new_count > 0,
                "prompt_len": (result.get("metadata") or {}).get("prompt_len"),
                "prompt_truncated": any("Truncat" in str(item.message) for item in caught),
                "warnings": [str(item.message) for item in caught],
                "saved_at": "generate_return",
            }
        )
        return result


def _install_extract_hook(factory, sink: list[dict]):
    original = factory._extract_tool_calls

    def wrapped(completion_msg, raw_text):
        # This is the text the loop passes in, before the parser strips
        # think tags or any later repair. It is not read from Step.thought.
        sink.append({"raw_output": raw_text})
        parsed = original(completion_msg, raw_text)
        sink[-1]["parser_result"] = parsed
        return parsed

    factory._extract_tool_calls = wrapped
    return original


def _tool_names_visible(rendered: str | None, schemas: list[dict]) -> dict:
    text = rendered or ""
    names = {}
    params = {}
    for schema in schemas:
        fn = schema.get("function") if isinstance(schema.get("function"), dict) else schema
        name = fn.get("name")
        if name:
            names[name] = name in text
        for pname in (fn.get("parameters") or {}).get("properties") or {}:
            params[pname] = pname in text
    return {"tool_names": names, "parameter_names": params}


def _probe(tokenizer, schemas: list[dict]) -> dict:
    from agenttune.agentic.rollout_engines.rollout_factory import _template_renders_tools

    probe = [{"role": "user", "content": "hi"}]
    with_tools = tokenizer.apply_chat_template(
        probe, tools=schemas, add_generation_prompt=True, tokenize=False
    )
    without = tokenizer.apply_chat_template(probe, add_generation_prompt=True, tokenize=False)
    return {
        "template_renders_tools": bool(_template_renders_tools(tokenizer, schemas)),
        "probe_with_tools_kw": with_tools,
        "probe_without_tools_kw": without,
        "probe_changed": with_tools != without,
    }


def _consume_guard_records(parsed, tool_names: set[str], episode, cursor: list[int]) -> list[dict]:
    """Read records the guard already stored. This does not call a tool."""
    records = []
    for call in parsed or []:
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = fn.get("name") or call.get("name") or ""
        if name not in tool_names:
            validation = "missing_name" if not name else "unknown_tool"
            records.append(
                string_after_structure(
                    {
                        "path": PATH,
                        "tool": name,
                        "validation": validation,
                        "arguments": None,
                        "problems": [validation],
                        "execution": "not_run",
                        "order_or_token": False,
                        "function_entered": False,
                        "structured_result": None,
                        "call": call,
                    }
                )
            )
            continue
        record = string_after_structure(episode.structured_results[cursor[0]])
        cursor[0] += 1
        record["call"] = call
        records.append(record)
    return records


def _base_step(source: str, episode: RefundEpisode, step_id: int) -> dict:
    return {
        "source": source,
        "episode_id": episode.episode_id,
        "step_id": step_id,
        "order_id": episode.order_id,
        "customer_id": episode.customer_id,
        "messages": None,
        "rendered_prompt": None,
        "tools_passed_to_generate": None,
        "template_renders_tools": None,
        "probe_with_tools_kw": None,
        "probe_without_tools_kw": None,
        "tool_names_in_rendered_prompt": None,
        "parameter_names_in_rendered_prompt": None,
        "hidden_strings_in_rendered_prompt": [],
        "raw_output": None,
        "raw_output_source": None,
        "loop_step_thought": None,
        "parser_invoked_by_loop": False,
        "parser_result": None,
        "diagnostic": None,
        "diagnostic_executed": False,
        "function_entries": [],
        "parse_status": None,
        "malformed_syntax": [],
        "omitted_by_native_parser": [],
        "truncated_fragments": [],
        "calls": [],
        "generation_truncated": False,
        "prompt_truncated": False,
        "new_token_count": None,
        "stop": "not_generated",
        "stop_reason": None,
        "round_limit_reached": False,
        "budget_exhausted": False,
        "pending_unexecuted_calls": [],
        "pending_unexecuted_reason": None,
        "elapsed_limit_exceeded": False,
        "timeout_enforced": False,
        "elapsed_s": None,
        "final_answer": None,
        "task_success_harness": False,
        "task_success_answer": False,
        "task_success": False,
    }


def _finish(steps: list[dict], episode: RefundEpisode, final_answer: str | None, elapsed: float | None) -> None:
    harness_ok = bool(episode.submitted)
    answer_ok = _answer_ok(final_answer, episode.amount_cents)
    # Measured after the episode returns. Nothing is interrupted when this is true.
    elapsed_limit_exceeded = elapsed is not None and elapsed > config.EPISODE_TIMEOUT_S
    for step in steps:
        step["final_answer"] = final_answer
        step["task_success_harness"] = harness_ok
        step["task_success_answer"] = answer_ok
        step["task_success"] = harness_ok and answer_ok
        step["elapsed_s"] = elapsed
        step["elapsed_limit_exceeded"] = elapsed_limit_exceeded
        step["timeout_enforced"] = False


def _summarize(steps: list[dict], label: str) -> dict:
    episodes: dict[str, list[dict]] = {}
    for step in steps:
        episodes.setdefault(step["episode_id"], []).append(step)
    n_episodes = len(episodes)
    parsed_calls = 0
    malformed = 0
    schema_invalid = 0
    unknown_tool = 0
    missing_name = 0
    tool_error = 0
    exception = 0
    order_or_token = 0
    truncated_steps = 0
    elapsed_limit_exceeded_episodes = 0
    harness_ok = 0
    answer_ok = 0
    both_ok = 0
    no_parsed_call = 0
    silent = 0
    rewards = []
    for episode_id, ep_steps in episodes.items():
        calls = [call for step in ep_steps for call in step.get("calls") or []]
        parsed_calls += len(calls)
        malformed += sum(len(_diagnostic(step).get("malformed_syntax") or []) for step in ep_steps)
        schema_invalid += sum(call.get("validation") == "schema_invalid" for call in calls)
        unknown_tool += sum(call.get("validation") == "unknown_tool" for call in calls)
        missing_name += sum(call.get("validation") == "missing_name" for call in calls)
        tool_error += sum(call.get("execution") == "tool_error" for call in calls)
        exception += sum(call.get("execution") == "exception" for call in calls)
        order_or_token += sum(bool(call.get("order_or_token")) for call in calls)
        truncated_steps += sum(bool(step.get("generation_truncated")) for step in ep_steps)
        head = ep_steps[0]
        if head.get("elapsed_limit_exceeded"):
            elapsed_limit_exceeded_episodes += 1
        if head.get("task_success_harness"):
            harness_ok += 1
        if head.get("task_success_answer"):
            answer_ok += 1
        if head.get("task_success"):
            both_ok += 1
        if not calls:
            no_parsed_call += 1
        attempts = sum(
            len(_diagnostic(step).get("malformed_syntax") or [])
            + len(_diagnostic(step).get("omitted_by_native_parser") or [])
            + len(_diagnostic(step).get("truncated_fragments") or [])
            for step in ep_steps
        )
        if not calls and attempts == 0 and all(
            step.get("parse_status") == "no_tool_call" for step in ep_steps
        ):
            silent += 1
        rewards.append(
            {
                "episode_id": episode_id,
                **_rewards(bool(head.get("task_success")), _n_invalid(ep_steps)),
            }
        )
    n_steps = len(steps)
    return {
        "split": label,
        "n_episodes": n_episodes,
        "n_steps": n_steps,
        "task_success_harness": _rate(harness_ok, n_episodes),
        "task_success_answer": _rate(answer_ok, n_episodes),
        "task_success": _rate(both_ok, n_episodes),
        "parsed_calls": parsed_calls,
        "malformed_syntax": malformed,
        "omitted_by_native_parser": sum(
            sum(
                len(item.get("dropped_by_full_parse") or [])
                for item in _diagnostic(step).get("omitted_by_native_parser") or []
            )
            for step in steps
        ),
        "truncated_fragments": sum(
            len(_diagnostic(step).get("truncated_fragments") or []) for step in steps
        ),
        "schema_invalid_calls": schema_invalid,
        "unknown_tool_calls": unknown_tool,
        "missing_name_calls": missing_name,
        "execution_errors": {
            "tool_error": tool_error,
            "exception": exception,
            "total": tool_error + exception,
        },
        "order_or_token": order_or_token,
        "calls_per_task": _rate(parsed_calls, n_episodes),
        "episodes_with_no_parsed_call": _rate(no_parsed_call, n_episodes),
        "episodes_with_no_attempt": _rate(silent, n_episodes),
        "schema_invalid_per_parsed_call": _rate(schema_invalid, parsed_calls),
        "generation_truncated_steps": _rate(truncated_steps, n_steps),
        "elapsed_limit_exceeded_episodes": _rate(elapsed_limit_exceeded_episodes, n_episodes),
        "timeout_enforced": False,
        "rewards": rewards,
        "N_invalid_components": _n_invalid_components(steps),
        "N_invalid_definition": N_INVALID_DEFINITION,
        "C_status": "preliminary",
    }


def _primary(seed: int) -> RefundEpisode:
    return RefundEpisode(
        episode_id="fixture-primary",
        order_id="ORD-1001",
        customer_id="CUS-501",
        amount_cents=1840,
        seed=seed,
    )


def _fixture_specs(seed: int) -> list[dict]:
    primary = _primary(seed)
    foreign = RefundEpisode(
        episode_id="fixture-foreign",
        order_id="ORD-9",
        customer_id="CUS-9",
        amount_cents=111,
        seed=seed,
    )
    correct = json.dumps(
        [{"tool_name": "lookup_order", "parameters": {"order_id": primary.order_id}}],
        ensure_ascii=False,
    )
    unknown = json.dumps(
        [{"tool_name": "delete_account", "parameters": {"order_id": primary.order_id}}],
        ensure_ascii=False,
    )
    bad_schema = json.dumps(
        [{"tool_name": "lookup_order", "parameters": {}}],
        ensure_ascii=False,
    )
    tool_error = json.dumps(
        [{"tool_name": "lookup_order", "parameters": {"order_id": "ORD-DOES-NOT-EXIST"}}],
        ensure_ascii=False,
    )
    order_bypass = json.dumps(
        [
            {
                "tool_name": "calculate_refund",
                "parameters": {
                    "order_context_token": "ord_forged",
                    "customer_validation_token": "cus_forged",
                },
            }
        ],
        ensure_ascii=False,
    )
    foreign_call = json.dumps(
        [
            {
                "tool_name": "calculate_refund",
                "parameters": {
                    "order_context_token": foreign.order_token,
                    "customer_validation_token": foreign.customer_token,
                },
            }
        ],
        ensure_ascii=False,
    )
    mixed = (
        correct
        + "\n<function=get_customer>{not json}</function>"
    )
    single_object = json.dumps(
        {"tool_name": "lookup_order", "parameters": {"order_id": primary.order_id}},
        ensure_ascii=False,
    )
    two_calls = [
        {"tool_name": "lookup_order", "parameters": {"order_id": primary.order_id}},
        {"tool_name": "get_customer", "parameters": {"customer_id": primary.customer_id}},
    ]
    two_call_list = json.dumps(two_calls, indent=2, ensure_ascii=False)
    cut_at = two_call_list.rfind(primary.customer_id)
    truncated_list = two_call_list[: cut_at + 3]
    return [
        {"episode_id": "f-correct", "case": "correct_call", "raw": correct},
        {"episode_id": "f-silence", "case": "silence", "raw": "I don't know the refund amount."},
        {
            "episode_id": "f-prose-json",
            "case": "non_call_json",
            "raw": "The only numbers I see are [2, 3].",
        },
        {
            "episode_id": "f-broken",
            "case": "broken_json",
            "raw": '<tool_call>{"tool_name": "lookup_order", "parameters": {</tool_call>',
        },
        {"episode_id": "f-unknown", "case": "unknown_tool", "raw": unknown},
        {"episode_id": "f-schema", "case": "schema_invalid", "raw": bad_schema},
        {"episode_id": "f-tool-error", "case": "tool_error", "raw": tool_error},
        {"episode_id": "f-order", "case": "order_bypass", "raw": order_bypass},
        {"episode_id": "f-foreign", "case": "foreign_token", "raw": foreign_call},
        {"episode_id": "f-mixed", "case": "mixed_valid_and_broken", "raw": mixed},
        {"episode_id": "f-parser-object", "case": "parser_single_object", "raw": single_object},
        {"episode_id": "f-parser-list", "case": "parser_two_call_list", "raw": two_call_list},
        {
            "episode_id": "f-parser-truncated",
            "case": "parser_truncated_list",
            "raw": truncated_list,
        },
        {
            "episode_id": "f-answer-negation",
            "case": "answer_negation",
            "raw": f"not {final_answer_text(primary.amount_cents)}",
            "final_answer": f"not {final_answer_text(primary.amount_cents)}",
            "submitted": True,
        },
        {
            "episode_id": "f-answer-both",
            "case": "answer_both_amounts",
            "raw": (
                f"{final_answer_text(primary.amount_cents)} "
                f"{final_answer_text(1500)}"
            ),
            "final_answer": (
                f"{final_answer_text(primary.amount_cents)} "
                f"{final_answer_text(1500)}"
            ),
            "submitted": True,
        },
        {
            "episode_id": "f-answer-wrong",
            "case": "answer_wrong_amount",
            "raw": final_answer_text(1500),
            "final_answer": final_answer_text(1500),
            "submitted": True,
        },
        {
            "episode_id": "f-answer-extra",
            "case": "answer_extra_text",
            "raw": f"{final_answer_text(primary.amount_cents)} The refund is complete.",
            "final_answer": f"{final_answer_text(primary.amount_cents)} The refund is complete.",
            "submitted": True,
        },
    ]


def _diagnose(raw: str, parsed, original, episode: RefundEpisode) -> dict:
    before = list(episode.function_entries)
    diagnosed = classify_parse(raw, parsed, original)
    if episode.function_entries != before:
        raise RuntimeError("diagnostic classification called a tool")
    return {
        "parse_status": diagnosed["parse_status"],
        "native_parse_status": diagnosed["native_parse_status"],
        "malformed_syntax": diagnosed["malformed_syntax"],
        "omitted_by_native_parser": diagnosed["omitted_by_native_parser"],
        "truncated_fragments": diagnosed["truncated_fragments"],
        "parsed_calls": diagnosed["parsed_calls"],
        "attempt_span_count": diagnosed["attempt_span_count"],
        "executed": False,
    }


def _generation_diagnosis(raw: str, parser_invoked: bool, parser_result, original, episode: RefundEpisode):
    """Keep a call the loop never parsed out of the omitted-by-parser bucket."""
    if parser_invoked:
        return _diagnose(raw, parser_result, original, episode), parser_result, [], None
    detected = list(original({"role": "assistant", "content": raw}, raw) or [])
    diagnostic = dict(_diagnose(raw, detected, original, episode))
    diagnostic["parsed_calls"] = []
    diagnostic["omitted_by_native_parser"] = []
    diagnostic["native_parse_status"] = "not_invoked"
    diagnostic["parse_status"] = "not_invoked" if detected else "no_tool_call"
    return diagnostic, None, detected, None


def _assign_budget(rows: list[dict]) -> None:
    rounds = sum(1 for step in rows if step.get("parser_invoked_by_loop") and step.get("parser_result"))
    exhausted = rounds >= config.MAX_STEPS
    reason = "max_steps" if exhausted else "ended_without_tool_call"
    for step in rows:
        step["stop_reason"] = reason
        step["round_limit_reached"] = exhausted
        # Alias kept so a full four-round success is not read as a failed stop.
        step["budget_exhausted"] = exhausted
    if not rows:
        return
    last = rows[-1]
    pending = last.get("pending_unexecuted_calls") or []
    if pending and not last.get("parser_invoked_by_loop"):
        if exhausted:
            last["pending_unexecuted_reason"] = (
                "stop_reason=max_steps. This generation is outside the tool loop, "
                "so the loop did not call the parser and did not execute the call."
            )
        else:
            last["pending_unexecuted_reason"] = (
                "The loop did not call the parser on this generation, so the call was not executed."
            )


def _direct_fixture_row(episode: RefundEpisode, spec: dict, original) -> dict:
    guards, _schemas = make_guarded_tools(episode)
    raw = spec["raw"]
    message = {"role": "assistant", "content": raw}
    parsed = original(message, raw)
    diagnostic = _diagnose(raw, parsed, original, episode)
    calls = validation_then_execution(parsed, guards)
    if spec.get("submitted"):
        episode.submitted = True
    step = _base_step("fixture", episode, 0)
    step.update(
        {
            "case": spec["case"],
            "raw_output": raw,
            "raw_output_source": "fixture_string_passed_to_parser",
            "parser_invoked_by_loop": False,
            "parser_result": parsed,
            "diagnostic": diagnostic,
            "diagnostic_executed": False,
            "parse_status": diagnostic["parse_status"],
            "malformed_syntax": diagnostic["malformed_syntax"],
            "omitted_by_native_parser": diagnostic["omitted_by_native_parser"],
            "truncated_fragments": diagnostic["truncated_fragments"],
            "attempt_span_count": diagnostic["attempt_span_count"],
            "calls": calls,
            "function_entries": list(episode.function_entries),
            "execution_path": PATH,
        }
    )
    _finish(steps=[step], episode=episode, final_answer=spec.get("final_answer"), elapsed=None)
    return step


class _ScriptedEngine:
    """Returns fixed completions and stores each one at generate() return."""

    def __init__(self, outputs: list[str]):
        self._outputs = list(outputs)
        self.records: list[dict] = []

    def _get_tokenizer(self):
        return None

    def generate(self, prompts, tools, gen_cfg):
        if not self._outputs:
            raise RuntimeError(
                f"extra generation after {len(self.records)} completions; "
                "the max_steps budget did not leave room for the scripted trajectory"
            )
        raw = self._outputs.pop(0)
        self.records.append(
            {
                "raw_output": raw,
                "messages": copy.deepcopy(prompts),
                "saved_at": "generate_return",
            }
        )
        return {
            "completions": [raw],
            "completions_raw": [raw],
            "completions_structured": [{"role": "assistant", "content": raw}],
            "logprobs": [None],
            "metadata": {"completion_ids": [[0]], "prompt_len": 0},
        }


def _call_json(name: str, parameters: dict) -> str:
    return json.dumps(
        [{"tool_name": name, "parameters": parameters}],
        ensure_ascii=False,
    )


def _scripted_outputs(episode: RefundEpisode, final_text: str) -> list[str]:
    return [
        _call_json("lookup_order", {"order_id": episode.order_id}),
        _call_json("get_customer", {"customer_id": episode.customer_id}),
        _call_json(
            "calculate_refund",
            {
                "order_context_token": episode.order_token,
                "customer_validation_token": episode.customer_token,
            },
        ),
        _call_json(
            "submit_refund",
            {
                "quote_token": episode.quote_token,
                "order_id": episode.order_id,
                "customer_id": episode.customer_id,
                "amount_cents": episode.amount_cents,
            },
        ),
        final_text,
    ]


def _budget_outputs(episode: RefundEpisode) -> list[str]:
    return [
        _call_json("lookup_order", {"order_id": episode.order_id}),
        _call_json("lookup_order", {"order_id": episode.order_id}),
        _call_json("get_customer", {"customer_id": episode.customer_id}),
        _call_json(
            "calculate_refund",
            {
                "order_context_token": episode.order_token,
                "customer_validation_token": episode.customer_token,
            },
        ),
        _call_json(
            "submit_refund",
            {
                "quote_token": episode.quote_token,
                "order_id": episode.order_id,
                "customer_id": episode.customer_id,
                "amount_cents": episode.amount_cents,
            },
        ),
    ]


def _extra_amount_call(episode: RefundEpisode) -> str:
    return json.dumps(
        [
            {
                "tool_name": "calculate_refund",
                "parameters": {
                    "order_context_token": episode.order_token,
                    "customer_validation_token": episode.customer_token,
                    "amount_cents": episode.amount_cents,
                },
            }
        ],
        ensure_ascii=False,
    )


def _run_scripted(factory, episode: RefundEpisode, outputs: list[str], case: str) -> list[dict]:
    from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn

    guards, _schemas = make_guarded_tools(episode)
    tool_names = {guard.__name__ for guard in guards}
    engine = _ScriptedEngine(outputs)
    extract_log: list[dict] = []
    original = factory._extract_tool_calls
    factory._extract_tool_calls = original
    _install_extract_hook(factory, extract_log)
    rollout = create_rollout_fn(
        rollout_engine=engine,
        tools=guards,
        max_steps=config.MAX_STEPS,
        system_prompt=SYSTEM_PROMPT,
        enable_thinking=config.ENABLE_THINKING,
        force_final_answer=config.FORCE_FINAL_ANSWER,
        force_action_on_stall=config.FORCE_ACTION_ON_STALL,
        context_length=config.CONTEXT_LENGTH,
        tools_fallback_prompt=TOOLS_FALLBACK_PROMPT,
        tool_result_format=TOOL_RESULT_FORMAT,
    )
    try:
        result = rollout(
            [user_prompt(episode.order_id, episode.customer_id)],
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            do_sample=config.DO_SAMPLE,
            max_length=config.CONTEXT_LENGTH,
        )
    finally:
        factory._extract_tool_calls = original
    final_answer = result["trajectories"][0].final_response
    cursor = [0]
    used = 0
    rows = []
    for step_id, generated in enumerate(engine.records):
        raw = generated["raw_output"]
        extract = None
        if used < len(extract_log) and extract_log[used]["raw_output"] == raw:
            extract = extract_log[used]
            used += 1
        if extract is not None:
            parsed = extract["parser_result"]
            parser_invoked = True
        else:
            parsed = None
            parser_invoked = False
        diagnostic, parsed, pending, _pending_reason = _generation_diagnosis(
            raw, parser_invoked, parsed, original, episode
        )
        calls = (
            _consume_guard_records(parsed, tool_names, episode, cursor)
            if parser_invoked
            else []
        )
        messages = generated["messages"]
        visible_text = json.dumps(messages, ensure_ascii=False, default=str)
        step = _base_step("fixture", episode, step_id)
        step.update(
            {
                "case": case,
                "messages": messages,
                "raw_output": raw,
                "raw_output_source": "generation_engine",
                "parser_invoked_by_loop": parser_invoked,
                "parser_result": parsed,
                "diagnostic": diagnostic,
                "diagnostic_executed": False,
                "parse_status": diagnostic["parse_status"],
                "malformed_syntax": diagnostic["malformed_syntax"],
                "omitted_by_native_parser": diagnostic["omitted_by_native_parser"],
                "truncated_fragments": diagnostic["truncated_fragments"],
                "attempt_span_count": diagnostic["attempt_span_count"],
                "calls": calls,
                "pending_unexecuted_calls": pending,
                "hidden_strings_in_rendered_prompt": _hidden_hits(visible_text, episode),
                "execution_path": PATH,
                "max_steps": config.MAX_STEPS,
            }
        )
        rows.append(step)
    if engine._outputs:
        raise RuntimeError(f"{case} left unused scripted generations: {len(engine._outputs)}")
    if cursor[0] != len(episode.structured_results):
        raise RuntimeError(
            f"{case} consumed {cursor[0]} guard records of {len(episode.structured_results)}"
        )
    for step in rows:
        step["function_entries"] = list(episode.function_entries)
    _assign_budget(rows)
    _finish(rows, episode, final_answer, elapsed=None)
    return rows


def run_fixtures(factory) -> list[dict]:
    original = factory._extract_tool_calls
    rows = [
        _direct_fixture_row(
            RefundEpisode(spec["episode_id"], "ORD-1001", "CUS-501", 1840, config.SEED),
            spec,
            original,
        )
        for spec in _fixture_specs(config.SEED)
    ]
    success = RefundEpisode("f-scripted-success", "ORD-1001", "CUS-501", 1840, config.SEED)
    wrong = RefundEpisode("f-scripted-wrong-answer", "ORD-1001", "CUS-501", 1840, config.SEED)
    schema_loop = RefundEpisode("f-schema-loop", "ORD-1001", "CUS-501", 1840, config.SEED)
    extra_direct = RefundEpisode("f-schema-extra", "ORD-1001", "CUS-501", 1840, config.SEED)
    extra_loop = RefundEpisode("f-schema-extra-loop", "ORD-1001", "CUS-501", 1840, config.SEED)
    budget = RefundEpisode("f-budget", "ORD-1001", "CUS-501", 1840, config.SEED)
    rows.append(
        _direct_fixture_row(
            extra_direct,
            {
                "episode_id": "f-schema-extra",
                "case": "schema_invalid_extra_amount",
                "raw": _extra_amount_call(extra_direct),
            },
            original,
        )
    )
    rows.extend(
        _run_scripted(
            factory,
            success,
            _scripted_outputs(
                success,
                f" {final_answer_text(success.amount_cents)}<|END_RESPONSE|><EOS_TOKEN> ",
            ),
            "scripted_success",
        )
    )
    rows.extend(
        _run_scripted(
            factory,
            wrong,
            _scripted_outputs(wrong, "The refund was submitted."),
            "scripted_wrong_answer",
        )
    )
    rows.extend(
        _run_scripted(
            factory,
            schema_loop,
            [
                '[{"tool_name": "lookup_order", "parameters": {}}]',
                "I cannot complete that call.",
            ],
            "schema_invalid_through_loop",
        )
    )
    rows.extend(
        _run_scripted(
            factory,
            extra_loop,
            [_extra_amount_call(extra_loop), "I cannot complete that call."],
            "schema_invalid_extra_through_loop",
        )
    )
    rows.extend(_run_scripted(factory, budget, _budget_outputs(budget), "budget_exhausted"))
    return rows


def _check_fixtures(rows: list[dict]) -> list[str]:
    problems = []
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["episode_id"], []).append(row)
    by_case = {steps[0]["case"]: steps for steps in grouped.values()}
    single_expected = {
        "correct_call": "parsed",
        "silence": "no_tool_call",
        "non_call_json": "no_tool_call",
        "broken_json": "malformed_syntax",
        "unknown_tool": "parsed",
        "schema_invalid": "parsed",
        "tool_error": "parsed",
        "order_bypass": "parsed",
        "foreign_token": "parsed",
        "mixed_valid_and_broken": "parsed",
        "parser_single_object": "parsed",
        "parser_two_call_list": "parsed",
        "parser_truncated_list": "parsed",
    }
    for case, expected in single_expected.items():
        row = by_case[case][0]
        if row["parse_status"] != expected:
            problems.append(f"{case}: parse_status {row['parse_status']} != {expected}")
        if row["diagnostic"]["executed"]:
            problems.append(f"{case}: diagnostic executed a call")
        if row["calls"] and row["calls"][0].get("path") != PATH:
            problems.append(f"{case}: execution did not use {PATH}")
    if not by_case["broken_json"][0]["malformed_syntax"]:
        problems.append("broken_json produced no malformed_syntax fragment")
    if by_case["broken_json"][0]["omitted_by_native_parser"] or by_case["broken_json"][0]["truncated_fragments"]:
        problems.append("broken_json was classified as omitted or truncated")
    mixed = by_case["mixed_valid_and_broken"][0]
    if not mixed["calls"] or not mixed["malformed_syntax"]:
        problems.append("mixed fixture hid either the valid call or the broken fragment")
    single = by_case["parser_single_object"][0]
    if len(single["parser_result"] or []) != 1 or single["omitted_by_native_parser"] or single["truncated_fragments"] or single["malformed_syntax"]:
        problems.append("single-object fixture was not accepted as one native call")
    if not single["calls"] or single["calls"][0]["execution"] != "success":
        problems.append("single-object fixture did not execute the parsed call")
    listed = by_case["parser_two_call_list"][0]
    if len(listed["parser_result"] or []) != 2 or listed["omitted_by_native_parser"] or listed["truncated_fragments"]:
        problems.append("complete two-call list was not returned in full")
    if [call["tool"] for call in listed["calls"]] != ["lookup_order", "get_customer"]:
        problems.append(f"complete two-call list executed {[call.get('tool') for call in listed['calls']]}")
    truncated = by_case["parser_truncated_list"][0]
    if len(truncated["parser_result"] or []) != 1:
        problems.append("truncated list native parser did not return only the first complete call")
    if truncated["malformed_syntax"]:
        problems.append("truncated list was labelled malformed_syntax")
    if not truncated["truncated_fragments"]:
        problems.append("truncated list produced no truncated_fragment")
    if truncated["generation_truncated"]:
        problems.append("fixture truncation was stored as generation_truncated")
    if truncated["calls"] and truncated["calls"][0].get("tool") != "lookup_order":
        problems.append("truncated list executed a call other than the native parser result")
    schema = by_case["schema_invalid"][0]
    schema_call = schema["calls"][0]
    if schema_call["validation"] != "schema_invalid" or schema_call["function_entered"]:
        problems.append("direct schema_invalid reached the tool function")
    if schema["function_entries"]:
        problems.append(f"direct schema_invalid entries: {schema['function_entries']}")
    loop_schema = by_case["schema_invalid_through_loop"]
    loop_calls = [call for step in loop_schema for call in step["calls"]]
    if (
        not loop_calls
        or loop_calls[0]["validation"] != "schema_invalid"
        or loop_calls[0]["function_entered"]
        or loop_calls[0]["path"] != PATH
        or loop_schema[-1]["function_entries"]
    ):
        problems.append("loop schema_invalid reached the tool function")
    if loop_schema[0]["raw_output_source"] != "generation_engine":
        problems.append("loop schema fixture did not save engine raw output")
    extra = by_case["schema_invalid_extra_amount"][0]
    extra_call = extra["calls"][0] if extra["calls"] else {}
    if (
        extra_call.get("validation") != "schema_invalid"
        or extra_call.get("execution") != "not_run"
        or extra_call.get("function_entered")
        or extra["function_entries"]
        or "unexpected amount_cents" not in extra_call.get("problems", [])
    ):
        problems.append("direct extra amount_cents was not rejected before the implementation")
    extra_loop_steps = by_case["schema_invalid_extra_through_loop"]
    extra_loop_calls = [call for step in extra_loop_steps for call in step["calls"]]
    if (
        not extra_loop_calls
        or extra_loop_calls[0]["validation"] != "schema_invalid"
        or extra_loop_calls[0]["execution"] != "not_run"
        or extra_loop_calls[0]["function_entered"]
        or extra_loop_calls[0]["path"] != PATH
        or extra_loop_steps[-1]["function_entries"]
        or "unexpected amount_cents" not in extra_loop_calls[0].get("problems", [])
    ):
        problems.append("loop extra amount_cents reached the tool function")
    _guards, published = make_guarded_tools(
        RefundEpisode("schema-check", "ORD-1001", "CUS-501", 1840, config.SEED)
    )
    if len(published) != 4:
        problems.append(f"expected four published schemas, got {len(published)}")
    for schema in published:
        function = schema.get("function") if isinstance(schema.get("function"), dict) else schema
        parameters = function.get("parameters") or {}
        if parameters.get("additionalProperties") is not False:
            problems.append(f"{function.get('name')} schema allows additional properties")
        guard = next(item for item in _guards if item.__name__ == function.get("name"))
        if guard.to_schema() is not schema:
            problems.append(f"{function.get('name')} prompt schema is not the validator schema")
    unknown = by_case["unknown_tool"][0]["calls"]
    if not unknown or unknown[0]["validation"] != "unknown_tool" or unknown[0]["function_entered"]:
        problems.append("unknown tool was called or mislabelled")
    tool_err = by_case["tool_error"][0]["calls"][0]
    if tool_err["execution"] != "tool_error" or not isinstance(tool_err["structured_result"], dict):
        problems.append("tool_error was not stored as a structured result")
    if not isinstance(tool_err["string_result"], str):
        problems.append("tool_error string form was not produced after the structured result")
    for case in ("order_bypass", "foreign_token"):
        calls = by_case[case][0]["calls"]
        if not calls or not calls[0]["order_or_token"]:
            problems.append(f"{case} was not labelled order_or_token")
    correct = by_case["correct_call"][0]
    if not correct["calls"] or correct["calls"][0]["execution"] != "success":
        problems.append("correct call did not execute")
    if correct["task_success"]:
        problems.append("a single valid fixture call was counted as task success")
    success = by_case["scripted_success"]
    if len(success) != 5:
        problems.append(f"scripted success generated {len(success)} times, expected 5")
    if success[-1]["function_entries"] != [
        "lookup_order",
        "get_customer",
        "calculate_refund",
        "submit_refund",
    ]:
        problems.append(f"scripted success entries: {success[-1]['function_entries']}")
    if not success[-1]["task_success"]:
        problems.append("scripted success did not reach task_success")
    if success[-1]["parser_invoked_by_loop"]:
        problems.append("scripted final generation was parsed by the loop instead of only saved")
    if any(not step["parser_invoked_by_loop"] for step in success[:-1]):
        problems.append("a scripted tool generation was missing the loop parser result")
    if success[-1]["raw_output_source"] != "generation_engine":
        problems.append("scripted final answer was not taken from the generation engine")
    if success[0]["hidden_strings_in_rendered_prompt"]:
        problems.append("scripted prompt contained hidden state")
    success_reward = _rewards(True if success[-1]["task_success"] else False, _n_invalid(success))
    if success_reward["B"] != 1 or success_reward["C"] != 1:
        problems.append(f"scripted success reward was {success_reward}")
    if success[-1]["function_entries"] != [
        "lookup_order",
        "get_customer",
        "calculate_refund",
        "submit_refund",
    ]:
        problems.append("scripted success repeated or skipped a tool")
    if success[-1].get("pending_unexecuted_calls"):
        problems.append("scripted success left a pending call")
    if not success[-1].get("round_limit_reached"):
        problems.append("scripted success did not record that the four rounds were used")
    if success[-1].get("timeout_enforced") is not False:
        problems.append("scripted success enforced a timeout")
    if "elapsed_s" not in success[-1] or "elapsed_limit_exceeded" not in success[-1]:
        problems.append("scripted success did not store elapsed fields")
    budget_steps = by_case["budget_exhausted"]
    if budget_steps[-1]["function_entries"] != [
        "lookup_order",
        "lookup_order",
        "get_customer",
        "calculate_refund",
    ]:
        problems.append(f"budget entries: {budget_steps[-1]['function_entries']}")
    if budget_steps[-1]["task_success_harness"] or budget_steps[-1]["task_success"]:
        problems.append("budget control submitted the refund")
    if budget_steps[-1]["stop_reason"] != "max_steps" or not budget_steps[-1]["budget_exhausted"]:
        problems.append(f"budget stop_reason was {budget_steps[-1]['stop_reason']}")
    if budget_steps[-1]["parser_invoked_by_loop"]:
        problems.append("budget final generation was parsed by the loop")
    pending_names = _call_names(budget_steps[-1]["pending_unexecuted_calls"])
    if pending_names != ["submit_refund"]:
        problems.append(f"budget pending calls were {pending_names}")
    if budget_steps[-1]["calls"] or budget_steps[-1]["omitted_by_native_parser"]:
        problems.append("budget final call was recorded as executed or omitted by the parser")
    if not budget_steps[-1]["pending_unexecuted_reason"]:
        problems.append("budget final call has no non-execution reason")
    if "submit_refund" in budget_steps[-1]["function_entries"]:
        problems.append("budget control executed submit_refund")
    wrong = by_case["scripted_wrong_answer"]
    if not wrong[-1]["task_success_harness"] or wrong[-1]["task_success_answer"] or wrong[-1]["task_success"]:
        problems.append("wrong-answer control did not keep harness success separate from task success")
    if not success[-1]["task_success_answer"] or success[-1]["final_answer"] is None:
        problems.append("scripted success final answer was not accepted")
    if not answer_matches(success[-1]["raw_output"], 1840):
        problems.append("scripted success raw answer was rejected")
    for case in (
        "answer_negation",
        "answer_both_amounts",
        "answer_wrong_amount",
        "answer_extra_text",
    ):
        row = by_case[case][0]
        if not row["task_success_harness"] or row["task_success_answer"] or row["task_success"]:
            problems.append(f"{case} did not stay task_success 0 with submitted true")
    return problems


def _arithmetic_pair(rows: list[dict]) -> dict:
    by_id = {}
    for row in rows:
        by_id.setdefault(row["episode_id"], []).append(row)
    silence = _rewards(bool(by_id["f-silence"][0]["task_success"]), _n_invalid(by_id["f-silence"]))
    broken = _rewards(bool(by_id["f-broken"][0]["task_success"]), _n_invalid(by_id["f-broken"]))
    return {
        "pair": ["f-silence", "f-broken"],
        "silence": silence,
        "malformed": broken,
        "B_equal": silence["B"] == broken["B"],
        "C_differ": silence["C"] != broken["C"],
        "passed": silence["B"] == broken["B"] and silence["C"] != broken["C"] and silence["task_success"] is False and broken["task_success"] is False,
    }


def _redact(text: str) -> str:
    text = re.sub(r"hf_[A-Za-z0-9]{8,}", "[redacted]", text)
    text = re.sub(r"(?i)bearer\s+\S+", "Bearer [redacted]", text)
    text = re.sub(r"(?i)(token=)[^&\s]+", r"\1[redacted]", text)
    return text


def _local_token_configured() -> bool:
    from huggingface_hub import get_token

    return bool(get_token())


def _snapshot_revision(path: str) -> str | None:
    parts = Path(path).parts
    if "snapshots" not in parts:
        return None
    return parts[parts.index("snapshots") + 1]


def _load_model():
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from agenttune.agentic.rollout_engines.transformers_engine import TransformersRolloutEngine

    config_path = hf_hub_download(
        config.MODEL_ID, "config.json", revision=config.MODEL_REVISION
    )
    tokenizer_path = hf_hub_download(
        config.MODEL_ID, "tokenizer.json", revision=config.MODEL_REVISION
    )
    config_revision = _snapshot_revision(config_path)
    tokenizer_revision = _snapshot_revision(tokenizer_path)
    if config_revision != config.MODEL_REVISION or tokenizer_revision != config.MODEL_REVISION:
        raise RuntimeError(
            "cached config.json or tokenizer.json is not the pinned revision "
            f"{config.MODEL_REVISION}; refusing to continue"
        )
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(config.SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if config.DTYPE == "float16" else None
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, revision=config.MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        revision=config.MODEL_REVISION,
        torch_dtype=dtype,
    )
    model.eval()
    if device != "cpu":
        model.to(device)
    engine = TransformersRolloutEngine(model, tokenizer)
    revision = {
        "model_id": config.MODEL_ID,
        "requested_revision": config.MODEL_REVISION,
        "config_json_revision": config_revision,
        "tokenizer_json_revision": tokenizer_revision,
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "device": str(device),
        "dtype": config.DTYPE,
        "quantization": config.QUANTIZATION,
        "distribution": "public_gated",
        "local_token_configured": _local_token_configured(),
        "config_json": "loaded",
        "tokenizer": "loaded",
    }
    return engine, tokenizer, revision


def run_model_episodes(factory, engine, tokenizer, schema_list, schemas) -> tuple[list[dict], list[str]]:
    from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn

    probe = _probe(tokenizer, schema_list)
    limitations = []
    if not probe["template_renders_tools"]:
        limitations.append(
            "The tokenizer chat template ignores tools=. Schemas reach the model "
            "only through tools_fallback_prompt text appended to the system message."
        )
    rows = []
    specs = [
        ("m0", "ORD-1001", "CUS-501", 1840),
        ("m1", "ORD-1002", "CUS-502", 2755),
        ("m2", "ORD-1003", "CUS-503", 930),
        ("m3", "ORD-1004", "CUS-504", 4120),
        ("m4", "ORD-1005", "CUS-505", 660),
    ]
    original = factory._extract_tool_calls
    for episode_id, order_id, customer_id, amount in specs:
        episode = RefundEpisode(
            episode_id=episode_id,
            order_id=order_id,
            customer_id=customer_id,
            amount_cents=amount,
            seed=config.SEED,
        )
        print(f"model_episode {episode_id}", flush=True)
        guards, _episode_schemas = make_guarded_tools(episode)
        tool_names = {guard.__name__ for guard in guards}
        tools = guards
        gen_log: list[dict] = []
        extract_log: list[dict] = []
        wrapped = _RecordingEngine(engine, gen_log, tokenizer)
        factory._extract_tool_calls = original
        _install_extract_hook(factory, extract_log)
        rollout = create_rollout_fn(
            rollout_engine=wrapped,
            tools=tools,
            max_steps=config.MAX_STEPS,
            system_prompt=SYSTEM_PROMPT,
            enable_thinking=config.ENABLE_THINKING,
            force_final_answer=config.FORCE_FINAL_ANSWER,
            force_action_on_stall=config.FORCE_ACTION_ON_STALL,
            context_length=config.CONTEXT_LENGTH,
            tools_fallback_prompt=TOOLS_FALLBACK_PROMPT,
            tool_result_format=TOOL_RESULT_FORMAT,
        )
        started = time.monotonic()
        result = rollout(
            [user_prompt(order_id, customer_id)],
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            do_sample=config.DO_SAMPLE,
            max_length=config.CONTEXT_LENGTH,
        )
        elapsed = time.monotonic() - started
        factory._extract_tool_calls = original
        trajectory = result["trajectories"][0]
        final_answer = trajectory.final_response
        loop_steps = list(trajectory.steps or [])
        cursor = [0]
        used_extract = 0
        ep_rows = []
        for step_id, generated in enumerate(gen_log):
            raw = generated["completions_raw"]
            extract = None
            if (
                used_extract < len(extract_log)
                and extract_log[used_extract]["raw_output"] == raw
            ):
                extract = extract_log[used_extract]
                used_extract += 1
            parser_invoked = extract is not None
            parser_result = extract["parser_result"] if parser_invoked else None
            diagnostic, parser_result, pending, _pending_reason = _generation_diagnosis(
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
            visible = _tool_names_visible(generated["rendered_prompt"], schema_list)
            row = _base_step("model", episode, step_id)
            row.update(
                {
                    "messages": generated["messages"],
                    "rendered_prompt": generated["rendered_prompt"],
                    "tools_passed_to_generate": [
                        ((item.get("function") or item).get("name"))
                        for item in (generated["tools_passed"] or [])
                    ],
                    "template_renders_tools": probe["template_renders_tools"],
                    "probe_with_tools_kw": probe["probe_with_tools_kw"],
                    "probe_without_tools_kw": probe["probe_without_tools_kw"],
                    "tool_names_in_rendered_prompt": visible["tool_names"],
                    "parameter_names_in_rendered_prompt": visible["parameter_names"],
                    "hidden_strings_in_rendered_prompt": _hidden_hits(
                        generated["rendered_prompt"], episode
                    ),
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
                    "generation_truncated": generated["generation_truncated"],
                    "prompt_truncated": generated["prompt_truncated"],
                    "new_token_count": generated["new_token_count"],
                    "stop": "generation_truncated" if generated["generation_truncated"] else "eos_or_other",
                    "prompt_len": generated["prompt_len"],
                }
            )
            ep_rows.append(row)
        if used_extract != len(extract_log):
            limitations.append(
                f"{episode_id}: extract hook records ({len(extract_log)}) did not align "
                f"with generate records ({len(gen_log)}); unused extracts={len(extract_log) - used_extract}."
            )
        initial = ep_rows[0]["rendered_prompt"] if ep_rows else ""
        leaked = _hidden_hits(initial, episode)
        if leaked:
            limitations.append(
                f"{episode_id}: initial rendered prompt contains hidden state {leaked}."
            )
        if ep_rows and not all(ep_rows[0]["tool_names_in_rendered_prompt"].values()):
            limitations.append(
                f"{episode_id}: rendered prompt does not contain every tool name."
            )
        _finish(ep_rows, episode, final_answer, elapsed)
        _assign_budget(ep_rows)
        rows.extend(ep_rows)
    return rows, limitations, probe


def _bind_output(out: Path) -> None:
    global ARTIFACTS, FIXTURE_PATH, MODEL_PATH, SUMMARY_PATH, MANIFEST_PATH, ERROR_PATH
    ARTIFACTS = out
    FIXTURE_PATH = out / "fixtures.jsonl"
    MODEL_PATH = out / "model_episodes.jsonl"
    SUMMARY_PATH = out / "summary.json"
    MANIFEST_PATH = out / "manifest.json"
    ERROR_PATH = out / "model_error.txt"


def _call_names(parsed) -> list[str]:
    names = []
    for call in parsed or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        names.append(function.get("name") or call.get("tool") or call.get("name"))
    return names


def _stored_episode_views(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("source") != "model":
            continue
        grouped.setdefault(row["episode_id"], []).append(row)
    views = {}
    for episode_id, steps in grouped.items():
        last = steps[-1]
        views[episode_id] = {
            "steps": len(steps),
            "generation_truncated": [bool(step.get("generation_truncated")) for step in steps],
            "new_token_count": [step.get("new_token_count") for step in steps],
            "parser_result": [_call_names(step.get("parser_result")) for step in steps],
            "executed_calls": last.get("function_entries"),
            "task_success": bool(last.get("task_success")),
        }
    return views


def _episode_views(rows: list[dict]) -> dict:
    views = _stored_episode_views(rows)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("source") != "model":
            continue
        grouped.setdefault(row["episode_id"], []).append(row)
    for episode_id, steps in grouped.items():
        last = steps[-1]
        views[episode_id]["components"] = _n_invalid_components(steps)
        omitted_calls = []
        for step in steps:
            for item in _diagnostic(step).get("omitted_by_native_parser") or []:
                omitted_calls.extend(_call_names(item.get("dropped_by_full_parse")))
        views[episode_id]["omitted_calls"] = omitted_calls
        views[episode_id]["reward"] = _rewards(
            bool(last.get("task_success")), _n_invalid(steps)
        )
    return views


def _reclassify_frozen(factory) -> dict:
    """Classify the frozen 128-token raw strings again. Do not rewrite that run."""
    source = FROZEN_128 / "model_episodes.jsonl"
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    original = factory._extract_tool_calls
    reclassified = []
    for row in rows:
        raw = row.get("raw_output") or ""
        parsed = original({"role": "assistant", "content": raw}, raw)
        episode = RefundEpisode(
            row["episode_id"], row["order_id"], row["customer_id"], 0, config.SEED
        )
        diagnostic = _diagnose(raw, parsed, original, episode)
        if episode.function_entries:
            raise RuntimeError("reclassification of a saved string called a tool")
        copied = dict(row)
        copied["parser_result"] = parsed
        copied["diagnostic"] = diagnostic
        copied["malformed_syntax"] = diagnostic["malformed_syntax"]
        copied["omitted_by_native_parser"] = diagnostic["omitted_by_native_parser"]
        copied["truncated_fragments"] = diagnostic["truncated_fragments"]
        copied["parser_result_matches_frozen"] = _call_names(parsed) == _call_names(
            row.get("parser_result")
        )
        reclassified.append(copied)
    prompt = next(
        row.get("rendered_prompt")
        for row in rows
        if row.get("episode_id") == "m0" and row.get("step_id") == 0
    )
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    (TASK_DIR / "rendered_prompt_m0_128.txt").write_text(prompt, encoding="utf-8")
    frozen_summary = json.loads((FROZEN_128 / "summary.json").read_text(encoding="utf-8"))
    report = {
        "source": str(source),
        "source_unchanged": True,
        "C_status": "preliminary",
        "definition": N_INVALID_DEFINITION,
        "note": (
            "The frozen summary counted complete calls omitted by the native parser "
            "as malformed fragments. That is not a model syntax error, and those "
            "frozen C values are not the provisional penalty."
        ),
        "frozen_summary_rewards": frozen_summary["model"]["rewards"],
        "episodes": _episode_views(reclassified),
        "totals": _n_invalid_components(reclassified),
        "prompt_instruction_in_rendered_prompt": {
            "json_list": "reply with only a JSON list" in prompt,
            "list_even_for_one_call": "Use a list even for one call." in prompt,
        },
    }
    _write_json(TASK_DIR / "n_invalid_128.json", report)
    return report


def _parser_fixture_trace(rows: list[dict]) -> list[dict]:
    wanted = ("parser_single_object", "parser_two_call_list", "parser_truncated_list")
    by_case = {row["case"]: row for row in rows if row.get("case") in wanted}
    trace = []
    for case in wanted:
        row = by_case[case]
        trace.append(
            {
                "case": case,
                "raw_output": row["raw_output"],
                "parser_result": row["parser_result"],
                "diagnostic": {
                    "parse_status": row["parse_status"],
                    "malformed_syntax": row["malformed_syntax"],
                    "omitted_by_native_parser": row["omitted_by_native_parser"],
                    "truncated_fragments": row["truncated_fragments"],
                },
                "executed": [call.get("tool") for call in row.get("calls") or []],
                "generation_truncated": row["generation_truncated"],
            }
        )
    return trace


def _prompt_diff(previous_fallback: str, previous_tool_result: str) -> str:
    import difflib

    previous_system = (
        "You are a refund agent. Use the tools. "
        "Do not invent tokens or amounts. "
        "The refund amount is unknown until a tool returns it. "
        "When the refund is submitted, give the amount in cents as the final answer."
    )
    previous_user = (
        "Refund order ORD-1001 for customer CUS-501. "
        "Look up the order, look up the customer, calculate the refund, "
        "then submit it. Finish with the refund amount in cents."
    )
    pairs = [
        ("tools_fallback_prompt", previous_fallback, TOOLS_FALLBACK_PROMPT),
        ("tool_result_format", previous_tool_result, TOOL_RESULT_FORMAT),
        ("system_prompt", previous_system, SYSTEM_PROMPT),
        ("user_prompt", previous_user, user_prompt("ORD-1001", "CUS-501")),
    ]
    chunks = []
    for name, old, new in pairs:
        old_text = old if old.endswith("\n") else old + "\n"
        new_text = new if new.endswith("\n") else new + "\n"
        chunks.extend(
            difflib.unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=f"a/{name}",
                tofile=f"b/{name}",
            )
        )
        if chunks and not chunks[-1].endswith("\n"):
            chunks.append("\n")
    return "".join(chunks)


def _trajectory_report(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("source") != "model":
            continue
        grouped.setdefault(row["episode_id"], []).append(row)
    report = []
    for episode_id, steps in grouped.items():
        last = steps[-1]
        calls = [call for step in steps for call in step.get("calls") or []]
        components = _n_invalid_components(steps)
        execution = {
            "success": sum(call.get("execution") == "success" for call in calls),
            "order_or_token": sum(bool(call.get("order_or_token")) for call in calls),
            "tool_error": sum(call.get("execution") == "tool_error" for call in calls),
            "exception": sum(call.get("execution") == "exception" for call in calls),
            "not_run": sum(call.get("execution") == "not_run" for call in calls),
        }
        report.append(
            {
                "episode_id": episode_id,
                "task_success_harness": bool(last.get("task_success_harness")),
                "task_success_answer": bool(last.get("task_success_answer")),
                "task_success": bool(last.get("task_success")),
                "stop_reason": last.get("stop_reason"),
                "budget_exhausted": bool(last.get("budget_exhausted")),
                "pending_unexecuted_calls": _call_names(last.get("pending_unexecuted_calls")),
                "pending_unexecuted_reason": last.get("pending_unexecuted_reason"),
                "final_answer": last.get("final_answer"),
                "executed_calls": [
                    {
                        "tool": call.get("tool"),
                        "validation": call.get("validation"),
                        "execution": call.get("execution"),
                        "order_or_token": bool(call.get("order_or_token")),
                    }
                    for call in calls
                ],
                "schema_invalid": components["schema_invalid"],
                "execution_exceptions": execution["exception"],
                "errors": {
                    "malformed_syntax": components["malformed_syntax"],
                    "omitted_by_native_parser": components["omitted_by_native_parser"],
                    "schema_invalid": components["schema_invalid"],
                    "unknown_tool": components["unknown_tool"],
                    "missing_name": components["missing_name"],
                    "truncated_fragment": components["truncated_fragment"],
                    "generation_truncated": components["generation_truncated"],
                    **execution,
                },
                "reward": _rewards(bool(last.get("task_success")), components["N_invalid_provisional"]),
            }
        )
    return report


def _model_revision_from_hub() -> dict:
    try:
        from huggingface_hub import model_info

        info = model_info(config.MODEL_ID)
        return {"hub_sha": info.sha, "hub_id": info.id}
    except Exception as exc:
        return {"hub_sha": None, "hub_error": _redact(f"{type(exc).__name__}: {exc}")}


def _environment_fingerprint(episode: RefundEpisode) -> str:
    import hashlib

    payload = "\n".join(episode.hidden_strings())
    return hashlib.sha256(payload.encode()).hexdigest()


def _seed_generation(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _reward_spread(values: list[float]) -> dict:
    import statistics

    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    return {
        "values": values,
        "mean": mean,
        "std": std,
        "ddof": 1,
        "n_distinct": len(set(values)),
        "zero_std": std == 0.0,
        "centered": [value - mean for value in values],
        "centered_is_trl_advantage": False,
        "note": "centered = r - mean(r) inside this group. A trainer was not run.",
    }


def _one_model_rollout(
    factory,
    engine,
    tokenizer,
    schema_list,
    probe,
    episode: RefundEpisode,
    *,
    generation_seed: int,
    environment_seed: int,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    top_p: float | None,
    system_prompt: str = SYSTEM_PROMPT,
    tools_fallback_prompt: str = TOOLS_FALLBACK_PROMPT,
    tool_result_format: str = TOOL_RESULT_FORMAT,
    prompt_text: str | None = None,
    prompt_protocol: str = "v1.4",
) -> tuple[list[dict], list[str]]:
    from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn

    _seed_generation(generation_seed)
    limitations = []
    guards, _episode_schemas = make_guarded_tools(episode)
    tool_names = {guard.__name__ for guard in guards}
    gen_log: list[dict] = []
    extract_log: list[dict] = []
    wrapped = _RecordingEngine(engine, gen_log, tokenizer)
    original = factory._extract_tool_calls
    factory._extract_tool_calls = original
    _install_extract_hook(factory, extract_log)
    rollout = create_rollout_fn(
        rollout_engine=wrapped,
        tools=guards,
        max_steps=config.MAX_STEPS,
        system_prompt=system_prompt,
        enable_thinking=config.ENABLE_THINKING,
        force_final_answer=config.FORCE_FINAL_ANSWER,
        force_action_on_stall=config.FORCE_ACTION_ON_STALL,
        context_length=config.CONTEXT_LENGTH,
        tools_fallback_prompt=tools_fallback_prompt,
        tool_result_format=tool_result_format,
    )
    started = time.monotonic()
    try:
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "max_length": config.CONTEXT_LENGTH,
        }
        if top_p is not None:
            kwargs["top_p"] = top_p
        text = prompt_text or user_prompt(episode.order_id, episode.customer_id)
        result = rollout([text], **kwargs)
    finally:
        factory._extract_tool_calls = original
    elapsed = time.monotonic() - started
    trajectory = result["trajectories"][0]
    final_answer = trajectory.final_response
    loop_steps = list(trajectory.steps or [])
    cursor = [0]
    used_extract = 0
    ep_rows = []
    fingerprint = _environment_fingerprint(episode)
    for step_id, generated in enumerate(gen_log):
        raw = generated["completions_raw"]
        extract = None
        if used_extract < len(extract_log) and extract_log[used_extract]["raw_output"] == raw:
            extract = extract_log[used_extract]
            used_extract += 1
        parser_invoked = extract is not None
        parser_result = extract["parser_result"] if parser_invoked else None
        diagnostic, parser_result, pending, _pending_reason = _generation_diagnosis(
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
        visible = _tool_names_visible(generated["rendered_prompt"], schema_list)
        row = _base_step("model", episode, step_id)
        row.update(
            {
                "rollout_id": f"{episode.episode_id}-g{generation_seed}",
                "generation_seed": generation_seed,
                "environment_seed": environment_seed,
                "environment_fingerprint": fingerprint,
                "temperature": temperature,
                "do_sample": do_sample,
                "top_p": top_p,
                "prompt_protocol": prompt_protocol,
                "messages": generated["messages"],
                "rendered_prompt": generated["rendered_prompt"],
                "tools_passed_to_generate": [
                    ((item.get("function") or item).get("name"))
                    for item in (generated["tools_passed"] or [])
                ],
                "template_renders_tools": probe["template_renders_tools"],
                "probe_with_tools_kw": probe["probe_with_tools_kw"],
                "probe_without_tools_kw": probe["probe_without_tools_kw"],
                "tool_names_in_rendered_prompt": visible["tool_names"],
                "parameter_names_in_rendered_prompt": visible["parameter_names"],
                "hidden_strings_in_rendered_prompt": _hidden_hits(
                    generated["rendered_prompt"], episode
                ),
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
                "generation_truncated": generated["generation_truncated"],
                "prompt_truncated": generated["prompt_truncated"],
                "new_token_count": generated["new_token_count"],
                "stop": "generation_truncated" if generated["generation_truncated"] else "eos_or_other",
                "prompt_len": generated["prompt_len"],
            }
        )
        ep_rows.append(row)
    if used_extract != len(extract_log):
        limitations.append(
            f"{episode.episode_id} g{generation_seed}: extract hook records did not align"
        )
    _finish(ep_rows, episode, final_answer, elapsed)
    _assign_budget(ep_rows)
    return ep_rows, limitations


def _rollout_view(rows: list[dict]) -> dict:
    last = rows[-1]
    components = _n_invalid_components(rows)
    calls = [call for step in rows for call in step.get("calls") or []]
    reward = _rewards(bool(last.get("task_success")), components["N_invalid_provisional"])
    return {
        "rollout_id": last.get("rollout_id"),
        "episode_id": last.get("episode_id"),
        "generation_seed": last.get("generation_seed"),
        "environment_seed": last.get("environment_seed"),
        "environment_fingerprint": last.get("environment_fingerprint"),
        "task_success": bool(last.get("task_success")),
        "B": reward["B"],
        "C": reward["C"],
        "C_status": "preliminary",
        "N_invalid": components,
        "parsed_calls": len(calls),
        "no_calls": len(calls) == 0,
        "generation_truncated_steps": sum(bool(step.get("generation_truncated")) for step in rows),
        "round_limit_reached": bool(last.get("round_limit_reached")),
        "pending_unexecuted_calls": _call_names(last.get("pending_unexecuted_calls")),
        "elapsed_s": last.get("elapsed_s"),
        "elapsed_limit_exceeded": bool(last.get("elapsed_limit_exceeded")),
        "timeout_enforced": False,
        "reward": reward,
    }


def run_variability(out: Path) -> int:
    """Eight sampled rollouts. Pins are written before the first generation."""
    _bind_output(out)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if config.FORCE_FINAL_ANSWER or config.FORCE_ACTION_ON_STALL:
        raise RuntimeError("automatic output repair flags must stay off")
    pinned = {
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "dtype": config.DTYPE,
        "quantization": config.QUANTIZATION,
        "environment_seed": config.VARIABILITY_ENVIRONMENT_SEED,
        "generation_seeds": list(config.VARIABILITY_GENERATION_SEEDS),
        "do_sample": config.VARIABILITY_DO_SAMPLE,
        "temperature": config.VARIABILITY_TEMPERATURE,
        "top_p": config.VARIABILITY_TOP_P,
        "max_steps": config.VARIABILITY_MAX_STEPS,
        "max_new_tokens": config.VARIABILITY_MAX_NEW_TOKENS,
        "tasks": [
            {
                "episode_id": episode_id,
                "order_id": order_id,
                "customer_id": customer_id,
                "amount_cents": amount,
            }
            for episode_id, order_id, customer_id, amount in config.VARIABILITY_TASKS
        ],
        "prompt": "v1.4 tools_fallback_prompt, tool_result_format, system prompt, user prompt",
        "parser": "unchanged AgentTune _extract_tool_calls",
        "reward": {
            "B": "I(task_success)",
            "C": "preliminary: I(task_success) - lambda * N_invalid",
            "lambda": config.LAMBDA,
            "N_invalid": N_INVALID_DEFINITION,
            "changed_after_results": False,
        },
        "timeout_enforced": False,
        "lora": False,
        "grpo": False,
        "trainer": False,
    }
    manifest = {
        "pinned_before_launch": pinned,
        "launch_command": (
            ".venv/bin/python run_contract.py --variability --out artifacts/task-2"
        ),
    }
    _write_json(MANIFEST_PATH, manifest)

    import agenttune.agentic.rollout_engines.rollout_factory as factory

    fixture_rows = run_fixtures(factory)
    fixture_problems = _check_fixtures(fixture_rows)
    _dump(FIXTURE_PATH, fixture_rows)
    if fixture_problems:
        _write_json(SUMMARY_PATH, {"fixture_problems": fixture_problems, "rollouts": []})
        return 1

    rows: list[dict] = []
    errors = []
    try:
        engine, tokenizer, loaded = _load_model()
        manifest["loaded_model"] = loaded
        _write_json(MANIFEST_PATH, manifest)
        _probe_guards, schema_list = make_guarded_tools(
            RefundEpisode("schema-probe", "ORD-1001", "CUS-501", 1840, config.SEED)
        )
        probe = _probe(tokenizer, schema_list)
        for episode_id, order_id, customer_id, amount in config.VARIABILITY_TASKS:
            for generation_seed in config.VARIABILITY_GENERATION_SEEDS:
                print(f"variability {episode_id} g{generation_seed}", flush=True)
                episode = RefundEpisode(
                    episode_id=episode_id,
                    order_id=order_id,
                    customer_id=customer_id,
                    amount_cents=amount,
                    seed=config.VARIABILITY_ENVIRONMENT_SEED,
                )
                try:
                    ep_rows, limits = _one_model_rollout(
                        factory,
                        engine,
                        tokenizer,
                        schema_list,
                        probe,
                        episode,
                        generation_seed=generation_seed,
                        environment_seed=config.VARIABILITY_ENVIRONMENT_SEED,
                        max_new_tokens=config.VARIABILITY_MAX_NEW_TOKENS,
                        temperature=config.VARIABILITY_TEMPERATURE,
                        do_sample=config.VARIABILITY_DO_SAMPLE,
                        top_p=config.VARIABILITY_TOP_P,
                    )
                    errors.extend(limits)
                except Exception as exc:
                    ERROR_PATH.write_text(_redact(traceback.format_exc()), encoding="utf-8")
                    errors.append(f"{episode_id} g{generation_seed}: {type(exc).__name__}: {_redact(str(exc))}")
                    _dump(MODEL_PATH, rows)
                    _write_json(
                        SUMMARY_PATH,
                        {"partial": True, "errors": errors, "completed_rollouts": len({r.get('rollout_id') for r in rows})},
                    )
                    print(errors[-1], file=sys.stderr)
                    return 2
                rows.extend(ep_rows)
                _dump(MODEL_PATH, rows)
    except Exception as exc:
        ERROR_PATH.write_text(_redact(traceback.format_exc()), encoding="utf-8")
        _dump(MODEL_PATH, rows)
        _write_json(SUMMARY_PATH, {"partial": True, "errors": [f"{type(exc).__name__}: {_redact(str(exc))}"]})
        return 2

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["rollout_id"], []).append(row)
    by_task: dict[str, list[dict]] = {}
    for rollout_rows in grouped.values():
        view = _rollout_view(rollout_rows)
        by_task.setdefault(view["episode_id"], []).append(view)
    groups = {}
    homogeneous = True
    for task, views in by_task.items():
        fingerprints = {view["environment_fingerprint"] for view in views}
        b_stats = _reward_spread([view["B"] for view in views])
        c_stats = _reward_spread([view["C"] for view in views])
        if not (b_stats["zero_std"] and c_stats["zero_std"]):
            homogeneous = False
        groups[task] = {
            "rollouts": views,
            "environment_fingerprints_identical": len(fingerprints) == 1,
            "B": b_stats,
            "C": c_stats,
        }
    summary = {
        "groups": groups,
        "homogeneous": homogeneous,
        "additional_rollouts_launched": False,
        "parameters_changed_after_results": False,
        "reward_changed_after_results": False,
        "centered_rewards_are_trl_advantages": False,
        "trainer_ran": False,
        "fixture_problems": fixture_problems,
        "errors": errors,
        "note": (
            "Both rewards are computed on the saved trajectories. "
            "Centered values are r - mean(r). They are not TRL advantages."
        ),
    }
    if homogeneous:
        summary["homogeneity_note"] = (
            "Both groups have zero standard deviation for B and for C. No further rollouts were launched."
        )
    _write_json(SUMMARY_PATH, summary)
    print(json.dumps({"homogeneous": homogeneous, "groups": {k: {"B": v["B"]["values"], "C": v["C"]["values"]} for k, v in groups.items()}}, indent=2))
    return 0


def _strip_ending_tokens(text: str) -> str:
    from contract.harness import ENDING_TOKENS

    cleaned = (text or "").strip()
    changed = True
    while changed and cleaned:
        changed = False
        for token in ENDING_TOKENS:
            if cleaned.endswith(token):
                cleaned = cleaned[: -len(token)].rstrip()
                changed = True
    return cleaned


def _exactly_one_call_no_extra(raw: str) -> dict:
    """Whether the generation is exactly one JSON list with one call and no other text."""
    cleaned = _strip_ending_tokens(raw)
    out = {
        "exactly_one_call_no_extra_text": False,
        "list_len": None,
        "parse_error": None,
        "leading_or_trailing_text": False,
    }
    if not cleaned:
        out["parse_error"] = "empty"
        return out
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        out["parse_error"] = f"{type(exc).__name__}: {exc}"
        # Still detect whether a call object sits inside non-JSON wrapper text.
        out["leading_or_trailing_text"] = True
        return out
    if not isinstance(value, list):
        out["parse_error"] = "not_a_list"
        return out
    out["list_len"] = len(value)
    if len(value) != 1 or not isinstance(value[0], dict):
        out["parse_error"] = "list_not_one_object"
        return out
    if "tool_name" not in value[0] and "tool_names" not in value[0]:
        out["parse_error"] = "missing_tool_name"
        return out
    out["exactly_one_call_no_extra_text"] = True
    return out


def _format_rollout_view(rows: list[dict]) -> dict:
    last = rows[-1]
    base = _rollout_view(rows)
    executed = []
    for step in rows:
        for call in step.get("calls") or []:
            if call.get("execution") == "success" or call.get("function_entered"):
                name = call.get("name") or (call.get("parsed") or {}).get("tool_name")
                if name:
                    executed.append(name)
    # Prefer the episode's function_entries when present: that is the real impl path.
    if last.get("function_entries"):
        executed = list(last["function_entries"])
    repeats = sorted({name for name in executed if executed.count(name) > 1})
    schema_invalid = []
    for step in rows:
        for call in step.get("calls") or []:
            if call.get("validation") == "schema_invalid":
                schema_invalid.append(
                    {
                        "step_id": step.get("step_id"),
                        "name": call.get("tool") or call.get("name"),
                        "problems": call.get("problems"),
                    }
                )
    turn_flags = []
    for step in rows:
        flag = _exactly_one_call_no_extra(step.get("raw_output") or "")
        flag["step_id"] = step.get("step_id")
        flag["parser_invoked_by_loop"] = bool(step.get("parser_invoked_by_loop"))
        flag["parse_status"] = step.get("parse_status")
        flag["n_parsed_calls"] = len(step.get("calls") or [])
        flag["n_omitted"] = len(step.get("omitted_by_native_parser") or [])
        flag["pending"] = _call_names(step.get("pending_unexecuted_calls"))
        turn_flags.append(flag)
    n_turns = len(rows)
    n_exact = sum(1 for item in turn_flags if item["exactly_one_call_no_extra_text"])
    return {
        **base,
        "n_turns": n_turns,
        "turns_exactly_one_call_no_extra_text": n_exact,
        "turn_format": turn_flags,
        "tools_executed": executed,
        "repeated_tools": repeats,
        "schema_invalid": schema_invalid,
        "submit_executed": "submit_refund" in executed,
        "task_success_harness": bool(last.get("task_success_harness")),
        "task_success_answer": bool(last.get("task_success_answer")),
        "final_answer": last.get("final_answer"),
    }


def _load_rollout_rows(path: Path, rollout_id: str) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("rollout_id") == rollout_id:
            rows.append(row)
    return rows


def _prompt_text_diff(before: str, after: str, label: str) -> str:
    import difflib

    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"v1.4/{label}",
            tofile=f"one_action/{label}",
        )
    )


def run_one_action(out: Path) -> int:
    """Two paired rollouts: m0 and m2 at seed 100 with the one-action prompt."""
    _bind_output(out)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if config.FORCE_FINAL_ANSWER or config.FORCE_ACTION_ON_STALL:
        raise RuntimeError("automatic output repair flags must stay off")

    prompt_diff = "\n".join(
        [
            _prompt_text_diff(TOOLS_FALLBACK_PROMPT, ONE_ACTION_TOOLS_FALLBACK_PROMPT, "tools_fallback_prompt"),
            _prompt_text_diff(TOOL_RESULT_FORMAT, ONE_ACTION_TOOL_RESULT_FORMAT, "tool_result_format"),
            _prompt_text_diff(SYSTEM_PROMPT, ONE_ACTION_SYSTEM_PROMPT, "system_prompt"),
            _prompt_text_diff(
                user_prompt("ORD-1001", "CUS-501"),
                one_action_user_prompt("ORD-1001", "CUS-501"),
                "user_prompt",
            ),
        ]
    )
    (ARTIFACTS / "prompt.diff").write_text(prompt_diff, encoding="utf-8")

    pinned = {
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "dtype": config.DTYPE,
        "quantization": config.QUANTIZATION,
        "environment_seed": config.ONE_ACTION_ENVIRONMENT_SEED,
        "generation_seed": config.ONE_ACTION_GENERATION_SEED,
        "do_sample": config.ONE_ACTION_DO_SAMPLE,
        "temperature": config.ONE_ACTION_TEMPERATURE,
        "top_p": config.ONE_ACTION_TOP_P,
        "max_steps": config.ONE_ACTION_MAX_STEPS,
        "max_new_tokens": config.ONE_ACTION_MAX_NEW_TOKENS,
        "tasks": [
            {
                "episode_id": episode_id,
                "order_id": order_id,
                "customer_id": customer_id,
                "amount_cents": amount,
            }
            for episode_id, order_id, customer_id, amount in config.ONE_ACTION_TASKS
        ],
        "prompt_protocol": "one_action_per_turn",
        "parser": "unchanged AgentTune _extract_tool_calls",
        "validator": "unchanged validation_then_execution",
        "harness": "unchanged RefundEpisode",
        "reward": {
            "B": "I(task_success)",
            "C": "preliminary: I(task_success) - lambda * N_invalid",
            "lambda": config.LAMBDA,
            "N_invalid": N_INVALID_DEFINITION,
            "changed_after_results": False,
        },
        "repair": False,
        "omitted_calls_not_executed": True,
        "lora": False,
        "grpo": False,
        "trainer": False,
        "baseline_dir": config.ONE_ACTION_BASELINE_DIR,
        "comparison": "paired format diagnostic against task-2 m0-g100 and m2-g100",
        "agenttune_patch": "patches/agenttune-d4db4c1-top_p.patch",
    }
    manifest = {
        "pinned_before_launch": pinned,
        "launch_command": ".venv/bin/python run_contract.py --one-action --out artifacts/task-3",
        "prompt_diff": str(ARTIFACTS / "prompt.diff"),
    }
    _write_json(MANIFEST_PATH, manifest)

    import agenttune.agentic.rollout_engines.rollout_factory as factory

    fixture_rows = run_fixtures(factory)
    fixture_problems = _check_fixtures(fixture_rows)
    _dump(FIXTURE_PATH, fixture_rows)
    if fixture_problems:
        _write_json(SUMMARY_PATH, {"fixture_problems": fixture_problems})
        return 1

    rows: list[dict] = []
    errors = []
    try:
        engine, tokenizer, loaded = _load_model()
        manifest["loaded_model"] = loaded
        _write_json(MANIFEST_PATH, manifest)
        _probe_guards, schema_list = make_guarded_tools(
            RefundEpisode("schema-probe", "ORD-1001", "CUS-501", 1840, config.SEED)
        )
        probe = _probe(tokenizer, schema_list)
        for episode_id, order_id, customer_id, amount in config.ONE_ACTION_TASKS:
            generation_seed = config.ONE_ACTION_GENERATION_SEED
            print(f"one_action {episode_id} g{generation_seed}", flush=True)
            episode = RefundEpisode(
                episode_id=episode_id,
                order_id=order_id,
                customer_id=customer_id,
                amount_cents=amount,
                seed=config.ONE_ACTION_ENVIRONMENT_SEED,
            )
            try:
                ep_rows, limits = _one_model_rollout(
                    factory,
                    engine,
                    tokenizer,
                    schema_list,
                    probe,
                    episode,
                    generation_seed=generation_seed,
                    environment_seed=config.ONE_ACTION_ENVIRONMENT_SEED,
                    max_new_tokens=config.ONE_ACTION_MAX_NEW_TOKENS,
                    temperature=config.ONE_ACTION_TEMPERATURE,
                    do_sample=config.ONE_ACTION_DO_SAMPLE,
                    top_p=config.ONE_ACTION_TOP_P,
                    system_prompt=ONE_ACTION_SYSTEM_PROMPT,
                    tools_fallback_prompt=ONE_ACTION_TOOLS_FALLBACK_PROMPT,
                    tool_result_format=ONE_ACTION_TOOL_RESULT_FORMAT,
                    prompt_text=one_action_user_prompt(order_id, customer_id),
                    prompt_protocol="one_action_per_turn",
                )
                errors.extend(limits)
            except Exception as exc:
                ERROR_PATH.write_text(_redact(traceback.format_exc()), encoding="utf-8")
                errors.append(f"{episode_id} g{generation_seed}: {type(exc).__name__}: {_redact(str(exc))}")
                _dump(MODEL_PATH, rows)
                _write_json(SUMMARY_PATH, {"partial": True, "errors": errors})
                return 2
            rows.extend(ep_rows)
            _dump(MODEL_PATH, rows)
    except Exception as exc:
        ERROR_PATH.write_text(_redact(traceback.format_exc()), encoding="utf-8")
        _dump(MODEL_PATH, rows)
        _write_json(SUMMARY_PATH, {"partial": True, "errors": [f"{type(exc).__name__}: {_redact(str(exc))}"]})
        return 2

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["rollout_id"], []).append(row)
    current = {rid: _format_rollout_view(rrows) for rid, rrows in grouped.items()}

    baseline_path = ROOT / config.ONE_ACTION_BASELINE_DIR / "model_episodes.jsonl"
    comparison = {}
    for episode_id, *_ in config.ONE_ACTION_TASKS:
        rid = f"{episode_id}-g{config.ONE_ACTION_GENERATION_SEED}"
        baseline_rows = _load_rollout_rows(baseline_path, rid)
        comparison[rid] = {
            "one_action": current[rid],
            "task_2": _format_rollout_view(baseline_rows) if baseline_rows else None,
        }

    summary = {
        "kind": "paired_one_action_format_diagnostic",
        "not_a_variability_rerun": True,
        "fixture_problems": fixture_problems,
        "errors": errors,
        "rollouts": current,
        "comparison": comparison,
        "stopped_after_two_rollouts": True,
        "additional_rollouts_launched": False,
        "parameters_changed_after_results": False,
        "reward_changed_after_results": False,
        "trainer_ran": False,
    }
    _write_json(SUMMARY_PATH, summary)
    _write_json(ARTIFACTS / "comparison.json", comparison)
    print(
        json.dumps(
            {
                "rollouts": {
                    rid: {
                        "exact_one_call_turns": view["turns_exactly_one_call_no_extra_text"],
                        "n_turns": view["n_turns"],
                        "tools_executed": view["tools_executed"],
                        "repeats": view["repeated_tools"],
                        "schema_invalid": len(view["schema_invalid"]),
                        "pending": view["pending_unexecuted_calls"],
                        "submit": view["submit_executed"],
                        "task_success": view["task_success"],
                    }
                    for rid, view in current.items()
                }
            },
            indent=2,
        )
    )
    return 0


def _failure_reasons(view: dict) -> list[str]:
    reasons = []
    if view.get("task_success"):
        return reasons
    if not view.get("submit_executed"):
        reasons.append("submit_refund_not_executed")
    if view.get("pending_unexecuted_calls"):
        reasons.append(
            "pending_unexecuted:" + ",".join(view["pending_unexecuted_calls"])
        )
    if view.get("schema_invalid"):
        reasons.append(f"schema_invalid_n={len(view['schema_invalid'])}")
    if view.get("repeated_tools"):
        reasons.append("repeated_tools:" + ",".join(view["repeated_tools"]))
    n_turns = view.get("n_turns") or 0
    exact = view.get("turns_exactly_one_call_no_extra_text") or 0
    # Final answer turn is not required to be a tool-call JSON list.
    tool_turns = max(n_turns - 1, 0)
    if tool_turns and exact < tool_turns:
        reasons.append(
            f"format_not_one_action_on_{tool_turns - exact}_of_{tool_turns}_tool_turns"
        )
    if view.get("task_success_harness") and not view.get("task_success_answer"):
        reasons.append("final_answer_not_accepted")
    if not view.get("task_success_harness") and view.get("submit_executed"):
        reasons.append("harness_not_submitted")
    n_invalid = (view.get("N_invalid") or {}).get("N_invalid_provisional")
    if n_invalid:
        reasons.append(f"N_invalid={n_invalid}")
    if not reasons:
        reasons.append("task_success_false")
    return reasons


def _verify_task3_reuse(task3_dir: Path) -> tuple[dict, dict[str, list[dict]], list[str]]:
    """Confirm task-3 pins and load seed-100 trajectories for reuse."""
    problems = []
    manifest = json.loads((task3_dir / "manifest.json").read_text(encoding="utf-8"))
    pinned = manifest.get("pinned_before_launch") or {}
    expected = {
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "environment_seed": config.ONE_ACTION_VARIABILITY_ENVIRONMENT_SEED,
        "generation_seed": config.ONE_ACTION_VARIABILITY_REUSED_SEED,
        "do_sample": config.ONE_ACTION_VARIABILITY_DO_SAMPLE,
        "temperature": config.ONE_ACTION_VARIABILITY_TEMPERATURE,
        "top_p": config.ONE_ACTION_VARIABILITY_TOP_P,
        "max_steps": config.ONE_ACTION_VARIABILITY_MAX_STEPS,
        "max_new_tokens": config.ONE_ACTION_VARIABILITY_MAX_NEW_TOKENS,
        "prompt_protocol": "one_action_per_turn",
    }
    for key, value in expected.items():
        if pinned.get(key) != value:
            problems.append(f"task-3 manifest {key}={pinned.get(key)!r} != {value!r}")
    expected_tasks = [
        {
            "episode_id": episode_id,
            "order_id": order_id,
            "customer_id": customer_id,
            "amount_cents": amount,
        }
        for episode_id, order_id, customer_id, amount in config.ONE_ACTION_VARIABILITY_TASKS
    ]
    if pinned.get("tasks") != expected_tasks:
        problems.append("task-3 manifest tasks do not match the pinned one-action tasks")

    jsonl = task3_dir / "model_episodes.jsonl"
    reused: dict[str, list[dict]] = {}
    for episode_id, order_id, customer_id, amount in config.ONE_ACTION_VARIABILITY_TASKS:
        rid = f"{episode_id}-g{config.ONE_ACTION_VARIABILITY_REUSED_SEED}"
        rows = _load_rollout_rows(jsonl, rid)
        if not rows:
            problems.append(f"missing reused trajectory {rid} in {jsonl}")
            continue
        head = rows[0]
        episode = RefundEpisode(
            episode_id=episode_id,
            order_id=order_id,
            customer_id=customer_id,
            amount_cents=amount,
            seed=config.ONE_ACTION_VARIABILITY_ENVIRONMENT_SEED,
        )
        expected_fp = _environment_fingerprint(episode)
        if head.get("environment_fingerprint") != expected_fp:
            problems.append(
                f"{rid}: environment_fingerprint mismatch with current harness"
            )
        if head.get("environment_seed") != config.ONE_ACTION_VARIABILITY_ENVIRONMENT_SEED:
            problems.append(f"{rid}: environment_seed mismatch")
        if head.get("generation_seed") != config.ONE_ACTION_VARIABILITY_REUSED_SEED:
            problems.append(f"{rid}: generation_seed mismatch")
        if head.get("prompt_protocol") != "one_action_per_turn":
            problems.append(f"{rid}: prompt_protocol is not one_action_per_turn")
        if head.get("temperature") != config.ONE_ACTION_VARIABILITY_TEMPERATURE:
            problems.append(f"{rid}: temperature mismatch")
        if head.get("top_p") != config.ONE_ACTION_VARIABILITY_TOP_P:
            problems.append(f"{rid}: top_p mismatch")
        if head.get("do_sample") != config.ONE_ACTION_VARIABILITY_DO_SAMPLE:
            problems.append(f"{rid}: do_sample mismatch")
        reused[rid] = rows
    return pinned, reused, problems


def run_one_action_variability(out: Path) -> int:
    """Six new one-action rollouts; reuse verified task-3 seed 100 in each group."""
    _bind_output(out)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if config.FORCE_FINAL_ANSWER or config.FORCE_ACTION_ON_STALL:
        raise RuntimeError("automatic output repair flags must stay off")

    task3_dir = ROOT / config.ONE_ACTION_VARIABILITY_TASK3_DIR
    task3_pinned, reused, reuse_problems = _verify_task3_reuse(task3_dir)
    if reuse_problems:
        _write_json(
            SUMMARY_PATH,
            {"reuse_verification_failed": True, "problems": reuse_problems},
        )
        print(json.dumps({"reuse_problems": reuse_problems}, indent=2), file=sys.stderr)
        return 1

    pinned = {
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "dtype": config.DTYPE,
        "quantization": config.QUANTIZATION,
        "environment_seed": config.ONE_ACTION_VARIABILITY_ENVIRONMENT_SEED,
        "reused_generation_seed": config.ONE_ACTION_VARIABILITY_REUSED_SEED,
        "new_generation_seeds": list(config.ONE_ACTION_VARIABILITY_NEW_SEEDS),
        "do_sample": config.ONE_ACTION_VARIABILITY_DO_SAMPLE,
        "temperature": config.ONE_ACTION_VARIABILITY_TEMPERATURE,
        "top_p": config.ONE_ACTION_VARIABILITY_TOP_P,
        "max_steps": config.ONE_ACTION_VARIABILITY_MAX_STEPS,
        "max_new_tokens": config.ONE_ACTION_VARIABILITY_MAX_NEW_TOKENS,
        "tasks": [
            {
                "episode_id": episode_id,
                "order_id": order_id,
                "customer_id": customer_id,
                "amount_cents": amount,
            }
            for episode_id, order_id, customer_id, amount in config.ONE_ACTION_VARIABILITY_TASKS
        ],
        "prompt_protocol": "one_action_per_turn",
        "prompt_source": "contract/harness.py ONE_ACTION_* (task 3)",
        "parser": "unchanged AgentTune _extract_tool_calls",
        "validator": "unchanged validation_then_execution",
        "harness": "unchanged RefundEpisode",
        "reward": {
            "B": "I(task_success)",
            "C": "preliminary: I(task_success) - lambda * N_invalid",
            "lambda": config.LAMBDA,
            "N_invalid": N_INVALID_DEFINITION,
            "changed_after_results": False,
        },
        "repair": False,
        "lora": False,
        "grpo": False,
        "trainer": False,
        "task3_dir": config.ONE_ACTION_VARIABILITY_TASK3_DIR,
        "task3_config_verified": True,
        "new_rollouts_planned": len(config.ONE_ACTION_VARIABILITY_TASKS)
        * len(config.ONE_ACTION_VARIABILITY_NEW_SEEDS),
        "agenttune_patch": "patches/agenttune-d4db4c1-top_p.patch",
    }
    manifest = {
        "pinned_before_launch": pinned,
        "task3_pinned_verified": {
            key: task3_pinned.get(key)
            for key in (
                "model_id",
                "model_revision",
                "environment_seed",
                "generation_seed",
                "do_sample",
                "temperature",
                "top_p",
                "max_steps",
                "max_new_tokens",
                "prompt_protocol",
                "tasks",
            )
        },
        "reused_rollouts": {
            rid: {
                "source": str(task3_dir / "model_episodes.jsonl"),
                "environment_fingerprint": rows[0].get("environment_fingerprint"),
                "n_steps": len(rows),
            }
            for rid, rows in reused.items()
        },
        "launch_command": (
            ".venv/bin/python run_contract.py --one-action-variability --out artifacts/task-4"
        ),
    }
    _write_json(MANIFEST_PATH, manifest)

    import agenttune.agentic.rollout_engines.rollout_factory as factory

    fixture_rows = run_fixtures(factory)
    fixture_problems = _check_fixtures(fixture_rows)
    _dump(FIXTURE_PATH, fixture_rows)
    if fixture_problems:
        _write_json(SUMMARY_PATH, {"fixture_problems": fixture_problems})
        return 1

    new_rows: list[dict] = []
    errors = []
    sources: dict[str, dict] = {
        rid: {
            "source": "artifacts/task-3/model_episodes.jsonl",
            "kind": "reused_task3",
            "generation_seed": config.ONE_ACTION_VARIABILITY_REUSED_SEED,
        }
        for rid in reused
    }
    try:
        engine, tokenizer, loaded = _load_model()
        manifest["loaded_model"] = loaded
        _write_json(MANIFEST_PATH, manifest)
        _probe_guards, schema_list = make_guarded_tools(
            RefundEpisode("schema-probe", "ORD-1001", "CUS-501", 1840, config.SEED)
        )
        probe = _probe(tokenizer, schema_list)
        new_count = 0
        for episode_id, order_id, customer_id, amount in config.ONE_ACTION_VARIABILITY_TASKS:
            for generation_seed in config.ONE_ACTION_VARIABILITY_NEW_SEEDS:
                print(f"one_action_variability {episode_id} g{generation_seed}", flush=True)
                episode = RefundEpisode(
                    episode_id=episode_id,
                    order_id=order_id,
                    customer_id=customer_id,
                    amount_cents=amount,
                    seed=config.ONE_ACTION_VARIABILITY_ENVIRONMENT_SEED,
                )
                try:
                    ep_rows, limits = _one_model_rollout(
                        factory,
                        engine,
                        tokenizer,
                        schema_list,
                        probe,
                        episode,
                        generation_seed=generation_seed,
                        environment_seed=config.ONE_ACTION_VARIABILITY_ENVIRONMENT_SEED,
                        max_new_tokens=config.ONE_ACTION_VARIABILITY_MAX_NEW_TOKENS,
                        temperature=config.ONE_ACTION_VARIABILITY_TEMPERATURE,
                        do_sample=config.ONE_ACTION_VARIABILITY_DO_SAMPLE,
                        top_p=config.ONE_ACTION_VARIABILITY_TOP_P,
                        system_prompt=ONE_ACTION_SYSTEM_PROMPT,
                        tools_fallback_prompt=ONE_ACTION_TOOLS_FALLBACK_PROMPT,
                        tool_result_format=ONE_ACTION_TOOL_RESULT_FORMAT,
                        prompt_text=one_action_user_prompt(order_id, customer_id),
                        prompt_protocol="one_action_per_turn",
                    )
                    errors.extend(limits)
                except Exception as exc:
                    ERROR_PATH.write_text(_redact(traceback.format_exc()), encoding="utf-8")
                    errors.append(
                        f"{episode_id} g{generation_seed}: {type(exc).__name__}: {_redact(str(exc))}"
                    )
                    _dump(MODEL_PATH, new_rows)
                    _write_json(SUMMARY_PATH, {"partial": True, "errors": errors})
                    return 2
                rid = ep_rows[-1]["rollout_id"]
                sources[rid] = {
                    "source": "artifacts/task-4/model_episodes.jsonl",
                    "kind": "new_task4",
                    "generation_seed": generation_seed,
                }
                new_rows.extend(ep_rows)
                new_count += 1
                _dump(MODEL_PATH, new_rows)
        if new_count != pinned["new_rollouts_planned"]:
            errors.append(
                f"expected {pinned['new_rollouts_planned']} new rollouts, got {new_count}"
            )
    except Exception as exc:
        ERROR_PATH.write_text(_redact(traceback.format_exc()), encoding="utf-8")
        _dump(MODEL_PATH, new_rows)
        _write_json(
            SUMMARY_PATH,
            {"partial": True, "errors": [f"{type(exc).__name__}: {_redact(str(exc))}"]},
        )
        return 2

    combined: dict[str, list[dict]] = {rid: list(rows) for rid, rows in reused.items()}
    for row in new_rows:
        combined.setdefault(row["rollout_id"], []).append(row)

    groups: dict[str, dict] = {}
    for episode_id, *_ in config.ONE_ACTION_VARIABILITY_TASKS:
        views = []
        for seed in (config.ONE_ACTION_VARIABILITY_REUSED_SEED, *config.ONE_ACTION_VARIABILITY_NEW_SEEDS):
            rid = f"{episode_id}-g{seed}"
            view = _format_rollout_view(combined[rid])
            view["source"] = sources[rid]
            view["failure_reasons"] = _failure_reasons(view)
            n_turns = view["n_turns"]
            tool_turns = max(n_turns - 1, 0)
            view["one_action_format"] = {
                "n_turns": n_turns,
                "tool_turns": tool_turns,
                "turns_exactly_one_call_no_extra_text": view[
                    "turns_exactly_one_call_no_extra_text"
                ],
                "all_tool_turns_exact": view["turns_exactly_one_call_no_extra_text"]
                >= tool_turns
                and tool_turns > 0,
            }
            views.append(view)
        fingerprints = {view["environment_fingerprint"] for view in views}
        b_values = [view["B"] for view in views]
        c_values = [view["C"] for view in views]
        success_values = [view["task_success"] for view in views]
        n_invalid_values = [
            view["N_invalid"]["N_invalid_provisional"] for view in views
        ]
        groups[episode_id] = {
            "rollouts": views,
            "environment_fingerprints_identical": len(fingerprints) == 1,
            "task_success": success_values,
            "N_invalid": n_invalid_values,
            "B": _reward_spread([float(v) for v in b_values]),
            "C": _reward_spread([float(v) for v in c_values]),
            "trajectory_sources": {
                view["rollout_id"]: view["source"] for view in views
            },
        }

    summary = {
        "kind": "one_action_reward_variability",
        "fixture_problems": fixture_problems,
        "errors": errors,
        "reused_from_task3": sorted(reused),
        "new_rollouts": sorted(
            rid for rid, meta in sources.items() if meta["kind"] == "new_task4"
        ),
        "new_rollouts_count": sum(
            1 for meta in sources.values() if meta["kind"] == "new_task4"
        ),
        "stopped_after_six_new_rollouts": True,
        "groups": groups,
        "parameters_changed_after_results": False,
        "reward_changed_after_results": False,
        "centered_rewards_are_trl_advantages": False,
        "trainer_ran": False,
        "grpo": False,
        "note": (
            "Seed 100 trajectories are reused from task 3 after config and "
            "environment fingerprint verification. Centered values are r - mean(r)."
        ),
    }
    _write_json(SUMMARY_PATH, summary)
    print(
        json.dumps(
            {
                "new_rollouts": summary["new_rollouts_count"],
                "groups": {
                    task: {
                        "task_success": data["task_success"],
                        "N_invalid": data["N_invalid"],
                        "B": data["B"]["values"],
                        "C": data["C"]["values"],
                        "B_std_ddof1": data["B"]["std"],
                        "C_std_ddof1": data["C"]["std"],
                        "sources": data["trajectory_sources"],
                    }
                    for task, data in groups.items()
                },
            },
            indent=2,
        )
    )
    return 0


def _protected_dirs() -> set[Path]:
    return {
        (ROOT / "artifacts" / "contract").resolve(),
        (ROOT / "artifacts" / "contract-128").resolve(),
        (ROOT / "artifacts" / "contract-512").resolve(),
        (ROOT / "artifacts" / "contract-512-v13").resolve(),
        (ROOT / "artifacts" / "contract-512-v14").resolve(),
        (ROOT / "artifacts" / "task-1-2").resolve(),
        (ROOT / "artifacts" / "task-1-3").resolve(),
        (ROOT / "artifacts" / "task-2").resolve(),
        (ROOT / "artifacts" / "task-3").resolve(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "contract"))
    parser.add_argument("--max-new-tokens", type=int, default=PINNED_MAX_NEW_TOKENS)
    parser.add_argument("--variability", action="store_true")
    parser.add_argument("--one-action", action="store_true")
    parser.add_argument("--one-action-variability", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    if args.variability:
        if out.resolve() in _protected_dirs() | {(ROOT / "artifacts" / "task-4").resolve()}:
            raise SystemExit(f"refusing to overwrite {out}")
        return run_variability(out)
    if args.one_action:
        if out.resolve() in _protected_dirs() | {(ROOT / "artifacts" / "task-4").resolve()}:
            raise SystemExit(f"refusing to overwrite {out}")
        return run_one_action(out)
    if args.one_action_variability:
        if out.resolve() in _protected_dirs():
            raise SystemExit(f"refusing to overwrite {out}")
        return run_one_action_variability(out)
    protected = _protected_dirs() | {(ROOT / "artifacts" / "task-4").resolve()}
    if out.resolve() in protected:
        raise SystemExit(f"refusing to overwrite {out}")
    config.MAX_NEW_TOKENS = args.max_new_tokens
    _bind_output(out)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if config.FORCE_FINAL_ANSWER or config.FORCE_ACTION_ON_STALL:
        raise RuntimeError("automatic output repair flags must stay off")
    if config.QUANTIZATION is not None or config.DTYPE != "float16":
        raise RuntimeError("dtype and quantization are pinned in config.py")

    import agenttune.agentic.rollout_engines.rollout_factory as factory
    from agenttune.agentic.rollout_engines.rollout_factory import (
        DEFAULT_TOOL_RESULT_FORMAT,
        DEFAULT_TOOLS_FALLBACK_PROMPT,
    )

    prompt_diff = _prompt_diff(DEFAULT_TOOLS_FALLBACK_PROMPT, DEFAULT_TOOL_RESULT_FORMAT)
    (ARTIFACTS / "prompt.diff").write_text(prompt_diff, encoding="utf-8")

    _probe_guards, schema_list = make_guarded_tools(
        RefundEpisode("schema-probe", "ORD-1001", "CUS-501", 1840, config.SEED)
    )
    schemas = schema_map(schema_list)

    fixture_rows = run_fixtures(factory)
    fixture_problems = _check_fixtures(fixture_rows)
    pair = _arithmetic_pair(fixture_rows)
    _dump(FIXTURE_PATH, fixture_rows)
    parser_trace = _parser_fixture_trace(fixture_rows)
    _write_json(ARTIFACTS / "parser_fixtures.json", parser_trace)

    launch = (
        f".venv/bin/python run_contract.py --out {out.relative_to(ROOT)} "
        f"--max-new-tokens {config.MAX_NEW_TOKENS}"
    )
    manifest = {
        "launch_command": launch,
        "agenttune_commit": _git_sha(),
        "agenttune_path": str(ROOT / "vendor" / "AgentTune"),
        "model_id": config.MODEL_ID,
        "dtype": config.DTYPE,
        "quantization": config.QUANTIZATION,
        "seed": config.SEED,
        "generation": {
            "max_new_tokens": config.MAX_NEW_TOKENS,
            "pinned_max_new_tokens": PINNED_MAX_NEW_TOKENS,
            "temperature": config.TEMPERATURE,
            "do_sample": config.DO_SAMPLE,
            "enable_thinking": config.ENABLE_THINKING,
        },
        "episode_limits": {
            "max_steps": config.MAX_STEPS,
            "context_length": config.CONTEXT_LENGTH,
            "episode_timeout_s": config.EPISODE_TIMEOUT_S,
            "n_model_episodes": config.N_MODEL_EPISODES,
        },
        "repair": {
            "force_final_answer": config.FORCE_FINAL_ANSWER,
            "force_action_on_stall": config.FORCE_ACTION_ON_STALL,
        },
        "lambda": config.LAMBDA,
        "reward_definitions": {
            "B": "I(task_success)",
            "C": "preliminary: I(task_success) - lambda * N_invalid",
            "C_status": "preliminary",
            "task_success": (
                "submitted is true AND the whole final answer is "
                "REFUND_SUBMITTED amount_cents=<the episode amount>, "
                "allowing only outer whitespace and trailing special tokens"
            ),
            "N_invalid": N_INVALID_DEFINITION,
        },
        "parse_rule": PARSE_RULE,
        "parser_format": {
            "source": "vendor/AgentTune/src/agenttune/agentic/rollout_engines/tool_call_parse.py:_parse_cohere_json",
            "prompt_source": "vendor/AgentTune/src/agenttune/agentic/rollout_engines/rollout_factory.py:DEFAULT_TOOLS_FALLBACK_PROMPT",
            "supported_format": PARSER_FORMAT,
            "prompt_changed": True,
            "parser_changed": False,
            "repair_added": False,
            "fixtures": parser_trace,
            "schema_additional_properties": False,
        },
        "loop": {
            "generation": "agenttune.agentic.rollout_engines.rollout_factory._execute_trajectory._gen -> TransformersRolloutEngine.generate",
            "parse": "agenttune.agentic.rollout_engines.rollout_factory._extract_tool_calls",
            "execute": "contract.execute.reject_or_run, called by the guarded tools the loop passes to _run_tools_parallel",
            "schema": "agenttune.agentic.rollout_engines.rollout_factory._build_tool_schema on the guarded tool",
            "raw_output": "TransformersRolloutEngine.generate return value, including the final generation; the parser result is stored separately",
            "max_steps_counts": MAX_STEPS_COUNTS,
        },
        "tools_fallback_prompt": TOOLS_FALLBACK_PROMPT,
        "tool_result_format": TOOL_RESULT_FORMAT,
        "previous_tools_fallback_prompt": DEFAULT_TOOLS_FALLBACK_PROMPT,
        "previous_tool_result_format": DEFAULT_TOOL_RESULT_FORMAT,
        "tool_schemas": schema_list,
        "packages": _package_versions(),
        "hardware": _hardware(),
        "environment_flags": {
            "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
        },
        "hub": _model_revision_from_hub(),
        "access": {
            "distribution": "public_gated",
            "requested_revision": config.MODEL_REVISION,
            "local_token_configured": _local_token_configured(),
        },
        "max_steps_counts": MAX_STEPS_COUNTS,
    }
    success_steps = [row for row in fixture_rows if row["episode_id"] == "f-scripted-success"]
    manifest["max_steps_observed"] = {
        "limit": config.MAX_STEPS,
        "limit_changed": False,
        "generations": len(success_steps),
        "function_entries": success_steps[-1]["function_entries"] if success_steps else [],
    }
    limitations = [
        "No LoRA and no GRPO. This run does not train.",
        "The loop stringifies the guard's return value after the structured result is stored. Its tool_failure_count only sees a dict that still contains an 'error' key, so the contract log uses the structured result.",
        "The loop does not parse the final generation after a tool result. That raw completion is saved from generate(), and the diagnostic classification does not execute it.",
        "Step.thought is stored only as loop_step_thought. raw_output is the generation-engine string.",
    ]
    model_rows: list[dict] = []
    model_error = None
    probe = None
    try:
        engine, tokenizer, loaded = _load_model()
        manifest["loaded_model"] = loaded
        model_rows, model_limits, probe = run_model_episodes(
            factory, engine, tokenizer, schema_list, schemas
        )
        limitations.extend(model_limits)
        _dump(MODEL_PATH, model_rows)
        ERROR_PATH.write_text(
            "\n".join(
                [
                    "status: no_error",
                    f"model_id: {config.MODEL_ID}",
                    f"revision: {config.MODEL_REVISION}",
                    "distribution: public_gated",
                    "config_json: loaded",
                    "tokenizer: loaded",
                    f"local_token_configured: {str(_local_token_configured()).lower()}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        model_error = _redact(traceback.format_exc())
        ERROR_PATH.write_text(
            "\n".join(
                [
                    "status: error",
                    f"model_id: {config.MODEL_ID}",
                    f"revision: {config.MODEL_REVISION}",
                    "distribution: public_gated",
                    f"local_token_configured: {str(_local_token_configured()).lower()}",
                    f"exception: {type(exc).__name__}",
                    "",
                    model_error,
                ]
            ),
            encoding="utf-8",
        )
        limitations.append(
            "CohereLabs/tiny-aya-global did not finish "
            f"({type(exc).__name__}). No other model was loaded. "
            "The redacted traceback is in " + str(ERROR_PATH) + "."
        )
        _dump(
            MODEL_PATH,
            [
                {
                    "record_type": "model_run_status",
                    "source": "model",
                    "status": "not_run",
                    "model_id": config.MODEL_ID,
                    "revision": (manifest.get("hub") or {}).get("hub_sha"),
                    "error_file": str(ERROR_PATH),
                }
            ],
        )

    if probe is not None:
        manifest["template_renders_tools"] = probe["template_renders_tools"]
        manifest["probe_with_tools_kw"] = probe["probe_with_tools_kw"]
        manifest["probe_without_tools_kw"] = probe["probe_without_tools_kw"]

    rendered_paths = []
    for row in model_rows:
        if row.get("episode_id") == "m0" and row.get("rendered_prompt"):
            name = f"rendered_prompt_m0_step{row['step_id']}.txt"
            path = ARTIFACTS / name
            path.write_text(row["rendered_prompt"], encoding="utf-8")
            rendered_paths.append(str(path))
            if row["step_id"] == 0:
                (ARTIFACTS / "rendered_prompt_m0.txt").write_text(row["rendered_prompt"], encoding="utf-8")
    previous_rows = [
        json.loads(line)
        for line in (ROOT / "artifacts" / "contract-512" / "model_episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    diagnostic = {
        "kind": "integration_diagnostic",
        "same_prompt": False,
        "same_answer_predicate": False,
        "same_reward_formula": True,
        "reward_formula_expanded": False,
        "parser_changed": False,
        "repair_added": False,
        "note": (
            "The 512-token run used the previous prompt and a substring check for the amount. "
            "This run uses the new prompt and the exact final-answer line. "
            "Its task_success and B/C are measurements of this contract, not a rescore of that run."
        ),
        "previous_dir": str(ROOT / "artifacts" / "contract-512"),
        "this_dir": str(ARTIFACTS),
        "previous_512": _stored_episode_views(previous_rows),
        "this_run": _episode_views(model_rows),
        "generation_truncated_steps": {
            "previous_512": sum(bool(row.get("generation_truncated")) for row in previous_rows),
            "this_run": sum(bool(row.get("generation_truncated")) for row in model_rows),
        },
    }
    trajectories = _trajectory_report(model_rows)
    truncated_now = diagnostic["generation_truncated_steps"]["this_run"]
    if truncated_now:
        limitations.append(
            f"max_new_tokens={config.MAX_NEW_TOKENS} still truncated "
            f"{truncated_now} generation(s). The limit was not increased again."
        )
    _write_json(ARTIFACTS / "integration_diagnostic.json", diagnostic)
    _write_json(ARTIFACTS / "trajectories.json", {"C_status": "preliminary", "episodes": trajectories})
    manifest["rendered_prompts"] = rendered_paths
    manifest["prompt_diff"] = str(ARTIFACTS / "prompt.diff")

    model_summary = _summarize(model_rows, "model")
    if model_error is not None:
        model_summary["note"] = "no model episode was produced"
        manifest["loaded_model"] = {
            "loaded": False,
            "model_commit": (manifest.get("hub") or {}).get("hub_sha"),
            "tokenizer_commit": None,
            "note": (
                "Weights and tokenizer files were not downloaded. "
                "hub_sha is the revision Hugging Face resolved for "
                "CohereLabs/tiny-aya-global. The model and the tokenizer "
                "are files in that same repository."
            ),
        }
    summary = {
        "parse_rule": PARSE_RULE,
        "fixture_problems": fixture_problems,
        "fixtures": _summarize(fixture_rows, "fixture"),
        "model": model_summary,
        "arithmetic_pair": pair,
        "model_error": model_error is not None,
        "limitations": limitations,
        "files": {
            "fixtures": str(FIXTURE_PATH),
            "model_episodes": str(MODEL_PATH),
            "manifest": str(MANIFEST_PATH),
            "rendered_prompt_m0": str(ARTIFACTS / "rendered_prompt_m0.txt"),
            "prompt_diff": str(ARTIFACTS / "prompt.diff"),
            "trajectories": str(ARTIFACTS / "trajectories.json"),
            "integration_diagnostic": str(ARTIFACTS / "integration_diagnostic.json"),
            "frozen_128": str(FROZEN_128),
            "previous_512": str(ROOT / "artifacts" / "contract-512"),
        },
    }
    manifest["limitations"] = limitations
    _write_json(SUMMARY_PATH, summary)
    _write_json(MANIFEST_PATH, manifest)
    print(json.dumps({"fixture_problems": fixture_problems, "arithmetic_pair": pair, "model_error": model_error is not None, "model_episodes": summary["model"].get("n_episodes")}, indent=2))
    if model_error is not None:
        print(model_error, file=sys.stderr)
        return 2
    if fixture_problems or not pair["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
