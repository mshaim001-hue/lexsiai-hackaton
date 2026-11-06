"""Diagnostic labels for one raw string. This module does not execute tools.

An attempt span is only one of these:

- a ``<tool_call>`` / ``<toolcall>`` tag, closed or still open
- a ``<function=...>`` block, closed or still open
- ``<|START_ACTION|>`` through ``<|END_ACTION|>`` or the end of the text
- an ``Action:`` fenced block, or ``Action:`` plus ``Action Input:``
- a JSON object or a JSON list of objects that carries ``tool_name`` /
  ``tool_names``, or ``name`` / ``tool`` / ``function_name`` together with
  ``arguments`` / ``parameters`` / ``args`` / ``input``, or ``type=function``

Prose is not an attempt. A JSON list that is not a list of objects, such as
``[2, 3]``, is not an attempt.

Categories, kept separate:

- ``malformed_syntax``: a finished attempt that the native parser does not
  accept and that does not itself parse into a call. A closed tag with invalid
  JSON is syntax. A complete object is never placed here.
- ``omitted_by_native_parser``: a syntactically complete call that parses on
  its own, while the native parser did not return it. This is not a syntax
  error.
- ``truncated_fragment``: an attempt that runs off the end of the text without
  a closer. Generation truncation is a separate flag supplied by the caller
  and can occur together with any of these categories.
- schema-invalid calls are a validation result, not a parse category.

``parsed`` means the native parser returned one or more calls. ``no_tool_call``
means it returned none. Spans strictly inside a larger span are not counted twice.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

_TOOL_TAG_RE = re.compile(
    r"<tool_?call\b[^>]*>.*?</tool_?call\s*>", re.IGNORECASE | re.DOTALL
)
_TOOL_TAG_OPEN_RE = re.compile(r"<tool_?call\b", re.IGNORECASE)
_FUNCTION_RE = re.compile(
    r"<function=[^>]*>.*?</function>", re.IGNORECASE | re.DOTALL
)
_FUNCTION_OPEN_RE = re.compile(r"<function=", re.IGNORECASE)
_START_ACTION_RE = re.compile(
    r"<\|START_ACTION\|>.*?(?:<\|END_ACTION\|>|$)", re.IGNORECASE | re.DOTALL
)
_ACTION_FENCE_RE = re.compile(r"Action:\s*```.*?```", re.IGNORECASE | re.DOTALL)
_REACT_RE = re.compile(
    r"Action:\s*\S[^\n]*\nAction Input:\s*[^\n]+", re.IGNORECASE
)
_ARG_KEYS = ("arguments", "parameters", "args", "input")


def _inside(start: int, end: int, spans: list[dict]) -> bool:
    return any(s["start"] <= start and end <= s["end"] and not (s["start"] == start and s["end"] == end) for s in spans)


def _add_span(spans: list[dict], kind: str, start: int, end: int, text: str) -> None:
    if _inside(start, end, spans):
        return
    spans.append({"kind": kind, "start": start, "end": end, "text": text})


def _looks_like_call_dict(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if obj.get("type") == "function" and isinstance(obj.get("function"), dict):
        return True
    if "tool_name" in obj or "tool_names" in obj:
        return True
    name = obj.get("name") or obj.get("tool") or obj.get("function_name")
    if isinstance(name, str) and any(k in obj for k in _ARG_KEYS):
        return True
    return False


def _looks_like_attempt_json(obj: Any) -> bool:
    if _looks_like_call_dict(obj):
        return True
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        return any(_looks_like_call_dict(x) for x in obj)
    return False


def attempt_spans(text: str) -> list[dict]:
    spans: list[dict] = []
    for pattern, kind in (
        (_TOOL_TAG_RE, "tool_call_tag"),
        (_FUNCTION_RE, "function_tag"),
        (_START_ACTION_RE, "start_action"),
        (_ACTION_FENCE_RE, "action_fence"),
        (_REACT_RE, "react"),
    ):
        for match in pattern.finditer(text):
            _add_span(spans, kind, match.start(), match.end(), match.group())
    for pattern, kind in (
        (_TOOL_TAG_OPEN_RE, "tool_call_tag_unclosed"),
        (_FUNCTION_OPEN_RE, "function_tag_unclosed"),
    ):
        for match in pattern.finditer(text):
            if _inside(match.start(), match.end(), spans):
                continue
            _add_span(spans, kind, match.start(), len(text), text[match.start() :])
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        if text[index] not in "{[":
            index += 1
            continue
        try:
            obj, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            index += 1
            continue
        if _looks_like_attempt_json(obj) and not _inside(index, end, spans):
            _add_span(spans, "json", index, end, text[index:end])
        index = max(end, index + 1)
    # Drop spans strictly contained in a span added later.
    kept: list[dict] = []
    for span in spans:
        if any(
            other["start"] <= span["start"]
            and span["end"] <= other["end"]
            and other is not span
            and (other["start"], other["end"]) != (span["start"], span["end"])
            for other in spans
        ):
            continue
        kept.append(span)
    return kept


def _call_key(call: dict) -> tuple:
    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = fn.get("name") or call.get("name") or ""
    args = fn.get("arguments", call.get("arguments", ""))
    return (name, args if isinstance(args, str) else json.dumps(args, sort_keys=True))


def _parser_message(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _span_is_closed(span: dict) -> bool:
    kind = span["kind"]
    text = span["text"]
    if kind == "tool_call_tag":
        return bool(re.search(r"</tool_?call\s*>", text, re.IGNORECASE))
    if kind == "function_tag":
        return "</function>" in text.lower()
    if kind in ("tool_call_tag_unclosed", "function_tag_unclosed"):
        return False
    if kind == "start_action":
        return "<|END_ACTION|>" in text.upper() or "<|end_action|>" in text.lower()
    return True


def _incomplete_tail(text: str, occupied: list[tuple[int, int]]) -> dict | None:
    """An opener that never finishes, and after which no complete value starts."""
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        if any(start <= index < end for start, end in occupied) or text[index] not in "{[":
            index += 1
            continue
        try:
            _obj, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            later = any(start > index for start, _end in occupied)
            if not later and text[index:].strip():
                return {
                    "kind": "truncated_fragment",
                    "text": text[index:],
                    "parsed_alone": [],
                    "dropped_by_full_parse": [],
                }
            index += 1
            continue
        index = max(end, index + 1)
    return None


def classify_parse(raw: str, parsed: list[dict] | None, extract: Callable) -> dict:
    text = raw or ""
    spans = attempt_spans(text)
    parsed_calls = list(parsed or [])
    parsed_keys = [_call_key(call) for call in parsed_calls]
    malformed_syntax: list[dict] = []
    omitted: list[dict] = []
    for span in spans:
        alone = extract(_parser_message(span["text"]), span["text"])
        alone_calls = list(alone or [])
        dropped = [call for call in alone_calls if _call_key(call) not in parsed_keys]
        record = {
            "kind": span["kind"],
            "text": span["text"],
            "parsed_alone": alone_calls,
            "dropped_by_full_parse": dropped,
        }
        if alone_calls and not dropped:
            continue
        if dropped:
            # A complete call the native parser did not return is not syntax.
            omitted.append(record)
            continue
        if _span_is_closed(span):
            malformed_syntax.append(record)
        else:
            # Unclosed delimiters are collected with the incomplete-tail pass
            # when they reach EOF; a closed-failure is syntax.
            malformed_syntax.append(record)
    # Unclosed tag spans were classified above. An unclosed tag that failed to
    # parse is an incomplete fragment, not malformed syntax.
    truncated: list[dict] = []
    still_syntax: list[dict] = []
    for record in malformed_syntax:
        if record["kind"] in ("tool_call_tag_unclosed", "function_tag_unclosed"):
            truncated.append({**record, "kind": "truncated_fragment"})
        elif record["kind"] == "start_action" and "<|END_ACTION|>" not in record["text"].upper():
            truncated.append({**record, "kind": "truncated_fragment"})
        else:
            still_syntax.append(record)
    malformed_syntax = still_syntax
    occupied = [(span["start"], span["end"]) for span in spans]
    tail = _incomplete_tail(text, occupied)
    if tail is not None:
        truncated.append(tail)
    if parsed_calls:
        status = "parsed"
    elif malformed_syntax:
        status = "malformed_syntax"
    elif truncated:
        status = "truncated_fragment"
    else:
        status = "no_tool_call"
    return {
        "parse_status": status,
        "native_parse_status": "parsed" if parsed_calls else "no_tool_call",
        "parsed_calls": parsed_calls,
        "malformed_syntax": malformed_syntax,
        "omitted_by_native_parser": omitted,
        "truncated_fragments": truncated,
        "attempt_span_count": len(spans),
    }


def _parameters_schema(schema: dict) -> dict:
    fn = schema.get("function") if isinstance(schema.get("function"), dict) else schema
    params = fn.get("parameters") or {"type": "object", "properties": {}}
    return params


def schema_map(schemas: list[dict]) -> dict[str, dict]:
    out = {}
    for schema in schemas:
        fn = schema.get("function") if isinstance(schema.get("function"), dict) else schema
        name = fn.get("name")
        if name:
            out[name] = schema
    return out


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return type(value) is int
    if expected == "number":
        return type(value) in (int, float) and type(value) is not bool
    if expected == "boolean":
        return type(value) is bool
    return True


def _schema_problems(args: Any, schema: dict) -> list[str]:
    params = _parameters_schema(schema)
    if not isinstance(args, dict):
        return ["arguments are not an object"]
    problems: list[str] = []
    required = params.get("required") or []
    properties = params.get("properties") or {}
    for key in required:
        if key not in args:
            problems.append(f"missing {key}")
    if params.get("additionalProperties") is False:
        for key in args:
            if key not in properties:
                problems.append(f"unexpected {key}")
    for key, spec in properties.items():
        if key not in args or not isinstance(spec, dict):
            continue
        expected = spec.get("type")
        if isinstance(expected, str) and not _type_ok(args[key], expected):
            problems.append(f"{key} is not {expected}")
    return problems


def _arguments(call: dict) -> tuple[str, Any, str | None]:
    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = fn.get("name") or call.get("name") or ""
    raw_args = fn.get(
        "arguments",
        fn.get("parameters", call.get("arguments", call.get("parameters", {}))),
    )
    if isinstance(raw_args, str):
        try:
            loaded = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError as exc:
            return name, raw_args, f"arguments are not JSON: {exc.msg}"
        return name, loaded, None
    return name, raw_args, None


def validate_call(call: dict, schemas: dict[str, dict], tool_names: set[str]) -> dict:
    name, args, load_error = _arguments(call)
    if not name:
        return {
            "validation": "missing_name",
            "name": name,
            "arguments": args,
            "problems": ["missing name"],
        }
    if name not in tool_names:
        return {
            "validation": "unknown_tool",
            "name": name,
            "arguments": args,
            "problems": [f"unknown tool {name}"],
        }
    problems = []
    if load_error:
        problems.append(load_error)
    else:
        problems.extend(_schema_problems(args, schemas.get(name, {})))
    if problems:
        return {
            "validation": "schema_invalid",
            "name": name,
            "arguments": args,
            "problems": problems,
        }
    return {"validation": "valid", "name": name, "arguments": args, "problems": []}


# Diagnosis stops here. Execution lives in contract.execute and is not called
# from classification. A malformed fragment is not rewritten into a call.
