"""One validation-then-execution path for fixtures and the AgentTune loop.

The loop calls the guarded tool. A missing or mistyped argument is rejected
inside the guard, and the implementation is not called. The structured result
is stored before the caller converts it to a string.
"""

from __future__ import annotations

import inspect

from contract.classify import validate_call
from contract.harness import make_tools

PATH = "validation_then_execution"


def _record(
    name: str,
    validation: str,
    arguments,
    problems: list,
    execution: str,
    order_or_token: bool,
    function_entered: bool,
    structured_result,
) -> dict:
    return {
        "path": PATH,
        "tool": name,
        "validation": validation,
        "arguments": arguments,
        "problems": problems,
        "execution": execution,
        "order_or_token": order_or_token,
        "function_entered": function_entered,
        "structured_result": structured_result,
    }


def reject_or_run(episode, name: str, impl, schema: dict, arguments: dict) -> dict:
    verdict = validate_call(
        {"type": "function", "function": {"name": name, "arguments": arguments}},
        {name: schema},
        {name},
    )
    if verdict["validation"] != "valid":
        record = _record(
            name,
            verdict["validation"],
            arguments,
            verdict["problems"],
            "not_run",
            False,
            False,
            None,
        )
        episode.structured_results.append(record)
        return record
    before = len(episode.function_entries)
    try:
        structured = impl(**arguments)
    except Exception as exc:
        record = _record(
            name,
            "valid",
            arguments,
            [],
            "exception",
            False,
            len(episode.function_entries) > before,
            {"exception": type(exc).__name__, "message": str(exc)},
        )
        episode.structured_results.append(record)
        return record
    record = _record(
        name,
        "valid",
        arguments,
        [],
        structured.get("execution", "success"),
        bool(structured.get("order_or_token")),
        True,
        structured,
    )
    episode.structured_results.append(record)
    return record


def _close_parameters(schema: dict) -> dict:
    """The schema published to the model and the schema used by the validator."""
    function = schema.get("function") if isinstance(schema.get("function"), dict) else schema
    parameters = function.setdefault("parameters", {"type": "object", "properties": {}})
    parameters["additionalProperties"] = False
    return schema


def make_guarded_tools(episode) -> tuple[list, list[dict]]:
    """Tools whose schema is the one AgentTune publishes, and whose body is the guard."""
    from agenttune.agentic.rollout_engines.rollout_factory import _build_tool_schema

    impls = make_tools(episode)
    guards = []
    cells = []
    for impl in impls:
        cell: dict = {"schema": None}

        def guard(_impl=impl, _cell=cell, _episode=episode, **kwargs):
            return reject_or_run(
                _episode, _impl.__name__, _impl, _cell["schema"], kwargs
            )

        guard.__name__ = impl.__name__
        guard.__doc__ = impl.__doc__
        guard.__signature__ = inspect.signature(impl)
        guard.__annotations__ = dict(getattr(impl, "__annotations__", {}))
        guards.append(guard)
        cells.append(cell)
    schemas = [_close_parameters(_build_tool_schema(guard)) for guard in guards]
    for guard, cell, schema in zip(guards, cells, schemas, strict=True):
        cell["schema"] = schema

        def to_schema(_schema=schema):
            return _schema

        guard.to_schema = to_schema
    return guards, schemas


def string_after_structure(record: dict) -> dict:
    """Attach the string form after the structured result is already stored."""
    payload = record.get("structured_result")
    copied = dict(record)
    copied["string_result"] = str(payload if payload is not None else record)
    return copied


def validation_then_execution(parsed_calls, guards: list) -> list[dict]:
    """Same decisions the loop makes: unknown names are not called; known names enter the guard."""
    by_name = {guard.__name__: guard for guard in guards}
    records = []
    for call in parsed_calls or []:
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = fn.get("name") or call.get("name") or ""
        if name not in by_name:
            validation = "missing_name" if not name else "unknown_tool"
            records.append(
                string_after_structure(
                    _record(name, validation, None, [validation], "not_run", False, False, None)
                )
            )
            continue
        raw_args = fn.get("arguments", fn.get("parameters", {}))
        if isinstance(raw_args, str):
            try:
                import json

                raw_args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                raw_args = {}
        if not isinstance(raw_args, dict):
            raw_args = {}
        records.append(string_after_structure(by_name[name](**raw_args)))
    return records
