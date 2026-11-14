"""Release-candidate verify: load frozen step_4 in a clean checkout and replay
two control episodes (h25_7-g101 fixed, h24_7-g100 still fails).

No training. Same one-action generation/scoring as 5.24/5.25.
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
from contract.grpo_prep import _package_versions, _write_json
from contract.grpo_train_diag import _row_from_steps
from contract.harness import (
    ONE_ACTION_SYSTEM_PROMPT,
    ONE_ACTION_TOOL_RESULT_FORMAT,
    ONE_ACTION_TOOLS_FALLBACK_PROMPT,
    RefundEpisode,
)
from contract.holdout_prompts_524 import holdout_524_user_prompt
from contract.holdout_prompts_525 import holdout_525_user_prompt
from contract.run import (
    _bind_output,
    _dump,
    _one_model_rollout,
    _probe,
    make_guarded_tools,
)
from contract.sft_eval_520 import _load_engine
from contract.sft_pilot import _calls_after_submit

OUT = ROOT / "artifacts" / "release-candidate"
ADAPTER = ROOT / "artifacts" / "task-5-23" / "adapters" / "step_4"
COMMAND_BASE = (
    "HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
    ".venv/bin/python run_release_candidate_verify.py"
)

# Control episodes from independent holdouts (pinned before any RC replay).
CONTROL_EPISODES = (
    {
        "rollout_id": "h25_7-g101",
        "episode_id": "h25_7",
        "order_id": "ORD-9918",
        "customer_id": "CUS-M08",
        "amount_cents": 2460,
        "generation_seed": 101,
        "source": "artifacts/task-5-25",
        "expected_prior": {
            "base_error_type": "final_answer_tool_call_after_submit",
            "adapter_transition": "fail_to_success",
        },
        "prompt_fn": "holdout_525",
    },
    {
        "rollout_id": "h24_7-g100",
        "episode_id": "h24_7",
        "order_id": "ORD-9908",
        "customer_id": "CUS-L08",
        "amount_cents": 2710,
        "generation_seed": 100,
        "source": "artifacts/task-5-24",
        "expected_prior": {
            "base_error_type": "final_answer_tool_call_after_submit",
            "adapter_transition": "unchanged_fail",
        },
        "prompt_fn": "holdout_524",
    },
)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _prompt_for(spec: dict[str, Any]) -> str:
    if spec["prompt_fn"] == "holdout_525":
        return holdout_525_user_prompt(spec["episode_id"], spec["order_id"], spec["customer_id"])
    if spec["prompt_fn"] == "holdout_524":
        return holdout_524_user_prompt(spec["episode_id"], spec["order_id"], spec["customer_id"])
    raise KeyError(spec["prompt_fn"])


def write_pinned_config(*, clean_checkout_note: str) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "base").mkdir(parents=True, exist_ok=True)
    (OUT / "step_4").mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "release_candidate_verify",
        "written_before_run": True,
        "no_training": True,
        "command": COMMAND_BASE,
        "git_head": _git_head(),
        "repo_root": str(ROOT),
        "clean_checkout_note": clean_checkout_note,
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "adapter": str(ADAPTER.relative_to(ROOT)),
        "control_episodes": [
            {
                "rollout_id": s["rollout_id"],
                "episode_id": s["episode_id"],
                "order_id": s["order_id"],
                "customer_id": s["customer_id"],
                "amount_cents": s["amount_cents"],
                "generation_seed": s["generation_seed"],
                "source": s["source"],
                "expected_prior": s["expected_prior"],
                "prompt": _prompt_for(s),
            }
            for s in CONTROL_EPISODES
        ],
        "generation": {
            "settings": "same as task-5-24 / task-5-25 / task-5-14 one-action",
            "environment_seed": config.ONE_ACTION_ENVIRONMENT_SEED,
            "temperature": config.ONE_ACTION_TEMPERATURE,
            "top_p": config.ONE_ACTION_TOP_P,
            "do_sample": config.ONE_ACTION_DO_SAMPLE,
            "max_new_tokens": config.ONE_ACTION_MAX_NEW_TOKENS,
            "max_steps": config.ONE_ACTION_MAX_STEPS,
            "protocol": "one_action_per_turn",
        },
        "load_instructions": {
            "base": (
                f"AutoModelForCausalLM.from_pretrained({config.MODEL_ID!r}, "
                f"revision={config.MODEL_REVISION!r}, local_files_only=True)"
            ),
            "adapter": (
                "PeftModel.from_pretrained(base, "
                "'artifacts/task-5-23/adapters/step_4')"
            ),
            "engine": "contract.sft_eval_520._load_engine",
        },
    }
    _write_json(OUT / "pinned_config.json", payload)
    (OUT / "commands.txt").write_text(
        "\n".join(
            [
                "# Clean-checkout release-candidate verify (no training)",
                "# 1) Fresh worktree or clone of this commit",
                "# 2) Env + AgentTune per README.md / RESUME.md",
                "# 3) Model cached at pinned revision, then:",
                COMMAND_BASE,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def verify_load(adapter_dir: Path) -> dict[str, Any]:
    """Load base at pinned revision, then PEFT adapter; confirm config identity."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

    t0 = time.time()
    base_cfg = AutoConfig.from_pretrained(
        config.MODEL_ID, revision=config.MODEL_REVISION, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID, revision=config.MODEL_REVISION, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        revision=config.MODEL_REVISION,
        dtype=torch.float16,
        local_files_only=True,
    )
    peft = PeftModel.from_pretrained(model, str(adapter_dir))
    peft.eval()
    adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    info = {
        "ok": True,
        "elapsed_s": round(time.time() - t0, 2),
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "base_config_model_type": getattr(base_cfg, "model_type", None),
        "tokenizer_vocab_size": len(tokenizer),
        "adapter_path": str(adapter_dir.relative_to(ROOT)),
        "adapter_base_model_name_or_path": adapter_cfg.get("base_model_name_or_path"),
        "adapter_peft_type": adapter_cfg.get("peft_type"),
        "adapter_r": adapter_cfg.get("r"),
        "adapter_alpha": adapter_cfg.get("lora_alpha"),
        "peft_active": True,
        "dtype": "float16",
        "device_probe": "mps" if torch.backends.mps.is_available() else "cpu",
    }
    if adapter_cfg.get("base_model_name_or_path") != config.MODEL_ID:
        raise RuntimeError(
            f"adapter base mismatch: {adapter_cfg.get('base_model_name_or_path')} "
            f"vs {config.MODEL_ID}"
        )
    del peft, model, tokenizer
    _release()
    return info


def _score_episode(ep_rows: list[dict], *, out_dir: Path) -> dict[str, Any]:
    scored = _row_from_steps(ep_rows, source=str(out_dir / "model_episodes.jsonl"))
    after = _calls_after_submit(ep_rows)
    if (
        scored.get("task_success_harness")
        and not scored.get("task_success_answer")
        and scored.get("error_type") == "final_answer_tool_call_after_submit"
    ):
        after = max(after, 1)
    return {
        "task_success": bool(scored["task_success"]),
        "confirmed": bool(scored["task_success_harness"]),
        "correct_final": bool(scored["task_success_answer"]),
        "error_type": scored["error_type"],
        "n_invalid": int(scored.get("n_invalid") or 0),
        "n_calls_after_confirmed_submit": int(after),
        "final_answer": scored.get("final_answer"),
        "function_entries": scored.get("function_entries"),
    }


def run_controls(adapter_dir: Path | None, *, out_dir: Path, label: str) -> dict[str, Any]:
    import agenttune.agentic.rollout_engines.rollout_factory as factory

    out_dir.mkdir(parents=True, exist_ok=True)
    engine, tokenizer, loaded = _load_engine(adapter_dir)
    _guards, schema_list = make_guarded_tools(
        RefundEpisode("schema-probe", "ORD-9918", "CUS-M08", 2460, config.SEED)
    )
    probe = _probe(tokenizer, schema_list)
    _bind_output(out_dir)
    all_steps: list[dict] = []
    pairs: list[dict[str, Any]] = []

    for spec in CONTROL_EPISODES:
        rid = spec["rollout_id"]
        print(f"RC[{label}] {rid}", flush=True)
        prompt = _prompt_for(spec)
        episode = RefundEpisode(
            episode_id=spec["episode_id"],
            order_id=spec["order_id"],
            customer_id=spec["customer_id"],
            amount_cents=spec["amount_cents"],
            seed=config.ONE_ACTION_ENVIRONMENT_SEED,
        )
        ep_rows, limits = _one_model_rollout(
            factory,
            engine,
            tokenizer,
            schema_list,
            probe,
            episode,
            generation_seed=spec["generation_seed"],
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
        for step in ep_rows:
            step["amount_cents"] = spec["amount_cents"]
            step["order_id"] = spec["order_id"]
            step["customer_id"] = spec["customer_id"]
            step["rollout_id"] = rid
        all_steps.extend(ep_rows)
        _dump(out_dir / "model_episodes.jsonl", all_steps)
        scored = _score_episode(ep_rows, out_dir=out_dir)
        pair = {
            "rollout_id": rid,
            "episode_id": spec["episode_id"],
            "generation_seed": spec["generation_seed"],
            "order_id": spec["order_id"],
            "customer_id": spec["customer_id"],
            "amount_cents": spec["amount_cents"],
            "source": spec["source"],
            "prompt": prompt,
            **scored,
        }
        pairs.append(pair)
        print(
            f"  success={pair['task_success']} confirmed={pair['confirmed']} "
            f"final={pair['correct_final']} invalid={pair['n_invalid']} "
            f"after={pair['n_calls_after_confirmed_submit']} err={pair['error_type']}",
            flush=True,
        )

    _write_json(out_dir / "per_pair.json", {"pairs": pairs})
    _write_json(
        out_dir / "traces.json",
        {
            "rollouts": {
                p["rollout_id"]: {
                    "task_success": p["task_success"],
                    "confirmed": p["confirmed"],
                    "correct_final": p["correct_final"],
                    "error_type": p["error_type"],
                    "n_invalid": p["n_invalid"],
                    "n_calls_after_confirmed_submit": p["n_calls_after_confirmed_submit"],
                    "final_answer": p.get("final_answer"),
                    "function_entries": p.get("function_entries"),
                }
                for p in pairs
            }
        },
    )
    del engine
    _release()
    return {"label": label, "loaded": loaded, "pairs": pairs}


def compare_controls(base_pairs: list[dict], ad_pairs: list[dict]) -> dict[str, Any]:
    base_by = {p["rollout_id"]: p for p in base_pairs}
    ad_by = {p["rollout_id"]: p for p in ad_pairs}
    rows = []
    for spec in CONTROL_EPISODES:
        rid = spec["rollout_id"]
        b, a = base_by[rid], ad_by[rid]
        if (not b["task_success"]) and a["task_success"]:
            transition = "fail_to_success"
        elif b["task_success"] and (not a["task_success"]):
            transition = "success_to_fail"
        elif b["task_success"]:
            transition = "unchanged_success"
        else:
            transition = "unchanged_fail"
        rows.append(
            {
                "rollout_id": rid,
                "transition": transition,
                "expected_prior_transition": spec["expected_prior"]["adapter_transition"],
                "matches_prior_holdout": transition
                == spec["expected_prior"]["adapter_transition"],
                "base": {
                    "task_success": b["task_success"],
                    "confirmed": b["confirmed"],
                    "correct_final": b["correct_final"],
                    "n_invalid": b["n_invalid"],
                    "n_calls_after_confirmed_submit": b["n_calls_after_confirmed_submit"],
                    "error_type": b["error_type"],
                },
                "step_4": {
                    "task_success": a["task_success"],
                    "confirmed": a["confirmed"],
                    "correct_final": a["correct_final"],
                    "n_invalid": a["n_invalid"],
                    "n_calls_after_confirmed_submit": a["n_calls_after_confirmed_submit"],
                    "error_type": a["error_type"],
                },
            }
        )
    return {"per_episode": rows}


def _write_report(
    summary: dict[str, Any], comparison: dict[str, Any], load_info: dict[str, Any]
) -> None:
    lines = [
        "# Release candidate verify — frozen step_4",
        "",
        f"- adapter: `{summary['adapter']}`",
        f"- base model: `{summary['model_id']}` @ `{summary['model_revision']}`",
        f"- git_head: `{summary['git_head']}`",
        f"- clean checkout: {summary.get('clean_checkout_note')}",
        f"- load ok: `{load_info.get('ok')}` (adapter base `{load_info.get('adapter_base_model_name_or_path')}`)",
        "",
        "## Control episodes",
        "",
        "| rollout | base ok | step_4 ok | base after | ad after | base err | ad err | transition | matches prior |",
        "|---|---|---|---:|---:|---|---|---|---|",
    ]
    for row in comparison["per_episode"]:
        b, a = row["base"], row["step_4"]
        lines.append(
            f"| {row['rollout_id']} | {b['task_success']} | {a['task_success']} | "
            f"{b['n_calls_after_confirmed_submit']} | {a['n_calls_after_confirmed_submit']} | "
            f"{b['error_type']} | {a['error_type']} | {row['transition']} | "
            f"{row['matches_prior_holdout']} |"
        )
    lines.append("")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(*, clean_checkout_note: str = "cwd is git worktree/clone used for RC verify") -> int:
    if not ADAPTER.exists():
        raise FileNotFoundError(ADAPTER)
    started = time.time()
    pinned = write_pinned_config(clean_checkout_note=clean_checkout_note)
    print(
        f"RC pinned controls={[s['rollout_id'] for s in CONTROL_EPISODES]} "
        f"adapter={pinned['adapter']} rev={pinned['model_revision']}",
        flush=True,
    )

    print("RC verifying adapter load on pinned Tiny Aya revision", flush=True)
    load_info = verify_load(ADAPTER)
    _write_json(OUT / "load_check.json", load_info)
    print(json.dumps(load_info, indent=2), flush=True)

    print("RC running base on control episodes", flush=True)
    base_run = run_controls(None, out_dir=OUT / "base", label="base")
    print("RC running step_4 on control episodes", flush=True)
    ad_run = run_controls(ADAPTER, out_dir=OUT / "step_4", label="step_4")

    comparison = compare_controls(base_run["pairs"], ad_run["pairs"])
    _write_json(OUT / "comparison.json", comparison)
    _write_json(OUT / "paired_outcomes.json", {"per_episode": comparison["per_episode"]})

    summary = {
        "kind": "release_candidate_verify",
        "no_training": True,
        "command": COMMAND_BASE,
        "git_head": _git_head(),
        "repo_root": str(ROOT),
        "clean_checkout_note": clean_checkout_note,
        "python": platform.python_version(),
        "versions": _package_versions(),
        "elapsed_s": round(time.time() - started, 1),
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "adapter": str(ADAPTER.relative_to(ROOT)),
        "load_check": load_info,
        "base_loaded": base_run["loaded"],
        "adapter_loaded": ad_run["loaded"],
        "comparison": comparison,
        "pinned_config": pinned,
    }
    _write_json(OUT / "summary.json", summary)
    _write_report(summary, comparison, load_info)
    print(json.dumps({"load_ok": load_info["ok"], "comparison": comparison}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
