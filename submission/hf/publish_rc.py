#!/usr/bin/env python3
"""Publish RC adapter + 5.23–5.25 traces to HF and add to first-submission collections.

Requires a Write (or fine-grained write) token for user mshaim001:
  .venv/bin/hf auth login
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo, whoami

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "mshaim001/lexsi-t3-refund-adapter-5-23-step4"
DATASET_ID = "mshaim001/lexsi-t3-refund-traces-5-23-5-25"
ADAPTER_COLLECTION = (
    "mshaim001/lexsi-hackathon-refund-model-adapters-6af3073f5bd1f23d4e0a82fe"
)
TRACES_COLLECTION = (
    "mshaim001/lexsi-hackathon-refund-data-and-traces-6af306a596f1197ded3aab30"
)

TRACE_FILES = [
    "artifacts/task-5-23/REPORT.md",
    "artifacts/task-5-23/summary.json",
    "artifacts/task-5-23/pinned_config.json",
    "artifacts/task-5-23/commands.txt",
    "artifacts/task-5-23/eval_514/comparison.json",
    "artifacts/task-5-23/eval_514/adapter_step_4/traces.json",
    "artifacts/task-5-23/eval_514/adapter_step_4/per_pair.json",
    "artifacts/task-5-23/eval_514/adapter_step_4/aggregates.json",
    "artifacts/task-5-14/baseline/per_pair.json",
    "artifacts/task-5-14/baseline/traces.json",
    "artifacts/task-5-14/baseline/aggregates.json",
    "artifacts/task-5-14/baseline/summary.json",
    "artifacts/task-5-24/REPORT.md",
    "artifacts/task-5-24/summary.json",
    "artifacts/task-5-24/pinned_config.json",
    "artifacts/task-5-24/commands.txt",
    "artifacts/task-5-24/comparison.json",
    "artifacts/task-5-24/paired_outcomes.json",
    "artifacts/task-5-24/base/traces.json",
    "artifacts/task-5-24/base/per_pair.json",
    "artifacts/task-5-24/base/aggregates.json",
    "artifacts/task-5-24/step_4/traces.json",
    "artifacts/task-5-24/step_4/per_pair.json",
    "artifacts/task-5-24/step_4/aggregates.json",
    "artifacts/task-5-24/datasets/holdout_matrix.jsonl",
    "artifacts/task-5-24/datasets/manifest.json",
    "artifacts/task-5-25/REPORT.md",
    "artifacts/task-5-25/summary.json",
    "artifacts/task-5-25/pinned_config.json",
    "artifacts/task-5-25/commands.txt",
    "artifacts/task-5-25/comparison.json",
    "artifacts/task-5-25/paired_outcomes.json",
    "artifacts/task-5-25/combined_with_524.json",
    "artifacts/task-5-25/base/traces.json",
    "artifacts/task-5-25/base/per_pair.json",
    "artifacts/task-5-25/base/aggregates.json",
    "artifacts/task-5-25/step_4/traces.json",
    "artifacts/task-5-25/step_4/per_pair.json",
    "artifacts/task-5-25/step_4/aggregates.json",
    "artifacts/task-5-25/datasets/holdout_matrix.jsonl",
    "artifacts/task-5-25/datasets/manifest.json",
    "artifacts/release-candidate/REPORT.md",
    "artifacts/release-candidate/summary.json",
    "artifacts/release-candidate/pinned_config.json",
    "artifacts/release-candidate/commands.txt",
    "artifacts/release-candidate/comparison.json",
    "artifacts/release-candidate/paired_outcomes.json",
    "artifacts/release-candidate/load_check.json",
    "artifacts/release-candidate/base/traces.json",
    "artifacts/release-candidate/base/per_pair.json",
    "artifacts/release-candidate/step_4/traces.json",
    "artifacts/release-candidate/step_4/per_pair.json",
]


def _require_write() -> None:
    info = whoami()
    at = (info.get("auth") or {}).get("accessToken") or {}
    role = at.get("role") if isinstance(at, dict) else None
    scoped = ((at.get("fineGrained") or {}).get("scoped") or []) if isinstance(at, dict) else []
    perms = []
    for scope in scoped:
        perms.extend(scope.get("permissions") or [])
    print(f"hf_user={info.get('name')} token_role={role} scoped_perms={sorted(set(perms))}")
    if role == "write":
        return
    if role == "fineGrained" and any(
        p in perms
        for p in (
            "repo.write",
            "repo.content.write",
            "repo.contribute.write",
            "collection.write",
        )
    ):
        return
    raise SystemExit(
        "HF token cannot create/upload repos. "
        "Run: .venv/bin/hf auth login\n"
        "Use a Write token, or fine-grained with repo.write "
        "(and collection write) on user mshaim001."
    )


def main() -> int:
    _require_write()
    api = HfApi()

    create_repo(MODEL_ID, repo_type="model", private=False, exist_ok=True)
    adapter = ROOT / "artifacts/task-5-23/adapters/step_4"
    api.upload_file(
        path_or_fileobj=str(adapter / "adapter_config.json"),
        path_in_repo="adapter_config.json",
        repo_id=MODEL_ID,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=str(adapter / "adapter_model.safetensors"),
        path_in_repo="adapter_model.safetensors",
        repo_id=MODEL_ID,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=str(ROOT / "submission/hf/model_card_step4.md"),
        path_in_repo="README.md",
        repo_id=MODEL_ID,
        repo_type="model",
    )
    print("uploaded", MODEL_ID)

    missing = [p for p in TRACE_FILES if not (ROOT / p).exists()]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    zip_path = Path("/tmp/lexsi-t3-refund-traces-5-23-5-25.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in TRACE_FILES:
            zf.write(ROOT / rel, arcname=rel)

    create_repo(DATASET_ID, repo_type="dataset", private=False, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(zip_path),
        path_in_repo="lexsi-t3-refund-traces-5-23-5-25.zip",
        repo_id=DATASET_ID,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=str(ROOT / "submission/hf/dataset_card_traces_523_525.md"),
        path_in_repo="README.md",
        repo_id=DATASET_ID,
        repo_type="dataset",
    )
    print("uploaded", DATASET_ID)

    api.add_collection_item(ADAPTER_COLLECTION, MODEL_ID, item_type="model", exists_ok=True)
    api.add_collection_item(TRACES_COLLECTION, DATASET_ID, item_type="dataset", exists_ok=True)
    print("added to collections")
    print("model", f"https://huggingface.co/{MODEL_ID}")
    print("dataset", f"https://huggingface.co/datasets/{DATASET_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
