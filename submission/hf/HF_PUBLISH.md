# Hugging Face publish audit (first submission → RC update)

Checked **without login** (anonymous HTTP 200). Form not submitted.

**Published (2026-11-14):**

- https://huggingface.co/mshaim001/lexsi-t3-refund-adapter-5-23-step4
- https://huggingface.co/datasets/mshaim001/lexsi-t3-refund-traces-5-23-5-25

Both repos are in the first-submission collections (verified anonymously).
Collection membership was added manually after `repo.write` upload;
`collection.write` was not on the fine-grained token.

## Existing collections (first application)

1. **Lexsi Hackathon — Refund Model Adapters**  
   https://huggingface.co/collections/mshaim001/lexsi-hackathon-refund-model-adapters-6af3073f5bd1f23d4e0a82fe  
   Public, ungated. Items:
   - https://huggingface.co/mshaim001/lexsi-t3-refund-adapter-5-9
   - https://huggingface.co/mshaim001/lexsi-t3-refund-adapter-5-5-b
   - https://huggingface.co/mshaim001/lexsi-t3-refund-adapter-5-5-c
   - https://huggingface.co/mshaim001/lexsi-t3-refund-adapter-5-10-b

2. **Lexsi Hackathon — Refund Data and Traces**  
   https://huggingface.co/collections/mshaim001/lexsi-hackathon-refund-data-and-traces-6af306a596f1197ded3aab30  
   Public, ungated. Items:
   - https://huggingface.co/datasets/mshaim001/lexsi-t3-refund-data-5-8  
     (`train.jsonl`, `test.jsonl`, `manifest.json`, `pinned_config.json`)
   - https://huggingface.co/datasets/mshaim001/lexsi-t3-refund-traces  
     (`lexsi-t3-refund-traces.zip` — older contract/5.2–5.10 artifacts only)

### Available now (anonymous)

| Artifact | URL | Notes |
| --- | --- | --- |
| Adapter collection | [collection](https://huggingface.co/collections/mshaim001/lexsi-hackathon-refund-model-adapters-6af3073f5bd1f23d4e0a82fe) | 5.5 / 5.9 / 5.10 only |
| Data+traces collection | [collection](https://huggingface.co/collections/mshaim001/lexsi-hackathon-refund-data-and-traces-6af306a596f1197ded3aab30) | 5.8 data + early traces zip |
| 5.9 adapter | [model](https://huggingface.co/mshaim001/lexsi-t3-refund-adapter-5-9) | public; weights downloadable |
| 5.5-B / 5.5-C / 5.10-B | linked in adapter collection | public |
| 5.8 split | [dataset](https://huggingface.co/datasets/mshaim001/lexsi-t3-refund-data-5-8) | public |
| Early traces zip | [dataset](https://huggingface.co/datasets/mshaim001/lexsi-t3-refund-traces) | **no** 5.14 / 5.23 / 5.24 / 5.25 / step_4 |

### Not published (needed for RC update)

| Needed | Exact local path / HF target | Status |
| --- | --- | --- |
| **step_4 adapter** | `artifacts/task-5-23/adapters/step_4/` → new model `mshaim001/lexsi-t3-refund-adapter-5-23-step4` in adapters collection | missing (HF 401/not found) |
| **5.23 paired + adapter traces** | `artifacts/task-5-23/eval_514/…` + REPORT/summary | missing |
| **5.23 base pairs/traces** | reused `artifacts/task-5-14/baseline/{per_pair,traces,aggregates,summary}.json` | missing from HF zip |
| **5.24 paired + traces** | `artifacts/task-5-24/{comparison,paired_outcomes,REPORT,summary}.json` + `base/` + `step_4/` | missing |
| **5.25 paired + traces + combined** | `artifacts/task-5-25/{comparison,paired_outcomes,combined_with_524,REPORT,summary}.json` + `base/` + `step_4/` | missing |
| **RC control replay** (optional but useful) | `artifacts/release-candidate/` | missing |

## Exact links to use in the form after publish

Until upload, point reviewers at GitHub; after upload use:

- Adapter: `https://huggingface.co/mshaim001/lexsi-t3-refund-adapter-5-23-step4`
- Traces dataset: `https://huggingface.co/datasets/mshaim001/lexsi-t3-refund-traces-5-23-5-25`  
  (new dataset in the existing Data and Traces collection; keep the old zip as history)

GitHub (already public) mirrors for the same files:

- Adapter: https://github.com/mshaim001-hue/lexsiai-hackaton/tree/main/artifacts/task-5-23/adapters/step_4
- 5.23 comparison: https://github.com/mshaim001-hue/lexsiai-hackaton/blob/main/artifacts/task-5-23/eval_514/comparison.json
- 5.24 paired: https://github.com/mshaim001-hue/lexsiai-hackaton/blob/main/artifacts/task-5-24/paired_outcomes.json
- 5.25 combined: https://github.com/mshaim001-hue/lexsiai-hackaton/blob/main/artifacts/task-5-25/combined_with_524.json

## File lists for upload

### A. Model → adapters collection

Repo: **`mshaim001/lexsi-t3-refund-adapter-5-23-step4`** (public, ungated)  
Card: [`model_card_step4.md`](model_card_step4.md)

```
artifacts/task-5-23/adapters/step_4/adapter_config.json
artifacts/task-5-23/adapters/step_4/adapter_model.safetensors
submission/hf/model_card_step4.md   → README.md
```

Then **Add to collection**  
`Lexsi Hackathon — Refund Model Adapters`.

### B. Dataset → data/traces collection

Repo: **`mshaim001/lexsi-t3-refund-traces-5-23-5-25`** (public, ungated)  
Card: [`dataset_card_traces_523_525.md`](dataset_card_traces_523_525.md)

Upload either as flat tree or one zip preserving paths:

```
# 5.23 (dev matrix paired + adapter traces)
artifacts/task-5-23/REPORT.md
artifacts/task-5-23/summary.json
artifacts/task-5-23/pinned_config.json
artifacts/task-5-23/commands.txt
artifacts/task-5-23/eval_514/comparison.json
artifacts/task-5-23/eval_514/adapter_step_4/traces.json
artifacts/task-5-23/eval_514/adapter_step_4/per_pair.json
artifacts/task-5-23/eval_514/adapter_step_4/aggregates.json
# 5.23 base came from frozen 5.14 baseline (document in card)
artifacts/task-5-14/baseline/per_pair.json
artifacts/task-5-14/baseline/traces.json
artifacts/task-5-14/baseline/aggregates.json
artifacts/task-5-14/baseline/summary.json

# 5.24 independent holdout
artifacts/task-5-24/REPORT.md
artifacts/task-5-24/summary.json
artifacts/task-5-24/pinned_config.json
artifacts/task-5-24/commands.txt
artifacts/task-5-24/comparison.json
artifacts/task-5-24/paired_outcomes.json
artifacts/task-5-24/base/traces.json
artifacts/task-5-24/base/per_pair.json
artifacts/task-5-24/base/aggregates.json
artifacts/task-5-24/step_4/traces.json
artifacts/task-5-24/step_4/per_pair.json
artifacts/task-5-24/step_4/aggregates.json
artifacts/task-5-24/datasets/holdout_matrix.jsonl
artifacts/task-5-24/datasets/manifest.json

# 5.25 independent holdout + combined
artifacts/task-5-25/REPORT.md
artifacts/task-5-25/summary.json
artifacts/task-5-25/pinned_config.json
artifacts/task-5-25/commands.txt
artifacts/task-5-25/comparison.json
artifacts/task-5-25/paired_outcomes.json
artifacts/task-5-25/combined_with_524.json
artifacts/task-5-25/base/traces.json
artifacts/task-5-25/base/per_pair.json
artifacts/task-5-25/base/aggregates.json
artifacts/task-5-25/step_4/traces.json
artifacts/task-5-25/step_4/per_pair.json
artifacts/task-5-25/step_4/aggregates.json
artifacts/task-5-25/datasets/holdout_matrix.jsonl
artifacts/task-5-25/datasets/manifest.json

# optional RC control replay
artifacts/release-candidate/REPORT.md
artifacts/release-candidate/summary.json
artifacts/release-candidate/pinned_config.json
artifacts/release-candidate/commands.txt
artifacts/release-candidate/comparison.json
artifacts/release-candidate/paired_outcomes.json
artifacts/release-candidate/load_check.json
artifacts/release-candidate/base/traces.json
artifacts/release-candidate/base/per_pair.json
artifacts/release-candidate/step_4/traces.json
artifacts/release-candidate/step_4/per_pair.json
```

Suggested zip name if mirroring the first submission style:  
`lexsi-t3-refund-traces-5-23-5-25.zip`

Then **Add to collection**  
`Lexsi Hackathon — Refund Data and Traces`.

## Suggested upload commands (when ready; not run)

```bash
# login once on the machine that owns mshaim001
.venv/bin/hf auth login

.venv/bin/hf upload mshaim001/lexsi-t3-refund-adapter-5-23-step4 \
  artifacts/task-5-23/adapters/step_4 \
  --repo-type model

.venv/bin/hf upload mshaim001/lexsi-t3-refund-adapter-5-23-step4 \
  submission/hf/model_card_step4.md README.md \
  --repo-type model

# after creating the dataset repo / zip:
.venv/bin/hf upload mshaim001/lexsi-t3-refund-traces-5-23-5-25 \
  /path/to/lexsi-t3-refund-traces-5-23-5-25.zip \
  --repo-type dataset
```

Add both new repos to the two existing collections in the HF UI (or `huggingface_hub` collection API).

## Claim reminder for cards / form

- Dev 5.23: **7/12 → 8/12**
- Independent 5.24+5.25: **27/40 → 28/40**, after-submit **2 → 1**, invalid **0 → 0**
- Target error hit base **twice**; adapter fixed **one** (`h25_7-g101`)
