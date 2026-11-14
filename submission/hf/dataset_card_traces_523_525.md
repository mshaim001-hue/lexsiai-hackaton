---
pretty_name: Lexsi T3 refund traces 5.23–5.25
tags:
- lexsi-hackathon
- agenttune
- tool-calling
- traces
license: other
---

# Lexsi Track 3 — paired results and traces (5.23–5.25)

Public paired outcomes and episode traces for the release-candidate adapter
[`mshaim001/lexsi-t3-refund-adapter-5-23-step4`](https://huggingface.co/mshaim001/lexsi-t3-refund-adapter-5-23-step4)
(after publish) vs untrained Tiny Aya Global
(`af89d219b53ed9b13b8a4645f9c8028973510324`).

Source of truth:
[mshaim001-hue/lexsiai-hackaton](https://github.com/mshaim001-hue/lexsiai-hackaton).

## Headline metrics

| Matrix | Full success (base → step_4) | After-submit | Invalid |
| --- | --- | --- | --- |
| 5.23 on 5.14 (dev, 12 eps) | **7/12 → 8/12** | 3 → 2 | 0 → 0 |
| 5.24 + 5.25 (independent, 40 eps) | **27/40 → 28/40** | **2 → 1** | **0 → 0** |

**Target error:** `final_answer_tool_call_after_submit` hit the **base twice**
on the independent matrix; the adapter **fixed one** (`h25_7-g101`) and left
one unchanged (`h24_7-g100`).

## Layout

Preserve repository-relative paths (zip or tree):

- `artifacts/task-5-23/` — REPORT, summary, `eval_514/comparison.json`,
  adapter `traces` / `per_pair` / `aggregates`
- `artifacts/task-5-14/baseline/` — base pairs/traces reused by 5.23
  (`comparison_source` when generation conditions matched)
- `artifacts/task-5-24/` — independent holdout 8×2; `comparison.json`,
  `paired_outcomes.json`, `base/`, `step_4/`
- `artifacts/task-5-25/` — independent holdout 12×2; same plus
  `combined_with_524.json`
- `artifacts/release-candidate/` — clean-checkout load check + control
  replays of `h25_7-g101` and `h24_7-g100`

Primary paired tables:

- 5.23: `artifacts/task-5-23/eval_514/comparison.json`
- 5.24: `artifacts/task-5-24/paired_outcomes.json`
- 5.25: `artifacts/task-5-25/paired_outcomes.json`
- Combined: `artifacts/task-5-25/combined_with_524.json`

## Relation to first-submission traces

The earlier dataset
[`mshaim001/lexsi-t3-refund-traces`](https://huggingface.co/datasets/mshaim001/lexsi-t3-refund-traces)
covers contract tooling through tasks ~5.10 and does **not** include 5.23–5.25.
Keep both datasets in
[Lexsi Hackathon — Refund Data and Traces](https://huggingface.co/collections/mshaim001/lexsi-hackathon-refund-data-and-traces-6af306a596f1197ded3aab30).

## License / access

Trace JSON is released for hackathon review. The base model remains gated;
adapter use requires compliance with CohereLabs Tiny Aya Global terms.
