# Submission checklist — release candidate

Public GitHub repo (main):
`https://github.com/mshaim001-hue/lexsiai-hackaton`

Release-candidate write-up:
[RELEASE_CANDIDATE.md](RELEASE_CANDIDATE.md). Annotation draft:
[annotation_draft.md](annotation_draft.md).

## In-repo (ready for form paste)

- [x] Adapter path: `artifacts/task-5-23/adapters/step_4`
- [x] Base revision: `CohereLabs/tiny-aya-global` @
      `af89d219b53ed9b13b8a4645f9c8028973510324`
- [x] Dev matrix 5.23: **7/12 → 8/12**
      ([REPORT](../artifacts/task-5-23/REPORT.md),
      [comparison](../artifacts/task-5-23/eval_514/comparison.json))
- [x] Independent 5.24+5.25: **27/40 → 28/40**, after-submit **2 → 1**,
      invalid **0 → 0**
      ([combined](../artifacts/task-5-25/combined_with_524.json))
- [x] Explicit note: target error hit base **twice**; adapter fixed **one**
- [x] Paired results / traces linked from RELEASE_CANDIDATE.md
- [x] Clean-checkout load + control replay:
      [artifacts/release-candidate/](../artifacts/release-candidate/)
- [x] Repro commands: README, `artifacts/task-5-*/commands.txt`,
      `artifacts/release-candidate/commands.txt`
- [x] Annotation draft ≤500 words

## Still needed for the Lexsi form (do not start re-submit yet)

- [ ] Confirm Track 3 registration email + Discord handle match the form
- [ ] Paste final annotation (edit draft if the form requires another language)
- [ ] Short project description / title field (if separate from the annotation)
- [ ] Any required team / affiliation fields

## Hugging Face — publication complete

Audit notes: [hf/HF_PUBLISH.md](hf/HF_PUBLISH.md). Cards:
[hf/model_card_step4.md](hf/model_card_step4.md),
[hf/dataset_card_traces_523_525.md](hf/dataset_card_traces_523_525.md).
Form not re-submitted.

### Direct links (anonymous OK)

- Adapter: https://huggingface.co/mshaim001/lexsi-t3-refund-adapter-5-23-step4
- Traces / paired results: https://huggingface.co/datasets/mshaim001/lexsi-t3-refund-traces-5-23-5-25

### Collections (anonymous OK; new repos listed)

- [Refund Model Adapters](https://huggingface.co/collections/mshaim001/lexsi-hackathon-refund-model-adapters-6af3073f5bd1f23d4e0a82fe) — includes `…-adapter-5-23-step4`
- [Refund Data and Traces](https://huggingface.co/collections/mshaim001/lexsi-hackathon-refund-data-and-traces-6af306a596f1197ded3aab30) — includes `…-traces-5-23-5-25`

### Done

- [x] Public model `mshaim001/lexsi-t3-refund-adapter-5-23-step4` (README card + weights)
- [x] Public dataset `mshaim001/lexsi-t3-refund-traces-5-23-5-25` (README card + zip)
- [x] Both repos added to the first-submission collections
- [x] Do **not** upload Hugging Face tokens or private `.env` files

## Earlier pilots (not the RC claim)

5.8/5.9 realistic matrix (no gain) and coached 5.5 remain under
`artifacts/task-5-{5,8,9}/` for history; do not mix with the RC tables above.
