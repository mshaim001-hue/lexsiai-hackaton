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

## Still needed on Hugging Face (optional until form requires URLs)

Create **public** collection(s) or model/dataset cards and paste URLs:

- [ ] **Adapter** repo with `artifacts/task-5-23/adapters/step_4`
- [ ] **Traces / paired tables** for 5.23 eval_514 and 5.24+5.25 holdouts
- [ ] Model card: base revision, AgentTune pin, one-action protocol, seeds,
      independent-holdout caveat (target error 2→1 fixed)
- [ ] License / gated-base note: HF access to Tiny Aya Global required
- [ ] Do **not** upload Hugging Face tokens or private `.env` files

## Suggested HF layout (names only — create manually later)

```
mshaim001/lexsi-t3-refund-adapter-step4
mshaim001/lexsi-t3-refund-traces-5-23-5-25
```

## Earlier pilots (not the RC claim)

5.8/5.9 realistic matrix (no gain) and coached 5.5 remain under
`artifacts/task-5-{5,8,9}/` for history; do not mix with the RC tables above.
