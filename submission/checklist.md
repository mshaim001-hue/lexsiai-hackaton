# Submission checklist — remaining items

Already in this public GitHub repo (main):

- [x] Reproducible commands in [README.md](../README.md) and `artifacts/task-5-*/commands.txt`
- [x] Pinned configs: `artifacts/task-5-8/pinned_config.json`, `artifacts/task-5-9/pilot_config.json`
- [x] Traces / metrics: `artifacts/task-5-8/baseline/`, `artifacts/task-5-9/eval_B/`, `artifacts/task-5-9/comparison.json`
- [x] Adapters in git: `artifacts/task-5-5/adapter_B|C`, `artifacts/task-5-9/adapter_B`
- [x] Annotation draft ≤500 words: [annotation_draft.md](annotation_draft.md)
- [x] Main table 5.8 vs 5.9; 5.5 shown separately
- [x] Explicit note: 5.8 matrix chose the target failure; pilot showed no gain

## Still needed for the Lexsi form

- [ ] Confirm Track 3 registration email + Discord handle match the form
- [ ] Paste final annotation (edit draft if the form requires another language)
- [ ] Public GitHub URL (repo is already public if visibility is public):  
      `https://github.com/mshaim001-hue/lexsiai-hackaton`
- [ ] Short project description / title field (if separate from the annotation)
- [ ] Link to README section “Main result” or a one-page summary screenshot if the form asks for figures
- [ ] Any required team / affiliation fields

## Still needed on Hugging Face

Create **public** collection(s) or model/dataset cards and paste URLs into the form:

- [ ] **Adapter collection or model repo** with `artifacts/task-5-9/adapter_B`  
      (optional: also upload 5.5 `adapter_B` / `adapter_C` as reference)
- [ ] **Dataset repo** with realistic split  
      `artifacts/task-5-8/datasets/train.jsonl`, `test.jsonl`, `manifest.json` / `pinned_config.json`
- [ ] **Traces repo or dataset** with  
      `artifacts/task-5-8/baseline/{traces,per_pair,summary}.json` and  
      `artifacts/task-5-9/{comparison.json,eval_B/traces.json,eval_B/per_pair.json,group_u1.json}`
- [ ] Model card text: base `CohereLabs/tiny-aya-global` @ `af89d219…`, AgentTune pin, one-action protocol, eval seeds, “observation on selection matrix” caveat
- [ ] License / gated-base note: users need HF access to Tiny Aya Global
- [ ] Do **not** upload Hugging Face tokens or private `.env` files

## Suggested HF layout (names only — create manually)

```
mshaim001/lexsi-t3-refund-adapter-5-9
mshaim001/lexsi-t3-refund-data-5-8
mshaim001/lexsi-t3-refund-traces-5-8-5-9
```

Upload from a machine with `huggingface-cli login` after the repos exist; keep this git history as the source of truth for commands.
