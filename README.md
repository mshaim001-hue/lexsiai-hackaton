# Lexsi hackathon — refund contract

Pinned AgentTune revision: `d4db4c12d52342227c4113a70183d0f64aa2f345`.

## Apply the top_p forward patch

AgentTune’s transformers engine does not pass `top_p` into `model.generate`. The
project patch for the pinned revision is
`patches/agenttune-d4db4c1-top_p.patch`.

From a clean checkout of that AgentTune commit:

```bash
git clone https://github.com/Lexsi-Labs/AgentTune.git
cd AgentTune
git checkout d4db4c12d52342227c4113a70183d0f64aa2f345
git apply --check /path/to/lexsiai-hackaton/patches/agenttune-d4db4c1-top_p.patch
git apply /path/to/lexsiai-hackaton/patches/agenttune-d4db4c1-top_p.patch
```

When AgentTune lives under `vendor/AgentTune` in this repo:

```bash
git -C vendor/AgentTune apply --check ../../patches/agenttune-d4db4c1-top_p.patch
git -C vendor/AgentTune apply ../../patches/agenttune-d4db4c1-top_p.patch
```

The parser is not changed by this patch.

## Contract runs

```bash
.venv/bin/python run_contract.py --out artifacts/<new-dir> --max-new-tokens 512
.venv/bin/python run_contract.py --variability --out artifacts/task-2
.venv/bin/python run_contract.py --one-action --out artifacts/task-3
.venv/bin/python run_contract.py --one-action-variability --out artifacts/task-4
.venv/bin/python run_grpo_prep.py
```

`run_grpo_prep.py` builds the GRPO pilot configs (B and C, `beta=0.0`),
replays the eight saved one-action trajectories through the shared reward
callback, writes train/held-out manifests, and records untrained held-out
metrics. It does not call `trainer.train()` and installs a stop callback that
refuses the first optimizer step.

Do not overwrite finished artifact directories.
