#!/usr/bin/env python3
"""Task 5.1: trainer → AgentTune rollout → reward scoring check (no optimizer).

Usage:
  .venv/bin/python run_grpo_path.py
  .venv/bin/python run_grpo_path.py --skip-live
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

from contract.grpo_path import main

if __name__ == "__main__":
    raise SystemExit(main())
