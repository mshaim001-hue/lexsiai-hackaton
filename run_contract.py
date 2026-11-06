"""Contract run. One command from the repository root:

.venv/bin/python run_contract.py
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

from contract.run import main

if __name__ == "__main__":
    raise SystemExit(main())
