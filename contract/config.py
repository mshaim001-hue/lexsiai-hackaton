"""Pinned before any episode runs. Do not derive these from results."""

SEED = 0
MODEL_ID = "CohereLabs/tiny-aya-global"
# Public gated repo. Resolved by Hugging Face before config.json returned 401.
MODEL_REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"
DTYPE = "float16"
QUANTIZATION = None
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.0
DO_SAMPLE = False
MAX_STEPS = 4
CONTEXT_LENGTH = 4096
EPISODE_TIMEOUT_S = 180
FORCE_FINAL_ANSWER = False
FORCE_ACTION_ON_STALL = False
ENABLE_THINKING = False
# C = I(task_success) - LAMBDA * N_invalid
LAMBDA = 0.5
N_MODEL_EPISODES = 5

# Provisional until the penalty definition is agreed. C must be labelled preliminary.
# Counted: malformed_syntax, missing_name, unknown_tool, schema_invalid.
# Not counted: a complete call omitted by the native parser, truncated fragments,
# generation_truncated, no_tool_call, tool_error, exception, order_or_token, timeout.
N_INVALID_VALIDATION = ("missing_name", "unknown_tool", "schema_invalid")
