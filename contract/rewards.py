"""GRPO reward callbacks for the refund contract.

B and C are separate pilot reward functions. C is a single callable:
I(task_success) - 0.5 * N_invalid. Both require contract_events (or trajectories
that carry them). Completions text alone is not enough.
"""

from __future__ import annotations

from contract import config
from contract.events import extract_events_from_reward_kwargs


LAMBDA = config.LAMBDA


def reward_B(completions=None, contract_events=None, trajectories=None, **kwargs):
    """B = I(task_success)."""
    completions = list(completions or [])
    events = extract_events_from_reward_kwargs(
        contract_events=contract_events,
        trajectories=trajectories,
        n=len(completions),
    )
    return [1.0 if item.get("task_success") else 0.0 for item in events]


def reward_C(completions=None, contract_events=None, trajectories=None, **kwargs):
    """One function: I(task_success) - 0.5 * N_invalid."""
    completions = list(completions or [])
    events = extract_events_from_reward_kwargs(
        contract_events=contract_events,
        trajectories=trajectories,
        n=len(completions),
    )
    out = []
    for item in events:
        success = 1.0 if item.get("task_success") else 0.0
        n_invalid = float(item.get("n_invalid") or 0)
        out.append(success - LAMBDA * n_invalid)
    return out


reward_B.__name__ = "reward_B_I_task_success"
reward_C.__name__ = "reward_C_I_success_minus_lambda_N_invalid"
