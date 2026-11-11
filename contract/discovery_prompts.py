"""Pinned discovery prompts for task 5.14 (6 queries × 2 seeds).

Fixed before any rollout. Distinct from v0–v3, u0–u3, s*/p*/e*, and 5.12 proposals.
Realistic paraphrases without tool-sequence coaching.
"""

from __future__ import annotations

# 6 new realistic refund requests. Distinct orders, customers, amounts.
DISCOVERY_TASKS = (
    ("d0", "ORD-9401", "CUS-G01", 1620),
    ("d1", "ORD-9402", "CUS-G02", 2480),
    ("d2", "ORD-9403", "CUS-G03", 1350),
    ("d3", "ORD-9404", "CUS-G04", 2790),
    ("d4", "ORD-9405", "CUS-G05", 1910),
    ("d5", "ORD-9406", "CUS-G06", 2240),
)

# Two seeds per query → 12 episodes max. Same seed family as task-5-8.
GENERATION_SEEDS = (100, 101)

_FORBIDDEN_ORDERS = {
    "ORD-1001",
    "ORD-1002",
    "ORD-1003",
    "ORD-1004",
    "ORD-1005",
    "ORD-2001",
    "ORD-2002",
    "ORD-2003",
    "ORD-2004",
    "ORD-3001",
    "ORD-3002",
    "ORD-3003",
    "ORD-3004",
    "ORD-4001",
    "ORD-4002",
    "ORD-4003",
    "ORD-4004",
    "ORD-5001",
    "ORD-5002",
    "ORD-5003",
    "ORD-5004",
}


def discovery_user_prompt(episode_id: str, order_id: str, customer_id: str) -> str:
    templates = {
        "d0": (
            f"Hi — please refund order {order_id}. Customer {customer_id}. "
            f"The wireless mouse double-clicks constantly after a week and I already "
            f"tried a firmware reset from the FAQ."
        ),
        "d1": (
            f"Account {customer_id}: I need money back for {order_id}. "
            f"The camping tent seam split on the first rain. Photos are in the return portal; "
            f"no exchange, original card only."
        ),
        "d2": (
            f"Customer service, customer {customer_id}. Order {order_id} arrived with the wrong "
            f"region power brick. I am traveling and cannot wait for a reship."
        ),
        "d3": (
            f"Formal refund request for {order_id} under {customer_id}. "
            f"We were charged for an annual plan after cancelling the trial in writing on the 12th."
        ),
        "d4": (
            f"Please reverse {order_id} for {customer_id}. The perfume sample set leaked in transit "
            f"and ruined the packing paper; I still have the stained outer carton."
        ),
        "d5": (
            f"{customer_id} here about order {order_id}. The desk chair gas lift will not stay up. "
            f"I do not want a replacement part — just the refund to the debit card."
        ),
    }
    try:
        return templates[episode_id]
    except KeyError as exc:
        raise KeyError(f"no discovery prompt for {episode_id}") from exc


def assert_discovery_split() -> None:
    seen_ids: set[str] = set()
    seen_orders: set[str] = set()
    for episode_id, order_id, customer_id, amount in DISCOVERY_TASKS:
        if order_id in _FORBIDDEN_ORDERS:
            raise RuntimeError(f"discovery order {order_id} collides with prior splits")
        if episode_id in seen_ids:
            raise RuntimeError(f"duplicate episode_id {episode_id}")
        if order_id in seen_orders:
            raise RuntimeError(f"duplicate order_id {order_id}")
        if amount <= 0:
            raise RuntimeError(f"non-positive amount for {episode_id}")
        seen_ids.add(episode_id)
        seen_orders.add(order_id)
    if len(DISCOVERY_TASKS) * len(GENERATION_SEEDS) > 12:
        raise RuntimeError("discovery matrix exceeds 12 episodes")
