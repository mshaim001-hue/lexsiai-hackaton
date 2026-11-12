"""Pinned independent holdout prompts for task 5.24 (8 queries × seeds 100/101).

New realistic refund paraphrases and ORD-99xx IDs. Excludes prior train/eval
orders from 5.14–5.23 (94xx–98xx and earlier splits).
"""

from __future__ import annotations

from contract.collect_prompts_521 import _FORBIDDEN_ORDERS as _BASE
from contract.collect_prompts_522 import COLLECT_522_TASKS

HOLDOUT_524_TASKS = (
    ("h24_0", "ORD-9901", "CUS-L01", 1470),
    ("h24_1", "ORD-9902", "CUS-L02", 2290),
    ("h24_2", "ORD-9903", "CUS-L03", 1680),
    ("h24_3", "ORD-9904", "CUS-L04", 2530),
    ("h24_4", "ORD-9905", "CUS-L05", 1860),
    ("h24_5", "ORD-9906", "CUS-L06", 2140),
    ("h24_6", "ORD-9907", "CUS-L07", 1750),
    ("h24_7", "ORD-9908", "CUS-L08", 2710),
)

HOLDOUT_524_SEEDS = (100, 101)

_FORBIDDEN_ORDERS = set(_BASE) | {t[1] for t in COLLECT_522_TASKS} | {
    # 5.21
    "ORD-9701",
    "ORD-9702",
    "ORD-9703",
    "ORD-9704",
    "ORD-9705",
    "ORD-9706",
    "ORD-9707",
    "ORD-9708",
}


def holdout_524_user_prompt(episode_id: str, order_id: str, customer_id: str) -> str:
    templates = {
        "h24_0": (
            f"Please refund order {order_id} for {customer_id}. "
            f"The air fryer basket coating peeled into the first batch of fries."
        ),
        "h24_1": (
            f"Account {customer_id}: reverse {order_id}. "
            f"The waterproof jacket leaked at both seams on a short walk in light rain."
        ),
        "h24_2": (
            f"Hi, {customer_id} about {order_id}. "
            f"The smartwatch band clasp snapped the first week and support only offered a tip sheet."
        ),
        "h24_3": (
            f"Formal refund for {order_id} under {customer_id}. "
            f"We were charged a restocking fee after the seller cancelled the backorder."
        ),
        "h24_4": (
            f"Customer {customer_id} wants {order_id} refunded. "
            f"The ceramic knife chipped on cardboard; I kept the packaging for inspection."
        ),
        "h24_5": (
            f"{customer_id} / {order_id}. The dehumidifier ices over within an hour "
            f"and never reaches the claimed tank fill."
        ),
        "h24_6": (
            f"Refund request {order_id}, id {customer_id}. "
            f"The board game arrived with a crushed box and a missing rulebook insert."
        ),
        "h24_7": (
            f"Process a refund for {order_id}. Customer {customer_id}. "
            f"The white-noise machine randomizes volume spikes that wake the baby."
        ),
    }
    try:
        return templates[episode_id]
    except KeyError as exc:
        raise KeyError(f"no 5.24 holdout prompt for {episode_id}") from exc


def assert_holdout_524_split() -> None:
    seen_ids: set[str] = set()
    seen_orders: set[str] = set()
    seen_customers: set[str] = set()
    for episode_id, order_id, customer_id, amount in HOLDOUT_524_TASKS:
        if order_id in _FORBIDDEN_ORDERS:
            raise RuntimeError(f"5.24 order {order_id} collides with prior splits")
        if episode_id in seen_ids:
            raise RuntimeError(f"duplicate episode_id {episode_id}")
        if order_id in seen_orders:
            raise RuntimeError(f"duplicate order_id {order_id}")
        if customer_id in seen_customers:
            raise RuntimeError(f"duplicate customer_id {customer_id}")
        if amount <= 0:
            raise RuntimeError(f"non-positive amount for {episode_id}")
        if not episode_id.startswith("h24_"):
            raise RuntimeError(f"unexpected episode_id {episode_id}")
        seen_ids.add(episode_id)
        seen_orders.add(order_id)
        seen_customers.add(customer_id)
    if len(HOLDOUT_524_TASKS) != 8:
        raise RuntimeError("expected 8 holdout queries")
    if len(HOLDOUT_524_TASKS) * len(HOLDOUT_524_SEEDS) != 16:
        raise RuntimeError("expected 16 episodes")
