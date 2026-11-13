"""Pinned independent holdout prompts for task 5.25 (12 queries × seeds 100/101).

New realistic refund paraphrases and ORD-991x/992x IDs. Excludes prior train/eval
and 5.24 holdout orders (94xx–99xx through ORD-9908).
"""

from __future__ import annotations

from contract.holdout_prompts_524 import HOLDOUT_524_TASKS, _FORBIDDEN_ORDERS as _BASE_524

HOLDOUT_525_TASKS = (
    ("h25_0", "ORD-9911", "CUS-M01", 1920),
    ("h25_1", "ORD-9912", "CUS-M02", 2380),
    ("h25_2", "ORD-9913", "CUS-M03", 1540),
    ("h25_3", "ORD-9914", "CUS-M04", 2670),
    ("h25_4", "ORD-9915", "CUS-M05", 1810),
    ("h25_5", "ORD-9916", "CUS-M06", 2090),
    ("h25_6", "ORD-9917", "CUS-M07", 1730),
    ("h25_7", "ORD-9918", "CUS-M08", 2460),
    ("h25_8", "ORD-9919", "CUS-M09", 1590),
    ("h25_9", "ORD-9920", "CUS-M10", 2840),
    ("h25_10", "ORD-9921", "CUS-M11", 1980),
    ("h25_11", "ORD-9922", "CUS-M12", 2210),
)

HOLDOUT_525_SEEDS = (100, 101)

_FORBIDDEN_ORDERS = set(_BASE_524) | {t[1] for t in HOLDOUT_524_TASKS}


def holdout_525_user_prompt(episode_id: str, order_id: str, customer_id: str) -> str:
    templates = {
        "h25_0": (
            f"Please issue a refund on {order_id} for {customer_id}. "
            f"The espresso machine's steam wand sprays scalding water sideways."
        ),
        "h25_1": (
            f"Account {customer_id} needs {order_id} reversed. "
            f"The hiking boots delaminated at the sole after two weekend trails."
        ),
        "h25_2": (
            f"Hi — {customer_id}, order {order_id}. "
            f"The desk lamp flickers on every dimmer setting and buzzes loudly."
        ),
        "h25_3": (
            f"Formal refund request for {order_id} ({customer_id}). "
            f"The yoga mat arrived sticky with a chemical smell that will not air out."
        ),
        "h25_4": (
            f"Customer {customer_id} asks to refund {order_id}. "
            f"The blender jar cracked while blending frozen fruit at the lowest speed."
        ),
        "h25_5": (
            f"{customer_id} / {order_id}. The portable SSD disconnects whenever "
            f"the laptop sleeps, and files corrupted twice."
        ),
        "h25_6": (
            f"Refund {order_id} for id {customer_id}. "
            f"The cast-iron skillet arrived warped and rocks on a flat stovetop."
        ),
        "h25_7": (
            f"Process a refund for {order_id}. Customer {customer_id}. "
            f"The wireless earbuds lose pairing every few minutes outdoors."
        ),
        "h25_8": (
            f"I'd like {order_id} refunded under {customer_id}. "
            f"The linen duvet cover bled dye onto the mattress on the first wash."
        ),
        "h25_9": (
            f"Please reverse charge {order_id} for {customer_id}. "
            f"The robot vacuum gets stuck on every threshold and never returns to dock."
        ),
        "h25_10": (
            f"{customer_id} reporting defect on {order_id}: "
            f"the standing desk motor stalls halfway and the legs are uneven."
        ),
        "h25_11": (
            f"Refund needed on {order_id}, customer {customer_id}. "
            f"The camping tent's rainfly seam leaked during a light overnight drizzle."
        ),
    }
    try:
        return templates[episode_id]
    except KeyError as exc:
        raise KeyError(f"no 5.25 holdout prompt for {episode_id}") from exc


def assert_holdout_525_split() -> None:
    seen_ids: set[str] = set()
    seen_orders: set[str] = set()
    seen_customers: set[str] = set()
    for episode_id, order_id, customer_id, amount in HOLDOUT_525_TASKS:
        if order_id in _FORBIDDEN_ORDERS:
            raise RuntimeError(f"5.25 order {order_id} collides with prior splits")
        if episode_id in seen_ids:
            raise RuntimeError(f"duplicate episode_id {episode_id}")
        if order_id in seen_orders:
            raise RuntimeError(f"duplicate order_id {order_id}")
        if customer_id in seen_customers:
            raise RuntimeError(f"duplicate customer_id {customer_id}")
        if amount <= 0:
            raise RuntimeError(f"non-positive amount for {episode_id}")
        if not episode_id.startswith("h25_"):
            raise RuntimeError(f"unexpected episode_id {episode_id}")
        seen_ids.add(episode_id)
        seen_orders.add(order_id)
        seen_customers.add(customer_id)
    if len(HOLDOUT_525_TASKS) != 12:
        raise RuntimeError("expected 12 holdout queries")
    if len(HOLDOUT_525_TASKS) * len(HOLDOUT_525_SEEDS) != 24:
        raise RuntimeError("expected 24 episodes")
