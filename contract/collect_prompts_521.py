"""Pinned collection prompts for task 5.21 (8 queries × seeds 100/101).

New realistic refund paraphrases. Distinct from 5.14 discovery (d*/ORD-94xx),
5.15 train/dev (t15_*/v15_*/ORD-95xx/96xx), and earlier u*/v*/s*/p*/e* splits.
"""

from __future__ import annotations

# 8 new realistic refund requests for post-submit train collection.
COLLECT_521_TASKS = (
    ("c21_0", "ORD-9701", "CUS-J01", 1580),
    ("c21_1", "ORD-9702", "CUS-J02", 2410),
    ("c21_2", "ORD-9703", "CUS-J03", 1290),
    ("c21_3", "ORD-9704", "CUS-J04", 2860),
    ("c21_4", "ORD-9705", "CUS-J05", 1740),
    ("c21_5", "ORD-9706", "CUS-J06", 2190),
    ("c21_6", "ORD-9707", "CUS-J07", 1930),
    ("c21_7", "ORD-9708", "CUS-J08", 2670),
)

# Match 5.14 seed family; 8 × 2 = 16 episodes max.
COLLECT_521_SEEDS = (100, 101)

_FORBIDDEN_ORDERS = {
    # Early contract / GRPO ranges
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
    # SFT / probe ranges
    "ORD-6101",
    "ORD-6102",
    "ORD-6103",
    "ORD-6104",
    "ORD-6105",
    "ORD-6106",
    "ORD-6107",
    "ORD-6108",
    "ORD-6109",
    "ORD-6110",
    "ORD-6111",
    "ORD-6112",
    "ORD-6113",
    "ORD-6114",
    "ORD-6115",
    "ORD-6116",
    "ORD-7101",
    "ORD-7102",
    "ORD-7103",
    "ORD-7104",
    "ORD-7105",
    "ORD-7106",
    "ORD-7107",
    "ORD-7108",
    "ORD-8101",
    "ORD-8102",
    "ORD-8103",
    "ORD-8104",
    # 5.14 discovery
    "ORD-9401",
    "ORD-9402",
    "ORD-9403",
    "ORD-9404",
    "ORD-9405",
    "ORD-9406",
    # 5.15 pilot train/dev
    "ORD-9501",
    "ORD-9502",
    "ORD-9503",
    "ORD-9504",
    "ORD-9505",
    "ORD-9506",
    "ORD-9507",
    "ORD-9508",
    "ORD-9601",
    "ORD-9602",
    "ORD-9603",
    "ORD-9604",
}


def collect_521_user_prompt(episode_id: str, order_id: str, customer_id: str) -> str:
    templates = {
        "c21_0": (
            f"Please refund {order_id} for customer {customer_id}. "
            f"The portable blender jar cracked after two uses and spilled smoothie into the motor base."
        ),
        "c21_1": (
            f"Account {customer_id} — I need {order_id} reversed. "
            f"The hiking poles telescoped shut on a downhill and I already filed photos in the returns app."
        ),
        "c21_2": (
            f"Hi support, {customer_id} here about {order_id}. "
            f"The noise-cancelling earbuds only mute the left side after the latest firmware."
        ),
        "c21_3": (
            f"Formal request: refund order {order_id} under {customer_id}. "
            f"We were billed twice for the same express courier pickup on the invoice."
        ),
        "c21_4": (
            f"Customer {customer_id} wants money back on {order_id}. "
            f"The cast-iron griddle arrived warped and will not sit flat on the burner."
        ),
        "c21_5": (
            f"{customer_id} / order {order_id}. The robot vacuum maps half the apartment then "
            f"dies mid-room; no replacement — refund the card."
        ),
        "c21_6": (
            f"Refund please for {order_id}, id {customer_id}. "
            f"The paperback arrived with a missing signature page and a torn spine from the depot."
        ),
        "c21_7": (
            f"Process a refund for {order_id}. Customer id {customer_id}. "
            f"Baby bottle warmer overheats past the marked setting and scorched the first feed."
        ),
    }
    try:
        return templates[episode_id]
    except KeyError as exc:
        raise KeyError(f"no 5.21 collect prompt for {episode_id}") from exc


def assert_collect_521_split() -> None:
    seen_ids: set[str] = set()
    seen_orders: set[str] = set()
    seen_customers: set[str] = set()
    for episode_id, order_id, customer_id, amount in COLLECT_521_TASKS:
        if order_id in _FORBIDDEN_ORDERS:
            raise RuntimeError(f"5.21 order {order_id} collides with prior splits")
        if episode_id in seen_ids:
            raise RuntimeError(f"duplicate episode_id {episode_id}")
        if order_id in seen_orders:
            raise RuntimeError(f"duplicate order_id {order_id}")
        if customer_id in seen_customers:
            raise RuntimeError(f"duplicate customer_id {customer_id}")
        if amount <= 0:
            raise RuntimeError(f"non-positive amount for {episode_id}")
        if episode_id not in {
            "c21_0",
            "c21_1",
            "c21_2",
            "c21_3",
            "c21_4",
            "c21_5",
            "c21_6",
            "c21_7",
        }:
            raise RuntimeError(f"unexpected episode_id {episode_id}")
        seen_ids.add(episode_id)
        seen_orders.add(order_id)
        seen_customers.add(customer_id)
    if len(COLLECT_521_TASKS) != 8:
        raise RuntimeError("expected 8 collection queries")
    if len(COLLECT_521_TASKS) * len(COLLECT_521_SEEDS) > 16:
        raise RuntimeError("5.21 matrix exceeds 16 episodes")
