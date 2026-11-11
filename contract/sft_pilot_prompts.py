"""Pinned train/dev prompts for task 5.15 SFT pilot.

Held out from the 5.14 discovery eval matrix (d0–d5). Fixed before any rollout.
"""

from __future__ import annotations

# Train queries. Separate from d*/v*/u*/s*/p*/e* and prior ORD ranges.
TRAIN_TASKS = (
    ("t15_0", "ORD-9501", "CUS-H01", 1730),
    ("t15_1", "ORD-9502", "CUS-H02", 2590),
    ("t15_2", "ORD-9503", "CUS-H03", 1460),
    ("t15_3", "ORD-9504", "CUS-H04", 2810),
    ("t15_4", "ORD-9505", "CUS-H05", 1980),
    ("t15_5", "ORD-9506", "CUS-H06", 2320),
    ("t15_6", "ORD-9507", "CUS-H07", 1670),
    ("t15_7", "ORD-9508", "CUS-H08", 2450),
)

# Dev queries for post-submit final probe (not used in training).
DEV_TASKS = (
    ("v15_0", "ORD-9601", "CUS-I01", 1840),
    ("v15_1", "ORD-9602", "CUS-I02", 2260),
    ("v15_2", "ORD-9603", "CUS-I03", 1590),
    ("v15_3", "ORD-9604", "CUS-I04", 2730),
)

# One seed each to keep collection within budget; 5.14 remains the 12-pair eval.
COLLECTION_SEEDS = (100,)

# 5.14 held-out eval matrix (do not train on these).
EVAL_514_TASKS = (
    ("d0", "ORD-9401", "CUS-G01", 1620),
    ("d1", "ORD-9402", "CUS-G02", 2480),
    ("d2", "ORD-9403", "CUS-G03", 1350),
    ("d3", "ORD-9404", "CUS-G04", 2790),
    ("d4", "ORD-9405", "CUS-G05", 1910),
    ("d5", "ORD-9406", "CUS-G06", 2240),
)
EVAL_514_SEEDS = (100, 101)

_FORBIDDEN = {
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
    "ORD-9401",
    "ORD-9402",
    "ORD-9403",
    "ORD-9404",
    "ORD-9405",
    "ORD-9406",
}


def pilot_user_prompt(episode_id: str, order_id: str, customer_id: str) -> str:
    templates = {
        "t15_0": (
            f"Please refund order {order_id}. Customer {customer_id}. "
            f"The espresso machine gasket failed and grounds flood the tray every shot."
        ),
        "t15_1": (
            f"Account {customer_id}: money back for {order_id}. "
            f"Running shoes arrived two sizes small despite the chart I followed."
        ),
        "t15_2": (
            f"Hi support — {customer_id} / {order_id}. The smart plug never joins Wi‑Fi "
            f"after three resets; I need the charge reversed."
        ),
        "t15_3": (
            f"Formal note: refund {order_id} under {customer_id}. "
            f"We were billed for express shipping that was never applied."
        ),
        "t15_4": (
            f"Customer {customer_id} wants {order_id} refunded. "
            f"The cast-iron skillet arrived with a cracked enamel rim."
        ),
        "t15_5": (
            f"{customer_id} here. Order {order_id} — the vacuum loses suction after ten minutes. "
            f"No replacement, original payment only."
        ),
        "t15_6": (
            f"Refund request {order_id}, id {customer_id}. "
            f"The novel's pages were water-stained on arrival from the depot."
        ),
        "t15_7": (
            f"Process a refund for {order_id}. Customer id {customer_id}. "
            f"Baby monitor audio cuts out past 5 meters indoors."
        ),
        "v15_0": (
            f"I need a refund for order {order_id}. Customer number {customer_id}. "
            f"The yoga mat smell will not off-gas after a week aired out."
        ),
        "v15_1": (
            f"Please reverse {order_id} for {customer_id}. "
            f"Warehouse sent a floor model with scratched legs."
        ),
        "v15_2": (
            f"Customer {customer_id} about {order_id}. "
            f"Subscription box month-two charged after I cancelled month-one."
        ),
        "v15_3": (
            f"Refund request: {order_id}, id {customer_id}. "
            f"Ceramic mug set had two chipped cups in the sealed carton."
        ),
    }
    try:
        return templates[episode_id]
    except KeyError as exc:
        raise KeyError(f"no pilot prompt for {episode_id}") from exc


def assert_pilot_split() -> None:
    groups = {"train": TRAIN_TASKS, "dev": DEV_TASKS, "eval_514": EVAL_514_TASKS}
    seen_ids: dict[str, str] = {}
    seen_orders: dict[str, str] = {}
    for name, tasks in groups.items():
        for episode_id, order_id, _customer_id, amount in tasks:
            if order_id in _FORBIDDEN and name != "eval_514":
                raise RuntimeError(f"{name} order {order_id} forbidden")
            if episode_id in seen_ids:
                raise RuntimeError(
                    f"episode_id {episode_id} overlaps {seen_ids[episode_id]} and {name}"
                )
            if order_id in seen_orders:
                raise RuntimeError(
                    f"order_id {order_id} overlaps {seen_orders[order_id]} and {name}"
                )
            if amount <= 0:
                raise RuntimeError(f"bad amount for {episode_id}")
            seen_ids[episode_id] = name
            seen_orders[order_id] = name
