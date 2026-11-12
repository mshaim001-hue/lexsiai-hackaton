"""Pinned collection prompts for task 5.22 (4 queries × seeds 100/101).

Continues 5.21 train collection with new paraphrases. Distinct from 5.14 (d*/94xx),
5.15 (t15_*/v15_*/95xx/96xx), 5.21 (c21_*/97xx), and earlier splits.
"""

from __future__ import annotations

from contract.collect_prompts_521 import COLLECT_521_TASKS, _FORBIDDEN_ORDERS as _BASE_FORBIDDEN

COLLECT_522_TASKS = (
    ("c22_0", "ORD-9801", "CUS-K01", 1520),
    ("c22_1", "ORD-9802", "CUS-K02", 2380),
    ("c22_2", "ORD-9803", "CUS-K03", 1810),
    ("c22_3", "ORD-9804", "CUS-K04", 2640),
)

COLLECT_522_SEEDS = (100, 101)

_FORBIDDEN_ORDERS = set(_BASE_FORBIDDEN) | {t[1] for t in COLLECT_521_TASKS}


def collect_522_user_prompt(episode_id: str, order_id: str, customer_id: str) -> str:
    templates = {
        "c22_0": (
            f"I need a refund on {order_id}, customer {customer_id}. "
            f"The electric kettle auto-shutoff failed and boiled dry, scorching the element."
        ),
        "c22_1": (
            f"{customer_id} calling about order {order_id}. "
            f"The yoga blocks arrived crushed and crumbly; store credit is not an option."
        ),
        "c22_2": (
            f"Please reverse {order_id} for {customer_id}. "
            f"The desk lamp flicker started on day three and the support chat never replied."
        ),
        "c22_3": (
            f"Refund request for {order_id} / {customer_id}. "
            f"The mechanical keyboard's spacebar sticks after every other press out of the box."
        ),
    }
    try:
        return templates[episode_id]
    except KeyError as exc:
        raise KeyError(f"no 5.22 collect prompt for {episode_id}") from exc


def assert_collect_522_split() -> None:
    seen_ids: set[str] = set()
    seen_orders: set[str] = set()
    seen_customers: set[str] = set()
    prior_ids = {t[0] for t in COLLECT_521_TASKS}
    for episode_id, order_id, customer_id, amount in COLLECT_522_TASKS:
        if order_id in _FORBIDDEN_ORDERS:
            raise RuntimeError(f"5.22 order {order_id} collides with prior splits")
        if episode_id in prior_ids:
            raise RuntimeError(f"5.22 episode_id {episode_id} collides with 5.21")
        if episode_id in seen_ids:
            raise RuntimeError(f"duplicate episode_id {episode_id}")
        if order_id in seen_orders:
            raise RuntimeError(f"duplicate order_id {order_id}")
        if customer_id in seen_customers:
            raise RuntimeError(f"duplicate customer_id {customer_id}")
        if amount <= 0:
            raise RuntimeError(f"non-positive amount for {episode_id}")
        if episode_id not in {"c22_0", "c22_1", "c22_2", "c22_3"}:
            raise RuntimeError(f"unexpected episode_id {episode_id}")
        seen_ids.add(episode_id)
        seen_orders.add(order_id)
        seen_customers.add(customer_id)
    if len(COLLECT_522_TASKS) != 4:
        raise RuntimeError("expected 4 collection queries")
    if len(COLLECT_522_TASKS) * len(COLLECT_522_SEEDS) > 8:
        raise RuntimeError("5.22 matrix exceeds 8 episodes")
