"""Realistic refund user prompts without tool-sequence coaching.

Pinned before any baseline or pilot run. User text may paraphrase and add
customer-service noise; it must not name the lookup → calculate → submit order.
One-action protocol and the final-answer line stay in the system / tool prompts.
"""

from __future__ import annotations

# Separate from task-5 t*/h* and diagnostic m0/m2. Fixed before generation.
TRAIN_TASKS = (
    ("u0", "ORD-4001", "CUS-801", 1750),
    ("u1", "ORD-4002", "CUS-802", 2340),
    ("u2", "ORD-4003", "CUS-803", 990),
    ("u3", "ORD-4004", "CUS-804", 2860),
)

TEST_TASKS = (
    ("v0", "ORD-5001", "CUS-901", 1520),
    ("v1", "ORD-5002", "CUS-902", 1180),
    ("v2", "ORD-5003", "CUS-903", 2675),
    ("v3", "ORD-5004", "CUS-904", 1430),
)

GENERATION_SEEDS = (100, 101, 102, 103)


def realistic_user_prompt(episode_id: str, order_id: str, customer_id: str) -> str:
    """Return the pinned natural-language refund request for this episode."""
    templates = {
        "u0": (
            f"Hi support — my headphones never arrived and I already opened a ticket last week. "
            f"Please refund order {order_id}. My customer id is {customer_id}. "
            f"I paid with the card ending 4412 and I do not want a replacement."
        ),
        "u1": (
            f"Customer {customer_id} here. Order {order_id} was delivered cracked, "
            f"photos are attached in the portal. I need the money back to the original payment method, "
            f"not store credit. Shipment left Denver on the 3rd."
        ),
        "u2": (
            f"Can you process a return refund for {order_id}? Account {customer_id}. "
            f"The size chart was wrong and the jacket does not fit. I still have the packing slip "
            f"and the item is unused in the original bag."
        ),
        "u3": (
            f"Formal request: reverse the charge for order {order_id} under customer {customer_id}. "
            f"We were billed twice after the site timed out at checkout. Finance needs confirmation "
            f"once the refund posts; ignore the complimentary tote offer."
        ),
        "v0": (
            f"Hello, I'd like my money back for order {order_id}. Customer number {customer_id}. "
            f"The blender motor failed on day two and support chat closed without resolving it. "
            f"I still have the serial sticker on the box."
        ),
        "v1": (
            f"Please refund {order_id} for {customer_id}. I cancelled within the free window "
            f"but the warehouse still shipped it. Tracking shows it sitting at the depot; "
            f"I will refuse delivery if needed."
        ),
        "v2": (
            f"This is about a duplicate shipment. Customer id {customer_id}, order {order_id}. "
            f"We only ordered one desk lamp and received two. Keep the extra if you want — "
            f"just put the second charge back on the Visa."
        ),
        "v3": (
            f"Need a refund on order {order_id}, customer {customer_id}. "
            f"Color was charcoal in the listing and olive in the box. I've already started the "
            f"carrier return label from the email receipt dated Monday."
        ),
    }
    try:
        return templates[episode_id]
    except KeyError as exc:
        raise KeyError(f"no realistic prompt pinned for episode_id={episode_id}") from exc
