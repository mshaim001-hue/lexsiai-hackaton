"""Pinned prompts and splits for task 5.11 SFT on final answers.

Train, probe, and full-eval IDs must not overlap with each other or with
task-5-8 v0–v3 / u0–u3.
"""

from __future__ import annotations

# 16 successful SFT chains. Distinct orders, customers, amounts, paraphrases.
TRAIN_TASKS = (
    ("s0", "ORD-6101", "CUS-A01", 1110),
    ("s1", "ORD-6102", "CUS-A02", 2220),
    ("s2", "ORD-6103", "CUS-A03", 3330),
    ("s3", "ORD-6104", "CUS-A04", 1440),
    ("s4", "ORD-6105", "CUS-A05", 2550),
    ("s5", "ORD-6106", "CUS-A06", 1660),
    ("s6", "ORD-6107", "CUS-A07", 2770),
    ("s7", "ORD-6108", "CUS-A08", 1880),
    ("s8", "ORD-6109", "CUS-A09", 2990),
    ("s9", "ORD-6110", "CUS-A10", 1210),
    ("s10", "ORD-6111", "CUS-A11", 2320),
    ("s11", "ORD-6112", "CUS-A12", 3430),
    ("s12", "ORD-6113", "CUS-A13", 1540),
    ("s13", "ORD-6114", "CUS-A14", 2650),
    ("s14", "ORD-6115", "CUS-A15", 1760),
    ("s15", "ORD-6116", "CUS-A16", 2870),
)

# 8 post-submit probe contexts. Never used in SFT training.
PROBE_TASKS = (
    ("p0", "ORD-7101", "CUS-B01", 1310),
    ("p1", "ORD-7102", "CUS-B02", 2420),
    ("p2", "ORD-7103", "CUS-B03", 3530),
    ("p3", "ORD-7104", "CUS-B04", 1640),
    ("p4", "ORD-7105", "CUS-B05", 2750),
    ("p5", "ORD-7106", "CUS-B06", 1860),
    ("p6", "ORD-7107", "CUS-B07", 2970),
    ("p7", "ORD-7108", "CUS-B08", 1080),
)

# Full-episode eval after a passed probe gate. 4 tasks × seeds 100–103.
EVAL_TASKS = (
    ("e0", "ORD-8101", "CUS-C01", 1710),
    ("e1", "ORD-8102", "CUS-C02", 2820),
    ("e2", "ORD-8103", "CUS-C03", 1930),
    ("e3", "ORD-8104", "CUS-C04", 2140),
)

GENERATION_SEEDS = (100, 101, 102, 103)

_FORBIDDEN = {
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


def sft_user_prompt(episode_id: str, order_id: str, customer_id: str) -> str:
    templates = {
        "s0": f"Please refund order {order_id}. Customer {customer_id}. The mug arrived chipped and I already recycled the box.",
        "s1": f"Account {customer_id}: cancel and refund {order_id}. Wrong voltage adapter shipped to me in Lisbon.",
        "s2": f"Hi — {customer_id} / {order_id}. The subscription trial charged early; put the money back on the debit card.",
        "s3": f"Need a refund for {order_id}, id {customer_id}. Fabric pills after one wash; photos are in the app.",
        "s4": f"Customer service: reverse {order_id} for {customer_id}. Gift order was never claimed at the locker.",
        "s5": f"{customer_id} here. Order {order_id} smells of chemicals; I cannot keep it. Original payment only.",
        "s6": f"Kindly process refund on {order_id} ({customer_id}). Size XXL was listed but XL arrived.",
        "s7": f"Refund request {order_id}. My number is {customer_id}. Courier marked delivered but porch cam is empty.",
        "s8": f"Formal note for finance: refund {order_id} under {customer_id} after the double capture on Amex.",
        "s9": f"Hey team, {order_id} for {customer_id} — battery swollen on day one. No replacement, just refund.",
        "s10": f"Please handle return funds for order {order_id}, customer {customer_id}. Color swatch mismatched the site.",
        "s11": f"{customer_id} requesting cash back on {order_id}. The ebook never unlocked in my library.",
        "s12": f"Support ticket reopen: refund {order_id} / {customer_id}. Missing HDMI cable in the kit.",
        "s13": f"I am {customer_id}. Refund {order_id}. Allergic reaction warning was not on the label.",
        "s14": f"Process a refund for {order_id}. Customer id {customer_id}. Event was cancelled by the venue.",
        "s15": f"Urgent: reverse charge {order_id} for {customer_id}. Fraud alert on the card already opened.",
        "p0": f"Refund {order_id} please. Customer {customer_id}. The stand does not fit the monitor model sold.",
        "p1": f"{customer_id} — money back for {order_id}. Package soaked in transit; contents ruined.",
        "p2": f"Handle refund on {order_id} ({customer_id}). Preorder slipped three months past the ETA.",
        "p3": f"Customer {customer_id} wants {order_id} refunded. Lens haze out of the sealed box.",
        "p4": f"Please refund order {order_id} for {customer_id}. Duplicate SKU billed after cart glitch.",
        "p5": f"Account {customer_id}: return funds for {order_id}. Assembly instructions missing half the steps.",
        "p6": f"Hi support, {order_id} / {customer_id}. The scent candle arrived melted into a brick.",
        "p7": f"Refund {order_id}, customer {customer_id}. Warranty void sticker was already scratched.",
        "e0": f"I need a refund for order {order_id}. Customer number {customer_id}. The zipper failed on first use.",
        "e1": f"Please reverse {order_id} for {customer_id}. Warehouse sent the floor demo unit by mistake.",
        "e2": f"Customer {customer_id} here about {order_id}. Cloud backup tier charged after I cancelled.",
        "e3": f"Refund request: {order_id}, id {customer_id}. Ceramic coated pan arrived with a hairline crack.",
    }
    try:
        return templates[episode_id]
    except KeyError as exc:
        raise KeyError(f"no sft prompt for {episode_id}") from exc


def assert_no_overlap() -> None:
    groups = {
        "train": TRAIN_TASKS,
        "probe": PROBE_TASKS,
        "eval": EVAL_TASKS,
    }
    seen_ids: dict[str, str] = {}
    seen_orders: dict[str, str] = {}
    for name, tasks in groups.items():
        for episode_id, order_id, customer_id, amount in tasks:
            if order_id in _FORBIDDEN:
                raise RuntimeError(f"{name} order {order_id} collides with prior splits")
            if episode_id in seen_ids:
                raise RuntimeError(f"episode overlap {episode_id}")
            if order_id in seen_orders:
                raise RuntimeError(f"order overlap {order_id}")
            seen_ids[episode_id] = name
            seen_orders[order_id] = name
            if amount <= 0:
                raise RuntimeError(f"bad amount for {episode_id}")
            _ = customer_id
