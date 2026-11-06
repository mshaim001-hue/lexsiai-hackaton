"""Local refund harness.

lookup_order and get_customer each return a different token for this episode.
calculate_refund requires both of those tokens and returns a quote_token.
submit_refund accepts only that quote_token together with the matching
amount and identifiers. A token from another episode, or a skipped step,
is rejected. The amount and the tokens are not placed in the user prompt.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


def episode_token(seed: int, episode_id: str, kind: str) -> str:
    digest = hashlib.sha256(f"{seed}:{episode_id}:{kind}".encode()).hexdigest()[:16]
    return f"{kind}_{digest}"


@dataclass
class RefundEpisode:
    episode_id: str
    order_id: str
    customer_id: str
    amount_cents: int
    seed: int
    order_token: str = ""
    customer_token: str = ""
    quote_token: str = ""
    lookup_done: bool = False
    customer_done: bool = False
    quote_done: bool = False
    submitted: bool = False
    # Names of implementations that actually ran. The schema guard must not append.
    function_entries: list[str] = field(default_factory=list)
    # Structured tool results, appended before any str() conversion.
    structured_results: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.order_token = episode_token(self.seed, self.episode_id, "order")
        self.customer_token = episode_token(self.seed, self.episode_id, "customer")
        self.quote_token = episode_token(self.seed, self.episode_id, "quote")

    def hidden_strings(self) -> list[str]:
        return [
            self.order_token,
            self.customer_token,
            self.quote_token,
            str(self.amount_cents),
        ]

    def lookup_order(self, order_id: str) -> dict:
        self.function_entries.append("lookup_order")
        if order_id != self.order_id:
            return self._tool_error("lookup_order", "no such order")
        self.lookup_done = True
        return {
            "ok": True,
            "tool": "lookup_order",
            "execution": "success",
            "order_or_token": False,
            "order_context_token": self.order_token,
        }

    def get_customer(self, customer_id: str) -> dict:
        self.function_entries.append("get_customer")
        if customer_id != self.customer_id:
            return self._tool_error("get_customer", "no such customer")
        self.customer_done = True
        return {
            "ok": True,
            "tool": "get_customer",
            "execution": "success",
            "order_or_token": False,
            "customer_validation_token": self.customer_token,
        }

    def calculate_refund(
        self, order_context_token: str, customer_validation_token: str
    ) -> dict:
        self.function_entries.append("calculate_refund")
        tokens_ok = (
            order_context_token == self.order_token
            and customer_validation_token == self.customer_token
        )
        steps_ok = self.lookup_done and self.customer_done
        if not tokens_ok or not steps_ok:
            return {
                "ok": False,
                "tool": "calculate_refund",
                "execution": "order_or_token",
                "order_or_token": True,
                "reason": "both tokens from this episode are required",
            }
        self.quote_done = True
        return {
            "ok": True,
            "tool": "calculate_refund",
            "execution": "success",
            "order_or_token": False,
            "quote_token": self.quote_token,
            "amount_cents": self.amount_cents,
        }

    def submit_refund(
        self,
        quote_token: str,
        order_id: str,
        customer_id: str,
        amount_cents: int,
    ) -> dict:
        self.function_entries.append("submit_refund")
        quote_ok = quote_token == self.quote_token and self.quote_done
        ids_ok = (
            order_id == self.order_id
            and customer_id == self.customer_id
            and amount_cents == self.amount_cents
        )
        if not quote_ok or not ids_ok:
            return {
                "ok": False,
                "tool": "submit_refund",
                "execution": "order_or_token",
                "order_or_token": True,
                "reason": "quote_token, amount, or identifiers do not match this episode",
            }
        self.submitted = True
        return {
            "ok": True,
            "tool": "submit_refund",
            "execution": "success",
            "order_or_token": False,
            "submitted": True,
        }

    def _tool_error(self, tool: str, reason: str) -> dict:
        return {
            "ok": False,
            "tool": tool,
            "execution": "tool_error",
            "order_or_token": False,
            "reason": reason,
        }


def make_tools(episode: RefundEpisode) -> list:
    def lookup_order(order_id: str) -> str:
        """Return this episode's order context token for an existing order.

        Args:
            order_id: Order identifier to look up.

        Returns:
            The order context token for this episode.
        """
        return episode.lookup_order(order_id)

    def get_customer(customer_id: str) -> str:
        """Return this episode's customer validation token.

        Args:
            customer_id: Customer identifier to look up.

        Returns:
            The customer validation token for this episode.
        """
        return episode.get_customer(customer_id)

    def calculate_refund(
        order_context_token: str, customer_validation_token: str
    ) -> str:
        """Calculate a refund from both tokens issued for this episode.

        Args:
            order_context_token: Token returned by lookup_order for this episode.
            customer_validation_token: Token returned by get_customer for this episode.

        Returns:
            The quote token and the refund amount in cents.
        """
        return episode.calculate_refund(order_context_token, customer_validation_token)

    def submit_refund(
        quote_token: str, order_id: str, customer_id: str, amount_cents: int
    ) -> str:
        """Submit a refund when the quote, identifiers, and amount match.

        Args:
            quote_token: Quote token returned by calculate_refund.
            order_id: Order identifier for this refund.
            customer_id: Customer identifier for this refund.
            amount_cents: Refund amount in cents.

        Returns:
            Whether the refund was submitted.
        """
        return episode.submit_refund(quote_token, order_id, customer_id, amount_cents)

    return [lookup_order, get_customer, calculate_refund, submit_refund]


# Passed to create_rollout_fn(tools_fallback_prompt=...). The JSON-list sentence
# and the {tools} schema slot are the pinned AgentTune wording. The sentence
# that told the model to answer in plain text after a tool result is replaced
# by the call contract. Doubled braces survive str.format(tools=...).
TOOLS_FALLBACK_PROMPT = (
    "You can call these tools:\n{tools}\n\n"
    "To call tools, reply with only a JSON list, one object per call, and no other text: "
    '[{{"tool_name": "<name>", "parameters": {{"<argument>": <value>}}}}]. '
    "Use a list even for one call. "
    'A user turn that starts with "Tool result" is an observation, not a new message from the user. '
    "After an observation you may continue with more tool calls. "
    "Take arguments of a dependent call only from observations you have already received. "
    "Do not put a dependent call in the same response as the call whose result it needs. "
    "After submit_refund is confirmed, the final answer must be exactly "
    "REFUND_SUBMITTED amount_cents=<integer>."
)

# Passed to create_rollout_fn(tool_result_format=...). This is the text folded
# into each observation. {name} and {content} are the AgentTune slots.
TOOL_RESULT_FORMAT = (
    "Tool result for {name}: {content}\n"
    "This is an observation. You may continue with more tool calls. "
    "Take arguments of a dependent call only from observations you have already received. "
    "Do not put a dependent call in the same response as the call whose result it needs. "
    "After submit_refund is confirmed, the final answer must be exactly "
    "REFUND_SUBMITTED amount_cents=<integer>."
)

SYSTEM_PROMPT = (
    "You are a refund agent. Use the tools. "
    "Do not invent tokens or amounts. "
    "The refund amount is unknown until a tool returns it. "
    "Tool results are observations, and you may continue calling tools after one. "
    "Take arguments of a dependent call only from observations you have already received. "
    "Do not put a dependent call in the same response as the call whose result it needs. "
    "After submit_refund is confirmed, the final answer must be exactly "
    "REFUND_SUBMITTED amount_cents=<integer>."
)


def user_prompt(order_id: str, customer_id: str) -> str:
    return (
        f"Refund order {order_id} for customer {customer_id}. "
        "Look up the order, look up the customer, calculate the refund, "
        "then submit it. After submit_refund is confirmed, finish with exactly "
        "REFUND_SUBMITTED amount_cents=<integer>."
    )


def final_answer_text(amount_cents: int) -> str:
    return f"REFUND_SUBMITTED amount_cents={amount_cents}"


# Special tokens of the pinned Tiny Aya tokenizer. Only a trailing run of these
# tokens may surround the final answer. A token inside the answer is not removed.
ENDING_TOKENS = (
    "<PAD>",
    "<MASK_TOKEN>",
    "<BOS_TOKEN>",
    "<EOS_TOKEN>",
    "<UNK>",
    "<|START_OF_TURN_TOKEN|>",
    "<|END_OF_TURN_TOKEN|>",
    "<|USER_TOKEN|>",
    "<|CHATBOT_TOKEN|>",
    "<|SYSTEM_TOKEN|>",
    "<|NEW_FILE|>",
    "<|BEGINNING_OF_PREFIX_FIM_TOKEN|>",
    "<|BEGINNING_OF_MIDDLE_FIM_TOKEN|>",
    "<|BEGINNING_OF_SUFFIX_FIM_TOKEN|>",
    "<|END_OF_MIDDLE_FIM_TOKEN|>",
    "<|START_RESPONSE|>",
    "<|END_RESPONSE|>",
    "<|START_ACTION|>",
    "<|END_ACTION|>",
    "<|START_TOOL_RESULT|>",
    "<|END_TOOL_RESULT|>",
    "<|START_THINKING|>",
    "<|END_THINKING|>",
)


def answer_matches(text: str | None, amount_cents: int) -> bool:
    """True only when the whole answer is the required line.

    Leading and trailing whitespace is ignored. Trailing special tokens of the
    pinned tokenizer are ignored. Any other character makes the answer fail.
    """
    if text is None:
        return False
    body = text.strip()
    tokens = sorted(ENDING_TOKENS, key=len, reverse=True)
    progressed = True
    while progressed and body:
        progressed = False
        for token in tokens:
            if body.endswith(token):
                body = body[: -len(token)].rstrip()
                progressed = True
                break
    return body == final_answer_text(amount_cents)
