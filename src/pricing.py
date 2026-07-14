"""USD-per-1K-token pricing table, used for cost estimation on every OpenAI call."""

from __future__ import annotations

# (input_price_per_1k, output_price_per_1k) in USD.
CHAT_PRICING = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
}
EMBEDDING_PRICING = {
    "text-embedding-3-small": 0.00002,
}


def estimate_chat_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    in_price, out_price = CHAT_PRICING.get(model, (0.0, 0.0))
    return (input_tokens / 1000) * in_price + (output_tokens / 1000) * out_price


def estimate_embedding_cost(model: str, total_tokens: int) -> float:
    price = EMBEDDING_PRICING.get(model, 0.0)
    return (total_tokens / 1000) * price
