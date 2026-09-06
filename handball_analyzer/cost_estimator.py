"""Grovt kostnadsanslag for en analysekjøring, vist FØR noe sendes til API-et.

Prisene under er hentet fra Anthropics offentlige prisliste på tidspunktet
dette ble skrevet og kan bli utdatert. Sjekk alltid gjeldende priser på
https://www.anthropic.com/pricing og faktisk forbruk på
https://console.anthropic.com før du kjører store analyser.
"""
from __future__ import annotations

import math
from typing import Tuple

# (input $/1M tokens, output $/1M tokens)
PRICING_PER_MILLION_TOKENS = {
    "claude-3-5-sonnet-latest": (3.00, 15.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-sonnet-20240620": (3.00, 15.00),
    "claude-3-5-haiku-latest": (0.80, 4.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-5": (5.00, 25.00),
}

# Brukes som konservativt (høyt) anslag for modeller vi ikke kjenner prisen på.
DEFAULT_PRICING = (3.00, 15.00)

# Anthropics tommelfingerregel for bildetokens: (bredde_px * høyde_px) / 750
TOKENS_PER_PIXEL_DIVISOR = 750

# Grovt anslag på tekst rundt hvert bilde og instruksjonstekst per batch.
TEXT_TOKENS_PER_FRAME = 20
TEXT_TOKENS_PER_BATCH_INSTRUCTION = 40
SYSTEM_PROMPT_TOKENS_ESTIMATE = 450
OUTPUT_TOKENS_PER_BATCH_ESTIMATE = 300


def estimate_frame_dimensions(
    original_width: int, original_height: int, max_dimension: int
) -> Tuple[int, int]:
    if original_width <= 0 or original_height <= 0:
        return max_dimension, max_dimension
    longest = max(original_width, original_height)
    if longest <= max_dimension:
        return original_width, original_height
    scale = max_dimension / longest
    return int(original_width * scale), int(original_height * scale)


def estimate_image_tokens(width: int, height: int) -> int:
    return max(1, math.ceil((width * height) / TOKENS_PER_PIXEL_DIVISOR))


def get_pricing(model: str) -> Tuple[float, float]:
    return PRICING_PER_MILLION_TOKENS.get(model, DEFAULT_PRICING)


def estimate_run(
    num_frames: int,
    batch_size: int,
    frame_width: int,
    frame_height: int,
    model: str,
) -> dict:
    """Grovt anslag på tokenforbruk og kostnad (USD) for en analysekjøring."""
    if num_frames <= 0:
        return {
            "num_frames": 0,
            "num_batches": 0,
            "image_tokens_per_frame": 0,
            "estimated_input_tokens": 0,
            "estimated_output_tokens": 0,
            "estimated_cost_usd": 0.0,
            "price_known": model in PRICING_PER_MILLION_TOKENS,
            "model": model,
        }

    num_batches = math.ceil(num_frames / batch_size)
    image_tokens_per_frame = estimate_image_tokens(frame_width, frame_height)

    total_image_tokens = num_frames * image_tokens_per_frame
    total_text_tokens = (
        num_frames * TEXT_TOKENS_PER_FRAME
        + num_batches * TEXT_TOKENS_PER_BATCH_INSTRUCTION
        + num_batches * SYSTEM_PROMPT_TOKENS_ESTIMATE
    )
    total_output_tokens = num_batches * OUTPUT_TOKENS_PER_BATCH_ESTIMATE

    input_price, output_price = get_pricing(model)
    input_tokens = total_image_tokens + total_text_tokens
    cost = (
        (input_tokens / 1_000_000) * input_price
        + (total_output_tokens / 1_000_000) * output_price
    )

    return {
        "num_frames": num_frames,
        "num_batches": num_batches,
        "image_tokens_per_frame": image_tokens_per_frame,
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": total_output_tokens,
        "estimated_cost_usd": round(cost, 2),
        "price_known": model in PRICING_PER_MILLION_TOKENS,
        "model": model,
    }


def format_estimate(estimate: dict) -> str:
    def thousands(n: int) -> str:
        return f"{n:,}".replace(",", " ")

    price_note = "" if estimate["price_known"] else "  (ukjent pris - bruker konservativt anslag)"
    lines = [
        "\n=== Kostnadsanslag (grovt, før analyse starter) ===",
        f"Modell: {estimate['model']}{price_note}",
        f"Antall frames: {thousands(estimate['num_frames'])}  "
        f"(API-kall/batcher: {thousands(estimate['num_batches'])})",
        f"Anslått input-tokens: ~{thousands(estimate['estimated_input_tokens'])}",
        f"Anslått output-tokens: ~{thousands(estimate['estimated_output_tokens'])}",
        f"Anslått kostnad: ~${estimate['estimated_cost_usd']:.2f} USD",
        "(Dette er kun et estimat, ikke en garanti - faktisk forbruk kan avvike. "
        "Sjekk gjeldende priser på anthropic.com/pricing og faktisk forbruk på "
        "console.anthropic.com.)",
    ]
    return "\n".join(lines)
