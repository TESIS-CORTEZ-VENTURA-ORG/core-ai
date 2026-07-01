"""LLM prompt builder for restaurant document extraction (E11 Smart Onboarding).

Centralised here so every adapter sends the same instructions.
The prompt is intentionally strict and conservative: the LLM is told to ONLY
extract items that are clearly present, never invent prices, and output raw JSON.
"""

from __future__ import annotations

EXTRACT_SYSTEM = """\
You are a restaurant data extraction specialist for a Peruvian restaurant management system.

Given raw text from a restaurant document (menu PDF, price list, Excel spreadsheet),
extract ONLY items that are CLEARLY AND EXPLICITLY present in the text.

STRICT OUTPUT RULES — output exactly one raw JSON object, nothing else:
1. Output ONLY valid JSON — no markdown, no code fences (```), no explanations, no commentary.
2. DO NOT invent or estimate prices. If a price is not clearly stated in the text → omit that item from menuItems.
3. For ingredients, estimatedCost is OPTIONAL — include it only if a cost/price appears next to that ingredient in the text.
4. Menu item prices must be non-negative numbers (the currency is {currency}).
5. Unit for ingredients must be a real measurement unit (kg, g, litro, ml, unidad, porcion, etc.).
6. All names must be non-empty strings (at least 1 character).
7. Only return items that appear VERBATIM or nearly verbatim in the provided text.
8. Be CONSERVATIVE: if you are uncertain about a price or name, omit the item entirely.
9. category and description for menu items are optional — include only if clearly present.
10. If the document contains no extractable items, return empty arrays.

Required JSON structure (output this and ONLY this):
{{
  "menuItems": [
    {{"name": "Dish Name", "price": 25.50, "category": "Category Name", "description": "Optional description"}}
  ],
  "ingredients": [
    {{"name": "Ingredient Name", "unit": "kg", "estimatedCost": 8.00}}
  ]
}}

Target: {target_instruction}
"""

_TARGET_INSTRUCTIONS: dict[str, str] = {
    "menu": "Extract ONLY menu items (dishes with prices). Leave 'ingredients' as an empty array [].",
    "ingredients": "Extract ONLY ingredients/insumos. Leave 'menuItems' as an empty array [].",
    "auto": "Extract BOTH menu items (dishes with prices) AND ingredients/insumos if present.",
}


def build_extract_prompt(
    text: str,
    target: str,
    currency: str = "PEN",
    max_chars: int = 40_000,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the extract LLM call.

    The text is truncated to max_chars to keep token costs manageable.
    A restaurant menu rarely exceeds 40k characters of plain text.
    """
    target_instruction = _TARGET_INSTRUCTIONS.get(target, _TARGET_INSTRUCTIONS["auto"])
    system = EXTRACT_SYSTEM.format(
        currency=currency,
        target_instruction=target_instruction,
    )
    # Truncate document text to avoid excessive token usage.
    truncated = text[:max_chars]
    suffix = (
        f"\n[Document truncated at {max_chars} characters]"
        if len(text) > max_chars
        else ""
    )
    user = (
        f"Restaurant document text:\n\n{truncated}{suffix}\n\n"
        "Extract the requested data and output ONLY the JSON object described above."
    )
    return system, user
