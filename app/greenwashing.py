from __future__ import annotations

import re
from typing import List, Tuple

from app.schemas import GreenwashingFlag, ProductData


CLAIM_KEYWORDS = {
    "natural": ["natural", "naturally", "all natural"],
    "clean label": ["clean label", "clean"],
    "healthy": ["healthy", "health", "fitness", "wellness"],
    "organic/bio": ["organic", "bio"],
    "no artificial": ["no artificial", "without artificial"],
}

PROCESSING_MARKERS = {
    "E-number additive": re.compile(r"\bE\s?\d{3,4}[a-z]?\b", re.IGNORECASE),
    "flavouring": re.compile(r"\bflavou?rings?\b", re.IGNORECASE),
    "sweetener": re.compile(r"\b(sweetener|aspartame|sucralose|acesulfame|saccharin)\b", re.IGNORECASE),
    "emulsifier": re.compile(r"\b(emulsifier|lecithin|mono- and diglycerides?)\b", re.IGNORECASE),
    "colour": re.compile(r"\b(colou?r|caramel|tartrazine)\b", re.IGNORECASE),
    "preservative": re.compile(r"\b(preservative|sorbate|benzoate|nitrite|sulphite|sulfite)\b", re.IGNORECASE),
    "stabiliser": re.compile(r"\b(stabili[sz]er|thickener|xanthan|guar gum)\b", re.IGNORECASE),
}


def _product_claim_text(product: ProductData) -> str:
    parts = [
        product.product_name,
        product.brands,
        " ".join(product.labels),
        " ".join(product.categories),
    ]
    return " ".join(part for part in parts if part).lower()


def _find_claims(product: ProductData) -> List[str]:
    claim_text = _product_claim_text(product)
    claims = []
    for claim, keywords in CLAIM_KEYWORDS.items():
        if any(keyword in claim_text for keyword in keywords):
            claims.append(claim)
    return claims


def find_processing_markers(ingredients_text: str) -> List[str]:
    markers = []
    for label, pattern in PROCESSING_MARKERS.items():
        if pattern.search(ingredients_text or ""):
            markers.append(label)
    return markers


def detect_greenwashing(product: ProductData) -> Tuple[List[GreenwashingFlag], List[str]]:
    claims = _find_claims(product)
    markers = find_processing_markers(product.ingredients_text)
    warnings: List[str] = []

    if not product.ingredients_text:
        warnings.append("Open Food Facts did not provide an ingredient list for this product.")

    flags: List[GreenwashingFlag] = []
    if product.nova_group == 4 and claims:
        flags.append(
            GreenwashingFlag(
                claim=", ".join(claims),
                issue="Marketing claims may conflict with an Open Food Facts NOVA 4 ultra-processed classification.",
                evidence=["Open Food Facts NOVA group: 4"] + markers,
                severity="high",
            )
        )

    if claims and markers:
        flags.append(
            GreenwashingFlag(
                claim=", ".join(claims),
                issue="Marketing language should be checked against processing markers in the ingredient list.",
                evidence=markers,
                severity="medium",
            )
        )

    if not claims and product.nova_group == 4 and markers:
        flags.append(
            GreenwashingFlag(
                claim="No explicit health/natural claim found",
                issue="The product still contains multiple ultra-processing markers.",
                evidence=markers,
                severity="low",
            )
        )

    return flags, warnings


def fallback_nova_prediction(product: ProductData) -> Tuple[int | None, str]:
    if product.nova_group:
        return product.nova_group, f"Using Open Food Facts NOVA group {product.nova_group} as the fallback verdict."

    markers = find_processing_markers(product.ingredients_text)
    if len(markers) >= 2:
        return 4, "Ingredient markers suggest an ultra-processed profile, but no official NOVA value was available."
    if markers:
        return 3, "One processing marker was found, so the fallback verdict is cautious rather than definitive."
    return None, "Not enough evidence to assign a NOVA group automatically."
