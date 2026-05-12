from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from app.schemas import ProductData


OFF_API_URL = "https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
OFF_USER_AGENT = "TruthBite/0.1 (student demo; Open Food Facts API)"


class OpenFoodFactsError(RuntimeError):
    """Raised when Open Food Facts cannot return a usable product."""


def _split_tags(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _first_text(product: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = product.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def normalize_product(barcode: str, payload: Dict[str, Any]) -> ProductData:
    product = payload.get("product") or {}
    nova_value: Optional[int] = None
    raw_nova = product.get("nova_group") or product.get("nova_groups")
    try:
        if raw_nova is not None:
            nova_value = int(raw_nova)
    except (TypeError, ValueError):
        nova_value = None

    return ProductData(
        barcode=barcode,
        product_name=_first_text(product, "product_name", "product_name_en", "generic_name"),
        brands=_first_text(product, "brands"),
        ingredients_text=_first_text(product, "ingredients_text", "ingredients_text_en"),
        labels=_split_tags(product.get("labels")) or _split_tags(product.get("labels_tags")),
        categories=_split_tags(product.get("categories")) or _split_tags(product.get("categories_tags")),
        countries=_split_tags(product.get("countries")) or _split_tags(product.get("countries_tags")),
        nova_group=nova_value,
        nutriscore_grade=product.get("nutriscore_grade"),
        image_url=product.get("image_front_url") or product.get("image_url"),
        source_url=f"https://world.openfoodfacts.org/product/{barcode}",
    )


def fetch_product_by_barcode(barcode: str, timeout: float = 10.0) -> ProductData:
    clean_barcode = barcode.strip()
    if not clean_barcode:
        raise OpenFoodFactsError("Barcode is required.")

    response = requests.get(
        OFF_API_URL.format(barcode=clean_barcode),
        headers={"User-Agent": OFF_USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != 1:
        raise OpenFoodFactsError(f"No Open Food Facts product found for barcode {clean_barcode}.")
    return normalize_product(clean_barcode, payload)
