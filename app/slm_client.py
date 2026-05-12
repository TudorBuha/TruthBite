from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

import requests

from app.schemas import ProductData


DEFAULT_OLLAMA_TIMEOUT = 180.0

SYSTEM_PROMPT = """You are a senior Food Scientist specialized in NOVA food processing classification.
Return STRICT JSON (no markdown) with these exact keys:
  ingredient_steps: list of objects each with: ingredient, analysis, nova_marker, e_number, cited_function
  reasoning_summary: string
  predicted_nova_group: integer 1-4
Rules:
1) Analyse each ingredient step-by-step.
2) When an additive appears, cite its E-number and function from the EU additive context.
3) Be factually grounded and concise.
4) Output only valid JSON, no markdown fences."""


def build_prompt(product: ProductData, additive_context: str = "No retrieved additive context available.") -> str:
    user_message = "\n".join(
        [
            f"Product: {product.product_name or 'Unknown product'}",
            f"Country: {', '.join(product.countries) or 'Unknown'}",
            f"Ingredients: {product.ingredients_text or 'No ingredients listed'}",
            "EU additive context:",
            additive_context,
        ]
    )
    return f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n<|user|>\n{user_message}<|end|>\n<|assistant|>\n"


class OllamaClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL") or "truthbite-phi4"
        self.timeout = timeout if timeout is not None else float(os.getenv("OLLAMA_TIMEOUT", DEFAULT_OLLAMA_TIMEOUT))

    def is_model_available(self) -> bool:
        response = requests.get(f"{self.base_url}/api/tags", timeout=5.0)
        response.raise_for_status()
        models = response.json().get("models", [])
        return any(model.get("name", "").split(":")[0] == self.model for model in models)

    @staticmethod
    def _parse_json_response(raw_text: str) -> Dict[str, Any]:
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    def analyze(self, product: ProductData, additive_context: str = "No retrieved additive context available.") -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "stream": False,
                "prompt": build_prompt(product, additive_context=additive_context),
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        raw_text = payload.get("response", "")
        return self._parse_json_response(raw_text)
