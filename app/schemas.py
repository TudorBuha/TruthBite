from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ProductData(BaseModel):
    barcode: Optional[str] = None
    product_name: str = ""
    brands: str = ""
    ingredients_text: str = ""
    labels: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    countries: List[str] = Field(default_factory=list)
    nova_group: Optional[int] = None
    nutriscore_grade: Optional[str] = None
    image_url: Optional[str] = None
    source_url: Optional[str] = None


class GreenwashingFlag(BaseModel):
    claim: str
    issue: str
    evidence: List[str] = Field(default_factory=list)
    severity: str = "medium"


class AnalysisRequest(BaseModel):
    barcode: Optional[str] = None
    product: Optional[ProductData] = None
    use_model: bool = True


class AnalysisResponse(BaseModel):
    product: ProductData
    predicted_nova_group: Optional[int] = None
    reasoning_summary: str
    greenwashing_flags: List[GreenwashingFlag] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    model_output: Optional[Dict[str, Any]] = None
    warnings: List[str] = Field(default_factory=list)
