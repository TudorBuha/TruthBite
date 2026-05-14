from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.greenwashing import detect_greenwashing, fallback_nova_prediction
from app.off_client import OpenFoodFactsError, fetch_product_by_barcode
from app.schemas import AnalysisRequest, AnalysisResponse, ProductData
from app.slm_client import OllamaClient

# RAG pipeline - fallback if Qdrant is not available
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    from pipeline import retrieve_context, validate_citations
    _RAG_AVAILABLE = True
except Exception:
    _RAG_AVAILABLE = False


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

app = FastAPI(title="TruthBite", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "truthbite-app"}


@app.get("/api/model/status")
def model_status() -> dict:
    client = OllamaClient()
    try:
        available = client.is_model_available()
    except requests.RequestException as exc:
        return {
            "available": False,
            "model": client.model,
            "ollama_url": client.base_url,
            "detail": f"Ollama is not reachable: {exc}",
        }
    return {
        "available": available,
        "model": client.model,
        "ollama_url": client.base_url,
        "detail": "Model is available." if available else "Ollama is reachable, but the model is not installed.",
    }


@app.get("/api/product/{barcode}", response_model=ProductData)
def get_product(barcode: str) -> ProductData:
    try:
        return fetch_product_by_barcode(barcode)
    except OpenFoodFactsError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Open Food Facts request failed: {exc}") from exc


def _resolve_product(request: AnalysisRequest) -> ProductData:
    if request.product:
        return request.product
    if request.barcode:
        return get_product(request.barcode)
    raise HTTPException(status_code=400, detail="Provide either a barcode or a product payload.")


@app.post("/api/analyze", response_model=AnalysisResponse)
def analyze(request: AnalysisRequest) -> AnalysisResponse:
    product = _resolve_product(request)
    flags, warnings = detect_greenwashing(product)
    predicted_nova, reasoning = fallback_nova_prediction(product)
    model_output = None
    sources: List[str] = []

    if product.source_url:
        sources.append(product.source_url)

    # RAG: retrieve additive context before calling the model
    additive_context = "No retrieved additive context available."
    if _RAG_AVAILABLE and product.ingredients_text:
        try:
            country = product.countries[0] if product.countries else None
            additive_context, rag_sources = retrieve_context(
                ingredients_text=product.ingredients_text,
                country=country,
                strategy=2,
                top_k=5,
            )
            sources += rag_sources
        except Exception as exc:
            warnings.append(f"RAG retrieval failed, proceeding without context: {exc}")

    if request.use_model:
        try:
            model_output = OllamaClient().analyze(product, additive_context=additive_context)
            predicted_nova = model_output.get("predicted_nova_group", predicted_nova)
            reasoning = model_output.get("reasoning_summary", reasoning)

            if _RAG_AVAILABLE and model_output:
                try:
                    from qdrant_client import QdrantClient
                    import os
                    _qclient = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"), check_compatibility=False)
                    valid, citation_errors = validate_citations(
                        model_output.get("ingredient_steps", []), _qclient
                    )
                    if not valid:
                        warnings.append(
                            "Citation issues detected: " + "; ".join(citation_errors)
                        )
                except Exception:
                    pass

        except (requests.RequestException, ValueError) as exc:
            warnings.append(f"Ollama model call failed, using fallback analysis: {exc}")

    return AnalysisResponse(
        product=product,
        predicted_nova_group=predicted_nova,
        reasoning_summary=reasoning,
        greenwashing_flags=flags,
        sources=sources,
        model_output=model_output,
        warnings=warnings,
    )
