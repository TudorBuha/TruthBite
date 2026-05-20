"""
TruthBite — RAG evaluation script

Compares three inference conditions on a stratified 100-example subset of the
test split that was used for fine-tuning evaluation (seed=42, 80/20 split):

  Condition A — SLM with no RAG context
  Condition B — Strategy 1: dense-only retrieval
  Condition C — Strategy 2: hybrid dense+BM25 + guaranteed E-number lookup

Metrics collected:
  - NOVA Accuracy          (exact match vs ground_truth_nova_group)
  - Faithfulness           (RAGAS — are model claims grounded in context?)
  - Context Precision      (RAGAS — is retrieved context relevant?)
  - Context Recall         (RAGAS — does retrieved context cover the answer?)
  - Answer Correctness     (RAGAS — does the answer match ground truth?)
  - Citation Validity      (% of examples where all E-number citations check out)
  - Parse Failures         (how often the model output was not valid JSON)

Results are saved as JSON after each condition completes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.model_selection import train_test_split
from qdrant_client import QdrantClient

from app.schemas import ProductData
from app.slm_client import OllamaClient
from pipeline import retrieve_context, validate_citations

COT_PATH      = PROJECT_ROOT / "data" / "processed" / "synthetic_cot_dataset.jsonl"
RESULTS_DIR   = PROJECT_ROOT / "results"
QDRANT_URL    = os.getenv("QDRANT_URL",   "http://localhost:6333")
OLLAMA_URL    = os.getenv("OLLAMA_URL",   "http://localhost:11434")
JUDGE_MODEL   = os.getenv("JUDGE_MODEL",  "llama3.1:8b")
INFER_MODEL   = os.getenv("OLLAMA_MODEL", "truthbite-phi4")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _extract_product_name(raw: Any) -> str:
    if isinstance(raw, dict):
        return str(raw.get("text") or raw.get("lang") or "Unknown")
    return str(raw) if raw else "Unknown"


def _extract_ingredients_text(raw: Any) -> str:
    if isinstance(raw, str):
        match = re.search(r"'text':\s*'([^']+)'", raw)
        if match:
            return match.group(1)
        return raw
    return str(raw)


def load_test_split(n_samples: int = 100, seed: int = 42) -> List[Dict]:
    """
    Load CoT dataset and reproduce the identical 80/20 stratified split
    used in notebooks/evaluate_phi4_mini.ipynb, then take a stratified
    subsample of n_samples for speed.
    """
    all_records = _load_jsonl(COT_PATH)

    # Keep only records with the required fields
    records = [
        r for r in all_records
        if r.get("ground_truth_nova_group") is not None
        and r.get("ingredients_text")
    ]

    labels = [r["ground_truth_nova_group"] for r in records]

    # Reproduce the exact 80/20 split
    _, test_records, _, test_labels = train_test_split(
        records, labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )

    if n_samples >= len(test_records):
        return test_records

    # Stratified subsample
    _, subset, _, _ = train_test_split(
        test_records, test_labels,
        test_size=n_samples / len(test_records),
        random_state=seed,
        stratify=test_labels,
    )
    return subset


def record_to_product(rec: Dict) -> ProductData:
    country = rec.get("country") or "United Kingdom"
    return ProductData(
        product_name=_extract_product_name(rec.get("product_name", "")),
        ingredients_text=_extract_ingredients_text(rec.get("ingredients_text", "")),
        countries=[str(country)],
        nova_group=rec.get("ground_truth_nova_group"),
    )


def run_condition(
    records: List[Dict],
    strategy: Optional[int],
    qdrant_client: QdrantClient,
    label: str,
    inference_timeout: float = 180.0,
) -> List[Dict]:
    """
    Run one inference condition over all records.

    strategy=None  →  no RAG (plain model call)
    strategy=1     →  dense-only retrieval
    strategy=2     →  hybrid dense+BM25 + guaranteed E-number lookup
    """
    results: List[Dict] = []
    client = OllamaClient(timeout=inference_timeout)

    for i, rec in enumerate(records):
        product_name = _extract_product_name(rec.get("product_name", ""))
        print(f"  [{label}] {i+1}/{len(records)}: {product_name[:55]}")

        product = record_to_product(rec)
        ground_truth_nova = int(rec["ground_truth_nova_group"])

        # Ground truth text for RAGAS — use the validated reasoning from the trace
        trace = rec.get("trace") or {}
        ground_truth_text = (
            trace.get("reasoning_summary")
            or f"This product is NOVA group {ground_truth_nova}."
        )

        additive_context = "No retrieved additive context available."
        retrieved_chunks: List[str] = []

        if strategy is not None and product.ingredients_text:
            try:
                ctx, _ = retrieve_context(
                    ingredients_text=product.ingredients_text,
                    country=(product.countries[0] if product.countries else None),
                    strategy=strategy,
                    top_k=5,
                    qdrant_client=qdrant_client,
                )
                additive_context = ctx
                # Split into individual lines for RAGAS contexts list
                retrieved_chunks = [
                    line.strip()
                    for line in ctx.splitlines()
                    if line.strip() and not line.startswith("===")
                ]
            except Exception as exc:
                print(f"    [RAG error] {exc}")

        # Call the inference model
        model_output: Optional[Dict] = None
        parse_failed = False
        t0 = time.time()
        try:
            model_output = client.analyze(product, additive_context=additive_context)
        except Exception as exc:
            print(f"    [model error] {exc}")
            parse_failed = True
        elapsed = time.time() - t0
        print(f"    done in {elapsed:.0f}s")

        predicted_nova: Optional[int] = None
        answer_text = ""
        citation_valid: Optional[bool] = None

        if model_output:
            raw_nova = model_output.get("predicted_nova_group")
            try:
                predicted_nova = int(raw_nova) if raw_nova is not None else None
            except (ValueError, TypeError):
                pass
            answer_text = json.dumps(model_output)

            # Citation validation only makes sense when RAG context was injected
            if strategy is not None:
                try:
                    valid, _ = validate_citations(
                        model_output.get("ingredient_steps", []), qdrant_client
                    )
                    citation_valid = valid
                except Exception:
                    pass
        else:
            parse_failed = True

        nova_correct: Optional[bool] = (
            (predicted_nova == ground_truth_nova)
            if predicted_nova is not None
            else None
        )

        results.append({
            "product_name": product_name,
            "ingredients_text": product.ingredients_text,
            "ground_truth_nova": ground_truth_nova,
            "predicted_nova": predicted_nova,
            "nova_correct": nova_correct,
            "answer": answer_text,
            "contexts": retrieved_chunks,
            "ground_truth_text": ground_truth_text,
            "citation_valid": citation_valid,
            "parse_failed": parse_failed,
        })

    return results


# RAGAS evaluation 

def run_ragas(
    results: List[Dict],
    judge_model: str,
    ollama_url: str,
    condition_label: str = "",
) -> Dict[str, float]:
    """
    Run RAGAS faithfulness / context_precision / context_recall / answer_correctness.
    Uses llama3.1:8b (or the specified judge) via Ollama — zero API cost.

    Only called for conditions that include retrieved context (Strategy 1 & 2).
    """
    # Filter to examples where the model actually produced an answer with contexts
    ragas_samples = [r for r in results if r["answer"] and r["contexts"]]
    if not ragas_samples:
        print(f"  [RAGAS] No samples with contexts for {condition_label}, skipping.")
        return {}

    print(f"  [RAGAS] Running on {len(ragas_samples)} samples with judge={judge_model} ...")

    try:
        # RAGAS >= 0.2.x API
        from ragas import evaluate, EvaluationDataset
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import Faithfulness, ContextPrecision, ContextRecall, AnswerCorrectness
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        # Try langchain-ollama first (recommended), fall back to langchain-community
        try:
            from langchain_ollama import ChatOllama, OllamaEmbeddings
        except ImportError:
            from langchain_community.chat_models import ChatOllama
            from langchain_community.embeddings import OllamaEmbeddings

        judge_llm = LangchainLLMWrapper(
            ChatOllama(model=judge_model, base_url=ollama_url, temperature=0)
        )
        judge_emb = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(model=judge_model, base_url=ollama_url)
        )

        samples = [
            SingleTurnSample(
                user_input=r["ingredients_text"],
                retrieved_contexts=r["contexts"],
                response=r["answer"],
                reference=r["ground_truth_text"],
            )
            for r in ragas_samples
        ]
        dataset = EvaluationDataset(samples=samples)

        metrics = [Faithfulness(), ContextPrecision(), ContextRecall(), AnswerCorrectness()]
        for m in metrics:
            m.llm = judge_llm
            if hasattr(m, "embeddings"):
                m.embeddings = judge_emb

        result = evaluate(dataset=dataset, metrics=metrics)
        scores = dict(result)
        print(f"  [RAGAS] Done: {scores}")
        return scores

    except ImportError as exc:
        print(f"  [RAGAS] Missing package: {exc}")
        print("  Install: pip install ragas langchain-ollama")
        return {}
    except Exception as exc:
        print(f"  [RAGAS] Evaluation failed: {exc}")
        return {}


# metric helpers 

def nova_accuracy(results: List[Dict]) -> Optional[float]:
    valid = [r for r in results if r["nova_correct"] is not None]
    if not valid:
        return None
    return sum(r["nova_correct"] for r in valid) / len(valid)


def citation_validity_rate(results: List[Dict]) -> Optional[float]:
    valid = [r for r in results if r["citation_valid"] is not None]
    if not valid:
        return None
    return sum(r["citation_valid"] for r in valid) / len(valid)


def parse_failure_count(results: List[Dict]) -> str:
    n = sum(r["parse_failed"] for r in results)
    return f"{n}/{len(results)}"



def print_table(summary: Dict[str, Dict]) -> None:
    conditions = ["no_rag", "strategy_1", "strategy_2"]
    labels     = ["SLM no RAG", "Strategy 1 (Dense)", "Strategy 2 (Hybrid)"]

    metrics_order = [
        ("nova_accuracy",       "NOVA Accuracy"),
        ("faithfulness",        "Faithfulness"),
        ("context_precision",   "Context Precision"),
        ("context_recall",      "Context Recall"),
        ("answer_correctness",  "Answer Correctness"),
        ("citation_validity",   "Citation Validity"),
        ("parse_failures",      "Parse Failures"),
    ]

    col_w = 22
    sep   = "=" * (25 + col_w * len(labels))
    header = f"{'Metric':<25}" + "".join(f"{lbl:>{col_w}}" for lbl in labels)

    print(f"\n{sep}")
    print(header)
    print(sep)
    for key, display in metrics_order:
        row = f"{display:<25}"
        for cond in conditions:
            val = summary.get(cond, {}).get(key)
            if val is None:
                row += f"{'—':>{col_w}}"
            elif isinstance(val, float):
                row += f"{val:>{col_w}.4f}"
            else:
                row += f"{str(val):>{col_w}}"
        print(row)
    print(sep)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare SLM-only vs Strategy 1 vs Strategy 2 on the TruthBite test split."
    )
    parser.add_argument(
        "--n-samples", type=int, default=100,
        help="Number of stratified test examples to evaluate (default: 100).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save full JSON results (default: results/rag_eval.json).",
    )
    parser.add_argument(
        "--skip-ragas", action="store_true",
        help="Skip RAGAS metrics (only compute NOVA accuracy + citation rate). "
             "Useful for a quick sanity check or when llama3.1:8b is not available.",
    )
    parser.add_argument(
        "--judge-model", type=str, default=JUDGE_MODEL,
        help=f"Ollama model used as RAGAS judge (default: {JUDGE_MODEL}).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed — must match the fine-tuning split seed (default: 42).",
    )
    parser.add_argument(
        "--timeout", type=float, default=180.0,
        help="Ollama inference timeout in seconds per call (default: 180).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("TruthBite — RAG Evaluation")
    print("=" * 60)

    print(f"\nLoading {args.n_samples} stratified test examples from CoT dataset ...")
    records = load_test_split(n_samples=args.n_samples, seed=args.seed)
    nova_dist: Dict[int, int] = {}
    for r in records:
        g = r["ground_truth_nova_group"]
        nova_dist[g] = nova_dist.get(g, 0) + 1
    print(f"  Loaded {len(records)} examples. NOVA distribution: {dict(sorted(nova_dist.items()))}")

    print(f"\nConnecting to Qdrant at {QDRANT_URL} ...")
    qdrant_client = QdrantClient(url=QDRANT_URL, check_compatibility=False)
    try:
        colls = [c.name for c in qdrant_client.get_collections().collections]
        print(f"  Collections available: {colls}")
        if "additives_corpus" not in colls or "cot_corpus" not in colls:
            print("  WARNING: additives_corpus and/or cot_corpus not found.")
            print("  Run: python scripts/ingest.py")
    except Exception as exc:
        print(f"  WARNING: Cannot reach Qdrant — {exc}")
        print("  RAG conditions will fail. Ensure Qdrant is running and re-run.")

    output_path = Path(args.output) if args.output else RESULTS_DIR / "rag_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, List[Dict]] = {}

    conditions = [
        ("no_rag",     None, "A — SLM no RAG"),
        ("strategy_1", 1,    "B — Strategy 1 (dense only)"),
        ("strategy_2", 2,    "C — Strategy 2 (hybrid + guaranteed lookup)"),
    ]
    for key, strategy, description in conditions:
        print(f"\n[Condition {description}]  ({len(records)} examples)")
        results = run_condition(
            records,
            strategy=strategy,
            qdrant_client=qdrant_client,
            label=key,
            inference_timeout=args.timeout,
        )
        all_results[key] = results

        # Checkpoint save after each condition — safe to stop and resume manually
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"summary": {}, "details": all_results}, f, indent=2, default=str)
        print(f"  Checkpoint saved → {output_path}")

    ragas_scores: Dict[str, Dict] = {k: {} for k in all_results}

    if not args.skip_ragas:
        for key in ("strategy_1", "strategy_2"):
            print(f"\n[RAGAS] Condition {key} — judge: {args.judge_model}")
            ragas_scores[key] = run_ragas(
                all_results[key],
                judge_model=args.judge_model,
                ollama_url=OLLAMA_URL,
                condition_label=key,
            )
    else:
        print("\n[RAGAS] Skipped (--skip-ragas flag set).")

    summary: Dict[str, Dict] = {}
    for key, results in all_results.items():
        summary[key] = {
            "nova_accuracy":     nova_accuracy(results),
            "citation_validity": citation_validity_rate(results),
            "parse_failures":    parse_failure_count(results),
            **ragas_scores.get(key, {}),
        }

    print_table(summary)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": all_results}, f, indent=2, default=str)
    print(f"\nFull results saved → {output_path}")
    print(
        "\nCopy the table above into README.md or CLAUDE.md under "
        '"Experiments and Results".'
    )


if __name__ == "__main__":
    main()
