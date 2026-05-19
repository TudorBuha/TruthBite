"""
Re-run the RAGAS phase only against a previously saved rag_eval_full.json.

Usage:
    python scripts/run_ragas_only.py
    python scripts/run_ragas_only.py --input data/rag_eval_full.json
    python scripts/run_ragas_only.py --judge-model llama3.1:8b --ollama-url http://localhost:11434
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OLLAMA_URL  = os.getenv("OLLAMA_URL",  "http://localhost:11434")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama3.1:8b")
DEFAULT_INPUT = PROJECT_ROOT / "results" / "rag_eval.json"


def run_ragas(
    results: List[Dict],
    judge_model: str,
    ollama_url: str,
    condition_label: str = "",
) -> Dict[str, float]:
    ragas_samples = [r for r in results if r.get("answer") and r.get("contexts")]
    if not ragas_samples:
        print(f"  [RAGAS] No samples with contexts for {condition_label}, skipping.")
        return {}

    print(f"  [RAGAS] {condition_label}: {len(ragas_samples)} samples, judge={judge_model}")

    try:
        from ragas import evaluate, EvaluationDataset
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import Faithfulness, ContextPrecision, ContextRecall, AnswerCorrectness
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

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
        print(f"  [RAGAS] Failed: {exc}")
        return {}


def print_table(summary: Dict[str, Dict]) -> None:
    conditions = ["no_rag", "strategy_1", "strategy_2"]
    labels     = ["SLM no RAG", "Strategy 1 (Dense)", "Strategy 2 (Hybrid)"]
    metrics_order = [
        ("nova_accuracy",      "NOVA Accuracy"),
        ("faithfulness",       "Faithfulness"),
        ("context_precision",  "Context Precision"),
        ("context_recall",     "Context Recall"),
        ("answer_correctness", "Answer Correctness"),
        ("citation_validity",  "Citation Validity"),
        ("parse_failures",     "Parse Failures"),
    ]
    col_w = 22
    sep    = "=" * (25 + col_w * len(labels))
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
    parser = argparse.ArgumentParser(description="Re-run RAGAS on saved rag_eval_full.json.")
    parser.add_argument("--input",       type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--judge-model", type=str, default=JUDGE_MODEL)
    parser.add_argument("--ollama-url",  type=str, default=OLLAMA_URL)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found.")
        sys.exit(1)

    print(f"Loading {input_path} ...")
    with input_path.open(encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    details = data.get("details", {})

    for key in ("strategy_1", "strategy_2"):
        rows = details.get(key, [])
        if not rows:
            print(f"No details found for {key}, skipping.")
            continue
        print(f"\n[RAGAS] Condition: {key}")
        scores = run_ragas(rows, judge_model=args.judge_model, ollama_url=args.ollama_url, condition_label=key)
        if scores:
            summary.setdefault(key, {}).update(scores)
            # Checkpoint after each condition
            data["summary"] = summary
            with input_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            print(f"  Checkpoint saved → {input_path}")

    print_table(summary)
    print(f"\nUpdated results saved → {input_path}")


if __name__ == "__main__":
    main()
