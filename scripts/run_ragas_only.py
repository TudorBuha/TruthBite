"""
Re-run the RAGAS phase only against a previously saved rag_eval_full.json.

Usage:
    # Smoke test (recommended first)
    python scripts/run_ragas_only.py --input results/rag_eval_full.json \\
        --n-samples 15 --metrics faithfulness --max-workers 1 --judge-backend ollama

    # API judge (best JSON compliance) — set GROQ_API_KEY or OPENAI_API_KEY
    python scripts/run_ragas_only.py --input results/rag_eval_full.json \\
        --n-samples 30 --metrics faithfulness,context_precision --judge-backend groq

    # Full local run (slower, more parse failures than API)
    python scripts/run_ragas_only.py --input results/rag_eval_full.json \\
        --max-workers 1 --timeout 600
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama3.1:8b")
DEFAULT_INPUT = PROJECT_ROOT / "results" / "rag_eval.json"
EMBEDDING_MODEL = os.getenv(
    "RAGAS_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

METRIC_CHOICES = (
    "faithfulness",
    "context_precision",
    "context_recall",
    "answer_correctness",
)


def subsample_rows(rows: List[Dict[str, Any]], n_samples: int, seed: int) -> List[Dict[str, Any]]:
    if n_samples >= len(rows):
        return rows
    from sklearn.model_selection import train_test_split

    labels = [r.get("ground_truth_nova") for r in rows]
    try:
        _, subset, _, _ = train_test_split(
            rows,
            labels,
            test_size=n_samples / len(rows),
            random_state=seed,
            stratify=labels,
        )
        return list(subset)
    except ValueError:
        _, subset, _, _ = train_test_split(
            rows,
            labels,
            test_size=n_samples / len(rows),
            random_state=seed,
            shuffle=True,
        )
        return list(subset)


def _truncate_text(text: str, max_chars: int) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _prepare_rows_for_ragas(
    rows: List[Dict[str, Any]],
    max_chars_answer: int,
    max_chars_context: int,
    max_contexts: int,
) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for row in rows:
        if not row.get("answer") or not row.get("contexts"):
            continue
        contexts = row.get("contexts") or []
        if isinstance(contexts, str):
            contexts = [contexts]
        contexts = [
            _truncate_text(c, max_chars_context)
            for c in list(contexts)[:max_contexts]
            if str(c).strip()
        ]
        if not contexts:
            continue
        prepared.append(
            {
                **row,
                "ingredients_text": _truncate_text(row.get("ingredients_text", ""), 2000),
                "answer": _truncate_text(row.get("answer", ""), max_chars_answer),
                "ground_truth_text": _truncate_text(
                    row.get("ground_truth_text", ""), max_chars_answer
                ),
                "contexts": contexts,
            }
        )
    return prepared


def _canonical_metric_name(name: str) -> str:
    lowered = str(name).lower().replace(" ", "_")
    for key in METRIC_CHOICES:
        if key in lowered:
            return key
    return lowered


def normalize_ragas_scores(result: Any) -> Dict[str, float]:
    if result is None:
        return {}

    scores: Dict[str, float] = {}

    if hasattr(result, "to_pandas"):
        try:
            df = result.to_pandas()
            if len(df) > 0:
                row = df.iloc[0]
                for col in df.columns:
                    val = row[col]
                    if val is None:
                        continue
                    try:
                        fv = float(val)
                    except (TypeError, ValueError):
                        continue
                    if fv != fv:
                        continue
                    scores[_canonical_metric_name(str(col))] = fv
            return scores
        except Exception:
            pass

    try:
        raw = dict(result)
    except Exception:
        return scores

    for key, value in raw.items():
        if value is None:
            continue
        try:
            fv = float(value)
        except (TypeError, ValueError):
            continue
        if fv != fv:
            continue
        scores[_canonical_metric_name(str(key))] = fv

    return scores


def _parse_metrics_arg(metrics_arg: str) -> List[str]:
    names = [m.strip().lower() for m in metrics_arg.split(",") if m.strip()]
    unknown = [m for m in names if m not in METRIC_CHOICES]
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}. Choose from: {', '.join(METRIC_CHOICES)}")
    return names


def _build_metric_objects(metric_names: Sequence[str]):
    from ragas.metrics import (
        AnswerCorrectness,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    factory = {
        "faithfulness": Faithfulness,
        "context_precision": ContextPrecision,
        "context_recall": ContextRecall,
        "answer_correctness": AnswerCorrectness,
    }
    return [factory[name]() for name in metric_names]


def _resolve_judge_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "ollama"


def _build_judge_llm(
    backend: str,
    model: Optional[str],
    ollama_url: str,
    timeout: float,
):
    from ragas.llms import LangchainLLMWrapper

    if backend == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:
            raise ImportError("pip install langchain-groq groq") from exc
        groq_model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        return LangchainLLMWrapper(
            ChatGroq(model=groq_model, temperature=0, timeout=int(timeout))
        )

    if backend == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            from langchain_community.chat_models import ChatOpenAI
        openai_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return LangchainLLMWrapper(
            ChatOpenAI(model=openai_model, temperature=0, timeout=timeout)
        )

    ollama_model = model or JUDGE_MODEL
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        from langchain_community.chat_models import ChatOllama
    return LangchainLLMWrapper(
        ChatOllama(
            model=ollama_model,
            base_url=ollama_url,
            temperature=0,
            timeout=timeout,
            num_ctx=8192,
            format="json",
        )
    )


def _build_embeddings():
    from ragas.embeddings import LangchainEmbeddingsWrapper

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )


def run_ragas(
    results: List[Dict[str, Any]],
    metric_names: Sequence[str],
    judge_backend: str,
    judge_model: Optional[str],
    ollama_url: str,
    condition_label: str = "",
    ollama_timeout: float = 600.0,
    ragas_timeout: int = 600,
    max_workers: int = 1,
    max_chars_answer: int = 6000,
    max_chars_context: int = 1500,
    max_contexts: int = 8,
) -> Dict[str, float]:
    ragas_samples = _prepare_rows_for_ragas(
        results, max_chars_answer, max_chars_context, max_contexts
    )
    if not ragas_samples:
        print(f"  [RAGAS] No samples with contexts for {condition_label}, skipping.")
        return {}

    backend = _resolve_judge_backend(judge_backend)
    print(
        f"  [RAGAS] {condition_label}: {len(ragas_samples)} samples, "
        f"metrics={list(metric_names)}, judge_backend={backend}, "
        f"max_workers={max_workers}, timeout={ragas_timeout}s"
    )

    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.dataset_schema import SingleTurnSample
        from ragas.run_config import RunConfig

        judge_llm = _build_judge_llm(backend, judge_model, ollama_url, ollama_timeout)
        judge_emb = _build_embeddings()

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
        metrics = _build_metric_objects(metric_names)
        for metric in metrics:
            metric.llm = judge_llm
            if hasattr(metric, "embeddings"):
                metric.embeddings = judge_emb

        run_config = RunConfig(
            timeout=ragas_timeout,
            max_workers=max_workers,
            max_retries=2,
        )
        result = evaluate(dataset=dataset, metrics=metrics, run_config=run_config)
        scores = normalize_ragas_scores(result)
        if scores:
            print(f"  [RAGAS] Done: {scores}")
        else:
            print(
                "  [RAGAS] WARNING: evaluate() finished but no numeric scores were parsed. "
                "Check judge JSON output / parser errors above."
            )
        return scores

    except ImportError as exc:
        print(f"  [RAGAS] Missing package: {exc}")
        return {}
    except Exception as exc:
        print(f"  [RAGAS] Failed: {exc}")
        return {}


def print_table(summary: Dict[str, Dict]) -> None:
    conditions = ["no_rag", "strategy_1", "strategy_2"]
    labels = ["SLM no RAG", "Strategy 1 (Dense)", "Strategy 2 (Hybrid)"]
    metrics_order = [
        ("nova_accuracy", "NOVA Accuracy"),
        ("faithfulness", "Faithfulness"),
        ("context_precision", "Context Precision"),
        ("context_recall", "Context Recall"),
        ("answer_correctness", "Answer Correctness"),
        ("citation_validity", "Citation Validity"),
        ("parse_failures", "Parse Failures"),
    ]
    col_w = 22
    sep = "=" * (25 + col_w * len(labels))
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
        description="Re-run RAGAS on saved rag_eval JSON (subset-friendly, API or Ollama judge)."
    )
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Stratified subsample size per condition (default: all rows with contexts).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Subsample seed (default: 42).")
    parser.add_argument(
        "--metrics",
        type=str,
        default="faithfulness,context_precision",
        help=f"Comma-separated metrics (default: faithfulness,context_precision). "
        f"Options: {', '.join(METRIC_CHOICES)}",
    )
    parser.add_argument(
        "--only",
        type=str,
        choices=("strategy_1", "strategy_2", "both"),
        default="both",
        help="Which RAG conditions to evaluate (default: both).",
    )
    parser.add_argument(
        "--judge-backend",
        type=str,
        choices=("auto", "ollama", "groq", "openai"),
        default="auto",
        help="LLM for RAGAS prompts. auto = groq if GROQ_API_KEY, else openai, else ollama.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Override judge model name (backend-specific default if omitted).",
    )
    parser.add_argument("--ollama-url", type=str, default=OLLAMA_URL)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-chars-answer", type=int, default=6000)
    parser.add_argument("--max-chars-context", type=int, default=1500)
    parser.add_argument("--max-contexts", type=int, default=8)
    args = parser.parse_args()

    metric_names = _parse_metrics_arg(args.metrics)
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found.")
        sys.exit(1)

    print(f"Loading {input_path} ...")
    with input_path.open(encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    details = data.get("details", {})

    conditions = ["strategy_1", "strategy_2"]
    if args.only != "both":
        conditions = [args.only]

    for key in conditions:
        rows = list(details.get(key, []))
        if not rows:
            print(f"No details found for {key}, skipping.")
            continue

        if args.n_samples is not None:
            rows = subsample_rows(rows, n_samples=args.n_samples, seed=args.seed)
            print(f"\n[{key}] Using stratified subsample n={len(rows)} (seed={args.seed})")

        print(f"\n[RAGAS] Condition: {key}")
        scores = run_ragas(
            rows,
            metric_names=metric_names,
            judge_backend=args.judge_backend,
            judge_model=args.judge_model,
            ollama_url=args.ollama_url,
            condition_label=key,
            ollama_timeout=float(args.timeout),
            ragas_timeout=args.timeout,
            max_workers=args.max_workers,
            max_chars_answer=args.max_chars_answer,
            max_chars_context=args.max_chars_context,
            max_contexts=args.max_contexts,
        )
        if scores:
            summary.setdefault(key, {}).update(scores)
            data["summary"] = summary
            with input_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            print(f"  Checkpoint saved → {input_path}")
        else:
            print(f"  No RAGAS scores written for {key} (see warnings above).")

    print_table(summary)
    print(f"\nUpdated results saved → {input_path}")


if __name__ == "__main__":
    main()
