from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from data_pipeline import EU_ADDITIVES_CACHE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "synthetic_cot_dataset.jsonl"


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _iter_ingredient_steps(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Support both list steps and legacy single-object steps from older exports."""
    steps = (trace or {}).get("ingredient_steps")
    if steps is None:
        return []
    if isinstance(steps, list):
        return [s for s in steps if isinstance(s, dict)]
    if isinstance(steps, dict):
        if "ingredient" in steps or "analysis" in steps:
            return [steps]
    return []


def count_non_empty_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
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


def check_citation_accuracy(
    records: List[Dict[str, Any]],
    additives: Dict[str, Dict[str, Any]],
) -> Tuple[int, int, int]:
    total_citations = 0
    valid_citations = 0
    unknown_enumbers = 0

    for rec in records:
        trace = rec.get("trace", {}) or {}
        for step in _iter_ingredient_steps(trace):
            e_number = str(step.get("e_number") or "").strip().upper()
            cited_function = _normalize_text(step.get("cited_function"))
            if not e_number:
                continue
            total_citations += 1
            if e_number not in additives:
                unknown_enumbers += 1
                continue
            known_functions = [
                _normalize_text(fn) for fn in additives[e_number].get("functions", []) if _normalize_text(fn)
            ]
            if not cited_function:
                # Count empty function as valid only if source has no known functions.
                if not known_functions:
                    valid_citations += 1
                continue
            if not known_functions or any(cited_function in fn for fn in known_functions):
                valid_citations += 1

    return total_citations, valid_citations, unknown_enumbers


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "TruthBite synthetic dataset QA: NOVA distribution, validation flags, and step-level E-number "
            "citation checks against the local additives cache. Run after generate_dataset.py (e.g. next morning)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/analyze_dataset.py --dataset-path data/processed/synthetic_cot_dataset.jsonl\n"
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        metavar="PATH",
        help="JSON Lines file from generate_dataset.py (default: data/processed/synthetic_cot_dataset.jsonl).",
    )
    parser.add_argument(
        "--additives-cache",
        type=Path,
        default=EU_ADDITIVES_CACHE,
        metavar="PATH",
        help="Normalized additives JSON cache (default: data/raw/eu_food_additives.json).",
    )
    args = parser.parse_args()

    if not args.dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_path}")
    if not args.additives_cache.exists():
        raise FileNotFoundError(f"Additives cache not found: {args.additives_cache}")

    line_estimate = count_non_empty_lines(args.dataset_path)
    records = load_jsonl(args.dataset_path)
    additives = json.loads(args.additives_cache.read_text(encoding="utf-8"))

    total = len(records)
    nova_counts = Counter()
    label_match_true = 0
    enum_ok_true = 0

    for rec in records:
        nova_counts[str(rec.get("ground_truth_nova_group", "unknown"))] += 1
        validation = rec.get("validation", {}) or {}
        if validation.get("nova_label_match") is True:
            label_match_true += 1
        if validation.get("enum_citation_ok") is True:
            enum_ok_true += 1

    total_citations, valid_citations, unknown_enumbers = check_citation_accuracy(records, additives)

    print("=== TruthBite Synthetic Dataset Report ===")
    print(f"Dataset: {args.dataset_path}")
    if line_estimate != total:
        print(f"Warning: non-empty lines={line_estimate}, parsed records={total} (some lines may be invalid JSON).")
    print(f"Records: {total}")
    print("")
    print("NOVA distribution (ground-truth):")
    for nova in sorted(nova_counts.keys()):
        count = nova_counts[nova]
        pct = (count / total * 100.0) if total else 0.0
        print(f"  NOVA {nova}: {count} ({pct:.2f}%)")
    print("")
    print("Validation flags:")
    print(f"  nova_label_match=True: {label_match_true}/{total} ({(label_match_true/total*100.0) if total else 0.0:.2f}%)")
    print(f"  enum_citation_ok=True: {enum_ok_true}/{total} ({(enum_ok_true/total*100.0) if total else 0.0:.2f}%)")
    print("")
    print("Citation accuracy (step-level):")
    print(f"  Total E-number citations: {total_citations}")
    print(f"  Valid citations: {valid_citations}")
    print(f"  Unknown E-numbers: {unknown_enumbers}")
    acc = (valid_citations / total_citations * 100.0) if total_citations else 0.0
    print(f"  Citation accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()
