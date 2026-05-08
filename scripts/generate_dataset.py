from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import ollama
import pandas as pd
from tqdm import tqdm

from data_pipeline import (
    DEFAULT_EU_ADDITIVES_URL,
    EU_ADDITIVES_CACHE,
    apply_limit,
    fetch_or_load_eu_additives,
    load_open_food_facts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "synthetic_cot_dataset.jsonl"

ENUM_RE = re.compile(r"\bE\d{3,4}[A-Z]?\b", re.IGNORECASE)


SYSTEM_PROMPT = """
You are a senior Food Scientist specialized in NOVA food processing classification.
Return STRICT JSON with keys:
{
  "ingredient_steps": [
    {"ingredient": "...", "analysis": "...", "nova_marker": "...", "e_number": "E### or null", "cited_function": "..." }
  ],
  "reasoning_summary": "...",
  "predicted_nova_group": 1|2|3|4
}
Rules:
1) Analyze ingredients step-by-step.
2) If an additive appears, cite the E-number function.
3) Be factually grounded and concise.
4) predicted_nova_group MUST match the provided ground truth.
5) Output only valid JSON, no markdown.
""".strip()


def ensure_output_dir() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def parse_ingredients(ingredients_text: str) -> List[str]:
    return [part.strip() for part in str(ingredients_text).split(",") if part.strip()]


def _scalarize(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return _scalarize(value[0]) if value else None
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            converted = value.tolist()
            if isinstance(converted, list):
                return _scalarize(converted[0]) if converted else None
            return converted
        except Exception:  # pylint: disable=broad-except
            return value
    return value


def _json_safe(value: Any) -> Any:
    """
    Make values JSON-serializable without collapsing lists.
    (_scalarize is only for OFF row fields like product_name arrays — not for model traces.)
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return _json_safe(value.tolist())
        except Exception:  # pylint: disable=broad-except
            return str(value)
    return str(value)


def build_source_key(row: pd.Series) -> str:
    barcode = str(_scalarize(row.get("code")) or "missing-barcode")
    nova_group = int(_scalarize(row.get("nova_group")) or -1)
    ingredients = str(
        _scalarize(row.get("ingredients_clean")) or _scalarize(row.get("ingredients_text")) or ""
    ).strip()
    digest = hashlib.sha1(ingredients.encode("utf-8")).hexdigest()[:16]
    return f"{barcode}::nova{nova_group}::{digest}"


def load_checkpoint_keys(output_path: Path) -> Set[str]:
    if not output_path.exists():
        return set()
    keys: Set[str] = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            source_key = obj.get("source_key")
            if source_key:
                keys.add(str(source_key))
                continue
            # Backward compatibility for older records without source_key.
            barcode = obj.get("barcode")
            nova_group = obj.get("ground_truth_nova_group")
            ingredients = obj.get("ingredients_text", "")
            fallback_digest = hashlib.sha1(str(ingredients).encode("utf-8")).hexdigest()[:16]
            keys.add(f"{barcode}::nova{nova_group}::{fallback_digest}")
    return keys


def build_user_prompt(row: pd.Series, eu_additives: Dict[str, Dict[str, Any]]) -> str:
    ingredients = _scalarize(row.get("ingredients_clean")) or _scalarize(row.get("ingredients_text")) or ""
    product = _scalarize(row.get("product_name", "Unknown Product")) or "Unknown Product"
    country = _scalarize(row.get("countries_en", "Unknown")) or "Unknown"
    ground_truth = int(row["nova_group"])

    present_codes = sorted({m.upper() for m in ENUM_RE.findall(str(ingredients).replace("-", ""))})
    additive_context = []
    for code in present_codes:
        if code in eu_additives:
            entry = eu_additives[code]
            additive_context.append(
                f"{code}: name={entry.get('name','')}, functions={entry.get('functions', [])}"
            )
    additive_block = "\n".join(additive_context) if additive_context else "No explicit E-number found."

    return (
        f"Product: {product}\n"
        f"Country: {country}\n"
        f"Ingredients: {ingredients}\n"
        f"Ground truth NOVA group: {ground_truth}\n"
        f"EU additive context:\n{additive_block}\n"
    )


def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def verify_nova_label(trace: Dict[str, Any], ground_truth: int) -> bool:
    return int(trace.get("predicted_nova_group", -1)) == int(ground_truth)


def normalize_trace_ingredient_steps(trace: Dict[str, Any]) -> None:
    """Coerce a single step object into a one-element list (common LLM mistake)."""
    steps = trace.get("ingredient_steps")
    if isinstance(steps, dict) and ("ingredient" in steps or "analysis" in steps):
        trace["ingredient_steps"] = [steps]


def verify_enumber_citations(trace: Dict[str, Any], eu_additives: Dict[str, Dict[str, Any]]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    steps = trace.get("ingredient_steps", [])
    if not isinstance(steps, list):
        return False, ["ingredient_steps must be a list of step objects"]

    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"step {idx} is not an object")
            continue

        e_number = str(step.get("e_number") or "").strip().upper()
        cited_function = str(step.get("cited_function") or "").strip().lower()
        if not e_number:
            continue
        if e_number not in eu_additives:
            errors.append(f"{e_number} not found in local EU additives DB")
            continue
        known_functions = [
            str(fn).strip().lower() for fn in eu_additives[e_number].get("functions", []) if str(fn).strip()
        ]
        if cited_function and known_functions:
            if all(cited_function not in known for known in known_functions):
                errors.append(
                    f"{e_number} cited as '{cited_function}', expected one of {known_functions}"
                )

    return len(errors) == 0, errors


def generate_synthetic_dataset(
    off_path: Path,
    target_count: int = 3000,
    output_path: Path = OUTPUT_PATH,
    model_name: str = "llama3.1:8b",
    num_ctx: int = 8192,
    batch_size: int = 8,
    seed: int = 7,
    limit: Optional[int] = None,
    sample_seed: int = 7,
    stratify_by_nova: bool = False,
    eu_additives_url: str = DEFAULT_EU_ADDITIVES_URL,
) -> None:
    ensure_output_dir()
    eu_additives = fetch_or_load_eu_additives(eu_url=eu_additives_url)
    df = load_open_food_facts(off_path, limit=limit)
    df = apply_limit(
        df=df,
        limit=limit,
        sample_seed=sample_seed,
        stratify_by_nova=stratify_by_nova,
    ).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid Open Food Facts records after filtering.")

    random.seed(seed)
    candidate_indices = list(range(len(df)))
    random.shuffle(candidate_indices)

    existing_keys = load_checkpoint_keys(output_path)
    if existing_keys:
        print(f"Resume mode: found {len(existing_keys)} existing records in {output_path.name}")
    if len(existing_keys) >= target_count:
        print(f"Target already satisfied ({len(existing_keys)}/{target_count}). Nothing to do.")
        return

    accepted = len(existing_keys)
    cursor = 0

    with output_path.open("a", encoding="utf-8") as f:
        pbar = tqdm(total=target_count, desc="synthetic-cot")
        pbar.update(min(accepted, target_count))
        while accepted < target_count and cursor < len(candidate_indices):
            batch_indices = candidate_indices[cursor : cursor + batch_size]
            cursor += batch_size

            for idx in batch_indices:
                row = df.iloc[idx]
                source_key = build_source_key(row)
                if source_key in existing_keys:
                    continue
                ground_truth = int(row["nova_group"])
                user_prompt = build_user_prompt(row, eu_additives)

                resp = ollama.chat(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    options={
                        "temperature": 0.2,
                        "num_ctx": num_ctx,
                    },
                )
                raw_text = resp.get("message", {}).get("content", "")
                trace = parse_json_response(raw_text)
                if not trace:
                    continue
                normalize_trace_ingredient_steps(trace)

                if not verify_nova_label(trace, ground_truth):
                    continue

                enum_ok, enum_errors = verify_enumber_citations(trace, eu_additives)
                if not enum_ok:
                    continue

                record = {
                    "id": f"synthetic::{accepted + 1}",
                    "source_key": source_key,
                    "product_name": _scalarize(row.get("product_name", "Unknown Product")),
                    "barcode": _scalarize(row.get("code")),
                    "country": _scalarize(row.get("countries_en")),
                    "ground_truth_nova_group": ground_truth,
                    "ingredients_text": _scalarize(
                        row.get("ingredients_clean", row.get("ingredients_text", ""))
                    ),
                    "trace": _json_safe(trace),
                    "validation": {
                        "nova_label_match": True,
                        "enum_citation_ok": True,
                        "enum_errors": _json_safe(enum_errors),
                    },
                }
                f.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")
                existing_keys.add(source_key)
                accepted += 1
                pbar.update(1)
                if accepted >= target_count:
                    break
        pbar.close()

    print(f"Saved {accepted} traces to {output_path}")
    if accepted < target_count:
        print("Warning: source data exhausted before hitting requested target count.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TruthBite synthetic CoT generation: sample Open Food Facts products, call Ollama, "
            "validate NOVA + E-number citations, append JSON Lines to an output file (resume-safe)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --open-food-facts-path data/raw/openfoodfacts.parquet --target-count 100 --limit 1000 --sample-seed 42 --batch-size 2 --model-name llama3.1:8b\n"
            "\n"
            "--limit: Cap how many filtered OFF rows are loaded from Parquet (chunked) and optionally subsampled; "
            "the model only sees this pool. Use with --sample-seed for reproducibility.\n"
            "--sample-seed: Seed for subsampling the pool when --limit is set (matches data_pipeline.py convention).\n"
            "--seed: Shuffles candidate row order before generation (different from sample-seed).\n"
            "Resume: if --output-path already exists, existing source_key values are skipped.\n"
        ),
    )
    parser.add_argument(
        "--open-food-facts-path",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to Open Food Facts .parquet (or .csv).",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=3000,
        metavar="N",
        help="Number of validated traces to collect (default: 3000).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_PATH,
        metavar="PATH",
        help=f"JSON Lines output file (default: {OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="llama3.1:8b",
        help="Ollama model name (default: llama3.1:8b).",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=8192,
        metavar="TOKENS",
        help="Ollama context length (default: 8192; lower if VRAM is tight).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="N",
        help="Internal batch stride for candidate indices (not parallel Ollama calls); lower on weak GPUs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        metavar="INT",
        help="Random seed for shuffling candidates (default: 7).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Max filtered OFF rows to load from Parquet and subsample into the generation pool. "
            "Omit only if you accept long reads / large memory."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=7,
        metavar="INT",
        help="Random seed for pool subsampling when --limit is set (default: 7).",
    )
    parser.add_argument(
        "--stratify-by-nova",
        action="store_true",
        help="When using --limit, preserve approximate NOVA 1–4 proportions in the pool.",
    )
    parser.add_argument(
        "--eu-additives-url",
        type=str,
        default=DEFAULT_EU_ADDITIVES_URL,
        help="Additives taxonomy URL if cache is missing (see data_pipeline).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_synthetic_dataset(
        off_path=args.open_food_facts_path,
        target_count=args.target_count,
        output_path=args.output_path,
        model_name=args.model_name,
        num_ctx=args.num_ctx,
        batch_size=args.batch_size,
        seed=args.seed,
        limit=args.limit,
        sample_seed=args.sample_seed,
        stratify_by_nova=args.stratify_by_nova,
        eu_additives_url=args.eu_additives_url,
    )
