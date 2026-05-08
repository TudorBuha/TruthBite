from __future__ import annotations

import argparse
import html
import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow.parquet as pq
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

EU_ADDITIVES_CACHE = RAW_DIR / "eu_food_additives.json"
DEFAULT_EU_ADDITIVES_URL = (
    "https://raw.githubusercontent.com/openfoodfacts/openfoodfacts-server/main/taxonomies/additives.txt"
)
FALLBACK_EU_ADDITIVES_URLS = [
    "https://raw.githubusercontent.com/openfoodfacts/openfoodfacts-server/main/taxonomies/additives.txt",
]

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INGREDIENTS_COLLECTION = "ingredients_corpus"
ADDITIVES_COLLECTION = "additives_corpus"

HTML_TAG_RE = re.compile(r"<[^>]+>")
ENUM_RE = re.compile(r"\b[eE][\s\-]?(\d{3,4}[a-zA-Z]?)\b")


def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def normalize_enumber(raw: str) -> Optional[str]:
    match = ENUM_RE.search(str(raw or ""))
    if not match:
        return None
    return f"E{match.group(1).upper()}"


def strip_html_text(text: str) -> str:
    no_tags = HTML_TAG_RE.sub(" ", str(text or ""))
    unescaped = html.unescape(no_tags)
    return re.sub(r"\s+", " ", unescaped).strip()


def normalize_ingredient_text(text: str) -> str:
    clean = strip_html_text(text)
    return re.sub(r"\b[eE]\s*-\s*(\d{3,4}[a-zA-Z]?)\b", r"E\1", clean)


def load_open_food_facts(path: Path, limit: Optional[int] = None) -> pd.DataFrame:
    required_cols = ["ingredients_text", "nova_group"]
    selected_cols = ["code", "product_name", "ingredients_text", "nova_group", "countries_en"]
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        pf = pq.ParquetFile(path)
        # schema_arrow reliably exposes top-level logical column names in OFF parquet exports.
        available_cols = set(pf.schema_arrow.names)
        parquet_cols = [c for c in selected_cols if c in available_cols]
        missing_required = [c for c in required_cols if c not in available_cols]
        if missing_required:
            raise ValueError(f"Missing required column(s): {missing_required}")

        frames: List[pd.DataFrame] = []
        rows_loaded = 0
        for batch in pf.iter_batches(batch_size=1000, columns=parquet_cols):
            batch_df = batch.to_pandas()
            frames.append(batch_df)
            rows_loaded += len(batch_df)
            if limit is not None and rows_loaded >= limit:
                break

        if not frames:
            return pd.DataFrame(columns=parquet_cols + ["ingredients_clean"])
        df = pd.concat(frames, ignore_index=True)
        if limit is not None and len(df) > limit:
            df = df.iloc[:limit].copy()
    else:
        # CSV path keeps low_memory mode and optionally short-circuits with nrows when limit is set.
        df = pd.read_csv(path, low_memory=True, nrows=limit)

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["nova_group"] = pd.to_numeric(df["nova_group"], errors="coerce")
    df["ingredients_text"] = df["ingredients_text"].astype(str)
    df = df[df["nova_group"].between(1, 4, inclusive="both")]
    df = df[df["ingredients_text"].str.strip().ne("")]
    df = df.copy()
    df["ingredients_clean"] = df["ingredients_text"].map(normalize_ingredient_text)
    return df


def apply_limit(
    df: pd.DataFrame,
    limit: Optional[int] = None,
    sample_seed: int = 7,
    stratify_by_nova: bool = False,
) -> pd.DataFrame:
    if limit is None:
        return df
    if limit <= 0:
        raise ValueError("--limit must be a positive integer")
    if len(df) <= limit:
        return df

    if not stratify_by_nova:
        return df.sample(n=limit, random_state=sample_seed).reset_index(drop=True)

    # Keep NOVA distribution balanced while honoring exact limit.
    group_sizes = df.groupby("nova_group").size().sort_index()
    total = int(group_sizes.sum())
    target_per_group: Dict[int, int] = {}
    remainders: List[tuple[float, int]] = []

    assigned = 0
    for g, size in group_sizes.items():
        raw_target = (int(size) / total) * limit
        base = min(int(size), int(raw_target))
        target_per_group[int(g)] = base
        assigned += base
        remainders.append((raw_target - base, int(g)))

    remaining = limit - assigned
    for _, g in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        available = int(group_sizes[g]) - target_per_group[g]
        if available > 0:
            target_per_group[g] += 1
            remaining -= 1

    parts: List[pd.DataFrame] = []
    for g, target in target_per_group.items():
        if target <= 0:
            continue
        group_df = df[df["nova_group"] == g]
        parts.append(group_df.sample(n=target, random_state=sample_seed))

    limited = pd.concat(parts, axis=0).sample(frac=1.0, random_state=sample_seed).reset_index(drop=True)
    return limited


def fetch_or_load_eu_additives(
    cache_path: Path = EU_ADDITIVES_CACHE,
    eu_url: str = DEFAULT_EU_ADDITIVES_URL,
) -> Dict[str, Dict[str, Any]]:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    errors: List[str] = []
    candidate_urls = [eu_url] + [u for u in FALLBACK_EU_ADDITIVES_URLS if u != eu_url]
    normalized: Optional[Dict[str, Dict[str, Any]]] = None

    for url in candidate_urls:
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            content_type = (response.headers.get("content-type") or "").lower()

            if url.lower().endswith(".json") or "json" in content_type:
                payload = response.json()
                normalized = normalize_eu_additives_payload(payload)
            else:
                normalized = normalize_eu_additives_text(response.text)

            if normalized:
                break
            errors.append(f"{url} returned empty/unsupported additive payload")
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(f"{url}: {exc}")

    if not normalized:
        raise RuntimeError(
            "Unable to fetch EU additives data from configured sources. Errors: " + " | ".join(errors)
        )

    cache_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized


def normalize_eu_additives_payload(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    normalized: Dict[str, Dict[str, Any]] = {}
    for key, data in payload.items():
        code = normalize_enumber(key)
        if not code:
            continue
        name = data.get("name") if isinstance(data, dict) else str(data)
        functions = data.get("wikidata", {}).get("en", []) if isinstance(data, dict) else []
        aliases = [str(name)] if name else []

        normalized[code] = {
            "code": code,
            "name": str(name or "").strip(),
            "functions": functions if isinstance(functions, list) else [],
            "aliases": aliases,
            "source": "EU Food Additives (normalized from OFF taxonomy)",
        }
    return normalized


def normalize_eu_additives_text(raw_text: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse OFF additives taxonomy text format into normalized E-number records.
    """
    normalized: Dict[str, Dict[str, Any]] = {}
    current_code: Optional[str] = None

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            current_code = None
            continue
        if line.startswith("#"):
            continue

        if line.startswith("en:"):
            value = line[3:].strip()
            parts = [p.strip() for p in value.split(",") if p.strip()]
            if not parts:
                current_code = None
                continue
            code = normalize_enumber(parts[0])
            if not code:
                current_code = None
                continue

            name = parts[1] if len(parts) > 1 else code
            aliases = parts[1:] if len(parts) > 1 else []
            normalized.setdefault(
                code,
                {
                    "code": code,
                    "name": name,
                    "functions": [],
                    "aliases": [],
                    "source": "OFF additives taxonomy text",
                },
            )
            normalized[code]["aliases"] = sorted(
                set(normalized[code].get("aliases", []) + aliases)
            )
            if normalized[code].get("name") in ("", code) and name:
                normalized[code]["name"] = name
            current_code = code
            continue

        if current_code is None or ":" not in line:
            continue

        # Pattern: property:lang:value (e.g., additive_class:en:preservative)
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        property_name, _lang, value = parts
        if any(token in property_name for token in ("class", "function", "role")):
            vals = [v.strip() for v in value.split(",") if v.strip()]
            if vals:
                normalized[current_code]["functions"] = sorted(
                    set(normalized[current_code].get("functions", []) + vals)
                )

    return normalized


def _off_value(row: pd.Series, keys: List[str], default: Any = "") -> Any:
    for key in keys:
        if key not in row.index:
            continue
        value = row[key]
        # OFF exports may store nested/list values in some fields; normalize safely.
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                continue
            value = value[0]
        elif hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
            try:
                arr = value.tolist()
                if isinstance(arr, list):
                    if len(arr) == 0:
                        continue
                    value = arr[0]
                else:
                    value = arr
            except Exception:  # pylint: disable=broad-except
                pass

        if pd.isna(value):
            continue
        return value
    return default


def build_ingredient_documents(df: pd.DataFrame) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        barcode = str(_off_value(row, ["code", "id"], f"missing-{idx}"))
        product_name = str(_off_value(row, ["product_name", "product_name_en"], "Unknown Product"))
        country = str(_off_value(row, ["countries_en", "countries"], "Unknown"))
        nova_group = int(row["nova_group"])
        ingredients_clean = str(row["ingredients_clean"])

        chunk_text = (
            f"Barcode: {barcode}\n"
            f"Product Name: {product_name}\n"
            f"Country: {country}\n"
            f"NOVA Group: {nova_group}\n"
            f"Ingredients: {ingredients_clean}"
        )
        source_id = f"ingredient::{barcode}::{idx}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, source_id))
        docs.append(
            {
                "id": point_id,
                "text": chunk_text,
                "payload": {
                    "source_id": source_id,
                    "barcode": barcode,
                    "product_name": product_name,
                    "country": country,
                    "nova_group": nova_group,
                    "doc_type": "ingredient_product",
                    "text": chunk_text,
                },
            }
        )
    return docs


def build_additives_documents(additives: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for code, data in additives.items():
        name = data.get("name", "")
        functions = data.get("functions", [])
        aliases = data.get("aliases", [])
        chunk_text = (
            f"E-Number: {code}\n"
            f"Name: {name}\n"
            f"Functions: {', '.join(functions) if functions else 'Unknown'}\n"
            f"Aliases: {', '.join(aliases) if aliases else 'N/A'}\n"
            "Jurisdiction: EU"
        )
        source_id = f"additive::{code}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, source_id))
        docs.append(
            {
                "id": point_id,
                "text": chunk_text,
                "payload": {
                    "source_id": source_id,
                    "e_number": code,
                    "name": name,
                    "functions": functions,
                    "aliases": aliases,
                    "jurisdiction": "EU",
                    "doc_type": "additive_entity",
                    "text": chunk_text,
                },
            }
        )
    return docs


def setup_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )


def upsert_documents(
    client: QdrantClient,
    encoder: SentenceTransformer,
    collection_name: str,
    docs: List[Dict[str, Any]],
    batch_size: int = 64,
) -> None:
    for start in tqdm(range(0, len(docs), batch_size), desc=f"upsert:{collection_name}"):
        batch = docs[start : start + batch_size]
        texts = [item["text"] for item in batch]
        vectors = encoder.encode(texts, normalize_embeddings=True).tolist()
        points = [
            models.PointStruct(id=item["id"], vector=vector, payload=item["payload"])
            for item, vector in zip(batch, vectors)
        ]
        client.upsert(collection_name=collection_name, points=points)


def run_pipeline(
    open_food_facts_path: Path,
    qdrant_url: str = "http://localhost:6333",
    eu_additives_url: str = DEFAULT_EU_ADDITIVES_URL,
    limit: Optional[int] = None,
    sample_seed: int = 7,
    stratify_by_nova: bool = False,
) -> None:
    ensure_directories()
    print("Loading Open Food Facts...")
    df = load_open_food_facts(open_food_facts_path, limit=limit)
    print(f"Filtered OFF records: {len(df)}")
    df = apply_limit(
        df=df,
        limit=limit,
        sample_seed=sample_seed,
        stratify_by_nova=stratify_by_nova,
    )
    if limit is not None:
        print(
            f"Embedding subset size: {len(df)} "
            f"(limit={limit}, stratify_by_nova={stratify_by_nova}, sample_seed={sample_seed})"
        )

    print("Loading EU additives (with cache)...")
    additives = fetch_or_load_eu_additives(eu_url=eu_additives_url)
    print(f"Loaded additives: {len(additives)}")

    ingredient_docs = build_ingredient_documents(df)
    additive_docs = build_additives_documents(additives)

    print("Initializing embedding model...")
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    vector_size = int(encoder.get_sentence_embedding_dimension())

    print("Connecting to Qdrant...")
    client = QdrantClient(url=qdrant_url)
    setup_qdrant_collection(client, INGREDIENTS_COLLECTION, vector_size)
    setup_qdrant_collection(client, ADDITIVES_COLLECTION, vector_size)

    print("Upserting ingredient chunks...")
    upsert_documents(client, encoder, INGREDIENTS_COLLECTION, ingredient_docs)
    print("Upserting additive chunks...")
    upsert_documents(client, encoder, ADDITIVES_COLLECTION, additive_docs)
    print("Pipeline complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TruthBite data ingestion: load Open Food Facts + EU additives taxonomy, "
            "preprocess, embed with all-MiniLM-L6-v2, upsert into Qdrant (ingredients_corpus, additives_corpus)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --open-food-facts-path data/raw/openfoodfacts.parquet --limit 5000 --sample-seed 42\n"
            "  %(prog)s --open-food-facts-path data/raw/openfoodfacts.parquet --limit 5000 --stratify-by-nova --sample-seed 42\n"
            "\n"
            "--limit: After filtering (NOVA 1–4, non-empty ingredients), cap how many rows are read from Parquet\n"
            "         (chunked read) and optionally subsample. Omit for full filtered dataset (very large).\n"
            "--sample-seed: Reproducible random seed when subsampling with --limit (and for --stratify-by-nova).\n"
        ),
    )
    parser.add_argument(
        "--open-food-facts-path",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to Open Food Facts export (.parquet or .csv).",
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default="http://localhost:6333",
        help="Qdrant HTTP URL (default: http://localhost:6333).",
    )
    parser.add_argument(
        "--eu-additives-url",
        type=str,
        default=DEFAULT_EU_ADDITIVES_URL,
        help="URL for OFF additives taxonomy (text/json); used if local cache is missing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Max rows to load/embed after filtering: stops Parquet batch read early, then optionally subsamples to N. "
            "Recommended for development (e.g. 3000–10000)."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=7,
        metavar="INT",
        help="Random seed for reproducible subsampling when --limit is set (default: 7).",
    )
    parser.add_argument(
        "--stratify-by-nova",
        action="store_true",
        help=(
            "When using --limit, sample so each NOVA group (1–4) is represented proportionally "
            "(approximate stratification)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        open_food_facts_path=args.open_food_facts_path,
        qdrant_url=args.qdrant_url,
        eu_additives_url=args.eu_additives_url,
        limit=args.limit,
        sample_seed=args.sample_seed,
        stratify_by_nova=args.stratify_by_nova,
    )
