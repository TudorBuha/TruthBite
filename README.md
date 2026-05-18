**Project Name:** TruthBite

**Team Name:** TruthBite

**Team Members:** Cocaina Mara, Ciobanu Eduarda, Bortos Alexia, Buha Tudor

**RAG Solution Design Document**

# **Getting Started**

Prerequisites: Python 3.11+, [Docker Desktop](https://www.docker.com/products/docker-desktop/), and (for synthetic dataset generation) [Ollama](https://ollama.com/) with your chosen model pulled (e.g. `llama3.1:8b`).

1. **Install Python dependencies** (from the repository root):

   ```bash
   pip install -r requirements.txt
   ```

2. **Start Qdrant** (local vector database on port `6333`):

   ```bash
   docker compose up -d
   ```

   Check it is running: `docker ps`, or open [http://localhost:6333](http://localhost:6333).

3. **Populate Qdrant** (EU additives + CoT reasoning examples):

   Place the required data files first:
   - `data/raw/eu_food_additives.json` — EU E-number definitions (540 entries)
   - `data/processed/synthetic_cot_dataset.jsonl` — CoT reasoning traces (1701 entries)

   Then run:

   ```bash
   python scripts/ingest.py
   ```

   This creates two Qdrant collections:
   - `additives_corpus` — one chunk per EU E-number
   - `cot_corpus` — one chunk per validated CoT reasoning trace

   To test without Docker (data lost on exit):

   ```bash
   python scripts/ingest.py --in-memory
   ```

### **Task D — Application demo**

The app layer can be run without downloading the full Open Food Facts export. It uses the live Open Food Facts barcode API for demo lookups and falls back to rule-based analysis if Ollama is not running.

Run locally:

```bash
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000), type a barcode, and click **Analyze**. Camera barcode scanning works on `localhost`; for scanning from a phone, expose the app over HTTPS, for example with `ngrok http 8000`.

By default the app uses the Task B fine-tuned `truthbite-phi4` model through Ollama. If Ollama is not running or the model call fails, the backend falls back to deterministic analysis so the demo still works. The app checks model availability at `GET /api/model/status`; model calls wait up to 180 seconds by default, which can be overridden with `OLLAMA_TIMEOUT`.

Run with Docker:

```bash
docker compose up --build
```

The backend is available at [http://localhost:8000](http://localhost:8000). If you want model-backed analysis, create the `truthbite-phi4` Ollama model as described below and keep Ollama running on the host.

4. **Open Food Facts data** (optional — only needed for `scripts/data_pipeline.py` and dataset generation): Download an Open Food Facts **Parquet** export manually (for example from [Hugging Face datasets](https://huggingface.co/datasets) — search for Open Food Facts / world food facts) and place the file under `data/raw/`, e.g. `data/raw/openfoodfacts.parquet`.

5. **Run OFF ingestion** (use `--limit` for a manageable subset; the full export is very large):

   ```bash
   python scripts/data_pipeline.py --open-food-facts-path "data/raw/openfoodfacts.parquet" --limit 5000 --stratify-by-nova --sample-seed 42
   ```

6. **Optional — synthetic CoT dataset** (requires Ollama running; output is gitignored by default):

   ```bash
   python scripts/generate_dataset.py --open-food-facts-path "data/raw/openfoodfacts.parquet" --target-count 100 --limit 1000 --sample-seed 42 --batch-size 2 --num-ctx 8192 --model-name "llama3.1:8b"
   python scripts/analyze_dataset.py --dataset-path "data/processed/synthetic_cot_dataset.jsonl"
   ```

   **Why the Ollama-generated file is not on GitHub:** the real output path `data/processed/synthetic_cot_dataset.jsonl` is listed in `.gitignore` (it can grow large and is easy to regenerate). The JSON Lines **schema** is illustrated in `examples/synthetic_cot_sample.jsonl`. For collaboration, share a full export via cloud storage or run the same command locally.

### **Overnight generation (RTX 5060 8GB, ~2000–3000 rows)**

Prerequisites: Ollama running with `llama3.1:8b` pulled, `openfoodfacts.parquet` in `data/raw/`, laptop plugged in and sleep disabled for the session.

The script **appends** to the JSONL file, **resumes** on restart (skips `source_key` values already present), **retries** failed Ollama calls, runs **GC every 100 LLM calls**, prints a **summary every 50 products**, and with **`--shutdown`** writes `data/processed/generate_dataset_done.txt` when the run finishes.

**Exactly 2500 validated traces:**

```bash
python scripts/generate_dataset.py --open-food-facts-path "data/raw/openfoodfacts.parquet" --target-count 2500 --limit 15000 --sample-seed 42 --stratify-by-nova --seed 7 --batch-size 2 --num-ctx 8192 --model-name "llama3.1:8b" --ollama-timeout 600 --max-retries 3 --gc-interval 100 --gc-sleep 0.5 --progress-interval 50 --shutdown
```

**Exactly 3000 validated traces:**

```bash
python scripts/generate_dataset.py --open-food-facts-path "data/raw/openfoodfacts.parquet" --target-count 3000 --limit 20000 --sample-seed 42 --stratify-by-nova --seed 7 --batch-size 2 --num-ctx 8192 --model-name "llama3.1:8b" --ollama-timeout 600 --max-retries 3 --gc-interval 100 --gc-sleep 0.5 --progress-interval 50 --shutdown
```

**Next morning — quality report:**

```bash
python scripts/analyze_dataset.py --dataset-path "data/processed/synthetic_cot_dataset.jsonl"
```

If `data/processed/generate_dataset_done.txt` exists and `status=complete`, the overnight job reached the target row count.

Use `python scripts/data_pipeline.py --help` and `python scripts/generate_dataset.py --help` for all options.

## Fine-tuned Model (Ollama)

The fine-tuned Phi-4-mini (QLoRA, NOVA classification) is hosted on HuggingFace as a GGUF:
**https://huggingface.co/alexiab05/truthbite-phi4**

No GPU or Python environment needed to run it — just Ollama.

1. **Download the Modelfile:**

   ```bash
   curl -L -o Modelfile https://huggingface.co/alexiab05/truthbite-phi4/resolve/main/Modelfile
   ```

2. **Create the Ollama model:**

   ```bash
   ollama create truthbite-phi4 -f Modelfile
   ollama list  # verify it appears
   ```

3. **Test with a query:**

   The model requires the Phi-4-mini chat template to be constructed manually and sent via the `/api/generate` endpoint. The `/api/chat` endpoint does **not** work with this GGUF. Save the following as `payload.json`:

   ```json
   {
     "model": "truthbite-phi4",
     "stream": false,
     "prompt": "<|system|>\nYou are a senior Food Scientist specialized in NOVA food processing classification.\nReturn STRICT JSON (no markdown) with these exact keys:\n  ingredient_steps: list of objects each with: ingredient, analysis, nova_marker, e_number, cited_function\n  reasoning_summary: string\n  predicted_nova_group: integer 1-4\nRules:\n1) Analyse each ingredient step-by-step.\n2) When an additive appears, cite its E-number and function from the EU additive context.\n3) Be factually grounded and concise.\n4) Output only valid JSON, no markdown fences.<|end|>\n<|user|>\nProduct: Coca-Cola Classic\nCountry: United Kingdom\nIngredients: Carbonated Water, Sugar, Colour (Caramel E150d), Phosphoric Acid, Natural Flavourings including Caffeine\nEU additive context:\nE150d: name=Sulfite ammonia caramel, functions=[colour]<|end|>\n<|assistant|>\n"
   }
   ```

   Then run:

   ```bash
   curl http://localhost:11434/api/generate -d @payload.json
   ```

   The response will contain `ingredient_steps`, `reasoning_summary`, and `predicted_nova_group`.

   **Template format reference** — when building prompts programmatically, the structure is:

   ```
   <|system|>
   {system prompt}<|end|>
   <|user|>
   {user message}<|end|>
   <|assistant|>
   ```

## **Team handover**

- **Fine-tuning / training data:** Validated chain-of-thought traces are written to `data/processed/synthetic_cot_dataset.jsonl` by `scripts/generate_dataset.py` (JSON Lines). That path is ignored by Git; each line is one record with `trace`, `ground_truth_nova_group`, and `validation` metadata. Regenerate or extend runs with the same flags; the generator supports resume (skips rows already present in the output file). Use `scripts/analyze_dataset.py` for a quick NOVA and citation report.

- **RAG / retrieval:** The RAG pipeline is implemented in `scripts/pipeline.py` and supports two retrieval strategies, both exposed through the same `retrieve_context(ingredients_text, country, strategy)` function:

  **Strategy 1 — Dense-only (baseline):**
  1. Extracts E-numbers from the ingredient list using a regex
  2. Embeds the full ingredient text with `all-MiniLM-L6-v2` and runs a nearest-neighbour search against `additives_corpus` (top 5 by cosine similarity)
  3. Runs the same dense search against `cot_corpus` for similar past product reasoning examples (few-shot)
  4. Returns results as-is without reranking — pure vector similarity ranking

  **Strategy 2 — Hybrid RAG with Metadata Filtering (deployed in app):**
  1. Extracts E-numbers from the ingredient list using a regex
  2. Runs hybrid search (dense all-MiniLM-L6-v2 + in-process BM25) against `additives_corpus` — BM25 re-scores the dense pool to boost exact E-number keyword matches
  3. Runs a direct payload lookup (`WHERE e_number = 'EXxx'`) for every E-number found in the text — guarantees definitions are included even when short codes score poorly in vector search
  4. Runs dense search against `cot_corpus` for similar past product reasoning examples (few-shot)
  5. Re-ranks all candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2`, keeping top 5 additives + top 3 CoT examples
  6. Returns a formatted `additive_context` string and a `sources` list

  The app (`app/main.py`) uses Strategy 2 by default. Pass `strategy=1` to `retrieve_context()` to use the dense-only baseline (used in RAGAS evaluation to measure the improvement).

  The context string is injected into the SLM prompt before every model call. If Qdrant is unreachable, the app falls back gracefully to `"No retrieved additive context available."`

  After the model responds, `validate_citations(ingredient_steps, qdrant_client)` checks each cited E-number against `additives_corpus` and flags mismatches in the response `warnings` field.

  **Standalone test (no app, no Ollama):**

  ```bash
  python scripts/pipeline.py "Carbonated Water, Sugar, Colour (E150d), Phosphoric Acid (E338), Caffeine"
  ```

  **Ingestion** (run once per environment):

  ```bash
  python scripts/ingest.py                   # real Qdrant on port 6333
  python scripts/ingest.py --in-memory       # ephemeral, no Docker needed
  ```

  **RAG evaluation** (`scripts/evaluate_rag.py`) — compares three conditions on a stratified test subset: SLM with no RAG, Strategy 1 (dense), Strategy 2 (hybrid). Produces NOVA accuracy, citation validity, and RAGAS metrics (faithfulness, context precision/recall, answer correctness).

  Prerequisites:

  ```bash
  pip install -r requirements.txt
  docker compose up -d qdrant
  python scripts/ingest.py
  ollama serve
  ollama pull llama3.1:8b
  # truthbite-phi4 must be installed
  ```

  Run from the project root:

  ```bash
  # Full run — 100 examples, all metrics
  python scripts/evaluate_rag.py

  # Quick check — 20 examples, no RAGAS judge
  python scripts/evaluate_rag.py --n-samples 20 --skip-ragas

  # Custom output path
  python scripts/evaluate_rag.py --output results/rag_eval.json
  ```

  The script saves a checkpoint to `results/rag_eval.json` after each condition (no RAG → Strategy 1 → Strategy 2), so a failed run does not lose completed work. Use `--judge-model llama3.1:8b` (default) to change the RAGAS judge. See `python scripts/evaluate_rag.py --help` for all options.

# **1\. Project Overview**

TruthBite is an autonomous agentic application that deconstructs food labels to identify Ultra-Processed Foods (UPF). Its reasoning core is a fine-tuned Small Language Model (SLM) that orchestrates specialized tools, distinguishing whole-food ingredients from industrial formulations and providing evidence-based verdicts on "natural" marketing claims.

This document describes the complete RAG (Retrieval-Augmented Generation) solution design, covering: the datasets selected, chunking strategies, vector database choice, model selections and the RAG strategies to be implemented and evaluated.

# **2\. Dataset Selection**

We chose the following datasets based on coverage, reliability, and open-access availability:

## **2.1 Primary Datasets**

[**Open Food Facts (OFF)**](https://www.kaggle.com/datasets/openfoodfacts/world-food-facts)

Open Food Facts is the foundation dataset for TruthBite. It is a crowdsourced, open-access database containing over 3 million food products from 180+ countries, each entry including full ingredient lists, nutritional tables, NOVA processing group scores, and packaging claims.

- Format: CSV / MongoDB dump / REST API
- Why selected: Direct NOVA group labels enable ground-truth supervision for UPF classification; ingredient lists provide the raw text for parser training.
- Preprocessing: Filter to products with complete ingredient lists and confirmed NOVA scores; deduplicate by barcode; strip HTML artefacts from ingredient text.

[**EU Food Additives Database**](https://developer.datalake.sante.service.ec.europa.eu/api-details#api=294321de-6daf-480b-9c7a-b7b19eeff462&operation=ea5e05d1-f567-4ed2-a316-b9466fd2f6e6) **(E-numbers)**

The Directorate-General for Health and Food Safety (DG SANTE) maintains the official Union list of authorized food additives, detailing their technological functions, conditions of use, and specific food categories as defined by Regulation (EC) No 1333/2008.

- Format: RESTful API (JSON/XML), CSV exports, and searchable web interface (FIP Database).
- Why selected: The database provides the authoritative mapping of E-numbers to chemical identities; essential for high-precision semantic matching in ingredient list parsing and NOVA classification (Ultra-processed food identification).
- Preprocessing: Map API functional classes (e.g., "emulsifier") to NOVA-4 industrial markers; cache JSON responses for E-number definitions to ensure 100% uptime; normalize chemical synonyms to match standardized E-number IDs.

# **3\. Chunking Strategy**

Different document types require different chunking approaches. A one-size-fits-all strategy would degrade retrieval quality \- regulatory tables need structure-aware chunking, while academic text benefits from semantic chunking.

## **3.1 Ingredient List Chunks (Open Food Facts)**

**Strategy: Structured Document Chunking** Because this data is about specific products, we shouldn't just cut the text in half. We need to keep the product information together so the AI doesn't get confused.

- **How it works:** Each product (like a specific brand of granola bar) is treated as one "chunk." We group the Barcode, Product Name, and Ingredients List into a single block of text.
- **The Logic:** If a user scans a barcode, the AI needs the _entire_ ingredient list at once to decide if it's ultra-processed. If we split the list in the middle, the AI might miss the "Emulsifier" at the very end.
- **Metadata Tagging:** We attach "tags" to each chunk, like `Country: France` or `NOVA Score: 4`. This helps the agent filter the data instantly.

## **3.2 EU Food Additives Database**

**Strategy: Entity-Centric Chunking & Relational Mapping** This data is more like a dictionary of chemicals. The goal here is high-precision matching.

- **How it works:** We create one chunk per **E-number** (e.g., E300, E415).
- **Content of the Chunk:** Each chunk contains the E-number, its scientific name (Ascorbic Acid), its "Job" (Antioxidant), and the "Safety Rules" (how much is allowed in bread vs. candy).
- **Small & Precise:** These chunks are very small (usually under 200 words) to ensure that when the AI searches for "E415," it gets the exact definition and nothing else.
- **Synonym Expansion:** We include "aliases" in the chunk. If the AI looks for "Xanthan Gum," the chunking strategy ensures it points directly to the "E415" entry.

# **4\. Vector Database Choice**

## **4.1 Selected Database: Qdrant**

Qdrant is selected as the primary vector store for TruthBite. It is an open-source, production-ready vector database written in Rust, optimised for filtered semantic search with rich payload metadata.

## **4.2 Justification**

###

- **Hybrid Search Support**: Qdrant enables simultaneous sparse and dense search. This allows TruthBite to combine semantic matching for marketing terms with exact keyword lookups for E-numbers without extra infrastructure.
- **Payload Filtering**: Metadata pre-filtering narrows ANN (Approximate Nearest Neighbour) searches by NOVA group or document type, significantly increasing precision and reducing irrelevant results.
- **Privacy & Control**: As a self-hosted, Docker-ready solution, Qdrant ensures proprietary ingredient data remains on-premises, avoiding the risks of managed cloud services.
- **High Performance**: The Rust-based engine and HNSW indexing provide sub-millisecond latency and low memory overhead, meeting the speed requirements for real-time nutritional auditing.

# **5\. Model Choices**

## **5.1 SLM Reasoning Core**

**Base Model: Phi-3-mini (3.8B parameters)**

Microsoft's Phi-3-mini is selected as the fine-tuning base. It achieves strong reasoning performance at 3.8B parameters, and has a 128K token context window that can accommodate long ingredient lists and retrieved regulatory passages simultaneously.

• Why not a larger model: TruthBite's design philosophy prioritises database citation over probabilistic guessing. A smaller, highly specialised model with constrained output ("No Source, No Flag") outperforms a general-purpose 70B model for this task while being orders of magnitude cheaper to serve.

• Fine-tuning method: QLoRA (4-bit quantised LoRA) on NOVA-labelled Chain-of-Thought reasoning traces derived from the Open Food Facts and NOVA definition datasets.

• Training objective: Given an ingredient list, produce a step-by-step NOVA classification with cited sources for each ingredient decision.

## **5.2 Embedding Model**

**Dense Embeddings: all-MiniLM-L6-v2 (Sentence-Transformers)**

For semantic dense vectors, the all-MiniLM-L6-v2 model from the Sentence-Transformers library is used. It produces 384-dimensional embeddings with excellent semantic similarity performance on short food-domain text, and its small size (22M parameters) enables fast batch embedding at ingestion time.

• Domain adaptation: The model will be fine-tuned for 2-3 epochs on ingredient-label sentence pairs from OFF to improve food-domain vocabulary alignment.

**Sparse Embeddings: BM25 (via Qdrant's built-in sparse vectors)**

For keyword-based E-number and additive name matching, BM25 sparse vectors are generated using Qdrant's native sparse vector support. This eliminates the need for a separate Elasticsearch or keyword search service.

##

##

## **5.3 Reranker**

**Cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2**

A cross-encoder validates the top 20 retrieved candidates, re-ranking them before passing the top 5 to the SLM. This acts as a **precision filter** to distinguish between natural ingredients and industrial additives with similar-sounding names. By scoring the actual text match rather than just vector distance, it prevents the agent from grounding its verdict on irrelevant or "near-miss" data.

# **6\. RAG Strategies**

TruthBite implements and evaluates three RAG strategies, progressing from a baseline to an advanced agentic pipeline. The strategies are designed to be compared using RAGAS metrics to identify the optimal configuration for production deployment.

## **Strategy 1 \- Naive RAG (Baseline)**

**Description**

A standard single-step retrieval pipeline: the raw ingredient list is embedded, the top-5 most similar chunks are retrieved from the ingredients_corpus collection, and these are concatenated into the SLM prompt as context. No filtering, no reranking, no multi-step reasoning.

**Purpose**

This baseline establishes a performance floor, quantifying the benefit of each subsequent enhancement. Expected weaknesses include low precision on E-numbers (pure semantic search misses exact additive codes) and hallucination risk when ingredient synonyms are not resolved.

**Metrics Targeted**

• Context Recall: proportion of relevant regulatory passages retrieved

• Answer Faithfulness (RAGAS): fraction of response claims grounded in retrieved context

## **Strategy 2 \- Hybrid RAG with Metadata Filtering (Intermediate)**

**Description**

The retriever is upgraded to a hybrid search combining dense semantic vectors (all-MiniLM-L6-v2) with sparse BM25 keyword vectors. Qdrant's payload filters are applied to restrict regulatory lookups by jurisdiction (EU or FDA, inferred from product country of origin). A cross-encoder reranker then reorders the top-20 candidates to produce the final top-5 context passages.

**Enhancements over Strategy 1**

• Hybrid search dramatically improves E-number recall \- exact codes are matched by BM25 even when semantic similarity fails

• Jurisdiction filtering reduces noise from irrelevant regulatory passages

• Reranker corrects synonym-induced ranking errors

**Metrics Targeted**

• Context Precision and Recall (RAGAS)

• Classification accuracy on NOVA group (compared to OFF ground-truth labels)

## **Strategy 3 \- Agentic RAG with ReAct Orchestration (Advanced)**

**Description**

The full TruthBite architecture: the fine-tuned Phi-3-mini SLM acts as a ReAct agent, dynamically choosing which tool to invoke at each reasoning step. The tools \- Ingredient Parser, Nova-Search, and Safety Monitor \- each perform targeted retrievals against different Qdrant collections and external APIs (PubChem). The agent iterates until all ingredients are resolved and cited, or flags that a source cannot be found (No Source, No Flag guardrail).

**Key RAG Innovations**

• Multi-hop retrieval: the agent may perform 3-7 sequential retrieval steps per query, resolving synonyms before classification and safety-checking after

• Tool-specific retrieval strategies: Ingredient Parser uses sparse synonym index; Nova-Search uses hybrid dense+sparse; Safety Monitor uses exact-match filtered queries

• Self-grounding: the SLM is trained (via CoT fine-tuning) to explicitly reference the retrieved chunk ID in its reasoning trace, enabling automated faithfulness checking

• Greenwashing detection: a dedicated semantic sub-query targets a curated index of greenwashing marketing terms, flagging claims unsupported by ingredient-level evidence

**Metrics Targeted**

• Answer Faithfulness (RAGAS): target \> 95% \- all claims must be grounded in retrieved documents

• Answer Correctness: verified against OFF NOVA ground-truth labels on held-out test set

• Hallucination Rate: number of uncited ingredient-to-class assignments (target: 0 under No Source, No Flag guardrail)

• Tool Call Efficiency: average number of retrieval steps per product; target \< 6 steps

## **6.1 Strategy Comparison Overview**

| Strategy                      | Key Characteristics                                                   |
| :---------------------------- | :-------------------------------------------------------------------- |
| **1\. Naive RAG**             | Single-step dense retrieval; no reranking; baseline accuracy          |
| **2\. Hybrid RAG \+ Filters** | Dense+sparse hybrid; jurisdiction filters; cross-encoder reranking    |
| **3\. Agentic ReAct RAG**     | Multi-hop tool-driven retrieval; CoT SLM; No Source No Flag guardrail |

# **7\. Summary**

TruthBite's RAG solution is designed around the principle that verifiable scientific grounding must take precedence over probabilistic inference for food safety classification. The selected components \- Open Food Facts as corpus, Qdrant for hybrid retrieval, Phi-3-mini as the fine-tuned reasoning core, and an agentic ReAct pipeline \- are each chosen specifically to enforce this principle while maintaining the low-latency performance required for consumer-facing nutritional auditing.

The three-strategy evaluation framework ensures that each architectural enhancement can be measured and justified empirically before production deployment, with RAGAS metrics providing the objective grounding that TruthBite's own outputs are designed to provide for food labels.

# **8\. Local Runbook (Pause / Resume)**

Use this section when you want to stop work now and continue later

## **8.1 Stop Everything Safely (Now)**

From the project root:

1. Stop Qdrant container:
   - `docker compose down`
2. Stop Ollama server (if running in a terminal):
   - press `Ctrl + C` in that Ollama terminal
3. Optional: close any terminal running data generation or ingestion scripts.

Notes:

- `docker compose down` stops containers but keeps your persistent Qdrant data in `data/qdrant_storage`.
- Generated dataset files remain in `data/processed/`.

## **8.2 Start Everything Later**

From the project root:

1. Start Qdrant:
   - `docker compose up -d`
2. Verify Qdrant is running:
   - `docker ps`
   - optional health check: open `http://localhost:6333`
3. Start Ollama (if not already running as a background service):
   - `ollama serve`
4. Resume pipeline:
   - Ingestion example:
     - `python scripts/data_pipeline.py --open-food-facts-path "data/raw/openfoodfacts.parquet" --limit 5000 --stratify-by-nova --sample-seed 42`
   - Synthetic generation example:
     - `python scripts/generate_dataset.py --open-food-facts-path "data/raw/openfoodfacts.parquet" --target-count 100 --batch-size 4 --num-ctx 8192 --model-name "llama3.1:8b"`

---

# **9. SLM Fine-tuning: Approach, Results & Data Improvement Recommendations**

This section documents the full fine-tuning cycle for the TruthBite SLM — what was done, what the evaluation showed, and concrete recommendations for whoever extends the training data next.

---

## **9.1 Fine-tuning Approach**

### Base model

**Phi-4-mini** (3.8B parameters, MIT licence) — note: the design document references Phi-3-mini, but the actual training used Phi-4-mini, which has better instruction-following and JSON output reliability.

### Method: QLoRA via Unsloth

- 4-bit quantised base model (bitsandbytes NF4); only the LoRA adapter layers (~20–30M parameters) are trained
- Library: [Unsloth](https://github.com/unslothai/unsloth) for memory-efficient fine-tuning
- Hardware: Google Colab T4 GPU (~1.5 hours for 3 epochs)
- Notebook: `notebooks/finetune_phi4_mini.ipynb`

### Training data

- **1,701 synthetic Chain-of-Thought traces** from `data/processed/synthetic_cot_dataset.jsonl`
- Each trace was generated by Llama 3.1 8B (via Ollama) acting as the "teacher" model and validated before inclusion:
  1. The JSON must parse correctly and contain all required keys (`ingredient_steps`, `reasoning_summary`, `predicted_nova_group`)
  2. `predicted_nova_group` must match the Open Food Facts ground-truth label
  3. Any cited E-number must exist in the EU additives database and the cited function must match the known functions for that code
- **80/20 stratified split** (seed = 42): 1,360 training examples / 341 test examples

### Data generation pipeline (for reference)

```bash
# Teacher model must be running
ollama serve

# Generate traces (resume-safe — skips already-generated products)
python scripts/generate_dataset.py \
  --open-food-facts-path data/raw/openfoodfacts.parquet \
  --target-count 3000 \
  --model-name llama3.1:8b \
  --num-ctx 8192 \
  --batch-size 8

# Check NOVA distribution and citation accuracy
python scripts/analyze_dataset.py --dataset-path data/processed/synthetic_cot_dataset.jsonl
```

### Output artefacts

| Artefact                         | Location                                                     |
| -------------------------------- | ------------------------------------------------------------ |
| LoRA adapter (HF format)         | Google Drive: `MyDrive/TruthBite/model/lora_adapter/`        |
| Merged model (HF format, 7.7 GB) | Google Drive: `MyDrive/TruthBite/model/gguf/truthbite-phi4/` |
| GGUF (Q4_K_M, ~2.2 GB)           | Google Drive + HuggingFace: `alexiab05/truthbite-phi4`       |
| Eval notebook                    | `notebooks/evaluate_phi4_mini.ipynb`                         |

> **Note on GGUF export:** Phi-4-mini uses a JSON tokenizer rather than SentencePiece. The standard `llama.cpp` converter fails on it. The fix requires patching the converter locally — this is a known issue and the patch is not upstreamed.

---

## **9.2 Evaluation Results**

The eval notebook (`notebooks/evaluate_phi4_mini.ipynb`) reproduces the identical 80/20 stratified split used during training and runs inference on the 341 held-out test examples using the fine-tuned model (LoRA adapter loaded via Unsloth).

### Methodology

For each test example the model receives the system prompt plus the user-turn (product name, country, ingredient list, and EU additive context). The generated JSON is parsed and `predicted_nova_group` is extracted. If parsing fails entirely, the prediction is recorded as a format failure and excluded from metric computation.

### Top-line metrics

| Metric            | Value                 |
| ----------------- | --------------------- |
| Test set size     | 341                   |
| Format failures   | 15 / 341 (4.4%)       |
| Evaluated on      | 326 valid predictions |
| **NOVA Accuracy** | **0.7331**            |
| **Macro F1**      | **0.4726**            |

### Per-class breakdown

| Class            | Precision | Recall   | F1       | Support |
| ---------------- | --------- | -------- | -------- | ------- |
| NOVA 1           | 0.64      | 0.57     | 0.60     | 44      |
| NOVA 2           | 0.00      | 0.00     | 0.00     | 4       |
| NOVA 3           | 0.50      | 0.38     | 0.43     | 68      |
| NOVA 4           | 0.82      | 0.90     | 0.85     | 210     |
| **weighted avg** | **0.72**  | **0.73** | **0.72** | **326** |
| **macro avg**    | **0.49**  | **0.46** | **0.47** | **326** |

### What each metric means

**Format failures (15/341):** The model output could not be parsed as valid JSON, or `predicted_nova_group` was missing/unreadable even with a fallback regex. These 15 examples are dropped; all other metrics are computed on the remaining 326.

**NOVA Accuracy (0.7331):** Simple percentage of predictions that matched the ground-truth NOVA group exactly among the 326 parseable outputs.

**Macro F1 (0.4726):** F1 is the harmonic mean of precision and recall. _Macro_ means every class is weighted equally regardless of how many examples it has — so NOVA 2 with 4 examples pulls the average down just as hard as NOVA 4 with 210. This is the honest measure of whether the model handles the full NOVA scale, not just the majority class.

**Weighted avg F1 (0.72):** Weights each class by its support. Close to overall accuracy because NOVA 4 dominates (64% of the test set). This number looks better but is misleading for a balanced use-case.

### Analysis

**NOVA 4 dominance.** NOVA 4 is 64% of the test set (210/326). The model learned to predict it aggressively — recall 0.90 means it catches 90% of ultra-processed products. This also inflates the overall accuracy to 73%, masking real weakness in the other classes.

**NOVA 2 is zero.** Only 4 NOVA 2 examples appear in the test set, meaning the training set had roughly 20 in total. The model never learned to confidently predict NOVA 2, so it always assigns something else. This is a data problem, not a model problem.

**NOVA 3 is the hardest class.** F1 = 0.43, recall = 0.38. Products in NOVA 3 ("processed foods" — canned fish with salt, cured meats, cheeses with a few additives) sit at a blurry boundary with NOVA 4. The model defaults to predicting NOVA 4 when uncertain, which costs NOVA 3 recall. More training data at the NOVA 3/4 boundary is the fix.

**Macro vs. weighted avg gap.** Macro F1 (0.47) is 25 points below weighted F1 (0.72). This gap measures how badly the class imbalance is hurting minority-class performance. Closing this gap is the primary goal for the next training iteration.

---

## **9.3 Recommendations for Improving Training Data**

The following are prioritised recommendations for whoever generates the next batch of CoT traces. They are ordered by expected impact.

### Priority 1 — Force per-class output targets (code change, ~30 min)

The existing `--stratify-by-nova` flag in `generate_dataset.py` stratifies the _source pool_, not the _output_. If NOVA 2 is rare in Open Food Facts, you still end up with almost no NOVA 2 traces regardless.

**Change:** Replace the single `target_count` with a per-class target dict, and stop each class independently when it reaches its quota.

```python
# Example target distribution
TARGET_PER_CLASS = {1: 500, 2: 500, 3: 500, 4: 700}
```

Track `accepted_per_nova = {1: 0, 2: 0, 3: 0, 4: 0}` and skip classes that are already full. This single change is the highest-leverage improvement because it directly attacks the root cause of the Macro F1 gap.

### Priority 2 — Upgrade the teacher model for the entire dataset

Llama 3.1 8B at temperature 0.2 produces plausible but shallow reasoning for minority classes. NOVA 2 products (pressed oils, butter, plain flour, pasta, canned plain vegetables, hard cheeses) have simple ingredient lists but require precise reasoning about what makes them _processed culinary ingredients_ rather than unprocessed or ultra-processed.

**Important:** do not use different teacher models for different NOVA classes. Mixing teachers introduces inconsistent reasoning styles and trace structure into the training set, which confuses the student model (Phi-4-mini) during fine-tuning. The teacher must be uniform across the entire dataset.

**The right approach:** upgrade the teacher for all new data generation in one go.

- **Groq API (Llama 3.3 70B)** — free tier, fast, easy to swap in `generate_dataset.py` by replacing the `ollama.chat` call with the Groq client; same message format
- **Claude Haiku via Anthropic API** — cheap per token, reliable JSON output, strong food science knowledge
- **Ollama with `llama3.3:70b` locally** — if sufficient VRAM is available

Existing 1,701 traces were generated with Llama 3.1 8B. If you switch teacher models, generate a fresh balanced dataset from scratch rather than appending to the existing file — mixing outputs from two different teachers in the same training set is worse than keeping one consistent weaker teacher.

### Priority 3 — Hand-curate a NOVA 2 gold set (~2 hours)

NOVA 2 is the best-defined class in the NOVA framework. EU Regulation defines it explicitly: cold-pressed oils, butter, cream, flour, pasta, plain canned/dried/frozen vegetables, salt, sugar, honey, unsweetened fruit juice. You can write 50–100 gold traces manually in an afternoon.

These hand-written traces will be higher quality than anything Llama 8B produces for this class, and 50 gold NOVA 2 examples matter enormously when the current dataset has ~20.

Format each trace to match the existing JSONL schema (`ingredient_steps`, `reasoning_summary`, `predicted_nova_group`, `ground_truth_nova_group`, `validation`).

### Priority 4 — Temperature augmentation for minority classes

The generator runs at `temperature=0.2` (near-deterministic). For the same OFF product, generating at `temperature=0.5` produces a meaningfully different reasoning path — different phrasing, different ordering of ingredient analysis.

**Change:** For NOVA 1 and NOVA 2 products, generate 2–3 traces per product at varying temperatures instead of 1. This is free augmentation that forces the model to learn the underlying concept rather than memorising a single trace pattern. No new data source needed.

### Priority 5 — Target the NOVA 3/4 boundary explicitly

NOVA 3 recall is 0.38 — the model classifies most NOVA 3 products as NOVA 4. The confusing cases are products with one or two additives (e.g. a preservative or a colour) but not the full ultra-processing profile that characterises NOVA 4.

**Change:** When building the generation pool for NOVA 3, filter OFF to products where the ingredient list contains at least one E-number. These are the hardest and most diagnostic examples for NOVA 3, and generating more of them directly addresses the model's main failure mode.

### Summary table

| Priority | Action                                                          | Estimated effort                | Expected impact                                |
| -------- | --------------------------------------------------------------- | ------------------------------- | ---------------------------------------------- |
| 1        | Add per-class target tracking to `generate_dataset.py`          | ~30 min                         | Fixes NOVA 2 zero-shot, closes Macro F1 gap    |
| 2        | Upgrade to a single stronger teacher (70B) for the full dataset | ~1 hour (integration) + compute | Better overall trace quality, consistent style |
| 3        | Hand-write 50 gold NOVA 2 traces                                | ~2 hours                        | Guaranteed signal for the hardest class        |
| 4        | Temperature augmentation for NOVA 1/2                           | Minor code change               | Free data diversity at no label cost           |
| 5        | Filter NOVA 3 pool to E-number-containing products              | Minor script change             | Improves NOVA 3/4 boundary recall              |
