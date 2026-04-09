 

**Project Name:** TruthBite 

**Team Name:** TruthBite  

**Team Members:** Cocaina Mara, Ciobanu Eduarda, Bortos Alexia, Buha Tudor

 
 **RAG Solution Design Document**

# **1\. Project Overview**

TruthBite is an autonomous agentic application that deconstructs food labels to identify Ultra-Processed Foods (UPF). Its reasoning core is a fine-tuned Small Language Model (SLM) that orchestrates specialized tools, distinguishing whole-food ingredients from industrial formulations and providing evidence-based verdicts on "natural" marketing claims.

This document describes the complete RAG (Retrieval-Augmented Generation) solution design, covering: the datasets selected, chunking strategies, vector database choice, model selections and the RAG strategies to be implemented and evaluated.

 

# **2\. Dataset Selection**

We chose the following datasets based on coverage, reliability, and open-access availability:

## **2.1 Primary Datasets**

[**Open Food Facts (OFF)**](https://www.kaggle.com/datasets/openfoodfacts/world-food-facts) 

Open Food Facts is the foundation dataset for TruthBite. It is a crowdsourced, open-access database containing over 3 million food products from 180+ countries, each entry including full ingredient lists, nutritional tables, NOVA processing group scores, and packaging claims.

* Format: CSV / MongoDB dump / REST API  
* Why selected: Direct NOVA group labels enable ground-truth supervision for UPF classification; ingredient lists provide the raw text for parser training.  
* Preprocessing: Filter to products with complete ingredient lists and confirmed NOVA scores; deduplicate by barcode; strip HTML artefacts from ingredient text.

 

[**EU Food Additives Database**](https://developer.datalake.sante.service.ec.europa.eu/api-details#api=294321de-6daf-480b-9c7a-b7b19eeff462&operation=ea5e05d1-f567-4ed2-a316-b9466fd2f6e6) **(E-numbers)**

The Directorate-General for Health and Food Safety (DG SANTE) maintains the official Union list of authorized food additives, detailing their technological functions, conditions of use, and specific food categories as defined by Regulation (EC) No 1333/2008.

* Format: RESTful API (JSON/XML), CSV exports, and searchable web interface (FIP Database).  
* Why selected: The database provides the authoritative mapping of E-numbers to chemical identities; essential for high-precision semantic matching in ingredient list parsing and NOVA classification (Ultra-processed food identification).  
* Preprocessing: Map API functional classes (e.g., "emulsifier") to NOVA-4 industrial markers; cache JSON responses for E-number definitions to ensure 100% uptime; normalize chemical synonyms to match standardized E-number IDs.

# **3\. Chunking Strategy**

Different document types require different chunking approaches. A one-size-fits-all strategy would degrade retrieval quality \- regulatory tables need structure-aware chunking, while academic text benefits from semantic chunking.

## **3.1 Ingredient List Chunks (Open Food Facts)**

**Strategy: Structured Document Chunking** Because this data is about specific products, we shouldn't just cut the text in half. We need to keep the product information together so the AI doesn't get confused.

* **How it works:** Each product (like a specific brand of granola bar) is treated as one "chunk." We group the Barcode, Product Name, and Ingredients List into a single block of text.  
* **The Logic:** If a user scans a barcode, the AI needs the *entire* ingredient list at once to decide if it's ultra-processed. If we split the list in the middle, the AI might miss the "Emulsifier" at the very end.  
* **Metadata Tagging:** We attach "tags" to each chunk, like `Country: France` or `NOVA Score: 4`. This helps the agent filter the data instantly.


 

## **3.2 EU Food Additives Database**

**Strategy: Entity-Centric Chunking & Relational Mapping** This data is more like a dictionary of chemicals. The goal here is high-precision matching.

* **How it works:** We create one chunk per **E-number** (e.g., E300, E415).  
* **Content of the Chunk:** Each chunk contains the E-number, its scientific name (Ascorbic Acid), its "Job" (Antioxidant), and the "Safety Rules" (how much is allowed in bread vs. candy).  
* **Small & Precise:** These chunks are very small (usually under 200 words) to ensure that when the AI searches for "E415," it gets the exact definition and nothing else.  
* **Synonym Expansion:** We include "aliases" in the chunk. If the AI looks for "Xanthan Gum," the chunking strategy ensures it points directly to the "E415" entry.


  

# **4\. Vector Database Choice**

## **4.1 Selected Database: Qdrant**

Qdrant is selected as the primary vector store for TruthBite. It is an open-source, production-ready vector database written in Rust, optimised for filtered semantic search with rich payload metadata.

## **4.2 Justification**

### 

* **Hybrid Search Support**: Qdrant enables simultaneous sparse and dense search. This allows TruthBite to combine semantic matching for marketing terms with exact keyword lookups for E-numbers without extra infrastructure.  
* **Payload Filtering**: Metadata pre-filtering narrows ANN (Approximate Nearest Neighbour)  searches by NOVA group or document type, significantly increasing precision and reducing irrelevant results.  
* **Privacy & Control**: As a self-hosted, Docker-ready solution, Qdrant ensures proprietary ingredient data remains on-premises, avoiding the risks of managed cloud services.  
* **High Performance**: The Rust-based engine and HNSW indexing provide sub-millisecond latency and low memory overhead, meeting the speed requirements for real-time nutritional auditing.

# **5\. Model Choices**

## **5.1 SLM Reasoning Core**

**Base Model: Phi-3-mini (3.8B parameters)**

Microsoft's Phi-3-mini is selected as the fine-tuning base. It achieves strong reasoning performance at 3.8B parameters, and has a 128K token context window that can accommodate long ingredient lists and retrieved regulatory passages simultaneously.

•       Why not a larger model: TruthBite's design philosophy prioritises database citation over probabilistic guessing. A smaller, highly specialised model with constrained output ("No Source, No Flag") outperforms a general-purpose 70B model for this task while being orders of magnitude cheaper to serve.

•       Fine-tuning method: QLoRA (4-bit quantised LoRA) on NOVA-labelled Chain-of-Thought reasoning traces derived from the Open Food Facts and NOVA definition datasets.

•       Training objective: Given an ingredient list, produce a step-by-step NOVA classification with cited sources for each ingredient decision.

 

## **5.2 Embedding Model**

**Dense Embeddings: all-MiniLM-L6-v2 (Sentence-Transformers)**

For semantic dense vectors, the all-MiniLM-L6-v2 model from the Sentence-Transformers library is used. It produces 384-dimensional embeddings with excellent semantic similarity performance on short food-domain text, and its small size (22M parameters) enables fast batch embedding at ingestion time.

•       Domain adaptation: The model will be fine-tuned for 2-3 epochs on ingredient-label sentence pairs from OFF to improve food-domain vocabulary alignment.

 

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

A standard single-step retrieval pipeline: the raw ingredient list is embedded, the top-5 most similar chunks are retrieved from the ingredients\_corpus collection, and these are concatenated into the SLM prompt as context. No filtering, no reranking, no multi-step reasoning.

**Purpose**

This baseline establishes a performance floor, quantifying the benefit of each subsequent enhancement. Expected weaknesses include low precision on E-numbers (pure semantic search misses exact additive codes) and hallucination risk when ingredient synonyms are not resolved.

**Metrics Targeted**

•       Context Recall: proportion of relevant regulatory passages retrieved

•       Answer Faithfulness (RAGAS): fraction of response claims grounded in retrieved context

 

## **Strategy 2 \- Hybrid RAG with Metadata Filtering (Intermediate)**

**Description**

The retriever is upgraded to a hybrid search combining dense semantic vectors (all-MiniLM-L6-v2) with sparse BM25 keyword vectors. Qdrant's payload filters are applied to restrict regulatory lookups by jurisdiction (EU or FDA, inferred from product country of origin). A cross-encoder reranker then reorders the top-20 candidates to produce the final top-5 context passages.

**Enhancements over Strategy 1**

•       Hybrid search dramatically improves E-number recall \- exact codes are matched by BM25 even when semantic similarity fails

•       Jurisdiction filtering reduces noise from irrelevant regulatory passages

•       Reranker corrects synonym-induced ranking errors

**Metrics Targeted**

•       Context Precision and Recall (RAGAS)

•       Classification accuracy on NOVA group (compared to OFF ground-truth labels)

 

## **Strategy 3 \- Agentic RAG with ReAct Orchestration (Advanced)**

**Description**

The full TruthBite architecture: the fine-tuned Phi-3-mini SLM acts as a ReAct agent, dynamically choosing which tool to invoke at each reasoning step. The tools \- Ingredient Parser, Nova-Search, and Safety Monitor \- each perform targeted retrievals against different Qdrant collections and external APIs (PubChem). The agent iterates until all ingredients are resolved and cited, or flags that a source cannot be found (No Source, No Flag guardrail).

**Key RAG Innovations**

•       Multi-hop retrieval: the agent may perform 3-7 sequential retrieval steps per query, resolving synonyms before classification and safety-checking after

•       Tool-specific retrieval strategies: Ingredient Parser uses sparse synonym index; Nova-Search uses hybrid dense+sparse; Safety Monitor uses exact-match filtered queries

•       Self-grounding: the SLM is trained (via CoT fine-tuning) to explicitly reference the retrieved chunk ID in its reasoning trace, enabling automated faithfulness checking

•       Greenwashing detection: a dedicated semantic sub-query targets a curated index of greenwashing marketing terms, flagging claims unsupported by ingredient-level evidence

**Metrics Targeted**

•       Answer Faithfulness (RAGAS): target \> 95% \- all claims must be grounded in retrieved documents

•       Answer Correctness: verified against OFF NOVA ground-truth labels on held-out test set

•       Hallucination Rate: number of uncited ingredient-to-class assignments (target: 0 under No Source, No Flag guardrail)

•       Tool Call Efficiency: average number of retrieval steps per product; target \< 6 steps

 

## **6.1 Strategy Comparison Overview**

| Strategy | Key Characteristics |
| :---- | :---- |
| **1\. Naive RAG** | Single-step dense retrieval; no reranking; baseline accuracy |
| **2\. Hybrid RAG \+ Filters** | Dense+sparse hybrid; jurisdiction filters; cross-encoder reranking |
| **3\. Agentic ReAct RAG** | Multi-hop tool-driven retrieval; CoT SLM; No Source No Flag guardrail |

 

 

# **7\. Summary**

TruthBite's RAG solution is designed around the principle that verifiable scientific grounding must take precedence over probabilistic inference for food safety classification. The selected components \- Open Food Facts as corpus, Qdrant for hybrid retrieval, Phi-3-mini as the fine-tuned reasoning core, and an agentic ReAct pipeline \- are each chosen specifically to enforce this principle while maintaining the low-latency performance required for consumer-facing nutritional auditing.

 

The three-strategy evaluation framework ensures that each architectural enhancement can be measured and justified empirically before production deployment, with RAGAS metrics providing the objective grounding that TruthBite's own outputs are designed to provide for food labels.

