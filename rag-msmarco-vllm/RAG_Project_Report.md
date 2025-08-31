# RAG MS MARCO vLLM — Code Structure and Data Flow Report (ROOT_DIR)

## 1) Project Overview
This repository implements a production-oriented Retrieval-Augmented Generation (RAG) system using the MS MARCO dataset for corpus and evaluation, Qdrant as the vector store, HuggingFace embeddings (BGE family) for dense retrieval, and a vLLM-backed OpenAI-compatible Chat API served behind Nginx. The FastAPI service exposes RAG query, ingestion, and health endpoints; LCEL chains orchestrate retrieval → context assembly → LLM generation with citations; tools provide index building, evaluation (Recall@k/MRR@10), and load testing; Docker Compose spins up Qdrant, the API, Nginx, and optional monitoring.

## 2) Directory Tree (annotated)
```
ROOT_DIR
├── app/                              # Core RAG services and pipeline
│   ├── api.py                        # FastAPI app; RAG endpoints
│   │   ├─ get_rag_chain()            # Initialize and cache RAG components
│   │   ├─ query_endpoint()           # Handle RAG queries (stream/non-stream)
│   │   ├─ stream_query_response()    # SSE wrapper for streaming chain
│   │   ├─ ingest_endpoint()          # Ingest MS MARCO or ad‑hoc text
│   │   ├─ health_endpoint()          # Service, Qdrant, LLM health
│   │   ├─ process_text_document()    # Preprocess/chunk/upsert single text
│   │   └─ rebuild_msmarco_index()    # Background index rebuild task
│   ├── pipeline.py                   # LCEL RAG chains + helpers
│   │   ├─ build_rag_chain()          # Full chain with citations/metadata
│   │   ├─ build_simple_rag_chain()   # Minimal debug chain (no metadata)
│   │   ├─ build_streaming_rag_chain()# Streaming chain (chunk iterator)
│   │   ├─ create_chain_with_timing() # Wrap chain; execution timing
│   │   └─ validate_chain()           # Sanity check on chain behavior
│   ├── retrieval.py                  # Dense/BM25/ensemble retrievers
│   │   ├─ build_dense_retriever()    # Qdrant dense retriever (k+threshold)
│   │   ├─ build_bm25_retriever()     # BM25 from in‑memory Documents
│   │   ├─ ensemble_retriever()       # RRF + MMR weighted ensemble
│   │   ├─ build_retrieval_pipeline() # Dense or dense+BM25 (+weights)
│   │   ├─ validate_retriever()       # Basic retrieval validation
│   │   └─ get_retrieval_stats()      # Latency and result stats
│   ├── index_qdrant.py               # Qdrant collection and ops
│   │   ├─ create_or_get_collection() # Ensure collection; HNSW config
│   │   ├─ upsert_documents()         # Batch embed + upsert points
│   │   ├─ as_retriever()             # LangChain retriever wrapper
│   │   ├─ get_collection_info()      # Stats and payload samples
│   │   └─ delete_collection()        # Drop collection
│   ├── embeddings.py                 # HF embeddings utilities
│   │   ├─ get_embeddings()           # Create CPU-optimized embeddings
│   │   ├─ get_production_embeddings()# BGE‑small production preset
│   │   ├─ validate_embeddings()      # Sanity check dimension/encode
│   │   └─ get_embedding_model_info() # Known models/dimensions
│   ├── prompting.py                  # Prompt template + citations
│   │   ├─ build_prompt_template()    # System/human template with [n]
│   │   ├─ format_context_with_citations() # Context join + mapping
│   │   ├─ format_citations()         # Build citation list
│   │   ├─ validate_citations_in_response() # Check [n] usage
│   │   └─ extract_query_intent()     # Simple intent + key terms
│   ├── chunking.py                   # Text splitting and stats
│   │   ├─ make_splitter()            # Recursive splitter by chars
│   │   ├─ split_documents_batch()    # Chunk + assign UUID metadata
│   │   └─ get_chunking_stats()       # Chunk count/length stats
│   ├── preprocess.py                 # Normalization and filtering
│   │   ├─ normalize_text()           # Unicode, whitespace cleanup
│   │   ├─ filter_record()            # Basic quality filter rules
│   │   ├─ clean_for_embedding()      # Replace/normalize punctuation
│   │   └─ prepare_for_chunking()     # Prepend source + final text
│   ├── dataset_msmarco.py            # HF Datasets → normalized dicts
│   │   ├─ load_msmarco()             # Load/slice corpus + queries
│   │   ├─ iter_corpus_passages()     # Expand nested passage lists
│   │   └─ get_dataset_stats()        # Corpus/query descriptive stats
│   ├── llm.py                        # vLLM ChatOpenAI configuration
│   │   ├─ get_llm()                  # Resolve settings; create client
│   │   ├─ get_production_llm()       # Defaults for production
│   │   ├─ validate_llm_connection()  # Smoke test invoke()
│   │   ├─ test_llm_streaming()       # Stream chunks and join
│   │   └─ create_llm_with_fallback() # Primary/fallback configs
│   ├── observability.py              # Logging/metrics/tracing helpers
│   ├── eval_msmarco.py               # Recall/MRR evaluation pipeline
│   ├── schemas.py                    # Pydantic models for API
│   ├── ids.py                        # Trace/request/session ID utils
│   ├── timing.py                     # Timers + function timing
│   └── __main__.py                   # CLI: build/evaluate/serve/validate
│       ├─ build_index_command()      # End‑to‑end indexing flow
│       ├─ evaluate_command()         # Build retriever + evaluate
│       ├─ serve_command()            # uvicorn runners
│       └─ validate_command()         # Component sanity checks
├── tools/                            # Operational scripts
│   ├── build_index.py                # Standalone index builder CLI
│   ├── evaluate.py                   # Standalone evaluation CLI
│   ├── load_test.py                  # Async load testing harness
│   └── debug_llm_connection.py       # Out‑of‑process LLM smoke test
├── ops/                              # Deployment and gateway
│   ├── compose.yaml                  # Qdrant/API/Nginx/monitoring stack
│   ├── nginx.conf                    # LB: LLM pool + API proxy
│   └── docker/                       # Dockerfiles (API, etc.)
├── configs/                          # Environment and monitoring
│   ├── .env.example                  # Full env var template
│   └── prometheus.yml                # Prometheus scrape config
├── web/                              # Static web client (served at /static)
├── Makefile                          # Developer entrypoints (make help)
├── requirements.txt                  # Python deps (LangChain/Qdrant/etc.)
└── README.md                         # Overview, quick start, commands
```

## 3) RAG Flowchart (files/functions + key config)
- Document Source → Slicer → Embedding Model → Vector Library/Index → Retrieval → Filtering/Rearrangement → Prompt Assembly → LLM Calls → Answer Post-Processing/Reference Annotations

- Document Source: app/dataset_msmarco.py: load_msmarco(), iter_corpus_passages()
  - Config: `HF_*` (dataset/config/splits), `CORPUS_SAMPLE_SIZE`
- Slicer: app/chunking.py: make_splitter(), split_documents_batch(); app/preprocess.py: prepare_for_chunking(), filter_record()
  - Config: `CHUNK_SIZE_CHARS`, `CHUNK_OVERLAP_CHARS`
- Embedding Model: app/embeddings.py: get_production_embeddings()
  - Model: `EMBED_MODEL_NAME` (default BAAI/bge-small-en-v1.5); Dim: 384
- Vector Library/Index: app/index_qdrant.py: create_or_get_collection(), upsert_documents()
  - Type: Qdrant; Distance: Cosine; Collection: `COLLECTION_NAME`; URL: `QDRANT_URL`
- Retrieval: app/retrieval.py: build_retrieval_pipeline() → build_dense_retriever() (+optional build_bm25_retriever())
  - Config: `TOPK_PRE` (dense pre‑k), `TOPK_FINAL` (final k), `USE_BM25`
- Filtering/Rearrangement: app/retrieval.py: ensemble_retriever() (RRF+MMR), app/preprocess.py: filter_record()
  - Config: ensemble weights (code default (0.7, 0.3)), search type MMR
- Prompt Assembly: app/prompting.py: build_prompt_template(), format_context_with_citations()
  - Variables: `{query}`, `{context}`; `MAX_CONTEXT_TOKENS`
- LLM Calls: app/llm.py: get_production_llm() (ChatOpenAI over vLLM)
  - Config: `OPENAI_API_BASE` (Nginx /llm/v1), `OPENAI_API_KEY`, `LLM_MODEL`, `TEMPERATURE`, `MAX_OUTPUT_TOKENS`
- Answer Post‑Processing/Refs: app/pipeline.py: build_rag_chain() → parse_response_with_metadata(); app/prompting.py: validate_citations_in_response(), format_citations()
  - Outputs: `answer`, `citations[]`, `metadata.usage`, `citation_validation`

## 4) Entry Points and Call Chain
- Service (FastAPI): `app/api.py:app` (run via `uvicorn app.api:app` or `python -m app serve`)
  1. HTTP `POST /v1/query` → `query_endpoint()`
  2. `verify_token()` checks `AUTH_BEARER_TOKEN`
  3. `get_rag_chain()` initializes singletons (Qdrant client, embeddings, LLM), builds retriever, prompt, and LCEL chains (cached globals)
  4. Non-stream: `rag_chain.invoke(query)` executed in `run_in_threadpool`
  5. `build_rag_chain()` stages: retrieve → context+citations → prompt+LLM → parse/metadata
  6. Response assembled (schemas.QueryResponse) and `store_trace()` saves preview/metrics
  7. Middleware `add_process_time_header` adds timing + `X-Trace-ID`
  - Concurrency: uvicorn workers (`UVICORN_WORKERS`), threadpool offload for blocking calls; Nginx load-balances upstream LLM
  - Exception handling: try/except in endpoints; 500 with error detail; traces stored on error
  - Caching: component singletons (`rag_chain`, `streaming_chain`, `qdrant_client`, `embeddings`, `llm`); in‑memory `trace_storage` (TTL by count)

- CLI (module): `python -m app <command>`
  - build-index → `build_index_command()`
    1. `load_msmarco()` → `filter_record()` → `prepare_for_chunking()`
    2. `make_splitter()` → `split_documents_batch()`
    3. `get_production_embeddings()` → dim validation
    4. `create_or_get_collection()` (Cosine, HNSW tuned)
    5. `upsert_documents()` in batches
    6. `get_collection_info()` summary
  - evaluate → `evaluate_command()`
    1. Create Qdrant client + embeddings
    2. `build_retrieval_pipeline()` (dense (+BM25 optional))
    3. `validate_retriever()` (optional)
    4. `evaluate_msmarco_retrieval()` → Recall@k, MRR@10
    5. `format_evaluation_report()` (+ optional save)
  - serve → `serve_command()` runs uvicorn; validate → `validate_command()` checks embeddings/Qdrant/LLM and builds a test RAG chain

## 5) Key Component List
- Slicing
  - Function: `app.chunking.make_splitter(chunk_size_chars, chunk_overlap_chars)`
  - Defaults (Settings): `CHUNK_SIZE_CHARS=1800`, `CHUNK_OVERLAP_CHARS=200`
  - Batch: `split_documents_batch(docs, splitter)` assigns UUID chunk IDs
- Embedding
  - Provider: HuggingFace (`langchain_huggingface.HuggingFaceEmbeddings`)
  - Default Model: `BAAI/bge-small-en-v1.5`
  - Dimensions: 384 (via `get_embedding_model_info`)
  - Device: CPU (avoid GPU contention with vLLM)
- Vector Library
  - Type: Qdrant
  - Collection: `COLLECTION_NAME` (default `msmarco_chunks_v21`)
  - Distance: Cosine (configurable map)
  - Persistence: Docker volume `qdrant_data` → `/qdrant/storage` (see ops/compose.yaml)
- Search
  - Dense k (pre): `TOPK_PRE` (default 50)
  - Final k: `TOPK_FINAL` (default 5)
  - Similarity: Cosine; threshold optional in `as_retriever(search_kwargs)`
  - Ensemble: `EnsembleRetriever` with `weights=(0.7, 0.3)`, `search_type="mmr"`
- Reranking
  - Config: `USE_RERANKER` (default false)
  - Status: Not implemented in base; no reranker component present
- Prompts
  - Template: `app/prompting.build_prompt_template()`
  - Variables: `{query}`, `{context}`
  - Guidance: context‑only answers; inline citations in `[n]`
- LLM
  - Provider: `langchain_openai.ChatOpenAI` (OpenAI-compatible over vLLM)
  - API Base: `OPENAI_API_BASE` (Nginx `/llm/v1`)
  - Model: `LLM_MODEL` (e.g., Llama‑3.1‑8B‑Instruct)
  - Temperature: `TEMPERATURE` (default 0.2)
  - Max Tokens: `MAX_OUTPUT_TOKENS` (default 1000)
  - Streaming: enabled by default

## 6) Configuration and Environment
- Sources
  - Code: `app/config.py` (`pydantic_settings.BaseSettings`)
  - Env template: `configs/.env.example` (copy to `.env` in ROOT_DIR)
  - Docker: `ops/compose.yaml`, `ops/nginx.conf` (proxy and LLM upstream)
- Required Environment Variables (for a working run)
  - `AUTH_BEARER_TOKEN`: API auth token
  - `OPENAI_API_BASE`: OpenAI-compatible base (e.g., `http://localhost:8080/llm/v1`)
  - `OPENAI_API_KEY`: Key passed to ChatOpenAI/vLLM
  - `LLM_MODEL`: Served model name
  - Additionally required in practice:
    - `QDRANT_URL`: e.g., `http://localhost:6333` (or `http://qdrant:6333` in Docker)
- Common Optional Variables
  - `COLLECTION_NAME`, `EMBED_MODEL_NAME`, `TOPK_PRE`, `TOPK_FINAL`, `USE_BM25`, `MAX_CONTEXT_TOKENS`
  - `CHUNK_SIZE_CHARS`, `CHUNK_OVERLAP_CHARS`, `LOG_LEVEL`, `UVICORN_WORKERS`, `PROMETHEUS_PORT`
  - `QDRANT_API_KEY` (if Qdrant auth enabled), `OTEL_EXPORTER_OTLP_ENDPOINT`

## 7) Testing and Evaluation
- Evaluation
  - Quick eval: `make eval-quick` → runs `tools/evaluate.py --max-queries 100`
  - Full eval: `make eval-full` → `--max-queries 5000 --output ...`
  - Metrics: Recall@5/10, MRR@10, average query time
- Load Testing
  - `make loadtest` → async concurrent users and streaming path
- Component Validation
  - `make validate` → embeddings, Qdrant, LLM, and a simple RAG chain
- Unit Tests
  - Pytest available in deps, but no `tests/` present in repo; focus is runtime validation and evaluation scripts

## 8) Risks and Improvement Suggestions
- Hard-coded path in `app/api.py` for `web_dir` (absolute user path). Suggest deriving from project root or env to avoid broken mounts in Docker/CI.
- Global singletons (`rag_chain`, `embeddings`, `llm`, `qdrant_client`) shared across workers; prefer FastAPI dependency injection per‑process, thread-safe initializers, and lifecycle hooks.
- Streaming endpoint buffers full response (`list(streaming_chain.stream(...))`) before yielding; convert to true incremental async streaming to reduce latency/memory.
- Secret leakage risk: `app/llm.get_llm()` prints `final_params` including `openai_api_key`. Avoid logging secrets; redact sensitive fields.
- BM25 path requires documents in memory; toggling `USE_BM25=true` without providing `bm25_docs` raises; guard at configuration level or precompute.
- Error handling for Qdrant/LLM timeouts is coarse; add typed exceptions, retries, circuit breakers, and clearer 5xx categorization.
- Observability partial: Prometheus client optional; integrate metrics into endpoints consistently (retrieval/generation durations, token usage) and add tracing spans.
- Configuration validation: `QDRANT_URL` defaults to empty string; enforce presence at startup if backend is qdrant; add URL format checks.
- Distance metric hardcoded to Cosine; consider exposing in settings with guard against collection mismatch.
- Token accounting is heuristic (char/4); consider using LLM usage metadata or tokenizer for accurate accounting.

## 9) Appendix: Glossary
- RAG: Retrieval‑Augmented Generation — generate answers grounded in retrieved context
- MS MARCO: Microsoft MAchine Reading COmprehension dataset used for corpora/queries
- Qdrant: Vector database used for similarity search over embeddings
- Embedding: Numerical vector representation of text for similarity search
- BGE: BAAI General Embeddings, HF models (e.g., bge‑small‑en‑v1.5)
- LCEL: LangChain Expression Language for composing pipelines
- BM25: Sparse lexical retrieval algorithm based on term frequency
- RRF: Reciprocal Rank Fusion for combining ranked lists
- MMR: Maximal Marginal Relevance to increase result diversity
- vLLM: High‑throughput LLM serving system with OpenAI API compatibility
- ChatOpenAI: LangChain wrapper for OpenAI‑style chat models
- HNSW: Hierarchical Navigable Small World graph index for ANN search
- Cosine Similarity: 1 - cosine distance of vectors (similarity metric)
- Top‑k: Number of top results to keep from retrieval
- SSE: Server‑Sent Events, used for streaming responses
- LCEL Runnable: Executable chain in LangChain’s expression language
- Pydantic: Data validation and settings management library
- Prometheus: Metrics collection and monitoring system
- Uvicorn: ASGI server running the FastAPI app
- Nginx: Reverse proxy/load balancer in front of API and LLMs
```

```
