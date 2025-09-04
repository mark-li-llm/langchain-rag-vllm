# RAG MS MARCO vLLM — Engineering Report

## 1) Technical Overview
This system is a production-oriented RAG stack: MS MARCO passages are preprocessed and chunked, embedded with HF BGE embeddings, and indexed into Qdrant. At query time, dense (and optional BM25) retrieval feeds a prompt builder that assembles a context with numbered citations, then a vLLM‑backed ChatOpenAI model generates the answer. The API is FastAPI; orchestration uses LCEL chains; operations run via Docker Compose behind Nginx; evaluation scripts compute Recall@k and MRR@10.

## 2) Architecture & Data Flow
- Document Source: `app/dataset_msmarco.py` → `load_msmarco()`, `iter_corpus_passages()`
- Preprocess: `app/preprocess.py` → `filter_record()`, `prepare_for_chunking()`
- Slicing: `app/chunking.py` → `make_splitter()`, `split_documents_batch()`
- Embeddings: `app/embeddings.py` → `get_production_embeddings()` (BGE‑small by default)
- Index/Vector Store: `app/index_qdrant.py` → `create_or_get_collection()`, `upsert_documents()`
- Retrieval: `app/retrieval.py` → `build_dense_retriever()` (+ optional `build_bm25_retriever()`), `ensemble_retriever()`
- Prompting: `app/prompting.py` → `build_prompt_template()`, `format_context_with_citations()`
- LLM: `app/llm.py` → `get_production_llm()` (OpenAI‑compatible to vLLM via Nginx)
- Chain & Post‑processing: `app/pipeline.py` → `build_rag_chain()` (citations, rough token usage, metadata)

Text flow (arrows): Document Source → Preprocess → Slicer → Embedding Model → Qdrant Collection → Retrieval (dense/BM25/ensemble) → Context Assembly + Citations → ChatOpenAI (vLLM) → Answer + Metadata

## 3) Code Map (modules → functions)
- `app/api.py`
  - `get_rag_chain()` initialize/cache: `QdrantClient`, embeddings, LLM; build retriever, prompt, LCEL chains
  - `query_endpoint()` non‑stream or `stream_query_response()` SSE streaming
  - `ingest_endpoint()` → background `rebuild_msmarco_index()` or `process_text_document()`
  - `health_endpoint()` checks API, Qdrant, and LLM
  - Global `trace_storage` with `store_trace()`; middleware adds `X-Trace-ID`
- `app/pipeline.py`
  - `build_rag_chain(retriever, llm, prompt, max_context_tokens)` retrieve → format context+citations → LLM → parse usage/metadata
  - `build_streaming_rag_chain()` chunked streaming; `create_chain_with_timing()`; `validate_chain()`
- `app/retrieval.py`
  - `build_dense_retriever(qdrant_client, collection_name, embeddings, k_pre, score_threshold)`
  - `build_bm25_retriever(docs_iter, k_pre)`; `ensemble_retriever(dense, sparse, weights, k_final)`
  - `build_retrieval_pipeline(..., use_bm25, dense_k, sparse_k, final_k, ensemble_weights)`
- `app/index_qdrant.py`
  - `create_or_get_collection(client, collection_name, vector_size, distance='Cosine')` (HNSW tuned)
  - `upsert_documents(client, collection_name, docs, embeddings, batch_size=128)`
  - `as_retriever(...)` wraps LangChain `Qdrant` vector store; `get_collection_info()`
- `app/embeddings.py`
  - `get_production_embeddings(model_name='BAAI/bge-small-en-v1.5')` (CPU; normalize; batch size)
- `app/prompting.py`
  - `build_prompt_template(max_context_tokens)` enforces context‑only answers, adds `[n]`
  - `format_context_with_citations(docs, max_context_tokens, chars_per_token)` returns context string + citation map
- `app/llm.py`
  - `get_llm(model, api_base, api_key, temperature, max_tokens, streaming=True, **kwargs)` resolves from settings/env; returns `ChatOpenAI`
- `app/dataset_msmarco.py`
  - `load_msmarco(corpus_split, queries_split, config, max_corpus_passages, seed)` uses HF Datasets; normalizes structure
- `app/__main__.py`
  - CLI: `build_index_command()`, `evaluate_command()`, `serve_command()`, `validate_command()`

## 4) Entry Points & Call Chains
- API (service):
  - `uvicorn app.api:app` (Makefile: `serve-api-dev` or Docker via Nginx)
  - `/v1/query` → `query_endpoint()` → `get_rag_chain()` → `rag_chain.invoke(query)` → response with `citations` and `metadata`
  - `/v1/query` with `stream=true` → `stream_query_response()` → `streaming_chain.stream(query)` (currently buffered then yielded)
  - `/v1/ingest` → `rebuild_msmarco_index()` (MS MARCO) or `process_text_document()`
  - Concurrency: uvicorn workers (`UVICORN_WORKERS`), threadpool for blocking LCEL invoke/stream, Nginx proxy
- CLI (module):
  - `python -m app build-index` → load → preprocess → chunk → embeddings → create collection → upsert → info
  - `python -m app evaluate` → build retriever → evaluate (Recall@k, MRR@10) → optional JSON/CSV outputs

## 5) Configuration & Defaults
- Settings: `app/config.py` (pydantic‑settings; loads `.env`)
  - Embeddings: `EMBED_MODEL_NAME=BAAI/bge-small-en-v1.5`
  - Index: `INDEX_BACKEND=qdrant`, `QDRANT_URL`, `QDRANT_API_KEY` (optional), `COLLECTION_NAME=msmarco_chunks_v21`
  - LLM: `OPENAI_API_BASE`, `OPENAI_API_KEY`, `LLM_MODEL`, `TEMPERATURE=0.2`, `MAX_OUTPUT_TOKENS=1000`
  - Retrieval: `TOPK_PRE=50`, `TOPK_FINAL=5`, `USE_BM25=false`, `MAX_CONTEXT_TOKENS=3000`
  - Chunking: `CHUNK_SIZE_CHARS=1800`, `CHUNK_OVERLAP_CHARS=200`
  - API: `AUTH_BEARER_TOKEN`, `UVICORN_WORKERS=2`, `LOG_LEVEL=info`
- Compose: `ops/compose.yaml` wires Qdrant (volumes, healthchecks), API, Nginx, optional Prometheus/Grafana.
- Nginx: `ops/nginx.conf` provides `/v1/*` proxy to API and `/llm/*` upstream pool to vLLM servers; streaming config and timeouts.

## 6) Retrieval & Prompt Details
- Vector Store: Qdrant (Cosine distance; HNSW `m=32`, `ef_construct=100`)
- Embeddings: HF `HuggingFaceEmbeddings` on CPU, normalized vectors for cosine similarity, configurable batch size
- Dense Retrieval: `k=TOPK_PRE` (default 50)
- Optional BM25: in‑memory index constructed from `Document` objects; ensemble uses RRF/`search_type='mmr'`
- Final K: `TOPK_FINAL` (default 5) — for dense‑only path, retriever `k` is updated to final K
- Prompt: system instructs to answer strictly from context and to include `[n]` citations; `format_context_with_citations` truncates to `MAX_CONTEXT_TOKENS` (character proxy)

## 7) Observability & Health
- Logging: `app/observability.py` structured JSON logger. System/health logs; request tracking helper with duration.
- Metrics: Prometheus (optional): request counts/durations, retrieval/generation histograms, token counters, active gauge. `start_metrics_server(PROMETHEUS_PORT)`.
- Health: `/v1/health` checks RAG API, Qdrant (collections count), LLM upstream (test message).
- Tracing: In‑memory `trace_storage` with `store_trace()` snapshot of docs/prompt/answer previews and timings.

## 8) Testing & Evaluation
- Quick eval: `make eval-quick` → `tools/evaluate.py --max-queries 100`
- Full eval: `make eval-full` → `--max-queries 5000 --output full_evaluation_results`
- Load testing: `make loadtest` → `tools/load_test.py` (async users; streaming path)
- Component validation: `make validate` → embeddings/Qdrant/LLM/chain checks

## 9) Deployment Notes
- Start stack: `make serve` → Docker Compose launches Qdrant, API, Nginx (and optional Prometheus/Grafana via profile)
- Env: copy `configs/.env.example` → `.env`; set `AUTH_BEARER_TOKEN`, `OPENAI_API_BASE`, `OPENAI_API_KEY`, `LLM_MODEL`, `QDRANT_URL`
- Nginx upstreams: update vLLM server hostnames/IPs in `ops/nginx.conf` (pool `vllm_pool`)

## 10) Risks, Constraints, Improvements
- Hard‑coded `web_dir` path in `app/api.py` (absolute, user‑specific): derive from project root or env.
- Streaming buffers: `stream_query_response()` collects all chunks before yielding → convert to true streaming (async generator relaying directly from LCEL stream).
- Secret leakage: `get_llm()` prints `final_params` including `openai_api_key` → redact secrets in logs.
- BM25 coupling: enabling `USE_BM25` requires supplying `bm25_docs` in context (API currently doesn’t) → feature‑flag at API or provide docs source.
- Config validation: enforce non‑empty `QDRANT_URL`, `OPENAI_API_BASE`, `LLM_MODEL`, `OPENAI_API_KEY` early (startup check) and validate URLs.
- Token usage estimation: heuristic (chars/4); prefer tokenizer or model usage metadata when available.
- Error handling/retries: wrap Qdrant/LLM ops with typed timeouts, backoff, circuit‑breaker semantics.
- Metrics coverage: ensure retrieval/generation timings and token counts recorded per request (currently partially wired).

## 11) Quick Commands
- Install deps: `make install`
- Build index: `make build-index`
- Serve API (dev): `make serve-api-dev`
- Serve full stack: `make serve`
- Health checks: `make health`
- Evaluate: `make eval-quick`
- Load test: `make loadtest`

---

# Appendix A — Key Defaults
- Embeddings: `BAAI/bge-small-en-v1.5` (384‑d)
- Vector store: Qdrant collection `msmarco_chunks_v21`, Cosine
- Retrieval: `TOPK_PRE=50`, `TOPK_FINAL=5`, `USE_BM25=false`
- Prompt context budget: `MAX_CONTEXT_TOKENS=3000`
- LLM: `TEMPERATURE=0.2`, `MAX_OUTPUT_TOKENS=1000`, streaming enabled

