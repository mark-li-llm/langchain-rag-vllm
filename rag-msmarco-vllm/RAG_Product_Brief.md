# RAG MS MARCO vLLM — Product Brief

## 1) Summary
This solution delivers accurate, cited answers by combining a curated knowledge base (MS MARCO passages) with a modern large language model. It retrieves relevant text, shows numbered references, and generates grounded, concise responses via an efficient, scalable API.

## 2) What It Does
- Answers user questions with evidence from a vetted corpus
- Includes inline citations (e.g., “[1]”) that map to source text
- Streams responses for fast perceived latency
- Scales horizontally behind Nginx load‑balancing and a vector database

## 3) Why It Matters
- Improves user trust with transparent sourcing
- Reduces hallucinations via retrieval‑augmented prompting
- Provides measurable quality (Recall@k, MRR) for continuous improvement
- Runs on your infrastructure; no proprietary data leaves your environment

## 4) How It Works (High‑Level)
User Question → Find Relevant Passages → Build an Answer Prompt with Sources → LLM Generates Answer → Return Answer + Citations

```
┌──────────┐   ┌─────────────┐   ┌────────────┐   ┌──────────┐
│  Client  │→→│  Retrieval   │→→│ Prompt w/   │→→│   LLM     │
│ (Web/API)│   │ (Vector DB) │   │ Citations  │   │ (vLLM)   │
└──────────┘   └─────────────┘   └────────────┘   └──────────┘
                     ↓
                References
```

## 5) Capabilities
- Grounded Answers: Uses only retrieved context to answer
- Citations: Inline `[n]` with a compact list of sources
- Streaming: Tokens stream as they’re generated
- Health & Monitoring: Health checks; optional Prometheus metrics
- Evaluation: Built‑in scripts to measure retrieval quality

## 6) Core Components (Non‑Technical)
- Knowledge Store: Vector database (Qdrant) holds searchable sentence embeddings of passages
- Retrieval: Finds the most similar passages to a question
- Prompt Assembly: Builds a short, readable context with numbered references
- Generation: Calls a locally‑hosted model (vLLM) through an OpenAI‑style API
- API: FastAPI endpoints for Query, Ingest, and Health

## 7) Operations
- Start (Docker): `make serve`
- Build Index: `make build-index` (prepares the knowledge base)
- Health Check: `make health`
- Evaluate Quality: `make eval-quick`

## 8) Performance & Expectations
- Latency: Sub‑second to a few seconds depending on model and passage count
- Throughput: Scales by adding vLLM servers; Nginx load‑balances
- Quality: Reported by Recall@k/MRR metrics from the MS MARCO evaluation scripts

## 9) Security & Access Control
- Authentication: Bearer token on the API (`AUTH_BEARER_TOKEN`)
- Private LLM: Your LLM runs behind Nginx; no external calls are required
- Secrets: Provide your own API key to access vLLM (or disable if not needed)

## 10) Monitoring & Reliability
- Health Endpoint: `/v1/health` returns API, vector store, and LLM status
- Metrics (optional): Prometheus counters/histograms for requests, latency
- Logs: Structured JSON logs for easy aggregation

## 11) What Needs Configuring (Minimum)
- API Token: `AUTH_BEARER_TOKEN`
- LLM Endpoint: `OPENAI_API_BASE` (e.g., `http://localhost:8080/llm/v1`)
- LLM Model Name: `LLM_MODEL`
- Vector DB: `QDRANT_URL` (e.g., `http://localhost:6333`)

## 12) Roadmap & Known Risks
- True Live Streaming: Current server buffers streaming chunks; move to fully live streaming
- Admin UI: Optional dashboard to browse sources, reindex, and rerun evals
- Reranking: Add re‑ranker for improved ordering beyond similarity
- Secret Hygiene: Remove sensitive values from logs entirely
- Search Tuning: Expose `TopK`/thresholds in UI for fast iteration

## 13) FAQ
- “Can we use our data?” Yes. Replace the dataset loader and re‑index.
- “Can we change the model?” Yes. Point to any model served by vLLM.
- “Do we get citations?” Yes, inline `[n]` and a concise source list.
- “Is it scalable?” Yes. Add more vLLM nodes, keep Qdrant persistent.

## 14) Glossary (Plain Language)
- Retrieval: Finding the most relevant pieces of text for a question
- Embedding: Turning text into numbers for fast similarity search
- Vector DB: Database optimized for searching embeddings
- RAG: Retrieval‑Augmented Generation; an LLM uses retrieved text as evidence
- Citation: A numbered reference pointing to source text
- vLLM: High‑throughput LLM server that exposes an OpenAI‑style API
- Nginx: The traffic manager/load‑balancer in front of services

