# RAG MS MARCO Template with vLLM Integration

A production-ready Retrieval-Augmented Generation (RAG) system using the MS MARCO dataset, Qdrant vector database, and vLLM for local LLM serving. Built with LangChain for robust document processing and retrieval.

## 🎯 Overview

This template demonstrates a complete RAG pipeline with:

- **MS MARCO Dataset Integration**: Uses the MS MARCO v2.1 dataset for both corpus and evaluation
- **Dual vLLM Servers**: Load-balanced LLM serving with Nginx for high availability  
- **Production Architecture**: FastAPI + Qdrant + Nginx with comprehensive monitoring
- **Evaluation Framework**: Built-in metrics using MS MARCO relevance labels (Recall@k, MRR@10)
- **LangChain Integration**: Leverages RecursiveCharacterTextSplitter, HuggingFaceEmbeddings, EnsembleRetriever, and LCEL

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────────┐
│   Client    │────│    Nginx    │ (Load Balancer)
└─────────────┘    └─────┬───────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  RAG API    │  │  vLLM-A     │  │  vLLM-B     │
│ (FastAPI)   │  │ (OpenAI     │  │ (OpenAI     │
└─────┬───────┘  │ Compatible) │  │ Compatible) │
      │          └─────────────┘  └─────────────┘
      ▼
┌─────────────┐
│   Qdrant    │ (Vector Store)
│ (Vectors +  │
│  Metadata)  │
└─────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- 2x GPU servers for vLLM (or modify for CPU inference)
- 8GB+ RAM for Qdrant and API services

### 1. Setup Environment

```bash
# Clone and navigate to template
cd templates/rag-msmarco-vllm/

# Copy and configure environment
cp configs/.env.example .env
# Edit .env with your settings (see Configuration section)
```

### 2. Start vLLM Servers

On your GPU servers, start two vLLM instances:

```bash
# Server A (e.g., 192.168.1.100)
docker run --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --served-model-name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 8000

# Server B (e.g., 192.168.1.101) 
docker run --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --served-model-name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 8000
```

Update `ops/nginx.conf` with your vLLM server IPs.

### 3. Build Index and Start Services

```bash
# Build the MS MARCO index
make build-index

# Start all services
make serve

# Check health
make health
```

### 4. Test the System

```bash
# Run a test query
make demo-query

# Run evaluation
make eval-quick

# Run load test
make loadtest
```

## 📋 Configuration

Key environment variables in `.env`:

```bash
# Data Configuration
CORPUS_SAMPLE_SIZE=200000          # Limit dataset size for development
EMBED_MODEL_NAME=BAAI/bge-small-en-v1.5

# LLM Configuration  
OPENAI_API_BASE=http://localhost:8080/llm/v1
LLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
OPENAI_API_KEY=your-api-key

# Security
AUTH_BEARER_TOKEN=your-secure-token

# Retrieval Settings
TOPK_FINAL=5
USE_BM25=true
MAX_CONTEXT_TOKENS=3000
```

See `configs/.env.example` for all available options.

## 🛠️ Available Commands

### Core Operations
- `make build-index` - Build Qdrant index from MS MARCO
- `make serve` - Start all services with Docker Compose  
- `make health` - Check service health
- `make stop` - Stop all services
- `make clean` - Stop services and remove data

### Development
- `make serve-api-dev` - Start API with auto-reload
- `make validate` - Validate all components
- `make logs` - View service logs

### Testing & Evaluation  
- `make eval` - Run retrieval evaluation (1000 queries)
- `make eval-quick` - Quick evaluation (100 queries)
- `make loadtest` - Performance load testing
- `make demo-query` - Test with example query

### Maintenance
- `make backup-data` - Backup Qdrant data
- `make lint` - Code linting and formatting
- `make requirements` - Generate requirements.txt

## 📊 Evaluation Metrics

The system uses MS MARCO relevance labels for evaluation:

- **Recall@5/10**: Percentage of queries with relevant results in top-k
- **MRR@10**: Mean Reciprocal Rank for ranking quality
- **Latency**: Query processing time breakdown

Example evaluation output:
```
MS MARCO Retrieval Evaluation Report
====================================

Dataset Information:
  Configuration: v2.1
  Evaluated Queries: 1000

Recall@5:
  Mean: 0.7234
  Hit Rate: 0.8123

MRR@10:
  Mean: 0.6891

Performance:  
  Average Query Time: 245.67ms
```

## 🏭 Production Deployment

### Resource Requirements

- **API Service**: 2-4 CPU cores, 4-8GB RAM
- **Qdrant**: 4-8 CPU cores, 8-16GB RAM, SSD storage
- **vLLM Servers**: GPU with 16GB+ VRAM each
- **Nginx**: 1-2 CPU cores, 1GB RAM

### Security Checklist

- [ ] Generate secure `AUTH_BEARER_TOKEN`
- [ ] Configure TLS termination
- [ ] Set up VPN/firewall for vLLM servers  
- [ ] Enable Qdrant authentication
- [ ] Configure log aggregation
- [ ] Set up monitoring alerts

### Scaling Considerations

- **Horizontal**: Add more vLLM servers to Nginx upstream
- **Vertical**: Increase vLLM GPU memory for larger models
- **Storage**: Use external Qdrant cluster for large datasets
- **Caching**: Add Redis for query result caching

## 🔧 Development

### Project Structure

```
├── app/                     # Core Python package
│   ├── api.py              # FastAPI application
│   ├── pipeline.py         # LCEL RAG chain
│   ├── retrieval.py        # Dense + BM25 + Ensemble
│   ├── eval_msmarco.py     # Evaluation framework
│   └── ...
├── tools/                  # CLI utilities
│   ├── build_index.py      # Index builder
│   ├── evaluate.py         # Evaluation runner  
│   └── load_test.py        # Load testing
├── ops/                    # Infrastructure
│   ├── nginx.conf          # Load balancer config
│   ├── compose.yaml        # Service orchestration
│   └── docker/             # Container definitions
└── configs/                # Configuration templates
```

### Key LangChain Components

- **Text Splitting**: `RecursiveCharacterTextSplitter` (1800 chars, 200 overlap)
- **Embeddings**: `HuggingFaceEmbeddings` with BGE-small-en-v1.5
- **Vector Store**: `Qdrant` with HNSW indexing
- **Retrieval**: `EnsembleRetriever` (dense + BM25 with RRF)
- **LLM**: `ChatOpenAI` configured for vLLM compatibility
- **Chain**: LCEL composition with streaming support

### Adding New Features

1. **Custom Retrievers**: Extend `app/retrieval.py`
2. **New Endpoints**: Add to `app/api.py` with proper schemas
3. **Evaluation Metrics**: Enhance `app/eval_msmarco.py`
4. **Monitoring**: Add metrics in `app/observability.py`

## 📚 API Reference

### Query Endpoint

```http
POST /v1/query
Authorization: Bearer <token>
Content-Type: application/json

{
  "query": "What is machine learning?",
  "top_k": 5,
  "stream": true
}
```

Response:
```json
{
  "answer": "Machine learning is a subset of artificial intelligence [1] that enables computers to learn patterns from data [2].",
  "citations": [
    {
      "number": 1,
      "doc_id": "doc_123",
      "source": "msmarco:v2.1:123:0", 
      "url": "https://example.com/ml-intro",
      "score": 0.89
    }
  ],
  "metadata": {
    "retrieval_count": 5,
    "total_time_ms": 245.67
  },
  "trace_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV"
}
```

### Health Endpoint

```http
GET /v1/health
```

### Ingestion Endpoint

```http  
POST /v1/ingest
Authorization: Bearer <token>

{
  "source_type": "text",
  "payload": "Custom document text to index"
}
```

## 🐛 Troubleshooting

### Common Issues

**Q: Index build fails with CUDA errors**
A: Check GPU availability on vLLM servers. Use CPU embedding models if needed.

**Q: Nginx returns 502 errors**  
A: Verify vLLM servers are running and accessible. Check `ops/nginx.conf` upstream configuration.

**Q: Low recall scores**
A: Increase `CORPUS_SAMPLE_SIZE` or tune embedding model. Check chunk size settings.

**Q: API timeouts**
A: Increase Nginx proxy timeouts for long queries. Check vLLM server performance.

### Debug Commands

```bash
make debug          # Show system information
make logs-api       # API service logs  
make logs-nginx     # Nginx logs
make validate       # Component validation
```

## 🤝 Contributing

1. Follow LangChain coding conventions
2. Add type hints and docstrings
3. Include tests for new features
4. Update documentation
5. Test with `make ci-test`

## 📄 License

This template follows the same license as the LangChain project.

## 🔗 Related Resources

- [LangChain Documentation](https://python.langchain.com/)
- [MS MARCO Dataset](https://huggingface.co/datasets/microsoft/ms_marco)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)

---

For more examples and advanced usage, see the [cookbook notebook](../../cookbook/msmarco_rag_end_to_end.ipynb) and [integration guide](../../docs/how_to/rag/msmarco_vllm.md).