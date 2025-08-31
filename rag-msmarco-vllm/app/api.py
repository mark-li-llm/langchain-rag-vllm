"""FastAPI application for RAG MS MARCO service.

This module provides the main API endpoints for querying, ingestion,
health checks, and tracing.
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, Optional

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from starlette.concurrency import run_in_threadpool

from .chunking import make_splitter, split_documents_batch
from .config import settings
from .dataset_msmarco import load_msmarco
from .embeddings import get_production_embeddings
from .ids import generate_trace_id, is_valid_trace_id
from .index_qdrant import (
    create_or_get_collection,
    get_collection_info,
    upsert_documents,
)
from .llm import get_production_llm, validate_llm_connection
from .pipeline import build_rag_chain, build_streaming_rag_chain
from .preprocess import filter_record, prepare_for_chunking
from .prompting import build_prompt_template
from .retrieval import build_retrieval_pipeline
from .schemas import (
    ErrorResponse,
    HealthResponse,
    HealthStatus,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    StreamingChunk,
    TraceResponse,
)
from .timing import TimingCollector

# Initialize FastAPI app
app = FastAPI(
    title="RAG MS MARCO API",
    description="Production RAG system using MS MARCO dataset with vLLM integration",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for web interface
import os

# Use absolute path to web directory
web_dir = "/Users/liyunxiao/rag-msmarco-vllm/web"
print(f"Mounting web directory: {web_dir}")  # Debug info
if os.path.exists(web_dir):
    app.mount("/static", StaticFiles(directory=web_dir), name="static")
    print("✅ Web static files mounted successfully")
else:
    print(f"❌ Web directory not found at: {web_dir}")

# Security
security = HTTPBearer()

# Global state (in production, use proper dependency injection)
rag_chain = None
streaming_chain = None
qdrant_client = None
embeddings = None
llm = None

# In-memory trace storage (in production, use Redis or database)
trace_storage: Dict[str, Dict[str, Any]] = {}


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify bearer token authentication."""
    if credentials.credentials != settings.auth_bearer_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    return credentials.credentials


def get_rag_chain():
    """Get or initialize RAG chain."""
    global rag_chain, streaming_chain, qdrant_client, embeddings, llm
    
    if rag_chain is None:
        # Initialize components
        qdrant_client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        embeddings = get_production_embeddings(settings.embed_model_name)
        llm = get_production_llm()
        
        # Build retrieval pipeline
        retriever = build_retrieval_pipeline(
            qdrant_client=qdrant_client,
            collection_name=settings.collection_name,
            embeddings=embeddings,
            use_bm25=settings.use_bm25,
            final_k=settings.topk_final
        )
        
        # Build prompt template
        prompt = build_prompt_template(max_context_tokens=settings.max_context_tokens)
        
        # Build chains
        rag_chain = build_rag_chain(retriever, llm, prompt, settings.max_context_tokens)
        streaming_chain = build_streaming_rag_chain(retriever, llm, prompt, settings.max_context_tokens)
    
    return rag_chain, streaming_chain


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    print("Initializing RAG MS MARCO API...")
    
    try:
        # Initialize chains
        get_rag_chain()
        print("RAG chains initialized successfully")
        
        # Validate LLM connection
        validation = validate_llm_connection(llm)
        if validation["status"] != "success":
            print(f"Warning: LLM validation failed: {validation.get('error', 'Unknown error')}")
        else:
            print("LLM connection validated")
        
        print("API startup completed")
        
    except Exception as e:
        print(f"Error during startup: {e}")
        raise


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add timing and trace ID headers to responses."""
    # Generate trace ID for request
    trace_id = generate_trace_id()
    request.state.trace_id = trace_id
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Trace-ID"] = trace_id
    
    return response


@app.post("/v1/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest,
    req: Request,
    token: str = Depends(verify_token)
):
    """Query endpoint for RAG responses."""
    trace_id = req.state.trace_id
    timing = TimingCollector()
    
    try:
        with timing.time("total"):
            # Get chains
            rag_chain, _ = get_rag_chain()
            
            # Handle streaming
            if request.stream:
                return StreamingResponse(
                    stream_query_response(request, trace_id, timing),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive"
                    }
                )
            
            # Non-streaming response
            with timing.time("rag_generation"):
                # Run synchronous LangChain invocation in a thread pool to avoid blocking the event loop
                result = await run_in_threadpool(rag_chain.invoke, request.query)
            
            # Build response
            response = QueryResponse(
                answer=result.get("answer", ""),
                citations=result.get("citations", []),
                metadata=result.get("metadata", {}),
                trace_id=trace_id
            )
            
            # Store trace
            store_trace(trace_id, request.query, result, timing.get_timings())
            
            return response
    
    except Exception as e:
        # Store error trace
        store_trace(trace_id, request.query, {"error": str(e)}, timing.get_timings())
        
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


async def stream_query_response(request: QueryRequest, trace_id: str, timing: TimingCollector):
    """Async generator to stream RAG responses, handling blocking calls in a thread pool."""
    try:
        _, streaming_chain = get_rag_chain()
        import json as _json

        # Run the entire blocking generator in a thread pool to get all chunks.
        # This prevents blocking the main event loop. It buffers the full response.
        all_chunks = await run_in_threadpool(list, streaming_chain.stream(request.query))
        
        # Now, asynchronously yield the collected chunks.
        for chunk_data in all_chunks:
            payload = dict(chunk_data)
            payload["delta"] = payload.pop("chunk", "")
            payload["trace_id"] = trace_id
            yield f"data: {_json.dumps(payload, ensure_ascii=False)}\n\n"

            if payload.get("is_final"):
                store_trace(trace_id, request.query, chunk_data, timing.get_timings())
        
        yield "data: [DONE]\n\n"

    except Exception as e:
        import json as _json
        # If an error occurs, yield a final error message.
        error_payload = {"delta": "", "trace_id": trace_id, "is_final": True, "error": str(e)}
        yield f"data: {_json.dumps(error_payload, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        # Also store the error trace
        store_trace(trace_id, request.query, {"error": str(e)}, timing.get_timings())


@app.post("/v1/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    request: IngestRequest,
    req: Request,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Ingest data into the system."""
    trace_id = req.state.trace_id
    start_time = time.time()
    
    try:
        if request.source_type == "msmarco":
            # Trigger MS MARCO index rebuild
            background_tasks.add_task(rebuild_msmarco_index)
            
            return IngestResponse(
                status="accepted",
                message="MS MARCO index rebuild started",
                documents_processed=0,
                chunks_created=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                trace_id=trace_id
            )
        
        elif request.source_type == "text":
            if not request.payload:
                raise HTTPException(status_code=400, detail="Text payload required")
            
            # Process single text document
            result = await process_text_document(request.payload, trace_id)
            
            return IngestResponse(
                status="completed",
                message="Text document processed",
                documents_processed=result["documents_processed"],
                chunks_created=result["chunks_created"],
                processing_time_ms=(time.time() - start_time) * 1000,
                trace_id=trace_id
            )
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported source type: {request.source_type}"
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )


@app.get("/")
async def root():
    """Redirect to web interface."""
    return RedirectResponse(url="/static/index.html")


@app.get("/v1/health", response_model=HealthResponse)
async def health_endpoint():
    """Health check endpoint."""
    timestamp = datetime.utcnow().isoformat() + "Z"
    components = {}
    
    # Check RAG API
    components["rag_api"] = HealthStatus(status="ok", response_time_ms=1.0)
    
    # Check Qdrant
    try:
        start_time = time.time()
        if qdrant_client:
            collections = qdrant_client.get_collections()
            response_time = (time.time() - start_time) * 1000
            components["qdrant"] = HealthStatus(
                status="ok",
                response_time_ms=response_time,
                details={"collections_count": len(collections.collections)}
            )
        else:
            components["qdrant"] = HealthStatus(
                status="degraded",
                error="Qdrant client not initialized"
            )
    except Exception as e:
        components["qdrant"] = HealthStatus(
            status="down",
            error=str(e)
        )
    
    # Check LLM upstream
    try:
        start_time = time.time()
        if llm:
            validation = validate_llm_connection(llm, "health check")
            response_time = (time.time() - start_time) * 1000
            
            if validation["status"] == "success":
                components["llm_upstream"] = HealthStatus(
                    status="ok",
                    response_time_ms=response_time,
                    details={"model": settings.llm_model}
                )
            else:
                components["llm_upstream"] = HealthStatus(
                    status="degraded",
                    response_time_ms=response_time,
                    error=validation.get("error", "Unknown error")
                )
        else:
            components["llm_upstream"] = HealthStatus(
                status="degraded",
                error="LLM not initialized"
            )
    except Exception as e:
        components["llm_upstream"] = HealthStatus(
            status="down",
            error=str(e)
        )
    
    # Determine overall status
    statuses = [comp.status for comp in components.values()]
    if "down" in statuses:
        overall_status = "down"
    elif "degraded" in statuses:
        overall_status = "degraded"
    else:
        overall_status = "ok"
    
    return HealthResponse(
        overall_status=overall_status,
        timestamp=timestamp,
        components=components
    )


@app.get("/v1/traces/{trace_id}", response_model=TraceResponse)
async def get_trace(trace_id: str, token: str = Depends(verify_token)):
    """Get trace information for debugging."""
    if not is_valid_trace_id(trace_id):
        raise HTTPException(status_code=400, detail="Invalid trace ID format")
    
    if trace_id not in trace_storage:
        return TraceResponse(trace=None, found=False)
    
    trace_data = trace_storage[trace_id]
    
    return TraceResponse(
        trace=trace_data,
        found=True
    )


def store_trace(trace_id: str, query: str, result: Dict[str, Any], timings: Dict[str, float]):
    """Store trace information for debugging."""
    trace_info = {
        "trace_id": trace_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "query": query,
        "retrieval_results": result.get("docs", [])[:3],  # Store first 3 docs
        "prompt_preview": result.get("context", "")[:500],  # First 500 chars
        "response_preview": result.get("answer", "")[:200],  # First 200 chars
        "timing": timings,
        "error": result.get("error")
    }
    
    # Store with TTL (in production, use Redis with expiration)
    trace_storage[trace_id] = trace_info
    
    # Simple cleanup: keep only last 1000 traces
    if len(trace_storage) > 1000:
        oldest_trace = min(trace_storage.keys())
        del trace_storage[oldest_trace]


async def process_text_document(text: str, trace_id: str) -> Dict[str, Any]:
    """Process a single text document."""
    # Create document record
    record = {
        "doc_id": f"text_{trace_id}",
        "text": text,
        "source": f"manual_upload:{trace_id}"
    }
    
    # Filter and prepare
    if not filter_record(record):
        raise HTTPException(status_code=400, detail="Document failed quality filters")
    
    prepared_text = prepare_for_chunking(record)
    record["text"] = prepared_text
    
    # Chunk document
    splitter = make_splitter(settings.chunk_size_chars, settings.chunk_overlap_chars)
    documents = split_documents_batch([record], splitter)
    
    if not documents:
        raise HTTPException(status_code=400, detail="No chunks created from document")
    
    # Get embedding model
    if embeddings is None:
        raise HTTPException(status_code=500, detail="Embeddings not initialized")
    
    # Upsert to Qdrant
    result = upsert_documents(
        qdrant_client, settings.collection_name, documents, embeddings
    )
    
    return {
        "documents_processed": 1,
        "chunks_created": len(documents)
    }


async def rebuild_msmarco_index():
    """Background task to rebuild MS MARCO index."""
    try:
        print("Starting MS MARCO index rebuild...")
        
        # Load MS MARCO data
        corpus_passages, queries = load_msmarco(
            corpus_split=settings.hf_split_corpus,
            queries_split=settings.hf_split_queries,
            config=settings.hf_dataset_config,
            max_corpus_passages=settings.corpus_sample_size
        )
        
        # Filter and prepare documents
        filtered_passages = [p for p in corpus_passages if filter_record(p)]
        print(f"Filtered to {len(filtered_passages)} valid passages")
        
        # Chunk documents
        splitter = make_splitter(settings.chunk_size_chars, settings.chunk_overlap_chars)
        documents = split_documents_batch(filtered_passages, splitter)
        print(f"Created {len(documents)} chunks")
        
        # Create collection
        embedding_model = get_production_embeddings(settings.embed_model_name)
        
        # Get vector size by testing embedding
        test_embedding = embedding_model.embed_query("test")
        vector_size = len(test_embedding)
        
        create_or_get_collection(
            qdrant_client, settings.collection_name, vector_size
        )
        
        # Upsert documents
        result = upsert_documents(
            qdrant_client, settings.collection_name, documents, embedding_model
        )
        
        print(f"Index rebuild completed: {result}")
        
    except Exception as e:
        print(f"Index rebuild failed: {e}")


if __name__ == "__main__":
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=8000,
        workers=settings.uvicorn_workers,
        log_level=settings.log_level
    )