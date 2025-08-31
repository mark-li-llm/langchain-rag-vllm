"""Pydantic models for API requests and responses.

This module defines the data models used by the FastAPI endpoints
for request validation and response serialization.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., description="User query text", min_length=1, max_length=1000)
    top_k: int = Field(default=5, description="Number of results to return", ge=1, le=20)
    stream: bool = Field(default=True, description="Enable streaming response")
    max_output_tokens: Optional[int] = Field(default=None, description="Override max output tokens", ge=1, le=2000)
    temperature: Optional[float] = Field(default=None, description="Override temperature", ge=0.0, le=2.0)
    include_sources: bool = Field(default=True, description="Include source information in response")
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query is not empty after stripping."""
        if not v.strip():
            raise ValueError('Query cannot be empty or only whitespace')
        return v.strip()


class Citation(BaseModel):
    """Citation information for a retrieved document chunk."""
    number: int = Field(..., description="Citation number in response")
    doc_id: str = Field(..., description="Original document ID")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    source: str = Field(default="", description="Source information")
    url: Optional[str] = Field(default=None, description="Source URL if available")
    score: Optional[float] = Field(default=None, description="Retrieval similarity score")
    preview: Optional[str] = Field(default=None, description="Content preview")


class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int = Field(..., description="Tokens used in prompt")
    completion_tokens: int = Field(..., description="Tokens generated in completion")
    total_tokens: int = Field(..., description="Total tokens used")


class QueryMetadata(BaseModel):
    """Metadata for query response."""
    retrieval_count: int = Field(..., description="Number of documents retrieved")
    retrieval_time_ms: Optional[float] = Field(default=None, description="Time spent on retrieval")
    generation_time_ms: Optional[float] = Field(default=None, description="Time spent on generation")
    total_time_ms: Optional[float] = Field(default=None, description="Total processing time")
    usage: Optional[UsageInfo] = Field(default=None, description="Token usage information")
    citation_validation: Optional[Dict[str, Any]] = Field(default=None, description="Citation validation results")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str = Field(..., description="Generated answer text")
    citations: List[Citation] = Field(default_factory=list, description="List of citations")
    metadata: QueryMetadata = Field(..., description="Response metadata")
    trace_id: str = Field(..., description="Unique trace ID for request")
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "answer": "Machine learning is a subset of artificial intelligence [1] that enables computers to learn patterns from data [2].",
                "citations": [
                    {
                        "number": 1,
                        "doc_id": "doc_123",
                        "chunk_id": "doc_123:chunk:0",
                        "source": "msmarco:v2.1:123:0",
                        "url": "https://example.com/ml-intro",
                        "score": 0.89,
                        "preview": "Machine learning is a subset of artificial intelligence that..."
                    }
                ],
                "metadata": {
                    "retrieval_count": 5,
                    "total_time_ms": 1250.5,
                    "usage": {
                        "prompt_tokens": 450,
                        "completion_tokens": 120,
                        "total_tokens": 570
                    }
                },
                "trace_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV"
            }
        }


class StreamingChunk(BaseModel):
    """Individual chunk in streaming response."""
    chunk: str = Field(default="", description="Text chunk content")
    citations: List[Citation] = Field(default_factory=list, description="Citations (included in final chunk)")
    metadata: Optional[QueryMetadata] = Field(default=None, description="Metadata (included in final chunk)")
    trace_id: str = Field(..., description="Trace ID for request")
    is_final: bool = Field(default=False, description="Whether this is the final chunk")


class IngestRequest(BaseModel):
    """Request model for ingestion endpoint."""
    source_type: str = Field(..., description="Type of source data")
    payload: Optional[str] = Field(default=None, description="Source-specific payload")
    options: Dict[str, Any] = Field(default_factory=dict, description="Additional options")
    
    @validator('source_type')
    def validate_source_type(cls, v):
        """Validate source type is supported."""
        valid_types = {'msmarco', 'text', 'url'}
        if v not in valid_types:
            raise ValueError(f'source_type must be one of {valid_types}')
        return v


class IngestResponse(BaseModel):
    """Response model for ingestion endpoint."""
    status: str = Field(..., description="Ingestion status")
    message: str = Field(..., description="Status message")
    documents_processed: int = Field(default=0, description="Number of documents processed")
    chunks_created: int = Field(default=0, description="Number of chunks created")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    trace_id: str = Field(..., description="Unique trace ID for request")


class HealthStatus(BaseModel):
    """Component health status."""
    status: str = Field(..., description="Component status: ok, degraded, down")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional status details")
    response_time_ms: Optional[float] = Field(default=None, description="Component response time")
    error: Optional[str] = Field(default=None, description="Error message if status is not ok")


class HealthResponse(BaseModel):
    """Response model for health endpoint."""
    overall_status: str = Field(..., description="Overall system status")
    timestamp: str = Field(..., description="Health check timestamp")
    components: Dict[str, HealthStatus] = Field(..., description="Individual component statuses")
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "overall_status": "ok",
                "timestamp": "2024-01-15T10:30:00Z",
                "components": {
                    "rag_api": {
                        "status": "ok",
                        "response_time_ms": 5.2
                    },
                    "qdrant": {
                        "status": "ok",
                        "details": {"collections": 1, "points": 150000},
                        "response_time_ms": 12.1
                    },
                    "llm_upstream": {
                        "status": "ok",
                        "details": {"models": ["meta-llama/Meta-Llama-3.1-8B-Instruct"]},
                        "response_time_ms": 245.3
                    }
                }
            }
        }


class TraceInfo(BaseModel):
    """Trace information for debugging."""
    trace_id: str = Field(..., description="Unique trace identifier")
    timestamp: str = Field(..., description="Request timestamp")
    query: str = Field(..., description="Original query")
    retrieval_results: List[Dict[str, Any]] = Field(..., description="Retrieved documents with scores")
    prompt_preview: str = Field(..., description="Preview of formatted prompt")
    response_preview: str = Field(..., description="Preview of generated response")
    timing: Dict[str, float] = Field(..., description="Timing breakdown")


class TraceResponse(BaseModel):
    """Response model for trace endpoint."""
    trace: TraceInfo = Field(..., description="Trace information")
    found: bool = Field(..., description="Whether trace was found")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type/category")
    trace_id: Optional[str] = Field(default=None, description="Trace ID if available")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "error": "Invalid query parameter",
                "error_type": "ValidationError",
                "trace_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
                "details": {
                    "field": "query",
                    "message": "Query cannot be empty"
                }
            }
        }
