"""Observability utilities for monitoring and logging.

This module provides structured logging, Prometheus metrics, and tracing
capabilities for monitoring the RAG application in production.
"""

import json
import time
import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Warning: prometheus_client not available, metrics disabled")

from .config import settings


# Prometheus metrics (if available)
if PROMETHEUS_AVAILABLE:
    # Create custom registry to avoid conflicts
    registry = CollectorRegistry()
    
    # Request metrics
    request_counter = Counter(
        'rag_requests_total',
        'Total number of requests by endpoint and status',
        ['endpoint', 'status'],
        registry=registry
    )
    
    request_duration = Histogram(
        'rag_request_duration_seconds',
        'Request duration in seconds',
        ['endpoint'],
        buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        registry=registry
    )
    
    # RAG-specific metrics
    retrieval_duration = Histogram(
        'rag_retrieval_duration_seconds',
        'Document retrieval duration in seconds',
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
        registry=registry
    )
    
    generation_duration = Histogram(
        'rag_generation_duration_seconds',
        'LLM generation duration in seconds',
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        registry=registry
    )
    
    documents_retrieved = Histogram(
        'rag_documents_retrieved',
        'Number of documents retrieved per query',
        buckets=[1, 3, 5, 10, 20, 50],
        registry=registry
    )
    
    token_usage = Counter(
        'rag_tokens_total',
        'Total tokens used by type',
        ['type'],  # 'prompt', 'completion'
        registry=registry
    )
    
    # System metrics
    qdrant_operations = Counter(
        'qdrant_operations_total',
        'Total Qdrant operations by type and status',
        ['operation', 'status'],
        registry=registry
    )
    
    active_requests = Gauge(
        'rag_active_requests',
        'Currently active requests',
        registry=registry
    )


class StructuredLogger:
    """Structured JSON logger for the application."""
    
    def __init__(self, name: str = "rag_api"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, settings.log_level.upper()))
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Add JSON formatter
        handler = logging.StreamHandler()
        formatter = JsonFormatter()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_request(
        self,
        trace_id: str,
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float,
        **kwargs
    ):
        """Log request information."""
        self.logger.info("request", extra={
            "trace_id": trace_id,
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "duration_ms": duration_ms,
            **kwargs
        })
    
    def log_rag_operation(
        self,
        trace_id: str,
        operation: str,
        query: str,
        retrieval_count: int,
        duration_ms: float,
        **kwargs
    ):
        """Log RAG operation details."""
        self.logger.info("rag_operation", extra={
            "trace_id": trace_id,
            "operation": operation,
            "query_length": len(query),
            "retrieval_count": retrieval_count,
            "duration_ms": duration_ms,
            **kwargs
        })
    
    def log_error(
        self,
        trace_id: str,
        error: str,
        error_type: str,
        **kwargs
    ):
        """Log error information."""
        self.logger.error("error", extra={
            "trace_id": trace_id,
            "error": error,
            "error_type": error_type,
            **kwargs
        })


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                              'pathname', 'filename', 'module', 'lineno', 
                              'funcName', 'created', 'msecs', 'relativeCreated', 
                              'thread', 'threadName', 'processName', 'process',
                              'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                    log_entry[key] = value
        
        return json.dumps(log_entry)


class MetricsCollector:
    """Collects and reports metrics."""
    
    def __init__(self):
        self.enabled = PROMETHEUS_AVAILABLE
    
    def record_request(self, endpoint: str, status: str, duration_seconds: float):
        """Record request metrics."""
        if not self.enabled:
            return
        
        request_counter.labels(endpoint=endpoint, status=status).inc()
        request_duration.labels(endpoint=endpoint).observe(duration_seconds)
    
    def record_retrieval(self, duration_seconds: float, document_count: int):
        """Record retrieval metrics."""
        if not self.enabled:
            return
        
        retrieval_duration.observe(duration_seconds)
        documents_retrieved.observe(document_count)
    
    def record_generation(self, duration_seconds: float, prompt_tokens: int, completion_tokens: int):
        """Record generation metrics."""
        if not self.enabled:
            return
        
        generation_duration.observe(duration_seconds)
        token_usage.labels(type='prompt').inc(prompt_tokens)
        token_usage.labels(type='completion').inc(completion_tokens)
    
    def record_qdrant_operation(self, operation: str, status: str):
        """Record Qdrant operation metrics."""
        if not self.enabled:
            return
        
        qdrant_operations.labels(operation=operation, status=status).inc()
    
    @contextmanager
    def active_request_counter(self):
        """Context manager for tracking active requests."""
        if self.enabled:
            active_requests.inc()
        try:
            yield
        finally:
            if self.enabled:
                active_requests.dec()


# Global instances
logger = StructuredLogger()
metrics = MetricsCollector()


def start_metrics_server(port: int = None):
    """Start Prometheus metrics server."""
    if not PROMETHEUS_AVAILABLE:
        print("Prometheus client not available, metrics server not started")
        return False
    
    port = port or settings.prometheus_port
    
    try:
        start_http_server(port, registry=registry)
        print(f"Prometheus metrics server started on port {port}")
        return True
    except Exception as e:
        print(f"Failed to start metrics server: {e}")
        return False


@contextmanager
def trace_operation(operation_name: str, trace_id: str, **metadata):
    """Context manager for tracing operations with metrics and logging."""
    start_time = time.time()
    
    try:
        yield
        
        # Success
        duration = time.time() - start_time
        logger.log_rag_operation(
            trace_id=trace_id,
            operation=operation_name,
            duration_ms=duration * 1000,
            status="success",
            **metadata
        )
        
    except Exception as e:
        # Error
        duration = time.time() - start_time
        logger.log_error(
            trace_id=trace_id,
            error=str(e),
            error_type=type(e).__name__,
            operation=operation_name,
            duration_ms=duration * 1000,
            **metadata
        )
        raise


def log_health_check(component: str, status: str, response_time_ms: float, **details):
    """Log health check results."""
    logger.logger.info("health_check", extra={
        "component": component,
        "status": status,
        "response_time_ms": response_time_ms,
        **details
    })


def log_system_event(event_type: str, **details):
    """Log system events (startup, shutdown, etc.)."""
    logger.logger.info("system_event", extra={
        "event_type": event_type,
        **details
    })


class RequestTracker:
    """Tracks request metrics and logging."""
    
    def __init__(self, trace_id: str, endpoint: str, method: str = "POST"):
        self.trace_id = trace_id
        self.endpoint = endpoint
        self.method = method
        self.start_time = time.time()
        self.metadata = {}
    
    def add_metadata(self, **kwargs):
        """Add metadata to be logged."""
        self.metadata.update(kwargs)
    
    def record_retrieval(self, duration_seconds: float, document_count: int):
        """Record retrieval metrics."""
        metrics.record_retrieval(duration_seconds, document_count)
        self.add_metadata(
            retrieval_duration_ms=duration_seconds * 1000,
            retrieval_count=document_count
        )
    
    def record_generation(self, duration_seconds: float, prompt_tokens: int, completion_tokens: int):
        """Record generation metrics."""
        metrics.record_generation(duration_seconds, prompt_tokens, completion_tokens)
        self.add_metadata(
            generation_duration_ms=duration_seconds * 1000,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
    
    def finish(self, status_code: int):
        """Finish request tracking."""
        duration = time.time() - self.start_time
        
        # Record metrics
        status = "success" if 200 <= status_code < 400 else "error"
        metrics.record_request(self.endpoint, status, duration)
        
        # Log request
        logger.log_request(
            trace_id=self.trace_id,
            endpoint=self.endpoint,
            method=self.method,
            status_code=status_code,
            duration_ms=duration * 1000,
            **self.metadata
        )


def setup_observability():
    """Setup observability components."""
    # Start metrics server if configured
    if hasattr(settings, 'prometheus_port') and settings.prometheus_port:
        start_metrics_server(settings.prometheus_port)
    
    # Log system startup
    log_system_event("application_startup", 
                     version="0.1.0",
                     log_level=settings.log_level)
    
    print("Observability setup completed")


def get_metrics_summary() -> Dict[str, Any]:
    """Get current metrics summary."""
    if not PROMETHEUS_AVAILABLE:
        return {"error": "Prometheus not available"}
    
    try:
        # This is a simplified version - in production you'd query the registry
        return {
            "metrics_enabled": True,
            "components": [
                "request_counter",
                "request_duration",
                "retrieval_duration", 
                "generation_duration",
                "documents_retrieved",
                "token_usage",
                "qdrant_operations",
                "active_requests"
            ]
        }
    except Exception as e:
        return {"error": str(e)}