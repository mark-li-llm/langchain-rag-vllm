"""Minimal smoke tests to validate basic imports and core components.

These tests avoid external services (LLM, Qdrant) and only check that
modules import and core models/utilities work as expected.
"""

from __future__ import annotations


def test_fastapi_app_import() -> None:
    from fastapi import FastAPI
    from app.api import app

    assert isinstance(app, FastAPI)


def test_schemas_query_request() -> None:
    from app.schemas import QueryRequest

    req = QueryRequest(query="hello world")
    assert req.query == "hello world"
    assert 1 <= req.top_k <= 20


def test_ids_generation() -> None:
    from app.ids import generate_trace_id, is_valid_trace_id, validate_uuid, generate_request_id

    trace_id = generate_trace_id()
    assert isinstance(trace_id, str)
    assert is_valid_trace_id(trace_id)

    req_id = generate_request_id()
    assert validate_uuid(req_id)


def test_preprocess_normalize_text() -> None:
    from app.preprocess import normalize_text

    assert normalize_text("  Hello\tWorld \n") == "Hello World"

