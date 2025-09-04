"""LCEL pipeline composition for RAG.

This module provides LangChain Expression Language (LCEL) composition
for the complete RAG pipeline including retrieval, prompting, and generation.
"""

import time
from typing import Dict, Any, List, Iterator, Optional

from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from .prompting import format_context_with_citations, format_citations, validate_citations_in_response


def build_rag_chain(
    retriever: BaseRetriever,
    llm: BaseChatModel,
    prompt: ChatPromptTemplate,
    max_context_tokens: int = 3000
) -> Runnable:
    """Build complete RAG chain using LCEL.
    
    This creates a LangChain Expression Language (LCEL) pipeline that:
    1. Takes a query as input
    2. Retrieves relevant documents 
    3. Formats context with citations
    4. Generates response with LLM
    5. Returns response with metadata
    
    Args:
        retriever: Configured retriever for document search
        llm: LLM for response generation
        prompt: Chat prompt template with context formatting
        max_context_tokens: Maximum tokens for context
        
    Returns:
        Configured LCEL Runnable chain
    """
    
    def retrieve_documents(query: str) -> List[Document]:
        """Retrieve documents for query."""
        start_time = time.time()
        docs = retriever.get_relevant_documents(query)
        retrieval_time = time.time() - start_time
        
        # Add retrieval timing to metadata
        for doc in docs:
            doc.metadata["retrieval_time_ms"] = retrieval_time * 1000
        
        return docs
    
    def format_context(data: Dict[str, Any]) -> Dict[str, Any]:
        """Format retrieved documents as context with citations."""
        query = data["query"]
        docs = data["docs"]
        
        if not docs:
            return {
                "query": query,
                "context": "No relevant context found.",
                "citations": [],
                "docs": docs
            }
        
        context, citations = format_context_with_citations(
            docs, max_context_tokens=max_context_tokens
        )
        
        return {
            "query": query,
            "context": context,
            "citations": citations,
            "docs": docs
        }
    
    def parse_response_with_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response and add metadata."""
        llm_response = data["llm_response"]
        citations = data["citations"]
        docs = data["docs"]

        # Extract text content from response
        if hasattr(llm_response, 'content'):
            response_text = llm_response.content
        else:
            response_text = str(llm_response)

        # Validate citations in response
        citation_validation = validate_citations_in_response(
            response_text, len(citations)
        )
        # Add warnings if placeholder-style IDs were used in citations
        try:
            import re
            placeholder_numbers = []
            for c in citations:
                did = str(c.get("doc_id", ""))
                cid = str(c.get("chunk_id", ""))
                if re.fullmatch(r"doc_\d+", did) or re.fullmatch(r"doc_\d+:chunk:\d+", cid):
                    # Prefer the assigned citation number, fallback to positional index
                    num = c.get("number")
                    if isinstance(num, int):
                        placeholder_numbers.append(num)
                    else:
                        placeholder_numbers.append(len(placeholder_numbers) + 1)
            if placeholder_numbers:
                msgs = [
                    f"placeholder ids used for citations: {placeholder_numbers}"
                ]
                citation_validation["warnings"] = msgs
        except Exception:
            # Do not fail response construction if warnings computation errs
            pass
        
        # Calculate token usage (rough estimation)
        prompt_tokens = len(data.get("context", "")) // 4  # Rough estimate
        completion_tokens = len(response_text) // 4
        
        return {
            "answer": response_text,
            "citations": citations,
            "metadata": {
                "retrieval_count": len(docs),
                "citation_validation": citation_validation,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }
        }
    
    # Build the LCEL chain
    chain = (
        # Input: {"query": str}
        {"query": RunnablePassthrough(), "docs": RunnableLambda(retrieve_documents)}
        | RunnableLambda(format_context)
        # Now we have: {"query": str, "context": str, "citations": list, "docs": list}
        | RunnableLambda(lambda x: {
            "llm_response": prompt | llm | StrOutputParser(),
            "citations": x["citations"],
            "docs": x["docs"],
            "context": x["context"]
        }.update({"llm_response": (prompt | llm | StrOutputParser()).invoke({"query": x["query"], "context": x["context"]})}) or {
            "llm_response": (prompt | llm | StrOutputParser()).invoke({"query": x["query"], "context": x["context"]}),
            "citations": x["citations"],
            "docs": x["docs"],
            "context": x["context"]
        })
        | RunnableLambda(parse_response_with_metadata)
    )
    
    return chain


def build_simple_rag_chain(
    retriever: BaseRetriever,
    llm: BaseChatModel,
    prompt: ChatPromptTemplate,
    max_context_tokens: int = 3000
) -> Runnable:
    """Build a simpler RAG chain for easier debugging.
    
    Args:
        retriever: Configured retriever for document search
        llm: LLM for response generation  
        prompt: Chat prompt template
        max_context_tokens: Maximum tokens for context
        
    Returns:
        Simple LCEL chain
    """
    def retrieve_and_format(query: str) -> Dict[str, str]:
        """Retrieve documents and format context."""
        docs = retriever.get_relevant_documents(query)
        
        if not docs:
            context = "No relevant context found."
        else:
            context, _ = format_context_with_citations(
                docs, max_context_tokens=max_context_tokens
            )
        
        return {"query": query, "context": context}
    
    # Simple chain: query -> retrieve -> format -> prompt -> llm -> parse
    chain = (
        RunnableLambda(retrieve_and_format)
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


def build_streaming_rag_chain(
    retriever: BaseRetriever,
    llm: BaseChatModel,
    prompt: ChatPromptTemplate,
    max_context_tokens: int = 3000
) -> Runnable:
    """Build RAG chain optimized for streaming responses.
    
    Args:
        retriever: Configured retriever for document search
        llm: LLM for response generation (must support streaming)
        prompt: Chat prompt template
        max_context_tokens: Maximum tokens for context
        
    Returns:
        Streaming-capable LCEL chain
    """
    def setup_streaming_context(query: str) -> Dict[str, Any]:
        """Set up context for streaming response."""
        start_time = time.time()
        docs = retriever.get_relevant_documents(query)
        retrieval_time = time.time() - start_time
        
        context, citations = format_context_with_citations(
            docs, max_context_tokens=max_context_tokens
        )
        
        return {
            "query": query,
            "context": context,
            "citations": citations,
            "docs": docs,
            "retrieval_time_ms": retrieval_time * 1000
        }
    
    # For streaming, we need to handle the context setup separately
    # and then stream the LLM response
    def create_streaming_response(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Create streaming response with metadata."""
        query = data["query"]
        context = data["context"]
        citations = data["citations"]
        
        # Generate prompt inputs
        prompt_inputs = {"query": query, "context": context}
        
        # Stream LLM response
        response_chunks = []
        for chunk in llm.stream(prompt.format_messages(**prompt_inputs)):
            if hasattr(chunk, 'content') and chunk.content:
                response_chunks.append(chunk.content)
                yield {
                    "chunk": chunk.content,
                    "citations": citations,
                    "is_final": False
                }
        
        # Send final message with complete metadata
        full_response = "".join(response_chunks)
        citation_validation = validate_citations_in_response(full_response, len(citations))
        
        yield {
            "chunk": "",
            "answer": full_response,
            "citations": citations,
            "metadata": {
                "retrieval_count": len(data["docs"]),
                "retrieval_time_ms": data["retrieval_time_ms"],
                "citation_validation": citation_validation
            },
            "is_final": True
        }
    
    # Chain for streaming
    chain = (
        RunnableLambda(setup_streaming_context)
        | RunnableLambda(create_streaming_response)
    )
    
    return chain


def create_chain_with_timing(base_chain: Runnable) -> Runnable:
    """Wrap a chain with timing information.
    
    Args:
        base_chain: Base LCEL chain to wrap
        
    Returns:
        Chain with timing metadata
    """
    def add_timing(inputs: Any) -> Dict[str, Any]:
        """Add timing to chain execution."""
        start_time = time.time()
        
        try:
            result = base_chain.invoke(inputs)
            execution_time = time.time() - start_time
            
            # Add timing to result
            if isinstance(result, dict):
                if "metadata" not in result:
                    result["metadata"] = {}
                result["metadata"]["execution_time_ms"] = execution_time * 1000
            else:
                # If result is not dict, wrap it
                result = {
                    "result": result,
                    "metadata": {"execution_time_ms": execution_time * 1000}
                }
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "metadata": {"execution_time_ms": execution_time * 1000}
            }
    
    return RunnableLambda(add_timing)


def validate_chain(
    chain: Runnable,
    test_query: str = "What is machine learning?"
) -> Dict[str, Any]:
    """Validate that a RAG chain is working correctly.
    
    Args:
        chain: RAG chain to validate
        test_query: Test query to use
        
    Returns:
        Validation results
    """
    try:
        start_time = time.time()
        result = chain.invoke(test_query)
        execution_time = time.time() - start_time
        
        # Analyze result structure
        has_answer = False
        has_citations = False
        has_metadata = False
        
        if isinstance(result, dict):
            has_answer = "answer" in result and result["answer"]
            has_citations = "citations" in result and isinstance(result["citations"], list)
            has_metadata = "metadata" in result
        elif isinstance(result, str):
            has_answer = bool(result)
        
        return {
            "status": "success",
            "execution_time_ms": execution_time * 1000,
            "has_answer": has_answer,
            "has_citations": has_citations,
            "has_metadata": has_metadata,
            "result_type": type(result).__name__,
            "result_preview": str(result)[:200] if result else "",
            "test_query": test_query
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "test_query": test_query
        }
