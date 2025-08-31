"""Retrieval system with dense, sparse, and ensemble options.

This module provides retriever construction including dense vector search,
BM25 sparse retrieval, and ensemble methods using Reciprocal Rank Fusion.
"""

from typing import List, Iterable, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from qdrant_client import QdrantClient

from .index_qdrant import as_retriever as qdrant_as_retriever


def build_dense_retriever(
    qdrant_client: QdrantClient,
    collection_name: str,
    embeddings: Embeddings,
    k_pre: int = 50,
    score_threshold: Optional[float] = None
) -> BaseRetriever:
    """Build dense vector retriever using Qdrant.
    
    Args:
        qdrant_client: Qdrant client instance
        collection_name: Name of the Qdrant collection
        embeddings: Embeddings instance for query encoding
        k_pre: Number of documents to retrieve (before reranking)
        score_threshold: Minimum similarity score threshold
        
    Returns:
        Configured dense retriever
    """
    return qdrant_as_retriever(
        client=qdrant_client,
        collection_name=collection_name,
        embeddings=embeddings,
        k=k_pre,
        score_threshold=score_threshold
    )


def build_bm25_retriever(
    docs_iter: Iterable[Document],
    k_pre: int = 50
) -> BM25Retriever:
    """Build BM25 sparse retriever from document collection.
    
    Args:
        docs_iter: Iterable of Document objects for BM25 index
        k_pre: Number of documents to retrieve
        
    Returns:
        Configured BM25 retriever
        
    Raises:
        ValueError: If no documents provided
    """
    # Convert to list if not already
    docs = list(docs_iter) if not isinstance(docs_iter, list) else docs_iter
    
    if not docs:
        raise ValueError("Cannot create BM25 retriever: no documents provided")
    
    print(f"Building BM25 retriever from {len(docs)} documents")
    
    # Create BM25 retriever
    retriever = BM25Retriever.from_documents(docs, k=k_pre)
    
    print(f"BM25 retriever created with k={k_pre}")
    return retriever


def ensemble_retriever(
    dense: BaseRetriever,
    sparse: BaseRetriever,
    weights: Tuple[float, float] = (0.7, 0.3),
    k_final: int = 5
) -> BaseRetriever:
    """Create ensemble retriever combining dense and sparse methods.
    
    Uses Reciprocal Rank Fusion (RRF) to combine results from multiple
    retrievers, providing robust retrieval that leverages both semantic
    similarity and exact term matching.
    
    Args:
        dense: Dense vector retriever (e.g., from Qdrant)
        sparse: Sparse retriever (e.g., BM25)
        weights: Relative weights for (dense, sparse) retrievers
        k_final: Final number of documents to return
        
    Returns:
        Configured ensemble retriever
        
    Raises:
        ValueError: If weights don't sum to approximately 1.0
    """
    # Validate weights
    if abs(sum(weights) - 1.0) > 0.01:
        raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")
    
    if len(weights) != 2:
        raise ValueError(f"Expected 2 weights for dense and sparse, got {len(weights)}")
    
    print(f"Creating ensemble retriever with weights {weights} and k_final={k_final}")
    
    # Create ensemble retriever
    ensemble = EnsembleRetriever(
        retrievers=[dense, sparse],
        weights=list(weights),
        search_type="mmr"  # Maximum Marginal Relevance for diversity
    )
    
    # Note: EnsembleRetriever uses its own k parameter, so we need to configure it
    # The k parameter controls how many results are returned from the ensemble
    ensemble.k = k_final
    
    return ensemble


def build_retrieval_pipeline(
    qdrant_client: QdrantClient,
    collection_name: str,
    embeddings: Embeddings,
    bm25_docs: Optional[Iterable[Document]] = None,
    use_bm25: bool = True,
    dense_k: int = 50,
    sparse_k: int = 50,
    final_k: int = 5,
    ensemble_weights: Tuple[float, float] = (0.7, 0.3),
    score_threshold: Optional[float] = None
) -> BaseRetriever:
    """Build complete retrieval pipeline with optional ensemble.
    
    Args:
        qdrant_client: Qdrant client instance
        collection_name: Name of the Qdrant collection
        embeddings: Embeddings instance
        bm25_docs: Documents for BM25 index (if use_bm25=True)
        use_bm25: Whether to include BM25 in ensemble
        dense_k: Number of documents from dense retriever
        sparse_k: Number of documents from sparse retriever
        final_k: Final number of documents to return
        ensemble_weights: Weights for (dense, sparse) in ensemble
        score_threshold: Minimum similarity score for dense retriever
        
    Returns:
        Configured retriever (dense-only or ensemble)
        
    Raises:
        ValueError: If BM25 requested but no documents provided
    """
    print("Building retrieval pipeline...")
    
    # Build dense retriever
    dense_retriever = build_dense_retriever(
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        embeddings=embeddings,
        k_pre=dense_k,
        score_threshold=score_threshold
    )
    print(f"Dense retriever built with k={dense_k}")
    
    # If BM25 not requested, return dense-only
    if not use_bm25:
        print("Using dense-only retrieval")
        # Adjust dense retriever to return final_k documents
        dense_retriever.search_kwargs["k"] = final_k
        return dense_retriever
    
    # Build BM25 retriever
    if bm25_docs is None:
        raise ValueError("BM25 requested but no documents provided for indexing")
    
    sparse_retriever = build_bm25_retriever(bm25_docs, k_pre=sparse_k)
    print(f"BM25 retriever built with k={sparse_k}")
    
    # Build ensemble
    retriever = ensemble_retriever(
        dense=dense_retriever,
        sparse=sparse_retriever,
        weights=ensemble_weights,
        k_final=final_k
    )
    print(f"Ensemble retriever built with final k={final_k}")
    
    return retriever


def validate_retriever(retriever: BaseRetriever, test_query: str = "test query") -> dict:
    """Validate that a retriever is working correctly.
    
    Args:
        retriever: Retriever instance to test
        test_query: Query string to use for testing
        
    Returns:
        Dictionary with validation results
    """
    try:
        # Test retrieval
        results = retriever.get_relevant_documents(test_query)
        
        # Analyze results
        if not results:
            return {
                "status": "warning",
                "message": "Retriever returned no results",
                "results_count": 0
            }
        
        # Check document structure
        first_doc = results[0]
        has_content = bool(first_doc.page_content)
        has_metadata = bool(first_doc.metadata)
        
        return {
            "status": "success",
            "results_count": len(results),
            "first_doc_has_content": has_content,
            "first_doc_has_metadata": has_metadata,
            "first_doc_content_preview": first_doc.page_content[:100] if has_content else "",
            "metadata_keys": list(first_doc.metadata.keys()) if has_metadata else []
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }


def get_retrieval_stats(
    retriever: BaseRetriever,
    test_queries: List[str],
    max_queries: int = 10
) -> dict:
    """Get statistics about retriever performance.
    
    Args:
        retriever: Retriever to analyze
        test_queries: List of test queries
        max_queries: Maximum number of queries to test
        
    Returns:
        Dictionary with retrieval statistics
    """
    import time
    
    if not test_queries:
        return {"error": "No test queries provided"}
    
    # Limit number of test queries
    queries_to_test = test_queries[:max_queries]
    
    results = []
    total_time = 0
    
    for query in queries_to_test:
        try:
            start_time = time.time()
            docs = retriever.get_relevant_documents(query)
            query_time = time.time() - start_time
            
            total_time += query_time
            
            results.append({
                "query": query,
                "results_count": len(docs),
                "query_time_ms": query_time * 1000,
                "success": True
            })
            
        except Exception as e:
            results.append({
                "query": query,
                "error": str(e),
                "success": False
            })
    
    # Calculate statistics
    successful_queries = [r for r in results if r["success"]]
    
    if not successful_queries:
        return {
            "status": "error",
            "message": "No successful queries",
            "results": results
        }
    
    result_counts = [r["results_count"] for r in successful_queries]
    query_times = [r["query_time_ms"] for r in successful_queries]
    
    return {
        "status": "success",
        "queries_tested": len(queries_to_test),
        "successful_queries": len(successful_queries),
        "avg_results_per_query": sum(result_counts) / len(result_counts),
        "min_results": min(result_counts),
        "max_results": max(result_counts),
        "avg_query_time_ms": sum(query_times) / len(query_times),
        "min_query_time_ms": min(query_times),
        "max_query_time_ms": max(query_times),
        "total_time_ms": total_time * 1000,
        "queries_per_second": len(successful_queries) / total_time if total_time > 0 else 0,
        "detailed_results": results
    }