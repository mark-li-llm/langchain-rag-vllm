"""MS MARCO evaluation framework for retrieval quality assessment.

This module provides evaluation utilities using MS MARCO relevance labels
to compute standard retrieval metrics like Recall@k and MRR@10.
"""

import time
import csv
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from .dataset_msmarco import load_msmarco


def evaluate_retrieval(
    retriever: BaseRetriever,
    queries: List[Dict[str, Any]],
    ground_truth_mapping: Dict[str, List[str]],
    k_list: List[int] = [5, 10],
    max_queries: Optional[int] = None
) -> Dict[str, Any]:
    """Evaluate retrieval performance using MS MARCO ground truth.
    
    Args:
        retriever: Configured retriever to evaluate
        queries: List of query dictionaries with query_id and query_text
        ground_truth_mapping: Mapping from query_id to list of relevant doc_ids
        k_list: List of k values for Recall@k computation
        max_queries: Maximum number of queries to evaluate (for speed)
        
    Returns:
        Dictionary with evaluation metrics
    """
    if max_queries:
        queries = queries[:max_queries]
    
    print(f"Evaluating retrieval on {len(queries)} queries...")
    
    results = []
    total_retrieval_time = 0
    
    for i, query_data in enumerate(queries):
        if i % 100 == 0:
            print(f"Processed {i}/{len(queries)} queries")
        
        query_id = query_data["query_id"]
        query_text = query_data["query_text"]
        
        # Get ground truth for this query
        relevant_docs = ground_truth_mapping.get(query_id, [])
        if not relevant_docs:
            continue  # Skip queries without ground truth
        
        # Retrieve documents
        start_time = time.time()
        try:
            retrieved_docs = retriever.get_relevant_documents(query_text)
            retrieval_time = time.time() - start_time
            total_retrieval_time += retrieval_time
        except Exception as e:
            print(f"Error retrieving for query {query_id}: {e}")
            continue
        
        # Extract doc IDs from retrieved documents
        retrieved_doc_ids = []
        for doc in retrieved_docs:
            # Extract original doc_id from metadata
            doc_id = doc.metadata.get("doc_id", "")
            if doc_id:
                retrieved_doc_ids.append(doc_id)
        
        # Compute metrics for this query
        query_result = {
            "query_id": query_id,
            "query_text": query_text,
            "retrieved_count": len(retrieved_doc_ids),
            "relevant_count": len(relevant_docs),
            "retrieval_time_ms": retrieval_time * 1000
        }
        
        # Compute Recall@k for each k
        for k in k_list:
            retrieved_at_k = retrieved_doc_ids[:k]
            hits = len(set(retrieved_at_k) & set(relevant_docs))
            recall_at_k = hits / len(relevant_docs) if relevant_docs else 0
            query_result[f"recall_at_{k}"] = recall_at_k
            query_result[f"hits_at_{k}"] = hits
        
        # Compute reciprocal rank for MRR
        reciprocal_rank = 0
        for rank, doc_id in enumerate(retrieved_doc_ids[:10], 1):  # MRR@10
            if doc_id in relevant_docs:
                reciprocal_rank = 1.0 / rank
                break
        query_result["reciprocal_rank"] = reciprocal_rank
        
        results.append(query_result)
    
    # Aggregate metrics
    if not results:
        return {"error": "No valid queries evaluated"}
    
    metrics = compute_aggregate_metrics(results, k_list)
    metrics["evaluation_summary"] = {
        "total_queries": len(results),
        "avg_retrieval_time_ms": total_retrieval_time * 1000 / len(results),
        "total_time_seconds": sum(r["retrieval_time_ms"] for r in results) / 1000
    }
    
    return metrics


def compute_aggregate_metrics(results: List[Dict[str, Any]], k_list: List[int]) -> Dict[str, Any]:
    """Compute aggregate metrics from individual query results.
    
    Args:
        results: List of per-query evaluation results
        k_list: List of k values used
        
    Returns:
        Dictionary with aggregate metrics
    """
    if not results:
        return {}
    
    # Compute mean metrics
    metrics = {}
    
    # Recall@k metrics
    for k in k_list:
        recall_values = [r[f"recall_at_{k}"] for r in results]
        metrics[f"recall_at_{k}"] = {
            "mean": sum(recall_values) / len(recall_values),
            "median": sorted(recall_values)[len(recall_values) // 2],
            "min": min(recall_values),
            "max": max(recall_values),
            "queries_with_hits": sum(1 for r in results if r[f"hits_at_{k}"] > 0),
            "hit_rate": sum(1 for r in results if r[f"hits_at_{k}"] > 0) / len(results)
        }
    
    # MRR@10
    rr_values = [r["reciprocal_rank"] for r in results]
    metrics["mrr_at_10"] = {
        "mean": sum(rr_values) / len(rr_values),
        "median": sorted(rr_values)[len(rr_values) // 2],
        "queries_with_hits": sum(1 for rr in rr_values if rr > 0),
        "hit_rate": sum(1 for rr in rr_values if rr > 0) / len(results)
    }
    
    # Retrieval statistics
    retrieved_counts = [r["retrieved_count"] for r in results]
    relevant_counts = [r["relevant_count"] for r in results]
    
    metrics["retrieval_stats"] = {
        "avg_retrieved": sum(retrieved_counts) / len(retrieved_counts),
        "avg_relevant": sum(relevant_counts) / len(relevant_counts),
        "min_retrieved": min(retrieved_counts),
        "max_retrieved": max(retrieved_counts)
    }
    
    return metrics


def compute_mrr_at_10(results: List[Dict[str, Any]]) -> float:
    """Compute Mean Reciprocal Rank at 10.
    
    Args:
        results: List of query evaluation results
        
    Returns:
        MRR@10 score
    """
    rr_values = [r["reciprocal_rank"] for r in results]
    return sum(rr_values) / len(rr_values) if rr_values else 0.0


def build_ground_truth_mapping(corpus_passages: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Build ground truth mapping from MS MARCO corpus.
    
    Args:
        corpus_passages: List of passage records with relevance labels
        
    Returns:
        Dictionary mapping query_id to list of relevant doc_ids
    """
    ground_truth = defaultdict(list)
    
    for passage in corpus_passages:
        query_id = passage.get("query_id")
        doc_id = passage.get("doc_id")
        is_selected = passage.get("label_is_selected", 0)
        
        if query_id and doc_id and is_selected == 1:
            ground_truth[query_id].append(doc_id)
    
    return dict(ground_truth)


def evaluate_msmarco_retrieval(
    retriever: BaseRetriever,
    corpus_split: str = "train",
    queries_split: str = "validation", 
    config: str = "v2.1",
    max_corpus_passages: Optional[int] = None,
    max_queries: int = 5000,
    k_list: List[int] = [5, 10]
) -> Dict[str, Any]:
    """End-to-end evaluation using MS MARCO dataset.
    
    Args:
        retriever: Configured retriever to evaluate
        corpus_split: MS MARCO corpus split to use
        queries_split: MS MARCO queries split to use
        config: MS MARCO dataset configuration
        max_corpus_passages: Maximum corpus passages to load
        max_queries: Maximum queries to evaluate
        k_list: List of k values for evaluation
        
    Returns:
        Complete evaluation results
    """
    print("Loading MS MARCO dataset for evaluation...")
    
    # Load dataset
    corpus_passages, queries = load_msmarco(
        corpus_split=corpus_split,
        queries_split=queries_split,
        config=config,
        max_corpus_passages=max_corpus_passages
    )
    
    print(f"Loaded {len(corpus_passages)} passages and {len(queries)} queries")
    
    # Build ground truth mapping
    ground_truth = build_ground_truth_mapping(corpus_passages)
    print(f"Built ground truth for {len(ground_truth)} queries")
    
    # Filter queries to those with ground truth
    valid_queries = [q for q in queries if q["query_id"] in ground_truth]
    print(f"Found {len(valid_queries)} queries with ground truth")
    
    if max_queries:
        valid_queries = valid_queries[:max_queries]
        print(f"Evaluating on {len(valid_queries)} queries")
    
    # Run evaluation
    results = evaluate_retrieval(
        retriever=retriever,
        queries=valid_queries,
        ground_truth_mapping=ground_truth,
        k_list=k_list,
        max_queries=max_queries
    )
    
    # Add dataset info
    results["dataset_info"] = {
        "config": config,
        "corpus_split": corpus_split,
        "queries_split": queries_split,
        "total_corpus_passages": len(corpus_passages),
        "total_queries": len(queries),
        "queries_with_ground_truth": len(ground_truth),
        "evaluated_queries": len(valid_queries)
    }
    
    return results


def save_evaluation_results(results: Dict[str, Any], output_path: str):
    """Save evaluation results to CSV and JSON files.
    
    Args:
        results: Evaluation results dictionary
        output_path: Base path for output files (without extension)
    """
    import json
    
    # Save aggregate results as JSON
    json_path = f"{output_path}_summary.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved summary to {json_path}")
    
    # Save per-query results as CSV if available
    if "query_results" in results:
        csv_path = f"{output_path}_details.csv"
        query_results = results["query_results"]
        
        if query_results:
            fieldnames = query_results[0].keys()
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(query_results)
            print(f"Saved detailed results to {csv_path}")


def format_evaluation_report(results: Dict[str, Any]) -> str:
    """Format evaluation results as a readable report.
    
    Args:
        results: Evaluation results dictionary
        
    Returns:
        Formatted report string
    """
    if "error" in results:
        return f"Evaluation Error: {results['error']}"
    
    lines = ["MS MARCO Retrieval Evaluation Report", "=" * 40, ""]
    
    # Dataset info
    if "dataset_info" in results:
        info = results["dataset_info"]
        lines.extend([
            "Dataset Information:",
            f"  Configuration: {info.get('config', 'N/A')}",
            f"  Corpus Split: {info.get('corpus_split', 'N/A')}",
            f"  Queries Split: {info.get('queries_split', 'N/A')}",
            f"  Evaluated Queries: {info.get('evaluated_queries', 'N/A')}",
            ""
        ])
    
    # Recall metrics
    for k in [5, 10]:
        metric_key = f"recall_at_{k}"
        if metric_key in results:
            recall = results[metric_key]
            lines.extend([
                f"Recall@{k}:",
                f"  Mean: {recall['mean']:.4f}",
                f"  Hit Rate: {recall['hit_rate']:.4f} ({recall['queries_with_hits']}/{results.get('evaluation_summary', {}).get('total_queries', 'N/A')} queries)",
                f"  Range: {recall['min']:.4f} - {recall['max']:.4f}",
                ""
            ])
    
    # MRR metric
    if "mrr_at_10" in results:
        mrr = results["mrr_at_10"]
        lines.extend([
            "MRR@10:",
            f"  Mean: {mrr['mean']:.4f}",
            f"  Hit Rate: {mrr['hit_rate']:.4f}",
            ""
        ])
    
    # Performance info
    if "evaluation_summary" in results:
        summary = results["evaluation_summary"]
        lines.extend([
            "Performance:",
            f"  Average Query Time: {summary.get('avg_retrieval_time_ms', 0):.2f}ms",
            f"  Total Evaluation Time: {summary.get('total_time_seconds', 0):.2f}s",
            ""
        ])
    
    return "\n".join(lines)