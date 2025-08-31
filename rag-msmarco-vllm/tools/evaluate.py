#!/usr/bin/env python3
"""Standalone script for evaluating RAG system performance.

This script runs comprehensive evaluation of the retrieval system
using MS MARCO ground truth labels.
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.embeddings import get_production_embeddings
from app.retrieval import build_retrieval_pipeline, validate_retriever, get_retrieval_stats
from app.eval_msmarco import evaluate_msmarco_retrieval, format_evaluation_report, save_evaluation_results
from app.observability import setup_observability, log_system_event


def main():
    """Main function for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval performance")
    
    parser.add_argument(
        "--max-queries",
        type=int,
        default=5000,
        help="Maximum number of queries to evaluate"
    )
    parser.add_argument(
        "--collection-name",
        default=settings.collection_name,
        help="Qdrant collection name"
    )
    parser.add_argument(
        "--embed-model",
        default=settings.embed_model_name,
        help="Embedding model name"
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[5, 10],
        help="K values for Recall@k evaluation"
    )
    parser.add_argument(
        "--use-bm25",
        action="store_true",
        default=settings.use_bm25,
        help="Include BM25 in ensemble retrieval"
    )
    parser.add_argument(
        "--topk-final",
        type=int,
        default=settings.topk_final,
        help="Final number of documents to retrieve"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file prefix for results (without extension)"
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed for query sampling"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Include detailed per-query results"
    )
    parser.add_argument(
        "--test-retriever",
        action="store_true",
        help="Run retriever validation tests first"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    print("MS MARCO RAG Evaluation")
    print("=" * 50)
    print(f"Max queries: {args.max_queries}")
    print(f"Collection: {args.collection_name}")
    print(f"Embedding model: {args.embed_model}")
    print(f"K values: {args.k_values}")
    print(f"Use BM25: {args.use_bm25}")
    print(f"Top-k final: {args.topk_final}")
    print(f"Sample seed: {args.sample_seed}")
    print("")
    
    try:
        # Setup observability
        setup_observability()
        log_system_event("evaluation_start_cli", max_queries=args.max_queries)
        
        start_time = time.time()
        
        # Step 1: Initialize components
        print("1. Initializing components...")
        
        # Connect to Qdrant
        from qdrant_client import QdrantClient
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        
        # Check collection exists
        try:
            from app.index_qdrant import get_collection_info
            collection_info = get_collection_info(client, args.collection_name)
            print(f"   Collection found: {collection_info['point_count']:,} points")
        except Exception as e:
            print(f"   Error: Collection '{args.collection_name}' not found: {e}")
            print("   Please run index build first")
            return False
        
        # Initialize embeddings
        embeddings = get_production_embeddings(args.embed_model)
        print(f"   Embeddings initialized: {args.embed_model}")
        
        # Build retrieval pipeline
        print("   Building retrieval pipeline...")
        retriever = build_retrieval_pipeline(
            qdrant_client=client,
            collection_name=args.collection_name,
            embeddings=embeddings,
            use_bm25=args.use_bm25,
            final_k=args.topk_final,
            dense_k=50,  # Get more for better recall measurement
            sparse_k=50 if args.use_bm25 else 0
        )
        print(f"   Retrieval pipeline built (BM25: {args.use_bm25})")
        print("")
        
        # Step 2: Validate retriever (optional)
        if args.test_retriever:
            print("2. Testing retriever...")
            validation = validate_retriever(retriever)
            
            if validation["status"] == "success":
                print(f"   ✓ Retriever validation passed")
                print(f"   - Results returned: {validation['results_count']}")
                print(f"   - Has content: {validation['first_doc_has_content']}")
                print(f"   - Has metadata: {validation['first_doc_has_metadata']}")
                if args.verbose and validation.get('metadata_keys'):
                    print(f"   - Metadata keys: {validation['metadata_keys']}")
            else:
                print(f"   ✗ Retriever validation failed: {validation.get('error', 'Unknown error')}")
                return False
            
            # Test performance on sample queries
            print("   Testing performance on sample queries...")
            test_queries = [
                "What is machine learning?",
                "How does neural network work?",
                "Define artificial intelligence",
                "Explain deep learning algorithms",
                "What are the applications of AI?"
            ]
            
            stats = get_retrieval_stats(retriever, test_queries)
            if stats.get("status") == "success":
                print(f"   ✓ Performance test passed")
                print(f"   - Avg query time: {stats['avg_query_time_ms']:.2f}ms")
                print(f"   - Avg results per query: {stats['avg_results_per_query']:.1f}")
                print(f"   - Queries per second: {stats['queries_per_second']:.2f}")
            else:
                print(f"   ⚠ Performance test had issues: {stats.get('message', 'Unknown')}")
            
            print("")
        
        # Step 3: Run MS MARCO evaluation
        print("3. Running MS MARCO evaluation...")
        
        # Set random seed for reproducibility
        import random
        random.seed(args.sample_seed)
        
        evaluation_start = time.time()
        
        results = evaluate_msmarco_retrieval(
            retriever=retriever,
            max_queries=args.max_queries,
            k_list=args.k_values
        )
        
        evaluation_time = time.time() - evaluation_start
        
        if "error" in results:
            print(f"   Error during evaluation: {results['error']}")
            return False
        
        print(f"   Evaluation completed in {evaluation_time:.2f} seconds")
        print("")
        
        # Step 4: Display results
        print("4. Evaluation Results")
        print("-" * 30)
        
        report = format_evaluation_report(results)
        print(report)
        
        # Step 5: Save results (optional)
        if args.output:
            print(f"5. Saving results to {args.output}...")
            
            # Include detailed results if requested
            if args.detailed and "query_results" in results:
                print(f"   Including detailed per-query results")
            
            save_evaluation_results(results, args.output)
            print(f"   Results saved")
            print("")
        
        # Final summary
        total_time = time.time() - start_time
        
        print("Evaluation Summary:")
        print(f"- Total time: {total_time:.2f} seconds")
        print(f"- Queries evaluated: {results.get('evaluation_summary', {}).get('total_queries', 'N/A')}")
        print(f"- Dataset: {results.get('dataset_info', {}).get('config', 'N/A')}")
        
        # Key metrics summary
        if "recall_at_5" in results:
            recall_5 = results["recall_at_5"]["mean"]
            print(f"- Recall@5: {recall_5:.4f}")
        
        if "recall_at_10" in results:
            recall_10 = results["recall_at_10"]["mean"]
            print(f"- Recall@10: {recall_10:.4f}")
        
        if "mrr_at_10" in results:
            mrr_10 = results["mrr_at_10"]["mean"]
            print(f"- MRR@10: {mrr_10:.4f}")
        
        # Performance summary
        eval_summary = results.get("evaluation_summary", {})
        if eval_summary:
            avg_time = eval_summary.get("avg_retrieval_time_ms", 0)
            print(f"- Avg query time: {avg_time:.2f}ms")
        
        # Log completion
        log_system_event("evaluation_complete_cli",
                         duration_seconds=total_time,
                         queries_evaluated=results.get("evaluation_summary", {}).get("total_queries", 0),
                         recall_at_5=results.get("recall_at_5", {}).get("mean", 0),
                         mrr_at_10=results.get("mrr_at_10", {}).get("mean", 0))
        
        return True
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)