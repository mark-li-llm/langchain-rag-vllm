"""CLI entrypoints for RAG MS MARCO application.

This module provides command-line interfaces for building indexes,
running evaluations, and serving the API.
"""

import argparse
import sys
import time
from typing import Optional

from .config import settings
from .dataset_msmarco import load_msmarco, get_dataset_stats
from .preprocess import filter_record, prepare_for_chunking
from .chunking import make_splitter, split_documents_batch, get_chunking_stats
from .embeddings import get_production_embeddings, validate_embeddings
from .index_qdrant import create_or_get_collection, upsert_documents, get_collection_info
from .retrieval import build_retrieval_pipeline, validate_retriever
from .eval_msmarco import evaluate_msmarco_retrieval, format_evaluation_report, save_evaluation_results
from .llm import get_production_llm, validate_llm_connection
from .prompting import build_prompt_template
from .pipeline import build_rag_chain, validate_chain
from .observability import setup_observability, log_system_event


def build_index_command(args):
    """Build or rebuild the Qdrant index from MS MARCO data."""
    print("Starting index build process...")
    
    try:
        # Setup observability
        setup_observability()
        log_system_event("index_build_start", corpus_sample_size=settings.corpus_sample_size)
        
        start_time = time.time()
        
        # Load MS MARCO dataset
        print(f"Loading MS MARCO dataset (sample size: {settings.corpus_sample_size})...")
        corpus_passages, queries = load_msmarco(
            corpus_split=settings.hf_split_corpus,
            queries_split=settings.hf_split_queries,
            config=settings.hf_dataset_config,
            max_corpus_passages=settings.corpus_sample_size
        )
        
        # Print dataset statistics
        stats = get_dataset_stats(corpus_passages, queries)
        print(f"Dataset stats: {stats}")
        
        # Filter passages
        print("Filtering passages...")
        filtered_passages = []
        for passage in corpus_passages:
            if filter_record(passage):
                # Prepare text for chunking
                passage["text"] = prepare_for_chunking(passage)
                filtered_passages.append(passage)
        
        print(f"Filtered to {len(filtered_passages)} valid passages from {len(corpus_passages)}")
        
        if not filtered_passages:
            print("Error: No valid passages after filtering")
            return False
        
        # Create text splitter
        splitter = make_splitter(
            chunk_size_chars=settings.chunk_size_chars,
            chunk_overlap_chars=settings.chunk_overlap_chars
        )
        
        # Split documents into chunks
        print("Splitting documents into chunks...")
        documents = split_documents_batch(filtered_passages, splitter)
        
        chunking_stats = get_chunking_stats(documents)
        print(f"Chunking stats: {chunking_stats}")
        
        if not documents:
            print("Error: No chunks created")
            return False
        
        # Initialize embeddings
        print(f"Initializing embeddings model: {settings.embed_model_name}")
        embeddings = get_production_embeddings(settings.embed_model_name)
        
        # Validate embeddings
        validation = validate_embeddings(embeddings)
        if validation["status"] != "success":
            print(f"Error: Embeddings validation failed: {validation}")
            return False
        
        print(f"Embeddings validated: {validation['embedding_dim']} dimensions")
        
        # Connect to Qdrant
        from qdrant_client import QdrantClient
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        
        # Create collection
        vector_size = validation["embedding_dim"]
        create_or_get_collection(client, settings.collection_name, vector_size)
        
        # Upsert documents
        print("Upserting documents to Qdrant...")
        upsert_result = upsert_documents(client, settings.collection_name, documents, embeddings)
        
        # Get final collection info
        collection_info = get_collection_info(client, settings.collection_name)
        
        total_time = time.time() - start_time
        
        # Print summary
        print("\nIndex build completed successfully!")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Collection: {settings.collection_name}")
        print(f"Documents processed: {upsert_result['processed']}")
        print(f"Total points in collection: {collection_info['point_count']}")
        print(f"Vector size: {collection_info['vector_size']}")
        
        log_system_event("index_build_complete", 
                         duration_seconds=total_time,
                         documents_processed=upsert_result['processed'],
                         total_points=collection_info['point_count'])
        
        return True
        
    except Exception as e:
        print(f"Error during index build: {e}")
        log_system_event("index_build_error", error=str(e))
        return False


def evaluate_command(args):
    """Run evaluation on the current retrieval system."""
    print("Starting evaluation...")
    
    try:
        setup_observability()
        log_system_event("evaluation_start", max_queries=args.max_queries)
        
        start_time = time.time()
        
        # Initialize components
        print("Initializing components...")
        from qdrant_client import QdrantClient
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        embeddings = get_production_embeddings(settings.embed_model_name)
        
        # Build retrieval pipeline
        print("Building retrieval pipeline...")
        retriever = build_retrieval_pipeline(
            qdrant_client=client,
            collection_name=settings.collection_name,
            embeddings=embeddings,
            use_bm25=settings.use_bm25,
            final_k=settings.topk_final
        )
        
        # Validate retriever
        validation = validate_retriever(retriever)
        if validation["status"] != "success":
            print(f"Warning: Retriever validation failed: {validation}")
        else:
            print(f"Retriever validated: {validation['results_count']} results")
        
        # Run evaluation
        print(f"Running evaluation on up to {args.max_queries} queries...")
        results = evaluate_msmarco_retrieval(
            retriever=retriever,
            max_queries=args.max_queries,
            k_list=[5, 10]
        )
        
        total_time = time.time() - start_time
        
        # Print report
        report = format_evaluation_report(results)
        print("\n" + report)
        
        # Save results if output specified
        if args.output:
            save_evaluation_results(results, args.output)
            print(f"Results saved to {args.output}")
        
        log_system_event("evaluation_complete", 
                         duration_seconds=total_time,
                         queries_evaluated=results.get("evaluation_summary", {}).get("total_queries", 0))
        
        return True
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        log_system_event("evaluation_error", error=str(e))
        return False


def serve_command(args):
    """Start the FastAPI server."""
    print("Starting RAG API server...")
    
    try:
        import uvicorn
        from .api import app
        
        setup_observability()
        log_system_event("server_start", 
                         host=args.host, 
                         port=args.port,
                         workers=args.workers)
        
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level=settings.log_level
        )
        
    except Exception as e:
        print(f"Error starting server: {e}")
        log_system_event("server_error", error=str(e))
        return False


def validate_command(args):
    """Validate all system components."""
    print("Validating system components...")
    
    all_valid = True
    
    try:
        # Validate configuration
        print("✓ Configuration loaded successfully")
        
        # Validate embeddings
        print("Validating embeddings...")
        embeddings = get_production_embeddings(settings.embed_model_name)
        validation = validate_embeddings(embeddings)
        if validation["status"] == "success":
            print(f"✓ Embeddings valid: {validation['embedding_dim']} dimensions")
        else:
            print(f"✗ Embeddings validation failed: {validation}")
            all_valid = False
        
        # Validate Qdrant connection
        print("Validating Qdrant connection...")
        from qdrant_client import QdrantClient
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        
        try:
            collections = client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if settings.collection_name in collection_names:
                info = get_collection_info(client, settings.collection_name)
                print(f"✓ Qdrant collection '{settings.collection_name}' found with {info['point_count']} points")
            else:
                print(f"⚠ Qdrant collection '{settings.collection_name}' not found (needs index build)")
                
        except Exception as e:
            print(f"✗ Qdrant connection failed: {e}")
            all_valid = False
        
        # Validate LLM connection
        print("Validating LLM connection...")
        llm = get_production_llm()
        llm_validation = validate_llm_connection(llm)
        if llm_validation["status"] == "success":
            print(f"✓ LLM connection valid")
        else:
            print(f"✗ LLM validation failed: {llm_validation}")
            all_valid = False
        
        # Validate full RAG chain if possible
        if all_valid:
            print("Validating RAG chain...")
            try:
                retriever = build_retrieval_pipeline(
                    qdrant_client=client,
                    collection_name=settings.collection_name,
                    embeddings=embeddings,
                    use_bm25=False,  # Skip BM25 for validation
                    final_k=3
                )
                
                prompt = build_prompt_template()
                chain = build_rag_chain(retriever, llm, prompt)
                
                chain_validation = validate_chain(chain)
                if chain_validation["status"] == "success":
                    print(f"✓ RAG chain validation successful")
                else:
                    print(f"✗ RAG chain validation failed: {chain_validation}")
                    all_valid = False
                    
            except Exception as e:
                print(f"✗ RAG chain validation failed: {e}")
                all_valid = False
        
        print("\nValidation Summary:")
        if all_valid:
            print("✓ All components validated successfully")
            return True
        else:
            print("✗ Some components failed validation")
            return False
            
    except Exception as e:
        print(f"Error during validation: {e}")
        return False


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="RAG MS MARCO CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Build index command
    build_parser = subparsers.add_parser("build-index", help="Build Qdrant index from MS MARCO")
    build_parser.set_defaults(func=build_index_command)
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run retrieval evaluation")
    eval_parser.add_argument("--max-queries", type=int, default=5000, help="Maximum queries to evaluate")
    eval_parser.add_argument("--output", type=str, help="Output file prefix for results")
    eval_parser.set_defaults(func=evaluate_command)
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    serve_parser.add_argument("--workers", type=int, default=settings.uvicorn_workers, help="Number of workers")
    serve_parser.set_defaults(func=serve_command)
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate system components")
    validate_parser.set_defaults(func=validate_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return False
    
    # Run command
    success = args.func(args)
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)