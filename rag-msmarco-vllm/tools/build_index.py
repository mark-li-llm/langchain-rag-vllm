#!/usr/bin/env python3
"""Standalone script for building the MS MARCO index.

This script can be run independently to build or rebuild the Qdrant index
from MS MARCO data without starting the full API server.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.chunking import get_chunking_stats, make_splitter, split_documents_batch
from app.config import settings
from app.dataset_msmarco import get_dataset_stats, load_msmarco
from app.embeddings import get_production_embeddings, validate_embeddings
from app.index_qdrant import (
    create_or_get_collection,
    delete_collection,
    get_collection_info,
    upsert_documents,
)
from app.observability import log_system_event, setup_observability
from app.preprocess import filter_record, prepare_for_chunking


def main():
    """Main function for index building."""
    parser = argparse.ArgumentParser(description="Build MS MARCO Qdrant index")
    
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=settings.corpus_sample_size,
        help="Number of corpus passages to sample"
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
        "--chunk-size",
        type=int,
        default=settings.chunk_size_chars,
        help="Chunk size in characters"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.chunk_overlap_chars,
        help="Chunk overlap in characters"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete existing collection and recreate"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process data but don't upload to Qdrant"
    )
    parser.add_argument(
        "--output-manifest",
        type=str,
        help="Path to save build manifest JSON"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    print("MS MARCO Index Builder")
    print("=" * 50)
    print(f"Sample size: {args.sample_size}")
    print(f"Collection: {args.collection_name}")
    print(f"Embedding model: {args.embed_model}")
    print(f"Chunk size: {args.chunk_size} chars")
    print(f"Chunk overlap: {args.chunk_overlap} chars")
    print(f"Dry run: {args.dry_run}")
    print("")
    
    try:
        # Setup observability
        if not args.dry_run:
            setup_observability()
        
        import time
        start_time = time.time()
        
        # Step 1: Load MS MARCO dataset
        print("1. Loading MS MARCO dataset...")
        corpus_passages, queries = load_msmarco(
            corpus_split=settings.hf_split_corpus,
            queries_split=settings.hf_split_queries,
            config=settings.hf_dataset_config,
            max_corpus_passages=args.sample_size
        )
        
        # Print dataset statistics
        stats = get_dataset_stats(corpus_passages, queries)
        print(f"   Dataset statistics:")
        print(f"   - Corpus passages: {stats['corpus']['total_passages']}")
        print(f"   - Selected passages: {stats['corpus']['selected_passages']}")
        print(f"   - Selection rate: {stats['corpus']['selection_rate']:.3f}")
        print(f"   - Avg passage length: {stats['corpus']['avg_passage_length']:.1f} chars")
        print(f"   - Queries: {stats['queries']['total_queries']}")
        print("")
        
        # Step 2: Filter and preprocess passages
        print("2. Filtering and preprocessing passages...")
        filtered_passages = []
        skipped_count = 0
        
        for i, passage in enumerate(corpus_passages):
            if args.verbose and i % 10000 == 0:
                print(f"   Processed {i}/{len(corpus_passages)} passages")
            
            if filter_record(passage):
                # Prepare text for chunking
                passage["text"] = prepare_for_chunking(passage)
                filtered_passages.append(passage)
            else:
                skipped_count += 1
        
        print(f"   Kept: {len(filtered_passages)}/{len(corpus_passages)} passages")
        print(f"   Filtered out: {skipped_count} passages")
        print("")
        
        if not filtered_passages:
            print("Error: No valid passages after filtering")
            return False
        
        # Convert filtered passages (dicts) to LangChain Document objects
        from langchain_core.documents import Document
        langchain_docs = [
            Document(page_content=p.get("text", ""), metadata=p) for p in filtered_passages
        ]

        # Step 3: Split documents into chunks
        print("3. Splitting documents into chunks...")
        splitter = make_splitter(
            chunk_size_chars=args.chunk_size,
            chunk_overlap_chars=args.chunk_overlap
        )
        
        documents = split_documents_batch(langchain_docs, splitter)
        
        chunking_stats = get_chunking_stats(documents)
        print(f"   Chunking statistics:")
        print(f"   - Total chunks: {chunking_stats['total_chunks']}")
        print(f"   - Source documents: {chunking_stats['source_documents']}")
        print(f"   - Avg chunks per doc: {chunking_stats['avg_chunks_per_doc']:.2f}")
        print(f"   - Avg chunk length: {chunking_stats['avg_chunk_length']:.1f} chars")
        print(f"   - Total characters: {chunking_stats['total_characters']:,}")
        print("")
        
        if not documents:
            print("Error: No chunks created")
            return False
        
        # Step 4: Initialize embeddings
        print("4. Initializing embeddings...")
        embeddings = get_production_embeddings(args.embed_model)
        
        # Validate embeddings
        validation = validate_embeddings(embeddings)
        if validation["status"] != "success":
            print(f"   Error: Embeddings validation failed: {validation}")
            return False
        
        print(f"   Embeddings validated:")
        print(f"   - Model: {args.embed_model}")
        print(f"   - Dimensions: {validation['embedding_dim']}")
        print(f"   - Type: {validation['single_embedding_type']}")
        print("")
        
        if args.dry_run:
            print("Dry run completed - no Qdrant operations performed")
            
            # Save manifest if requested
            if args.output_manifest:
                manifest = create_build_manifest(
                    args, stats, chunking_stats, validation, 
                    len(corpus_passages), len(filtered_passages), len(documents)
                )
                with open(args.output_manifest, 'w') as f:
                    json.dump(manifest, f, indent=2)
                print(f"Manifest saved to {args.output_manifest}")
            
            return True
        
        # Step 5: Connect to Qdrant and setup collection
        print("5. Setting up Qdrant collection...")
        from qdrant_client import QdrantClient
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        
        # Delete existing collection if recreate flag is set
        if args.recreate:
            print(f"   Deleting existing collection '{args.collection_name}'...")
            try:
                delete_collection(client, args.collection_name)
            except Exception as e:
                print(f"   Warning: Could not delete collection: {e}")
        
        # Create or verify collection
        vector_size = validation["embedding_dim"]
        create_or_get_collection(client, args.collection_name, vector_size)
        print(f"   Collection '{args.collection_name}' ready")
        print("")
        
        # Step 6: Upsert documents
        print("6. Upserting documents to Qdrant...")
        upsert_result = upsert_documents(
            client, args.collection_name, documents, embeddings, batch_size=128
        )
        
        print(f"   Upsert completed:")
        print(f"   - Documents processed: {upsert_result['processed']}")
        print(f"   - Batches: {upsert_result['batches']}")
        print(f"   - Duration: {upsert_result['duration_seconds']:.2f}s")
        print(f"   - Rate: {upsert_result['rate_docs_per_second']:.2f} docs/sec")
        print("")
        
        # Step 7: Verify final collection state
        print("7. Verifying collection...")
        collection_info = get_collection_info(client, args.collection_name)
        print(f"   Collection info:")
        print(f"   - Name: {collection_info['name']}")
        print(f"   - Status: {collection_info['status']}")
        print(f"   - Point count: {collection_info['point_count']:,}")
        print(f"   - Vector size: {collection_info['vector_size']}")
        print(f"   - Distance metric: {collection_info['distance_metric']}")
        print("")
        
        # Final summary
        total_time = time.time() - start_time
        print("Build completed successfully!")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Final collection: {collection_info['point_count']:,} points")
        
        # Save manifest if requested
        if args.output_manifest:
            manifest = create_build_manifest(
                args, stats, chunking_stats, validation,
                len(corpus_passages), len(filtered_passages), len(documents),
                upsert_result, collection_info, total_time
            )
            with open(args.output_manifest, 'w') as f:
                json.dump(manifest, f, indent=2)
            print(f"Build manifest saved to {args.output_manifest}")
        
        # Log completion
        log_system_event("index_build_complete_cli",
                         duration_seconds=total_time,
                         documents_processed=upsert_result['processed'],
                         total_points=collection_info['point_count'])
        
        return True
        
    except Exception as e:
        print(f"Error during index build: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_build_manifest(args, dataset_stats, chunking_stats, embedding_validation, 
                         total_passages, filtered_passages, total_chunks,
                         upsert_result=None, collection_info=None, total_time=None):
    """Create build manifest with all metadata."""
    manifest = {
        "build_info": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sample_size": args.sample_size,
            "collection_name": args.collection_name,
            "embed_model": args.embed_model,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "recreated": args.recreate,
            "total_time_seconds": total_time
        },
        "dataset_stats": dataset_stats,
        "processing": {
            "total_passages_loaded": total_passages,
            "passages_after_filtering": filtered_passages,
            "chunks_created": total_chunks,
            "filter_rate": filtered_passages / total_passages if total_passages else 0
        },
        "chunking_stats": chunking_stats,
        "embedding_validation": embedding_validation
    }
    
    if upsert_result:
        manifest["upsert_result"] = upsert_result
        
    if collection_info:
        manifest["collection_info"] = collection_info
    
    return manifest


if __name__ == "__main__":
    import time
    success = main()
    sys.exit(0 if success else 1)