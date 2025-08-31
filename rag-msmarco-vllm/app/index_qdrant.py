"""Qdrant vector store integration and management.

This module provides utilities for creating, managing, and querying Qdrant
collections optimized for RAG applications.
"""

import time
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, PointStruct, VectorParams


def create_or_get_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance: str = "Cosine"
) -> None:
    """Create a Qdrant collection if it doesn't exist, or verify existing one.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the collection to create/verify
        vector_size: Dimension of the embedding vectors
        distance: Distance metric ('Cosine', 'Euclidean', 'Dot')
        
    Raises:
        ValueError: If distance metric is invalid
        Exception: If collection creation fails
    """
    # Validate distance metric
    distance_map = {
        "Cosine": Distance.COSINE,
        "Euclidean": Distance.EUCLID, 
        "Dot": Distance.DOT
    }
    
    if distance not in distance_map:
        raise ValueError(f"Invalid distance metric: {distance}. Must be one of {list(distance_map.keys())}")
    
    try:
        # Check if collection already exists
        collections = client.get_collections()
        existing_names = [col.name for col in collections.collections]
        
        if collection_name in existing_names:
            print(f"Collection '{collection_name}' already exists")
            
            # Verify the configuration matches
            collection_info = client.get_collection(collection_name)
            existing_size = collection_info.config.params.vectors.size
            existing_distance = collection_info.config.params.vectors.distance
            
            if existing_size != vector_size:
                raise ValueError(
                    f"Existing collection has vector size {existing_size}, "
                    f"but requested size is {vector_size}"
                )
            
            print(f"Collection configuration verified: size={vector_size}, distance={existing_distance}")
            return
        
        # Create new collection with optimized HNSW parameters
        print(f"Creating collection '{collection_name}' with vector size {vector_size}")
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance_map[distance]
            ),
            hnsw_config=models.HnswConfigDiff(
                m=32,  # Number of bi-directional links for each node
                ef_construct=100,  # Size of the dynamic candidate list
                full_scan_threshold=10000,  # Use HNSW for collections larger than this
                max_indexing_threads=0,  # Use all available threads
            ),
            optimizers_config=models.OptimizersConfigDiff(
                default_segment_number=2,  # Number of segments per shard
                max_segment_size=None,  # Let Qdrant choose optimal size
                memmap_threshold=None,  # Let Qdrant choose optimal threshold
                indexing_threshold=20000,  # Start indexing after this many points
                flush_interval_sec=5,  # Flush to disk every 5 seconds
            )
        )
        
        print(f"Successfully created collection '{collection_name}'")
        
    except Exception as e:
        print(f"Error creating/verifying collection: {e}")
        raise


def upsert_documents(
    client: QdrantClient,
    collection_name: str,
    docs: List[Document],
    embeddings: Embeddings,
    batch_size: int = 128
) -> Dict[str, Any]:
    """Upsert documents to Qdrant collection with embeddings.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the target collection
        docs: List of Document objects to upsert
        embeddings: Embeddings instance for generating vectors
        batch_size: Number of documents to process per batch
        
    Returns:
        Dictionary with operation statistics
        
    Raises:
        Exception: If upsert operation fails
    """
    if not docs:
        return {"status": "success", "processed": 0, "batches": 0, "duration_seconds": 0}
    
    start_time = time.time()
    total_processed = 0
    batch_count = 0
    
    print(f"Upserting {len(docs)} documents in batches of {batch_size}")
    
    try:
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            batch_count += 1
            
            print(f"Processing batch {batch_count} ({len(batch)} documents)")
            
            # Extract texts for embedding
            texts = [doc.page_content for doc in batch]
            
            # Generate embeddings for this batch
            batch_embeddings = embeddings.embed_documents(texts)
            
            # Prepare points for Qdrant
            points = []
            for j, (doc, embedding) in enumerate(zip(batch, batch_embeddings)):
                # Use chunk_id as the point ID, fallback to doc_id + index
                point_id = doc.metadata.get("chunk_id")
                if not point_id:
                    doc_id = doc.metadata.get("doc_id", f"doc_{i + j}")
                    point_id = f"{doc_id}:chunk:{j}"
                
                # Prepare payload with metadata and text
                payload = {
                    "page_content": doc.page_content,
                    **doc.metadata
                }
                
                # Create point
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
                points.append(point)
            
            # Upsert batch to Qdrant
            client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True  # Wait for the operation to complete
            )
            
            total_processed += len(batch)
            print(f"Completed batch {batch_count}, total processed: {total_processed}")
    
    except Exception as e:
        print(f"Error during upsert operation: {e}")
        raise
    
    duration = time.time() - start_time
    
    result = {
        "status": "success",
        "processed": total_processed,
        "batches": batch_count,
        "duration_seconds": duration,
        "rate_docs_per_second": total_processed / duration if duration > 0 else 0
    }
    
    print(f"Upsert completed: {result}")
    return result


def as_retriever(
    client: QdrantClient,
    collection_name: str,
    embeddings: Embeddings,
    k: int = 5,
    score_threshold: Optional[float] = None
) -> BaseRetriever:
    """Create a LangChain retriever from Qdrant collection.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the collection
        embeddings: Embeddings instance for query encoding
        k: Number of documents to retrieve
        score_threshold: Minimum similarity score threshold
        
    Returns:
        Configured BaseRetriever instance
    """
    # Create Qdrant vector store wrapper
    vector_store = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
        content_payload_key="page_content"
    )
    
    # Configure search parameters
    search_kwargs = {"k": k}
    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold
    
    # Return as retriever
    return vector_store.as_retriever(search_kwargs=search_kwargs)


def get_collection_info(client: QdrantClient, collection_name: str) -> Dict[str, Any]:
    """Get detailed information about a Qdrant collection.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the collection
        
    Returns:
        Dictionary with collection information
        
    Raises:
        Exception: If collection doesn't exist or query fails
    """
    try:
        # Get collection info
        collection_info = client.get_collection(collection_name)
        
        # Count points in collection
        count_result = client.count(collection_name)
        point_count = count_result.count
        
        # Get some sample points to understand the structure
        sample_points = client.scroll(
            collection_name=collection_name,
            limit=3,
            with_payload=True,
            with_vectors=False
        )[0]  # scroll returns (points, next_page_offset)
        
        return {
            "name": collection_name,
            "status": collection_info.status,
            "vector_size": collection_info.config.params.vectors.size,
            "distance_metric": collection_info.config.params.vectors.distance,
            "point_count": point_count,
            "indexed": collection_info.status == models.CollectionStatus.GREEN,
            "sample_payloads": [
                {k: str(v)[:100] + "..." if len(str(v)) > 100 else v 
                 for k, v in point.payload.items()}
                for point in sample_points[:2]
            ] if sample_points else []
        }
        
    except Exception as e:
        print(f"Error getting collection info: {e}")
        raise


def delete_collection(client: QdrantClient, collection_name: str) -> bool:
    """Delete a Qdrant collection.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the collection to delete
        
    Returns:
        True if deletion successful, False if collection didn't exist
        
    Raises:
        Exception: If deletion fails for other reasons
    """
    try:
        client.delete_collection(collection_name)
        print(f"Successfully deleted collection '{collection_name}'")
        return True
        
    except Exception as e:
        if "not found" in str(e).lower():
            print(f"Collection '{collection_name}' not found (already deleted?)")
            return False
        else:
            print(f"Error deleting collection: {e}")
            raise