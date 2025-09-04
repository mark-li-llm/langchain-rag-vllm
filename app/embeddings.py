"""Embedding model configuration and utilities.

This module provides HuggingFace embeddings integration optimized for CPU usage
to avoid GPU contention with vLLM servers.
"""

from typing import Any, Dict, Optional

from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings(
    model_name: str,
    model_kwargs: Optional[Dict[str, Any]] = None,
    encode_kwargs: Optional[Dict[str, Any]] = None,
    cache_folder: Optional[str] = None
) -> HuggingFaceEmbeddings:
    """Create HuggingFaceEmbeddings instance optimized for production use.
    
    This function creates embeddings configured for CPU usage to avoid GPU
    contention with vLLM servers. The embeddings are optimized for retrieval
    tasks with appropriate normalization and batching.
    
    Args:
        model_name: Name of the HuggingFace model (e.g., 'BAAI/bge-small-en-v1.5')
        model_kwargs: Additional arguments for the model
        encode_kwargs: Additional arguments for encoding
        cache_folder: Directory to cache downloaded models
        
    Returns:
        Configured HuggingFaceEmbeddings instance
        
    Raises:
        ValueError: If model_name is invalid
    """
    if not model_name or not isinstance(model_name, str):
        raise ValueError("model_name must be a non-empty string")
    
    # Default model kwargs optimized for CPU inference
    default_model_kwargs = {
        'device': 'cpu',  # Use CPU to avoid GPU contention with vLLM
        'trust_remote_code': False,  # Security: don't execute remote code
    }
    
    # Default encode kwargs for optimal retrieval performance
    default_encode_kwargs = {
        'normalize_embeddings': True,  # Normalize for cosine similarity
        'batch_size': 32,  # Reasonable batch size for CPU
    }
    
    # Merge with user-provided kwargs
    final_model_kwargs = {**default_model_kwargs, **(model_kwargs or {})}
    final_encode_kwargs = {**default_encode_kwargs, **(encode_kwargs or {})}
    
    # Add cache folder if specified
    if cache_folder:
        final_model_kwargs['cache_folder'] = cache_folder
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=final_model_kwargs,
        encode_kwargs=final_encode_kwargs
    )


def get_production_embeddings(model_name: str = "BAAI/bge-small-en-v1.5") -> HuggingFaceEmbeddings:
    """Get embeddings configured for production RAG use.
    
    Uses BGE-small-en-v1.5 as default, which provides good quality embeddings
    with reasonable computational requirements for CPU inference.
    
    Args:
        model_name: Model to use (default: BAAI/bge-small-en-v1.5)
        
    Returns:
        Production-configured HuggingFaceEmbeddings
    """
    return get_embeddings(
        model_name=model_name,
        model_kwargs={
            'device': 'cpu',
            'trust_remote_code': False,
        },
        encode_kwargs={
            'normalize_embeddings': True,
            'batch_size': 64,  # Larger batch for better throughput
        }
    )


def validate_embeddings(embeddings: HuggingFaceEmbeddings, test_text: str = "Hello world") -> dict:
    """Validate that embeddings are working correctly.
    
    Args:
        embeddings: HuggingFaceEmbeddings instance to test
        test_text: Text to use for testing
        
    Returns:
        Dictionary with validation results
        
    Raises:
        Exception: If embeddings fail validation
    """
    try:
        # Test single embedding
        embedding = embeddings.embed_query(test_text)
        
        # Test batch embedding
        batch_embeddings = embeddings.embed_documents([test_text, "Another test"])
        
        # Validate dimensions match
        if len(embedding) != len(batch_embeddings[0]):
            raise ValueError("Embedding dimensions don't match between single and batch")
        
        return {
            "status": "success",
            "embedding_dim": len(embedding),
            "single_embedding_type": type(embedding).__name__,
            "batch_embedding_type": type(batch_embeddings).__name__,
            "test_text": test_text,
            "embedding_preview": embedding[:5] if len(embedding) >= 5 else embedding
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }


def get_embedding_model_info(model_name: str) -> dict:
    """Get information about an embedding model.
    
    Args:
        model_name: Name of the HuggingFace model
        
    Returns:
        Dictionary with model information
    """
    # Common model information (this could be extended to query HuggingFace Hub)
    model_info = {
        "BAAI/bge-small-en-v1.5": {
            "dimension": 384,
            "max_sequence_length": 512,
            "description": "Compact English embedding model, good for retrieval",
            "recommended_use": "General purpose retrieval, CPU-friendly"
        },
        "BAAI/bge-base-en-v1.5": {
            "dimension": 768,
            "max_sequence_length": 512,
            "description": "Balanced English embedding model",
            "recommended_use": "Higher quality retrieval, more compute required"
        },
        "BAAI/bge-large-en-v1.5": {
            "dimension": 1024,
            "max_sequence_length": 512,
            "description": "High-quality English embedding model",
            "recommended_use": "Best quality retrieval, significant compute required"
        },
        "sentence-transformers/all-MiniLM-L6-v2": {
            "dimension": 384,
            "max_sequence_length": 256,
            "description": "Lightweight sentence transformer",
            "recommended_use": "Fast inference, lower quality"
        }
    }
    
    return model_info.get(model_name, {
        "dimension": "unknown",
        "max_sequence_length": "unknown",
        "description": f"Model: {model_name}",
        "recommended_use": "Check model documentation"
    })