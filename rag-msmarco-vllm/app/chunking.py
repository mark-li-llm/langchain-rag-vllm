"""Text chunking utilities using LangChain's RecursiveCharacterTextSplitter.

This module provides text splitting functionality optimized for RAG applications,
preserving document boundaries and creating meaningful chunk sizes.
"""

import uuid
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def make_splitter(
    chunk_size_chars: int = 500,
    chunk_overlap_chars: int = 50
) -> RecursiveCharacterTextSplitter:
    """Create a text splitter for document chunking."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars,
        chunk_overlap=chunk_overlap_chars,
        length_function=len,
        is_separator_regex=False,
    )


def split_documents_batch(
    docs: Iterable[Document], 
    text_splitter: RecursiveCharacterTextSplitter
) -> List[Document]:
    """Split a batch of documents into chunks with unique UUID4 identifiers."""
    all_chunks = []
    
    for i, doc in enumerate(docs):
        # Split this document into chunks
        sub_chunks = text_splitter.split_documents([doc])
        for j, chunk in enumerate(sub_chunks):
            # Create a unique UUID4 for the chunk
            chunk_uuid = str(uuid.uuid4())

            chunk.metadata["doc_id"] = doc.metadata.get("doc_id", f"doc_{i}")
            chunk.metadata["chunk_id"] = chunk_uuid
            chunk.metadata["chunk_id_str"] = chunk_uuid
            chunk.metadata["chunk_index"] = j
            chunk.metadata["total_chunks"] = len(sub_chunks)
            chunk.metadata["chunk_char_count"] = len(chunk.page_content)
            
            # Carry over original metadata
            for key, value in doc.metadata.items():
                if key not in chunk.metadata:
                    chunk.metadata[key] = value
                    
            all_chunks.append(chunk)
            
    return all_chunks


def get_chunking_stats(chunks: list[Document]) -> dict:
    """Calculate statistics for a list of document chunks."""
    if not chunks:
        return {
            "total_chunks": 0,
            "source_documents": 0,
            "avg_chunks_per_doc": 0,
            "avg_chunk_length": 0,
            "total_characters": 0
        }
    
    source_doc_ids = {chunk.metadata.get("doc_id") for chunk in chunks}
    char_counts = [len(chunk.page_content) for chunk in chunks]
    
    return {
        "total_chunks": len(chunks),
        "source_documents": len(source_doc_ids),
        "avg_chunks_per_doc": len(chunks) / len(source_doc_ids) if source_doc_ids else 0,
        "avg_chunk_length": sum(char_counts) / len(char_counts) if char_counts else 0,
        "total_characters": sum(char_counts)
    }