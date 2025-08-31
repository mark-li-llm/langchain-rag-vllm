"""Prompting templates and citation formatting for RAG.

This module provides prompt templates optimized for retrieval-augmented generation
with proper citation formatting and context management.
"""

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


def build_prompt_template(max_context_tokens: int = 3000) -> ChatPromptTemplate:
    """Build chat prompt template for RAG with citation support.
    
    The template enforces that the model only uses information from the provided
    context and formats citations as numbered references.
    
    Args:
        max_context_tokens: Approximate token budget for context (rough estimate)
        
    Returns:
        Configured ChatPromptTemplate
    """
    system_message = """You are a helpful assistant that answers questions using ONLY the information provided in the context below. 

IMPORTANT INSTRUCTIONS:
1. Answer ONLY with information from the provided context
2. If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided context to answer this question."
3. Include inline citations using [n] format where n is the numbered context chunk
4. Be concise but comprehensive in your answer
5. Do not add information from your general knowledge
6. If you're uncertain about something in the context, acknowledge the uncertainty

Format your citations like this: "According to the research [1], machine learning algorithms [2] can improve performance."
"""

    human_message = """Question: {query}

[Context]
{context}

Please answer the question using only the information from the context above, including appropriate citations [n] for each piece of information you reference."""

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_message),
        HumanMessagePromptTemplate.from_template(human_message)
    ])
    
    return prompt


def format_context_with_citations(
    docs: List[Document], 
    max_context_tokens: int = 3000,
    chars_per_token: float = 4.0
) -> tuple[str, List[Dict[str, Any]]]:
    """Format retrieved documents as numbered context with citation mapping.
    
    Args:
        docs: List of retrieved Document objects
        max_context_tokens: Maximum tokens to use for context
        chars_per_token: Rough estimate of characters per token
        
    Returns:
        Tuple of (formatted_context_string, citations_list)
    """
    if not docs:
        return "No relevant context found.", []
    
    max_context_chars = int(max_context_tokens * chars_per_token)
    
    formatted_chunks = []
    citations = []
    total_chars = 0
    
    for i, doc in enumerate(docs, 1):
        # Extract metadata for citation
        doc_id = doc.metadata.get("doc_id") or doc.metadata.get("metadata", {}).get("doc_id", f"doc_{i}")
        chunk_id = doc.metadata.get("chunk_id") or doc.metadata.get("metadata", {}).get("chunk_id", f"{doc_id}:chunk:{i}")
        source = doc.metadata.get("source", "")
        url = doc.metadata.get("url", "")
        
        # Create citation entry
        citation = {
            "number": i,
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "source": source,
            "url": url if url else None,
            "score": doc.metadata.get("score", None)
        }
        citations.append(citation)
        
        # Format document content
        content = doc.page_content.strip()
        
        # Truncate if too long
        max_chunk_chars = min(len(content), max_context_chars - total_chars - 100)  # Reserve space
        if max_chunk_chars <= 0:
            break
            
        if len(content) > max_chunk_chars:
            content = content[:max_chunk_chars] + "..."
        
        # Create formatted chunk with header
        header = f"[{i}]"
        if url:
            # Extract domain from URL for brief context
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc
                if domain:
                    header += f" Source: {domain}"
            except:
                pass
        elif source:
            header += f" Source: {source}"
        
        chunk_text = f"{header}\n{content}\n"
        formatted_chunks.append(chunk_text)
        
        total_chars += len(chunk_text)
        
        # Stop if we're approaching token limit
        if total_chars >= max_context_chars:
            break
    
    # Join all chunks
    context = "\n".join(formatted_chunks)
    
    return context, citations


def format_citations(docs: List[Document]) -> List[Dict[str, Any]]:
    """Format citations list from retrieved documents.
    
    Args:
        docs: List of retrieved Document objects
        
    Returns:
        List of citation dictionaries
    """
    citations = []
    
    for i, doc in enumerate(docs, 1):
        citation = {
            "number": i,
            "doc_id": doc.metadata.get("doc_id", f"doc_{i}"),
            "chunk_id": doc.metadata.get("chunk_id", f"chunk_{i}"),
            "source": doc.metadata.get("source", ""),
            "url": doc.metadata.get("url") if doc.metadata.get("url") else None,
            "score": doc.metadata.get("score"),
            "preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        }
        citations.append(citation)
    
    return citations


def validate_citations_in_response(response_text: str, expected_citations: int) -> Dict[str, Any]:
    """Validate that response contains proper citations.
    
    Args:
        response_text: Generated response text
        expected_citations: Number of context chunks provided
        
    Returns:
        Dictionary with citation validation results
    """
    import re
    
    # Find all citation patterns [n]
    citation_pattern = r'\[(\d+)\]'
    found_citations = re.findall(citation_pattern, response_text)
    
    # Convert to integers and get unique citations
    try:
        cited_numbers = list(set(int(n) for n in found_citations))
        cited_numbers.sort()
    except ValueError:
        cited_numbers = []
    
    # Check for issues
    missing_citations = []
    invalid_citations = []
    
    for i in range(1, expected_citations + 1):
        if i not in cited_numbers:
            missing_citations.append(i)
    
    for num in cited_numbers:
        if num < 1 or num > expected_citations:
            invalid_citations.append(num)
    
    return {
        "total_citations_found": len(found_citations),
        "unique_citations": len(cited_numbers),
        "cited_numbers": cited_numbers,
        "expected_citations": expected_citations,
        "missing_citations": missing_citations,
        "invalid_citations": invalid_citations,
        "citation_coverage": len(cited_numbers) / expected_citations if expected_citations > 0 else 0,
        "has_citations": len(cited_numbers) > 0,
        "all_citations_valid": len(invalid_citations) == 0
    }


def create_no_context_response() -> str:
    """Create standard response when no context is available.
    
    Returns:
        Standard no-context response message
    """
    return "I don't have any relevant context to answer this question. Please try rephrasing your question or providing more specific details."


def create_insufficient_context_response(query: str) -> str:
    """Create response when context is insufficient to answer question.
    
    Args:
        query: Original user query
        
    Returns:
        Appropriate insufficient context response
    """
    return f"I don't have enough information in the provided context to fully answer your question about '{query}'. The available context may be incomplete or not directly relevant to your specific question."


def extract_query_intent(query: str) -> Dict[str, Any]:
    """Extract intent and key terms from user query for better retrieval.
    
    Args:
        query: User query string
        
    Returns:
        Dictionary with query analysis
    """
    import re
    
    # Simple intent classification based on question words
    question_words = {
        'what': 'definition',
        'how': 'process',
        'why': 'reason',
        'when': 'time',
        'where': 'location',
        'who': 'person',
        'which': 'selection'
    }
    
    query_lower = query.lower()
    detected_intent = 'general'
    
    for word, intent in question_words.items():
        if query_lower.startswith(word):
            detected_intent = intent
            break
    
    # Extract potential key terms (simple approach)
    # Remove common words and extract meaningful terms
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
    
    words = re.findall(r'\b\w+\b', query_lower)
    key_terms = [word for word in words if len(word) > 2 and word not in stop_words]
    
    return {
        "intent": detected_intent,
        "key_terms": key_terms[:10],  # Limit to top 10 terms
        "query_length": len(query),
        "word_count": len(words),
        "has_question_word": detected_intent != 'general'
    }