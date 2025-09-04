"""Text preprocessing utilities.

This module provides text normalization and filtering functions
to prepare text for embedding and retrieval.
"""

import re
import unicodedata
from typing import Dict, Any


def normalize_text(text: str) -> str:
    """Normalize text using NFC normalization and cleanup.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text string
    """
    if not isinstance(text, str):
        return ""
    
    # NFC Unicode normalization
    text = unicodedata.normalize('NFC', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Collapse multiple whitespace characters to single spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text


def filter_record(record: Dict[str, Any]) -> bool:
    """Filter passage records based on quality criteria.
    
    Args:
        record: Passage record dictionary
        
    Returns:
        True if record should be kept, False if it should be filtered out
    """
    text = record.get("text", "")
    
    # Must have text
    if not text or not isinstance(text, str):
        return False
    
    # Length constraints
    min_length = 10  # Minimum 10 characters
    max_length = 10000  # Maximum 10k characters to avoid very long passages
    
    text_length = len(text.strip())
    if text_length < min_length or text_length > max_length:
        return False
    
    # Must contain some alphanumeric content
    if not re.search(r'[a-zA-Z0-9]', text):
        return False
    
    # Filter out passages that are mostly punctuation or whitespace
    alphanumeric_chars = len(re.findall(r'[a-zA-Z0-9]', text))
    if alphanumeric_chars / text_length < 0.3:  # At least 30% alphanumeric
        return False
    
    return True


def clean_for_embedding(text: str) -> str:
    """Clean text specifically for embedding generation.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text suitable for embedding
    """
    # Normalize first
    text = normalize_text(text)
    
    # Remove or replace common problematic characters
    # Replace em dashes and en dashes with regular hyphens
    text = re.sub(r'[—–]', '-', text)
    
    # Replace smart quotes with regular quotes
    text = re.sub(r'[""''`]', '"', text)
    
    # Remove or replace other Unicode punctuation that might cause issues
    text = re.sub(r'[…]', '...', text)
    
    # Normalize multiple consecutive punctuation
    text = re.sub(r'[.]{4,}', '...', text)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    
    return text


def extract_title_and_content(text: str) -> tuple[str, str]:
    """Extract title and content from a passage if possible.
    
    Attempts to identify if the passage starts with a title-like string.
    
    Args:
        text: Input passage text
        
    Returns:
        Tuple of (title, content) where title may be empty
    """
    text = text.strip()
    lines = text.split('\n', 1)
    
    if len(lines) == 1:
        # Single line - check if it looks like a title
        if len(text) < 100 and not text.endswith('.'):
            return text, ""
        else:
            return "", text
    
    first_line = lines[0].strip()
    rest = lines[1].strip()
    
    # Heuristics for title detection
    # Title is likely if:
    # - Short (< 100 chars)
    # - Doesn't end with sentence punctuation
    # - Is followed by more content
    
    if (len(first_line) < 100 and 
        not first_line.endswith(('.', '!', '?')) and 
        len(rest) > 20):
        return first_line, rest
    else:
        return "", text


def prepare_for_chunking(record: Dict[str, Any]) -> str:
    """Prepare a passage record for text chunking.
    
    Args:
        record: Passage record with text and metadata
        
    Returns:
        Prepared text string ready for chunking
    """
    text = record.get("text", "")
    url = record.get("url", "")
    
    # Clean the text
    text = clean_for_embedding(text)
    
    # Optionally prepend URL as context if available and meaningful
    if url and url.startswith(('http://', 'https://')) and len(url) < 200:
        # Extract domain for context
        domain_match = re.search(r'https?://([^/]+)', url)
        if domain_match:
            domain = domain_match.group(1)
            text = f"Source: {domain}\n\n{text}"
    
    return text