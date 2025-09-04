"""ID generation utilities.

This module provides utilities for generating unique identifiers
for tracing and tracking requests.
"""

import uuid
import time
from typing import Optional


def generate_trace_id() -> str:
    """Generate a unique trace ID for request tracking.
    
    Uses a combination of timestamp and UUID for uniqueness
    and rough temporal ordering.
    
    Returns:
        Unique trace ID string
    """
    # Use timestamp for rough ordering + UUID for uniqueness
    timestamp = int(time.time() * 1000)  # milliseconds
    unique_part = str(uuid.uuid4()).replace('-', '')[:12]
    return f"{timestamp}_{unique_part}"


def generate_ulid() -> str:
    """Generate a ULID (Universally Unique Lexicographically Sortable Identifier).
    
    This is a simplified ULID implementation that provides lexicographic ordering
    and uniqueness. For production use, consider using a proper ULID library.
    
    Returns:
        ULID string
    """
    try:
        # Try using python-ulid if available
        from ulid import ULID
        return str(ULID())
    except ImportError:
        # Fallback to timestamp + UUID
        timestamp = int(time.time() * 1000)
        unique_part = str(uuid.uuid4()).replace('-', '')[:16]
        # Format as ULID-like string (26 characters)
        return f"{timestamp:013x}{unique_part}".upper()[:26]


def generate_request_id() -> str:
    """Generate a request ID for API requests.
    
    Returns:
        Unique request ID
    """
    return str(uuid.uuid4())


def generate_session_id() -> str:
    """Generate a session ID for user sessions.
    
    Returns:
        Unique session ID
    """
    return str(uuid.uuid4())


def extract_timestamp_from_trace_id(trace_id: str) -> Optional[int]:
    """Extract timestamp from trace ID if possible.
    
    Args:
        trace_id: Trace ID to parse
        
    Returns:
        Timestamp in milliseconds, or None if not extractable
    """
    try:
        if '_' in trace_id:
            timestamp_part = trace_id.split('_')[0]
            return int(timestamp_part)
        return None
    except (ValueError, IndexError):
        return None


def is_valid_trace_id(trace_id: str) -> bool:
    """Validate trace ID format.
    
    Args:
        trace_id: Trace ID to validate
        
    Returns:
        True if trace ID appears valid
    """
    if not trace_id or not isinstance(trace_id, str):
        return False
    
    # Check basic format requirements
    if len(trace_id) < 10:
        return False
    
    # If it contains underscore, check timestamp part
    if '_' in trace_id:
        parts = trace_id.split('_')
        if len(parts) != 2:
            return False
        
        timestamp_part, unique_part = parts
        
        # Timestamp should be numeric and reasonable
        try:
            timestamp = int(timestamp_part)
            # Should be within reasonable range (after 2020, before year 3000)
            if not (1577836800000 <= timestamp <= 32503680000000):  
                return False
        except ValueError:
            return False
        
        # Unique part should be alphanumeric
        if not unique_part.replace('-', '').isalnum():
            return False
    
    return True


def validate_uuid(uuid_string: str) -> bool:
    """Validate UUID format.
    
    Args:
        uuid_string: UUID string to validate
        
    Returns:
        True if valid UUID format
    """
    try:
        uuid.UUID(uuid_string)
        return True
    except (ValueError, TypeError):
        return False