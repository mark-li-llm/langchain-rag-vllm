"""MS MARCO dataset loading and normalization.

This module handles loading MS MARCO v2.1 dataset from Hugging Face,
normalizing passages and queries, and providing iterators for processing.
"""

import random
from typing import Iterator, Optional

from datasets import load_dataset


def load_msmarco(
    corpus_split: str,
    queries_split: str,
    config: str = "v2.1",
    max_corpus_passages: Optional[int] = None,
    seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Load MS MARCO corpus and queries from Hugging Face Datasets.
    
    Args:
        corpus_split: Split name for corpus (e.g., 'train')
        queries_split: Split name for queries (e.g., 'validation')
        config: Dataset configuration (default: 'v2.1')
        max_corpus_passages: Maximum number of passages to load (uses reservoir sampling)
        seed: Random seed for sampling
        
    Returns:
        Tuple of (corpus_passages, queries) as lists of normalized dictionaries
        
    Raises:
        ValueError: If dataset fields are missing or invalid
    """
    print(f"Loading MS MARCO {config} dataset...")
    
    # Load the full dataset splits
    corpus_dataset = load_dataset("microsoft/ms_marco", config, split=corpus_split)
    print(f"Loaded {len(corpus_dataset)} examples from corpus split '{corpus_split}'")
    
    queries_dataset = load_dataset("microsoft/ms_marco", config, split=queries_split)
    print(f"Loaded {len(queries_dataset)} queries from queries split '{queries_split}'")

    # Process corpus passages by iterating through the dataset
    corpus_passages = list(iter_corpus_passages(corpus_dataset))
    print(f"Extracted {len(corpus_passages)} total passages from corpus")

    # Apply sampling if requested
    if max_corpus_passages is not None and len(corpus_passages) > max_corpus_passages:
        print(f"Sampling {max_corpus_passages} passages...")
        random.seed(seed)
        corpus_passages = random.sample(corpus_passages, max_corpus_passages)
    
    # Process queries
    queries = []
    for example in queries_dataset:
        if "query_id" not in example or "query" not in example:
            continue
            
        query_record = {
            "query_id": str(example["query_id"]),
            "query_text": example["query"],
            "answer_texts": example.get("answers", []) if "answers" in example else []
        }
        queries.append(query_record)
    
    print(f"Processed {len(queries)} queries")
    return corpus_passages, queries


def iter_corpus_passages(dataset_split) -> Iterator[dict]:
    """
    Iterate over all passages in a dataset split.
    MS MARCO v2.1 train/validation sets have a nested structure:
    Each example contains a query and a 'passages' dict with lists of texts, urls, etc.
    """
    passage_count = 0
    for example in dataset_split:
        query_id = str(example.get("query_id", ""))
        passages_dict = example.get("passages")

        if not isinstance(passages_dict, dict):
            continue

        passage_texts = passages_dict.get("passage_text", [])
        urls = passages_dict.get("url", [])
        is_selected = passages_dict.get("is_selected", [])

        num_passages = len(passage_texts)
        if num_passages == 0:
            continue

        # Ensure other lists are of the same length, pad with defaults if not
        urls.extend([""] * (num_passages - len(urls)))
        is_selected.extend([0] * (num_passages - len(is_selected)))

        for i in range(num_passages):
            passage_text = passage_texts[i]
            if not passage_text or not isinstance(passage_text, str):
                continue
            
            passage_record = {
                "doc_id": f"msmarco_{query_id}_{i}",
                "text": passage_text.strip(),
                "url": urls[i],
                "label_is_selected": int(is_selected[i]),
                "source": f"msmarco:v2.1:{query_id}:{i}",
                "query_id": query_id,
                "passage_idx": i
            }
            yield passage_record
            passage_count += 1


def validate_dataset_fields(example: dict) -> bool:
    """Validate that a dataset example has required MS MARCO fields.
    
    Args:
        example: Dataset example dictionary
        
    Returns:
        True if example has required fields, False otherwise
    """
    required_fields = ["query_id", "query", "passages"]
    
    # Check top-level fields
    for field in required_fields:
        if field not in example:
            return False
    
    # Check passages structure
    passages = example.get("passages", [])
    if not isinstance(passages, list) or not passages:
        return False
    
    # Check at least one passage has required fields
    for passage in passages:
        if not isinstance(passage, dict):
            continue
            
        if "passage_text" in passage and isinstance(passage["passage_text"], str):
            return True
    
    return False


def get_dataset_stats(corpus_passages: list[dict], queries: list[dict]) -> dict:
    """Calculate statistics for loaded dataset.
    
    Args:
        corpus_passages: List of normalized passage records
        queries: List of normalized query records
        
    Returns:
        Dictionary with dataset statistics
    """
    if not corpus_passages:
        return {"error": "No corpus passages provided"}
    
    # Passage statistics
    passage_lengths = [len(p["text"]) for p in corpus_passages]
    selected_passages = sum(1 for p in corpus_passages if p["label_is_selected"] == 1)
    
    # Query statistics  
    query_lengths = [len(q["query_text"]) for q in queries] if queries else []
    
    stats = {
        "corpus": {
            "total_passages": len(corpus_passages),
            "selected_passages": selected_passages,
            "selection_rate": selected_passages / len(corpus_passages) if corpus_passages else 0,
            "avg_passage_length": sum(passage_lengths) / len(passage_lengths) if passage_lengths else 0,
            "min_passage_length": min(passage_lengths) if passage_lengths else 0,
            "max_passage_length": max(passage_lengths) if passage_lengths else 0
        },
        "queries": {
            "total_queries": len(queries),
            "avg_query_length": sum(query_lengths) / len(query_lengths) if query_lengths else 0,
            "min_query_length": min(query_lengths) if query_lengths else 0,
            "max_query_length": max(query_lengths) if query_lengths else 0
        }
    }
    
    return stats