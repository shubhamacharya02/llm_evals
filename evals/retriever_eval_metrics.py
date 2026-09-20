import re
from typing import List, Dict, Any, Set


def extract_keywords(text: str) -> Set[str]:
    """
    Extracts informative alphanumeric keywords and numbers from text,
    filtering out short words and common English stop words.
    """
    if not text:
        return set()
    words = re.findall(r'\b[a-zA-Z0-9_\-\$%]+\b', text.lower())
    stop_words = {
        'the', 'and', 'for', 'are', 'with', 'this', 'that', 'from', 'have',
        'what', 'which', 'when', 'where', 'how', 'who', 'must', 'will', 'all',
        'per', 'day', 'days', 'year', 'years', 'their', 'such', 'upon', 'been',
        'were', 'into', 'some', 'than', 'them', 'then', 'also', 'about', 'over',
        'after', 'only', 'more', 'most', 'other', 'each', 'does', 'can', 'may',
        'any', 'our', 'out', 'not', 'your', 'you', 'its', 'under', 'between'
    }
    return {w for w in words if len(w) > 1 and w not in stop_words and (not w.isdigit() or len(w) >= 2)}


def is_chunk_relevant(chunk_text: str, expected_chunks: List[str]) -> bool:
    """
    Determines if a retrieved chunk is relevant to any of the expected clauses.
    
    A chunk is relevant if it shares at least 25% of keywords or >= 3 distinctive terms
    with at least one expected clause.
    """
    if not expected_chunks or not chunk_text:
        return False

    chunk_keywords = extract_keywords(chunk_text)
    if not chunk_keywords:
        return False

    for exp_clause in expected_chunks:
        exp_keywords = extract_keywords(exp_clause)
        if not exp_keywords:
            continue
        overlap = exp_keywords.intersection(chunk_keywords)
        if len(overlap) >= 3 or (len(overlap) / len(exp_keywords) >= 0.25):
            return True

    return False


def compute_context_recall(expected_chunks: List[str], retrieved_chunks: List[Dict[str, Any]]) -> float:
    """
    Calculates Context Recall across all expected clauses/chunks.
    
    For multi-hop queries with multiple expected chunks, checks what proportion
    of each expected clause's information is retrieved.
    
    Formula:
        Recall = (1 / M) * sum_i ( |Retrieved Keywords ∩ Expected_i Keywords| / |Expected_i Keywords| )
    
    Returns:
        float: Score between 0.0 and 1.0.
    """
    # Negative / out-of-scope query handling (no expected chunks)
    if not expected_chunks:
        return 1.0

    if not retrieved_chunks:
        return 0.0

    combined_retrieved = " ".join([c.get("content", "") for c in retrieved_chunks])
    retrieved_keywords = extract_keywords(combined_retrieved)

    clause_recalls = []
    for exp_chunk in expected_chunks:
        exp_keywords = extract_keywords(exp_chunk)
        if not exp_keywords:
            clause_recalls.append(1.0)
            continue
        captured = exp_keywords.intersection(retrieved_keywords)
        clause_recalls.append(len(captured) / len(exp_keywords))

    overall_recall = sum(clause_recalls) / len(clause_recalls) if clause_recalls else 0.0
    return round(overall_recall, 4)


def compute_context_precision(expected_chunks: List[str], retrieved_chunks: List[Dict[str, Any]]) -> float:
    """
    Calculates Rank-Weighted Context Precision (RAGAS / IR Standard):
    
    Context Precision = sum_{k=1}^K (Precision@k * v_k) / sum_{k=1}^K v_k
    where v_k = 1 if chunk_k is relevant, else 0.
          Precision@k = (Number of relevant chunks in top-k) / k
          
    If no relevant chunks are found or query is negative, returns 0.0.
    
    Returns:
        float: Score between 0.0 and 1.0.
    """
    if not retrieved_chunks or not expected_chunks:
        return 0.0

    relevance_flags = []
    for chunk in retrieved_chunks:
        chunk_text = chunk.get("content", "")
        relevance_flags.append(1 if is_chunk_relevant(chunk_text, expected_chunks) else 0)

    total_relevant = sum(relevance_flags)
    if total_relevant == 0:
        return 0.0

    running_relevant = 0
    weighted_precisions = []

    for k, is_rel in enumerate(relevance_flags, start=1):
        if is_rel:
            running_relevant += 1
            precision_at_k = running_relevant / k
            weighted_precisions.append(precision_at_k)

    context_precision = sum(weighted_precisions) / total_relevant
    return round(context_precision, 4)


def compute_mrr(expected_chunks: List[str], retrieved_chunks: List[Dict[str, Any]]) -> float:
    """
    Calculates Mean Reciprocal Rank (MRR) measuring how high the first
    relevant chunk is positioned in the ranking (1/rank).
    
    Returns:
        float: 1.0 (Rank 1), 0.5 (Rank 2), 0.3333 (Rank 3), 0.25 (Rank 4), or 0.0.
    """
    if not expected_chunks or not retrieved_chunks:
        return 0.0

    for rank, chunk in enumerate(retrieved_chunks, 1):
        chunk_text = chunk.get("content", "")
        if is_chunk_relevant(chunk_text, expected_chunks):
            return round(1.0 / rank, 4)

    return 0.0


def compute_context_f1(recall: float, precision: float) -> float:
    """Calculates the harmonic mean (F1 score) of Context Recall and Precision."""
    if (precision + recall) == 0:
        return 0.0
    return round(2 * (precision * recall) / (precision + recall), 4)


def compute_retrieval_metrics(
    expected_chunks: List[str],
    retrieved_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Comprehensive retrieval evaluation returning:
    - context_recall: Completeness of retrieved relevant facts (0.0 - 1.0)
    - context_precision: Rank-weighted precision of relevant chunks (0.0 - 1.0)
    - mrr: Mean Reciprocal Rank of first relevant chunk (1/rank)
    - f1_score: Harmonic mean of context recall and precision
    - hit_at_1: True if the rank #1 retrieved chunk is relevant
    """
    recall = compute_context_recall(expected_chunks, retrieved_chunks)
    precision = compute_context_precision(expected_chunks, retrieved_chunks)
    mrr = compute_mrr(expected_chunks, retrieved_chunks)
    f1 = compute_context_f1(recall, precision)
    hit_at_1 = (mrr == 1.0)

    return {
        "context_recall": recall,
        "context_precision": precision,
        "mrr": mrr,
        "f1_score": f1,
        "hit_at_1": hit_at_1
    }
