"""
hybrid_search_tool.py
----------------------
LangChain Tool: Hybrid search using Reciprocal Rank Fusion (RRF).

Combines vector (semantic) and FTS (keyword) results.
Best for: short, ambiguous queries or when both exact-match and
          conceptual recall are desirable simultaneously.

RRF formula: score(chunk) = Σ 1 / (rank + 60)
             summed across both vector and FTS result lists.
"""

from langchain.tools import tool
from src.core.helper import get_vector_store
from src.api.v1.tools.fts_search_tool import fts_search

_RRF_K = 60  # dampening constant — prevents top outliers from dominating


@tool
def hybrid_search(query: str, k: int = 5) -> list[dict]:
    """
    Perform a hybrid search (vector + full-text) against the Insurance & policy Knowledge Base
    using Reciprocal Rank Fusion to merge and re-rank results.

    Use this tool when:
    - The query is short (3 words or fewer)
    - The query is ambiguous or could benefit from both semantic and keyword matching
    - You are unsure whether to use vector_search or fts_search

    Args:
        query: The user's search query.
        k:     Number of top fused results to return (default 5).

    Returns:
        A list of dicts with 'content' and 'metadata' keys, ranked by RRF score.
    """
    print("Calling hybrid_search")
    # --- Dense vector results ---
    vector_store = get_vector_store()
    vector_docs = vector_store.similarity_search(query, k=k)

    # --- Sparse FTS results ---
    # fts_search is a tool; call its underlying function directly (no tool wrapper overhead)
    fts_docs = fts_search.func(query=query, k=k)  # type: ignore[attr-defined]

    # --- Reciprocal Rank Fusion ---
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, doc in enumerate(vector_docs):
        key = doc.page_content[:120]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
        chunk_map[key] = {"content": doc.page_content, "metadata": doc.metadata}

    for rank, item in enumerate(fts_docs):
        key = item["content"][:120]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
        chunk_map[key] = {"content": item["content"], "metadata": item["metadata"]}

    # Sort by descending RRF score and return top-k
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[key] for key, _ in ranked[:k]]
