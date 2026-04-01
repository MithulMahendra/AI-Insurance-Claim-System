"""
vector_search_tool.py
---------------------
LangChain Tool: Semantic / dense vector search via PGVector similarity.

Best for: natural-language questions, conceptual / explanatory queries.
"""

from langchain.tools import tool
from src.core.helper import get_vector_store


@tool
def vector_search(query: str, k: int = 5) -> list[dict]:
    """
    Perform a semantic vector search against the Insurance & Policy Knowledge Base.

    Use this tool when the query is a natural-language question or
    describes a concept (e.g., 'What is the vehicle flood policy?').

    Args:
        query: The user's search query.
        k:     Number of top results to return (default 5).

    Returns:
        A list of dicts, each with 'content' and 'metadata' keys.
        metadata includes: source, document_name, page, category.
    """
    print("Calling vector_search")
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(query, k=k)
    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]
