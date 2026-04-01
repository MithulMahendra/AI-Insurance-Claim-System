"""
fts_search_tool.py
------------------
LangChain Tool: Full-Text Search (FTS) via PostgreSQL tsvector / ts_rank.

"""

import os
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

# psycopg requires standard postgresql:// scheme (not postgresql+psycopg://)
_RAW_CONN = os.getenv("PG_CONNECTION_STRING", "").replace(
    "postgresql+psycopg", "postgresql"
)

_COLLECTION_NAME = "hr_support_desk"

_FTS_SQL = """
    SELECT
        e.document                                               AS content,
        e.cmetadata                                              AS metadata,
        ts_rank(
            to_tsvector('english', e.document),
            plainto_tsquery('english', %(query)s)
        )                                                        AS fts_rank
    FROM  langchain_pg_embedding  e
    JOIN  langchain_pg_collection c ON c.uuid = e.collection_id
    WHERE c.name = %(collection)s
      AND to_tsvector('english', e.document)
          @@ plainto_tsquery('english', %(query)s)
    ORDER BY fts_rank DESC
    LIMIT %(k)s;
"""


@tool
def fts_search(query: str, k: int = 5) -> list[dict]:
    """
    Perform a full-text (keyword) search against the Insurance & Policy Knowledge Base using
    PostgreSQL tsvector and ts_rank scoring.

    Use this tool when the query contains:
    - Policy or ticket codes  (e.g. 'POL-2024-HR-007')
    - Search through the policies
    - define claim id
    - Exact keyword matches

    Args:
        query: The user's search query (plain text).
        k:     Number of top results to return (default 5).

    Returns:
        A list of dicts with 'content', 'metadata', and 'fts_rank' keys.
    """
    print("Calling fts_search")
    with psycopg.connect(_RAW_CONN, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                _FTS_SQL,
                {"query": query, "collection": _COLLECTION_NAME, "k": k},
            )
            rows = cur.fetchall()
    
    return [
        {
            "content": row["content"],
            "metadata": row["metadata"],
            "fts_rank": round(float(row["fts_rank"]), 4),
        }
        for row in rows
    ]
