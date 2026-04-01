from typing import Any
from src.api.v1.agents.rag_answer_agent import run_rag_agent


def query_documents(
    query: str,
    session_id: str = "default",
    customer_context: dict[str, Any] | None = None,
):
    return run_rag_agent(query, session_id=session_id, customer_context=customer_context)
