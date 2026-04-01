from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query")
    session_id: str = Field(
        default="default",
        description="Conversation session ID for chat history (use a unique ID per user/session)"
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional metadata filter (e.g., policy_docs)"
    )
    customer_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional JSON object with details about the customer asking the question "
            "Used to personalise the LLM response."
        )
    )
