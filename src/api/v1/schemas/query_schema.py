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


class QueryResponse(BaseModel):
    query: str = Field(..., description="The original user query.")
    answer: str = Field(..., description="Synthesised answer from retrieved chunks.")
    citation: str = Field(..., description="Verbatim excerpt from the source document.")
    page_no: Optional[int] = Field(
        default=None, description="Page number in the source PDF (0-indexed)."
    )
    document_name: str = Field(
        ..., description="Filename of the source document."
    )
    relevant_chunks: List[str] = Field(
        default_factory=list, 
        description="List of the exact text chunks retrieved from the tool that were used to answer the query."
    )

    
