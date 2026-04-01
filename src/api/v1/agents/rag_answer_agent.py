from __future__ import annotations

import json
import os
from typing import Any
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from src.api.v1.tools.vector_search_tool import vector_search
from src.api.v1.tools.fts_search_tool import fts_search
from src.api.v1.tools.hybrid_search_tool import hybrid_search
from src.api.v1.schemas.query_schema import QueryResponse

load_dotenv()

# ---------------------------------------------------------------------------
# 1. LLM
# ---------------------------------------------------------------------------
model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)

# ---------------------------------------------------------------------------
# 2. System Prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are an expert Insurance and Policy assistant.

You have access to three retrieval tools:
vector_search     → best for natural language / conceptual questions
fts_search        → best for codes, IDs, abbreviations, exact keywords
hybrid_search     → best for short or ambiguous queries

Rules:
1. Choose exactly one tool and call it with the original user query.
2. Use ONLY the returned document chunks to answer.
3. Synthesise a clear, concise answer.
4. Always return your final response using the structured format below.
5. For 'page_no': use the value of metadata['page'] from the most relevant chunk.
   - If metadata['page'] is missing, set page_no to null — NEVER default to 0.
6. For 'document_name': use metadata['document_name'] from the most relevant chunk.
7. For 'relevant_chunks': include a list of the exact text chunks returned by the tool that you used to form your answer.

Do not add extra explanation after the structured output.
"""

# ---------------------------------------------------------------------------
# 3. Base agent (QueryResponse used directly as response_format)
#    'query' field is injected manually after the agent returns.
# ---------------------------------------------------------------------------
_base_agent = create_agent(
    model=model,
    tools=[vector_search, fts_search, hybrid_search],
    system_prompt=_SYSTEM_PROMPT,
    response_format=QueryResponse,
)

# ---------------------------------------------------------------------------
# 4. In-memory chat history store  { session_id → ChatMessageHistory }
# ---------------------------------------------------------------------------
_history_store: dict[str, ChatMessageHistory] = {}

def _get_session_history(session_id: str) -> ChatMessageHistory:
    """Return (or create) the ChatMessageHistory for the given session."""
    if session_id not in _history_store:
        _history_store[session_id] = ChatMessageHistory()
    return _history_store[session_id]

# ---------------------------------------------------------------------------
# 5. Wrap the agent with RunnableWithMessageHistory
#    - input_messages_key  : the key in the invoke dict that carries the new message
#    - history_messages_key: the key the agent expects for past messages
# ---------------------------------------------------------------------------
agent_with_history = RunnableWithMessageHistory(
    _base_agent,
    _get_session_history,
    input_messages_key="messages",
    history_messages_key="chat_history",
)

# ---------------------------------------------------------------------------
# 6. Public entry-point
# ---------------------------------------------------------------------------
def run_rag_agent(
    query: str,
    session_id: str = "default",
    customer_context: dict[str, Any] | None = None,
) -> QueryResponse:
    """Run the RAG agent with chat history and optional customer context."""

    # Build the message payload.
    if customer_context:
        context_block = json.dumps(customer_context, indent=2, ensure_ascii=False)
        full_message = (
            f"[Customer Context]\n{context_block}\n\n"
            f"[Question]\n{query}"
        )
    else:
        full_message = query

    result = agent_with_history.invoke(
        {"messages": [{"role": "user", "content": full_message}]},
        config={"configurable": {"session_id": session_id}},
    )
    print(result)

    structured = result.get("structured_response")

    if isinstance(structured, QueryResponse):
        # Agent returned a fully populated QueryResponse — inject query
        structured.query = query

        # -------------------------------------------------------------------
        # NEW CODE BLOCK: Intercept the exact tool output instead of relying on the LLM
        # -------------------------------------------------------------------
        # Look backwards through the messages to find the last search tool execution
        for msg in reversed(result.get("messages", [])):
            # Check if this message is a tool response and from one of our search tools
            if getattr(msg, "type", "") == "tool" and getattr(msg, "name", "") in ["vector_search", "fts_search", "hybrid_search"]:
                try:
                    # The tool returned a JSON string containing all the chunks
                    raw_chunks = json.loads(msg.content)
                    
                    # Overwrite whatever the LLM generated with the actual raw tool chunks
                    structured.relevant_chunks = [json.dumps(chunk) for chunk in raw_chunks]
                except Exception as e:
                    print(f"Could not parse tool message: {e}")
                break # Stop searching once we find the most recent tool call
        # -------------------------------------------------------------------

        return structured

    elif isinstance(structured, dict):
        # Fallback: agent returned a plain dict
        return QueryResponse(
            query=query,
            answer=structured.get("answer", "I could not generate a proper answer."),
            citation=structured.get("citation", ""),
            page_no=structured.get("page_no"),
            document_name=structured.get("document_name", ""),
            relevant_chunks=structured.get("relevant_chunks", []) 
        )

    else:
        # Last resort: extract text from the final message
        final_msg = result.get("messages", [])[-1]
        content = final_msg.content if hasattr(final_msg, "content") else str(final_msg)
        return QueryResponse(
            query=query,
            answer=str(content)[:500],
            citation="",
            page_no=None,
            document_name="",
            relevant_chunks=[] 
        )

