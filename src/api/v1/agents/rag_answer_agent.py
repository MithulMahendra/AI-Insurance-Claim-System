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

# init gemini 
model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)

# prompt for the routing agent
SYSTEM_PROMPT = """You are an expert Insurance and Policy assistant.

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

# setup the base agent forcing our QueryResponse schema
base_agent = create_agent(
    model=model,
    tools=[vector_search, fts_search, hybrid_search],
    system_prompt=_SYSTEM_PROMPT,
    response_format=QueryResponse,
)

# basic in-memory session cache
history_store: dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _history_store:
        history_store[session_id] = ChatMessageHistory()
    return history_store[session_id]

# wrap agent to handle conversation turns automatically
agent_with_history = RunnableWithMessageHistory(
    base_agent,
    get_session_history,
    input_messages_key="messages",
    history_messages_key="chat_history",
)


def run_rag_agent(
    query: str,
    session_id: str = "default",
    customer_context: dict[str, Any] | None = None,
) -> QueryResponse:
    """Main entry point for the RAG agent."""

    # prepend customer data to the prompt if we have it
    if customer_context:
        context_block = json.dumps(customer_context, indent=2, ensure_ascii=False)
        full_message = (
            f"[Customer Context]\n{context_block}\n\n"
            f"[Question]\n{query}"
        )
    else:
        full_message = query

    # fire off the request
    result = agent_with_history.invoke(
        {"messages": [{"role": "user", "content": full_message}]},
        config={"configurable": {"session_id": session_id}},
    )
    print(result)

    structured = result.get("structured_response")

    # happy path: we got our pydantic model back
    if isinstance(structured, QueryResponse):
        structured.query = query

        # hack to fix LLMs hallucinating or truncating chunk data.
        # we parse backwards through the messages to find the raw tool output 
        # and inject it directly into the response instead of trusting the agent's summary.
        for msg in reversed(result.get("messages", [])):
            if getattr(msg, "type", "") == "tool" and getattr(msg, "name", "") in ["vector_search", "fts_search", "hybrid_search"]:
                try:
                    raw_chunks = json.loads(msg.content)
                    structured.relevant_chunks = [json.dumps(chunk) for chunk in raw_chunks]
                except Exception as e:
                    print(f"Failed to parse chunks from tool message: {e}")
                break 

        return structured

    # fallback if the model dumps a plain dict instead of the schema object
    elif isinstance(structured, dict):
        return QueryResponse(
            query=query,
            answer=structured.get("answer", "I could not generate a proper answer."),
            citation=structured.get("citation", ""),
            page_no=structured.get("page_no"),
            document_name=structured.get("document_name", ""),
            relevant_chunks=structured.get("relevant_chunks", []) 
        )

    # worst-case fallback: try to scrape something usable out of the raw text
    else:
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
