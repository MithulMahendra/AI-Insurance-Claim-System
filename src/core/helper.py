import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

PG_CONNECTION = os.getenv("PG_CONNECTION_STRING")
if not PG_CONNECTION:
    raise ValueError("PG_CONNECTION_STRING is not set")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set")

EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDINGS_MODEL")
if not EMBEDDING_MODEL:
    raise ValueError("GOOGLE_EMBEDDINGS_MODEL is not set")


def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=GOOGLE_API_KEY,
        output_dimensionality=1536
    )


def get_vector_store(collection_name: str = "policy_docs"):
    return PGVector(
        collection_name=collection_name,
        connection=PG_CONNECTION,
        embeddings=get_embedding_model(),
        use_jsonb=True
    )


