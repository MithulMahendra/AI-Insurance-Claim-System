import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.core.helper import get_vector_store
from sqlalchemy import create_engine, text

load_dotenv()
PG_CONNECTION = os.getenv("PG_CONNECTION_STRING")

def create_ivf_index(db_connection_string, embedding_dim=1536):
    """
    Alters the embedding column to the specified dimension and 
    creates an IVFFlat index for faster cosine similarity search.
    """
    try:
        engine = create_engine(db_connection_string)
        
        with engine.connect() as conn:
            # Step 1: Alter column to ensure vector has correct dimensions (1536)
            # USING clause casts existing data to the new vector type
            alter_sql = text(f"""
                ALTER TABLE langchain_pg_embedding 
                ALTER COLUMN embedding TYPE vector({embedding_dim}) 
                USING embedding::vector({embedding_dim});
            """)
            conn.execute(alter_sql)
            print(f"✓ Embedding column altered to vector({embedding_dim})")
            
            # Step 2: Create IVFFlat index for efficient retrieval
            index_sql = text(f"""
                CREATE INDEX IF NOT EXISTS ivfflat_policy_idx 
                ON langchain_pg_embedding 
                USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100);
            """)
            conn.execute(index_sql)
            conn.commit()
            print("✓ IVFFlat index created successfully on 'langchain_pg_embedding'")
            
    except Exception as e:
        print(f"✗ Failed to setup IVFFlat index: {e}")
        print("Tip: Ensure pgvector extension is enabled: CREATE EXTENSION IF NOT EXISTS vector;")

def ingest_pdf(file_path):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    if not docs:
        raise ValueError("No content extracted from PDF.")

    print(f"Pages loaded: {len(docs)}")

    for doc in docs:
        doc.metadata.update({
            "source": file_path,
            "document_extension": "pdf",
            "page": doc.metadata.get("page"),
            "category": "policy_docs",
            "last_updated": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
        })

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=520,   
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(docs)
    print(f"Chunks created: {len(chunks)}")

    vector_store = get_vector_store(collection_name="policy_docs")
    vector_store.add_documents(chunks)

    print("Ingestion Completed Successfully")
    
    # --- ADD ON: Setup IVF Index after indexing ---
    create_ivf_index(PG_CONNECTION, embedding_dim=1536)


if __name__ == "__main__":
    ingest_pdf(r"data\Insurance Claims Processing & Intelligence System - FAQ.pdf")

