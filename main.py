from fastapi import FastAPI
from src.api.v1.routes.query import router as query_router
from src.api.v1.routes.admin import router as admin_router


app = FastAPI(
    title="Policy RAG API",
    description="Hybrid RAG system for Insurance policy document Q&A",
    version="1.0.0",
)

#Prefixes
app.include_router(query_router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1/admin")
