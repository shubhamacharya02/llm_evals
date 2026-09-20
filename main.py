from fastapi import FastAPI
from src.router import app as router

app = FastAPI(
    title="Enterprise HR RAG & Evaluation Pipeline API",
    description="Full-stack Enterprise RAG Pipeline with Retriever Benchmarks and LLM-as-a-Judge Evaluation Engine.",
    version="1.0.0"
)

app.include_router(router)


@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "Enterprise HR RAG & Evaluation Pipeline",
        "version": "1.0.0",
        "docs_url": "/docs"
    }
