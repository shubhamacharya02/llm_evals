from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from src.document_process import (
    save_uploaded_file,
    process_and_index_document,
    retrieve_relevant_chunks
)

app = APIRouter()


class RetrieveRequest(BaseModel):
    query: str
    top_k: Optional[int] = None


@app.post("/upload_docs")
def upload_docs(doc: UploadFile = File(...)):
    try:
        saved_file_path = save_uploaded_file(doc)
        total_chunks = process_and_index_document(saved_file_path)
        return {
            "status": "success",
            "message": f"File '{doc.filename}' uploaded and indexed successfully.",
            "file_path": saved_file_path,
            "total_chunks_indexed": total_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve")
def retrieve_documents(request: RetrieveRequest):
    try:
        retrieved_docs = retrieve_relevant_chunks(
            query=request.query,
            top_k=request.top_k
        )
        return {
            "status": "success",
            "query": request.query,
            "count": len(retrieved_docs),
            "retrieved_documents": retrieved_docs
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

