from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from src.document_process import (
    save_uploaded_file,
    process_and_index_document,
    retrieve_relevant_chunks
)
from evals.retriever_eval_process import run_evaluation_benchmark, DEFAULT_DATASET_PATH

app = APIRouter()


class RetrieveRequest(BaseModel):
    query: str
    top_k: Optional[int] = None


class EvalRequest(BaseModel):
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


@app.post("/run_eval")
def trigger_evaluation(request: Optional[EvalRequest] = None):
    """
    Runs the retriever evaluation benchmark against the golden dataset,
    updates golden_dataset.json with retrieved chunks and precision/recall metrics,
    and returns the comprehensive evaluation report.
    """
    try:
        top_k = request.top_k if request else None
        report = run_evaluation_benchmark(dataset_path=DEFAULT_DATASET_PATH, top_k=top_k)
        return report
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/eval_report")
def get_evaluation_report():
    """
    Returns the latest evaluation report from the saved golden dataset without re-running queries.
    """
    try:
        import json
        with open(DEFAULT_DATASET_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        total = len(dataset)
        recalls = [item["metrics"]["context_recall"] for item in dataset if item.get("metrics", {}).get("context_recall") is not None]
        precisions = [item["metrics"]["context_precision"] for item in dataset if item.get("metrics", {}).get("context_precision") is not None]

        return {
            "status": "success",
            "total_records": total,
            "evaluated_records": len(recalls),
            "average_context_recall": round(sum(recalls)/len(recalls), 4) if recalls else None,
            "average_context_precision": round(sum(precisions)/len(precisions), 4) if precisions else None,
            "records": dataset
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Golden dataset not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


