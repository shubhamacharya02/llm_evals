import os
import json
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from src.document_process import (
    save_uploaded_file,
    process_and_index_document,
    retrieve_relevant_chunks
)
from src.generator import generate_answer
from evals.retriever_eval_process import run_evaluation_benchmark, DEFAULT_DATASET_PATH
from evals.rag_eval_process import run_rag_eval_pipeline, REPORT_JSON_PATH

app = APIRouter()


# =====================================================================
# REQUEST SCHEMAS
# =====================================================================

class AskPolicyRequest(BaseModel):
    query: str = Field(..., description="The user question regarding HR policy")


class RetrieveChunksRequest(BaseModel):
    query: str = Field(..., description="The search query to retrieve relevant chunks from vector store")


# =====================================================================
# 1. RAG CORE PIPELINE ENDPOINTS
# =====================================================================

@app.post(
    "/rag/ask",
    tags=["RAG Core Pipeline"],
    summary="Ask HR Policy Question (RAG Synthesis)",
    description="End-to-end RAG question answering. Retrieves relevant policy chunks (using settings.TOP_K) and synthesizes a grounded response with citations."
)
def ask_policy_question(request: AskPolicyRequest):
    try:
        result = generate_answer(query=request.query)
        return {
            "status": "success",
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/documents/upload",
    tags=["RAG Core Pipeline"],
    summary="Upload & Index Document",
    description="Uploads a PDF policy document, extracts text, generates embeddings, and indexes chunks into Chroma vector store."
)
def upload_and_index_document(doc: UploadFile = File(...)):
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


@app.post(
    "/retriever/search",
    tags=["RAG Core Pipeline"],
    summary="Retrieve Relevant Chunks",
    description="Queries the vector store directly and returns the top-K relevant text chunks with similarity scores and metadata."
)
def search_relevant_chunks(request: RetrieveChunksRequest):
    try:
        retrieved_docs = retrieve_relevant_chunks(query=request.query)
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


# =====================================================================
# 2. RETRIEVER EVALUATION BENCHMARK ENDPOINTS
# =====================================================================

@app.post(
    "/evals/retriever/run",
    tags=["Retriever Evaluation"],
    summary="Run Retriever Benchmark",
    description="Executes the retriever evaluation benchmark across golden_dataset.json and calculates Context Precision, Recall, MRR, and F1 Score."
)
def run_retriever_eval_benchmark():
    try:
        report = run_evaluation_benchmark()
        return report
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/evals/retriever/report",
    tags=["Retriever Evaluation"],
    summary="Get Retriever Evaluation Report",
    description="Fetches the latest cached retriever evaluation report from disk without re-executing searches."
)
def get_latest_retriever_report():
    try:
        if not os.path.exists(DEFAULT_DATASET_PATH):
            raise HTTPException(status_code=404, detail="Golden dataset not found.")

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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# 3. END-TO-END RAG LLM-AS-A-JUDGE EVALUATION ENDPOINTS
# =====================================================================

@app.post(
    "/evals/rag-judge/run",
    tags=["RAG LLM Judge Evaluation"],
    summary="Run RAG LLM-as-a-Judge Benchmark",
    description="Executes the full RAG generation and LLM Judge pipeline across all in-scope and adversarial/false test cases. Scores Faithfulness, Relevancy, Semantic Correctness, and Refusal Accuracy."
)
def run_rag_judge_eval_benchmark():
    try:
        report = run_rag_eval_pipeline()
        return report
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/evals/rag-judge/report",
    tags=["RAG LLM Judge Evaluation"],
    summary="Get RAG LLM Judge Report",
    description="Returns the latest generated RAG LLM-as-a-Judge evaluation report from disk without re-running model calls."
)
def get_latest_rag_judge_report():
    try:
        if not os.path.exists(REPORT_JSON_PATH):
            raise HTTPException(
                status_code=404,
                detail="No RAG evaluation report found yet. Run POST /evals/rag-judge/run first."
            )
        with open(REPORT_JSON_PATH, "r", encoding="utf-8") as f:
            report = json.load(f)
        return report
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
