import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from typing import List, Dict, Any, Optional

from src.settings import settings
from src.document_process import (
    retrieve_relevant_chunks,
    process_and_index_document,
    get_chroma_vector_store
)
from evals.retriever_eval_metrics import compute_retrieval_metrics

DEFAULT_DATASET_PATH = os.path.join(settings.BASE_DIR, "datasets", "golden_dataset.json")


class RetrieverEvaluator:
    """
    Evaluator dedicated to measuring Vector Store & Retriever performance
    (Context Recall, Context Precision, Ranking Accuracy, Hit Rate).
    """

    def __init__(
        self,
        dataset_path: str = DEFAULT_DATASET_PATH,
        top_k: Optional[int] = None
    ):
        self.dataset_path = dataset_path
        self.top_k = top_k if top_k is not None and top_k > 0 else settings.TOP_K

    def load_dataset(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Golden dataset file not found at: {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_dataset(self, dataset: List[Dict[str, Any]]) -> None:
        with open(self.dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)

    def ensure_document_indexed(self, force_reset: bool = False) -> None:
        try:
            store = get_chroma_vector_store()
            current_count = store._collection.count()
        except Exception:
            current_count = 0

        if current_count > 0 and not force_reset:
            print(f"[RetrieverEvaluator] Vector store ready with {current_count} indexed chunks.")
            return

        candidate_paths = [
            os.path.join(settings.DATA_DIR, "hr_policy.pdf"),
            os.path.join(settings.BASE_DIR, "hr_policy.pdf"),
            "hr_policy.pdf"
        ]
        pdf_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if pdf_path:
            count = process_and_index_document(pdf_path, reset=force_reset)
            print(f"[RetrieverEvaluator] Successfully indexed {count} chunks from '{pdf_path}'.")
        else:
            print("[RetrieverEvaluator] Warning: No hr_policy.pdf found to index.")

    def evaluate(self, force_reset: bool = False) -> Dict[str, Any]:
        """
        Executes the evaluation benchmark:
        - Ensures documents are indexed in Chroma DB.
        - Runs queries against the retriever.
        - Logs detailed progress step-by-step to the terminal.
        - Computes precision and recall metrics.
        - Persists retrieved results into the dataset file.
        - Returns a structured summary report.
        """
        print("\n" + "=" * 80)
        print(f"🚀 STARTING RETRIEVER EVALUATION BENCHMARK (Top-K = {self.top_k})")
        print("=" * 80)

        self.ensure_document_indexed(force_reset=force_reset)
        dataset = self.load_dataset()
        total = len(dataset)
        print(f"Loaded {total} benchmark queries from: {self.dataset_path}\n")

        evaluated_records = []
        recalls = []
        precisions = []
        mrrs = []
        f1s = []

        for idx, item in enumerate(dataset, start=1):
            query = item["query"]
            category = item.get("category", "General")
            chapter = item.get("chapter", "N/A")
            expected_chunks = item.get("expected_chunks", [])

            print(f"[{idx:02d}/{total:02d}] Test ID: {item['id']} | Category: {category}")
            print(f"       Query: \"{query}\"")
            print(f"       Chapter Ref: {chapter} | Expected Clauses: {len(expected_chunks)}")

            # Perform semantic retrieval
            retrieved_chunks = retrieve_relevant_chunks(query=query, top_k=self.top_k)

            # Compute detailed retrieval metrics
            metrics = compute_retrieval_metrics(expected_chunks, retrieved_chunks)
            recalls.append(metrics["context_recall"])
            precisions.append(metrics["context_precision"])
            mrrs.append(metrics["mrr"])
            f1s.append(metrics["f1_score"])

            # Determine pass/fail status
            status = "PASS" if metrics["context_recall"] >= 0.6 else "FAIL"
            status_badge = "✅ PASS" if status == "PASS" else "❌ FAIL"
            top1_badge = "🎯 Hit@1" if metrics["hit_at_1"] else "⚠️ Miss@1"

            print(f"       Metrics -> Recall: {metrics['context_recall']:.4f} | "
                  f"Precision@{self.top_k}: {metrics['context_precision']:.4f} | "
                  f"MRR: {metrics['mrr']:.4f} | F1: {metrics['f1_score']:.4f} | "
                  f"{top1_badge} | Status: {status_badge}")
            print("-" * 80)

            # Update item with execution results
            item["retrieved_chunks"] = retrieved_chunks
            item["metrics"] = metrics

            evaluated_records.append({
                "id": item["id"],
                "query": query,
                "category": category,
                "difficulty": item.get("difficulty", "Medium"),
                "chapter": chapter,
                "expected_chunks_count": len(expected_chunks),
                "retrieved_chunks_count": len(retrieved_chunks),
                "context_recall": metrics["context_recall"],
                "context_precision": metrics["context_precision"],
                "mrr": metrics["mrr"],
                "f1_score": metrics["f1_score"],
                "hit_at_1": metrics["hit_at_1"],
                "status": status
            })

        # Persist updated dataset
        self.save_dataset(dataset)

        avg_recall = round(sum(recalls) / total, 4) if total else 0.0
        avg_precision = round(sum(precisions) / total, 4) if total else 0.0
        avg_mrr = round(sum(mrrs) / total, 4) if total else 0.0
        avg_f1 = round(sum(f1s) / total, 4) if total else 0.0
        top1_hits = sum(1 for m in mrrs if m == 1.0)
        passed_count = sum(1 for r in evaluated_records if r["status"] == "PASS")

        print("\n" + "=" * 80)
        print("🏁 RETRIEVER BENCHMARK EVALUATION SUMMARY")
        print("=" * 80)
        print(f"  • Total Test Queries Evaluated : {total}")
        print(f"  • Passed Queries               : {passed_count} / {total} ({(passed_count / total * 100):.1f}%)")
        print(f"  • Average Context Recall       : {avg_recall:.4f}")
        print(f"  • Average Context Precision    : {avg_precision:.4f}")
        print(f"  • Average MRR                  : {avg_mrr:.4f}")
        print(f"  • Average Context F1 Score     : {avg_f1:.4f}")
        print(f"  • Top-1 Hit Accuracy           : {(top1_hits / total * 100):.1f}%")
        print(f"  • Embedding Model              : {settings.EMBEDDING_MODEL_NAME}")
        print(f"  • Vector Store                 : Chroma DB")
        print(f"  • Results Persisted At         : {self.dataset_path}")
        print("=" * 80 + "\n")

        return {
            "status": "success",
            "benchmark_summary": {
                "total_test_queries": total,
                "passed_queries": passed_count,
                "success_rate": f"{(passed_count / total * 100):.1f}%" if total else "0%",
                "average_context_recall": avg_recall,
                "average_context_precision_at_k": avg_precision,
                "average_mrr": avg_mrr,
                "average_f1_score": avg_f1,
                "top_1_accuracy": f"{(top1_hits / total * 100):.1f}%" if total else "0%",
                "retriever_top_k": self.top_k,
                "embedding_model": settings.EMBEDDING_MODEL_NAME,
                "vector_store": "Chroma DB",
                "dataset_updated": self.dataset_path
            },
            "query_results": evaluated_records
        }
