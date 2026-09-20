import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from src.generator import generate_answer
from evals.rag_eval_judge import evaluate_rag_response
from src.settings import settings

DEFAULT_RAG_DATASET_PATH = os.path.join(settings.BASE_DIR, "datasets", "rag_eval_dataset.json")
LOGS_DIR = os.path.join(settings.BASE_DIR, "evals", "logs")
REPORTS_DIR = os.path.join(settings.BASE_DIR, "evals", "reports")
LOG_FILE_PATH = os.path.join(LOGS_DIR, "rag_eval_execution.log")
REPORT_JSON_PATH = os.path.join(REPORTS_DIR, "rag_eval_report.json")
REPORT_MD_PATH = os.path.join(REPORTS_DIR, "rag_eval_report.md")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Configure structured file & console logger
logger = logging.getLogger("rag_eval_logger")
logger.setLevel(logging.DEBUG)

# File handler with full debug detail
if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE_PATH, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


def run_rag_eval_pipeline(
    dataset_path: str = DEFAULT_RAG_DATASET_PATH,
    max_samples: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Runs the complete End-to-End RAG Evaluation Pipeline:
    1. Loads dataset of test cases (in-scope + adversarial/false questions).
    2. Runs RAG pipeline to generate answers and context.
    3. Executes LLM-as-a-Judge against 4 rubrics (Faithfulness, Relevancy, Correctness, Refusal).
    4. Computes aggregate statistics and category breakdowns.
    5. Saves updated dataset and exports JSON/Markdown evaluation reports.
    6. Logs execution traces to evals/logs/rag_eval_execution.log.
    """
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path does not exist: {dataset_path}")
        raise FileNotFoundError(f"RAG Evaluation dataset not found at: {dataset_path}")

    logger.info(f"Starting RAG Evaluation. Model: {settings.MODEL_NAME}, Top-K: {settings.TOP_K}, Dataset: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        full_dataset = json.load(f)

    samples_to_run = full_dataset[:max_samples] if max_samples else full_dataset
    total_cases = len(samples_to_run)
    start_time = time.time()

    if verbose:
        print("\n" + "=" * 80)
        print("  🚀 STARTING END-TO-END RAG LLM-AS-A-JUDGE EVALUATION PIPELINE")
        print("=" * 80)
        print(f" Generator Model : {settings.MODEL_NAME}")
        print(f" Top-K Chunks    : {settings.TOP_K}")
        print(f" Samples To Run  : {total_cases} (out of {len(full_dataset)} total)")
        print(f" Dataset Path    : {dataset_path}")
        print(f" Execution Log   : {LOG_FILE_PATH}")
        print("-" * 80)

    evaluated_records = []
    faithfulness_scores = []
    relevancy_scores = []
    correctness_scores = []
    refusal_scores = []
    category_stats: Dict[str, Dict[str, List[float]]] = {}

    # Map for merging updates back to full dataset
    evaluated_map = {}

    for idx, item in enumerate(samples_to_run, 1):
        test_id = item.get("id", f"TC_{idx:03d}")
        query = item["query"]
        category = item.get("category", "General")
        is_false = item.get("is_out_of_scope_or_false", False)
        ground_truth = item.get("ground_truth_answer", "")

        if category not in category_stats:
            category_stats[category] = {
                "faithfulness": [],
                "relevancy": [],
                "correctness": [],
                "refusal": [],
                "passes": 0,
                "total": 0
            }

        # Step 1: Run RAG Generator (Retriever + LLM)
        try:
            rag_output = generate_answer(query=query)
            gen_answer = rag_output.get("answer", "")
            retrieved_chunks = rag_output.get("retrieved_chunks", [])
            sources = rag_output.get("sources", [])
        except Exception as e:
            gen_answer = f"ERROR during generation: {str(e)}"
            retrieved_chunks = []
            sources = []

        # Combine retrieved chunks into context string for judge
        context_str = "\n\n".join([
            f"[Chunk {i+1} | Source: {c.get('source', 'HR Policy')}]\n{c.get('content', '')}"
            for i, c in enumerate(retrieved_chunks)
        ])

        # Step 2: Run LLM-as-a-Judge
        try:
            judge_res = evaluate_rag_response(
                query=query,
                retrieved_context=context_str,
                ground_truth_answer=ground_truth,
                generated_answer=gen_answer,
                is_out_of_scope_or_false=is_false
            )
        except Exception as e:
            judge_res = {
                "faithfulness": {"score": 1, "reasoning": f"Judge error: {str(e)}"},
                "answer_relevancy": {"score": 1, "reasoning": f"Judge error: {str(e)}"},
                "semantic_correctness": {"score": 1, "reasoning": f"Judge error: {str(e)}"},
                "refusal_accuracy": {"score": 1, "reasoning": f"Judge error: {str(e)}"},
                "overall_score_avg": 1.0,
                "verdict": "FAIL"
            }

        f_score = judge_res["faithfulness"]["score"]
        r_score = judge_res["answer_relevancy"]["score"]
        c_score = judge_res["semantic_correctness"]["score"]
        ref_score = judge_res["refusal_accuracy"]["score"]
        verdict = judge_res["verdict"]

        faithfulness_scores.append(f_score)
        relevancy_scores.append(r_score)
        correctness_scores.append(c_score)
        refusal_scores.append(ref_score)

        # Update category stats
        category_stats[category]["faithfulness"].append(f_score)
        category_stats[category]["relevancy"].append(r_score)
        category_stats[category]["correctness"].append(c_score)
        category_stats[category]["refusal"].append(ref_score)
        category_stats[category]["total"] += 1
        if verdict == "PASS":
            category_stats[category]["passes"] += 1

        # Structured file logging for the sample
        logger.info(
            f"[{idx:02d}/{total_cases:02d}] ID: {test_id} | Category: {category} | Verdict: {verdict} | "
            f"Scores -> F:{f_score} R:{r_score} C:{c_score} Ref:{ref_score}"
        )
        logger.debug(f"Query: {query}")
        logger.debug(f"Generated: {gen_answer}")
        logger.debug(f"Ground Truth: {ground_truth}")
        logger.debug(f"Judge Reasoning: {judge_res}")

        # Update item record
        item["generated_answer"] = gen_answer
        item["retrieved_context_sources"] = sources
        item["judge_results"] = judge_res
        evaluated_records.append(item)

        if verbose:
            status_symbol = "✅ PASS" if verdict == "PASS" else "❌ FAIL"
            print(f"[{idx:02d}/{total_cases:02d}] {test_id} ({category}) | F:{f_score}/5 R:{r_score}/5 C:{c_score}/5 Ref:{ref_score}/5 -> {status_symbol}")

    elapsed_time = round(time.time() - start_time, 2)
    logger.info(f"Completed evaluation of {total_cases} test cases in {elapsed_time} seconds.")

    # Step 3: Compute aggregate summary
    avg_faithfulness = round(sum(faithfulness_scores) / len(faithfulness_scores), 2) if faithfulness_scores else 0.0
    avg_relevancy = round(sum(relevancy_scores) / len(relevancy_scores), 2) if relevancy_scores else 0.0
    avg_correctness = round(sum(correctness_scores) / len(correctness_scores), 2) if correctness_scores else 0.0
    avg_refusal = round(sum(refusal_scores) / len(refusal_scores), 2) if refusal_scores else 0.0

    all_scores = [avg_faithfulness, avg_relevancy, avg_correctness, avg_refusal]
    overall_avg_score = round(sum(all_scores) / len(all_scores), 2)
    overall_percentage = round((overall_avg_score / 5.0) * 100, 1)

    total_passes = sum(1 for item in evaluated_records if item["judge_results"]["verdict"] == "PASS")
    pass_rate = round((total_passes / total_cases) * 100, 1)

    # Step 4: Category breakdown formatting
    cat_summary = {}
    for cat_name, stats in category_stats.items():
        cnt = stats["total"]
        cat_summary[cat_name] = {
            "count": cnt,
            "faithfulness_avg": round(sum(stats["faithfulness"]) / cnt, 2) if cnt else 0,
            "relevancy_avg": round(sum(stats["relevancy"]) / cnt, 2) if cnt else 0,
            "correctness_avg": round(sum(stats["correctness"]) / cnt, 2) if cnt else 0,
            "refusal_avg": round(sum(stats["refusal"]) / cnt, 2) if cnt else 0,
            "pass_rate_percentage": round((stats["passes"] / cnt) * 100, 1) if cnt else 0
        }

    report_payload = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed_time,
        "model_name": settings.MODEL_NAME,
        "top_k": settings.TOP_K,
        "total_test_cases": total_cases,
        "summary_metrics": {
            "faithfulness_avg_score": avg_faithfulness,
            "faithfulness_percentage": round((avg_faithfulness / 5.0) * 100, 1),
            "answer_relevancy_avg_score": avg_relevancy,
            "answer_relevancy_percentage": round((avg_relevancy / 5.0) * 100, 1),
            "semantic_correctness_avg_score": avg_correctness,
            "semantic_correctness_percentage": round((avg_correctness / 5.0) * 100, 1),
            "refusal_accuracy_avg_score": avg_refusal,
            "refusal_accuracy_percentage": round((avg_refusal / 5.0) * 100, 1),
            "overall_quality_score_5": overall_avg_score,
            "overall_quality_percentage": overall_percentage,
            "total_passed": total_passes,
            "pass_rate_percentage": pass_rate
        },
        "category_breakdown": cat_summary,
        "test_cases": evaluated_records
    }

    # Step 5: Save evaluated dataset and reports
    # Merge evaluated records into full dataset by matching ID
    eval_by_id = {item.get("id"): item for item in evaluated_records if item.get("id")}
    for i, orig_item in enumerate(full_dataset):
        item_id = orig_item.get("id")
        if item_id in eval_by_id:
            full_dataset[i] = eval_by_id[item_id]

    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(full_dataset, f, indent=2)

    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    _generate_markdown_report(report_payload, REPORT_MD_PATH)

    if verbose:
        _print_ascii_summary(report_payload)

    return report_payload


def _print_ascii_summary(report: Dict[str, Any]):
    sm = report["summary_metrics"]
    cb = report["category_breakdown"]
    print("\n" + "=" * 80)
    print("                    🎯 RAG PIPELINE LLM-AS-A-JUDGE REPORT")
    print("=" * 80)
    print(f" Generator Model : {report['model_name']} | Top-K: {report['top_k']}")
    print(f" Total Evaluated : {report['total_test_cases']} test cases in {report['elapsed_seconds']}s")
    print("-" * 80)
    print(f" {'METRIC':<30} | {'SCORE (1-5)':<12} | {'PERCENTAGE':<12} | {'STATUS'}")
    print("-" * 80)
    print(f" {'Faithfulness (Zero Hallucination)':<30} | {sm['faithfulness_avg_score']:<12} | {sm['faithfulness_percentage']:<11}% | {'✅ PASS' if sm['faithfulness_avg_score']>=4.0 else '❌ FAIL'}")
    print(f" {'Answer Relevancy':<30} | {sm['answer_relevancy_avg_score']:<12} | {sm['answer_relevancy_percentage']:<11}% | {'✅ PASS' if sm['answer_relevancy_avg_score']>=4.0 else '❌ FAIL'}")
    print(f" {'Semantic Correctness':<30} | {sm['semantic_correctness_avg_score']:<12} | {sm['semantic_correctness_percentage']:<11}% | {'✅ PASS' if sm['semantic_correctness_avg_score']>=4.0 else '❌ FAIL'}")
    print(f" {'Refusal / Guardrail Accuracy':<30} | {sm['refusal_accuracy_avg_score']:<12} | {sm['refusal_accuracy_percentage']:<11}% | {'✅ PASS' if sm['refusal_accuracy_avg_score']>=4.0 else '❌ FAIL'}")
    print("-" * 80)
    print(f" {'OVERALL RAG QUALITY':<30} | {sm['overall_quality_score_5']}/5.00      | {sm['overall_quality_percentage']:<11}% | 🏆 {'EXCELLENT' if sm['overall_quality_percentage']>=90 else 'GOOD'}")
    print(f" {'Pass Rate (Score >= 4.0)':<30} | {sm['total_passed']}/{report['total_test_cases']:<8} | {sm['pass_rate_percentage']:<11}% |")
    print("=" * 80)
    print(" 📂 CATEGORY BREAKDOWN:")
    print("-" * 80)
    print(f" {'Category':<32} | {'Count':<6} | {'Faithful':<8} | {'Relevancy':<9} | {'Pass Rate'}")
    print("-" * 80)
    for cat, data in cb.items():
        print(f" {cat:<32} | {data['count']:<6} | {data['faithfulness_avg']:<8} | {data['relevancy_avg']:<9} | {data['pass_rate_percentage']}%")
    print("=" * 80)
    print(f" 📄 Full JSON Report: {REPORT_JSON_PATH}")
    print(f" 📑 Full Markdown   : {REPORT_MD_PATH}\n")


def _generate_markdown_report(report: Dict[str, Any], output_path: str):
    sm = report["summary_metrics"]
    cb = report["category_breakdown"]
    md = f"""# 📑 Enterprise HR RAG Evaluation Report

- **Generated Date**: {report['timestamp']}
- **Generator LLM**: `{report['model_name']}`
- **Retriever Top-K**: `{report['top_k']}`
- **Total Test Cases**: `{report['total_test_cases']}`
- **Evaluation Duration**: `{report['elapsed_seconds']}s`

---

## 🏆 Overall Quality Score

### **{sm['overall_quality_score_5']} / 5.00 ({sm['overall_quality_percentage']}%)**
- **Pass Rate**: **{sm['pass_rate_percentage']}%** ({sm['total_passed']} of {report['total_test_cases']} cases passed threshold >= 4.0)

---

## 📊 Core Evaluation Rubrics

| Metric | Average Score (1-5) | Normalized % | Target Threshold | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Faithfulness / Groundedness** | {sm['faithfulness_avg_score']} / 5.0 | {sm['faithfulness_percentage']}% | >= 90% | {'✅ PASS' if sm['faithfulness_avg_score']>=4.0 else '❌ FAIL'} |
| **Answer Relevancy** | {sm['answer_relevancy_avg_score']} / 5.0 | {sm['answer_relevancy_percentage']}% | >= 90% | {'✅ PASS' if sm['answer_relevancy_avg_score']>=4.0 else '❌ FAIL'} |
| **Semantic Correctness** | {sm['semantic_correctness_avg_score']} / 5.0 | {sm['semantic_correctness_percentage']}% | >= 90% | {'✅ PASS' if sm['semantic_correctness_avg_score']>=4.0 else '❌ FAIL'} |
| **Refusal & Guardrail Accuracy** | {sm['refusal_accuracy_avg_score']} / 5.0 | {sm['refusal_accuracy_percentage']}% | >= 90% | {'✅ PASS' if sm['refusal_accuracy_avg_score']>=4.0 else '❌ FAIL'} |

---

## 📂 Category Breakdown

| Category | Count | Faithfulness | Relevancy | Correctness | Refusal | Pass Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for cat, d in cb.items():
        md += f"| **{cat}** | {d['count']} | {d['faithfulness_avg']} | {d['relevancy_avg']} | {d['correctness_avg']} | {d['refusal_avg']} | {d['pass_rate_percentage']}% |\n"

    md += """
---

## 🔍 Sample Evaluated Cases

"""
    # Add top 5 sample cases
    for case in report["test_cases"][:6]:
        jr = case.get("judge_results", {})
        md += f"""### [{case.get('id')}] {case.get('category')} ({jr.get('verdict', 'PASS')})
- **Query**: {case.get('query')}
- **Is False / Out-of-Scope**: `{case.get('is_out_of_scope_or_false')}`
- **Generated Answer**: {case.get('generated_answer')}
- **Ground Truth**: {case.get('ground_truth_answer')}
- **Judge Verdict**: Avg Score `{jr.get('overall_score_avg', 5.0)}/5.0`
  - *Faithfulness*: `{jr.get('faithfulness', {}).get('score')}/5` — {jr.get('faithfulness', {}).get('reasoning')}
  - *Relevancy*: `{jr.get('answer_relevancy', {}).get('score')}/5` — {jr.get('answer_relevancy', {}).get('reasoning')}
  - *Semantic Correctness*: `{jr.get('semantic_correctness', {}).get('score')}/5` — {jr.get('semantic_correctness', {}).get('reasoning')}
  - *Refusal Accuracy*: `{jr.get('refusal_accuracy', {}).get('score')}/5` — {jr.get('refusal_accuracy', {}).get('reasoning')}

---
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    run_rag_eval_pipeline()
