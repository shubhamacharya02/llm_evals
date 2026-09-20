import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, Optional
from evals.retriever_eval import RetrieverEvaluator, DEFAULT_DATASET_PATH


def run_evaluation_benchmark(
    dataset_path: str = DEFAULT_DATASET_PATH,
    top_k: Optional[int] = None
) -> Dict[str, Any]:
    """
    Runs the retriever benchmark against the golden dataset.
    Delegates execution to RetrieverEvaluator.
    """
    evaluator = RetrieverEvaluator(dataset_path=dataset_path, top_k=top_k)
    return evaluator.evaluate()


if __name__ == "__main__":
    import json
    report = run_evaluation_benchmark()
    print("\nBenchmark Summary:")
    print(json.dumps(report["benchmark_summary"], indent=2))
