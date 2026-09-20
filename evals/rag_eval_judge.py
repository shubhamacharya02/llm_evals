import json
import re
from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from src.generator import get_llm

# =====================================================================
# RAG EVALUATION LLM-AS-A-JUDGE PROMPT TEMPLATES (LangChain)
# =====================================================================

RAG_JUDGE_SYSTEM_PROMPT = """You are an expert, impartial AI Judge evaluating an Enterprise HR Policy Retrieval-Augmented Generation (RAG) system.

You will be given:
1. User Query
2. Retrieved Context (Passages retrieved from the HR Policy Handbook)
3. Ground Truth Answer (Reference standard)
4. Generated Answer (The RAG model's response)
5. Is Out Of Scope / False Premise Flag (Boolean indicating if this question contains a false claim or is outside the policy scope)

Your job is to evaluate the Generated Answer across 4 distinct rubrics on a strict 1 to 5 scale:

1. FAITHFULNESS / GROUNDEDNESS (Context vs Answer)
- Score 5: Every single claim and number is 100% verified by the retrieved context. Zero hallucinations.
- Score 4: Minor conversational phrasing or extrapolation, but all core facts match context.
- Score 3: Partially supported by context, but contains at least one unverified claim or assumption.
- Score 2: Substantial claims contradict or are missing from the retrieved context.
- Score 1: Pure hallucination completely unsupported by the retrieved context.

2. ANSWER RELEVANCY (Query vs Answer)
- Score 5: Directly, precisely, and completely answers what the user asked without evading or fluff.
- Score 4: Answers the main question with minor extra details or slight wordiness.
- Score 3: Answers only part of the question or misses a key condition.
- Score 2: Tangentially related but does not provide the primary answer.
- Score 1: Off-topic or completely unresponsive.

3. SEMANTIC CORRECTNESS (Ground Truth vs Answer)
- Score 5: Semantically identical in facts, numbers, conditions, and conclusion to the Ground Truth.
- Score 4: Minor semantic drift in phrasing, but core facts and outcome match Ground Truth.
- Score 3: Partially correct, but gets a key number or rule wrong compared to Ground Truth.
- Score 2: Mostly incorrect or reaches an opposite conclusion.
- Score 1: Completely incorrect compared to Ground Truth.

4. REFUSAL & GUARDRAIL ACCURACY
- If 'Is Out Of Scope / False Premise' is TRUE:
  - Score 5: Politely states information is not found OR explicitly disproves the false claim using handbook rules.
  - Score 4: Refuses to answer, but explanation is slightly vague.
  - Score 3: Acknowledges missing context but offers speculative/general advice.
  - Score 2: Invented hypothetical rules while admitting uncertainty.
  - Score 1: Confidently hallucinated an answer to the false premise.
- If 'Is Out Of Scope / False Premise' is FALSE:
  - Score 5: Correctly answered the in-scope question without unnecessary refusal.
  - Score 1: Incorrectly refused to answer an in-scope question that had valid context.

You MUST respond strictly with a valid JSON object in the following format (no markdown fences, no text outside JSON):
{{
  "faithfulness": {{
    "score": <integer 1-5>,
    "reasoning": "<concise explanation>"
  }},
  "answer_relevancy": {{
    "score": <integer 1-5>,
    "reasoning": "<concise explanation>"
  }},
  "semantic_correctness": {{
    "score": <integer 1-5>,
    "reasoning": "<concise explanation>"
  }},
  "refusal_accuracy": {{
    "score": <integer 1-5>,
    "reasoning": "<concise explanation>"
  }},
  "overall_score_avg": <float between 1.0 and 5.0>,
  "verdict": "<PASS or FAIL>"
}}
"""

RAG_JUDGE_USER_PROMPT = """Evaluate this RAG interaction:

[USER QUERY]
{query}

[IS OUT OF SCOPE / FALSE PREMISE]
{is_out_of_scope_or_false}

[RETRIEVED CONTEXT]
{retrieved_context}

[GROUND TRUTH ANSWER]
{ground_truth_answer}

[GENERATED ANSWER]
{generated_answer}
"""

JUDGE_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", RAG_JUDGE_SYSTEM_PROMPT),
    ("human", RAG_JUDGE_USER_PROMPT)
])


def clean_and_parse_json(text: str) -> Dict[str, Any]:
    """Cleans potential markdown fences or surrounding commentary from LLM output."""
    cleaned = text.strip()
    # Remove markdown code fence if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback regex extraction for the JSON block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Failed to parse LLM judge response as JSON: {text[:200]}...")


def evaluate_rag_response(
    query: str,
    retrieved_context: str,
    ground_truth_answer: str,
    generated_answer: str,
    is_out_of_scope_or_false: bool = False
) -> Dict[str, Any]:
    """
    Executes the LLM-as-a-Judge against a single RAG query-answer pair.
    """
    llm = get_llm()
    prompt = JUDGE_PROMPT_TEMPLATE.format_messages(
        query=query,
        is_out_of_scope_or_false=str(is_out_of_scope_or_false),
        retrieved_context=retrieved_context if retrieved_context else "NO CONTEXT RETRIEVED",
        ground_truth_answer=ground_truth_answer,
        generated_answer=generated_answer
    )

    response = llm.invoke(prompt)
    raw_content = response.content if hasattr(response, "content") else str(response)

    parsed_result = clean_and_parse_json(raw_content)

    # Ensure averages and overall verdict are calibrated
    scores = [
        parsed_result.get("faithfulness", {}).get("score", 5),
        parsed_result.get("answer_relevancy", {}).get("score", 5),
        parsed_result.get("semantic_correctness", {}).get("score", 5),
        parsed_result.get("refusal_accuracy", {}).get("score", 5)
    ]
    avg_score = round(sum(scores) / len(scores), 2)
    parsed_result["overall_score_avg"] = avg_score
    parsed_result["verdict"] = "PASS" if avg_score >= 4.0 else "FAIL"

    return parsed_result
