# 📑 Enterprise HR RAG Evaluation Report

- **Generated Date**: 2026-09-20T22:44:20.042814
- **Generator LLM**: `openai/gpt-oss-120b`
- **Retriever Top-K**: `4`
- **Total Test Cases**: `40`
- **Evaluation Duration**: `910.74s`

---

## 🏆 Overall Quality Score

### **4.36 / 5.00 (87.2%)**
- **Pass Rate**: **85.0%** (34 of 40 cases passed threshold >= 4.0)

---

## 📊 Core Evaluation Rubrics

| Metric | Average Score (1-5) | Normalized % | Target Threshold | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Faithfulness / Groundedness** | 4.3 / 5.0 | 86.0% | >= 90% | ✅ PASS |
| **Answer Relevancy** | 4.4 / 5.0 | 88.0% | >= 90% | ✅ PASS |
| **Semantic Correctness** | 4.35 / 5.0 | 87.0% | >= 90% | ✅ PASS |
| **Refusal & Guardrail Accuracy** | 4.4 / 5.0 | 88.0% | >= 90% | ✅ PASS |

---

## 📂 Category Breakdown

| Category | Count | Faithfulness | Relevancy | Correctness | Refusal | Pass Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **In-Scope Factual** | 19 | 4.95 | 5.0 | 4.89 | 5.0 | 100.0% |
| **Numerical & Tenure Calculation** | 2 | 5.0 | 5.0 | 5.0 | 5.0 | 100.0% |
| **Multi-Hop Cross-Chapter** | 4 | 2.5 | 3.0 | 3.0 | 3.0 | 50.0% |
| **Classification Disambiguation** | 2 | 3.0 | 3.0 | 3.0 | 3.0 | 50.0% |
| **False Premise & Out-of-Scope** | 13 | 4.0 | 4.08 | 4.08 | 4.08 | 76.9% |

---

## 🔍 Sample Evaluated Cases

### [RAG_TC_001] In-Scope Factual (PASS)
- **Query**: Can the company reimburse me for buying a standing desk or ergonomic chair for my home setup?
- **Is False / Out-of-Scope**: `False`
- **Generated Answer**: Yes. According to **Chapter 5, Section 5.4 – Remote Workstation & Ergonomic Standards**, remote employees are eligible for a **one‑time home‑office setup reimbursement of up to $1,000**. This reimbursement can be used for items such as **desks, ergonomic chairs, and external monitors**.
- **Ground Truth**: Yes, remote employees are eligible for a one-time home office setup reimbursement of up to $1,000 for desks, ergonomic chairs, and external monitors.
- **Judge Verdict**: Avg Score `5.0/5.0`
  - *Faithfulness*: `5/5` — All statements in the answer are directly supported by Chunk 1 of the retrieved context with no extra or missing information.
  - *Relevancy*: `5/5` — The answer directly addresses the user's question about reimbursement for a standing desk or ergonomic chair, providing the relevant policy details.
  - *Semantic Correctness*: `5/5` — The answer matches the ground truth verbatim, conveying the same facts and limits.
  - *Refusal Accuracy*: `5/5` — The question is in scope; the answer correctly provides the information without any refusal.

---
### [RAG_TC_002] Numerical & Tenure Calculation (PASS)
- **Query**: How many days of paid vacation do I accrue annually if I have 4 years of continuous tenure, and what is my maximum accrual cap?
- **Is False / Out-of-Scope**: `False`
- **Generated Answer**: Based on **Chapter 10: Paid Time Off (PTO) & Holidays**:

- **Annual vacation accrual** for employees with **3 to 5 years of continuous tenure** is **25 days per year** (2.08 days per month).  
- The **maximum accrual cap** for this tenure band is **37.5 days**.

**Reference:** Excerpt 1, Page 33 – “10.1 Annual Vacation Accrual Rates.”
- **Ground Truth**: Employees with 3 to 5 years of continuous tenure accrue 25 days of paid vacation per year (2.08 days/month), with a maximum accrual cap of 37.5 days.
- **Judge Verdict**: Avg Score `5.0/5.0`
  - *Faithfulness*: `5/5` — All statements (25 days per year, 2.08 days/month, 37.5 day cap) are directly supported by Chunk 1 of the retrieved policy.
  - *Relevancy*: `5/5` — The response directly answers both parts of the query (annual accrual and maximum cap) without extraneous information.
  - *Semantic Correctness*: `5/5` — The answer matches the ground truth exactly in numbers and conditions.
  - *Refusal Accuracy*: `5/5` — The question is in scope and the model correctly provides the answer without any refusal.

---
### [RAG_TC_003] Multi-Hop Cross-Chapter (PASS)
- **Query**: If I take primary caregiver parental leave and then use the phased return program, how does my schedule and salary work during those first few weeks back?
- **Is False / Out-of-Scope**: `False`
- **Generated Answer**: **Schedule and salary during the first 4 weeks after you return from primary‑caregiver parental leave**

| Item | What the policy says | How it applies to you |
|------|----------------------|-----------------------|
| **Leave taken** | Primary caregivers receive **16 weeks of paid leave at 100 % of base salary**. (Chapter 11, 11.1) | You will have completed up to 16 weeks of fully‑paid leave. |
| **Phased return‑to‑work** | “New parents may work an **80 % part‑time schedule (4 days/week) at 100 % pay** for the **first 4 weeks** following return from parental leave.” (Chapter 11, 11.2) | When you come back, you will work **four days each week** (i.e., 80 % of a full‑time schedule) **but will continue to be paid at 100 % of your base salary** for those four weeks. |
| **Duration of the phased schedule** | Fixed at **4 weeks** after the leave ends. | After the fourth week, you would transition back to the standard full‑time schedule (5 days/week) and continue at 100 % salary as normal. |

**Summary**

- After your 16‑week primary‑caregiver parental leave (paid at full salary), you may elect the phased return‑to‑work program.
- For the **first four weeks back**, you work **four days per week** (80 % of a normal workweek) **while still receiving 100 % of your base salary**.
- This arrangement lasts **exactly four weeks**; thereafter you resume the regular full‑time schedule and pay.

*References: Chapter 11 – Parental & Caregiver Leaves, Sections 11.1 (Primary Caregiver Parental Leave) and 11.2 (Phased Return‑to‑Work Program).*
- **Ground Truth**: Primary caregivers receive 16 weeks of 100% paid parental leave. Upon returning, the phased return-to-work program allows working an 80% part-time schedule (4 days/week) at 100% full pay for the first 4 weeks.
- **Judge Verdict**: Avg Score `4.75/5.0`
  - *Faithfulness*: `4/5` — All core facts (16 weeks paid leave, 80% schedule, 100% pay for first 4 weeks) are directly supported by the retrieved context. The statement about returning to full‑time after the 4‑week period is a logical inference not explicitly stated, causing a minor extrapolation.
  - *Relevancy*: `5/5` — The answer directly addresses the user's request about schedule and salary during the first weeks back, providing precise details without irrelevant information.
  - *Semantic Correctness*: `5/5` — The generated answer matches the ground truth verbatim in facts, numbers, and conclusions.
  - *Refusal Accuracy*: `5/5` — The question is in scope and the model correctly provides the answer without any unnecessary refusal.

---
### [RAG_TC_004] Classification Disambiguation (PASS)
- **Query**: What is the resignation notice period required for engineering managers and directors versus individual contributors?
- **Is False / Out-of-Scope**: `False`
- **Generated Answer**: The HR manual specifies two notice‑period categories:

- **Individual Contributors (IC1 – IC5):** minimum **30 calendar days** notice.  
- **Managers, Directors, and Executive Staff (including engineering managers and directors):** minimum **60 calendar days** notice.  

**Reference:** CHAPTER 15 – “Voluntary Resignation Notice Periods,” page 48.
- **Ground Truth**: Individual contributors (IC1-IC5) must provide a minimum of 30 calendar days notice, while Managers, Directors, and Executive Staff must provide a minimum of 60 calendar days notice.
- **Judge Verdict**: Avg Score `5.0/5.0`
  - *Faithfulness*: `5/5` — All statements about notice periods exactly match the retrieved policy text; no hallucinations.
  - *Relevancy*: `5/5` — Directly answers the user's specific comparison of notice periods for engineering managers/directors and individual contributors.
  - *Semantic Correctness*: `5/5` — Matches the ground‑truth answer word‑for‑word; facts and numbers are identical.
  - *Refusal Accuracy*: `5/5` — Question is in‑scope and the model provides the correct answer without any unnecessary refusal.

---
### [RAG_TC_005] Multi-Hop Cross-Chapter (PASS)
- **Query**: What is the daily meal per diem for international business travel, and what flight class can I book for an 8-hour international flight?
- **Is False / Out-of-Scope**: `False`
- **Generated Answer**: - **Daily meal per‑diem for international business travel:** $100 per day (before taxes). *See Chapter 12, Section 12.4 (Meal & Incidentals Per Diem) and the “Meal Travel Per Diem” line in the Quick Reference Limits table (Excerpt 3, Page 50).*

- **Flight class you may book for an 8‑hour international flight:** **Premium Economy**. International flights longer than 6 hours are limited to Premium Economy (Business Class requires >10 hours and VP‑level sign‑off). *See Chapter 12, Section 12.2 (Air Travel Cabin Class Rules) (Excerpt 2, Page 40).*
- **Ground Truth**: The daily meal per diem for international travel is up to $100/day. For an 8-hour international flight (exceeding 6 hours), you are eligible to book Premium Economy class.
- **Judge Verdict**: Avg Score `4.75/5.0`
  - *Faithfulness*: `4/5` — All core facts ( $100 international per diem and Premium Economy for >6h flights) are supported by the retrieved context. The added phrase "before taxes" is not in the source but does not contradict the policy, representing minor extrapolation.
  - *Relevancy*: `5/5` — The response directly answers both parts of the user query: the daily meal per diem and the allowable flight class for an 8‑hour international flight.
  - *Semantic Correctness*: `5/5` — The answer matches the ground truth exactly in both amount ($100) and flight class (Premium Economy).
  - *Refusal Accuracy*: `5/5` — The question is in‑scope and the model provides a complete answer without any unnecessary refusal.

---
### [RAG_TC_006] In-Scope Factual (PASS)
- **Query**: During what specific hours must both remote and in-office employees be accessible on Slack and Google Workspace?
- **Is False / Out-of-Scope**: `False`
- **Generated Answer**: Both remote and in‑office employees must be reachable on Slack and Google Workspace during **Core Collaboration Hours: 10:00 AM – 4:00 PM** in their respective home time zone. 【Excerpt 1, CHAPTER 5 §5.2】
- **Ground Truth**: Employees must be accessible during Core Collaboration Hours: 10:00 AM to 4:00 PM in their respective home time zone.
- **Judge Verdict**: Avg Score `5.0/5.0`
  - *Faithfulness*: `5/5` — All statements exactly match the retrieved policy text; no hallucinations.
  - *Relevancy*: `5/5` — Directly answers the user's question about the required accessibility hours.
  - *Semantic Correctness*: `5/5` — Identical to the ground‑truth answer in facts, wording, and conclusion.
  - *Refusal Accuracy*: `5/5` — Question is in‑scope and the model provides the correct answer without any refusal.

---
