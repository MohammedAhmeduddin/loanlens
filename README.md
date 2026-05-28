---
title: LoanLens
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.9.1"
python_version: "3.11"
app_file: app.py
pinned: false
---

# LoanLens 🔍

### AI-Powered Credit Risk Explainer Using RAG over CFPB Regulations

[![LoanLens CI](https://github.com/MohammedAhmeduddin/loanlens/actions/workflows/ci.yml/badge.svg)](https://github.com/MohammedAhmeduddin/loanlens/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Tests](https://img.shields.io/badge/tests-96%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-89%25-green)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## The Problem (2 sentences a CFO understands)

Credit analysts at fintech lenders spend **3-5 hours per loan application** manually
cross-referencing borrower data against CFPB regulations to write decline explanations.
A wrong or incomplete adverse action explanation exposes the company to **fair lending
lawsuits costing $1M+**.

---

## Live Demo

**[HuggingFace Spaces](https://huggingface.co/spaces/AhmeduddinMohammed/loanlens)**

---

## What Makes This Different

Every DS portfolio has a credit scoring model. LoanLens adds a **RAG compliance layer**
on top — connecting XGBoost decisions to actual CFPB regulatory text through vector
retrieval, generating legally-grounded adverse action notices in under 4 seconds.
Borrower Data
|
v
XGBoost (val AUC 0.77) --> SHAP top-5 risk factors
|
v
Feature Translator (CFPB adverse action codes)
|
v
ChromaDB query (9,977 CFPB chunks)
cosine similarity retrieval
|
v
GPT-4o-mini --> ECOA-compliant adverse action notice
grounding score: 1.0

---

## System Architecture

Home Credit CSVs (307K rows, 6 tables)
|
v dbt SQL transformations
PostgreSQL feature store (90+ engineered features)
|
v XGBoost XGBClassifier
MLflow Model Registry (CreditScoringModel v1 -> Production)
|
v SHAP TreeExplainer (top-5 risk factors per borrower)
|
v Feature Translator (internal names -> CFPB regulatory language)
|
v ChromaDB Vector Store (9,977 CFPB regulation chunks)
| sentence-transformers/all-MiniLM-L6-v2 embeddings
| cosine similarity search
|
v GPT-4o-mini (structured JSON output, function calling)
| grounding score verification
|
v FastAPI REST endpoints (/score, /explain, /health, /model/info)
|
v Gradio analyst dashboard (HuggingFace Spaces)

---

## Quickstart

```bash
# 1. Clone and setup
git clone https://github.com/MohammedAhmeduddin/loanlens
cd loanlens
cp .env.example .env       # add OPENAI_API_KEY and POSTGRES_PASSWORD

# 2. Start infrastructure
docker compose up -d       # PostgreSQL + MLflow

# 3. Load data (Home Credit dataset from Kaggle)
python scripts/load_data.py

# 4. Build feature store
cd dbt/loanlens && dbt run && cd ../..

# 5. Train credit scoring model
python src/loanlens/model/train.py

# 6. Index CFPB regulations
python src/loanlens/rag/ingest.py

# 7. Start API
TOKENIZERS_PARALLELISM=false uvicorn src.loanlens.api.main:app --port 8000

# 8. Launch dashboard
python src/loanlens/dashboard/app.py
# Open http://localhost:7860
```

---

## Tech Stack

| Layer        | Technology                                 | Purpose                             |
| ------------ | ------------------------------------------ | ----------------------------------- |
| Credit Model | XGBoost, SHAP, scikit-learn                | Default prediction + explainability |
| RAG Pipeline | LangChain, ChromaDB, sentence-transformers | Regulatory text retrieval           |
| LLM          | GPT-4o-mini (OpenAI)                       | Adverse action notice generation    |
| MLOps        | MLflow, PSI drift detection                | Model versioning + monitoring       |
| Data         | dbt, PostgreSQL, pandas                    | Feature engineering pipeline        |
| API          | FastAPI, Pydantic, uvicorn                 | Production REST endpoints           |
| Testing      | pytest, 96 tests, 89% coverage             | Quality assurance                   |
| Deploy       | Docker, GitHub Actions, HuggingFace Spaces | CI/CD + demo                        |

---

## Model Performance

| Metric           | Value   | Industry Benchmark     |
| ---------------- | ------- | ---------------------- |
| Validation AUC   | 0.7700  | > 0.75 acceptable      |
| Test AUC         | 0.7672  | Consistent with val    |
| KS Statistic     | 0.4110  | > 0.30 good            |
| Gini Coefficient | 0.5400  | > 0.50 good            |
| Training samples | 215,257 |                        |
| Features         | 75      |                        |
| Best iteration   | 432     | Early stopping applied |

Baseline progression tracked in MLflow:

- Logistic Regression baseline: AUC 0.70
- XGBoost no feature engineering: AUC 0.74
- XGBoost + application features: AUC 0.77
- XGBoost + all tables + tuning: AUC 0.77

---

## RAG Pipeline Quality

| Metric              | Value                                              |
| ------------------- | -------------------------------------------------- |
| Knowledge base      | 9,977 CFPB regulatory chunks                       |
| Embedding model     | sentence-transformers/all-MiniLM-L6-v2 (free, CPU) |
| Retrieval           | Cosine similarity, top-3 passages                  |
| Top retrieval score | 0.75 (supervision_manual.pdf)                      |
| Grounding score     | 1.0 (phrase-level verification)                    |
| Generation time     | ~3.5 seconds                                       |
| Regulatory source   | CFPB Supervision and Examination Manual            |

---

## Feature Engineering

90+ features engineered across 6 joined tables using dbt SQL:

**From application_train.csv:**

- debt_to_income, credit_to_annuity, income_per_person
- age_years, employment_years, employed_to_age_ratio
- ext_source_mean (average of 3 external credit scores)

**From bureau.csv (aggregated per customer):**

- bureau_overdue_count, bureau_max_overdue_days
- bureau_debt_to_credit, bureau_active_loans
- bureau_total_debt, bureau_credit_type_count

**From previous_application.csv:**

- prev_refused_count, prev_approval_rate
- prev_max_credit, prev_avg_credit

**From installments_payments.csv:**

- late_payment_rate, avg_payment_ratio
- late_payment_count, avg_days_late

---

## API Endpoints

GET /health — Service status + model loaded + ChromaDB chunks
POST /score — Credit score only (fast, no LLM)
POST /explain — Full RAG explanation (score + SHAP + notice)
GET /model/info — Production model metadata from MLflow
Example /explain response:

```json
{
  "risk_score": 89.5,
  "decision": "decline",
  "shap_factors": [...],
  "adverse_action_notice": "We regret to inform you...",
  "regulatory_basis": "12 CFR 1002.9(a)(2)",
  "grounding_score": 1.0,
  "generation_time_ms": 3458
}
```

---

## MLOps Pipeline

- **MLflow** tracks every training run — hyperparameters, AUC, KS, Gini
- **Champion/challenger registry** — None -> Staging -> Production workflow
- **PSI drift detection** on 10 key features — Slack alert when PSI > 0.20
- **GitHub Actions CI** — 96 tests on every push, 85% coverage threshold
- **Weekly drift check** — scheduled GitHub Actions workflow

---

## Regulatory Compliance Approach

LoanLens maps model features to official CFPB Regulation B adverse action codes:

| Feature              | CFPB Code                   | Regulation |
| -------------------- | --------------------------- | ---------- |
| ext_source_mean      | A9 - Credit score           | FCRA       |
| debt_to_income       | A6 - Debt-to-income ratio   | ECOA       |
| bureau_overdue_count | A1 - Delinquent obligations | FCRA       |
| employment_years     | A13 - Length of employment  | ECOA       |
| late_payment_rate    | A1 - Delinquent obligations | FCRA       |

Note: age_years maps to A8 (length of credit history), NOT age — age is a
prohibited basis under ECOA and cannot be cited as an adverse action reason.

---

## Dataset

**Home Credit Default Risk** (Kaggle)

- 307,511 loan applications
- 6 related tables (bureau, installments, previous applications, credit card)
- 8% default rate (class imbalance handled with scale_pos_weight)
- Real production-like data with missing values and mixed types

**CFPB Knowledge Base**

- CFPB Supervision and Examination Manual (1,814 pages)
- 9,977 chunks after RecursiveCharacterTextSplitter (512 tokens, 50 overlap)
- Free public government document — no licensing issues

---

## Project Structure

![alt text](image.png)

## Resume Bullets

- Built RAG-powered credit explainability system over 9,977 CFPB regulatory chunks
  using ChromaDB and GPT-4o-mini, generating compliant adverse action notices in
  under 4 seconds vs 3-5 hour manual analyst process

- Trained XGBoost credit scoring model on 307K loan applications achieving val AUC
  0.77, KS 0.41, Gini 0.54 with SHAP TreeExplainer providing top-5 risk factor
  attribution meeting ECOA/FCRA adverse action requirements

- Engineered 90+ financial features using dbt SQL transformations across 6 joined
  tables; deployed MLflow champion/challenger registry with automated PSI drift
  monitoring on 10 key features

- Achieved 89% test coverage across 96 pytest unit and integration tests with
  GitHub Actions CI/CD pipeline — all OpenAI and ML model calls mocked

---

## STAR Interview Story

**Situation:** Credit analysts at lending companies spend hours manually justifying
loan decline decisions against regulations, creating compliance risk and inconsistency.

**Task:** Build a system that automatically generates regulation-grounded explanations
by combining a credit scoring model with a RAG pipeline over CFPB documents.

**Action:** Trained XGBoost on 307K applications, extracted SHAP risk drivers,
built LangChain RAG chain over 9,977 regulatory chunks served via FastAPI with
MLflow model versioning and PSI drift alerting.

**Result:** System generates structured compliant explanations in under 4 seconds,
achieves grounding score of 1.0 (all citations traceable to retrieved regulatory
text), and MLflow registry tracks every model version with automated drift alerts.

---

## Contact

**Ahmeduddin Mohammed** | Newark, NJ | MS Computer Science, NJIT

[GitHub](https://github.com/MohammedAhmeduddin) |
[LinkedIn](https://linkedin.com/in/ahmeduddinmohammed) |
[HuggingFace](https://huggingface.co/AhmeduddinMohammed)
