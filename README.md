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

<div align="center">

<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/XGBoost-2.0-189AB4?style=flat-square"/>
<img src="https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=flat-square"/>
<img src="https://img.shields.io/badge/ChromaDB-Vector%20DB-8B5CF6?style=flat-square"/>
<img src="https://img.shields.io/badge/GPT--4o--mini-OpenAI-412991?style=flat-square&logo=openai&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/Tests-96%20passed-22C55E?style=flat-square"/>
<img src="https://img.shields.io/badge/Coverage-89%25-22C55E?style=flat-square"/>
<img src="https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?style=flat-square&logo=huggingface&logoColor=black"/>

<br/><br/>

# LoanLens 🔍

### AI-Powered Credit Risk Explainer Using RAG over CFPB Regulations

*Automatically generates regulation-grounded loan decline explanations by combining
XGBoost credit scoring, SHAP explainability, and RAG retrieval over 9,977 CFPB
regulatory chunks — in under 4 seconds per application.*

<br/>

[![Live Demo](https://img.shields.io/badge/Demo-Live%20on%20HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/AhmeduddinMohammed/loanlens)
[![CI](https://img.shields.io/badge/CI-passing-22C55E?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/MohammedAhmeduddin/loanlens/actions/workflows/ci.yml)

<br/>

</div>

---

## Live Demo

| Component | URL |
|---|---|
| 🎛️ **Dashboard** | [huggingface.co/spaces/AhmeduddinMohammed/loanlens](https://huggingface.co/spaces/AhmeduddinMohammed/loanlens) |
| 💻 **GitHub** | [github.com/MohammedAhmeduddin/loanlens](https://github.com/MohammedAhmeduddin/loanlens) |

**Try these demo scenarios in the dashboard:**

| Scenario | Income | Loan | Risk Score | Expected Result |
|---|---|---|---|---|
| High Risk | $24,000 | $380,000 | 89.5/100 | 🔴 **DECLINED** — low credit score, high debt |
| Medium Risk | $55,000 | $200,000 | 43.2/100 | 🟡 **REVIEW** — borderline creditworthiness |
| Low Risk | $120,000 | $180,000 | 18.6/100 | 🟢 **APPROVED** — strong financial profile |

---

## The Problem

Credit analysts at fintech lenders spend **3–5 hours per loan application** manually
cross-referencing borrower financial data against CFPB, ECOA, and FCRA regulatory
guidelines to write adverse action notices.

A wrong or incomplete explanation exposes the company to **fair lending lawsuits
costing $1M+** and CFPB enforcement actions.

**LoanLens replaces that process end-to-end.**

A single API call scores a borrower with XGBoost, extracts SHAP risk drivers,
retrieves the most relevant CFPB regulatory passages from ChromaDB, and generates
a legally-grounded adverse action notice — in **under 4 seconds on average**.

---

## Key Results

| Metric | Value |
|---|---|
| Validation AUC | **0.7700** |
| Test AUC | **0.7672** |
| KS Statistic | **0.4110** (> 0.30 = good) |
| Gini Coefficient | **0.5400** (> 0.50 = good) |
| Training samples | **215,257** (stratified 70/15/15 split) |
| Features engineered | **75** across 6 joined tables |
| Regulatory chunks | **9,977** CFPB chunks indexed in ChromaDB |
| RAG grounding score | **1.0** (phrase-level citation verification) |
| Generation time | **~3.5 seconds** end-to-end |
| Test coverage | **89.26%** |
| Total tests | **96** — unit + integration |

---

## System Architecture

```
Home Credit CSVs (307K rows, 6 tables)
        │
        ▼  dbt SQL transformations (4 staging views + feat_master)
PostgreSQL feature store — 307,511 rows × 75 features
        │
        ▼  XGBoost XGBClassifier (scale_pos_weight=11.39)
MLflow Model Registry — CreditScoringModel v1 → Production
        │
        ▼  SHAP TreeExplainer — top-5 risk factors per borrower
Feature Translator — internal names → CFPB adverse action codes
        │
        ▼  query construction from SHAP factors
ChromaDB Vector Store — 9,977 CFPB regulation chunks
│   sentence-transformers/all-MiniLM-L6-v2 (384-dim, CPU, free)
│   cosine similarity search → top-3 passages (threshold ≥ 0.30)
        │
        ▼  GPT-4o-mini — structured JSON output (function calling)
        │   grounding score verification
        │
        ▼  FastAPI REST endpoints
        │   GET  /health    POST /score    POST /explain    GET /model/info
        │
        ▼  Gradio analyst dashboard (HuggingFace Spaces)
```

### Why RAG and not fine-tuning?

The CFPB updates regulations regularly. RAG retrieves from the actual document —
every citation is traceable to a specific page. Fine-tuning bakes regulations into
weights with no auditability. For compliance use cases, grounded retrieval is
the only defensible approach.

---

## Pipeline Components

### 1 · XGBoost Credit Scoring Model
- Trained on **307,511 Home Credit loan applications** with early stopping
- **scale_pos_weight = 11.39** handles 8% default rate class imbalance
- 75 features engineered from 6 joined tables via dbt SQL
- MLflow tracks every run — hyperparameters, AUC, KS, Gini
- Champion/challenger registry: `None → Staging → Production`

Baseline progression tracked in MLflow:

| Run | Model | AUC |
|---|---|---|
| 1 | Logistic Regression baseline | 0.70 |
| 2 | XGBoost, no feature engineering | 0.74 |
| 3 | XGBoost + application features | 0.77 |
| 4 | XGBoost + all tables + tuning | **0.77** |

### 2 · SHAP Explainer
- **TreeExplainer** extracts per-borrower risk factors (not global importance)
- Top-5 features by absolute SHAP value returned per application
- Each factor mapped to official CFPB Regulation B adverse action codes:

| Feature | CFPB Code | Regulation |
|---|---|---|
| ext_source_mean | A9 - Credit score | FCRA |
| debt_to_income | A6 - Debt-to-income ratio | ECOA |
| bureau_overdue_count | A1 - Delinquent obligations | FCRA |
| employment_years | A13 - Length of employment | ECOA |
| late_payment_rate | A1 - Delinquent obligations | FCRA |

> **Note:** `age_years` maps to A8 (length of credit history), NOT age.
> Age is a prohibited basis under ECOA and cannot be cited as an adverse action reason.

### 3 · RAG Retrieval Pipeline
- **Document:** CFPB Supervision and Examination Manual (1,814 pages)
- **Chunking:** RecursiveCharacterTextSplitter — 512 tokens, 50 token overlap
- **Embedding:** sentence-transformers/all-MiniLM-L6-v2 — free, runs on CPU
- **Storage:** ChromaDB persistent store — 9,977 chunks
- **Retrieval:** cosine similarity, top-5 → filtered to top-3 (similarity ≥ 0.30)
- **Top scores achieved:** 0.7499, 0.7373, 0.7177

### 4 · GPT-4o-mini Generation
- **Function calling** enforces structured JSON output — no format hallucinations
- Temperature 0.1 for factual consistency in compliance text
- Grounding score verifies citations against retrieved passages
- **Grounding score: 1.0** on all benchmark runs

### 5 · PSI Drift Monitoring
- Population Stability Index computed on 10 key features weekly
- `PSI < 0.10` → stable, `0.10–0.20` → monitor, `> 0.20` → retrain
- Slack webhook alert triggered on severe drift
- GitHub Actions scheduled workflow every Monday 9AM UTC

---

## Tech Stack

| Layer | Technology | Justification |
|---|---|---|
| Credit Model | **XGBoost 2.0 + SHAP** | SOTA tabular, regulatory explainability requirement |
| RAG Pipeline | **LangChain + ChromaDB** | Document chunking + persistent vector store |
| Embeddings | **sentence-transformers/all-MiniLM-L6-v2** | Free, CPU-only, 384-dim, production-fast |
| LLM | **GPT-4o-mini + function calling** | Cheapest capable LLM, structured output |
| MLOps | **MLflow** | Experiment tracking + model registry |
| Data Pipeline | **dbt + PostgreSQL** | Modern data stack, SQL-native feature engineering |
| API | **FastAPI + Pydantic v2** | Typed endpoints, auto OpenAPI docs |
| Drift Monitoring | **PSI + Slack webhooks** | Industry-standard credit model monitoring |
| CI/CD | **GitHub Actions** | 96 tests + 85% coverage gate on every push |
| Dashboard | **Gradio** | Python-native, HuggingFace native deployment |
| Containers | **Docker + docker-compose** | PostgreSQL + MLflow + API orchestration |

---

## Feature Engineering

90+ features engineered across 6 joined tables using dbt SQL transformations:

**From `application_train.csv` (307K rows):**
- `debt_to_income` = amt_credit / amt_income_total — maps to CFPB A6
- `credit_to_annuity` = amt_credit / amt_annuity
- `income_per_person` = amt_income_total / cnt_fam_members
- `age_years`, `employment_years`, `employed_to_age_ratio`
- `ext_source_mean` = mean(EXT_SOURCE_1, 2, 3) — maps to CFPB A9

**From `bureau.csv` (1.7M rows, aggregated per customer):**
- `bureau_overdue_count`, `bureau_max_overdue_days` — maps to CFPB A1
- `bureau_debt_to_credit`, `bureau_active_loans`, `bureau_total_debt`

**From `previous_application.csv` (1.67M rows):**
- `prev_refused_count`, `prev_approval_rate`
- `prev_max_credit`, `prev_avg_credit`

**From `installments_payments.csv` (500K sample):**
- `late_payment_rate`, `avg_payment_ratio` — maps to CFPB A1
- `late_payment_count`, `avg_days_late`

---

## API Reference

### `POST /explain`

Runs the full pipeline — score + SHAP + RAG + generation.

```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{
    "amt_income_total": 24000,
    "amt_credit": 380000,
    "age_years": 23,
    "employment_years": 0.5,
    "ext_source_2": 0.12,
    "bureau_overdue_count": 4,
    "prev_refused_count": 3,
    "late_payment_rate": 0.6
  }'
```

**Response:**
```json
{
  "risk_score": 89.47,
  "probability": 0.8947,
  "decision": "decline",
  "shap_factors": [
    {
      "rank": 1,
      "feature": "ext_source_mean",
      "shap_value": -0.85,
      "direction": "increases_risk",
      "label": "External credit score assessment",
      "cfpb_code": "A9 - Credit score",
      "regulation": "FCRA"
    }
  ],
  "adverse_action_notice": "We regret to inform you that your application for credit has been declined per 12 CFR 1002.9(a)(2)...",
  "regulatory_basis": "12 CFR 1002.9(a)(2) — ECOA Regulation B; FCRA Section 615(a)",
  "grounding_score": 1.0,
  "generation_time_ms": 3458
}
```

### `POST /score`
Fast scoring only — no LLM call, < 50ms.

### `GET /health`
```json
{
  "status": "ok",
  "model_loaded": true,
  "chromadb_chunks": 9977,
  "version": "0.1.0"
}
```

---

## Project Structure

```
loanlens/
├── src/loanlens/
│   ├── api/
│   │   ├── main.py              # FastAPI app — lifespan pipeline loading
│   │   └── schemas.py           # Pydantic v2 request/response contracts
│   ├── dashboard/
│   │   └── app.py               # Gradio UI — precomputed + live API mode
│   ├── model/
│   │   ├── train.py             # XGBoost training + MLflow logging
│   │   ├── evaluate.py          # AUC, KS statistic, Gini coefficient
│   │   ├── explain.py           # SHAP TreeExplainer + CFPB code mapping
│   │   ├── registry.py          # MLflow champion/challenger promotion
│   │   └── data_loader.py       # PostgreSQL feature store loader
│   ├── rag/
│   │   ├── ingest.py            # PDF chunking + ChromaDB indexing
│   │   ├── retriever.py         # Cosine similarity search + filtering
│   │   ├── generator.py         # GPT-4o-mini + function calling
│   │   └── pipeline.py          # End-to-end orchestration
│   ├── monitoring/
│   │   ├── drift.py             # PSI computation (10 features)
│   │   └── alerts.py            # Slack webhook notifications
│   ├── utils/logging.py         # Loguru structured logging
│   └── config.py                # Pydantic settings — all env vars
├── dbt/loanlens/
│   └── models/
│       ├── staging/             # 4 views: application, bureau,
│       │                        #   previous_application, installments
│       └── features/            # feat_master — 307K rows × 75 features
├── tests/
│   ├── test_api.py              # 23 tests — endpoints, validation, 503 handling
│   ├── test_model.py            # 21 tests — data loader, metrics, SHAP
│   ├── test_rag.py              # 16 tests — generator, retriever, ingest
│   ├── test_pipeline.py         # 15 tests — full pipeline, decisions
│   ├── test_monitoring.py       # 12 tests — PSI, drift, Slack alerts
│   └── test_train_and_ingest.py #  9 tests — training, ChromaDB insert
├── scripts/
│   ├── load_data.py             # Chunked CSV → PostgreSQL (4.19M rows)
│   ├── run_drift_check.py       # Standalone PSI check + Slack alert
│   └── test_pipeline.py         # End-to-end integration smoke test
├── .github/workflows/
│   ├── ci.yml                   # 96 tests + 85% coverage gate on push
│   └── drift_check.yml          # Weekly scheduled PSI monitoring
├── app.py                       # HuggingFace Spaces entry point
├── docker-compose.yml           # PostgreSQL 16 + MLflow server
├── pyproject.toml               # All dependencies + pytest config
└── .env.example                 # Required environment variables
```

---

## CI/CD Pipeline

GitHub Actions runs on every push to `main`:

```
Push to main
    │
    ├── Job: test
    │   ├── Set up Python 3.11
    │   ├── pip install -e ".[dev]" + langchain-text-splitters + pypdf
    │   ├── pytest tests/ --cov=src/loanlens --cov-fail-under=85
    │   │   └── All OpenAI + MLflow + ChromaDB calls mocked (zero API cost)
    │   └── Upload coverage artifact
    │
    └── Weekly: drift_check.yml
        ├── Scheduled Monday 9AM UTC
        ├── Loads recent 10K feature rows from PostgreSQL
        ├── Computes PSI on 10 key features
        └── Sends Slack alert if PSI > 0.20
```

---

## Testing Strategy

All external dependencies mocked — OpenAI, MLflow, ChromaDB — zero API costs in CI.

```
tests/
├── test_api.py              23 tests — FastAPI endpoints, Pydantic validation,
│                                       503 when pipeline not loaded
├── test_model.py            21 tests — data loader, AUC/KS/Gini metrics,
│                                       SHAP factors, CFPB code mapping
├── test_rag.py              16 tests — generator JSON parsing, retriever
│                                       similarity filtering, ingest chunking
├── test_pipeline.py         15 tests — score/review/approve decisions,
│                                       explain skips generation for approvals
├── test_monitoring.py       12 tests — PSI formula, drift detection,
│                                       Slack webhook sending
└── test_train_and_ingest.py  9 tests — MLflow param/metric logging,
                                        batch ChromaDB insert
```

---

## Quickstart

```bash
# 1. Clone and setup
git clone https://github.com/MohammedAhmeduddin/loanlens
cd loanlens
cp .env.example .env           # add OPENAI_API_KEY and POSTGRES_PASSWORD

# 2. Start infrastructure
docker compose up -d           # PostgreSQL 16 + MLflow server

# 3. Download Home Credit data from Kaggle
# kaggle.com/c/home-credit-default-risk/data — place CSVs in data/raw/

# 4. Load raw data
python scripts/load_data.py    # loads 4.19M rows across 4 tables

# 5. Build feature store
cd dbt/loanlens && dbt run && cd ../..

# 6. Train credit scoring model
python src/loanlens/model/train.py   # val AUC ~0.77, logged to MLflow

# 7. Index CFPB regulations
python src/loanlens/rag/ingest.py    # 9,977 chunks into ChromaDB

# 8. Start API
TOKENIZERS_PARALLELISM=false uvicorn src.loanlens.api.main:app --port 8000

# 9. Launch dashboard
python src/loanlens/dashboard/app.py
# Open http://localhost:7860
```

---

## Limitations and Future Work

| Area | Current State | Planned |
|---|---|---|
| **RAG evaluation** | Grounding score only | RAGAS faithfulness + context precision |
| **Regulatory coverage** | 1 PDF (CFPB manual) | FCRA + ECOA source documents |
| **Embedding model** | MiniLM (general purpose) | Legal-domain fine-tuned embeddings |
| **Retrieval** | Dense cosine similarity | Hybrid sparse+dense (BM25 + vector) |
| **AUC ceiling** | 0.77 with 75 features | credit_card_balance.csv adds ~5 more features |
| **Deployment** | Local + HuggingFace UI | FastAPI on GCP Cloud Run |

---

## Resume Bullets

```
Built RAG-powered credit explainability system over 9,977 CFPB regulatory chunks
using ChromaDB and GPT-4o-mini, generating compliant adverse action notices in
under 4 seconds vs 3-5 hour manual analyst process — grounding score 1.0

Trained XGBoost credit scoring model on 307K loan applications achieving val AUC
0.77, KS 0.41, Gini 0.54 with SHAP TreeExplainer providing top-5 per-borrower
risk factor attribution mapped to CFPB Regulation B adverse action codes

Engineered 90+ financial features using dbt SQL transformations across 6 joined
tables; deployed MLflow champion/challenger registry with automated PSI drift
monitoring on 10 key features and Slack alerting

Achieved 89% test coverage across 96 pytest unit and integration tests with
GitHub Actions CI/CD pipeline — all OpenAI and ML model calls mocked for
cost-free CI execution
```

---

## STAR Interview Story

**Situation:** Credit analysts at lending companies spend 3–5 hours manually
justifying loan decline decisions against CFPB regulations, creating compliance
risk and inconsistency across analysts.

**Task:** Build a system that automatically generates regulation-grounded
explanations by combining a credit scoring model with a RAG pipeline over CFPB
documents, meeting ECOA and FCRA adverse action notice requirements.

**Action:** Trained XGBoost on 307K applications, extracted SHAP risk drivers
per borrower, built a feature-to-regulation translation layer using official CFPB
Regulation B codes, indexed 9,977 regulatory chunks in ChromaDB, and served the
full pipeline via FastAPI with MLflow model versioning and PSI drift alerting.

**Result:** System generates structured compliant explanations in under 4 seconds,
achieves grounding score of 1.0 on all benchmark runs, and MLflow registry tracks
every model version with automated drift alerts triggered when PSI exceeds 0.20.

---

## Author

**Ahmeduddin Mohammed** 

- GitHub: [@MohammedAhmeduddin](https://github.com/MohammedAhmeduddin)
- LinkedIn: [linkedin.com/in/ahmeduddinmohammed](https://linkedin.com/in/ahmeduddinmohammed)

---

<div align="center">
<sub>Built with XGBoost · SHAP · LangChain · ChromaDB · GPT-4o-mini · FastAPI · dbt · MLflow</sub>
</div>
