"""
LoanLens Gradio Dashboard.
Analyst-facing UI for credit risk explanation.
Deployed on HuggingFace Spaces.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import gradio as gr
import httpx
import json
from pathlib import Path

# API URL — override with environment variable for production
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Pre-computed demo samples — works without live API
DEMO_SAMPLES = [
    {
        "label": "High Risk — Young borrower, high debt",
        "inputs": {
            "amt_income_total": 24000,
            "amt_credit": 380000,
            "amt_annuity": 19000,
            "age_years": 23,
            "employment_years": 0.5,
            "ext_source_2": 0.12,
            "bureau_overdue_count": 4,
            "prev_refused_count": 3,
            "late_payment_rate": 0.6,
            "bureau_debt_to_credit": 0.92,
        }
    },
    {
        "label": "Medium Risk — Average borrower",
        "inputs": {
            "amt_income_total": 55000,
            "amt_credit": 200000,
            "amt_annuity": 10000,
            "age_years": 38,
            "employment_years": 4.0,
            "ext_source_2": 0.42,
            "bureau_overdue_count": 1,
            "prev_refused_count": 1,
            "late_payment_rate": 0.1,
            "bureau_debt_to_credit": 0.4,
        }
    },
    {
        "label": "Low Risk — Stable borrower",
        "inputs": {
            "amt_income_total": 120000,
            "amt_credit": 180000,
            "amt_annuity": 9000,
            "age_years": 45,
            "employment_years": 12.0,
            "ext_source_2": 0.74,
            "bureau_overdue_count": 0,
            "prev_refused_count": 0,
            "late_payment_rate": 0.0,
            "bureau_debt_to_credit": 0.15,
        }
    },
]


def analyze_application(
    income: float,
    loan_amount: float,
    annuity: float,
    age: float,
    employment_years: float,
    ext_source_2: float,
    bureau_overdue: int,
    prev_refused: int,
    late_payment_rate: float,
    bureau_debt_ratio: float,
) -> tuple:
    """Call LoanLens API and return formatted results."""

    payload = {
        "amt_income_total": income,
        "amt_credit": loan_amount,
        "amt_annuity": annuity,
        "age_years": age,
        "employment_years": employment_years,
        "ext_source_2": ext_source_2,
        "bureau_overdue_count": int(bureau_overdue),
        "prev_refused_count": int(prev_refused),
        "late_payment_rate": late_payment_rate,
        "bureau_debt_to_credit": bureau_debt_ratio,
        "cnt_children": 0,
    }

    try:
        response = httpx.post(
            f"{API_URL}/explain",
            json=payload,
            timeout=30.0
        )
        response.raise_for_status()
        result = response.json()

    except httpx.ConnectError:
        return (
            "⚠️ API Offline",
            "Cannot connect to LoanLens API. Please ensure the backend is running.",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
        )
    except Exception as e:
        return (
            "❌ Error",
            str(e),
            "N/A", "N/A", "N/A", "N/A",
        )

    # Format decision badge
    decision = result["decision"]
    risk_score = result["risk_score"]

    if decision == "decline":
        decision_display = f"🔴 DECLINED — Risk Score: {risk_score:.1f}/100"
    elif decision == "review":
        decision_display = f"🟡 REVIEW REQUIRED — Risk Score: {risk_score:.1f}/100"
    else:
        decision_display = f"🟢 APPROVED — Risk Score: {risk_score:.1f}/100"

    # Format SHAP factors
    shap_lines = []
    for f in result.get("shap_factors", []):
        direction = "↑ risk" if f["direction"] == "increases_risk" else "↓ risk"
        shap_lines.append(
            f"{f['rank']}. {f['label']}\n"
            f"   SHAP: {f['shap_value']:.4f} ({direction})\n"
            f"   Code: {f['cfpb_code']}"
        )
    shap_display = "\n\n".join(shap_lines) if shap_lines else "No factors available"

    # Adverse action notice
    notice = result.get("adverse_action_notice") or "No adverse action notice generated."

    # Regulatory citations
    passages = result.get("retrieved_passages", [])
    citations = "\n\n".join([
        f"📄 {p['source']} (p.{p['page']}) — Relevance: {p['similarity_score']:.2%}\n{p['text'][:200]}..."
        for p in passages
    ]) if passages else "No citations available"

    # Grounding score
    grounding = result.get("grounding_score")
    gen_time = result.get("generation_time_ms", 0)
    metadata = (
        f"Grounding Score: {grounding:.0%}\n"
        f"Generation Time: {gen_time}ms\n"
        f"Model: CreditScoringModel v1\n"
        f"Regulatory Basis: {result.get('regulatory_basis', 'N/A')}"
    ) if grounding is not None else "Approved — no explanation required"

    return (
        decision_display,
        shap_display,
        notice,
        citations,
        metadata,
        f"Applicant Rights: {result.get('applicant_rights', 'N/A')}",
    )


def load_demo_sample(sample_label: str) -> tuple:
    """Load a pre-defined demo sample."""
    for sample in DEMO_SAMPLES:
        if sample["label"] == sample_label:
            s = sample["inputs"]
            return (
                s["amt_income_total"],
                s["amt_credit"],
                s["amt_annuity"],
                s["age_years"],
                s["employment_years"],
                s["ext_source_2"],
                s["bureau_overdue_count"],
                s["prev_refused_count"],
                s["late_payment_rate"],
                s["bureau_debt_to_credit"],
            )
    return (75000, 200000, 10000, 35, 5.0, 0.5, 0, 0, 0.0, 0.2)


# ── Build Gradio Interface ─────────────────────────────────────

with gr.Blocks(
    title="LoanLens — Credit Risk Explainer",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown("""
    # 🔍 LoanLens
    ### AI-Powered Credit Risk Explainer | RAG over CFPB Regulations
    
    Enter borrower details to generate a regulation-grounded decline explanation.
    Uses XGBoost credit scoring + SHAP explainability + ChromaDB retrieval + GPT-4o-mini.
    """)

    # Demo sample loader
    with gr.Row():
        sample_dropdown = gr.Dropdown(
            choices=[s["label"] for s in DEMO_SAMPLES],
            label="📋 Load Demo Sample",
            value=DEMO_SAMPLES[0]["label"],
        )
        load_btn = gr.Button("Load Sample", variant="secondary", scale=0)

    gr.Markdown("---")

    with gr.Row():
        # Left column — inputs
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Borrower Information")

            income = gr.Number(
                label="Annual Income ($)",
                value=45000,
                minimum=1000,
            )
            loan_amount = gr.Number(
                label="Loan Amount Requested ($)",
                value=250000,
                minimum=1000,
            )
            annuity = gr.Number(
                label="Annual Loan Payment ($)",
                value=12000,
                minimum=100,
            )
            age = gr.Slider(
                label="Age (years)",
                minimum=18, maximum=75, value=35, step=1,
            )
            employment_years = gr.Slider(
                label="Years Employed",
                minimum=0, maximum=40, value=3.0, step=0.5,
            )
            ext_source_2 = gr.Slider(
                label="External Credit Score (0=poor, 1=excellent)",
                minimum=0.0, maximum=1.0, value=0.35, step=0.01,
            )
            bureau_overdue = gr.Slider(
                label="Overdue Accounts in Credit History",
                minimum=0, maximum=20, value=1, step=1,
            )
            prev_refused = gr.Slider(
                label="Previous Loan Applications Refused",
                minimum=0, maximum=10, value=1, step=1,
            )
            late_payment_rate = gr.Slider(
                label="Late Payment Rate (0=never, 1=always)",
                minimum=0.0, maximum=1.0, value=0.1, step=0.01,
            )
            bureau_debt_ratio = gr.Slider(
                label="Bureau Debt-to-Credit Ratio",
                minimum=0.0, maximum=1.0, value=0.3, step=0.01,
            )

            analyze_btn = gr.Button(
                "🔍 Analyze Application",
                variant="primary",
                size="lg",
            )

        # Right column — outputs
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Risk Assessment")

            decision_output = gr.Textbox(
                label="Decision",
                interactive=False,
                lines=1,
            )
            shap_output = gr.Textbox(
                label="Top Risk Factors (SHAP Analysis)",
                interactive=False,
                lines=10,
            )
            notice_output = gr.Textbox(
                label="📄 Adverse Action Notice (ECOA Compliant)",
                interactive=False,
                lines=5,
            )
            citations_output = gr.Textbox(
                label="📚 Regulatory Citations (Retrieved from CFPB Manual)",
                interactive=False,
                lines=8,
            )
            metadata_output = gr.Textbox(
                label="📈 Model Metadata",
                interactive=False,
                lines=4,
            )
            rights_output = gr.Textbox(
                label="⚖️ Applicant Rights",
                interactive=False,
                lines=2,
            )

    # Wire up buttons
    all_inputs = [
        income, loan_amount, annuity, age,
        employment_years, ext_source_2,
        bureau_overdue, prev_refused,
        late_payment_rate, bureau_debt_ratio,
    ]
    all_outputs = [
        decision_output, shap_output,
        notice_output, citations_output,
        metadata_output, rights_output,
    ]

    analyze_btn.click(
        fn=analyze_application,
        inputs=all_inputs,
        outputs=all_outputs,
    )

    load_btn.click(
        fn=load_demo_sample,
        inputs=[sample_dropdown],
        outputs=all_inputs,
    )

    gr.Markdown("""
    ---
    **Tech Stack:** XGBoost · SHAP · LangChain · ChromaDB · GPT-4o-mini · FastAPI · MLflow
    
    **Regulatory Coverage:** CFPB Supervision Manual (9,977 indexed chunks)
    
    [GitHub](https://github.com/MohammedAhmeduddin/loanlens) | Built by Ahmeduddin Mohammed
    """)


if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
