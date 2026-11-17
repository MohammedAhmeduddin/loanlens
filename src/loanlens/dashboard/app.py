import os
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loanlens.dashboard.samples import (  # noqa: E402
    SAMPLES,
    build_payload,
    format_result,
    load_demo_samples,
)

API_URL = os.getenv("API_URL", "")
DEMO = load_demo_samples()

NO_API_MESSAGE = (
    "This public demo has no live API, so it can only show the stored results "
    "for the three unchanged samples. Load a sample to see real pipeline output, "
    "or run the API locally (see README) to score any input."
)


def analyze(income, loan, annuity, emp, ext, overdue, refused, late, debt, sample):
    inputs = (income, loan, annuity, emp, ext, overdue, refused, late, debt)
    if API_URL:
        import httpx
        try:
            r = httpx.post(f"{API_URL}/explain", json=build_payload(*inputs), timeout=60)
            r.raise_for_status()
            return format_result(r.json(), footer="Live result from the LoanLens API")
        except Exception:
            return ("⚠️ The API request failed. Please try again.", "", "", "", "", "")

    stored = DEMO.get(sample)
    if stored and [float(v) for v in inputs] == [float(v) for v in stored["inputs"]]:
        footer = (
            f"Stored output of a real pipeline run (model v{stored['model_version']}, "
            f"{stored['llm_model']}, {stored['generated_at']})"
        )
        return format_result(stored["result"], footer=footer)
    return (NO_API_MESSAGE, "", "", "", "", "")


def load_sample(s):
    return SAMPLES.get(s, next(iter(SAMPLES.values())))


SAMPLE_NAMES = list(SAMPLES)

with gr.Blocks(title="LoanLens — Credit Risk Explainer") as demo:
    gr.Markdown("""# 🔍 LoanLens
### AI-Powered Credit Risk Explainer | RAG over CFPB Regulations
Enter borrower details to generate a regulation-grounded decline explanation.
Uses XGBoost credit scoring + SHAP explainability + ChromaDB retrieval + an OpenAI model for the notice text.""")

    with gr.Row():
        sample_dd = gr.Dropdown(choices=SAMPLE_NAMES, value=SAMPLE_NAMES[0], label="📋 Load Demo Sample")
        load_btn = gr.Button("Load Sample", variant="secondary", scale=0)

    gr.Markdown("---")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Borrower Information")
            income = gr.Number(label="Annual Income ($)", value=24000, minimum=1000)
            loan = gr.Number(label="Loan Amount Requested ($)", value=380000, minimum=1000)
            annuity = gr.Number(label="Annual Loan Payment ($)", value=19000, minimum=100)
            emp = gr.Slider(label="Years Employed", minimum=0, maximum=40, value=0.5, step=0.5)
            ext = gr.Slider(label="External Credit Score (0=poor, 1=excellent)", minimum=0.0, maximum=1.0, value=0.12, step=0.01)
            overdue = gr.Slider(label="Overdue Accounts in Credit History", minimum=0, maximum=20, value=4, step=1)
            refused = gr.Slider(label="Previous Loan Applications Refused", minimum=0, maximum=10, value=3, step=1)
            late = gr.Slider(label="Late Payment Rate (0=never, 1=always)", minimum=0.0, maximum=1.0, value=0.6, step=0.01)
            debt = gr.Slider(label="Bureau Debt-to-Credit Ratio", minimum=0.0, maximum=1.0, value=0.92, step=0.01)
            btn = gr.Button("🔍 Analyze Application", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Risk Assessment")
            decision_out = gr.Textbox(label="Decision", interactive=False, lines=1)
            shap_out = gr.Textbox(label="Top Factors (SHAP)", interactive=False, lines=10)
            notice_out = gr.Textbox(label="📄 Adverse Action Notice (declines only)", interactive=False, lines=5)
            citations_out = gr.Textbox(label="📚 Regulatory Citations (Retrieved from CFPB Manual)", interactive=False, lines=8)
            metadata_out = gr.Textbox(label="📈 Details", interactive=False, lines=4)
            rights_out = gr.Textbox(label="⚖️ Applicant Rights", interactive=False, lines=2)

    inputs = [income, loan, annuity, emp, ext, overdue, refused, late, debt, sample_dd]
    outputs = [decision_out, shap_out, notice_out, citations_out, metadata_out, rights_out]

    btn.click(fn=analyze, inputs=inputs, outputs=outputs)
    load_btn.click(fn=load_sample, inputs=[sample_dd], outputs=inputs[:-1])

    gr.Markdown("""---
**Tech Stack:** XGBoost · SHAP · LangChain · ChromaDB · OpenAI · FastAPI · MLflow

**Regulatory Coverage:** CFPB Supervision Manual (9,977 indexed chunks)

[GitHub](https://github.com/MohammedAhmeduddin/loanlens) | Built by Ahmeduddin Mohammed""")

if __name__ == "__main__":
    demo.launch()
