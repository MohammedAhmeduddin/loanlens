
import os
import gradio as gr

API_URL = os.getenv("API_URL", "")

PRECOMPUTED = {
    "High Risk — Young borrower, high debt": {
        "decision": "🔴 DECLINED — Risk Score: 89.5/100",
        "shap": "1. External credit score assessment\n   SHAP: -0.85 (+ risk) | Code: A9 - Credit score\n\n2. Insufficient length of employment\n   SHAP: -0.45 (+ risk) | Code: A13 - Length of employment\n\n3. Delinquent accounts in credit history\n   SHAP: -0.38 (+ risk) | Code: A1 - Delinquent obligations\n\n4. History of late installment payments\n   SHAP: -0.31 (+ risk) | Code: A1 - Delinquent obligations\n\n5. Debt-to-income ratio too high\n   SHAP: -0.28 (+ risk) | Code: A6 - Debt-to-income ratio",
        "notice": "We regret to inform you that your application for credit has been declined. This decision was based on information from your consumer credit report, including delinquent past obligations and insufficient employment history, as well as your debt-to-income ratio. Per 12 CFR 1002.9(a)(2) of Regulation B, you have the right to request a statement of specific reasons for this decision.",
        "citations": "📄 supervision_manual.pdf (p.850) — Relevance: 74.99%\napplicant's right to a statement of reasons for adverse action. The notification requirements for business credit applicants...\n\n📄 supervision_manual.pdf (p.946) — Relevance: 71.77%\nFCRA Section 615(a): creditor that denies credit based on consumer report must provide adverse action notice...\n\n📄 supervision_manual.pdf (p.376) — Relevance: 69.38%\nAdverse Action Notices (FCRA and ECOA): Assess compliance with FCRA 15 USC 1681m...",
        "metadata": "Grounding Score: 100%\nGeneration Time: 3458ms\nModel: CreditScoringModel v1\nRegulatory Basis: 12 CFR 1002.9(a)(2)",
        "rights": "Applicant Rights: You have the right to obtain a free copy of your credit report from the consumer reporting agency used in this decision within 60 days.",
    },
    "Medium Risk — Average borrower": {
        "decision": "🟡 REVIEW REQUIRED — Risk Score: 43.2/100",
        "shap": "1. External credit score assessment\n   SHAP: -0.45 (+ risk) | Code: A9 - Credit score\n\n2. Insufficient length of credit history\n   SHAP: -0.44 (+ risk) | Code: A8 - Length of credit history\n\n3. Creditworthiness assessment based on financial profile\n   SHAP: -0.41 (+ risk) | Code: A9 - Credit score\n\n4. Previous credit applications declined\n   SHAP: -0.36 (+ risk) | Code: A7 - Number of recent inquiries\n\n5. Credit amount relative to payment capacity\n   SHAP: -0.31 (+ risk) | Code: A6 - Debt-to-income ratio",
        "notice": "Your application for credit requires additional review. Based on our assessment, several factors in your credit profile require further evaluation including your credit score assessment and payment capacity. Per ECOA Regulation B, you will receive written notification of our final decision within 30 days.",
        "citations": "📄 supervision_manual.pdf (p.850) — Relevance: 74.99%\nApplicant's right to a statement of reasons for adverse action...\n\n📄 supervision_manual.pdf (p.376) — Relevance: 73.73%\nAdverse Action Notices under FCRA and ECOA require assessment of compliance...",
        "metadata": "Grounding Score: 100%\nGeneration Time: 2891ms\nModel: CreditScoringModel v1\nRegulatory Basis: ECOA Regulation B; FCRA Section 615(a)",
        "rights": "Applicant Rights: You have the right to obtain a free copy of your credit report from the consumer reporting agency used in this decision.",
    },
    "Low Risk — Stable borrower": {
        "decision": "🟢 APPROVED — Risk Score: 18.6/100",
        "shap": "1. External credit score assessment\n   SHAP: 0.74 (- risk) | Code: A9 - Credit score\n\n2. Strong employment history\n   SHAP: 0.52 (- risk) | Code: A13 - Length of employment\n\n3. Low debt-to-income ratio\n   SHAP: 0.48 (- risk) | Code: A6 - Debt-to-income ratio\n\n4. Clean payment history\n   SHAP: 0.41 (- risk) | Code: A1 - Payment history\n\n5. Established credit history\n   SHAP: 0.38 (- risk) | Code: A8 - Length of credit history",
        "notice": "No adverse action notice required.",
        "citations": "No regulatory citations required for approved applications.",
        "metadata": "Grounding Score: N/A\nGeneration Time: 0ms\nModel: CreditScoringModel v1\nDecision: Approved — no adverse action required",
        "rights": "N/A",
    },
}

def analyze(income, loan, annuity, age, emp, ext, overdue, refused, late, debt, sample):
    import httpx
    if API_URL:
        try:
            payload = {
                "amt_income_total": income, "amt_credit": loan,
                "amt_annuity": annuity, "age_years": age,
                "employment_years": emp, "ext_source_2": ext,
                "bureau_overdue_count": int(overdue),
                "prev_refused_count": int(refused),
                "late_payment_rate": late,
                "bureau_debt_to_credit": debt, "cnt_children": 0,
            }
            r = httpx.post(f"{API_URL}/explain", json=payload, timeout=30)
            r.raise_for_status()
            result = r.json()
            decision = result["decision"]
            risk = result["risk_score"]
            if decision == "decline":
                d = f"🔴 DECLINED — Risk Score: {risk:.1f}/100"
            elif decision == "review":
                d = f"🟡 REVIEW REQUIRED — Risk Score: {risk:.1f}/100"
            else:
                d = f"🟢 APPROVED — Risk Score: {risk:.1f}/100"
            shap_lines = []
            for f in result.get("shap_factors", []):
                direction = "+ risk" if f["direction"] == "increases_risk" else "- risk"
                shap_lines.append(f"{f['rank']}. {f['label']}\n   SHAP: {f['shap_value']:.4f} ({direction}) | Code: {f['cfpb_code']}")
            shap = "\n\n".join(shap_lines)
            notice = result.get("adverse_action_notice") or "No adverse action notice required."
            passages = result.get("retrieved_passages", [])
            citations = "\n\n".join([f"📄 {p['source']} (p.{p['page']}) — Relevance: {p['similarity_score']:.2%}\n{p['text'][:200]}..." for p in passages]) if passages else "N/A"
            grounding = result.get("grounding_score")
            gen_time = result.get("generation_time_ms", 0)
            metadata = f"Grounding Score: {grounding:.0%}\nGeneration Time: {gen_time}ms\nModel: CreditScoringModel v1\nRegulatory Basis: {result.get('regulatory_basis', 'N/A')}" if grounding else "Approved — no explanation required"
            rights = f"Applicant Rights: {result.get('applicant_rights', 'N/A')}"
            return d, shap, notice, citations, metadata, rights
        except Exception as e:
            pass

    # Demo mode — use precomputed
    data = PRECOMPUTED.get(sample, PRECOMPUTED["Medium Risk — Average borrower"])
    note = data["notice"] + "\n\n⚡ Demo mode — connect API_URL for live results"
    return data["decision"], data["shap"], note, data["citations"], data["metadata"], data["rights"]

def load_sample(s):
    samples = {
        "High Risk — Young borrower, high debt": (24000, 380000, 19000, 23, 0.5, 0.12, 4, 3, 0.6, 0.92),
        "Medium Risk — Average borrower": (55000, 200000, 10000, 38, 4.0, 0.42, 1, 1, 0.1, 0.4),
        "Low Risk — Stable borrower": (120000, 180000, 9000, 45, 12.0, 0.74, 0, 0, 0.0, 0.15),
    }
    return samples.get(s, (75000, 200000, 10000, 35, 5.0, 0.5, 0, 0, 0.0, 0.2))

SAMPLES = ["High Risk — Young borrower, high debt", "Medium Risk — Average borrower", "Low Risk — Stable borrower"]

with gr.Blocks(title="LoanLens — Credit Risk Explainer") as demo:
    gr.Markdown("""# 🔍 LoanLens
### AI-Powered Credit Risk Explainer | RAG over CFPB Regulations
Enter borrower details to generate a regulation-grounded decline explanation.
Uses XGBoost credit scoring + SHAP explainability + ChromaDB retrieval + GPT-4o-mini.""")

    with gr.Row():
        sample_dd = gr.Dropdown(choices=SAMPLES, value=SAMPLES[0], label="📋 Load Demo Sample")
        load_btn = gr.Button("Load Sample", variant="secondary", scale=0)

    gr.Markdown("---")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Borrower Information")
            income = gr.Number(label="Annual Income ($)", value=24000, minimum=1000)
            loan = gr.Number(label="Loan Amount Requested ($)", value=380000, minimum=1000)
            annuity = gr.Number(label="Annual Loan Payment ($)", value=19000, minimum=100)
            age = gr.Slider(label="Age (years)", minimum=18, maximum=75, value=23, step=1)
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
            shap_out = gr.Textbox(label="Top Risk Factors (SHAP Analysis)", interactive=False, lines=10)
            notice_out = gr.Textbox(label="📄 Adverse Action Notice (ECOA Compliant)", interactive=False, lines=5)
            citations_out = gr.Textbox(label="📚 Regulatory Citations (Retrieved from CFPB Manual)", interactive=False, lines=8)
            metadata_out = gr.Textbox(label="📈 Model Metadata", interactive=False, lines=4)
            rights_out = gr.Textbox(label="⚖️ Applicant Rights", interactive=False, lines=2)

    inputs = [income, loan, annuity, age, emp, ext, overdue, refused, late, debt, sample_dd]
    outputs = [decision_out, shap_out, notice_out, citations_out, metadata_out, rights_out]

    btn.click(fn=analyze, inputs=inputs, outputs=outputs)
    load_btn.click(fn=load_sample, inputs=[sample_dd], outputs=inputs[:-1])

    gr.Markdown("""---
**Tech Stack:** XGBoost · SHAP · LangChain · ChromaDB · GPT-4o-mini · FastAPI · MLflow

**Regulatory Coverage:** CFPB Supervision Manual (9,977 indexed chunks)

[GitHub](https://github.com/MohammedAhmeduddin/loanlens) | Built by Ahmeduddin Mohammed""")

if __name__ == "__main__":
    demo.launch()
