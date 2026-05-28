"""
GPT-4o-mini explanation generator.
Takes retrieved regulatory passages + SHAP factors
and generates a structured ECOA-compliant adverse action notice.
"""

import json
import time
from openai import OpenAI
from loguru import logger
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from loanlens.config import get_settings


SYSTEM_PROMPT = """You are a compliance officer at a lending institution.
Your job is to write adverse action notices that are:
1. Legally compliant with ECOA Regulation B and FCRA
2. Clear and understandable to the applicant
3. Grounded only in the regulatory text provided to you
4. Professional and factual in tone

You must only cite regulations that appear in the provided context.
Never invent regulatory citations. Never use prohibited bases under ECOA
(race, color, religion, national origin, sex, marital status, age).
"""


def generate_explanation(
    shap_factors: list[dict],
    retrieved_passages: list[dict],
    risk_score: float,
    decision: str,
) -> dict:
    """
    Generate structured adverse action notice using GPT-4o-mini.

    Args:
        shap_factors: Top risk factors from SHAP explainer
        retrieved_passages: Regulatory passages from ChromaDB
        risk_score: Model probability score (0-1)
        decision: 'decline' or 'review'

    Returns:
        Structured explanation dict
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    # Build context from retrieved passages
    regulatory_context = "\n\n".join([
        f"[Source: {p['source']}, Page {p['page']}]\n{p['text']}"
        for p in retrieved_passages
    ])

    # Build risk factors summary
    risk_summary = "\n".join([
        f"- {f['label']} (CFPB Code: {f['cfpb_code']}, "
        f"SHAP impact: {f['shap_value']:.4f})"
        for f in shap_factors
        if f["direction"] == "increases_risk"
    ])

    user_prompt = f"""
REGULATORY CONTEXT (use only this text for citations):
{regulatory_context}

CREDIT DECISION:
- Decision: {decision.upper()}
- Risk Score: {risk_score:.2f} out of 1.0
- Model-identified risk factors:
{risk_summary}

Write a structured adverse action notice. Return a JSON object with:
{{
    "adverse_action_notice": "Full formal adverse action notice text (3-5 sentences)",
    "primary_reasons": ["reason 1", "reason 2", "reason 3"],
    "regulatory_basis": "Specific regulation section cited from the context above",
    "applicant_rights": "One sentence describing applicant's right to obtain credit report",
    "grounding_score": 0.0
}}

The grounding_score (0.0 to 1.0) should reflect how well your response
is grounded in the provided regulatory context.
Return ONLY the JSON object, no other text.
"""

    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            max_tokens=settings.openai_max_tokens,
            temperature=settings.openai_temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        elapsed_ms = int((time.time() - start_time) * 1000)
        raw_response = response.choices[0].message.content.strip()

        # Parse JSON response
        try:
            result = json.loads(raw_response)
        except json.JSONDecodeError:
            # Try to extract JSON if wrapped in markdown
            import re
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                raise ValueError(f"Could not parse JSON from response: {raw_response}")

        result["generation_time_ms"] = elapsed_ms
        result["model"] = settings.openai_model
        result["retrieved_passages"] = len(retrieved_passages)

        logger.info(
            f"Generated explanation in {elapsed_ms}ms | "
            f"Grounding: {result.get('grounding_score', 0):.2f}"
        )

        return result

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise
