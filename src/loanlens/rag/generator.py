"""
Adverse action notice generator (OpenAI chat model, JSON mode).
Takes retrieved regulatory passages + SHAP factors
and drafts an adverse action notice. Reasons and citation checks are done in code.
"""

import json
import sys
import time
from pathlib import Path

from loguru import logger
from openai import BadRequestError, OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from loanlens.config import get_settings
from loanlens.model.explain import FEATURE_TO_REGULATORY, is_citable_reason
from loanlens.rag.citations import check_citations, check_reasons

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


def _create_completion(client, settings, messages):
    """Call the chat API in JSON mode.

    Some models (reasoning models such as gpt-6-luna) only accept their
    default temperature; if the model rejects it, retry once without it.
    """
    kwargs = dict(
        model=settings.openai_model,
        messages=messages,
        max_completion_tokens=settings.openai_max_completion_tokens,
        response_format={"type": "json_object"},
        temperature=settings.openai_temperature,
    )
    try:
        return client.chat.completions.create(**kwargs)
    except BadRequestError as e:
        if "temperature" not in str(e).lower():
            raise
        logger.info(f"{settings.openai_model} rejected temperature; retrying with the model default")
        kwargs.pop("temperature")
        return client.chat.completions.create(**kwargs)


def principal_reasons(shap_factors: list[dict]) -> list[str]:
    """Citable risk factors as reason statements, one per reason code."""
    seen, reasons = set(), []
    for f in shap_factors:
        if is_citable_reason(f) and f["cfpb_code"] not in seen:
            seen.add(f["cfpb_code"])
            reasons.append(f["label"])
    return reasons[:4]


def generate_explanation(
    shap_factors: list[dict],
    retrieved_passages: list[dict],
    risk_score: float,
    decision: str,
) -> dict:
    """
    Draft an adverse action notice with the configured OpenAI model.

    Args:
        shap_factors: Top risk factors from SHAP explainer
        retrieved_passages: Regulatory passages from ChromaDB
        risk_score: Model probability score (0-1)
        decision: 'decline' or 'review'

    Returns:
        Structured explanation dict
    """
    settings = get_settings()
    client = OpenAI(
        api_key=settings.openai_api_key,
        timeout=settings.openai_timeout_seconds,
        max_retries=settings.openai_max_retries,
    )

    # Build context from retrieved passages
    regulatory_context = "\n\n".join([
        f"[Source: {p['source']}, Page {p['page']}]\n{p['text']}"
        for p in retrieved_passages
    ])

    # The reasons are decided by the model, not by the LLM: only factors that
    # raise risk and have an honest reason-code mapping, deduplicated by code.
    reasons = principal_reasons(shap_factors)
    reason_lines = "\n".join(f"- {r}" for r in reasons) or (
        "- No specific factor met the citation standard; refer to overall credit risk."
    )

    user_prompt = f"""
REGULATORY CONTEXT (use only this text for citations):
{regulatory_context}

CREDIT DECISION:
- Decision: {decision.upper()}
- Principal reasons (state exactly these, add no others):
{reason_lines}

Write a draft adverse action notice that states the principal reasons above.
Because the reasons are stated, do not tell the applicant to request a
statement of reasons. Return a JSON object with:
{{
    "adverse_action_notice": "Full formal adverse action notice text (3-5 sentences)",
    "primary_reasons": ["reason 1", "reason 2", "reason 3"],
    "regulatory_basis": "Specific regulation section cited from the context above",
    "applicant_rights": "One sentence: the right to a free copy of the credit report used, and to dispute its accuracy"
}}

Return ONLY the JSON object, no other text.
"""

    start_time = time.time()

    try:
        response = _create_completion(
            client,
            settings,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        elapsed_ms = int((time.time() - start_time) * 1000)
        choice = response.choices[0]
        raw_response = (choice.message.content or "").strip()
        if not raw_response:
            raise ValueError(
                f"Empty response from {settings.openai_model} (finish_reason={choice.finish_reason}); "
                "raise OPENAI_MAX_COMPLETION_TOKENS if it is 'length'"
            )

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

        # Verified in code, never taken from the LLM
        result["primary_reasons"] = reasons
        generated = " ".join(
            str(result.get(k) or "") for k in ("adverse_action_notice", "regulatory_basis")
        )
        check = check_citations(generated, retrieved_passages)
        result["citation_check"] = check
        result["grounding_score"] = check["grounding_score"]
        if check["unsupported"]:
            logger.warning(f"Citations not found in retrieved passages: {check['unsupported']}")

        other_labels = sorted({v["label"] for v in FEATURE_TO_REGULATORY.values()})
        result["reasons_check"] = check_reasons(result.get("adverse_action_notice", ""), reasons, other_labels)
        if not result["reasons_check"]["ok"]:
            logger.warning(f"Notice reasons differ from the model's: {result['reasons_check']}")
        result["notice_status"] = (
            "drafted" if result["reasons_check"]["ok"] and not check["unsupported"] else "needs_review"
        )

        result["generation_time_ms"] = elapsed_ms
        result["model"] = settings.openai_model
        result["retrieved_passages"] = len(retrieved_passages)

        logger.info(
            f"Generated explanation in {elapsed_ms}ms | "
            f"Citations verified: {len(check['supported'])}/{len(check['cited'])}"
        )

        return result

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise
