"""Citation checks and model-decided reasons for generated notices."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loanlens.rag.citations import check_citations, extract_citations  # noqa: E402

PASSAGES = [
    {"text": "Under 12 CFR 1002.9(a)(2), a creditor must provide a statement of specific reasons.",
     "source": "manual.pdf", "page": 1},
    {"text": "FCRA Section 615(a) requires notice when a consumer report is used (15 U.S.C. 1681m).",
     "source": "manual.pdf", "page": 2},
]


class TestExtractCitations:
    def test_common_formats(self):
        keys = {k for _, k in extract_citations(
            "Per 12 C.F.R. § 1002.9(a)(2), §1002.6(b), 15 USC 1681m and Section 615(a)."
        )}
        assert keys == {"1002.9", "1002.6", "1681m", "section 615"}

    def test_no_citations(self):
        assert extract_citations("Your application was declined.") == []


class TestCheckCitations:
    def test_all_supported(self):
        r = check_citations("Pursuant to 12 CFR 1002.9 and Section 615(a).", PASSAGES)
        assert r["grounding_score"] == 1.0
        assert r["unsupported"] == []

    def test_invented_citation_is_caught(self):
        r = check_citations("Pursuant to 12 CFR 1002.9 and 12 CFR 1026.18.", PASSAGES)
        assert r["grounding_score"] == 0.5
        assert r["unsupported"] == ["12 CFR 1026.18"]

    def test_nothing_cited_is_not_scored(self):
        r = check_citations("Your application was declined.", PASSAGES)
        assert r["grounding_score"] is None


FACTORS = [
    {"label": "History of late installment payments", "direction": "increases_risk",
     "cfpb_code": "A1 - Delinquent past or present credit obligations", "shap_value": 0.9},
    {"label": "Severity of past delinquency", "direction": "increases_risk",
     "cfpb_code": "A1 - Delinquent past or present credit obligations", "shap_value": 0.4},
    {"label": "Other model factor (not a citable reason)", "direction": "increases_risk",
     "cfpb_code": None, "shap_value": 0.3},
    {"label": "External credit score assessment", "direction": "decreases_risk",
     "cfpb_code": "A9 - Credit score", "shap_value": -0.5},
]


def _llm_reply(payload: dict):
    return MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps(payload)))])


class TestGeneratorUsesModelReasons:
    @patch("loanlens.rag.generator.OpenAI")
    def test_reasons_come_from_model_not_llm(self, mock_openai):
        mock_openai.return_value.chat.completions.create.return_value = _llm_reply({
            "adverse_action_notice": "Declined per 12 CFR 1002.9.",
            "primary_reasons": ["Invented reason the model never produced"],
            "regulatory_basis": "12 CFR 1002.9",
            "applicant_rights": "You may request a free credit report.",
            "grounding_score": 1.0,
        })
        from loanlens.rag.generator import generate_explanation
        result = generate_explanation(FACTORS, PASSAGES, risk_score=0.4, decision="decline")
        # one reason per code, only risk-increasing citable factors
        assert result["primary_reasons"] == ["History of late installment payments"]
        assert result["grounding_score"] == 1.0  # computed from passages, not the LLM's claim

    @patch("loanlens.rag.generator.OpenAI")
    def test_llm_self_score_is_ignored(self, mock_openai):
        mock_openai.return_value.chat.completions.create.return_value = _llm_reply({
            "adverse_action_notice": "Declined per 12 CFR 1026.18.",
            "regulatory_basis": "12 CFR 1026.18",
            "applicant_rights": "Rights.",
            "grounding_score": 1.0,
        })
        from loanlens.rag.generator import generate_explanation
        result = generate_explanation(FACTORS, PASSAGES, risk_score=0.4, decision="decline")
        assert result["grounding_score"] == 0.0
        assert result["citation_check"]["unsupported"] == ["12 CFR 1026.18"]

    @patch("loanlens.rag.generator.OpenAI")
    def test_uses_json_mode_and_lists_only_model_reasons(self, mock_openai):
        create = mock_openai.return_value.chat.completions.create
        create.return_value = _llm_reply({"adverse_action_notice": "x", "regulatory_basis": "", "applicant_rights": ""})
        from loanlens.rag.generator import generate_explanation
        generate_explanation(FACTORS, PASSAGES, risk_score=0.4, decision="decline")
        kwargs = create.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}
        prompt = kwargs["messages"][1]["content"]
        assert "History of late installment payments" in prompt
        assert "External credit score" not in prompt   # a strength, not a reason
        assert "not a citable reason" not in prompt


class TestModelVersionConsistency:
    def test_pipeline_loads_the_version_its_preprocessing_came_from(self):
        with patch("loanlens.rag.pipeline.load_production_model") as load_model, \
             patch("loanlens.rag.pipeline.CreditExplainer"), \
             patch("loanlens.rag.pipeline.RegulatoryRetriever"), \
             patch("loanlens.rag.pipeline.load_preprocessing_artifacts",
                   return_value=(["ext_source_mean"], {}, 7)):
            from loanlens.rag.pipeline import LoanLensPipeline
            pipeline = LoanLensPipeline()
        load_model.assert_called_once_with(7)
        assert pipeline.model_version == 7


class TestOpenAICall:
    def _bad_request(self, message):
        from openai import BadRequestError
        response = MagicMock(status_code=400, headers={}, request=MagicMock())
        return BadRequestError(message, response=response, body=None)

    @patch("loanlens.rag.generator.OpenAI")
    def test_uses_max_completion_tokens_and_configured_model(self, mock_openai):
        create = mock_openai.return_value.chat.completions.create
        create.return_value = _llm_reply({"adverse_action_notice": "x", "regulatory_basis": "", "applicant_rights": ""})
        from loanlens.config import get_settings
        from loanlens.rag.generator import generate_explanation
        generate_explanation(FACTORS, PASSAGES, risk_score=0.4, decision="decline")
        kwargs = create.call_args.kwargs
        assert kwargs["model"] == get_settings().openai_model
        assert "max_completion_tokens" in kwargs and "max_tokens" not in kwargs

    @patch("loanlens.rag.generator.OpenAI")
    def test_retries_without_temperature_when_model_rejects_it(self, mock_openai):
        create = mock_openai.return_value.chat.completions.create
        ok = _llm_reply({"adverse_action_notice": "x", "regulatory_basis": "", "applicant_rights": ""})
        create.side_effect = [self._bad_request("Unsupported value: 'temperature'"), ok]
        from loanlens.rag.generator import generate_explanation
        generate_explanation(FACTORS, PASSAGES, risk_score=0.4, decision="decline")
        assert create.call_count == 2
        assert "temperature" in create.call_args_list[0].kwargs
        assert "temperature" not in create.call_args_list[1].kwargs

    @patch("loanlens.rag.generator.OpenAI")
    def test_other_bad_requests_are_not_retried(self, mock_openai):
        create = mock_openai.return_value.chat.completions.create
        create.side_effect = self._bad_request("Invalid model")
        from openai import BadRequestError

        from loanlens.rag.generator import generate_explanation
        with pytest.raises(BadRequestError):
            generate_explanation(FACTORS, PASSAGES, risk_score=0.4, decision="decline")
        assert create.call_count == 1

    @patch("loanlens.rag.generator.OpenAI")
    def test_empty_reply_is_a_clear_error(self, mock_openai):
        reply = MagicMock(choices=[MagicMock(message=MagicMock(content=None), finish_reason="length")])
        mock_openai.return_value.chat.completions.create.return_value = reply
        from loanlens.rag.generator import generate_explanation
        with pytest.raises(ValueError, match="finish_reason=length"):
            generate_explanation(FACTORS, PASSAGES, risk_score=0.4, decision="decline")
