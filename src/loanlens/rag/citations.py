"""
Checks on generated adverse action notices, done in code, not by the LLM.

- Citation check: every regulation the notice cites (e.g. "12 CFR 1002.9",
  "Section 615(a)", "15 U.S.C. 1681m") must appear in the retrieved passages.
  grounding_score = share of cited references found in those passages.

The principal reasons themselves are not checked here: they are set from
the model's citable SHAP factors in code, never written by the LLM.
"""

from __future__ import annotations

import re

_CITATION_PATTERNS = [
    # 12 CFR 1002.9(a)(2), 12 C.F.R. § 1002.9
    (re.compile(r"\b\d+\s*C\.?F\.?R\.?\s*(?:§+\s*)?(\d+\.\d+)(?:\([a-z0-9]+\))*", re.I), "{}"),
    # § 1002.9 / §1002.6(b)
    (re.compile(r"§+\s*(\d+\.\d+)(?:\([a-z0-9]+\))*"), "{}"),
    # 15 U.S.C. 1681m / 15 USC § 1691(d)
    (re.compile(r"\b\d+\s*U\.?\s*S\.?\s*C\.?\s*(?:§+\s*)?(\d+[a-z]*)", re.I), "{}"),
    # Section 615(a) / Sec. 701
    (re.compile(r"\bSec(?:tion|\.)\s+(\d+[a-z]?)(?:\([a-z0-9]+\))*", re.I), "section {}"),
]

def _normalize(text: str) -> str:
    text = text.lower().replace("§", " ").replace("sec.", "section")
    return re.sub(r"\s+", " ", text)


def extract_citations(text: str) -> list[tuple[str, str]]:
    """Return (as_written, key) pairs; key is what must appear in the sources."""
    found: dict[str, str] = {}
    for pattern, key_fmt in _CITATION_PATTERNS:
        for m in pattern.finditer(text or ""):
            key = key_fmt.format(m.group(1).lower())
            found.setdefault(key, m.group(0).strip())
    return [(written, key) for key, written in found.items()]


def check_citations(generated_text: str, passages: list[dict]) -> dict:
    sources = _normalize(" ".join(p.get("text", "") for p in passages))
    cited = extract_citations(generated_text)
    supported = [w for w, key in cited if key in sources]
    unsupported = [w for w, key in cited if key not in sources]
    return {
        "cited": [w for w, _ in cited],
        "supported": supported,
        "unsupported": unsupported,
        # None when nothing is cited: there is nothing to verify.
        "grounding_score": round(len(supported) / len(cited), 4) if cited else None,
    }



def check_reasons(notice: str, reasons: list[str], other_labels: list[str]) -> dict:
    """Does the notice state every principal reason, and no reason the model did not give?

    reasons: labels decided from the model's SHAP factors.
    other_labels: every other citable reason label; finding one in the notice
    means the LLM added a reason of its own.
    """
    text = " ".join(re.sub(r"[^a-z0-9 ]+", " ", (notice or "").lower()).split())

    def found(label: str) -> bool:
        return " ".join(re.sub(r"[^a-z0-9 ]+", " ", label.lower()).split()) in text

    missing = [r for r in reasons if not found(r)]
    extra = [label for label in other_labels if label not in reasons and found(label)]
    return {"stated": [r for r in reasons if found(r)], "missing": missing, "extra": extra,
            "ok": not missing and not extra}
