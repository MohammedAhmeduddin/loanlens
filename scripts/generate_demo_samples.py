"""
Regenerate the dashboard's demo samples from the real pipeline.

Runs each sample in loanlens.dashboard.samples through the production model,
SHAP, retrieval and (for declines) the notice generator, and writes the
results to src/loanlens/dashboard/demo_samples.json. The public demo shows
these stored results, so they must come from a real run, never be hand-written.

Usage (needs the trained model, ChromaDB index and OPENAI_API_KEY):
    python scripts/generate_demo_samples.py
"""

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlflow.tracking import MlflowClient  # noqa: E402

from loanlens.api.main import request_to_dataframe  # noqa: E402
from loanlens.api.schemas import ExplainRequest  # noqa: E402
from loanlens.config import get_settings  # noqa: E402
from loanlens.dashboard.samples import DEMO_SAMPLES_PATH, SAMPLES, build_payload  # noqa: E402
from loanlens.rag.pipeline import LoanLensPipeline  # noqa: E402


def main() -> None:
    settings = get_settings()
    pipeline = LoanLensPipeline()
    versions = MlflowClient().search_model_versions(f"name='{settings.mlflow_model_name}'")
    model_version = max(int(v.version) for v in versions)

    out = {}
    for name, inputs in SAMPLES.items():
        request = ExplainRequest(**build_payload(*inputs))
        X = request_to_dataframe(request, pipeline.feature_names, pipeline.category_maps)
        result = pipeline.explain(X)
        out[name] = {
            "inputs": list(inputs),
            "model_version": model_version,
            "llm_model": settings.openai_model,
            "generated_at": date.today().isoformat(),
            "result": result,
        }
        print(f"{name}: {result['decision']} (P(default) = {result['probability']:.3f}) "
              f"| codes: {result['adverse_action_codes']}")

    DEMO_SAMPLES_PATH.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nWrote {DEMO_SAMPLES_PATH}")


if __name__ == "__main__":
    main()
