import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, 'src')

# Retriever first
from loanlens.rag.retriever import RegulatoryRetriever
retriever = RegulatoryRetriever()

# Then XGBoost + SHAP
from loanlens.model.data_loader import load_features, preprocess_features
from loanlens.model.explain import load_production_model, CreditExplainer
from loanlens.rag.generator import generate_explanation

model = load_production_model()
explainer = CreditExplainer(model)

df = load_features(limit=100)
X, y, _ = preprocess_features(df)
probs = model.predict_proba(X)[:, 1]
high_risk_idx = probs.argmax()
borrower = X.iloc[[high_risk_idx]]

print(f'Borrower prob: {probs[high_risk_idx]:.4f}')

factors = explainer.get_shap_factors(borrower, n_top=5)
query = explainer.build_rag_query(factors)
passages = retriever.retrieve_for_explanation(query)
result = generate_explanation(
    factors, passages,
    risk_score=probs[high_risk_idx],
    decision='decline'
)

print(f'Risk Score: {round(probs[high_risk_idx]*100, 2)}')
print(f'Grounding: {result["grounding_score"]}')
print(f'Time: {result["generation_time_ms"]}ms')
print(f'\nAdverse Action Notice:')
print(result['adverse_action_notice'])
print(f'\nPrimary Reasons:')
for r in result['primary_reasons']:
    print(f'  - {r}')
