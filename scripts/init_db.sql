-- ============================================
-- LoanLens Database Schema
-- ============================================

-- Schemas only — raw tables created automatically by pandas
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS ml;

-- ============================================
-- ML TABLES (hand-crafted, not from CSVs)
-- ============================================

CREATE TABLE IF NOT EXISTS ml.predictions (
    id                  SERIAL PRIMARY KEY,
    sk_id_curr          BIGINT,
    risk_score          NUMERIC(6,4),
    decision            VARCHAR(20),
    model_version       VARCHAR(50),
    shap_factors        JSONB,
    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ml.explanations (
    id                      SERIAL PRIMARY KEY,
    prediction_id           INTEGER REFERENCES ml.predictions(id),
    adverse_action_notice   TEXT,
    regulatory_citations    JSONB,
    retrieved_chunks        JSONB,
    grounding_score         NUMERIC(5,4),
    generation_time_ms      INTEGER,
    created_at              TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ml.feature_baseline (
    id              SERIAL PRIMARY KEY,
    feature_name    VARCHAR(100),
    mean            NUMERIC(15,6),
    std             NUMERIC(15,6),
    p25             NUMERIC(15,6),
    p50             NUMERIC(15,6),
    p75             NUMERIC(15,6),
    computed_at     TIMESTAMP DEFAULT NOW()
);

-- ============================================
-- INDEXES
-- ============================================

CREATE INDEX IF NOT EXISTS idx_predictions_sk_id
    ON ml.predictions(sk_id_curr);

CREATE INDEX IF NOT EXISTS idx_predictions_created
    ON ml.predictions(created_at);
