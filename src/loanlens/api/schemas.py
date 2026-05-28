"""
Pydantic request/response schemas for LoanLens API.
Strict typing on all inputs and outputs.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


class Decision(str, Enum):
    APPROVE = "approve"
    REVIEW = "review"
    DECLINE = "decline"


# ── Request Schemas ────────────────────────────────────────────

class ScoreRequest(BaseModel):
    """
    Borrower features for credit scoring.
    Matches columns in staging_features.feat_master.
    """
    # Core financials
    amt_income_total: float = Field(..., gt=0, description="Annual income in currency units")
    amt_credit: float = Field(..., gt=0, description="Loan amount requested")
    amt_annuity: Optional[float] = Field(None, gt=0, description="Loan annuity amount")
    amt_goods_price: Optional[float] = Field(None, gt=0)

    # Demographics
    age_years: float = Field(..., gt=18, lt=100, description="Applicant age in years")
    cnt_children: int = Field(0, ge=0, le=20)
    cnt_fam_members: Optional[float] = Field(None, ge=1)
    name_family_status: Optional[str] = None
    name_education_type: Optional[str] = None
    name_income_type: Optional[str] = None
    name_housing_type: Optional[str] = None
    code_gender: Optional[str] = None
    name_contract_type: Optional[str] = None

    # Employment
    employment_years: float = Field(0.0, ge=0)
    is_unemployed: bool = False
    occupation_type: Optional[str] = None
    organization_type: Optional[str] = None

    # External scores
    ext_source_1: Optional[float] = Field(None, ge=0, le=1)
    ext_source_2: Optional[float] = Field(None, ge=0, le=1)
    ext_source_3: Optional[float] = Field(None, ge=0, le=1)
    ext_source_mean: Optional[float] = Field(None, ge=0, le=1)

    # Derived ratios
    debt_to_income: Optional[float] = None
    credit_to_annuity: Optional[float] = None
    income_per_person: Optional[float] = None
    employed_to_age_ratio: Optional[float] = None

    # Flags
    flag_own_car: Optional[str] = None
    flag_own_realty: Optional[str] = None
    flag_work_phone: Optional[int] = Field(None, ge=0, le=1)
    flag_email: Optional[int] = Field(None, ge=0, le=1)

    # Region
    region_rating_client: Optional[int] = Field(None, ge=1, le=3)
    region_rating_client_w_city: Optional[int] = Field(None, ge=1, le=3)
    reg_city_not_live_city: Optional[int] = Field(None, ge=0, le=1)
    reg_city_not_work_city: Optional[int] = Field(None, ge=0, le=1)

    # Bureau features
    bureau_loan_count: int = Field(0, ge=0)
    bureau_active_loans: int = Field(0, ge=0)
    bureau_overdue_count: int = Field(0, ge=0)
    bureau_max_overdue_days: int = Field(0, ge=0)
    bureau_total_debt: float = Field(0.0, ge=0)
    bureau_debt_to_credit: float = Field(0.0, ge=0, le=1)
    has_bureau_history: int = Field(0, ge=0, le=1)

    # Previous application features
    prev_app_count: int = Field(0, ge=0)
    prev_refused_count: int = Field(0, ge=0)
    prev_approval_rate: float = Field(0.0, ge=0, le=1)
    has_prev_application: int = Field(0, ge=0, le=1)

    # Installment features
    late_payment_rate: float = Field(0.0, ge=0, le=1)
    avg_payment_ratio: Optional[float] = None
    has_instalment_history: int = Field(0, ge=0, le=1)

    @field_validator("debt_to_income", mode="before")
    @classmethod
    def compute_dti(cls, v, info):
        if v is None and "amt_credit" in info.data and "amt_income_total" in info.data:
            income = info.data["amt_income_total"]
            credit = info.data["amt_credit"]
            if income > 0:
                return round(credit / income, 4)
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "amt_income_total": 75000,
                "amt_credit": 250000,
                "amt_annuity": 12000,
                "age_years": 35,
                "cnt_children": 2,
                "employment_years": 3.5,
                "ext_source_2": 0.35,
                "bureau_overdue_count": 1,
                "prev_refused_count": 1,
            }
        }


class ExplainRequest(ScoreRequest):
    """Extended request that triggers full RAG explanation."""
    pass


# ── Response Schemas ───────────────────────────────────────────

class ShapFactor(BaseModel):
    rank: int
    feature: str
    feature_value: Optional[float]
    shap_value: float
    direction: str
    label: str
    cfpb_code: str
    regulation: str


class RetrievedPassage(BaseModel):
    text: str
    source: str
    page: int
    similarity_score: float
    rank: int


class ScoreResponse(BaseModel):
    risk_score: float = Field(..., description="Risk score 0-100")
    probability: float = Field(..., description="Default probability 0-1")
    decision: Decision
    model_version: str = "1"


class ExplainResponse(BaseModel):
    # Score
    risk_score: float
    probability: float
    decision: Decision
    model_version: str = "1"

    # SHAP
    shap_factors: list[ShapFactor]
    adverse_action_codes: list[str]

    # RAG
    retrieved_passages: list[RetrievedPassage]
    adverse_action_notice: Optional[str]
    primary_reasons: list[str]
    regulatory_basis: Optional[str]
    applicant_rights: Optional[str]
    grounding_score: Optional[float]
    generation_time_ms: int

    # Demo mode flag
    demo_mode: bool = False


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    chromadb_chunks: int
    version: str
