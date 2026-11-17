"""
Pydantic request/response schemas for LoanLens API.
Strict typing on all inputs and outputs.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
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

    # Household (sex, marital status and age are prohibited bases under
    # ECOA / Regulation B and are deliberately not accepted)
    cnt_children: int = Field(0, ge=0, le=20)
    cnt_fam_members: Optional[float] = Field(None, ge=1)
    name_education_type: Optional[str] = None
    name_income_type: Optional[str] = None
    name_housing_type: Optional[str] = None
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

    # Credit bureau inquiries before the application
    amt_req_credit_bureau_hour: Optional[int] = Field(None, ge=0)
    amt_req_credit_bureau_day: Optional[int] = Field(None, ge=0)
    amt_req_credit_bureau_week: Optional[int] = Field(None, ge=0)
    amt_req_credit_bureau_mon: Optional[int] = Field(None, ge=0)
    amt_req_credit_bureau_qrt: Optional[int] = Field(None, ge=0)
    amt_req_credit_bureau_year: Optional[int] = Field(None, ge=0)

    # Bureau features
    bureau_loan_count: int = Field(0, ge=0)
    bureau_active_loans: int = Field(0, ge=0)
    bureau_overdue_count: int = Field(0, ge=0)
    bureau_max_overdue_days: int = Field(0, ge=0)
    bureau_total_debt: float = Field(0.0, ge=0)
    bureau_debt_to_credit: float = Field(0.0, ge=0, le=1)
    bureau_closed_loans: Optional[int] = Field(None, ge=0)
    bureau_oldest_credit_days: Optional[float] = Field(None, le=0, description="Days before application (negative)")
    bureau_newest_credit_days: Optional[float] = Field(None, le=0, description="Days before application (negative)")
    bureau_avg_overdue_days: Optional[float] = Field(None, ge=0)
    bureau_total_credit: Optional[float] = Field(None, ge=0)
    bureau_total_overdue_amt: Optional[float] = Field(None, ge=0)
    bureau_credit_type_count: Optional[int] = Field(None, ge=0)
    bureau_total_prolongations: Optional[int] = Field(None, ge=0)
    has_bureau_history: Optional[int] = Field(None, ge=0, le=1, description="Derived from bureau_loan_count if omitted")

    # Previous application features
    prev_app_count: int = Field(0, ge=0)
    prev_refused_count: int = Field(0, ge=0)
    prev_approval_rate: float = Field(0.0, ge=0, le=1)
    prev_approved_count: Optional[int] = Field(None, ge=0)
    prev_last_decision_days: Optional[float] = Field(None, le=0, description="Days before application (negative)")
    prev_first_decision_days: Optional[float] = Field(None, le=0, description="Days before application (negative)")
    prev_canceled_count: Optional[int] = Field(None, ge=0)
    prev_avg_application_amt: Optional[float] = Field(None, ge=0)
    prev_max_credit: Optional[float] = Field(None, ge=0)
    prev_avg_credit: Optional[float] = Field(None, ge=0)
    has_prev_application: Optional[int] = Field(None, ge=0, le=1, description="Derived from prev_app_count if omitted")

    # Installment features
    late_payment_rate: float = Field(0.0, ge=0, le=1)
    avg_payment_ratio: Optional[float] = None
    late_payment_count: Optional[int] = Field(None, ge=0)
    avg_payment_ratio: Optional[float] = Field(None, ge=0)
    avg_payment_deficit: Optional[float] = None
    avg_days_late: Optional[float] = Field(None, ge=0)
    instalment_count: int = Field(0, ge=0, description="Past instalments on record")
    has_instalment_history: Optional[int] = Field(None, ge=0, le=1, description="Derived from instalment_count if omitted")

    @model_validator(mode="after")
    def history_is_consistent(self):
        """Reject inputs no real record could produce (the model would be
        scoring a combination it never saw in training)."""
        problems = []
        if self.bureau_overdue_count > self.bureau_loan_count:
            problems.append("bureau_overdue_count cannot exceed bureau_loan_count")
        if self.bureau_active_loans > self.bureau_loan_count:
            problems.append("bureau_active_loans cannot exceed bureau_loan_count")
        if self.prev_refused_count > self.prev_app_count:
            problems.append("prev_refused_count cannot exceed prev_app_count")
        if self.late_payment_rate > 0 and self.instalment_count == 0:
            problems.append("late_payment_rate > 0 requires instalment_count > 0")
        if self.has_bureau_history == 0 and self.bureau_loan_count > 0:
            problems.append("has_bureau_history=0 contradicts bureau_loan_count > 0")
        if self.has_prev_application == 0 and self.prev_app_count > 0:
            problems.append("has_prev_application=0 contradicts prev_app_count > 0")
        if self.has_instalment_history == 0 and self.instalment_count > 0:
            problems.append("has_instalment_history=0 contradicts instalment_count > 0")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    @field_validator("debt_to_income", mode="before")
    @classmethod
    def compute_dti(cls, v, info):
        if v is None and "amt_credit" in info.data and "amt_income_total" in info.data:
            income = info.data["amt_income_total"]
            credit = info.data["amt_credit"]
            if income > 0:
                return round(credit / income, 4)
        return v

    model_config = ConfigDict(
        # Unknown fields are an error, not silently dropped. This also rejects
        # prohibited-basis fields (age_years, code_gender, ...) explicitly.
        extra="forbid",
        json_schema_extra={
            "example": {
                "amt_income_total": 75000,
                "amt_credit": 250000,
                "amt_annuity": 12000,
                "name_contract_type": "Cash loans",
                "employment_years": 3.5,
                "ext_source_2": 0.35,
                "bureau_loan_count": 4,
                "bureau_overdue_count": 1,
                "prev_app_count": 2,
                "prev_refused_count": 1,
                "instalment_count": 24,
                "late_payment_rate": 0.05,
            }
        },
    )


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
    cfpb_code: Optional[str] = None
    regulation: Optional[str] = None


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
