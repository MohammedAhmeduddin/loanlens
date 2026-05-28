"""
Application configuration via environment variables.
Never hardcode secrets — all sensitive values come from .env
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """
    Central config for LoanLens.
    All values loaded from environment / .env file.
    """

    # Application
    app_name: str = "LoanLens"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "loanlens"
    postgres_user: str = "loanlens"
    postgres_password: str = Field(...)

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:"
            f"{self.postgres_password}@{self.postgres_host}:"
            f"{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_sync(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:"
            f"{self.postgres_password}@{self.postgres_host}:"
            f"{self.postgres_port}/{self.postgres_db}"
        )

    # OpenAI
    openai_api_key: str = Field(...)
    openai_model: str = "gpt-4o-mini"
    openai_max_tokens: int = 1000
    openai_temperature: float = 0.1  # Low temp for consistent compliance text

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "cfpb_regulations"

    # RAG
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_k: int = 5          # Initial retrieval count
    rerank_top_n: int = 3         # After reranking, keep top 3

    # MLflow
    mlflow_tracking_uri: str = "./mlflow/mlruns"
    mlflow_experiment_name: str = "credit_scoring"
    mlflow_model_name: str = "CreditScoringModel"

    # Model thresholds
    decline_threshold: float = 0.5
    review_threshold: float = 0.35

    # Monitoring
    psi_moderate_threshold: float = 0.10
    psi_severe_threshold: float = 0.20
    slack_webhook_url: str = ""   # Optional — empty string disables alerts

    # Data paths
    raw_data_dir: str = "./data/raw"
    processed_data_dir: str = "./data/processed"
    demo_data_dir: str = "./data/demo"
    pdf_dir: str = "./data/regulations"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "case_sensitive": False}


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings instance.
    Use this everywhere — avoids re-reading .env on every call.
    """
    return Settings()
