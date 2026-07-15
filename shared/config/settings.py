"""
Centralized config, loaded once from environment variables (via .env).

WHY: without this, every phase script does its own `os.getenv("MLFLOW_TRACKING_URI")`
with slightly different defaults, and Phase 4's serving layer silently points at the
wrong MLflow instance in someone's local run. Pydantic BaseSettings validates types
and fails fast at import time if something required is missing, instead of failing
three hours into a training run.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "claim-denial-prediction"

    # Phase 4
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    faiss_index_path: str = "./phase4_rag_agentic/index/claims.faiss"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Phase 3
    mimic_note_data_dir: str = ""

    # Serving
    api_port: int = 8000
    model_stage: str = "production"

    # GCP
    gcp_project_id: str = ""
    gcp_region: str = "us-central1"


settings = Settings()
