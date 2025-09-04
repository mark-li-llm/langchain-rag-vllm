"""Configuration management for RAG MS MARCO template.

This module handles all environment variable validation and provides
a centralized settings object for the application.
"""

from typing import Optional

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable validation."""

    # -----------------------------
    # Data and Index Configuration
    # -----------------------------
    hf_dataset_name: str = "microsoft/ms_marco"
    hf_dataset_config: str = "v2.1"
    hf_split_corpus: str = "train"
    hf_split_queries: str = "validation"
    corpus_sample_size: Optional[int] = 3000

    # -----------------------------
    # Embedding Configuration
    # 与 .env 的 EMBED_MODEL_NAME 对应
    # -----------------------------
    embed_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        validation_alias=AliasChoices("EMBED_MODEL_NAME", "embed_model_name"),
    )

    # -----------------------------
    # Index Backend (Qdrant) Configuration
    # 与 .env 的 INDEX_BACKEND / QDRANT_URL / QDRANT_API_KEY / COLLECTION_NAME 对应
    # -----------------------------
    index_backend: str = Field(
        default="qdrant",
        validation_alias=AliasChoices("INDEX_BACKEND", "index_backend"),
    )
    qdrant_url: str = Field(
        default="",
        validation_alias=AliasChoices("QDRANT_URL", "qdrant_url"),
    )
    qdrant_api_key: Optional[str] = Field(
        default="",
        validation_alias=AliasChoices("QDRANT_API_KEY", "qdrant_api_key"),
    )
    collection_name: str = Field(
        default="msmarco_chunks_v21",
        validation_alias=AliasChoices("COLLECTION_NAME", "collection_name"),
    )

    # -----------------------------
    # LLM Configuration
    # (这些通常也放到 .env：OPENAI_API_KEY、OPENAI_API_BASE、LLM_MODEL 等)
    # -----------------------------
    openai_api_key: str = Field(validation_alias=AliasChoices("OPENAI_API_KEY", "openai_api_key"))
    openai_api_base: str = Field(validation_alias=AliasChoices("OPENAI_API_BASE", "openai_api_base"))
    llm_model: str = Field(validation_alias=AliasChoices("LLM_MODEL", "llm_model"))
    max_output_tokens: int = 1000
    temperature: float = 0.2

    # -----------------------------
    # API and Controls Configuration
    # -----------------------------
    auth_bearer_token: str = Field(validation_alias=AliasChoices("AUTH_BEARER_TOKEN", "auth_bearer_token"))
    topk_pre: int = 50
    topk_final: int = 5
    use_bm25: bool = False
    use_reranker: bool = False
    max_context_tokens: int = 3000

    # -----------------------------
    # Operational Configuration
    # -----------------------------
    otel_exporter_otlp_endpoint: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("OTEL_EXPORTER_OTLP_ENDPOINT", "otel_exporter_otlp_endpoint"),
    )
    prometheus_port: int = 9100
    log_level: str = "info"
    uvicorn_workers: int = 2

    # -----------------------------
    # Chunking Configuration
    # -----------------------------
    chunk_size_chars: int = 1800
    chunk_overlap_chars: int = 200

    # -----------------------------
    # Validators
    # -----------------------------
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = {"debug", "info", "warning", "error", "critical"}
        if v.lower() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.lower()

    @field_validator("index_backend")
    @classmethod
    def validate_index_backend(cls, v):
        valid_backends = {"qdrant"}
        if v.lower() not in valid_backends:
            raise ValueError(f"index_backend must be one of {valid_backends}")
        return v.lower()

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v

    @field_validator("topk_pre", "topk_final")
    @classmethod
    def validate_topk(cls, v):
        if v <= 0:
            raise ValueError("top-k values must be positive integers")
        return v

    @field_validator("corpus_sample_size")
    @classmethod
    def validate_corpus_sample_size(cls, v):
        if v is not None and v <= 0:
            raise ValueError("corpus_sample_size must be a positive integer")
        return v

    # -----------------------------
    # Settings Config
    # -----------------------------
    model_config = SettingsConfigDict(
        case_sensitive=False,           # 环境变量名大小写不敏感
        env_file=".env",                # 直接读取本目录的 .env
        env_file_encoding="utf-8",
        extra="ignore",                 # 允许多余的环境变量
        env_prefix="",                  # 不加前缀
    )


# Global settings instance - loaded once at module import
settings = Settings()
