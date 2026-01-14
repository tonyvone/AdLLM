"""
Application settings and configuration management.

Uses pydantic-settings for environment-based configuration with validation.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database connection settings."""

    model_config = SettingsConfigDict(env_prefix="DB_")

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="adtech_llm", description="Database name")
    user: str = Field(default="adtech", description="Database user")
    password: str = Field(default="", description="Database password")
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Max overflow connections")

    @property
    def url(self) -> str:
        """Generate database URL."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def sync_url(self) -> str:
        """Generate synchronous database URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseSettings):
    """Redis connection settings."""

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    db: int = Field(default=0, description="Redis database number")
    password: str | None = Field(default=None, description="Redis password")

    @property
    def url(self) -> str:
        """Generate Redis URL."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class VectorDBSettings(BaseSettings):
    """Vector database settings for embeddings storage."""

    model_config = SettingsConfigDict(env_prefix="VECTOR_DB_")

    provider: str = Field(default="chromadb", description="Vector DB provider: chromadb, pinecone")
    pinecone_api_key: str | None = Field(default=None, description="Pinecone API key")
    pinecone_environment: str | None = Field(default=None, description="Pinecone environment")
    pinecone_index_name: str = Field(default="adtech-llm", description="Pinecone index name")
    chromadb_path: str = Field(default="./data/chromadb", description="ChromaDB persistence path")
    embedding_dimension: int = Field(default=768, description="Embedding vector dimension")


class LLMSettings(BaseSettings):
    """LLM model settings."""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    provider: str = Field(default="huggingface", description="LLM provider")
    model_name: str = Field(default="meta-llama/Llama-3.1-8B-Instruct", description="Base model")
    max_tokens: int = Field(default=4096, description="Maximum tokens for generation")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.95, description="Top-p sampling parameter")
    device: str = Field(default="auto", description="Device for inference: auto, cuda, cpu")
    quantization: str | None = Field(default="4bit", description="Quantization: 4bit, 8bit, None")
    use_flash_attention: bool = Field(default=True, description="Enable Flash Attention 2")

    # Fine-tuning settings
    lora_r: int = Field(default=16, description="LoRA rank")
    lora_alpha: int = Field(default=32, description="LoRA alpha")
    lora_dropout: float = Field(default=0.05, description="LoRA dropout")
    lora_target_modules: list[str] = Field(
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        description="Target modules for LoRA",
    )

    # OpenAI fallback
    openai_api_key: str | None = Field(default=None, description="OpenAI API key for fallback")
    openai_model: str = Field(default="gpt-4-turbo-preview", description="OpenAI model")


class DataSettings(BaseSettings):
    """Data pipeline settings."""

    model_config = SettingsConfigDict(env_prefix="DATA_")

    raw_path: Path = Field(default=Path("./data/raw"), description="Raw data directory")
    processed_path: Path = Field(default=Path("./data/processed"), description="Processed data")
    models_path: Path = Field(default=Path("./data/models"), description="Model checkpoints")
    cache_path: Path = Field(default=Path("./data/cache"), description="Cache directory")

    kaggle_username: str | None = Field(default=None, description="Kaggle username")
    kaggle_key: str | None = Field(default=None, description="Kaggle API key")

    max_dataset_size_gb: float = Field(default=100.0, description="Max dataset size in GB")
    batch_size: int = Field(default=1000, description="Processing batch size")
    num_workers: int = Field(default=4, description="Number of data processing workers")


class SecuritySettings(BaseSettings):
    """Security and authentication settings."""

    model_config = SettingsConfigDict(env_prefix="SECURITY_")

    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="JWT secret key",
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiry")
    refresh_token_expire_days: int = Field(default=7, description="Refresh token expiry")
    api_key_header: str = Field(default="X-API-Key", description="API key header name")

    # Rate limiting
    rate_limit_requests: int = Field(default=100, description="Requests per minute")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")


class AgentSettings(BaseSettings):
    """Agentic system settings."""

    model_config = SettingsConfigDict(env_prefix="AGENT_")

    orchestrator: str = Field(default="langgraph", description="Agent orchestrator: langgraph")
    max_iterations: int = Field(default=10, description="Max agent iterations per task")
    timeout_seconds: int = Field(default=300, description="Agent task timeout")
    memory_type: str = Field(default="redis", description="Agent memory backend")
    enable_human_in_loop: bool = Field(default=False, description="Require human approval")
    parallel_tasks: int = Field(default=4, description="Max parallel agent tasks")


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings."""

    model_config = SettingsConfigDict(env_prefix="MONITORING_")

    prometheus_port: int = Field(default=9090, description="Prometheus metrics port")
    enable_tracing: bool = Field(default=True, description="Enable distributed tracing")
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format: json, text")


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="AdTech LLM", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment: development, production")

    # API
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_prefix: str = Field(default="/api/v1", description="API prefix")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="CORS allowed origins",
    )

    # Sub-settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    vector_db: VectorDBSettings = Field(default_factory=VectorDBSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    agents: AgentSettings = Field(default_factory=AgentSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
