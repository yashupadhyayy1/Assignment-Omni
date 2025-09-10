from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WeatherSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    openweather_api_key: Optional[str] = Field(default=None, alias="OPENWEATHER_API_KEY")


class QdrantSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")
    collection: str = Field(default="assignment_omni", alias="QDRANT_COLLECTION")


class LangsmithSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    api_key: Optional[str] = Field(default=None, alias="LANGSMITH_API_KEY")
    tracing: bool = Field(default=True, alias="LANGSMITH_TRACING")
    project: str = Field(default="assignment-omni", alias="LANGSMITH_PROJECT")


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    provider: str = Field(default="ollama", alias="LLM_PROVIDER")
    ollama_model: str = Field(default="llama3.2", alias="OLLAMA_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")


class Settings(BaseModel):
    weather: WeatherSettings
    qdrant: QdrantSettings
    langsmith: LangsmithSettings
    llm: LLMSettings

    @staticmethod
    def load() -> "Settings":
        # Load .env if present
        load_dotenv()
        return Settings(
            weather=WeatherSettings(),
            qdrant=QdrantSettings(),
            langsmith=LangsmithSettings(),
            llm=LLMSettings(),
        )


def ensure_langsmith_env(cfg: Settings) -> None:
    if cfg.langsmith.api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", cfg.langsmith.api_key)
    os.environ.setdefault("LANGSMITH_TRACING", "true" if cfg.langsmith.tracing else "false")
    os.environ.setdefault("LANGSMITH_PROJECT", cfg.langsmith.project)


