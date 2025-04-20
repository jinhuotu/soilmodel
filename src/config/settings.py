from typing import Optional

from pydantic import BaseSettings


class Settings(BaseSettings):
    UPLOAD_DIR: str = "data/uploads"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_FILE_TYPES: list = ["pdf", "docx", "txt", "md"]
    DATABASE_URL = "postgresql://root:postgres@localhost:5432/soilmodel"
    SECRET_KEY: str = "your-secret-key-here"  # 生产环境应从环境变量读取
    TOKEN_EXPIRE_MINUTES: int = 30
    LLM_MODEL: str = "gpt-4-0125-preview"
    LLM_API_KEY: Optional[str] = None
    LLM_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.7

    class Config:
        env_file = ".env"


settings = Settings()