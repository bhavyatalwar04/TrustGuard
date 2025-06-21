"""
TruthGuard Configuration
Application configuration settings
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, validator
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings"""
    
    # Application settings
    APP_NAME: str = "TruthGuard"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Security
    SECRET_KEY: str = "your-secret-key-here"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"
    
    # Database settings
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/truthguard"
    DATABASE_ECHO: bool = False
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    
    # Redis settings (for caching)
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL: int = 3600  # 1 hour
    
    # AI/ML settings
    OPENAI_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None
    MODEL_NAME: str = "distilbert-base-uncased"
    EMBEDDING_DIMENSION: int = 768
    
    # External APIs
    NEWS_API_KEY: Optional[str] = None
    GOOGLE_SEARCH_API_KEY: Optional[str] = None
    GOOGLE_CSE_ID: Optional[str] = None
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:3001"]
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    
    # Background tasks
    TASK_QUEUE_URL: str = "redis://localhost:6379/1"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # File storage
    UPLOAD_DIR: str = "data/uploads"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # Fact-checking settings
    CONFIDENCE_THRESHOLD: float = 0.7
    MAX_CLAIMS_PER_REQUEST: int = 50
    FACT_CHECK_TIMEOUT: int = 30  # seconds
    
    # Knowledge graph settings
    NEO4J_URI: Optional[str] = None
    NEO4J_USERNAME: Optional[str] = None
    NEO4J_PASSWORD: Optional[str] = None
    
    @validator("ALLOWED_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

# Global settings instance
settings = get_settings()

# Environment-specific configurations
class DevelopmentConfig(Settings):
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    DATABASE_ECHO: bool = True

class ProductionConfig(Settings):
    DEBUG: bool = False
    LOG_LEVEL: str = "WARNING"
    DATABASE_ECHO: bool = False

class TestingConfig(Settings):
    DEBUG: bool = True
    DATABASE_URL: str = "sqlite:///./test.db"
    CACHE_TTL: int = 1

# Configuration factory
def get_config():
    """Get configuration based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()