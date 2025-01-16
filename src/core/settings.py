from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env', env_file_encoding='utf-8', extra='allow'
    )

    DATABASE_URL: str = 'postgresql://admin:admin@localhost:5432/vector_db'
    MAX_TOKENS_PER_MINUITE: int = 25000


settings = Settings()
