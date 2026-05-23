from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LM Studio
    lm_studio_base_url: str = "http://localhost:1234/v1"
    lm_studio_api_key: str = "lm-studio"
    lm_studio_model: str = "gemma4"

    # LangFuse
    langfuse_public_key: str = "pk-lf-842235fb-1b92-4a6c-97b6-4ab5c6c6a956"
    langfuse_secret_key: str = "sk-lf-ab50df0c-608f-4657-8f22-307967e3c610"
    langfuse_host: str = "http://localhost:3000"

settings = Settings()
