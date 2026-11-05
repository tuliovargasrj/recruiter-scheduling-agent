from zoneinfo import ZoneInfo

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: SecretStr = Field(min_length=1)
    openai_model: str = "gpt-4o-mini"
    timezone: ZoneInfo = ZoneInfo("America/Sao_Paulo")
