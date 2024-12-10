from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_SERVER: str
    POSTGRES_PORT: int
    POSTGRES_DB_NAME: str

    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8'
    )


config = Settings()


def db_url():
    return (f"postgresql+asyncpg://{config.POSTGRES_USER}:{config.POSTGRES_PASSWORD}@"
            f"{config.POSTGRES_SERVER}:{config.POSTGRES_PORT}/{config.POSTGRES_DB_NAME}")


def auth_data():
    return {
        "secret_key": config.SECRET_KEY,
        "algorithm": config.ALGORITHM,
        "access_token_expire_minutes": config.ACCESS_TOKEN_EXPIRE_MINUTES
    }
