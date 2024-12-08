from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr


class Settings(BaseSettings):
    postgres_user = SecretStr
    postgres_password = SecretStr
    postgres_server = SecretStr
    postgres_port = int
    postgres_db = SecretStr
    db_url = (f'postgresql://'
              f'{postgres_user}:{postgres_password}@'
              f'{postgres_server}:{postgres_port}/{postgres_db}')
    model_config = SettingsConfigDict(env_file='.env',
                                      env_file_encoding='utf-8')


config = Settings()
