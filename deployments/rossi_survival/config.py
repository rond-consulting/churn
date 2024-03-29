from pydantic import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    API_KEY: str

    class Config:
        # dot.env file
        env_file = ".env"


# New decorator for cache
@lru_cache()
def get_settings():
    return Settings()
