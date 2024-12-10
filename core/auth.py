from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
from jose import jwt

from core.config import config

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def auth_data():
    return {
        "secret_key": config.SECRET_KEY,
        "algorithm": config.ALGORITHM,
        "access_token_expire_minutes": config.ACCESS_TOKEN_EXPIRE_MINUTES
    }


def verify_password(plain_password, hashed_password) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=30)
    to_encode.update({"exp": expire})

    auth_params = auth_data()

    encoded_jwt = jwt.encode(
        to_encode,
        auth_params['secret_key'],
        algorithm=auth_params['algorithm']
    )
    return encoded_jwt
