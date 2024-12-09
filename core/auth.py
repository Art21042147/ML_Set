from datetime import datetime, timedelta, timezone
from jose import jwt
from core.config import auth_data


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
