from datetime import datetime, timedelta, UTC
from passlib.context import CryptContext
from authlib.jose import JoseError, JsonWebToken
from fastapi import Request, HTTPException, status

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

    if "exp" not in to_encode:
        expire = datetime.now(UTC) + timedelta(days=30)
        to_encode.update({"exp": int(expire.timestamp())})

    auth_params = auth_data()
    jwt = JsonWebToken(auth_params["algorithm"])

    return jwt.encode(
        header={"alg": auth_params["algorithm"]},
        payload=to_encode,
        key=auth_params["secret_key"]
    )


def verify_access_token(token: str) -> dict:
    try:
        auth_params = auth_data()
        jwt = JsonWebToken(auth_params["algorithm"])
        decoded = jwt.decode(token, auth_params["secret_key"])

        if "exp" in decoded:
            exp_datetime = datetime.fromtimestamp(decoded["exp"], UTC)
            if datetime.now(UTC) > exp_datetime:
                raise ValueError("Token has expired")

        return decoded
    except JoseError as e:
        raise ValueError(f"Token verification failed: {str(e)}") from e


def cookie_extractor(request: Request) -> str:
    token = request.cookies.get("Authorization")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization cookie not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token
