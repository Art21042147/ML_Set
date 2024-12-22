from fastapi import HTTPException, status, Cookie
import jwt

from app.core.config import config
from app.db.base import SessionDep
from app.db.session import get_user_by_username


def decode_token(token: str):
    """
    Decodes the JWT and returns the decoded data if the token is valid.
    If the token is invalid, the function raises an exception.

    :param token: str, contains the JSON Web Token to be decoded.
    """
    try:
        return jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
        session: SessionDep,
        token: str = Cookie(None, alias="Authorization")
):
    """
    Checks that token is not empty and starts with `Bearer `.
    If it is not, the function throws an exception.
    If token passes the check, the function removes the `Bearer ` prefix from token
    and decodes it using the `decode_token function`.
    The decoding result is stored in the `payload` variable.

    :param session: SQLAlchemy AsyncSession object, used to interact with the database.
    :param token: str, contains the JSON Web Token to be decoded.
    """
    if not token or not token.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token format",
        )
    try:
        # Remove the prefix "Bearer "
        token = token.split("Bearer ")[1]
        # Decrypting the token
        payload = decode_token(token)
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )
        # Checking the user in the database
        user = await get_user_by_username(session, username)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )
        return user
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )
