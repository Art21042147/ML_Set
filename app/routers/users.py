from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import RedirectResponse
from datetime import timedelta

from app.core.config import config
from app.core.auth import create_access_token
from app.core.auth import verify_password, hash_password
from app.db.session import get_user_by_username, create_user
from app.db.base import SessionDep
from app.schemas.users import UserCreate

user_router = APIRouter()


async def validate_or_create_user(
        session: SessionDep,
        username: str,
        password: str = None,
        email: str = None,
        create_new: bool = False
):
    user = await get_user_by_username(session, username)

    if create_new:  # Логика для регистрации
        if user:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")
        # Создаём нового пользователя
        hashed_password = hash_password(password)
        user = await create_user(session, username, email, hashed_password)
    else:  # Логика для авторизации
        if not user or not verify_password(password, user.password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    return user


@user_router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(session: SessionDep, user_data: UserCreate = Depends(UserCreate.as_form)):
    # Используем validate_or_create_user для регистрации
    user = await validate_or_create_user(
        session,
        username=user_data.username,
        password=user_data.password,
        email=user_data.email,
        create_new=True
    )

    # Генерируем токен
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    # Создаём ответ с токеном в cookie
    response = RedirectResponse(
        url=f"/user_page?user_name={user.username}",
        status_code=status.HTTP_303_SEE_OTHER
    )
    response.set_cookie(
        key="Authorization",
        value=f"Bearer {access_token}",
        httponly=True
    )
    return response


@user_router.post("/token")
async def login(session: SessionDep, form_data: OAuth2PasswordRequestForm = Depends()):
    # Используем validate_or_create_user для авторизации
    user = await validate_or_create_user(
        session,
        username=form_data.username,
        password=form_data.password
    )

    # Генерируем токен
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    # Создаём ответ с токеном в cookie
    response = RedirectResponse(
        url=f"/user_page?user_name={user.username}",
        status_code=status.HTTP_303_SEE_OTHER
    )
    response.set_cookie(
        key="Authorization",
        value=f"Bearer {access_token}",
        httponly=True
    )
    return response

