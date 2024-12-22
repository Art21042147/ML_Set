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


# Function of checking the presence of a user in the database, and creating a new one if it is absent.
async def validate_or_create_user(
        session: SessionDep,
        username: str,
        password: str = None,
        email: str = None,
        create_new: bool = False
):
    user = await get_user_by_username(session, username)

    if create_new:  # Register logic
        if user:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")
        # Create a new user
        hashed_password = hash_password(password)
        user = await create_user(session, username, email, hashed_password)
    else:  # Log In logic
        if not user or not verify_password(password, user.password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    return user


@user_router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(session: SessionDep, user_data: UserCreate = Depends(UserCreate.as_form)):
    """
    This function is called when a POST request is sent to `/register` endpoint.
    Returns a `response` object that will be sent to the user.

    :param session: SQLAlchemy AsyncSession object, used to interact with the database.
    :param user_data: an object containing the user data that was passed in the request.
    """
    # Use validate_or_create_user for registration
    user = await validate_or_create_user(
        session,
        username=user_data.username,
        password=user_data.password,
        email=user_data.email,
        create_new=True
    )

    # Generate a token
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    # Create a response with a token in a cookie
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
    """
    This function is called when a POST request is sent to `/token` endpoint.
    Returns a `response` object that will be sent to the user.

    :param session: SQLAlchemy AsyncSession object, used to interact with the database.
    :param form_data: a form containing authorization data.
    """
    # Use validate_or_create_user for authorization
    user = await validate_or_create_user(
        session,
        username=form_data.username,
        password=form_data.password
    )

    # Generate a token
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    # Create a response with a token in a cookie
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
