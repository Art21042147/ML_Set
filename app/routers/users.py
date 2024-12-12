from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy.future import select
from fastapi import HTTPException, status

from app.schemas.users import UserCreate
from app.db.models import User
from app.db.base import SessionDep
from core.auth import get_password_hash

user_router = APIRouter()


@user_router.post("/register")
async def register_user(
        session: SessionDep,
        user_data: UserCreate = Depends(UserCreate.as_form), ):
    result = await session.execute(select(User).where(User.email == user_data.email))
    existing_user = result.scalars().first()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email already exists."
        )

    hashed_password = get_password_hash(user_data.password)
    new_user = User(username=user_data.username, email=user_data.email, password=hashed_password)
    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)

    return RedirectResponse(url="/user_page", status_code=status.HTTP_303_SEE_OTHER)
