from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.exceptions import HTTPException

from app.db.base import connection, async_session_maker
from app.db.models import User
from app.schemas.users import UserInDB
from core.auth import get_password_hash, verify_password, create_access_token


async def get_db_session():
    async with async_session_maker() as session:
        yield session


@connection
async def register_user(user: UserInDB, session: AsyncSession):
    existing_user = await session.execute(
        session.query(User).filter_by(username=user.username)
    )
    if existing_user.scalar():
        raise HTTPException(status_code=400, detail="User already exists")

    hashed_password = get_password_hash(user.password)
    new_user = User(
        username=user.username,
        email=user.email,
        password=hashed_password
    )
    session.add(new_user)
    await session.commit()
    return new_user


async def login_user(username: str, password: str, session: AsyncSession):
    user_query = await session.execute(select(User).where(User.username == username))
    user = user_query.scalar()
    if not user or not verify_password(password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@connection
async def add_user(username: str, email: str, password: str, session: AsyncSession):
    new_user = User(username=username, email=email, password=password)
    session.add(new_user)
    await session.commit()
    return new_user
