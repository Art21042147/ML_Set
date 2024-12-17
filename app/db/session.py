from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.models import User
from core.auth import hash_password


async def get_user_by_username(session: AsyncSession, username: str) -> User | None:
    result = await session.execute(select(User).where(User.username == username))
    return result.scalars().first()


async def create_user(session: AsyncSession, username: str, email: str, password: str) -> User:
    hashed_pwd = hash_password(password)
    user = User(username=username, email=email, password=hashed_pwd)
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user
