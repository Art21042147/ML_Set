from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import RedirectResponse

from app.db.session import get_user_by_username, create_user
from app.db.base import SessionDep
from app.core.auth import verify_password
from app.schemas.users import UserCreate

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="users/token")

user_router = APIRouter()


@user_router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(session: SessionDep, user_data: UserCreate = Depends(UserCreate.as_form)):
    existing_user = await get_user_by_username(session, user_data.username)
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")

    user = await create_user(session, user_data.username, user_data.email, user_data.password)
    return RedirectResponse(url=f"/user_page?user_name={user.username}", status_code=status.HTTP_303_SEE_OTHER)


@user_router.post("/token")
async def login(session: SessionDep, form_data: OAuth2PasswordRequestForm = Depends()):
    user = await get_user_by_username(session, form_data.username)
    if not user or not verify_password(form_data.password, user.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    # После успешного входа перенаправляем на страницу пользователя
    return RedirectResponse(url=f"/user_page?user_name={form_data.username}", status_code=status.HTTP_303_SEE_OTHER)
