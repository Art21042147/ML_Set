from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from app.db.session import get_user_by_username, create_user
from app.db.base import SessionDep
from core.auth import create_access_token, verify_password, token_expires
from app.schemas.users import UserCreate

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="users/token")

user_router = APIRouter()


@user_router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(session: SessionDep, user_data: UserCreate = Depends(UserCreate.as_form)):
    existing_user = await get_user_by_username(session, user_data.username)
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")
    user = await create_user(session, user_data.username, user_data.email, user_data.password)
    return {"message": "User created successfully", "user_id": user.id}


@user_router.post("/token")
async def login(session: SessionDep, form_data: OAuth2PasswordRequestForm = Depends()):
    user = await get_user_by_username(session, form_data.username)
    if not user or not verify_password(form_data.password, user.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token({"sub": user.username}, expires_delta=token_expires)
    return {"access_token": token, "token_type": "bearer"}
