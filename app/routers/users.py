from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import register_user, login_user, get_db_session
from app.schemas.users import UserInDB, Token

templates = Jinja2Templates(directory="templates")
user_router = APIRouter()


@user_router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("registration.html", {"request": request})


@user_router.post("/register", response_model=UserInDB)
async def register(user: UserInDB, db: AsyncSession = Depends(get_db_session)):
    return await register_user(user, session=db)


@user_router.post("/login", response_model=Token)
async def login(username: str, password: str, db: AsyncSession = Depends(get_db_session)):
    try:
        return await login_user(username, password, session=db)
    except HTTPException as e:
        raise e
