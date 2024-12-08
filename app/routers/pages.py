from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from app.schemas import models
from app.texts import APP_DESCRIPTION

page_router = APIRouter()
templates = Jinja2Templates(directory="templates")


def get_user(request: Request):
    user = request.cookies.get("user_name")
    return {"authenticated": bool(user), "name": user}


@page_router.get("/", response_class=HTMLResponse)
async def main_page(request: Request, user=Depends(get_user)):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "models": models,
            "app_description": APP_DESCRIPTION,
            "user_authenticated": user["authenticated"],
            "user_name": user["name"] if user["authenticated"] else None,
        }
    )
