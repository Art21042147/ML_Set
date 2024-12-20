from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from app.schemas.ml_mod import models
from app.schemas.ml_set import datasets
from app.texts import APP_DESCRIPTION, LOG_INFO, INSTRUCTION

templates = Jinja2Templates(directory="templates")

page_router = APIRouter()


@page_router.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "models": models,
            "app_description": APP_DESCRIPTION,
            "log_info": LOG_INFO
        }
    )


@page_router.get("/user_page", response_class=HTMLResponse)
async def user_page(request: Request, user_name: str):
    return templates.TemplateResponse(
        "user_page.html",
        {
            "request": request,
            "models": models,
            "instruction": INSTRUCTION,
            "user_name": user_name,
            "datasets": datasets,
            "tasks": ["Classification", "Regression"]
        }
    )
