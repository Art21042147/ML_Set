from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from app.schemas import models

page_router = APIRouter()
templates = Jinja2Templates(directory="templates")


@page_router.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "models": models}
    )


@page_router.get("/models/{model_name}", response_class=HTMLResponse)
async def model_page(request: Request, model_name: str):
    for model in models:
        if model.id == model_name:
            return templates.TemplateResponse(
                "ml_mod.html", {"request": request, "model": model}
            )
    return templates.TemplateResponse(
        "404.html", {"request": request, "model_id": model_name}
    )
