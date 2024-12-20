from fastapi import HTTPException, APIRouter, Depends
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import subprocess

from app.schemas.ml_mod import models
from app.schemas.ml_set import datasets, LearningSet
from app.texts import INSTRUCTION

templates = Jinja2Templates(directory="templates")

ml_router = APIRouter()

@ml_router.post("/start-learning", response_class=HTMLResponse)
async def start_learning(request: Request, ml_data: LearningSet = Depends(LearningSet.as_form)):
    # Найти соответствующий объект Dataset
    dataset_obj = next((ds for ds in datasets if ds.title.lower() == ml_data.dataset.lower()), None)
    if not dataset_obj:
        raise HTTPException(status_code=400, detail="Invalid dataset selected.")

    # Извлечь необходимые параметры
    dataset_path = dataset_obj.path
    target_column = dataset_obj.target_column
    save_path = f"ml/predictions/{ml_data.dataset.lower().replace(' ', '_')}_model"

    # Определить скрипт для запуска
    if ml_data.library.lower() == "scikit-learn":
        script = f"ml/{ml_data.task.lower()}_ml/skl_model.py"
    elif ml_data.library.lower() == "tensorflow":
        script = f"ml/{ml_data.task.lower()}_ml/tf_model.py"
    elif ml_data.library.lower() == "pytorch":
        script = f"ml/{ml_data.task.lower()}_ml/torch_model.py"
    else:
        raise HTTPException(status_code=400, detail="Invalid library selected.")

    # Запуск скрипта с передачей параметров
    try:
        result = subprocess.run(
            [
                "python", script,
                "--dataset-path", dataset_path,
                "--target-column", target_column,
                "--save-path", save_path
            ],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout
    except subprocess.CalledProcessError as e:
        output = f"Error during training: {e.stderr}"

    return templates.TemplateResponse(
        "user_page.html",
        {
            "request": request,
            "models": models,
            "instruction": INSTRUCTION,
            "user_name": "to Machine Learning",
            "datasets": datasets,
            "tasks": ["Classification", "Regression"],
            "results": output  # Передача результатов обучения
        }
    )
