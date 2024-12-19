from fastapi import HTTPException, APIRouter, Depends
from fastapi.responses import JSONResponse
import subprocess

from app.schemas.ml_set import datasets, LearningSet

ml_router = APIRouter()

@ml_router.post("/start-learning")
async def start_learning(ml_data: LearningSet = Depends(LearningSet.as_form)):
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
        return JSONResponse(content={"message": "Learning completed", "output": result.stdout})
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error during training: {e.stderr}")
