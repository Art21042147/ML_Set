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
    """
    This function searches for the corresponding `Dataset` object in the `datasets` list by the name,
    specified in `ml_data.dataset`, extracts the necessary parameters from the found `Dataset` object
    and defines a script to run depending on the selected machine learning library (ml_data.library).
    The script is launched with the parameters passed through the command line.
    As a result of the function, the `user_page.html` template is returned with the training results
    passed in the `results` variable.

    :param request: SQLAlchemy AsyncSession object, used to interact with the database.
    :param ml_data: an object containing the user data that was passed in the request.
    """
    # Find the corresponding Dataset object
    dataset_obj = next((ds for ds in datasets if ds.title.lower() == ml_data.dataset.lower()), None)
    if not dataset_obj:
        raise HTTPException(status_code=400, detail="Invalid dataset selected.")

    # Extract the required parameters
    dataset_path = dataset_obj.path
    target_column = dataset_obj.target_column
    save_path = f"ml/predictions/{ml_data.dataset.lower().replace(' ', '_')}_model"

    # Define a script to run
    if ml_data.library.lower() == "scikit-learn":
        script = f"ml/{ml_data.task.lower()}_ml/skl_model.py"
    elif ml_data.library.lower() == "tensorflow":
        script = f"ml/{ml_data.task.lower()}_ml/tf_model.py"
    elif ml_data.library.lower() == "pytorch":
        script = f"ml/{ml_data.task.lower()}_ml/torch_model.py"
    else:
        raise HTTPException(status_code=400, detail="Invalid library selected.")

    # Running a script with passing parameters
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
            "results": output  # Transfer of learning outcomes
        }
    )
