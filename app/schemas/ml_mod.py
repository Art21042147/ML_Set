from pydantic import BaseModel
from typing import List

from app.texts import MODELS_DESCRIPTION, MODELS_URL


class LearningModel(BaseModel):
    name: str
    description: str
    url: str


models: List[LearningModel] = [
    LearningModel(
        name="scikit-learn",
        description=MODELS_DESCRIPTION["scikit-learn"],
        url=MODELS_URL["scikit-learn"]
    ),
    LearningModel(
        name="tensorflow",
        description=MODELS_DESCRIPTION["tensorflow"],
        url=MODELS_URL["tensorflow"]
    ),
    LearningModel(
        name="pytorch",
        description=MODELS_DESCRIPTION["pytorch"],
        url=MODELS_URL["pytorch"]
    ),
]
