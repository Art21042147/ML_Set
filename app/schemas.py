from pydantic import BaseModel
from typing import List

from app.texts import MODELS_DESCRIPTION, MODELS_URL


class LearningModel(BaseModel):
    name: str
    description: str
    url: str


models: List[LearningModel] = [
    LearningModel(
        name="sklearn",
        description=MODELS_DESCRIPTION["sklearn"],
        url=MODELS_URL["sklearn"]
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
