from pydantic import BaseModel
from typing import List


class LearningModel(BaseModel):
    name: str
    description: str


models: List[LearningModel] = [
    LearningModel(name="sklearn", description="Scikit-learn"),
    LearningModel(name="tensorflow", description="TensorFlow"),
    LearningModel(name="pytorch", description="PyTorch")
]
