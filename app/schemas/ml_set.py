from pydantic import BaseModel
from typing import List
from fastapi import Form
from app.texts import DATASET_DESCRIPTION, DATASET_URL


class Dataset(BaseModel):
    title: str
    description: str
    url: str
    path: str
    target_column: str


datasets: List[Dataset] = [
    Dataset(title="Air Pollution",
            description=DATASET_DESCRIPTION["Air Pollution"],
            url=DATASET_URL["Air Pollution"],
            path="ml/datasets/pollution_dataset.csv",
            target_column="Air Quality"),
    Dataset(title="Renewable Energy",
            description=DATASET_DESCRIPTION["Renewable Energy"],
            url=DATASET_URL["Renewable Energy"],
            path="ml/datasets/renewable_energy.csv",
            target_column="Energy_Level")
]


class LearningSet(BaseModel):
    library: str
    dataset: str
    task: str

    @classmethod
    def as_form(
            cls,
            library: str = Form(...),
            dataset: str = Form(...),
            task: str = Form(...),
    ):
        return cls(library=library, dataset=dataset, task=task)
