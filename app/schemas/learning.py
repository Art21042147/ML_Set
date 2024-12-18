from pydantic import BaseModel
from typing import List
from fastapi import Form
from app.texts import DATASET_DESCRIPTION, DATASET_URL


class Dataset(BaseModel):
    title: str
    description: str
    url: str


datasets: List[Dataset] = [
    Dataset(title="Air Pollution",
            description=DATASET_DESCRIPTION["Air Pollution"],
            url=DATASET_URL["Air Pollution"]),
    Dataset(title="Renewable Energy",
            description=DATASET_DESCRIPTION["Renewable Energy"],
            url=DATASET_URL["Renewable Energy"]),
]


class LearningSet(BaseModel):
    library: str
    dataset: str

    @classmethod
    def as_form(
            cls,
            library: str = Form(...),
            dataset: str = Form(...),
    ):
        return cls(library=library, dataset=dataset)
