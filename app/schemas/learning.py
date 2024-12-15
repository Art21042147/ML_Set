from pydantic import BaseModel
from typing import List
from fastapi import Form


class Dataset(BaseModel):
    title: str
    file_path: str


datasets: List[Dataset] = [
    Dataset(title="Dataset 1", file_path="dataset1.csv"),
    Dataset(title="Dataset 2", file_path="dataset2.csv"),
    Dataset(title="Dataset 3", file_path="dataset3.csv"),
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
