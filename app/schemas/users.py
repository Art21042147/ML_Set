from pydantic import BaseModel, EmailStr
from fastapi import Form

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

    @classmethod
    def as_form(
            cls,
            username: str = Form(...),
            email: str = Form(...),
            password: str = Form(...),
    ):
        return cls(username=username, email=email, password=password)
