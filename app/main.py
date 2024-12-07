from fastapi import FastAPI
from app.routers.pages import page_router
import uvicorn


app = FastAPI()

app.include_router(page_router, tags=["Pages"])


if __name__ == '__main__':
    uvicorn.run("main:app",
                reload=True)
