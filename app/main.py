from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

from routers.pages import page_router
from routers.users import user_router

app = FastAPI()

app.include_router(page_router, tags=["Pages"])
app.include_router(user_router, tags=["Users"])

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == '__main__':
    uvicorn.run("main:app",
                reload=True)
