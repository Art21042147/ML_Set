from fastapi import FastAPI
from routers.pages import page_router
from routers.users import user_router
import uvicorn


app = FastAPI()

app.include_router(page_router, tags=["Pages"])
app.include_router(user_router, tags=["Users"])


if __name__ == '__main__':
    uvicorn.run("main:app",
                reload=True)
