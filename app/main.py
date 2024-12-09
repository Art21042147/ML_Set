from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager

from core.config import config
from db.base import Base
from routers.pages import page_router
from routers.users import user_router

engine = create_engine(config.db.url)
SessionLocal = sessionmaker(bind=engine)


@asynccontextmanager
async def lifespan(_):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(page_router, tags=["Pages"])
app.include_router(user_router, tags=["Users"])

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == '__main__':
    import uvicorn

    uvicorn.run("main:app", reload=True)
