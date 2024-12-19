from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from app.db.base import Base, engine

from routers.pages import page_router
from routers.users import user_router
from routers.ml_start import ml_router


@asynccontextmanager
async def lifespan(_):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(page_router)
app.include_router(user_router, prefix="/users", tags=["Users"])
app.include_router(ml_router, tags=["Start Learning"])

app.mount("/app/static", StaticFiles(directory="static"), name="static")

if __name__ == '__main__':
    import uvicorn

    uvicorn.run("main:app", reload=True)
