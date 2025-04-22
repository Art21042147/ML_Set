import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import status

from app.main import app
from app.core.auth_depends import get_current_user
from app.db.models import User


# mock for get_current_user
async def override_get_current_user():
    return User(username="testuser", email="test@example.com")


@pytest.mark.asyncio
async def test_main_page():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == status.HTTP_200_OK
    assert "text/html" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_user_page_authorized():
    app.dependency_overrides[get_current_user] = override_get_current_user

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/user_page")

    assert response.status_code == status.HTTP_200_OK
    assert "text/html" in response.headers["content-type"]
    assert "testuser" in response.text
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_user_page_unauthorized():
    app.dependency_overrides.clear()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/user_page")

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
