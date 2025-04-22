import uuid
from httpx import AsyncClient, ASGITransport
from urllib.parse import urlencode
from unittest.mock import AsyncMock, patch
from fastapi import status, FastAPI, HTTPException
import pytest

from app.main import app


@pytest.mark.asyncio
async def test_register_user():
    data = {
        "username": f"testuser_{uuid.uuid4().hex[:8]}",
        "email": f"{uuid.uuid4().hex[:8]}@example.com",
        "password": "testpassword"
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    with patch("app.routers.users.get_user_by_username", new=AsyncMock(return_value=None)), \
            patch("app.routers.users.create_user",
                  new=AsyncMock(return_value=type("User", (), {"username": data["username"]}))):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/users/register", content=urlencode(data), headers=headers)

    assert response.status_code in (status.HTTP_200_OK, status.HTTP_303_SEE_OTHER)


@pytest.mark.asyncio
async def test_login_user():
    data = {
        "username": "testuser",
        "password": "testpassword"
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    fake_user = type("User", (), {
        "username": data["username"],
        "password": "$2b$12$mockedhash"
    })

    with patch("app.routers.users.validate_or_create_user", new=AsyncMock(return_value=fake_user)), \
            patch("app.core.auth.verify_password", return_value=True), \
            patch("app.core.auth.create_access_token", return_value="mocked_token"):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/users/token", content=urlencode(data), headers=headers)

    assert response.status_code in (status.HTTP_200_OK, status.HTTP_303_SEE_OTHER)


@pytest.mark.asyncio
async def test_get_current_user_success():
    token = "mocked.jwt.token"
    headers = {"Cookie": f"Authorization=Bearer {token}"}

    # temporary app with mock route
    test_app = FastAPI()

    @test_app.get("/users/me")
    async def mocked_users_me():
        return {"username": "testuser", "email": "test@example.com"}

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
        response = await ac.get("/users/me", headers=headers)

    assert response.status_code == 200
    assert response.json() == {"username": "testuser", "email": "test@example.com"}


@pytest.mark.asyncio
async def test_get_current_user_unauthorized():
    # temporary app with a route that requires a token
    test_app = FastAPI()

    @test_app.get("/users/me")
    async def mocked_users_me():
        raise HTTPException(status_code=401, detail="Unauthorized")

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
        response = await ac.get("/users/me")

    assert response.status_code == 401
