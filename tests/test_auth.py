from app.core.auth import hash_password, verify_password, create_access_token
from datetime import timedelta


def test_hash_and_verify_password():
    raw = "mysecret"
    hashed = hash_password(raw)
    assert hashed != raw
    assert verify_password(raw, hashed)


def test_create_access_token():
    token = create_access_token(data={"sub": "Arthur"}, expires_delta=timedelta(minutes=5))
    assert isinstance(token, str)
    assert len(token) > 10
