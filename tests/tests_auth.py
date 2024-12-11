import pytest
from datetime import datetime, timedelta, UTC

from core.auth import create_access_token, verify_access_token


def test_expired_token():
    expired_token = create_access_token({
        "sub": "test_user",
        "exp": int((datetime.now(UTC) - timedelta(minutes=1)).timestamp())
    })

    with pytest.raises(ValueError, match="Token has expired"):
        verify_access_token(expired_token)


def test_invalid_token():
    invalid_token = "abc.def.ghi"
    with pytest.raises(ValueError, match="Token verification failed"):
        verify_access_token(invalid_token)
