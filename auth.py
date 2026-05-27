# auth.py
import os
import time
from typing import Dict, Any

from jose import jwt, JWTError
import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from env_loader import load_app_env

load_app_env()

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
JWKS_URL = os.getenv("SUPABASE_JWKS_URL", f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json")

security = HTTPBearer()

# In-memory JWKS cache
_jwks_cache: Dict[str, Any] = {
    "keys": None,
    "fetched_at": 0.0,
    "ttl": 600.0,  # 10 minutes
}


def _get_jwks() -> Dict[str, Any]:
    """Fetch JWKS from Supabase, with basic caching."""
    now = time.time()
    if _jwks_cache["keys"] is not None and now - _jwks_cache["fetched_at"] < _jwks_cache["ttl"]:
        return _jwks_cache["keys"]

    with httpx.Client(timeout=5.0) as client:
        resp = client.get(JWKS_URL)
        resp.raise_for_status()
        jwks = resp.json()

    _jwks_cache["keys"] = jwks
    _jwks_cache["fetched_at"] = now
    return jwks


def _get_key_for_token(token: str) -> Dict[str, Any]:
    """Find the JWK whose kid matches the token header."""
    header = jwt.get_unverified_header(token)
    kid = header.get("kid")
    if not kid:
        raise JWTError("Token missing 'kid' header")

    jwks = _get_jwks()
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            return key

    raise JWTError(f"No matching JWK for kid={kid}")


def _verify_token(token: str) -> Dict[str, Any]:
    """Verify a Supabase Auth JWT using JWKS."""
    key = _get_key_for_token(token)
    alg = key.get("alg", "ES256")  # Supabase recommends ES256

    payload = jwt.decode(
        token,
        key,
        algorithms=[alg],
        audience="authenticated",
        issuer=f"{SUPABASE_URL}/auth/v1",
    )
    return payload


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    """
    FastAPI dependency: verifies Authorization header and returns a user dict.
    """
    token = credentials.credentials
    try:
        payload = _verify_token(token)
    except JWTError as e:
        # You can log e for debugging
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        ) from e

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject",
        )

    return {
        "id": user_id,
        "email": payload.get("email"),
        "role": payload.get("role"),
        "raw": payload,
    }
