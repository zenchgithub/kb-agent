import os
from urllib.parse import urlparse

from qdrant_client import QdrantClient
from env_loader import load_app_env

load_app_env()


def get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def csv_env(name: str, default: str = "") -> list[str]:
    return [item.strip() for item in get_env(name, default).split(",") if item.strip()]


def cors_origins() -> list[str]:
    default_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ]

    env_value = os.getenv("CORS_ORIGINS")
    if env_value is None or env_value.strip() == "":
        return default_origins

    return [item.strip() for item in env_value.split(",") if item.strip()]


def get_qdrant_client() -> QdrantClient:
    api_key = get_env("QDRANT_API_KEY") or None
    url = get_env("QDRANT_URL")
    if url:
        parsed = urlparse(url)
        port = parsed.port
        if port is None and parsed.scheme == "https":
            port = 443
        return QdrantClient(
            url=url,
            port=port,
            api_key=api_key,
            prefer_grpc=False,
            timeout=60,
            check_compatibility=False,
        )

    host = get_env("QDRANT_HOST", "localhost")
    port = int(get_env("QDRANT_PORT", "6333"))
    return QdrantClient(
        host=host,
        port=port,
        api_key=api_key,
        prefer_grpc=False,
        timeout=60,
        check_compatibility=False,
    )
