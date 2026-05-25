import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()


def get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def csv_env(name: str, default: str = "") -> list[str]:
    return [item.strip() for item in get_env(name, default).split(",") if item.strip()]


def cors_origins() -> list[str]:
    return csv_env(
        "CORS_ORIGINS",
        ",".join([
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5174",
        ]),
    )


def get_qdrant_client() -> QdrantClient:
    api_key = get_env("QDRANT_API_KEY") or None
    url = get_env("QDRANT_URL")
    if url:
        return QdrantClient(url=url, api_key=api_key)

    host = get_env("QDRANT_HOST", "localhost")
    port = int(get_env("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port, api_key=api_key)
