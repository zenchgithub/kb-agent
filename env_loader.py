import os
from pathlib import Path

from dotenv import load_dotenv


def load_app_env() -> Path | None:
    """Load the backend env file.

    Local development defaults to `.env.dev`. Set ENV_FILE to override:
    `ENV_FILE=.env.qa .venv/bin/uvicorn api:app --reload --port 8000`.
    Deployed environments normally provide real environment variables and do
    not need an env file.
    """

    root = Path(__file__).resolve().parent
    explicit_env_file = os.getenv("ENV_FILE")

    if explicit_env_file:
        path = Path(explicit_env_file)
        if not path.is_absolute():
            path = root / path
        candidates = [path]
    else:
        candidates = [root / ".env.dev", root / ".env"]

    for path in candidates:
        if path.exists():
            load_dotenv(path, override=False)
            return path

    load_dotenv(override=False)
    return None
