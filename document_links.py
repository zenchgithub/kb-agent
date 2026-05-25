import os
from pathlib import Path
from urllib.parse import quote, unquote


def document_name(raw_source: str) -> str:
    return unquote(Path(raw_source).name).strip()


def document_url(raw_source: str) -> str:
    base_url = os.getenv("NAS_BASE_URL", "").strip()
    name = document_name(raw_source)
    if base_url and name:
        return f"{base_url.rstrip('/')}/{quote(name)}"
    return f"/documents?source={quote(raw_source, safe='')}"
