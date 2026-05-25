from pathlib import Path
from urllib.parse import quote, unquote


def document_name(raw_source: str) -> str:
    return unquote(Path(raw_source).name).strip()


def document_url(raw_source: str) -> str:
    return f"/documents?source={quote(raw_source, safe='')}"
