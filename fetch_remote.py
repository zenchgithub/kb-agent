import os
import requests
from urllib.parse import unquote, urlparse
from env_loader import load_app_env

load_app_env()

WEBDAV_USER = os.getenv("WEBDAV_USER")
WEBDAV_PASS = os.getenv("WEBDAV_PASS")

def fetch_to_local(url: str, dest_dir: str = "data/remote") -> str:
    os.makedirs(dest_dir, exist_ok=True)
    name = unquote(os.path.basename(urlparse(url).path)).strip()
    local_path = os.path.join(dest_dir, name)

    auth = (WEBDAV_USER, WEBDAV_PASS) if WEBDAV_USER else None
    resp = requests.get(url, auth=auth, stream=True, timeout=60)
    resp.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"Downloaded {url} -> {local_path}")
    return local_path
