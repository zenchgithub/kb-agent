# ingest_remote_dir.py
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from env_loader import load_app_env
from fetch_remote import fetch_to_local
from ingest import ingest

load_app_env()

BASE_URL = os.getenv("NAS_BASE_URL", "https://files.bezench.com/kb-agent/")

def list_pdfs():
    auth = (os.getenv("WEBDAV_USER"), os.getenv("WEBDAV_PASS")) if os.getenv("WEBDAV_USER") else None
    resp = requests.get(BASE_URL, auth=auth, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    pdf_urls = []
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        if href.lower().endswith(".pdf"):
            pdf_urls.append(urljoin(BASE_URL, href))
    return pdf_urls

def ingest_remote_dir(collection: str = "nas_docs"):
    pdfs = list_pdfs()
    print(f"Found {len(pdfs)} PDF(s):")
    for url in pdfs:
        print("  ", url)
        local_path = fetch_to_local(url)
        ingest(local_path, collection)

if __name__ == "__main__":
    import sys
    collection = sys.argv[1] if len(sys.argv) > 1 else "nas_docs"
    ingest_remote_dir(collection)
