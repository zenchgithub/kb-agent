import sys
from env_loader import load_app_env
from fetch_remote import fetch_to_local
from ingest import ingest

load_app_env()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_remote.py <url> [collection]")
        sys.exit(1)

    url = sys.argv[1]
    collection = sys.argv[2] if len(sys.argv) > 2 else "personal_docs"

    local_path = fetch_to_local(url)
    ingest(local_path, collection)
