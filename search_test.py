import sys
import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_ADMIN_KEY")
if not openai_key:
    print("Missing OPENAI_API_KEY or OPENAI_ADMIN_KEY. Set it in .env or export it in your shell.")
    sys.exit(1)

oai = OpenAI(api_key=openai_key)

if len(sys.argv) < 2:
    print('Usage: python search_test.py "query" [collection]')
    sys.exit(1)

if len(sys.argv) == 2:
    query = sys.argv[1]
    collection = "default"
else:
    query = " ".join(sys.argv[1:-1])
    collection = sys.argv[-1]

# 1) Embed the query with OpenAI
resp = oai.embeddings.create(model="text-embedding-3-small", input=query)
vec = resp.data[0].embedding

# 2) Call Qdrant's HTTP search API directly
url = "http://localhost:6333/collections/{}/points/search".format(collection)

payload = {
    "vector": vec,
    "limit": 3,
    "with_payload": True,
    "with_vector": False,
}

r = requests.post(url, json=payload)
r.raise_for_status()
data = r.json()

hits = data.get("result", [])

for i, h in enumerate(hits, 1):
    payload = h.get("payload", {}) or {}
    score = h.get("score", 0.0)
    print(f"\n--- Hit {i} · score {score:.3f} · {payload.get('source')} p.{payload.get('page')} ---")
    print((payload.get("text") or "")[:300])