import sys
import uuid
import hashlib
import os
from pathlib import Path
from urllib.parse import unquote
from pypdf import PdfReader
from openai import OpenAI
from qdrant_client import models
from config import get_qdrant_client
from env_loader import load_app_env

load_app_env()

# OpenAI + Qdrant clients (assumes local Qdrant)
# Require an OpenAI API key in env for safety; prefer `OPENAI_API_KEY`
_openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_ADMIN_KEY")
if not _openai_key:
	print("Missing OPENAI_API_KEY or OPENAI_ADMIN_KEY. Set it in .env or export it in your shell.")
	sys.exit(1)

oai = OpenAI(api_key=_openai_key)
qc = get_qdrant_client()

"""
Simple PDF text extractor script.

This module provides a tiny command-line helper to read a PDF file and
print a short preview of each page's extracted text. It's intentionally
minimal: the goal is to inspect that text extraction is working and to
see the first ~500 characters of each page for quick debugging.

Notes:
- Uses `pypdf.PdfReader` to read the file and access `reader.pages`.
- `page.extract_text()` may return None when no text can be extracted
  (for scanned pages or complex encodings); we normalize that to an
  empty string before measuring or printing.
- This script prints only the first 500 characters per page to avoid
  flooding the terminal for large pages.
"""


# Chunking configuration (word counts)
# CHUNK_SIZE: approximate number of words per chunk
# OVERLAP: number of words to overlap between consecutive chunks
# MIN_CHARS: discard chunks smaller than this many characters
CHUNK_SIZE = 800
OVERLAP = 100
MIN_CHARS = 50
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536


def display_source(path: Path) -> str:
	"""Return a clean filename for API citations."""
	return unquote(path.name).strip()


def chunk_text(text: str):
	"""Yield chunks of `text` as strings.

	Splits `text` into words and yields pieces of approximately
	`CHUNK_SIZE` words, advancing by `CHUNK_SIZE - OVERLAP` words each
	step. Chunks shorter than `MIN_CHARS` characters are skipped.
	"""

	words = text.split()
	if not words:
		return

	step = CHUNK_SIZE - OVERLAP
	if step <= 0:
		step = CHUNK_SIZE

	for i in range(0, len(words), step):
		piece_words = words[i : i + CHUNK_SIZE]
		piece = " ".join(piece_words).strip()
		if len(piece) > MIN_CHARS:
			yield piece


def extract(
	path: str,
	user_id: str | None = None,
	is_public: bool = False,
	indexed_by_email: str | None = None,
):
	"""Extract and print a short preview of each page in `path`.

	Parameters
	- path: filesystem path to a PDF file. Can be a str or path-like.

	Behavior
	- If the file does not exist, prints an error and returns.
	- Otherwise opens the PDF with `PdfReader`, prints the page count,
	  then iterates pages and prints a 500-character preview per page.
	"""

	# Normalize to a Path object for convenience and safer existence checks
	path = Path(path)

	# Inform the user if the file is missing rather than letting PdfReader
	# raise a lower-level exception. This makes the script friendlier.
	if not path.exists():
		print(f"File not found: {path}")
		return

	# Create a PdfReader instance which parses the PDF structure.
	# Note: PdfReader may still raise for corrupt or unsupported PDFs.
	reader = PdfReader(path)

	# `reader.pages` is a sequence-like of page objects.
	print(f"Pages: {len(reader.pages)}")

	# Collect chunks with metadata for the whole document if needed.
	chunks_with_meta = []

	# Loop over pages and extract text, then chunk each page's full text.
	for i, page in enumerate(reader.pages, 1):
		# extract_text() can return None; use empty string as fallback.
		text = page.extract_text() or ""

		# Create chunks from the full page text.
		page_chunks = list(chunk_text(text))

		# Attach metadata to each chunk (source path, page number, index).
		for j, chunk in enumerate(page_chunks, 1):
			meta = {
				"source": display_source(path),
				"page": i,
				"chunk_index": j,
				"text": chunk,
			}
			if user_id:
				meta["user_id"] = str(user_id)
			meta["isPublic"] = bool(is_public)
			if indexed_by_email:
				meta["indexed_by_email"] = indexed_by_email
				meta["indexed_by"] = indexed_by_email
			chunks_with_meta.append(meta)

		# Print summary info for this page: number of chunks and first preview.
		print(f"--- Page {i} ({len(text)} chars) ---")
		print(f"Chunks: {len(page_chunks)}")
		if page_chunks:
			# Preview the first chunk (trim to 500 chars for display).
			print("First chunk:", page_chunks[0][:500])
		print()

	# Document-level summary: total chunks found
	print(f"Total chunks collected: {len(chunks_with_meta)}")
    
	# Return collected chunks for possible downstream ingestion
	return chunks_with_meta


def ensure_collection(name: str):
	"""Create the Qdrant collection if it doesn't exist."""
	try:
		qc.get_collection(name)
	except Exception:
		qc.create_collection(
			collection_name=name,
			vectors_config=models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE),
		)
		print(f"Created collection: {name}")


def embed_batch(texts: list[str]) -> list[list[float]]:
	"""Get embeddings for a batch of texts from OpenAI.

	Returns a list of vectors (lists of floats) matching `texts` order.
	"""
	resp = oai.embeddings.create(model=EMBED_MODEL, input=texts)
	return [d.embedding for d in resp.data]


def ingest_to_qdrant(chunks_with_meta: list[dict], collection: str):
	"""Embed chunks and upsert into Qdrant with metadata payload.

	- chunks_with_meta: list of dicts with at least keys `text`, `source`,
	  `page`, `chunk_index`.
	- collection: target Qdrant collection name.
	"""

	ensure_collection(collection)

	# Batch size for embedding API calls
	BATCH = 100
	total_upserted = 0

	for i in range(0, len(chunks_with_meta), BATCH):
		batch = chunks_with_meta[i : i + BATCH]
		texts = [c["text"] for c in batch]

		# Request embeddings
		embeddings = embed_batch(texts)

		points = []
		for c, vec in zip(batch, embeddings):
			h = hashlib.md5(c["text"].encode()).hexdigest()
			payload = {**c, "hash": h}
			points.append(
				models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload)
			)

		# Upsert this batch into Qdrant
		qc.upsert(collection_name=collection, points=points)
		total_upserted += len(points)

	print(f"✓ Ingested {total_upserted} chunks into '{collection}'")


def ingest(
	pdf_path: str,
	collection: str,
	user_id: str | None = None,
	is_public: bool = False,
	indexed_by_email: str | None = None,
):
	"""High-level helper: extract chunks from a PDF and push them into Qdrant."""
	chunks_with_meta = extract(
		pdf_path,
		user_id=user_id,
		is_public=is_public,
		indexed_by_email=indexed_by_email,
	)
	if not chunks_with_meta:
		print("No chunks to ingest.")
		return
	ingest_to_qdrant(chunks_with_meta, collection)


if __name__ == "__main__":

	# Basic CLI argument validation so the user sees a helpful message
	# instead of an index error when no path is provided.
	if len(sys.argv) < 2:
		print("Usage: python ingest.py <pdf-path> [collection]")
		sys.exit(1)

	pdf_path = sys.argv[1]
	collection = sys.argv[2] if len(sys.argv) > 2 else None

	# If a collection was provided, run the high-level ingest helper.
	if collection:
		ingest(pdf_path, collection)
	else:
		# Otherwise just extract and print a preview of chunks.
		extract(pdf_path)
