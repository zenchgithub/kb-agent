# kb-agent Code Explanation

This document explains the `kb-agent` backend code step by step, starting from how the app starts and then describing each module and key function.

## 1. Entry point and startup behavior

### `main.py`

- This file is very small.
- It defines a `main()` function that prints a greeting and is executed only when the file is run directly.
- It is not the main application entrypoint used by the FastAPI server in normal operation.

```python
def main():
    print("Hello from kb-agent!")

if __name__ == "__main__":
    main()
```

### Actual startup: `api.py`

- The backend is started with a command like:
  `uvicorn api:app --reload --host 0.0.0.0 --port 8000`
- This tells Uvicorn to import `api.py` and use the `app` object defined there.
- Therefore, `api.py` is the effective application entrypoint.

## 2. Environment loading and configuration

### `env_loader.py`

- Provides `load_app_env()`.
- Loads environment variables from one of these files:
  - `.env.dev`
  - `.env`
- If `ENV_FILE` is set, it loads that file instead.
- Uses `python-dotenv` to inject values into `os.environ`.

### `config.py`

- Provides environment helpers used across the app.
- `get_env(name, default)` reads a trimmed environment value.
- `csv_env(name, default)` reads a comma-separated env var and returns trimmed values.
- `cors_origins()` returns a safe list of allowed CORS origins, with defaults for local dev.
- `get_qdrant_client()` creates a `qdrant_client.QdrantClient` instance, using either:
  - `QDRANT_URL` / `QDRANT_API_KEY`
  - or `QDRANT_HOST` / `QDRANT_PORT`
- `load_app_env()` is called inside `config.py`, so env vars are loaded before the client is created.

## 3. Auth: Supabase JWT verification

### `auth.py`

- Responsible for verifying Supabase authentication tokens.
- Uses `python-jose` and `httpx`.
- Key concepts:
  - `SUPABASE_URL`: base Supabase URL.
  - `JWKS_URL`: Supabase JWKS endpoint.
  - `HTTPBearer` security dependency from FastAPI.
- Functions:
  - `_get_jwks()`: fetches the JWKS keys from Supabase and caches them for 10 minutes.
  - `_get_key_for_token(token)`: extracts the JWT header, finds the JWK matching the token `kid`.
  - `_verify_token(token)`: validates the JWT, checks audience and issuer.
  - `get_current_user(credentials)`: FastAPI dependency that extracts the bearer token, verifies it, and returns a user dict:
    - `id` from `sub`
    - `email`, `role`, and raw payload
- This dependency is used in API endpoints to secure requests.

### `check_token.py`

- A small helper script that decodes a Supabase JWT using a secret.
- Not part of the main API flow.
- Uses `jose.jwt.decode()` with `HS256`.

## 4. Database helpers

### `db.py`

- Manages Postgres access using SQLAlchemy.
- Reads `DATABASE_URL` from env vars.
- Creates a global `engine` and session factory `SessionLocal`.
- Provides `get_db()` as a FastAPI dependency.

Key functions:

- `create_conversation(db, user_id)`
  - creates a new `conversations` row with a generated UUID.
  - returns the conversation id.

- `user_owns_conversation(db, conversation_id, user_id)`
  - returns `True` if the given conversation belongs to the user.

- `list_conversations(db, user_id)`
  - returns a list of conversations for a user.
  - sorts by last activity.
  - derives the title from the first user message.

- `load_messages(db, conversation_id)`
  - returns all messages for a conversation in ascending order.
  - each message is normalized to `{'role': ..., 'content': ...}`.

- `load_messages_for_user(db, conversation_id, user_id)`
  - only returns messages if the conversation is owned by the user.
  - adds `created_at` timestamps to each message.

- `load_messages_owned(db, conversation_id, user_id)`
  - alias for `load_messages` with ownership checking.

- `delete_conversation(db, conversation_id, user_id)`
  - deletes a conversation and its messages only when the user owns it.

- `save_message(db, conversation_id, role, content)`
  - inserts a message row into the database.
  - `role` is expected to be `user` or `assistant`.

## 5. Ingestion: PDF processing and vector indexing

### `ingest.py`

- This file is the backend ingestion pipeline for PDFs.
- It routes PDFs through digital extraction, scanned/OCR extraction, and structured identity-field extraction, then writes enriched chunks into Qdrant.
- It uses these libraries:
  - `PyMuPDF` (`fitz`) for digital PDF block/table extraction and scanned page rendering.
  - `pydantic` models for normalized document, page, chunk, OCR, and identity schemas.
  - `openai.OpenAI` to create embeddings.
  - `qdrant_client.models` to define point payload and vector collections.
  - `hashlib` to create stable chunk hashes.
  - `shutil` / `subprocess` / `tempfile` / `tesseract` for local OCR fallback.

Main functions:

- `display_source(path)`
  - returns a cleaned source filename.
  - used to normalize document citation metadata.

- `is_scanned_pdf(pdf_path)`
  - detects image-only PDFs by checking whether pages contain usable text.

- `detect_document_type(pdf_path, sample_text)`
  - classifies documents as passport, ID card, visa, USCIS form, affidavit, lease, or generic PDF.

- `extract_digital_pdf(pdf_path)`
  - uses PyMuPDF block extraction for better reading order than raw PDF stream extraction.
  - keeps tables as separate table chunks where PyMuPDF can detect them.

- `extract_scanned_pdf(pdf_path, ocr_provider=None)`
  - routes scanned PDFs through an OCR provider.
  - defaults to local Tesseract OCR, but the provider interface can be replaced by Azure Document Intelligence or another OCR service.

- `extract_identity_fields(pdf_path, pages, identity_provider=None)`
  - extracts structured ID/passport fields when possible.
  - fields include full name, nationality, passport number, date of birth, issue/expiry dates, and country of issue.

- `chunk_document(document)`
  - creates structure-aware chunks:
    - identity field chunks
    - key-value chunks
    - table chunks
    - body chunks with heading/section metadata.

- `ocr_pdf_page(pdf_path, page_index)`
  - optionally OCRs a PDF page if text extraction is poor.
  - uses PyMuPDF to render page images.
  - runs `tesseract` to extract text.
  - returns OCR text or empty string on failure.

- `extract(path, user_id=None, is_public=False, indexed_by_email=None)`
  - builds a normalized `Document`, chunks it, and returns Qdrant payload dictionaries.
  - attaches retrieval metadata:
    - `source`, `page`, `chunk_index`, `document_id`
    - `doc_type`, `chunk_type`, `field_label`, `person_name`, `keywords`
    - `user_id`, `isPublic`, `indexed_by_email`.
  - returns a list of chunk metadata objects.

- `ensure_collection(name)`
  - creates the Qdrant collection if it does not exist.
  - uses cosine distance and the embedding dimension `EMBED_DIM`.

- `embed_batch(texts)`
  - sends a batch of texts to OpenAI embeddings.
  - returns a list of vectors.

- `ingest_to_qdrant(chunks_with_meta, collection)`
  - ensures the collection exists.
  - batches chunks into groups of 100.
  - embeds each batch.
  - upserts points into Qdrant with payload metadata.
  - payload includes `hash`, `text`, `page`, `source`, and access metadata.

### `ingest_remote_dir.py`

- Lists remote PDF files from a NAS/WebDAV-style directory.
- Uses `requests` and `BeautifulSoup` to parse HTML directory listings.
- Fetches each PDF locally using `fetch_remote.fetch_to_local()`.
- Calls `ingest.ingest()` to index each downloaded PDF.
- `BASE_URL` comes from `NAS_BASE_URL` env var.

## 6. Document helper functions

### `document_links.py`

- Contains two small helper functions:
  - `document_name(raw_source)` returns the filename from a source path.
  - `document_url(raw_source)` builds a document query URL for use in frontend links.

## 7. Knowledge agent workflow

### `graph.py`

- Implements the retrieval and answer-generation pipeline.
- Uses `langgraph` to define a state graph of nodes.
- Uses OpenAI for planning, reranking, and synthesis.
- Uses Qdrant for vector search.
- Uses `collections.yaml` as the list of searchable collections.

Important concepts:

- `State` typed dict defines the working state passed between nodes.
- `COLLECTIONS` loads available collections and descriptions from `collections.yaml`.
- `qc = get_qdrant_client()` creates a Qdrant client.

Pipeline nodes:

### `plan(state)`

- Sends the user question to OpenAI.
- Asks the model to decompose the question into subqueries.
- Asks the model to choose which collections to search.
- Stores `subqueries` and `collections` in state.

### `require_user_id(state)`

- Enforces that `state['user_id']` exists.
- Raises `PermissionError` when missing.

### `user_payload_filter(user_id, raw_source=None)`

- Builds a Qdrant filter that allows:
  - documents owned by the user,
  - public documents,
  - legacy documents with no `user_id`.
- Optionally filters to a specific source.

### `assert_owned_candidates(candidates, user_id)`

- Verifies that each retrieved candidate is either public or belongs to the user.
- Throws an error if any cross-tenant hit is found.

### `access(state)`

- Checks user access.
- Filters requested collections to only valid ones.
- Ensures a fallback to `nas_docs` when nothing valid is selected.
- This avoids accidentally searching documentation-only or not-yet-created collections.

### `normalize_search_text(value)` and `significant_terms(question)`

- Basic text normalization for keyword matching.
- Removes non-alphanumeric characters and stopwords.

### `collection_sources(collection, user_id)`

- Scrolls through Qdrant points and collects all source filenames.
- Only returns sources the user may access.

### `mentioned_sources(question, collections, user_id)`

- Finds document names that appear in the user’s question.
- Helps restrict retrieval to documents the user explicitly mentions.

### `keyword_match_chunks(collections, question, source_matches, user_id)`

- Performs a keyword search over loaded Qdrant payload text.
- Looks for chunks containing all significant terms.
- Returns up to `KEYWORD_TOPK` matches with a higher score.

### `one_search(query, collection, user_id, raw_source=None)`

- Embeds a query with OpenAI.
- Uses `qc.query_points()` to search Qdrant by vector similarity.
- Applies the access filter.
- Returns the raw matching points.

### `retrieve_all(queries, collections, source_matches, user_id)`

- Runs vector search for every query and selected collection.
- If document names are matched, adds source-specific searches.
- Uses `asyncio.gather()` to parallelize across queries.

### `source_context_chunks(collection, raw_source, user_id)`

- Loads a few chunks from a source to preserve document context.
- Returns a small number of nearby chunks for citation.

### `retrieve(state)`

- Builds the set of query texts, including subqueries.
- Uses structured-first retrieval for identity/key-value chunks when planner filters include fields like nationality, DOB, passport number, attorney, address, or employer.
- Finds mentioned sources in the question and history.
- Performs vector search and keyword search.
- Skips collections that are configured but missing in Qdrant.
- Falls back to legacy access-only retrieval if metadata filters return no results.
- Collects and deduplicates candidate chunks.
- Returns candidates in `state['candidates']`.

### `rerank(state)`

- Sends the candidate chunks back to OpenAI for relevance ranking.
- If exact keyword matches exist, it ranks only those.
- Outputs `state['ranked']` in best-to-worst order.

### `normalize(state)`

- Builds a single text context string from ranked chunks.
- Creates a `sources` list with citation metadata.
- Prepares citations with document names and page numbers.

### `synthesize(state)`

- Uses OpenAI chat completion to answer the user.
- Passes a system prompt that requires answers only from provided context.
- Produces `state['answer']`.

### Agent construction

- `build_agent()` defines the graph of nodes:
  1. `plan`
  2. `access`
  3. `retrieve`
  4. `rerank`
  5. `normalize`
  6. `synthesize`
- The graph is compiled and assigned to `agent`.
- When run as a script, it can answer a question from the command line.

## 8. `api.py` details and how requests flow

### Top-level imports used in `api.py`

- `fastapi` and `FastAPI` to create the application.
- `Header`, `HTTPException`, `Depends`, `Request`, `Response` for request handling.
- `CORSMiddleware` for browser access.
- `StaticFiles`, `FileResponse`, `StreamingResponse` for static assets and file downloads.
- `json`, `os`, `datetime`, `Path`, `quote`, `unquote`.
- `requests` for server-to-server HTTP calls.
- `pydantic.BaseModel` for request validation.
- `qdrant_client.models` to build filters.
- `env_loader.load_app_env()` to load env vars.
- `OpenAI` to call OpenAI APIs.
- `graph.agent` and `graph.qc` for the retrieval agent and Qdrant client.
- `auth.get_current_user` for JWT verification.
- `db.*` for conversation/message persistence.
- `ingest.ingest` and `ingest_remote_dir.ingest_remote_dir` for document ingestion.
- `document_links.document_name` and `document_links.document_url` for URL helpers.

### Startup actions in `api.py`

- Calls `load_app_env()` to load `.env`.
- Initializes `oai = OpenAI()`.
- Creates `app = FastAPI(title="Knowledge Agent API")`.
- Reads important env vars like `WEBDAV_USER`, `SUPABASE_URL`, `FRONTEND_URL`, and admin table names.
- Adds CORS middleware using `cors_origins()` from `config.py`.

### Helper functions in `api.py`

- `supabase_rest_headers()`
  - returns headers for Supabase REST requests.
  - requires the service role key.

- `get_admin_record(user_id)`
  - queries a Supabase table to verify admin membership.

- `is_admin_user(user_id)`
  - returns `True` if the user has an admin row.

- `require_admin()` and `require_current_admin()`
  - FastAPI dependencies that enforce admin access.

- `validate_collection_name(collection)`
  - validates Qdrant collection names before using them in admin maintenance routes.
  - allows only letters, numbers, underscores, dots, and hyphens.

- `load_invites()` / `save_invites(invites)`
  - read and write `data/invites.json` for admin invite tracking.

- `require_supabase_service_role()`
  - validates that the service role key is present.

- `clean_upload_filename(filename)`
  - validates and normalizes uploaded PDF filenames.

- `document_candidates(raw_source)` and `find_document_file(raw_source)`
  - resolve a source string to actual files in the `data/remote` directory.

- `format_file_size(size)`
  - formats file size values for display.

- `source_access_filter(user_id, raw_source)`
  - builds a Qdrant filter used by API retrieval endpoints.

- `indexed_documents_filter(user_id, visibility)`
  - returns a Qdrant filter for collections based on visibility modes.

- `payload_is_legacy(payload)` and `payload_is_public(payload)`
  - helpers to detect public or legacy document payloads.

- `source_is_accessible(collection, source, user_id)`
  - checks whether a given source is available to a user.

### Admin Qdrant maintenance endpoints

These endpoints are used by the admin-only Settings UI. The frontend calls the backend with the Supabase JWT, and the backend talks to Qdrant with server-side credentials.

- `GET /admin/qdrant/collections/{collection}`
  - returns collection status such as point count, indexed vector count, segment count, and optimizer status.

- `POST /admin/qdrant/reindex`
  - calls `ingest_remote_dir(collection)` to re-download NAS PDFs and rebuild/upsert Qdrant points.

- `POST /admin/qdrant/delete-source`
  - deletes only points whose Qdrant payload `source` exactly matches the provided value.
  - requires confirmation text: `DELETE SOURCE <source>`.

- `POST /admin/qdrant/delete-collection`
  - deletes the full Qdrant collection.
  - requires confirmation text: `DELETE <collection>`.

## 9. External libraries used in `kb-agent`

Key third-party packages:

- `fastapi`: web framework for the API.
- `uvicorn[standard]`: ASGI server.
- `python-dotenv`: loads `.env` files.
- `pydantic`: request validation.
- `requests`: HTTP client for server-side calls.
- `httpx`: HTTP client used in `auth.py`.
- `python-jose[cryptography]`: JWT parsing and verification.
- `openai`: OpenAI API client.
- `qdrant-client`: Qdrant vector database client.
- `sqlalchemy`: database connection and query execution.
- `psycopg2-binary`: Postgres driver.
- `PyMuPDF`: digital PDF extraction, table detection, and scanned page rendering for OCR.
- `pydantic`: ingestion schemas for documents, chunks, OCR responses, and identity fields.
- `beautifulsoup4`: HTML parsing for remote PDF directory listing.
- `pyyaml`: loads `collections.yaml`.
- `langgraph`: builds the retrieval workflow graph.

## 10. How the system works end-to-end

1. User sends a request to the FastAPI backend.
2. `api.py` verifies the JWT using `auth.get_current_user()`.
3. For chat/conversation endpoints, `db.py` stores or loads messages.
4. For document search, `graph.py` performs planning, retrieval, reranking, normalization, and synthesis.
5. Documents are stored in Qdrant via `ingest.py`.
6. Remote PDF ingestion can be done through `ingest_remote_dir.py`.
7. Document citations are generated by `document_links.py` and passed back to the frontend.

## 11. Notes

- The actual request routing in `api.py` is not listed here because the file is partial in the repo view, but the helper functions and app setup are the main control flow.
- `main.py` is not used in production; `api.py` is the service entrypoint.
- `graph.py` is the core knowledge-retrieval pipeline and is the most important piece for Q&A logic.
- `ingest.py` is the main document indexing pipeline.

---

If you want, I can also add a second markdown file that documents the API endpoints and request/response shapes in `api.py`.
