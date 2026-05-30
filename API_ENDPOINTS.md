# kb-agent API Endpoint Documentation

This file documents the API endpoints implemented in `kb-agent/api.py`, including authentication requirements, request formats, and response shapes.

## Authentication

- Most endpoints require a valid Supabase JWT.
- The JWT is sent as an `Authorization` header:
  - `Authorization: Bearer <token>`
- Token verification happens in `auth.py` using Supabase JWKS.
- Admin-only endpoints additionally validate the user against a Supabase admin lookup table.

## Standard response format

- Successful endpoints generally return JSON objects.
- Error responses use FastAPI HTTP exceptions and typically include `detail`.

---

## `GET /health`

- Public health check.
- No authentication required.
- Response:
  ```json
  {"ok": true}
  ```

---

## `GET /debug-token`

- Debug endpoint that returns the raw `Authorization` header value.
- Useful for verifying if the token reaches the backend.
- No authentication required.
- Response:
  ```json
  {"authorization": "Bearer ..."}
  ```

---

## `POST /query`

- Main non-streaming query endpoint.
- Requires authentication.
- Request body shape:
  ```json
  {
    "question": "string",
    "conversation_id": "string | null"
  }
  ```
- Behavior:
  1. Runs OpenAI moderation on `question`.
  2. Creates a new conversation if `conversation_id` is null.
  3. Loads conversation history from Postgres.
  4. Invokes the LangGraph agent with `question`, `history`, and `user_id`.
  5. Saves both the user message and the assistant response.
- Response shape:
  ```json
  {
    "conversation_id": "string",
    "answer": "string",
    "sources": [
      {
        "id": number,
        "document_name": "string",
        "source": "string",
        "original_source": "string",
        "page": number,
        "matched_text": "string"
      }
    ]
  }
  ```

---

## `POST /query-stream`

- Streaming query endpoint using SSE.
- Requires authentication.
- Request body shape:
  ```json
  {
    "question": "string",
    "conversation_id": "string | null"
  }
  ```
- Behavior:
  1. Runs OpenAI moderation.
  2. Creates or validates a conversation.
  3. Loads history.
  4. Saves the user message before streaming begins.
  5. Invokes the LangGraph agent and then streams tokens from OpenAI.
- Stream events:
  - `: stream-open` initial event.
  - `event: meta` contains `{ "conversation_id", "sources" }`.
  - `data: { "token": "..." }` for each streamed token.
  - `event: done` at the end.

---

## `GET /me`

- Returns the current user profile.
- Requires authentication.
- Response shape:
  ```json
  {
    "id": "string",
    "email": "string | null",
    "role": "admin" | "user",
    "is_admin": true | false
  }
  ```

---

## `GET /conversations`

- Returns the conversation list for the authenticated user.
- Requires authentication.
- Response shape:
  ```json
  {
    "conversations": [
      {
        "id": "string",
        "created_at": "string | null",
        "updated_at": "string | null",
        "title": "string"
      }
    ]
  }
  ```

---

## `GET /conversations/{conversation_id}/messages`

- Returns all messages for a specific conversation.
- Requires authentication.
- Only returns messages if the conversation belongs to the current user.
- Response shape:
  ```json
  {
    "conversation_id": "string",
    "messages": [
      { "role": "user" | "assistant", "content": "string", "created_at": "string" }
    ]
  }
  ```

---

## `DELETE /conversations/{conversation_id}`

- Deletes a conversation and its messages.
- Requires authentication.
- Only deletes when the conversation belongs to the current user.
- Response shape:
  ```json
  {
    "success": true,
    "conversation_id": "string"
  }
  ```

---

## `GET /documents`

- Serves a PDF document by `source`.
- Requires authentication.
- Query parameters:
  - `source`: document path or identifier
  - `collection`: optional, defaults to `nas_docs`
- Behavior:
  - Verifies `source` access through Qdrant metadata.
  - Attempts to serve a local file from `data/remote`.
  - If not found locally, fetches from NAS via `BASE_URL`.
- Response:
  - Returns a `application/pdf` response body.
- Errors:
  - `404` if the document is not accessible or not found.
  - `502` if NAS fetch fails.

---

## `GET /indexed-documents`

- Lists indexed documents in Qdrant.
- Requires authentication.
- Query parameters:
  - `collection` (default: `nas_docs`)
  - `visibility` (`private`, `public`, or `old`; default: `private`)
- Response shape:
  ```json
  {
    "collection": "string",
    "visibility": "string",
    "documents": [
      {
        "id": "string",
        "name": "string",
        "collection": "string",
        "pages": number,
        "chunks": number,
        "original_source": "string",
        "source": "string",
        "isPublic": true | false,
        "isLegacy": true | false,
        "indexed_by_email": "string | null",
        "size": "string",
        "last_indexed": number | null
      }
    ]
  }
  ```

---

## `POST /upload-document`

- Uploads a PDF document and indexes it.
- Requires authentication.
- Upload details are passed in request headers, not body fields.
- Required headers:
  - `X-Filename`: filename of the PDF.
- Optional headers:
  - `X-Collection`: target collection name (`nas_docs` by default).
  - `X-Is-Public`: `true|false|yes|1` (defaults to `false`).
- Request body: raw PDF bytes.
- Behavior:
  1. Validates `X-Filename` and content exists.
  2. Uploads the PDF to NAS via WebDAV-style PUT.
  3. Writes the uploaded bytes locally to `data/remote`.
  4. Calls `ingest.ingest()` to index the document in Qdrant.
     - Digital PDFs are extracted with PyMuPDF.
     - Scanned/image-only pages use OCR fallback.
     - Passports/IDs/visas may produce structured identity/key-value chunks.
     - Qdrant payloads include retrieval metadata such as `doc_type`, `chunk_type`, `field_label`, `person_name`, `keywords`, `user_id`, and `isPublic`.
- Response shape:
  ```json
  {
    "status": "ok",
    "document_name": "string",
    "collection": "string",
    "chunks_indexed": number,
    "isPublic": true | false,
    "indexed_by_email": "string | null",
    "source": "string",
    "nas_url": "string"
  }
  ```

---

## `POST /ingest-remote`

- Ingests remote PDFs from the configured NAS directory.
- Requires admin privileges.
- Query parameter:
  - `collection` (default: `nas_docs`)
- Response shape:
  ```json
  { "status": "ok", "collection": "string" }
  ```

---

## Admin endpoints

### `GET /admin/invites`

- Lists tracked invite records from `data/invites.json`.
- Requires admin authentication.
- Response shape:
  ```json
  {
    "invites": [
      {
        "email": "string",
        "invitedAt": "string",
        "invitedBy": "string"
      }
    ]
  }
  ```

### `POST /admin/invite`

- Sends a Supabase invite email.
- Requires admin authentication.
- Request body shape:
  ```json
  {
    "email": "string",
    "redirectTo": "string | null"
  }
  ```
- Response shape:
  ```json
  { "success": true }
  ```

### `DELETE /admin/invites/{email}`

- Removes an invite record from `data/invites.json`.
- Requires admin authentication.
- Response shape:
  ```json
  { "success": true }
  ```

### `GET /admin/qdrant/collections/{collection}`

- Admin-only Qdrant status check.
- Uses the backend Qdrant client, so the browser never receives the Qdrant API key.
- Response shape:
  ```json
  {
    "status": "ok",
    "collection": "nas_docs",
    "points_count": 149,
    "indexed_vectors_count": 0,
    "segments_count": 3,
    "optimizer_status": "ok"
  }
  ```

### `POST /admin/qdrant/reindex`

- Admin-only reindex operation for the NAS WebDAV folder.
- Calls `ingest_remote_dir(collection)`, which downloads PDFs from the configured NAS URL and upserts chunks into Qdrant.
- Request body:
  ```json
  { "collection": "nas_docs" }
  ```
- Response shape:
  ```json
  { "status": "ok", "collection": "nas_docs" }
  ```

### `POST /admin/qdrant/delete-source`

- Admin-only deletion for one document source inside a collection.
- Deletes only points whose payload `source` exactly matches the requested value.
- Request body:
  ```json
  {
    "collection": "nas_docs",
    "source": "I-485.pdf",
    "confirm": "DELETE SOURCE I-485.pdf"
  }
  ```
- Response shape:
  ```json
  { "status": "deleted", "collection": "nas_docs", "source": "I-485.pdf" }
  ```

### `POST /admin/qdrant/delete-collection`

- Admin-only destructive cleanup for an entire Qdrant collection.
- Requires typed confirmation to avoid accidental deletion.
- Request body:
  ```json
  {
    "collection": "nas_docs",
    "confirm": "DELETE nas_docs"
  }
  ```
- Response shape:
  ```json
  { "status": "deleted", "collection": "nas_docs" }
  ```

---

## `OPTIONS /{path}`

- Generic CORS preflight handler.
- Returns `204 No Content`.
- Sets permissive CORS headers based on request headers.

---

## Static file serving

- `app.mount("/", StaticFiles(directory="static", html=True), name="static")`
- Serves files from the `static/` directory.
- Includes a fallback to `static/index.html` for HTML requests.

---

## Notes on request models

### `Query`

Defined in `api.py` with Pydantic:
```python
class Query(BaseModel):
    question: str
    conversation_id: str | None = None
```

### `InviteRequest`

Defined in `api.py` with Pydantic:
```python
class InviteRequest(BaseModel):
    email: str
    redirectTo: str | None = None
```

---

## Access control details

- `get_current_user()` in `auth.py` validates the JWT and returns the user object.
- `require_admin()` and `require_current_admin()` call `is_admin_user()`.
- `is_admin_user()` checks Supabase admin membership via the `ADMIN_LOOKUP_TABLE`.
- `require_supabase_service_role()` ensures service role credentials are present for Supabase REST calls.

## Request flow summary for `/query`

1. Authenticated request reaches `api.py`.
2. Token validated by `get_current_user()`.
3. Content moderation runs on the question.
4. Conversation is created or validated.
5. Conversation history is loaded.
6. LangGraph agent answers from Qdrant context.
7. The turn is persisted to Postgres.
8. The backend returns `conversation_id`, `answer`, and `sources`.
