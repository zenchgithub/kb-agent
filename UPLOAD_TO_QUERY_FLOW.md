# End-to-End Flow: Upload Document → Ask Question → Receive Answer

This document describes the complete user flow for the `kb-agent` backend, from uploading a PDF document to querying the knowledge base and returning an answer.

## 1. User uploads a document

### 1.1. Client request

- The frontend sends a `POST /upload-document` request.
- Required headers:
  - `Authorization: Bearer <token>`
  - `X-Filename: <filename>.pdf`
- Optional headers:
  - `X-Collection: nas_docs` (default)
  - `X-Is-Public: true|false` (default false)
- Body:
  - raw PDF bytes

### 1.2. Server receives the upload

- `api.py` handles this request in the `upload_document()` endpoint.
- `current_user` is resolved via `auth.get_current_user()`.
- The upload filename is validated by `clean_upload_filename()`.
- The backend confirms the content body is not empty.

### 1.3. Upload file storage

- The server writes the PDF to the local `data/remote` folder.
- It also attempts to upload the PDF to the NAS using `requests.put()`.
- The URL used for NAS upload is built from `BASE_URL`.
- If the NAS upload fails, the request returns an HTTP `502`.

### 1.4. Document indexing

- The endpoint calls `ingest.ingest()` with:
  - local file path
  - collection name
  - current user id
  - public flag
  - user email
- `ingest.py` does the following:
  1. Detects whether the PDF is digital or scanned.
  2. Extracts digital PDFs with PyMuPDF block/table extraction.
  3. Uses OCR fallback for scanned or image-only pages.
  4. Detects document type such as passport, visa, USCIS form, affidavit, lease, or generic PDF.
  5. Builds structured chunks for identity fields, key-value lines, tables, and body text.
  6. Attaches rich metadata:
     - `source`, `page`, `chunk_index`, `document_id`
     - `doc_type`, `chunk_type`, `field_label`, `person_name`, `keywords`
     - `user_id`, `isPublic`, `indexed_by_email`
  7. Creates a Qdrant collection if needed.
  8. Embeds chunk text with OpenAI embeddings.
  9. Upserts vector points into Qdrant.

### 1.5. Upload response

- After successful ingest, the backend returns:
  - `status: ok`
  - `document_name`
  - `collection`
  - `chunks_indexed`
  - `isPublic`
  - `indexed_by_email`
  - `source`
  - `nas_url`

## 2. User asks a question

### 2.1. Client request

- The frontend sends a `POST /query` or `POST /query-stream` request.
- Request body JSON:
  ```json
  {
    "question": "<user question>",
    "conversation_id": "<id>" // or null for new conversation
  }
  ```
- The `Authorization` header is required.

### 2.2. Request authentication

- The backend uses `auth.get_current_user()`.
- The JWT is validated against Supabase JWKS.
- The token payload is parsed into a user object with `id`, `email`, and `role`.

### 2.3. Moderation

- The backend calls OpenAI moderation on the question.
- If the question is flagged, the request fails with HTTP `400`.

### 2.4. Conversation handling

- If `conversation_id` is null:
  - `db.create_conversation()` inserts a new row in Postgres.
  - A new conversation id is returned.
- If `conversation_id` exists:
  - `db.load_messages_owned()` verifies the user owns the conversation.
  - If not owned, the request returns HTTP `404`.

### 2.5. Load conversation history

- The backend calls `db.load_messages()`.
- It loads all messages for the conversation in chronological order.
- History is provided to the LangGraph agent.

## 3. Knowledge retrieval and answer generation

### 3.1. Invoke the LangGraph agent

- The backend calls `graph.agent.invoke()` with state:
  - `question`
  - `history`
  - `user_id`
- The LangGraph agent executes these nodes in order:
  1. `plan`
  2. `access`
  3. `retrieve`
  4. `rerank`
  5. `normalize`
  6. `synthesize`

### 3.2. Planning step

- `graph.plan()` sends a prompt to OpenAI.
- The model decomposes the user question into subqueries.
- It also chooses which document collections to search and returns retrieval filters.
- The result populates `state['subqueries']`, `state['collections']`, and `state['filters']`.

### 3.3. Access enforcement

- `graph.access()` validates that the chosen collections are allowed.
- It ensures the user only searches valid collections.
- If the selection is empty or invalid, it falls back to `nas_docs`.
- During retrieval, collections that are not present in Qdrant are skipped safely.

### 3.4. Retrieval from Qdrant

- `graph.retrieve()` performs vector search:
  - first checks structured identity/key-value chunks when the question asks for fields like nationality, DOB, passport number, attorney, address, or employer
  - obtains candidate chunks from Qdrant using vector search
  - applies filters to enforce user access and public documents
  - applies metadata filters such as `doc_type` when available
  - adds keyword-matched chunks from the local payload text
  - deduplicates candidates
- It may also use source-matching logic to focus on documents mentioned in the question.
- `state['candidates']` is populated with document chunks.

### 3.5. Reranking chunks

- `graph.rerank()` sends the candidate chunks to OpenAI.
- The model ranks chunks by relevance to the question.
- The best chunk indices are returned in `state['ranked']`.

### 3.6. Normalizing context

- `graph.normalize()` builds the final context string.
- It creates a list of citation metadata in `state['sources']`.
- Each source item includes:
  - document name
  - source URL
  - page
  - matched text

### 3.7. Synthesizing the answer

- `graph.synthesize()` sends the compiled context and question to OpenAI.
- It uses a system prompt that instructs the model to answer strictly from provided context.
- The output is stored in `state['answer']`.

## 4. Returning the result

### 4.1. Non-streaming `/query`

- After the agent runs, the backend saves:
  - the user message via `db.save_message()`
  - the assistant answer via `db.save_message()`
- Response JSON includes:
  - `conversation_id`
  - `answer`
  - `sources`

### 4.2. Streaming `/query-stream`

- The backend saves the user message before streaming.
- It then opens a server-sent event stream.
- It first sends metadata:
  - `conversation_id`
  - `sources`
- It then streams tokens from OpenAI back to the client.
- Finally, it sends a `done` event.

## 5. Relevant backend modules in the flow

- `api.py`
  - upload handling
  - query endpoints
  - conversation management
- `auth.py`
  - JWT validation
  - Supabase auth integration
- `db.py`
  - conversation and message persistence
- `ingest.py`
  - PDF text extraction
  - chunking
  - embedding and Qdrant indexing
- `graph.py`
  - planning, retrieval, reranking, normalization, synthesis
- `config.py`
  - Qdrant client creation
  - CORS origins
- `env_loader.py`
  - env file loading

## 6. Summary of data flow

1. User uploads a PDF to `/upload-document`.
2. Backend stores it locally and uploads it to NAS.
3. Backend extracts text, creates embeddings, and stores vectors in Qdrant.
4. User asks a question with `/query` or `/query-stream`.
5. Backend verifies auth and conversation ownership.
6. The LangGraph agent searches Qdrant, ranks chunks, and synthesizes the answer.
7. Backend stores the conversation turn and returns the answer and citations.
