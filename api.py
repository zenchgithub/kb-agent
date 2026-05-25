# api.py
#
# FastAPI backend for ChatMyDocs.ai
# - Auth: Supabase Auth (JWT verified via auth.get_current_user)
# - Data: Neon Postgres via db.get_db / create_conversation / load_messages / save_message
# - LLM: OpenAI client
# - Orchestration: LangGraph agent

# FastAPI core imports
from fastapi import FastAPI, Header, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, unquote
import requests
from ingest import ingest
from ingest_remote_dir import BASE_URL, ingest_remote_dir
from config import cors_origins
from document_links import document_name, document_url

# Data validation for request bodies
from pydantic import BaseModel

# Env + OpenAI client
from dotenv import load_dotenv
from openai import OpenAI

# LangGraph agent (build once)
from graph import agent, qc

# Supabase auth + Neon DB helpers
from auth import get_current_user  # verifies Supabase JWT, returns {"id": user_id, ...}
from db import get_db, create_conversation, load_messages, save_message

# Load .env for OPENAI_API_KEY and Supabase / Neon settings
load_dotenv()
oai = OpenAI()

# Create the FastAPI app object
app = FastAPI(title="Knowledge Agent API")
DATA_REMOTE_DIR = (Path(__file__).resolve().parent / "data" / "remote").resolve()
WEBDAV_USER = os.getenv("WEBDAV_USER")
WEBDAV_PASS = os.getenv("WEBDAV_PASS")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
INVITES_FILE = (Path(__file__).resolve().parent / "data" / "invites.json").resolve()
ADMIN_LOOKUP_TABLE = os.getenv("ADMIN_LOOKUP_TABLE", "user_admins")
ADMIN_LOOKUP_USER_ID_COLUMN = os.getenv("ADMIN_LOOKUP_USER_ID_COLUMN", "user_id")

# Enable CORS so a browser frontend (even on a different origin) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

def supabase_rest_headers() -> dict[str, str]:
    service_role = require_supabase_service_role()
    return {
        "apikey": service_role,
        "Authorization": f"Bearer {service_role}",
        "Content-Type": "application/json",
    }

def get_admin_record(user_id: str) -> dict | None:
    resp = requests.get(
        f"{SUPABASE_URL.rstrip('/')}/rest/v1/{ADMIN_LOOKUP_TABLE}",
        params={
            "select": ADMIN_LOOKUP_USER_ID_COLUMN,
            ADMIN_LOOKUP_USER_ID_COLUMN: f"eq.{user_id}",
            "limit": "1",
        },
        headers=supabase_rest_headers(),
        timeout=15,
    )
    if resp.status_code >= 400:
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to read admin membership from Supabase table",
                "table": ADMIN_LOOKUP_TABLE,
                "detail": detail,
            },
        )

    rows = resp.json()
    if not rows:
        return None
    return rows[0]

def is_admin_user(user_id: str) -> bool:
    return get_admin_record(user_id) is not None

def require_admin(current_user: dict = Depends(get_current_user)):
    if is_admin_user(current_user["id"]):
        return current_user
    raise HTTPException(status_code=403, detail="Forbidden: admin role required")

def require_current_admin(current_user: dict = Depends(get_current_user)) -> dict:
    if is_admin_user(current_user["id"]):
        return current_user
    raise HTTPException(status_code=403, detail="Forbidden: admin role required")

def load_invites() -> list[dict]:
    if not INVITES_FILE.exists():
        return []
    try:
        return json.loads(INVITES_FILE.read_text())
    except json.JSONDecodeError:
        return []

def save_invites(invites: list[dict]) -> None:
    INVITES_FILE.parent.mkdir(parents=True, exist_ok=True)
    INVITES_FILE.write_text(json.dumps(invites, indent=2))

def require_supabase_service_role() -> str:
    if not SUPABASE_URL:
        raise HTTPException(status_code=500, detail="SUPABASE_URL is missing in backend .env")
    if not SUPABASE_SERVICE_ROLE_KEY or SUPABASE_SERVICE_ROLE_KEY == "PASTE_FROM_DASHBOARD":
        raise HTTPException(
            status_code=500,
            detail="SUPABASE_SERVICE_ROLE_KEY is missing in /Users/zelalemsirag/kb-agent/.env",
        )
    return SUPABASE_SERVICE_ROLE_KEY

def clean_upload_filename(filename: str) -> str:
    name = unquote(Path(filename).name).strip()
    if not name:
        raise HTTPException(status_code=400, detail="Missing filename")
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported")
    return name

def document_candidates(raw_source: str) -> list[Path]:
    source_path = Path(raw_source)
    if source_path.is_absolute():
        return [source_path]
    return [
        (Path(__file__).resolve().parent / source_path).resolve(),
        (DATA_REMOTE_DIR / source_path.name).resolve(),
        (DATA_REMOTE_DIR / unquote(source_path.name).strip()).resolve(),
    ]

def find_document_file(raw_source: str) -> Path | None:
    for candidate in document_candidates(raw_source):
        try:
            candidate.relative_to(DATA_REMOTE_DIR)
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    return None

def format_file_size(size: int | None) -> str:
    if size is None:
        return "Unknown"
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"

@app.get("/documents")
def open_document(source: str, current_user: dict = Depends(get_current_user)):
    raw_source = source
    candidate = find_document_file(raw_source)
    if candidate:
        return FileResponse(
            candidate,
            media_type="application/pdf",
            headers={
                "Content-Disposition": (
                    f'inline; filename="{unquote(candidate.name).strip()}"'
                )
            },
        )

    filename = document_name(raw_source)
    if filename:
        nas_url = f"{BASE_URL.rstrip('/')}/{quote(filename)}"
        auth = (WEBDAV_USER, WEBDAV_PASS) if WEBDAV_USER else None
        try:
            resp = requests.get(nas_url, auth=auth, timeout=60)
            if resp.status_code == 404:
                raise HTTPException(status_code=404, detail="Document not found")
            resp.raise_for_status()
        except HTTPException:
            raise
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail=f"NAS document fetch failed: {exc}") from exc

        return Response(
            content=resp.content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'inline; filename="{filename}"',
                "Cache-Control": "private, max-age=300",
            },
        )

    raise HTTPException(status_code=404, detail="Document not found")

@app.get("/indexed-documents")
def indexed_documents(
    collection: str = "nas_docs",
    current_user: dict = Depends(get_current_user),
):
    docs: dict[str, dict] = {}
    next_offset = None

    while True:
        points, next_offset = qc.scroll(
            collection_name=collection,
            limit=256,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            raw_source = str(payload.get("source") or "")
            if not raw_source:
                continue

            clean_document_name = document_name(raw_source)
            entry = docs.setdefault(raw_source, {
                "id": raw_source,
                "name": clean_document_name,
                "collection": collection,
                "pages": set(),
                "chunks": 0,
                "original_source": raw_source,
                "source": document_url(raw_source),
                "size": None,
                "last_indexed": None,
            })
            entry["chunks"] += 1
            if payload.get("page") is not None:
                entry["pages"].add(payload["page"])

        if next_offset is None:
            break

    items = []
    for entry in docs.values():
        local_file = find_document_file(entry["original_source"])
        size = local_file.stat().st_size if local_file else None
        last_indexed = (
            local_file.stat().st_mtime if local_file else None
        )
        items.append({
            **entry,
            "pages": len(entry["pages"]),
            "size": format_file_size(size),
            "last_indexed": last_indexed,
        })

    items.sort(key=lambda d: d["name"].lower())
    return {"collection": collection, "documents": items}

@app.get("/me")
def current_user_profile(current_user: dict = Depends(get_current_user)):
    is_admin = is_admin_user(current_user["id"])
    return {
        "id": current_user["id"],
        "email": current_user.get("email"),
        "role": "admin" if is_admin else "user",
        "is_admin": is_admin,
    }

@app.post("/upload-document")
async def upload_document(
    request: Request,
    x_filename: str = Header(..., alias="X-Filename"),
    x_collection: str = Header("nas_docs", alias="X-Collection"),
    current_user: dict = Depends(get_current_user),
):
    filename = clean_upload_filename(x_filename)
    content = await request.body()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    DATA_REMOTE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = (DATA_REMOTE_DIR / filename).resolve()
    try:
        local_path.relative_to(DATA_REMOTE_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid filename") from exc

    nas_url = f"{BASE_URL.rstrip('/')}/{quote(filename)}"
    auth = (WEBDAV_USER, WEBDAV_PASS) if WEBDAV_USER else None
    try:
        resp = requests.put(
            nas_url,
            data=content,
            auth=auth,
            headers={"Content-Type": "application/pdf"},
            timeout=60,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"NAS upload failed: {exc}") from exc

    local_path.write_bytes(content)
    try:
        ingest(str(local_path), x_collection)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Uploaded to NAS but ingest failed: {exc}") from exc

    return {
        "status": "ok",
        "document_name": filename,
        "collection": x_collection,
        "source": document_url(filename),
        "nas_url": nas_url,
    }

@app.post("/ingest-remote")
async def ingest_remote(collection: str = "nas_docs", user=Depends(require_admin)):
    try:
        ingest_remote_dir(collection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok", "collection": collection}

@app.options("/{path:path}")
async def options_handler(request: Request) -> Response:
    headers = {
        "Access-Control-Allow-Origin": request.headers.get("Origin", "*"),
        "Access-Control-Allow-Methods": request.headers.get("Access-Control-Request-Method", "POST, GET, OPTIONS"),
        "Access-Control-Allow-Headers": request.headers.get("Access-Control-Request-Headers", "*"),
    }
    if request.headers.get("Access-Control-Request-Private-Network") == "true":
        headers["Access-Control-Allow-Private-Network"] = "true"
    return Response(status_code=204, headers=headers)

# ----- Request models -----


class Query(BaseModel):
    """
    Request body for /query and /query-stream.

    - question: current user message
    - conversation_id: if null, backend creates a new conversation;
                       if set, backend loads history for that conversation
    """
    question: str
    conversation_id: str | None = None


class InviteRequest(BaseModel):
    email: str
    redirectTo: str | None = None


@app.get("/admin/invites")
def list_invites(current_user: dict = Depends(require_current_admin)):
    return {"invites": load_invites()}


@app.post("/admin/invite")
def invite_user(body: InviteRequest, current_user: dict = Depends(require_current_admin)):
    email = body.email.strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="email is required")

    redirect_to = body.redirectTo or FRONTEND_URL
    resp = requests.post(
        f"{SUPABASE_URL.rstrip('/')}/auth/v1/invite",
        params={"redirect_to": redirect_to},
        headers=supabase_rest_headers(),
        json={
            "email": email,
            "data": {"invited_by": current_user.get("id")},
        },
        timeout=30,
    )
    if resp.status_code >= 400:
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text
        raise HTTPException(status_code=resp.status_code, detail=detail)

    invites = [invite for invite in load_invites() if invite.get("email") != email]
    invites.append({
        "email": email,
        "invitedAt": datetime.now(UTC).isoformat(),
        "invitedBy": current_user.get("email") or "unknown",
    })
    save_invites(invites)
    return {"success": True}


@app.delete("/admin/invites/{email}")
def remove_invite(email: str, current_user: dict = Depends(require_current_admin)):
    target = email.strip().lower()
    save_invites([invite for invite in load_invites() if invite.get("email") != target])
    return {"success": True}


# ----- Debug endpoint to inspect Authorization header -----


@app.get("/debug-token")
async def debug_token(request: Request):
    auth = request.headers.get("authorization")
    return {"authorization": auth}


# ----- Main /query endpoint (non-streaming) -----


@app.post("/query")
def query(
    body: Query,
    current_user: dict = Depends(get_current_user),  # Supabase user from JWT
    db=Depends(get_db),                              # SQLAlchemy session for Neon Postgres
):
    # 1) Content moderation via OpenAI
    mod = oai.moderations.create(input=body.question).results[0]
    if mod.flagged:
        raise HTTPException(
            status_code=400,
            detail={
                "blocked": True,
                "categories": mod.categories.model_dump(),
            },
        )

    # Supabase auth.users.id (UUID) from JWT "sub" claim
    user_id = current_user["id"]

    # 2) Resolve or create conversation_id
    if body.conversation_id is None:
        # New conversation in Neon
        conversation_id = create_conversation(db, user_id)
    else:
        conversation_id = body.conversation_id
        # Optional: verify this conversation belongs to user_id with a SELECT

    # 3) Load history for this conversation from Neon
    history = load_messages(db, conversation_id)

    # 4) Run LangGraph agent with question + history
    state = agent.invoke({"question": body.question, "history": history})
    answer = state.get("answer", "")

    # 5) Persist the new turn in the DB
    save_message(db, conversation_id, "user", body.question)
    save_message(db, conversation_id, "assistant", answer)

    # 6) Return answer + sources + conversation_id
    return {
        "conversation_id": conversation_id,
        "answer": answer,
        "sources": state.get("sources", []),
    }


# ----- /query-stream endpoint (SSE streaming) -----


@app.post("/query-stream")
def query_stream(
    body: Query,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    # 1) Moderation
    mod = oai.moderations.create(input=body.question).results[0]
    if mod.flagged:
        raise HTTPException(400, detail={"blocked": True})

    user_id = current_user["id"]

    # 2) Resolve or create conversation_id
    if body.conversation_id is None:
        conversation_id = create_conversation(db, user_id)
    else:
        conversation_id = body.conversation_id

    # 3) Load history for this conversation
    history = load_messages(db, conversation_id)

    # 4) Save the user message before starting the response stream
    save_message(db, conversation_id, "user", body.question)

    def gen():
        yield ": stream-open\n\n"

        # Run graph once to get context, answer, and sources. Keeping this
        # inside the generator lets the browser see an open stream immediately.
        state = agent.invoke(
            {"question": body.question, "history": history},
            {"recursion_limit": 25},
        )
        answer = state["answer"]
        save_message(db, conversation_id, "assistant", answer)

        # First event: send conversation_id + sources to the client
        meta = {
            "conversation_id": conversation_id,
            "sources": state["sources"],
        }
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

        # Build messages for OpenAI streaming
        messages = [
            {"role": "system", "content": "Answer only from context with [n] citations."}
        ]

        # Add prior turns from history
        for msg in history:
            messages.append(msg)

        # Current turn: include context + question
        user_content = (
            f"Context:\n{state['context']}\n\n"
            f"Question: {body.question}\n\n"
            "Answer with inline citations:"
        )
        messages.append({"role": "user", "content": user_content})

        # 6) Stream from OpenAI
        stream = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield f"data: {json.dumps({'token': delta})}\n\n"

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# Simple healthcheck to see if the server is up
@app.get("/health")
def health():
    return {"ok": True}


# Serve static files – e.g., static/index.html as a simple UI
app.mount("/", StaticFiles(directory="static", html=True), name="static")
