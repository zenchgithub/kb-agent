# db.py
#
# Database helpers for ChatMyDocs.ai.
# This file:
#   - Creates a SQLAlchemy engine using Neon POSTGRES_URL (DATABASE_URL)
#   - Provides a FastAPI dependency get_db()
#   - Implements simple helpers for conversations + messages

import os
import uuid

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from env_loader import load_app_env

load_app_env()

# Neon connection string from .env, for example:
# DATABASE_URL=postgresql://neondb_owner:<PASSWORD>@ep-....neon.tech/neondb?sslmode=require
DATABASE_URL = os.environ["DATABASE_URL"]

# Create one engine for the whole app.
# pool_pre_ping=True avoids stale connection errors.
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Session factory: each request gets its own SessionLocal() via get_db()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """
    FastAPI dependency that yields a SQLAlchemy Session.

    Usage:
        def endpoint(db = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ----- Conversation + message helpers -----


def create_conversation(db, user_id: str) -> str:
    """
    Create a new conversation row for this user and return its id.

    Expects a conversations table like:
        create table conversations (
            id uuid primary key,
            user_id uuid not null,
            created_at timestamptz default now()
        );
    """
    conv_id = str(uuid.uuid4())
    db.execute(
        text("insert into conversations (id, user_id) values (:id, :user_id)"),
        {"id": conv_id, "user_id": user_id},
    )
    db.commit()
    return conv_id


def user_owns_conversation(db, conversation_id: str, user_id: str) -> bool:
    result = db.execute(
        text(
            "select 1 from conversations "
            "where id = :cid and user_id = :user_id "
            "limit 1"
        ),
        {"cid": conversation_id, "user_id": user_id},
    )
    return result.fetchone() is not None


def list_conversations(db, user_id: str):
    """
    Return conversation summaries for a user, newest activity first.
    The title is derived from the first user message when possible.
    """
    result = db.execute(
        text(
            """
            select
              c.id,
              c.created_at,
              coalesce(max(m.created_at), c.created_at) as last_message_at,
              (
                select m2.content
                from messages m2
                where m2.conversation_id = c.id and m2.role = 'user'
                order by m2.created_at asc
                limit 1
              ) as first_user_message
            from conversations c
            left join messages m on m.conversation_id = c.id
            where c.user_id = :user_id
            group by c.id, c.created_at
            order by coalesce(max(m.created_at), c.created_at) desc
            """
        ),
        {"user_id": user_id},
    )
    return [
        {
            "id": str(row[0]),
            "created_at": row[1].isoformat() if row[1] else None,
            "updated_at": row[2].isoformat() if row[2] else None,
            "title": (row[3] or "New conversation")[:60],
        }
        for row in result.fetchall()
    ]


def load_messages(db, conversation_id: str):
    """
    Load all messages for a conversation ordered by time.

    Expects a messages table like:
        create table messages (
            id uuid primary key default gen_random_uuid(),
            conversation_id uuid not null,
            role text not null,          -- 'user' or 'assistant'
            content text not null,
            created_at timestamptz default now()
        );
    Returns a list of dicts in the format your agent expects:
        [{"role": "...", "content": "..."}, ...]
    """
    result = db.execute(
        text(
            "select role, content "
            "from messages "
            "where conversation_id = :cid "
            "order by created_at asc"
        ),
        {"cid": conversation_id},
    )
    rows = result.fetchall()
    return [
        {"role": row[0], "content": row[1]}
        for row in rows
    ]


def load_messages_for_user(db, conversation_id: str, user_id: str):
    if not user_owns_conversation(db, conversation_id, user_id):
        return None

    result = db.execute(
        text(
            "select role, content, created_at "
            "from messages "
            "where conversation_id = :cid "
            "order by created_at asc"
        ),
        {"cid": conversation_id},
    )
    return [
        {
            "role": row[0],
            "content": row[1],
            "created_at": row[2].isoformat() if row[2] else None,
        }
        for row in result.fetchall()
    ]


def load_messages_owned(db, conversation_id: str, user_id: str):
    if not user_owns_conversation(db, conversation_id, user_id):
        return None
    return load_messages(db, conversation_id)


def delete_conversation(db, conversation_id: str, user_id: str) -> bool:
    """
    Delete a conversation and its messages only if it belongs to the user.
    Returns False when the conversation does not exist for that user.
    """
    if not user_owns_conversation(db, conversation_id, user_id):
        return False

    db.execute(
        text("delete from messages where conversation_id = :cid"),
        {"cid": conversation_id},
    )
    db.execute(
        text("delete from conversations where id = :cid and user_id = :user_id"),
        {"cid": conversation_id, "user_id": user_id},
    )
    db.commit()
    return True


def save_message(db, conversation_id: str, role: str, content: str):
    """
    Insert a new message row for this conversation.

    role: 'user' or 'assistant'
    content: text of the message
    """
    db.execute(
        text(
            "insert into messages (conversation_id, role, content) "
            "values (:cid, :role, :content)"
        ),
        {"cid": conversation_id, "role": role, "content": content},
    )
    db.commit()
