# ChatMyDocs.ai Backend

FastAPI backend for ChatMyDocs.ai. It verifies Supabase JWTs, stores conversations in Postgres, retrieves document chunks from Qdrant, calls OpenAI, uploads PDFs to the NAS WebDAV folder, and sends Supabase invitation emails for admins.

## Local Development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## Required Services

- Supabase Auth
- Postgres database for conversations/messages
- Qdrant for document vectors
- OpenAI API
- NAS/WebDAV folder for uploaded PDFs

For deployment, do not point `QDRANT_HOST` at `localhost` unless Qdrant is running in the same deploy network. Use `QDRANT_URL` and `QDRANT_API_KEY` for Qdrant Cloud or another reachable Qdrant service.

## Environment Variables

See [.env.example](.env.example).

Core values:

```bash
OPENAI_API_KEY=
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
DATABASE_URL=
QDRANT_URL=
QDRANT_API_KEY=
FRONTEND_URL=
CORS_ORIGINS=
NAS_BASE_URL=
WEBDAV_USER=
WEBDAV_PASS=
ADMIN_LOOKUP_TABLE=user_admins
ADMIN_LOOKUP_USER_ID_COLUMN=user_id
```

`CORS_ORIGINS` is comma-separated:

```bash
CORS_ORIGINS=http://localhost:5173,https://your-frontend.vercel.app
```

## Deploy With Docker

Build locally:

```bash
docker build -t kb-agent .
```

Run locally:

```bash
docker run --env-file .env -p 8000:8000 kb-agent
```

The container starts:

```bash
uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
```

## Deploy On Render

This repo includes [render.yaml](render.yaml).

1. Push this repo to GitHub.
2. In Render, create a new Blueprint or Web Service from `zenchgithub/kb-agent`.
3. Use Docker runtime.
4. Add all required environment variables from `.env.example`.
5. Set `FRONTEND_URL` to your frontend URL.
6. Set `CORS_ORIGINS` to both local and production frontend URLs.
7. Set `QDRANT_URL` and `QDRANT_API_KEY` to a hosted Qdrant instance.

After deployment, test:

```bash
curl https://your-backend-domain/health
```

Then update frontend:

```bash
VITE_CHATMYDOCS_API_URL=https://your-backend-domain
```

## Admin Roles And Invites

Admin authorization is enforced by the FastAPI backend. The frontend cannot grant itself admin access.

The backend checks whether the signed-in user's UUID exists in:

```text
public.user_admins.user_id
```

Create the admin lookup table:

```sql
create table if not exists public.user_admins (
  user_id uuid primary key references auth.users(id) on delete cascade,
  created_at timestamptz not null default now()
);

alter table public.user_admins enable row level security;
```

Safer RLS policy for admin membership:

```sql
drop policy if exists "user_admins_insert_self" on public.user_admins;
drop policy if exists "user_admins_update_self" on public.user_admins;
drop policy if exists "user_admins_delete_self" on public.user_admins;

create policy "user_admins_select_self"
  on public.user_admins for select
  to authenticated
  using (user_id = auth.uid());
```

Do not allow authenticated users to insert/update/delete their own `user_admins` row. Otherwise any signed-in user can make themselves admin. Add admins only from Supabase SQL Editor or another trusted server-side process.

Grant admin to a user:

```sql
insert into public.user_admins (user_id)
values ('USER_UUID_FROM_AUTH_USERS')
on conflict (user_id) do nothing;
```

Admin-only endpoints:

- `GET /admin/invites`
- `POST /admin/invite`
- `DELETE /admin/invites/{email}`

Invite behavior:

1. Admin calls `POST /admin/invite`.
2. Backend verifies the caller exists in `public.user_admins`.
3. Backend sends the invite through Supabase Auth Admin API with the service role key.
4. The invite redirects to the frontend with `?invite=1`.
5. The frontend shows a set-password screen and calls `supabase.auth.updateUser({ password })`.
