from pathlib import Path
from typing import Any, TypedDict, List, Dict
from urllib.parse import quote, unquote

from qdrant_client import models
from config import get_qdrant_client
from document_links import document_name, document_url
from env_loader import load_app_env
from model_config import embedding_dimensions, model_for_step

class State(TypedDict, total=False):
    question: str            # original user question
    user_id: str             # Supabase auth.users.id from the verified JWT
    history: List[Dict]      # [{"role": "user"|"assistant", "content": "..."}] from the conversation so far
    subqueries: List[str]    # from planning stage
    collections: List[str]   # from planning stage
    filters: Dict[str, List[str]]
    candidates: List[Dict]   # raw retrieved chunks
    ranked: List[Dict]       # reranked chunks
    context: str             # concatenated context for the LLM
    answer: str              # final answer text
    sources: List[Dict]      # citations metadata
    
import json, yaml
from openai import OpenAI
load_app_env()
oai = OpenAI()

with open("collections.yaml") as f:
 COLLECTIONS = yaml.safe_load(f)["collections"]

DEFAULT_COLLECTION = "nas_docs"

def plan(state: State) -> State:
    coll_desc = "\n".join(f"- {c['name']}: {c['description']}" for c in COLLECTIONS)

    prompt = f"""You are planning retrieval for a PDF RAG system.

Question:
{state['question']}

Conversation history, if useful:
{json.dumps(state.get('history', [])[-6:], ensure_ascii=False)}

Available collections:
{coll_desc}

Return JSON only:
{{
  "subqueries": ["..."],
  "collections": ["..."],
  "filters": {{
    "doc_type": [],
    "field_labels": [],
    "person_names": [],
    "document_names": []
  }}
}}

Rules:
- Create 1 to 5 short retrieval-oriented subqueries.
- Preserve exact person names, document names, IDs, receipt numbers, and quoted phrases.
- If the question asks for nationality, citizenship, passport number, DOB, attorney, address, employer, status, date of issue, or date of expiry:
  - Include one query combining the exact person name and field label.
  - Include close field synonyms.
- For immigration/legal identity questions, prefer doc_type filters like passport, id_card, visa, uscis_form, affidavit.
- Do not invent names or identifiers not present in the question/history."""

    out = oai.chat.completions.create(
        model=model_for_step("planner_model"),
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    ).choices[0].message.content

    print("[raw json from model]", out)
    plan_data = json.loads(out)
    print(f"[plan] subqueries={plan_data.get('subqueries')} collections={plan_data.get('collections')} filters={plan_data.get('filters')}")

    new_state: State = dict(state)
    subqueries, filters = enrich_plan_with_field_hints(state, plan_data)
    new_state["subqueries"] = subqueries
    new_state["collections"] = list(plan_data.get("collections") or [])
    new_state["filters"] = filters
    return new_state

def require_user_id(state: State) -> str:
    user_id = str(state.get("user_id") or "").strip()
    if not user_id:
        raise PermissionError("access node: user_id missing from state")
    return user_id

def user_payload_filter(user_id: str, raw_source: str | None = None) -> models.Filter:
    must = [
        models.Filter(
            should=[
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id),
                ),
                models.FieldCondition(
                    key="isPublic",
                    match=models.MatchValue(value=True),
                ),
                models.IsEmptyCondition(
                    is_empty=models.PayloadField(key="user_id"),
                ),
            ]
        )
    ]
    if raw_source:
        must.append(
            models.FieldCondition(
                key="source",
                match=models.MatchValue(value=raw_source),
            )
        )
    return models.Filter(must=must)

def assert_owned_candidates(candidates: list[dict], user_id: str):
    for candidate in candidates:
        if candidate.get("isPublic") is True:
            continue
        candidate_user_id = candidate.get("user_id")
        if candidate_user_id is not None and str(candidate_user_id) != user_id:
            raise PermissionError("cross-tenant hit detected")

def payload_is_public(payload: dict) -> bool:
    return payload.get("isPublic") is True or not payload.get("user_id")

def metadata_should_filter(key: str, values: list[str]) -> models.Filter | None:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    if not cleaned:
        return None
    return models.Filter(
        should=[
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for value in cleaned
        ]
    )

def retrieval_filter(user_id: str, raw_source: str | None = None, filters: dict[str, list[str]] | None = None) -> models.Filter:
    base = user_payload_filter(user_id, raw_source)
    filters = filters or {}
    must = list(base.must or [])

    doc_type_filter = metadata_should_filter("doc_type", filters.get("doc_type") or [])
    if doc_type_filter:
        must.append(doc_type_filter)

    return models.Filter(must=must)

def payload_to_candidate(payload: dict, score: float = 1.0) -> dict:
    raw_source = str(payload.get("source") or payload.get("file_name") or "")
    return {
        "text": str(payload.get("text") or ""),
        "score": score,
        "source": raw_source,
        "page": payload.get("page"),
        "hash": payload.get("hash", f"{raw_source}:{payload.get('page')}:{payload.get('chunk_index')}"),
        "user_id": payload.get("user_id"),
        "isPublic": payload_is_public(payload),
        "doc_type": payload.get("doc_type"),
        "chunk_type": payload.get("chunk_type"),
        "field_label": payload.get("field_label"),
        "person_name": payload.get("person_name"),
        "section": payload.get("section"),
        "heading": payload.get("heading"),
        "keywords": payload.get("keywords") or [],
        "entities": payload.get("entities") or [],
    }

def access(state: State) -> State:
    user_id = require_user_id(state)
    assert_owned_candidates(state.get("candidates", []), user_id)
    assert_owned_candidates(state.get("ranked", []), user_id)

    allowed = {c["name"] for c in COLLECTIONS}
    filtered = [c for c in state.get("collections", []) if c in allowed]
    if not filtered:
        # Safe production fallback: keep today's known-good collection instead
        # of searching every configured collection. Some configured collections
        # may be documentation-only or not yet present in Qdrant.
        filtered = [DEFAULT_COLLECTION]
    new_state: State = dict(state)
    new_state["collections"] = filtered
    print(f"[access] user_id={user_id} collections={filtered}")
    return new_state

import asyncio
import re
qc = get_qdrant_client()
TOPK = 8
DOC_CONTEXT_CHUNKS = 3
KEYWORD_TOPK = 8
STOPWORDS = {
    "about", "after", "also", "and", "any", "are", "can", "could", "did",
    "does", "for", "from", "have", "how", "into", "is", "me", "of", "on",
    "or", "show", "tell", "that", "the", "this", "to", "was", "what",
    "when", "where", "which", "who", "with", "would", "you",
}

FIELD_SYNONYMS = {
    "attorney": [
        "attorney",
        "lawyer",
        "representative",
        "preparer",
        "accredited representative",
        "g 28",
        "g-28",
        "counsel",
    ],
    "nationality": ["nationality", "citizenship", "country of citizenship", "citizen", "ethopian", "ethiopian"],
    "passport_number": ["passport number", "passport no", "passport no.", "document number"],
    "date_of_birth": ["date of birth", "birth date", "dob"],
    "date_of_issue": ["date of issue", "issued on", "issue date"],
    "date_of_expiry": ["date of expiry", "expiration date", "expiry date", "expires"],
    "address": ["address", "street", "city", "state", "zip"],
    "employer": ["employer", "company", "occupation"],
    "alien_number": ["alien number", "a-number", "a number", "uscis number"],
}

FIELD_CANONICAL = {
    synonym: field
    for field, synonyms in FIELD_SYNONYMS.items()
    for synonym in synonyms
}

def normalize_search_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

def canonical_field_label(value: str) -> str:
    normalized = normalize_search_text(value).replace(" ", "_")
    for synonym, field in FIELD_CANONICAL.items():
        if normalize_search_text(synonym).replace(" ", "_") == normalized:
            return field
    return normalized

def detect_question_fields(question: str, filters: dict[str, list[str]] | None = None) -> list[str]:
    haystack = normalize_search_text(question)
    detected: list[str] = []
    for raw in (filters or {}).get("field_labels", []) or []:
        field = canonical_field_label(raw)
        if field and field not in detected:
            detected.append(field)
    for field, synonyms in FIELD_SYNONYMS.items():
        if any(normalize_search_text(synonym) in haystack for synonym in synonyms):
            if field not in detected:
                detected.append(field)
    return detected

def field_terms(fields: list[str]) -> list[str]:
    terms: list[str] = []
    for field in fields:
        for synonym in FIELD_SYNONYMS.get(canonical_field_label(field), [field]):
            normalized = normalize_search_text(synonym)
            if normalized and normalized not in terms:
                terms.append(normalized)
    return terms

def text_has_any_term(text: str, terms: list[str]) -> bool:
    haystack = normalize_search_text(text)
    return any(term and term in haystack for term in terms)

def enrich_plan_with_field_hints(state: State, plan_data: dict) -> tuple[list[str], dict[str, list[str]]]:
    filters = plan_data.get("filters") or {}
    new_filters = {
        "doc_type": list(filters.get("doc_type") or []),
        "field_labels": [canonical_field_label(value) for value in list(filters.get("field_labels") or [])],
        "person_names": list(filters.get("person_names") or []),
        "document_names": list(filters.get("document_names") or []),
    }
    detected_fields = detect_question_fields(state["question"], new_filters)
    for field in detected_fields:
        if field not in new_filters["field_labels"]:
            new_filters["field_labels"].append(field)

    subqueries = list(plan_data.get("subqueries") or [state["question"]])[:5]
    for field in detected_fields:
        people = new_filters["person_names"] or []
        if people:
            for person in people[:2]:
                q = f"{person} {' '.join(FIELD_SYNONYMS.get(field, [field])[:3])}"
                if q not in subqueries:
                    subqueries.append(q)
        else:
            q = " ".join(FIELD_SYNONYMS.get(field, [field])[:4])
            if q not in subqueries:
                subqueries.append(q)
    return subqueries[:5], new_filters

def source_label(raw_source: str) -> str:
    return unquote(Path(str(raw_source)).name).strip()

def qdrant_collection_exists(collection: str) -> bool:
    try:
        qc.get_collection(collection)
        return True
    except Exception as exc:
        print(f"[retrieve] skipping unavailable collection={collection}: {exc}")
        return False

def existing_qdrant_collections(collections: list[str]) -> list[str]:
    existing = [collection for collection in collections if qdrant_collection_exists(collection)]
    if existing:
        return existing
    if DEFAULT_COLLECTION not in collections and qdrant_collection_exists(DEFAULT_COLLECTION):
        print(f"[retrieve] falling back to {DEFAULT_COLLECTION}")
        return [DEFAULT_COLLECTION]
    return []

def collection_sources(collection: str, user_id: str) -> list[str]:
    if not qdrant_collection_exists(collection):
        return []
    sources = set()
    next_offset = None
    while True:
        points, next_offset = qc.scroll(
            collection_name=collection,
            scroll_filter=user_payload_filter(user_id),
            limit=256,
            offset=next_offset,
            with_payload=["source"],
            with_vectors=False,
        )
        for point in points:
            raw_source = (point.payload or {}).get("source")
            if raw_source:
                sources.add(str(raw_source))
        if next_offset is None:
            break
    return sorted(sources, key=source_label)

def mentioned_sources(question: str, collections: list[str], user_id: str) -> dict[str, list[str]]:
    haystack = normalize_search_text(question)
    matches: dict[str, list[str]] = {}

    for collection in collections:
        for raw_source in collection_sources(collection, user_id):
            name = source_label(raw_source)
            stem = Path(name).stem
            candidates = {
                normalize_search_text(name),
                normalize_search_text(stem),
                normalize_search_text(raw_source),
            }
            if any(candidate and candidate in haystack for candidate in candidates):
                matches.setdefault(collection, []).append(raw_source)

    return matches

def recent_history_text(history: list[dict], limit: int = 4) -> str:
    recent = history[-limit:] if history else []
    return " ".join(str(item.get("content") or "") for item in recent)

def significant_terms(question: str) -> list[str]:
    terms = []
    for term in normalize_search_text(question).split():
        if len(term) < 3 or term in STOPWORDS:
            continue
        if term not in terms:
            terms.append(term)
    return terms[:8]

def keyword_match_chunks(
    collections: list[str],
    question: str,
    source_matches: dict[str, list[str]],
    user_id: str,
    filters: dict[str, list[str]] | None = None,
) -> list[dict]:
    terms = significant_terms(question)
    fields = detect_question_fields(question, filters)
    field_term_list = field_terms(fields)
    if not terms:
        terms = field_term_list
    query_persons = [normalize_search_text(value) for value in (filters or {}).get("person_names", [])]

    matches = []
    for collection in collections:
        if not qdrant_collection_exists(collection):
            continue
        source_filter = source_matches.get(collection) or []
        next_offset = None
        while True:
            points, next_offset = qc.scroll(
                collection_name=collection,
                scroll_filter=retrieval_filter(user_id, filters=filters),
                limit=256,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                raw_source = str(payload.get("source") or "")
                if source_filter and raw_source not in source_filter:
                    continue

                text = str(payload.get("text") or "")
                metadata_text = " ".join(
                    str(payload.get(key) or "")
                    for key in ("field_label", "person_name", "section", "heading", "doc_type", "chunk_type")
                )
                haystack = normalize_search_text(f"{source_label(raw_source)} {metadata_text} {text}")
                field_hit = bool(field_term_list) and any(term in haystack for term in field_term_list)
                all_terms_hit = bool(terms) and all(term in haystack for term in terms)
                person_hit = not query_persons or any(person and person in haystack for person in query_persons)
                if all_terms_hit or (field_hit and (person_hit or source_filter)):
                    matches.append(payload_to_candidate(payload, score=1.8 if field_hit else 1.2))
                    if len(matches) >= KEYWORD_TOPK:
                        return matches

            if next_offset is None:
                break

    return matches

def field_source_chunks(
    collections: list[str],
    source_matches: dict[str, list[str]],
    user_id: str,
    filters: dict[str, list[str]] | None = None,
) -> list[dict]:
    fields = detect_question_fields(" ".join((filters or {}).get("field_labels", [])), filters)
    terms = field_terms(fields)
    if not terms or not source_matches:
        return []

    matches: list[dict] = []
    for collection, raw_sources in source_matches.items():
        if not qdrant_collection_exists(collection):
            continue
        for raw_source in raw_sources:
            points, _ = qc.scroll(
                collection_name=collection,
                scroll_filter=user_payload_filter(user_id, raw_source),
                limit=512,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                text = " ".join(
                    str(payload.get(key) or "")
                    for key in ("source", "text", "field_label", "section", "heading", "keywords")
                )
                if text_has_any_term(text, terms):
                    matches.append(payload_to_candidate(payload, score=2.0))
                    if len(matches) >= KEYWORD_TOPK:
                        return matches
    return matches

async def one_search(
    query: str,
    collection: str,
    user_id: str,
    raw_source: str | None = None,
    filters: dict[str, list[str]] | None = None,
):
    if not qdrant_collection_exists(collection):
        return []

    # embed the query
    vec = oai.embeddings.create(
        model=model_for_step("embedding_model"),
        input=query,
        dimensions=embedding_dimensions(),
    ).data[0].embedding

    query_filter = retrieval_filter(user_id, raw_source, filters)

    # vector search in Qdrant
    hits = qc.query_points(
        collection_name=collection,
        query=vec,
        query_filter=query_filter,
        limit=TOPK,
    ).points

    return hits

async def retrieve_all(queries, collections, source_matches, user_id, filters):
    tasks = []
    for collection in collections:
        matched = source_matches.get(collection) or []
        if matched:
            tasks.extend(one_search(q, collection, user_id, s, filters) for q in queries for s in matched)
        else:
            tasks.extend(one_search(q, collection, user_id, filters=filters) for q in queries)
    return await asyncio.gather(*tasks)

def source_context_chunks(collection: str, raw_source: str, user_id: str) -> list:
    if not qdrant_collection_exists(collection):
        return []
    points, _ = qc.scroll(
        collection_name=collection,
        scroll_filter=user_payload_filter(user_id, raw_source),
        limit=256,
        with_payload=True,
        with_vectors=False,
    )
    return sorted(
        points,
        key=lambda p: (
            (p.payload or {}).get("page") or 0,
            (p.payload or {}).get("chunk_index") or 0,
        ),
    )[:DOC_CONTEXT_CHUNKS]

def structured_field_candidates(
    collections: list[str],
    user_id: str,
    filters: dict[str, list[str]],
) -> list[dict]:
    """Structured-first lookup for facts like nationality, DOB, attorney.

    These chunks are first-class evidence and should be checked before vector
    search. This is the path that should answer questions like "What is
    Zelalem's nationality?" when a passport/ID field was extracted.
    """

    field_labels = [value.lower().replace(" ", "_") for value in filters.get("field_labels", [])]
    person_names = [normalize_search_text(value) for value in filters.get("person_names", [])]
    if not field_labels and not person_names:
        return []

    matches: list[dict] = []
    for collection in collections:
        if not qdrant_collection_exists(collection):
            continue
        next_offset = None
        while True:
            points, next_offset = qc.scroll(
                collection_name=collection,
                scroll_filter=retrieval_filter(user_id, filters=filters),
                limit=256,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                chunk_type = str(payload.get("chunk_type") or "")
                if chunk_type not in {"identity_field", "key_value"}:
                    continue
                field_label = str(payload.get("field_label") or "").lower()
                text = str(payload.get("text") or "")
                text_norm = normalize_search_text(text)
                field_ok = not field_labels or any(
                    label == field_label or label.replace("_", " ") in text_norm
                    for label in field_labels
                )
                person = normalize_search_text(str(payload.get("person_name") or ""))
                person_ok = not person_names or any(name and (name in person or name in text_norm) for name in person_names)
                if field_ok and person_ok:
                    matches.append(payload_to_candidate(payload, score=2.0))
            if next_offset is None:
                break
    return matches[:KEYWORD_TOPK]

def retrieve(state: State) -> State:
    user_id = require_user_id(state)
    filters = state.get("filters", {})
    queries = [state["question"]]
    for query in state.get("subqueries", []):
        if query not in queries:
            queries.append(query)
    collections = existing_qdrant_collections(state.get("collections") or [DEFAULT_COLLECTION])
    if not collections:
        new_state: State = dict(state)
        new_state["candidates"] = []
        print("[retrieve] no available Qdrant collections")
        return new_state
    current_query_text = " ".join(queries)
    source_matches = mentioned_sources(current_query_text, collections, user_id)
    if not source_matches:
        source_matches = mentioned_sources(
            f"{current_query_text} {recent_history_text(state.get('history', []))}",
            collections,
            user_id,
        )
    results = asyncio.run(retrieve_all(queries, collections, source_matches, user_id, filters))

    flat = structured_field_candidates(collections, user_id, filters)
    flat.extend(field_source_chunks(collections, source_matches, user_id, filters))
    flat.extend(keyword_match_chunks(collections, state["question"], source_matches, user_id, filters))
    for hits in results:
        for p in hits:
            flat.append(payload_to_candidate(p.payload or {}, score=p.score))

    for collection, sources in source_matches.items():
        for raw_source in sources:
            for p in source_context_chunks(collection, raw_source, user_id):
                payload = p.payload or {}
                flat.append(payload_to_candidate(payload, score=1.0))

    deduped = []
    seen = set()
    for candidate in flat:
        key = (candidate.get("source"), candidate.get("page"), candidate.get("hash"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)

    if not deduped and filters.get("doc_type"):
        # Older Qdrant points may not have doc_type metadata. Prefer metadata
        # filters when present, but fall back to access-only retrieval so legacy
        # public documents are not hidden from search.
        relaxed_filters = {**filters, "doc_type": []}
        print("[retrieve] no metadata-filtered hits; retrying without doc_type filter")
        relaxed_results = asyncio.run(retrieve_all(queries, collections, source_matches, user_id, relaxed_filters))
        relaxed_flat = keyword_match_chunks(collections, state["question"], source_matches, user_id, relaxed_filters)
        for hits in relaxed_results:
            for p in hits:
                relaxed_flat.append(payload_to_candidate(p.payload or {}, score=p.score))
        for candidate in relaxed_flat:
            key = (candidate.get("source"), candidate.get("page"), candidate.get("hash"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)

    print(f"[retrieve] got {len(deduped)} candidate chunks")
    if source_matches:
        print(f"[retrieve] document filters={source_matches}")
    keyword_count = sum(1 for candidate in deduped if candidate.get("score") == 1.2)
    if keyword_count:
        print(f"[retrieve] exact keyword matches={keyword_count}")
    assert_owned_candidates(deduped, user_id)
    new_state: State = dict(state)
    new_state["candidates"] = deduped
    return new_state

def rerank(state: State) -> State:
    question = state["question"]
    candidates = state.get("candidates", [])
    filters = state.get("filters", {})
    question_fields = detect_question_fields(question, filters)
    question_field_terms = field_terms(question_fields)
    source_matches_present = any(
        text_has_any_term(source_label(c.get("source", "")), [normalize_search_text(value)])
        for c in candidates
        for value in filters.get("document_names", [])
    )
    if question_field_terms:
        field_candidates = [
            c for c in candidates
            if text_has_any_term(
                " ".join(
                    str(c.get(key) or "")
                    for key in ("text", "field_label", "section", "heading", "keywords", "chunk_type")
                ),
                question_field_terms,
            )
        ]
        if field_candidates:
            candidates = field_candidates
        elif source_matches_present:
            candidates = []

    exact_candidates = [c for c in candidates if c.get("score", 0) >= 1.2]
    if exact_candidates:
        candidates = exact_candidates

    candidate_items = []
    for i, c in enumerate(candidates):
        candidate_items.append({
            "index": i,
            "text": c.get("text"),
            "source": source_label(c.get("source", "")),
            "page": c.get("page"),
            "metadata": {
                "doc_type": c.get("doc_type"),
                "chunk_type": c.get("chunk_type"),
                "field_label": c.get("field_label"),
                "person_name": c.get("person_name"),
                "section": c.get("section"),
                "heading": c.get("heading"),
                "keywords": c.get("keywords") or [],
                "entities": c.get("entities") or [],
                "score": c.get("score"),
            },
        })

    prompt = f"""You are reranking retrieved PDF chunks for document QA.

Question:
{question}

Candidate chunks:
{json.dumps(candidate_items, ensure_ascii=False)}

Return JSON only:
{{"ranked_indices": [0, 2, 5]}}

Reranking rules:
- Strongly prefer chunks that directly answer the question.
- Strongly prefer exact matches for person name, document name, identifier, and field label.
- Prefer explicit key-value or structured identity chunks such as "Nationality: Ethiopian" over generic narrative.
- Prefer metadata matches: doc_type, field_label, person_name, chunk_type.
- Penalize chunks about a different person or irrelevant entity.
- Keep only chunks that genuinely help answer the question.
- If no chunk is useful, return an empty list.
- For identity/legal questions, rank identity_field/key_value chunks first, then exact field+person chunks, then passport/id/visa chunks, then nearby narrative chunks."""

    out = oai.chat.completions.create(
        model=model_for_step("rerank_model"),
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    ).choices[0].message.content
    

    data = json.loads(out)
    idxs = data.get("ranked_indices", [])

    ranked = [candidates[i] for i in idxs if 0 <= i < len(candidates)]
    if question_field_terms:
        ranked = [
            c for c in ranked
            if text_has_any_term(
                " ".join(
                    str(c.get(key) or "")
                    for key in ("text", "field_label", "section", "heading", "keywords", "chunk_type")
                ),
                question_field_terms,
            )
        ]

    new_state: State = dict(state)
    new_state["ranked"] = ranked
    print(f"[rerank] kept {len(ranked)} chunks")
    return new_state

def normalize(state: State) -> State:
    parts = []
    sources = []
    for i, c in enumerate(state["ranked"], 1):
        raw_source = str(c["source"])
        clean_document_name = document_name(raw_source)
        source_url = document_url(raw_source)
        metadata = (
            f"doc_type={c.get('doc_type')}, "
            f"chunk_type={c.get('chunk_type')}, "
            f"field_label={c.get('field_label')}, "
            f"person_name={c.get('person_name')}"
        )
        parts.append(f"[{i}] {clean_document_name} p.{c.get('page')}\nmetadata: {metadata}\ntext: {c['text']}")
        sources.append({
            "id": i,
            "document_name": clean_document_name,
            "source": source_url,
            "original_source": raw_source,
            "page": c.get("page"),
            "matched_text": c["text"],
        })

    new_state: State = dict(state)
    new_state["context"] = "\n\n".join(parts)
    new_state["sources"] = sources
    return new_state

SYSPROMPT = """You are ChatMyDocs.ai, a document question-answering assistant.

Rules:
- Use ONLY the provided context chunks and structured fields.
- Do not use outside knowledge.
- Do not guess, infer, or assume identity/legal facts.
- Every factual claim must include an inline citation like [1] or [2].
- Use the source numbers exactly as provided in the context.
- For identity/legal facts such as nationality, citizenship, passport number, date of birth, attorney, address, employer, immigration status, or document expiration:
  - Answer only when the context explicitly states the fact.
  - Prefer structured identity fields and key-value lines over narrative text.
  - If a structured field directly answers the question, use it.
- Evidence policy:
  - Fully supported: answer directly and cite.
  - Partial evidence: say what is supported and what is missing.
  - No relevant evidence: say “I don’t have enough information in the retrieved documents.”
- Be concise and precise.
"""

def synthesize(state: State) -> State:
    if not state.get("context"):
        new_state: State = dict(state)
        new_state["answer"] = "I don’t have enough information in the retrieved documents."
        return new_state

    prompt = f"""Question:
{state['question']}

Context chunks:
{state['context']}

Each context chunk is formatted as:
[ID] document_name p.page
metadata: doc_type=..., chunk_type=..., field_label=..., person_name=...
text: ...

Answer the question using only the context above.
Use inline citations like [1], [2] after every factual claim.
If the context does not clearly contain the answer, say:
“I don’t have enough information in the retrieved documents.”"""

    resp = oai.chat.completions.create(
        model=model_for_step("answer_model"),
        messages=[
            {"role": "system", "content": SYSPROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    answer = resp.choices[0].message.content

    new_state: State = dict(state)
    new_state["answer"] = answer
    return new_state

from langgraph.graph import StateGraph, END

def build_agent():
    g = StateGraph(State)

    g.add_node("plan", plan)
    g.add_node("access", access)
    g.add_node("retrieve", retrieve)
    g.add_node("rerank", rerank)
    g.add_node("normalize", normalize)
    g.add_node("synthesize", synthesize)

    g.set_entry_point("plan")
    
    g.add_edge("plan", "access")
    g.add_edge("access", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "normalize")
    g.add_edge("normalize", "synthesize")
    g.add_edge("synthesize", END)

    return g.compile()

agent = build_agent()

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "who is the attorney of zelalem sirag"
    result = agent.invoke({"question": q})

    print("ANSWER:")
    print(result.get("answer", ""))

    print("\nSOURCES:")
    for s in result.get("sources", []):
        print(s["id"], s["source"], "p.", s["page"])        
