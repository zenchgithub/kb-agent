from pathlib import Path
from typing import TypedDict, List, Dict
from urllib.parse import quote, unquote

from dotenv import load_dotenv
from qdrant_client import models
from config import get_qdrant_client

class State(TypedDict, total=False):
    question: str            # original user question
    history: List[Dict]      # [{"role": "user"|"assistant", "content": "..."}] from the conversation so far
    subqueries: List[str]    # from planning stage
    collections: List[str]   # from planning stage
    candidates: List[Dict]   # raw retrieved chunks
    ranked: List[Dict]       # reranked chunks
    context: str             # concatenated context for the LLM
    answer: str              # final answer text
    sources: List[Dict]      # citations metadata
    
import json, yaml
from openai import OpenAI
load_dotenv() 
oai = OpenAI()

with open("collections.yaml") as f:
 COLLECTIONS = yaml.safe_load(f)["collections"]

def plan(state: State) -> State:
    coll_desc = "\n".join(f"- {c['name']}: {c['description']}" for c in COLLECTIONS)

    prompt = f"""Decompose the user's question into 1-3 focused sub-queries
and pick which collections to search.

Available collections:
{coll_desc}

User question: {state['question']}

Respond in JSON:
{{"subqueries": ["...","..."], "collections": ["name1","name2"]}}"""

    out = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    ).choices[0].message.content

    print("[raw json from model]", out)
    plan_data = json.loads(out)
    print(f"[plan] subqueries={plan_data['subqueries']} collections={plan_data['collections']}")

    new_state: State = dict(state)
    new_state["subqueries"] = plan_data["subqueries"]
    new_state["collections"] = plan_data["collections"]
    return new_state

def access(state: State) -> State:
    # In a real system, this would filter collections by user/tenant/role.
    # For your personal agent, we just keep whatever the planner chose.
    allowed = {c["name"] for c in COLLECTIONS}
    filtered = [c for c in state.get("collections", []) if c in allowed]
    if not filtered:
        # fallback: if nothing valid, search all collections
        filtered = list(allowed)
    new_state: State = dict(state)
    new_state["collections"] = filtered
    print(f"[access] collections={filtered}")
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

def normalize_search_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

def source_label(raw_source: str) -> str:
    return unquote(Path(str(raw_source)).name).strip()

def collection_sources(collection: str) -> list[str]:
    sources = set()
    next_offset = None
    while True:
        points, next_offset = qc.scroll(
            collection_name=collection,
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

def mentioned_sources(question: str, collections: list[str]) -> dict[str, list[str]]:
    haystack = normalize_search_text(question)
    matches: dict[str, list[str]] = {}

    for collection in collections:
        for raw_source in collection_sources(collection):
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
) -> list[dict]:
    terms = significant_terms(question)
    if not terms:
        return []

    matches = []
    for collection in collections:
        source_filter = source_matches.get(collection) or []
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
                if source_filter and raw_source not in source_filter:
                    continue

                text = str(payload.get("text") or "")
                haystack = normalize_search_text(f"{source_label(raw_source)} {text}")
                if all(term in haystack for term in terms):
                    matches.append({
                        "text": text,
                        "score": 1.2,
                        "source": raw_source,
                        "page": payload.get("page"),
                        "hash": payload.get(
                            "hash",
                            f"{raw_source}:{payload.get('page')}:{payload.get('chunk_index')}",
                        ),
                    })
                    if len(matches) >= KEYWORD_TOPK:
                        return matches

            if next_offset is None:
                break

    return matches

async def one_search(query: str, collection: str, raw_source: str | None = None):
    # embed the query
    vec = oai.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    ).data[0].embedding

    query_filter = None
    if raw_source:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="source",
                    match=models.MatchValue(value=raw_source),
                )
            ]
        )

    # vector search in Qdrant
    hits = qc.query_points(
        collection_name=collection,
        query=vec,
        query_filter=query_filter,
        limit=TOPK,
    ).points

    return hits

async def retrieve_all(queries, collections, source_matches):
    tasks = []
    for collection in collections:
        matched = source_matches.get(collection) or []
        if matched:
            tasks.extend(one_search(q, collection, s) for q in queries for s in matched)
        else:
            tasks.extend(one_search(q, collection) for q in queries)
    return await asyncio.gather(*tasks)

def source_context_chunks(collection: str, raw_source: str) -> list:
    points, _ = qc.scroll(
        collection_name=collection,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="source",
                    match=models.MatchValue(value=raw_source),
                )
            ]
        ),
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

def retrieve(state: State) -> State:
    queries = [state["question"]]
    for query in state.get("subqueries", []):
        if query not in queries:
            queries.append(query)
    #collections = state["collections"]
        # Force only nas_docs for debugging
    collections = ["nas_docs"]
    current_query_text = " ".join(queries)
    source_matches = mentioned_sources(current_query_text, collections)
    if not source_matches:
        source_matches = mentioned_sources(
            f"{current_query_text} {recent_history_text(state.get('history', []))}",
            collections,
        )
    results = asyncio.run(retrieve_all(queries, collections, source_matches))

    flat = keyword_match_chunks(collections, state["question"], source_matches)
    for hits in results:
        for p in hits:
            flat.append({
                "text": p.payload["text"],
                "score": p.score,
                "source": p.payload["source"],
                "page": p.payload["page"],
                "hash": p.payload["hash"],
            })

    for collection, sources in source_matches.items():
        for raw_source in sources:
            for p in source_context_chunks(collection, raw_source):
                payload = p.payload or {}
                flat.append({
                    "text": payload.get("text", ""),
                    "score": 1.0,
                    "source": payload.get("source", raw_source),
                    "page": payload.get("page"),
                    "hash": payload.get("hash", f"{raw_source}:{payload.get('page')}:{payload.get('chunk_index')}"),
                })

    deduped = []
    seen = set()
    for candidate in flat:
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
    new_state: State = dict(state)
    new_state["candidates"] = deduped
    return new_state

def rerank(state: State) -> State:
    question = state["question"]
    candidates = state.get("candidates", [])
    exact_candidates = [c for c in candidates if c.get("score") == 1.2]
    if exact_candidates:
        candidates = exact_candidates

    # keep just the text, with indices so we can map back
    # build a compact prompt
    prompt = f"""You are reranking retrieved document chunks for a question.

Question:
{question}

Here are the candidate chunks, numbered. The Source and page are part of relevance:

""" + "\n\n".join(
        f"[{i}] Source: {source_label(c['source'])}, page {c.get('page')}\nText: {c['text']}"
        for i, c in enumerate(candidates)
    ) + """

Return a JSON object with the indices of the most relevant chunks in best-to-worst order.
Only include indices that truly help answer the question.
If the user names a document, prefer chunks from that document.
If the user asks about a specific person, business, address, identifier, or exact phrase,
only include chunks that contain that entity or directly answer the question.

Example format:
{"ranked_indices": [3, 0, 5, 2]}
"""

    out = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    ).choices[0].message.content
    

    data = json.loads(out)
    idxs = data.get("ranked_indices", [])

    ranked = [candidates[i] for i in idxs if 0 <= i < len(candidates)]

    new_state: State = dict(state)
    new_state["ranked"] = ranked
    print(f"[rerank] kept {len(ranked)} chunks")
    return new_state

def normalize(state: State) -> State:
    parts = []
    sources = []
    for i, c in enumerate(state["ranked"], 1):
        raw_source = str(c["source"])
        document_name = unquote(Path(raw_source).name).strip()
        source_url = f"/documents?source={quote(raw_source, safe='')}"
        parts.append(f"[{i}] ({document_name} p.{c['page']}) {c['text']}")
        sources.append({
            "id": i,
            "document_name": document_name,
            "source": source_url,
            "original_source": raw_source,
            "page": c["page"],
            "matched_text": c["text"],
        })

    new_state: State = dict(state)
    new_state["context"] = "\n\n".join(parts)
    new_state["sources"] = sources
    return new_state

SYSPROMPT = """You are a helpful assistant that answers questions strictly from the provided context.

Rules:
- Use ONLY the information in the context below.
- Add inline citations like [1], [2] after every claim, where the number matches the source id.
- Use plain square brackets exactly like [1]. Do not use decorative citation brackets.
- If the context does not contain the answer, say you don’t have enough information.
- Choose the best format (paragraph, bullet list, or step-by-step) based on the question type.
"""

def synthesize(state: State) -> State:
    prompt = f"""Context:
{state['context']}

Question:
{state['question']}

Answer with inline citations using the source ids like [1], [2]."""

    resp = oai.chat.completions.create(
        model="gpt-4o-mini",
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
