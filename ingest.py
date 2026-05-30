from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import unquote

import fitz  # PyMuPDF
from openai import OpenAI
from pydantic import BaseModel, Field
from qdrant_client import models

from config import get_qdrant_client
from env_loader import load_app_env
from model_config import embedding_dimensions, model_for_step

load_app_env()

DocType = Literal[
    "passport",
    "id_card",
    "visa",
    "uscis_form",
    "affidavit",
    "lease",
    "generic_pdf",
]
ChunkType = Literal["body", "heading", "table", "key_value", "identity_field", "header", "footer"]

MIN_CHARS = 50
OCR_DPI = 220
DEFAULT_CHUNK_CHARS = 1800
DEFAULT_OVERLAP_CHARS = 260

_openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_ADMIN_KEY")
if not _openai_key:
    print("Missing OPENAI_API_KEY or OPENAI_ADMIN_KEY. Set it in .env or export it in your shell.")
    sys.exit(1)
if not shutil.which("tesseract"):
    print("Warning: tesseract is not installed. Scanned/image-only PDFs will not be OCR-indexed.")

oai = OpenAI(api_key=_openai_key)
qc = get_qdrant_client()


class IdentityFields(BaseModel):
    full_name: str | None = None
    given_name: str | None = None
    surname: str | None = None
    nationality: str | None = None
    passport_number: str | None = None
    id_number: str | None = None
    country_of_issue: str | None = None
    issuing_state: str | None = None
    date_of_birth: str | None = None
    date_of_issue: str | None = None
    date_of_expiry: str | None = None
    document_type: str | None = None


class PageText(BaseModel):
    page: int
    text: str
    markdown: str | None = None
    key_values: dict[str, str] = Field(default_factory=dict)
    tables: list[str] = Field(default_factory=list)


class Document(BaseModel):
    document_id: str
    file_name: str
    doc_type: DocType
    full_text: str
    pages: list[PageText]
    identity_fields: IdentityFields | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    file_name: str
    doc_type: DocType
    page: int | None = None
    section: str | None = None
    heading: str | None = None
    field_label: str | None = None
    person_name: str | None = None
    entities: list[str] = Field(default_factory=list)
    chunk_type: ChunkType = "body"
    text: str
    summary: str | None = None
    keywords: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OCRResult(BaseModel):
    pages: list[PageText]
    raw_response: dict[str, Any] = Field(default_factory=dict)


class IdentityExtractionResult(BaseModel):
    fields: IdentityFields
    confidence: float | None = None
    raw_response: dict[str, Any] = Field(default_factory=dict)


class OCRProvider(ABC):
    @abstractmethod
    def extract_pdf(self, pdf_path: Path) -> OCRResult:
        """Return normalized page text, key-values, and tables from a scanned PDF."""


class IdentityProvider(ABC):
    @abstractmethod
    def extract_identity(self, pdf_path: Path, pages: list[PageText]) -> IdentityExtractionResult:
        """Return normalized passport/ID fields."""


class AzureDocumentIntelligenceOCRProvider(OCRProvider):
    """Adapter placeholder for Azure Document Intelligence Read/Layout.

    Keep provider output normalized to PageText so the rest of the pipeline does
    not depend on Azure. Swap this class for Google Document AI, Textract, etc.
    without changing chunking or Qdrant code.
    """

    def __init__(self, endpoint: str, key: str):
        self.endpoint = endpoint
        self.key = key

    def extract_pdf(self, pdf_path: Path) -> OCRResult:
        raise NotImplementedError("Wire Azure Document Intelligence Read/Layout here")


class AzureIdentityProvider(IdentityProvider):
    """Adapter placeholder for Azure prebuilt ID/custom passport model."""

    def __init__(self, endpoint: str, key: str):
        self.endpoint = endpoint
        self.key = key

    def extract_identity(self, pdf_path: Path, pages: list[PageText]) -> IdentityExtractionResult:
        raise NotImplementedError("Wire Azure identity extraction here")


class LocalTesseractOCRProvider(OCRProvider):
    """Local fallback OCR.

    This is useful for development and simple scanned PDFs. For production
    passports, visas, stamps, and rotated IDs, prefer a document-intelligence
    provider that returns layout and key-value fields.
    """

    def extract_pdf(self, pdf_path: Path) -> OCRResult:
        pages: list[PageText] = []
        doc = fitz.open(str(pdf_path))
        try:
            for index, page in enumerate(doc, start=1):
                pages.append(PageText(page=index, text=ocr_pdf_page(pdf_path, index - 1)))
        finally:
            doc.close()
        return OCRResult(pages=pages)


class HeuristicIdentityProvider(IdentityProvider):
    """Best-effort local field extraction from OCR/digital text.

    This is deliberately conservative. It helps retrieval immediately, but a
    real ID provider should replace it for production-grade passports/IDs.
    """

    def extract_identity(self, pdf_path: Path, pages: list[PageText]) -> IdentityExtractionResult:
        text = "\n".join(page.text for page in pages)
        fields = IdentityFields(document_type="passport" if "passport" in text.lower() else None)

        def first(patterns: list[str]) -> str | None:
            for pattern in patterns:
                match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
                if match:
                    return clean_text(match.group(1))
            return None

        fields.nationality = first([
            r"\bNationality\s*[:\-]?\s*([A-Z][A-Za-z \-/]{2,40})",
            r"\bCitizenship\s*[:\-]?\s*([A-Z][A-Za-z \-/]{2,40})",
        ])
        fields.passport_number = first([
            r"\bPassport\s*(?:No\.?|Number|#)\s*[:\-]?\s*([A-Z0-9]{5,20})",
            r"\bDocument\s*(?:No\.?|Number|#)\s*[:\-]?\s*([A-Z0-9]{5,20})",
        ])
        fields.date_of_birth = first([
            r"\bDate of Birth\s*[:\-]?\s*([0-9A-Za-z/ .\-]{6,25})",
            r"\bDOB\s*[:\-]?\s*([0-9A-Za-z/ .\-]{6,25})",
        ])
        fields.date_of_expiry = first([
            r"\bDate of Expiry\s*[:\-]?\s*([0-9A-Za-z/ .\-]{6,25})",
            r"\bExpiration Date\s*[:\-]?\s*([0-9A-Za-z/ .\-]{6,25})",
        ])
        fields.date_of_issue = first([
            r"\bDate of Issue\s*[:\-]?\s*([0-9A-Za-z/ .\-]{6,25})",
            r"\bIssue Date\s*[:\-]?\s*([0-9A-Za-z/ .\-]{6,25})",
        ])
        fields.country_of_issue = first([
            r"\bCountry of Issue\s*[:\-]?\s*([A-Z][A-Za-z \-/]{2,40})",
            r"\bIssuing Country\s*[:\-]?\s*([A-Z][A-Za-z \-/]{2,40})",
        ])
        fields.full_name = first([
            r"\bFull Name\s*[:\-]?\s*([A-Z][A-Za-z ,.'\-]{3,80})",
            r"\bName\s*[:\-]?\s*([A-Z][A-Za-z ,.'\-]{3,80})",
        ])
        return IdentityExtractionResult(fields=fields)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def stable_id(*parts: str) -> str:
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()


def stable_uuid(*parts: str) -> str:
    """Return a deterministic UUID string accepted by Qdrant point IDs."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "::".join(parts)))


def display_source(path: Path) -> str:
    return unquote(path.name).strip()


def is_scanned_pdf(pdf_path: str | Path, min_chars_per_page: int = 40) -> bool:
    """Detect image-only PDFs by checking for usable text layer.

    A PDF can contain images and still be digital, so image presence alone is
    not enough. If fewer than ~35% of pages have meaningful text, route to OCR.
    """

    path = Path(pdf_path)
    doc = fitz.open(str(path))
    try:
        if len(doc) == 0:
            return True
        text_pages = 0
        for page in doc:
            if len(clean_text(page.get_text("text") or "")) >= min_chars_per_page:
                text_pages += 1
        return (text_pages / max(len(doc), 1)) < 0.35
    finally:
        doc.close()


def detect_document_type(pdf_path: str | Path, sample_text: str = "") -> DocType:
    path = Path(pdf_path)
    haystack = f"{path.name}\n{sample_text[:4000]}".lower()
    if any(term in haystack for term in ("passport", "nationality", "date of expiry", "place of birth")):
        return "passport"
    if any(term in haystack for term in ("identity card", "id card", "identification card")):
        return "id_card"
    if any(term in haystack for term in ("visa", "nonimmigrant", "immigrant visa")):
        return "visa"
    if any(term in haystack for term in ("i-485", "i-130", "i-589", "uscis", "alien registration")):
        return "uscis_form"
    if "affidavit" in haystack:
        return "affidavit"
    if "lease" in haystack:
        return "lease"
    return "generic_pdf"


def extract_digital_pdf(pdf_path: str | Path) -> list[PageText]:
    """Extract digital PDF text with PyMuPDF.

    PyMuPDF is preferred over pypdf because block extraction gives us geometry.
    Sorting blocks top-to-bottom and left-to-right mitigates broken stream order
    and many multi-column layouts. `find_tables()` keeps tables as table chunks
    instead of flattening them into unreadable prose. Font encoding and ligature
    problems can still happen in malformed PDFs, but PyMuPDF generally handles
    embedded font maps better than pypdf.
    """

    path = Path(pdf_path)
    doc = fitz.open(str(path))
    pages: list[PageText] = []
    try:
        for page_index, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks") or []
            blocks = sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))
            parts = [clean_text(str(block[4] or "")) for block in blocks]
            text = clean_text("\n\n".join(part for part in parts if part))

            tables: list[str] = []
            try:
                for table in page.find_tables().tables:
                    rows = table.extract()
                    rendered = "\n".join(
                        " | ".join(clean_text(str(cell or "")) for cell in row)
                        for row in rows
                    )
                    if rendered.strip():
                        tables.append(rendered)
            except Exception:
                pass

            pages.append(PageText(page=page_index, text=text, tables=tables))
    finally:
        doc.close()
    return pages


def ocr_pdf_page(pdf_path: Path, page_index: int) -> str:
    if not shutil.which("tesseract"):
        return ""
    try:
        doc = fitz.open(str(pdf_path))
        try:
            page = doc.load_page(page_index)
            pix = page.get_pixmap(dpi=OCR_DPI, alpha=False)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as image_file:
                pix.save(image_file.name)
                result = subprocess.run(
                    ["tesseract", image_file.name, "stdout", "--psm", "6"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=90,
                )
                if result.returncode != 0:
                    print(f"OCR failed on page {page_index + 1}: {result.stderr.strip()}")
                    return ""
                return clean_text(result.stdout)
        finally:
            doc.close()
    except Exception as exc:
        print(f"OCR failed on page {page_index + 1}: {exc}")
        return ""


def extract_scanned_pdf(pdf_path: str | Path, ocr_provider: OCRProvider | None = None) -> list[PageText]:
    provider = ocr_provider or LocalTesseractOCRProvider()
    return provider.extract_pdf(Path(pdf_path)).pages


def extract_identity_fields(
    pdf_path: str | Path,
    pages: list[PageText],
    identity_provider: IdentityProvider | None = None,
) -> IdentityFields | None:
    provider = identity_provider or HeuristicIdentityProvider()
    fields = provider.extract_identity(Path(pdf_path), pages).fields
    return fields if fields.model_dump(exclude_none=True) else None


def build_document(
    pdf_path: str | Path,
    ocr_provider: OCRProvider | None = None,
    identity_provider: IdentityProvider | None = None,
) -> Document:
    path = Path(pdf_path)
    scanned = is_scanned_pdf(path)
    pages = extract_scanned_pdf(path, ocr_provider) if scanned else extract_digital_pdf(path)

    # Some PDFs have a text layer but broken extraction. OCR weak pages only.
    repaired_pages: list[PageText] = []
    for page in pages:
        if len(page.text.strip()) < MIN_CHARS:
            ocr_text = ocr_pdf_page(path, page.page - 1)
            if len(ocr_text) > len(page.text):
                page = page.model_copy(update={"text": ocr_text})
        repaired_pages.append(page)
    pages = repaired_pages

    full_text = clean_text("\n\n".join(page.text for page in pages))
    doc_type = detect_document_type(path, full_text)
    identity_fields = None
    if doc_type in {"passport", "id_card", "visa"}:
        identity_fields = extract_identity_fields(path, pages, identity_provider)

    return Document(
        document_id=str(uuid.uuid4()),
        file_name=display_source(path),
        doc_type=doc_type,
        full_text=full_text,
        pages=pages,
        identity_fields=identity_fields,
        metadata={"is_scanned": scanned, "page_count": len(pages)},
    )


def derive_keywords(text: str) -> list[str]:
    labels = [
        "nationality",
        "citizenship",
        "passport number",
        "date of birth",
        "date of expiry",
        "date of issue",
        "attorney",
        "employer",
        "address",
        "alien number",
        "receipt number",
    ]
    lower = text.lower()
    return [label for label in labels if label in lower]


def infer_heading(text: str) -> str | None:
    first = text.splitlines()[0].strip() if text.splitlines() else ""
    if 3 <= len(first) <= 100 and not first.endswith("."):
        return first
    return None


def chunk_text_block(text: str, max_chars: int = DEFAULT_CHUNK_CHARS, overlap_chars: int = DEFAULT_OVERLAP_CHARS) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}".strip()
        else:
            if current:
                chunks.append(current)
            overlap = current[-overlap_chars:] if current else ""
            current = f"{overlap}\n\n{paragraph}".strip() if overlap else paragraph
    if current:
        chunks.append(current)
    return chunks


def identity_field_chunks(document: Document) -> list[Chunk]:
    if not document.identity_fields:
        return []
    fields = document.identity_fields.model_dump(exclude_none=True)
    person_name = (
        document.identity_fields.full_name
        or " ".join(x for x in [document.identity_fields.given_name, document.identity_fields.surname] if x)
        or None
    )
    chunks: list[Chunk] = []
    for label, value in fields.items():
        text = f"{label.replace('_', ' ').title()}: {value}"
        chunks.append(
            Chunk(
                chunk_id=stable_id(document.document_id, label, str(value)),
                document_id=document.document_id,
                file_name=document.file_name,
                doc_type=document.doc_type,
                field_label=label,
                person_name=person_name,
                entities=[person_name] if person_name else [],
                chunk_type="identity_field",
                text=text,
                keywords=[label, str(value)],
                metadata={"structured_identity": True},
            )
        )
    return chunks


def enrich_chunk_metadata(chunk: Chunk, chunk_index: int) -> Chunk:
    text = clean_text(chunk.text)
    summary = text[:240] + "..." if len(text) > 240 else text
    metadata = {
        **chunk.metadata,
        "document_id": chunk.document_id,
        "file_name": chunk.file_name,
        "doc_type": chunk.doc_type,
        "page": chunk.page,
        "chunk_index": chunk_index,
        "field_label": chunk.field_label,
        "person_name": chunk.person_name,
        "chunk_type": chunk.chunk_type,
        "keywords": chunk.keywords,
        "section": chunk.section,
        "heading": chunk.heading,
    }
    return chunk.model_copy(update={"text": text, "summary": summary, "metadata": metadata})


def chunk_document(document: Document) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunks.extend(identity_field_chunks(document))

    for page in document.pages:
        for label, value in page.key_values.items():
            chunks.append(
                Chunk(
                    chunk_id=stable_id(document.document_id, str(page.page), label, value),
                    document_id=document.document_id,
                    file_name=document.file_name,
                    doc_type=document.doc_type,
                    page=page.page,
                    field_label=label,
                    chunk_type="key_value",
                    text=f"{label}: {value}",
                    keywords=[label, value],
                )
            )
        for table_index, table in enumerate(page.tables):
            chunks.append(
                Chunk(
                    chunk_id=stable_id(document.document_id, str(page.page), "table", str(table_index)),
                    document_id=document.document_id,
                    file_name=document.file_name,
                    doc_type=document.doc_type,
                    page=page.page,
                    chunk_type="table",
                    text=table,
                    keywords=["table"],
                )
            )
        for index, text in enumerate(chunk_text_block(page.text), start=1):
            heading = infer_heading(text)
            chunks.append(
                Chunk(
                    chunk_id=stable_id(document.document_id, str(page.page), str(index), text[:80]),
                    document_id=document.document_id,
                    file_name=document.file_name,
                    doc_type=document.doc_type,
                    page=page.page,
                    heading=heading,
                    section=heading,
                    chunk_type="body",
                    text=text,
                    keywords=derive_keywords(text),
                )
            )

    return [enrich_chunk_metadata(chunk, index) for index, chunk in enumerate(chunks, start=1)]


def build_qdrant_payloads(
    chunks: list[Chunk],
    user_id: str | None = None,
    is_public: bool = False,
    indexed_by_email: str | None = None,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for chunk in chunks:
        payload = {
            **chunk.metadata,
            "chunk_id": chunk.chunk_id,
            "point_id": stable_uuid(chunk.document_id, chunk.chunk_id),
            "source": chunk.file_name,
            "text": chunk.text,
            "summary": chunk.summary,
            "entities": chunk.entities,
            "hash": hashlib.md5(chunk.text.encode("utf-8")).hexdigest(),
            "isPublic": bool(is_public),
        }
        if user_id:
            payload["user_id"] = str(user_id)
        if indexed_by_email:
            payload["indexed_by_email"] = indexed_by_email
            payload["indexed_by"] = indexed_by_email
        payloads.append(payload)
    return payloads


def ensure_collection(name: str) -> None:
    try:
        qc.get_collection(name)
    except Exception:
        qc.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=embedding_dimensions(), distance=models.Distance.COSINE),
        )
        print(f"Created collection: {name}")


def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = oai.embeddings.create(
        model=model_for_step("embedding_model"),
        input=texts,
        dimensions=embedding_dimensions(),
    )
    return [item.embedding for item in resp.data]


def ingest_to_qdrant(chunks_with_meta: list[dict[str, Any]], collection: str) -> int:
    ensure_collection(collection)
    total_upserted = 0
    batch_size = 100

    for start in range(0, len(chunks_with_meta), batch_size):
        batch = chunks_with_meta[start : start + batch_size]
        embeddings = embed_batch([item["text"] for item in batch])
        points = [
            models.PointStruct(
                id=item.get("point_id") or stable_uuid(str(item.get("chunk_id") or uuid.uuid4())),
                vector=vector,
                payload=item,
            )
            for item, vector in zip(batch, embeddings)
        ]
        qc.upsert(collection_name=collection, points=points)
        total_upserted += len(points)

    print(f"✓ Ingested {total_upserted} chunks into '{collection}'")
    return total_upserted


def extract(
    path: str,
    user_id: str | None = None,
    is_public: bool = False,
    indexed_by_email: str | None = None,
) -> list[dict[str, Any]]:
    pdf_path = Path(path)
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        return []

    document = build_document(pdf_path)
    chunks = chunk_document(document)
    payloads = build_qdrant_payloads(chunks, user_id=user_id, is_public=is_public, indexed_by_email=indexed_by_email)

    print(f"Document type: {document.doc_type}")
    print(f"Scanned: {document.metadata.get('is_scanned')}")
    print(f"Pages: {len(document.pages)}")
    print(f"Total chunks collected: {len(payloads)}")
    if document.identity_fields:
        print(f"Identity fields: {document.identity_fields.model_dump(exclude_none=True)}")
    if payloads:
        print("First chunk:", payloads[0]["text"][:500])
    return payloads


def ingest(
    pdf_path: str,
    collection: str,
    user_id: str | None = None,
    is_public: bool = False,
    indexed_by_email: str | None = None,
) -> int:
    chunks_with_meta = extract(
        pdf_path,
        user_id=user_id,
        is_public=is_public,
        indexed_by_email=indexed_by_email,
    )
    if not chunks_with_meta:
        print("No chunks to ingest.")
        return 0
    return ingest_to_qdrant(chunks_with_meta, collection)


def ingest_pdf(
    pdf_path: str | Path,
    collection: str,
    user_id: str | None,
    embed: Callable[[str], list[float]] | None = None,
    is_public: bool = False,
    indexed_by_email: str | None = None,
) -> dict[str, Any]:
    """Example composable flow for tests or future API code."""

    document = build_document(pdf_path)
    chunks = chunk_document(document)
    payloads = build_qdrant_payloads(chunks, user_id=user_id, is_public=is_public, indexed_by_email=indexed_by_email)
    if embed is None:
        indexed = ingest_to_qdrant(payloads, collection)
    else:
        ensure_collection(collection)
        points = [
            models.PointStruct(
                id=payload.get("point_id") or stable_uuid(str(payload["chunk_id"])),
                vector=embed(payload["text"]),
                payload=payload,
            )
            for payload in payloads
        ]
        if points:
            qc.upsert(collection_name=collection, points=points)
        indexed = len(points)
    return {
        "document": document.model_dump(),
        "chunks": [chunk.model_dump() for chunk in chunks],
        "chunks_indexed": indexed,
        "full_text": document.full_text,
        "identity_fields": document.identity_fields.model_dump() if document.identity_fields else None,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <pdf-path> [collection]")
        sys.exit(1)

    pdf_path_arg = sys.argv[1]
    collection_arg = sys.argv[2] if len(sys.argv) > 2 else None
    if collection_arg:
        ingest(pdf_path_arg, collection_arg)
    else:
        extract(pdf_path_arg)
