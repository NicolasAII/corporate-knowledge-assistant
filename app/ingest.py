"""Document ingestion: extract text -> section-aware chunking -> embed -> upsert to Chroma."""

import io
import re
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from app.config import CHROMA_PATH, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME, EMBEDDING_MODEL

# A heading is a line like "BAB II - WAKTU KERJA" or "Pasal 3 - Jam Kerja".
# Requiring the dash avoids matching in-text references such as "Pasal 5 tetap berlaku".
HEADING_RE = re.compile(r"^[ \t]*(?:BAB\s+[IVXLCDM]+|Pasal\s+\d+)\s*[-–]\s*\S.*$", re.I | re.M)
PARAGRAPH_RE = re.compile(r"\S(?:(?!\n[ \t]*\n).)*", re.S)
# Sentence/clause boundaries: after ". ", "; " or ": " (not "1. " numbering),
# or before a new numbered/lettered clause line.
SENTENCE_SPLIT_RE = re.compile(r"(?<![0-9])[.;:]\s+|\n(?=[ \t]*(?:\d+|[a-z])\.\s)")


# ---------- Text extraction ----------

def extract_text(filename: str, content: bytes) -> list[tuple[int, str]]:
    """Return a list of (page_number, text) for a PDF, TXT or MD file."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(content))
            return [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]
        except Exception as exc:
            raise ValueError("File PDF tidak dapat dibaca") from exc
    if ext in (".txt", ".md"):
        return [(1, content.decode("utf-8", errors="replace"))]
    raise ValueError("Unsupported file type")


# ---------- Chunking ----------

def _split_sections(text: str) -> list[tuple[str, str, int]]:
    """Split text into (heading, body, body_offset) at every BAB/Pasal heading line."""
    headings = list(HEADING_RE.finditer(text))
    sections = []
    first_start = headings[0].start() if headings else len(text)
    sections.append(("Pendahuluan", text[:first_start], 0))
    for i, m in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        sections.append((m.group().strip(), text[m.end():end], m.end()))
    return sections


def _atoms(body: str, limit: int) -> list[tuple[int, int]]:
    """Break a body into (start, end) spans: paragraphs, then sentences, then hard cuts."""
    spans = []
    for para in PARAGRAPH_RE.finditer(body):
        p_start, p_end = para.start(), para.end()
        if p_end - p_start <= limit:
            spans.append((p_start, p_end))
            continue
        cuts = [p_start] + [m.end() for m in SENTENCE_SPLIT_RE.finditer(body, p_start, p_end)] + [p_end]
        for s, e in zip(cuts, cuts[1:]):
            while e - s > limit:  # a single sentence longer than the limit
                spans.append((s, s + limit))
                s += limit
            if body[s:e].strip():
                spans.append((s, e))
    return spans


def _merge(body: str, spans: list[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
    """Greedily merge consecutive spans into chunks of at most `limit` characters."""
    chunks: list[tuple[int, int]] = []
    for s, e in spans:
        if chunks and e - chunks[-1][0] <= limit:
            chunks[-1] = (chunks[-1][0], e)
        else:
            chunks.append((s, e))
    return chunks


def chunk_text(text: str) -> list[dict]:
    """Section-aware recursive chunking.

    Returns dicts with `text` (heading-prefixed), `section` and `offset`
    (start position of the chunk in `text`, used for page lookup).
    """
    result = []
    for heading, body, body_offset in _split_sections(text):
        if not body.strip():
            continue
        limit = max(CHUNK_SIZE - len(heading) - 1, 100)
        pieces = _merge(body, _atoms(body, limit), limit)
        prev = ""
        for s, e in pieces:
            piece = body[s:e].strip()
            overlap = ""
            if prev and CHUNK_OVERLAP > 0:
                tail = prev[-CHUNK_OVERLAP:]
                # start the overlap at a word boundary
                overlap = (tail.split(" ", 1)[-1] if " " in tail else tail).strip() + "\n"
            prev = piece
            chunk = f"{heading}\n{overlap}{piece}".strip()
            if chunk:
                result.append({"text": chunk, "section": heading, "offset": body_offset + s})
    return result


# ---------- Embedding and storage ----------

@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load the embedding model once per process."""
    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts with normalized vectors (cosine similarity == dot product)."""
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()


@lru_cache(maxsize=1)
def get_collection() -> chromadb.Collection:
    """Return the persistent Chroma collection (cosine distance)."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def ingest_file(filename: str, content: bytes) -> dict:
    """Extract, chunk, embed and store a document, replacing any previous version."""
    filename = Path(filename).name
    pages = extract_text(filename, content)

    # Concatenate pages so sections spanning a page break stay intact; remember page offsets.
    page_starts, parts, pos = [], [], 0
    for _, page_text in pages:
        page_starts.append(pos)
        parts.append(page_text)
        pos += len(page_text) + 1
    full_text = "\n".join(parts)

    chunks = chunk_text(full_text)
    if not chunks:
        raise ValueError("Tidak ada teks yang dapat diekstrak")

    collection = get_collection()
    collection.delete(where={"source": filename})
    collection.upsert(
        ids=[f"{filename}::{i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        embeddings=embed([c["text"] for c in chunks]),
        metadatas=[
            {
                "source": filename,
                "section": c["section"],
                "page": pages[bisect_right(page_starts, c["offset"]) - 1][0],
                "chunk_index": i,
            }
            for i, c in enumerate(chunks)
        ],
    )
    return {"filename": filename, "chunks": len(chunks)}
