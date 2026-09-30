"""FastAPI entry point: /health, /ingest and /chat."""

import logging
from pathlib import Path

import openai
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.ingest import get_collection, ingest_file
from app.rag import answer_question

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Intelligent Corporate Knowledge Assistant API",
    description=(
        "RAG backend that answers employee questions about internal company policies. "
        "Upload policy documents with `/ingest`, then ask questions with `/chat`. "
        "Out-of-scope questions are politely refused."
    ),
    version="1.0.0",
)

# Simple browser demo UI (demo/index.html + compiled demo/app.js) at /demo
app.mount("/demo", StaticFiles(directory=Path(__file__).resolve().parent.parent / "demo", html=True), name="demo")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["Berapa hari jatah cuti tahunan saya?"])


class Source(BaseModel):
    source: str = Field(..., examples=["Kebijakan_SDM_2025.pdf"])
    section: str = Field(..., examples=["Pasal 5 - Cuti Tahunan"])
    page: int = Field(..., examples=[2])
    similarity: float = Field(..., examples=[0.62])


class ChatResponse(BaseModel):
    answer: str = Field(..., examples=["Jatah cuti tahunan adalah 14 hari kerja per tahun..."])
    sources: list[Source]
    refused: bool = Field(..., examples=[False])


class IngestResponse(BaseModel):
    filename: str = Field(..., examples=["Kebijakan_SDM_2025.pdf"])
    chunks: int = Field(..., examples=[14])


@app.get("/health")
def health() -> dict:
    """Liveness check; also reports how many chunks are in the vector store."""
    return {"status": "ok", "chunks_in_store": get_collection().count()}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(..., description="Policy document (.pdf, .txt or .md)")) -> IngestResponse:
    """Upload a policy document. It is chunked, embedded and stored in ChromaDB.

    Re-uploading a file with the same name replaces its previous chunks.
    """
    content = await file.read()
    try:
        result = ingest_file(file.filename or "", content)
    except ValueError as exc:
        msg = str(exc)
        if msg == "Unsupported file type":
            msg = "Unsupported file type. Please upload a .pdf, .txt or .md file."
        raise HTTPException(status_code=400, detail=msg)
    return IngestResponse(**result)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Ask a question about internal policies. Answers cite their sources.

    Questions outside the ingested policies return the refusal message with `refused: true`.
    """
    try:
        return ChatResponse(**answer_question(req.question))
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="Terlalu banyak permintaan, coba lagi sebentar.")
    except openai.APIError as exc:
        logger.error("LLM API error: %s", exc)
        raise HTTPException(status_code=502, detail="Layanan LLM sedang bermasalah, coba lagi nanti.")
