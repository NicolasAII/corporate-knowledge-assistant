"""Question answering: embed query -> retrieve -> guardrail -> LLM answer."""

import logging
from functools import lru_cache

from openai import OpenAI

from app.config import GROQ_API_KEY, GROQ_BASE_URL, LLM_MODEL, MIN_SIMILARITY, TOP_K
from app.ingest import embed, get_collection
from app.prompts import REFUSAL_MESSAGE, SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

EMPTY_STORE_MESSAGE = "Belum ada dokumen yang di-ingest. Silakan unggah dokumen melalui /ingest."


@lru_cache(maxsize=1)
def get_llm() -> OpenAI:
    """Groq client via its OpenAI-compatible endpoint, created once."""
    return OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)


def _refusal() -> dict:
    return {"answer": REFUSAL_MESSAGE, "sources": [], "refused": True}


def _is_refusal(answer: str) -> bool:
    """True if the model returned (or contained) the refusal sentence."""
    return not answer or REFUSAL_MESSAGE.rstrip(".").lower() in answer.lower()


def answer_question(question: str) -> dict:
    """Answer a question from the ingested policies, or refuse if out of scope."""
    collection = get_collection()
    if collection.count() == 0:
        return {"answer": EMPTY_STORE_MESSAGE, "sources": [], "refused": True}

    res = collection.query(
        query_embeddings=embed([question]),
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    hits = [
        {**meta, "text": doc, "similarity": round(1 - dist, 4)}
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        if meta is not None and doc is not None
    ]
    best = max((h["similarity"] for h in hits), default=0.0)

    # Guardrail layer 1: nothing relevant retrieved -> refuse without calling the LLM.
    if best < MIN_SIMILARITY:
        logger.info("question=%r best_similarity=%.4f refused=True (threshold)", question, best)
        return _refusal()

    chunks = [h for h in hits if h["similarity"] >= MIN_SIMILARITY]
    resp = get_llm().chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, chunks)},
        ],
        temperature=0,
        reasoning_effort="low",  # gpt-oss is a reasoning model; low keeps it fast
        max_tokens=1024,
    )
    answer = (resp.choices[0].message.content or "").strip()

    # Guardrail layer 2: the model itself decided the question is out of scope.
    if _is_refusal(answer):
        logger.info("question=%r best_similarity=%.4f refused=True (llm)", question, best)
        return _refusal()

    logger.info("question=%r best_similarity=%.4f refused=False", question, best)
    return {
        "answer": answer,
        "sources": [
            {k: c[k] for k in ("source", "section", "page", "similarity")} for c in chunks
        ],
        "refused": False,
    }
