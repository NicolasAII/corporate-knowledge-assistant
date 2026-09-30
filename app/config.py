"""Application settings loaded from environment variables / .env file."""

import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "policies")
TOP_K: int = int(os.getenv("TOP_K", "4"))
MIN_SIMILARITY: float = float(os.getenv("MIN_SIMILARITY", "0.45"))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

if not GROQ_API_KEY or GROQ_API_KEY == "your-groq-key-here":
    raise RuntimeError(
        "GROQ_API_KEY is not set. Copy .env.example to .env and paste your key "
        "from https://console.groq.com/keys"
    )
