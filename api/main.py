import os
import json
import sys
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

load_dotenv(os.path.join(BASE_DIR, ".env"))

FASTAPI_HOST     = os.getenv("FASTAPI_HOST",     "0.0.0.0")
FASTAPI_PORT     = int(os.getenv("FASTAPI_PORT", "8000"))
TOP_K_DEFAULT    = int(os.getenv("TOP_K_DEFAULT", "5"))
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "index/faiss.index")
METADATA_PATH    = os.getenv("METADATA_PATH",    "index/metadata.json")
DATA_PATH        = os.getenv("DATA_PATH",        "data/poems_raw.json")
CHUNKS_PATH      = os.getenv("CHUNKS_PATH",      "data/poems_chunks.json")
PRIMARY_MODEL    = os.getenv("PRIMARY_MODEL",    "gpt-oss:20b-cloud")
JAMBA_MODEL      = os.getenv("JAMBA_MODEL",      "jamba-mini")

app = FastAPI(
    title="Arabic Poetry RAG API",
    description="نظام استرجاع وتوليد للشعر العربي | RAG system for Arabic poetry",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="السؤال باللغة العربية")
    model:    str = Field(default="gpt-oss",  description="'gpt-oss' أو 'jamba'")
    top_k:    int = Field(default=5, ge=1, le=50, description="عدد المقاطع المسترجعة")


class SourceChunk(BaseModel):
    title:      str   = ""
    poet_name:  str   = ""
    era:        str   = ""
    num_lines:  str   = ""
    word_count: int   = 0
    poem_url:   str   = ""
    score:      float = 0.0
    text:       str   = ""


class AskResponse(BaseModel):
    answer:     str
    sources:    list[SourceChunk]
    model_used: str
    time_ms:    float


class HealthResponse(BaseModel):
    status:           str
    models_available: list[str]
    index_loaded:     bool
    timestamp:        str


class StatsResponse(BaseModel):
    total_poems:  int
    total_chunks: int
    index_size:   int


_retriever = None
_generator = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from rag.retriever import retrieve, format_context
        _retriever = {"retrieve": retrieve, "format_context": format_context}
    return _retriever


def _get_generator():
    global _generator
    if _generator is None:
        from rag.generator import generate_answer
        _generator = generate_answer
    return _generator


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Answer a question about Arabic poetry using the RAG pipeline.

    - **question**: The Arabic question to answer.
    - **model**: LLM to use — `gpt-oss` (Ollama) or `jamba` (AI21).
    - **top_k**: Number of poem chunks to retrieve (1–50).
    """
    if request.model not in ("gpt-oss", "jamba"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{request.model}'. Choose 'gpt-oss' or 'jamba'.",
        )

    try:
        retriever = _get_retriever()
        chunks    = retriever["retrieve"](request.question, top_k=request.top_k)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")

    try:
        generate_answer = _get_generator()
        result          = generate_answer(request.question, chunks, model_name=request.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")

    sources = [
        SourceChunk(
            title      = c.get("title",      ""),
            poet_name  = c.get("poet_name",  ""),
            era        = c.get("era",        ""),
            num_lines  = c.get("num_lines",  ""),
            word_count = c.get("word_count", 0),
            poem_url   = c.get("poem_url",   ""),
            score      = c.get("score",      0.0),
            text       = c.get("text",       "")[:300],
        )
        for c in chunks
    ]

    return AskResponse(
        answer     = result["answer"],
        sources    = sources,
        model_used = result["model_used"],
        time_ms    = result["time_ms"],
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check service health and model availability."""
    index_loaded = os.path.exists(
        os.path.join(BASE_DIR, FAISS_INDEX_PATH)
    ) or os.path.exists(FAISS_INDEX_PATH)

    available_models = []

    try:
        import ollama
        ollama_models = [m.model for m in ollama.list().models]
        primary       = os.getenv("PRIMARY_MODEL", "gpt-oss:20b-cloud")
        if any(primary in m for m in ollama_models):
            available_models.append("gpt-oss")
        else:
            available_models.append("gpt-oss (not pulled)")
    except Exception:
        available_models.append("gpt-oss (ollama unavailable)")

    if os.getenv("JAMBA_API_KEY", ""):
        available_models.append("jamba")
    else:
        available_models.append("jamba (no API key)")

    return HealthResponse(
        status           = "ok",
        models_available = available_models,
        index_loaded     = index_loaded,
        timestamp        = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Return statistics about the loaded data and index."""
    total_poems  = 0
    total_chunks = 0
    index_size   = 0

    data_path = os.path.join(BASE_DIR, DATA_PATH) if not os.path.isabs(DATA_PATH) else DATA_PATH
    if os.path.exists(data_path):
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                total_poems = len(json.load(f))
        except Exception:
            pass

    chunks_path = os.path.join(BASE_DIR, CHUNKS_PATH) if not os.path.isabs(CHUNKS_PATH) else CHUNKS_PATH
    if os.path.exists(chunks_path):
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                total_chunks = len(json.load(f))
        except Exception:
            pass

    faiss_path = os.path.join(BASE_DIR, FAISS_INDEX_PATH) if not os.path.isabs(FAISS_INDEX_PATH) else FAISS_INDEX_PATH
    if os.path.exists(faiss_path):
        try:
            import faiss
            idx        = faiss.read_index(faiss_path)
            index_size = idx.ntotal
        except Exception:
            pass

    return StatsResponse(
        total_poems  = total_poems,
        total_chunks = total_chunks,
        index_size   = index_size,
    )


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("[AR] تشغيل خادم FastAPI للشعر العربي")
    print("[EN] Starting Arabic Poetry RAG FastAPI server")
    print(f"[EN] Host: {FASTAPI_HOST}  Port: {FASTAPI_PORT}")
    print(f"[EN] Docs: http://localhost:{FASTAPI_PORT}/docs")
    print("=" * 60)

    uvicorn.run(
        "main:app",
        host     = FASTAPI_HOST,
        port     = FASTAPI_PORT,
        reload   = False,
        log_level= "info",
    )