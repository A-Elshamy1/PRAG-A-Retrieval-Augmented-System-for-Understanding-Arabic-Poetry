import os
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "index/faiss.index")
METADATA_PATH    = os.getenv("METADATA_PATH",    "index/metadata.json")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL",  "intfloat/multilingual-e5-small")
TOP_K_DEFAULT    = int(os.getenv("TOP_K_DEFAULT", "5"))

_faiss_index  = None
_metadata     = []
_embed_model  = None
_resources_loaded = False


def _load_resources():
    global _faiss_index, _metadata, _embed_model, _resources_loaded

    if _resources_loaded:
        return

    import faiss
    from sentence_transformers import SentenceTransformer

    print("[EN] Loading FAISS index...")
    print("[AR] جاري تحميل الفهرس...")

    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"[EN] FAISS index not found at {FAISS_INDEX_PATH}. "
            "Run embeddings/embed_and_index.py first."
        )
    _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    print(f"[EN] Loaded FAISS index with {_faiss_index.ntotal} vectors")

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"[EN] Metadata not found at {METADATA_PATH}. "
            "Run embeddings/embed_and_index.py first."
        )
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        _metadata = json.load(f)
    print(f"[EN] Loaded metadata for {len(_metadata)} chunks")

    _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"[EN] Embedding model loaded: {EMBEDDING_MODEL}")

    _resources_loaded = True
    print("[EN] All resources ready")
    print("[AR] جميع الموارد جاهزة\n")


def retrieve(query: str, top_k: int = TOP_K_DEFAULT) -> list[dict]:
    _load_resources()

    query_vec = _embed_model.encode(
        [f"query: {query}"],
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    scores, indices = _faiss_index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_metadata):
            continue
        chunk = _metadata[idx]
        results.append({
            "score":      float(score),
            "text":       chunk.get("text",       ""),
            "title":      chunk.get("title",      ""),
            "poet_name":  chunk.get("poet_name",  ""),
            "era":        chunk.get("era",        ""),
            "num_lines":  chunk.get("num_lines",  ""),
            "word_count": chunk.get("word_count", 0),
            "poem_url":   chunk.get("poem_url",   ""),
            "chunk_id":   chunk.get("chunk_id",   ""),
        })

    return results


def format_context(chunks: list[dict]) -> str:
    context_parts = []
    for i, c in enumerate(chunks, 1):
        context_parts.append(
            f"[{i}] عنوان القصيدة: {c['title']}\n"
            f"    الشاعر: {c['poet_name']} — العصر: {c['era']} — عدد الأبيات: {c['num_lines']}\n"
            f"    النص:\n{c['text']}\n"
            f"    المصدر: {c['poem_url']}"
        )
    return "\n\n".join(context_parts)


if __name__ == "__main__":
    print("=" * 60)
    print("[AR] اختبار نظام الاسترجاع")
    print("[EN] Testing the retriever")
    print("=" * 60)

    test_queries = [
        "قصائد عن الحب والغزل",
        "شعر المتنبي في المدح",
        "القصائد الأندلسية",
    ]

    for query in test_queries:
        print(f"\n[EN] Query: {query}")
        try:
            results = retrieve(query, top_k=3)
            for r in results:
                print(f"  Score: {r['score']:.4f} | {r['title']} — {r['poet_name']}")
        except Exception as e:
            print(f"  [EN] ERROR: {e}")