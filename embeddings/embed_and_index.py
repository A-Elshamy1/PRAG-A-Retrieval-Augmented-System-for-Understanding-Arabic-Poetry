import os
import json
import time
import numpy as np
from dotenv import load_dotenv

load_dotenv()

CHUNKS_PATH      = os.getenv("CHUNKS_PATH",      "data/poems_chunks.json")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "index/faiss.index")
METADATA_PATH    = os.getenv("METADATA_PATH",    "index/metadata.json")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL",  "intfloat/multilingual-e5-small")
BATCH_SIZE       = 32


def load_chunks(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_chunks(chunks: list[dict], model_name: str, batch_size: int = BATCH_SIZE) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    print(f"[EN] Loading embedding model: {model_name}")
    print(f"[AR] جاري تحميل نموذج التضمين: {model_name}")
    model = SentenceTransformer(model_name)

    texts      = [f"passage: {c['text']}" for c in chunks]
    total      = len(texts)
    start_time = time.time()

    print(f"[EN] Embedding {total} chunks with batch size {batch_size}")
    print(f"[AR] جاري تضمين {total} قطعة نصية بحجم دفعة {batch_size}\n")

    all_embeddings = []
    for i in range(0, total, batch_size):
        batch      = texts[i : i + batch_size]
        embeddings = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(embeddings)

        done    = min(i + batch_size, total)
        percent = (done / total) * 100
        print(f"[EN] Progress: {done}/{total} ({percent:.1f}%)", end="\r")

    all_embeddings = np.vstack(all_embeddings).astype("float32")
    elapsed        = time.time() - start_time

    print(f"\n[EN] Embedding complete in {elapsed:.1f}s")
    print(f"[AR] اكتمل التضمين في {elapsed:.1f} ثانية")
    print(f"[EN] Embedding matrix shape: {all_embeddings.shape}")
    return all_embeddings


def build_faiss_index(embeddings: np.ndarray):
    import faiss

    dim   = embeddings.shape[1]
    print(f"\n[EN] Building FAISS IndexFlatIP with dimension {dim}")
    print(f"[AR] بناء فهرس FAISS بأبعاد {dim}")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"[EN] Added {index.ntotal} vectors to the FAISS index")
    print(f"[AR] تم إضافة {index.ntotal} متجه إلى الفهرس")
    return index


def save_index(index, path: str) -> None:
    import faiss
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    faiss.write_index(index, path)
    print(f"[EN] FAISS index saved to: {path}")
    print(f"[AR] تم حفظ الفهرس في: {path}")


def save_metadata(chunks: list[dict], path: str) -> None:
    metadata = [
        {
            "chunk_id":     c.get("chunk_id",     ""),
            "chunk_index":  c.get("chunk_index",  0),
            "total_chunks": c.get("total_chunks", 1),
            "text":         c.get("text",         ""),
            "word_count":   c.get("word_count",   0),
            "title":        c.get("title",        ""),
            "poet_name":    c.get("poet_name",    ""),
            "era":          c.get("era",          ""),
            "num_lines":    c.get("num_lines",    ""),
            "poem_url":     c.get("poem_url",     ""),
        }
        for c in chunks
    ]
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[EN] Metadata saved to: {path}")
    print(f"[AR] تم حفظ البيانات الوصفية في: {path}")


def main():
    print("=" * 60)
    print("[AR] بدء عملية التضمين وبناء الفهرس")
    print("[EN] Starting embedding and FAISS index creation")
    print("=" * 60)

    if not os.path.exists(CHUNKS_PATH):
        print(f"[EN] ERROR: Chunks file not found at {CHUNKS_PATH}")
        print(f"[AR] خطأ: ملف القطع غير موجود في {CHUNKS_PATH}")
        print("[EN] Please run preprocessing/clean_and_chunk.py first.")
        return

    chunks = load_chunks(CHUNKS_PATH)
    print(f"[EN] Loaded {len(chunks)} chunks")
    print(f"[AR] تم تحميل {len(chunks)} قطعة نصية\n")

    if not chunks:
        print("[EN] ERROR: No chunks to embed.")
        return

    embeddings = embed_chunks(chunks, EMBEDDING_MODEL)
    index      = build_faiss_index(embeddings)

    save_index(index, FAISS_INDEX_PATH)
    save_metadata(chunks, METADATA_PATH)

    print("\n[EN] Summary:")
    print(f"  Embedding shape : {embeddings.shape}")
    print(f"  FAISS vectors   : {index.ntotal}")
    print(f"  Index path      : {FAISS_INDEX_PATH}")
    print(f"  Metadata path   : {METADATA_PATH}")
    print("\n[AR] اكتملت العملية بنجاح")
    print("[EN] Pipeline complete")


if __name__ == "__main__":
    main()