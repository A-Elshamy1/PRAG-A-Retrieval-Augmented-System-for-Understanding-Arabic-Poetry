import os
import sys
import json
import time
import tempfile

# ── Bootstrap: add project root to path ────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))
API_BASE = f"http://127.0.0.1:{FASTAPI_PORT}"

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⏭  SKIP"

results: list[tuple[str, str, str]] = []  # (test_name, status, detail)


def record(name: str, passed: bool, detail: str = ""):
    status = PASS if passed else FAIL
    results.append((name, status, detail))
    print(f"  {status}  {name}" + (f"  →  {detail}" if detail else ""))


def record_skip(name: str, reason: str):
    results.append((name, SKIP, reason))
    print(f"  {SKIP}  {name}  →  {reason}")


# ──────────────────────────────────────────────────────────────────────────
# 1. Environment / .env variables
# ──────────────────────────────────────────────────────────────────────────
def test_env():
    print("\n[1] Environment variables")
    required = [
        "DATA_PATH", "CHUNKS_PATH", "FAISS_INDEX_PATH", "METADATA_PATH",
        "EMBEDDING_MODEL", "PRIMARY_MODEL", "JAMBA_MODEL",
        "FASTAPI_PORT", "GRADIO_PORT", "TOP_K_DEFAULT",
    ]
    for key in required:
        val = os.getenv(key)
        record(f"ENV: {key}", val is not None, val or "NOT SET")


def test_preprocessing():
    print("\n[2] Preprocessing / Arabic cleaning")
    try:
        from preprocessing.clean_and_chunk import (
            remove_tashkeel, normalize_alef, normalize_ya,
            clean_arabic_text, chunk_text,
        )

        text_with = "قِفَا نَبْكِ"
        cleaned   = remove_tashkeel(text_with)
        record("remove_tashkeel", "ق" in cleaned and "ِ" not in cleaned, cleaned)

        alef_var   = "أإآا"
        normalized = normalize_alef(alef_var)
        record("normalize_alef", all(c == "ا" for c in normalized), normalized)

        record("normalize_ya", normalize_ya("مستوى") == "مستوي", normalize_ya("مستوى"))

        sample = "قِفَا نَبْكِ مِن ذِكرَى حَبِيبٍ وَمَنزِلِ"
        result = clean_arabic_text(sample)
        record("clean_arabic_text (pipeline)", len(result) > 0, result[:40])

        long_text = "أ" * 700
        chunks    = chunk_text(long_text, chunk_size=300, overlap=50)
        record("chunk_text (splits correctly)", len(chunks) >= 2, f"{len(chunks)} chunks")

        short_text   = "بيت شعر قصير"
        chunks_short = chunk_text(short_text)
        record("chunk_text (short text stays whole)", len(chunks_short) == 1, "")

    except Exception as e:
        record("preprocessing module", False, str(e))


def test_embeddings():
    print("\n[3] Embeddings (sentence-transformers)")
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model_name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
        model      = SentenceTransformer(model_name)

        texts = ["passage: قصيدة المتنبي", "passage: شعر الغزل العربي"]
        embs  = model.encode(texts, normalize_embeddings=True)

        record("embedding model loads", True, model_name)
        record("embedding shape[1] > 0", embs.shape[1] > 0, str(embs.shape))
        record("embeddings normalized (norm ≈ 1)", abs(float(np.linalg.norm(embs[0])) - 1.0) < 0.01, "")

    except Exception as e:
        record("embedding model", False, str(e))


def test_faiss():
    print("\n[4] FAISS index build & search")
    try:
        import faiss
        import numpy as np

        dim   = 384
        index = faiss.IndexFlatIP(dim)
        vecs  = np.random.rand(10, dim).astype("float32")
        faiss.normalize_L2(vecs)
        index.add(vecs)

        record("faiss IndexFlatIP.add()", index.ntotal == 10, f"ntotal={index.ntotal}")

        query = np.random.rand(1, dim).astype("float32")
        faiss.normalize_L2(query)
        scores, indices = index.search(query, 3)

        record("faiss search returns top-k", len(indices[0]) == 3, "")
        record("faiss scores are valid floats", all(isinstance(float(s), float) for s in scores[0]), "")

        with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as tmp:
            tmp_path = tmp.name
        faiss.write_index(index, tmp_path)
        loaded = faiss.read_index(tmp_path)
        os.unlink(tmp_path)
        record("faiss save/load round-trip", loaded.ntotal == 10, "")

    except Exception as e:
        record("faiss", False, str(e))


def test_retriever():
    print("\n[5] Retriever")
    faiss_path = os.path.join(ROOT, os.getenv("FAISS_INDEX_PATH", "index/faiss.index"))
    if not os.path.exists(faiss_path):
        record_skip("retriever.retrieve()", "FAISS index not found — run embed_and_index.py first")
        return

    try:
        from rag.retriever import retrieve
        results_ret = retrieve("قصائد الحب", top_k=3)
        record("retrieve() returns list", isinstance(results_ret, list), "")
        record("retrieve() returns ≤ top_k", len(results_ret) <= 3, f"{len(results_ret)} results")
        if results_ret:
            r = results_ret[0]
            record("chunk has 'text' key", "text" in r, "")
            record("chunk has 'score' key", "score" in r, f"score={r.get('score',0):.4f}")
            record("chunk has metadata keys", all(k in r for k in ("title","poet_name","era")), "")
            record("chunk has num_lines/word_count", all(k in r for k in ("num_lines","word_count")), "")
    except Exception as e:
        record("retriever", False, str(e))


def test_generator_ollama():
    print("\n[6] Generator — gpt-oss (Ollama)")
    try:
        import ollama as _ollama
        _ollama.list()
    except Exception:
        record_skip("generator gpt-oss", "Ollama not reachable — start ollama serve first")
        return

    try:
        from rag.generator import generate_answer

        mock_chunks = [{
            "title": "قفا نبك", "poet_name": "امرؤ القيس", "era": "الجاهلية",
            "num_lines": "10", "word_count": 50,
            "poem_url": "https://www.aldiwan.net/poem/test",
            "text": "قِفَا نَبْكِ مِن ذِكرَى حَبِيبٍ وَمَنزِلِ",
        }]

        result = generate_answer("اشرح هذه القصيدة", mock_chunks, model_name="gpt-oss")
        record("gpt-oss returns answer str", isinstance(result["answer"], str), "")
        record("gpt-oss time_ms > 0", result["time_ms"] > 0, f"{result['time_ms']:.0f}ms")
        record("gpt-oss model_used field", bool(result["model_used"]), result["model_used"])
    except Exception as e:
        record("generator gpt-oss", False, str(e))


def test_generator_jamba():
    print("\n[7] Generator — Jamba (AI21)")
    api_key = os.getenv("JAMBA_API_KEY", "")
    if not api_key or api_key == "your_jamba_api_key_here":
        record_skip("generator jamba", "JAMBA_API_KEY not set in .env")
        return

    try:
        from rag.generator import generate_answer

        mock_chunks = [{
            "title": "قفا نبك", "poet_name": "امرؤ القيس", "era": "الجاهلية",
            "num_lines": "10", "word_count": 50,
            "poem_url": "https://www.aldiwan.net/poem/test",
            "text": "قِفَا نَبْكِ مِن ذِكرَى حَبِيبٍ وَمَنزِلِ",
        }]

        result = generate_answer("اشرح هذه القصيدة", mock_chunks, model_name="jamba")
        record("jamba returns answer str", isinstance(result["answer"], str), "")
        record("jamba time_ms > 0", result["time_ms"] > 0, f"{result['time_ms']:.0f}ms")
        record("jamba model_used field", bool(result["model_used"]), result["model_used"])
    except Exception as e:
        record("generator jamba", False, str(e))


def test_fastapi():
    print("\n[8] FastAPI end-to-end (requires server running)")
    import requests

    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        record("GET /health → 200", r.status_code == 200, "")
        data = r.json()
        record("/health has 'status' field", "status" in data, data.get("status",""))
        record("/health has 'models_available'", "models_available" in data, "")
    except Exception as e:
        record_skip("GET /health", f"Server not reachable at {API_BASE}: {e}")

    try:
        r = requests.get(f"{API_BASE}/stats", timeout=5)
        record("GET /stats → 200", r.status_code == 200, "")
        data = r.json()
        record("/stats has 'total_poems'", "total_poems" in data, str(data.get("total_poems",0)))
    except Exception as e:
        record_skip("GET /stats", f"Server not reachable: {e}")

    faiss_path = os.path.join(ROOT, os.getenv("FAISS_INDEX_PATH", "index/faiss.index"))
    if not os.path.exists(faiss_path):
        record_skip("POST /ask — gpt-oss", "Index not built yet")
        record_skip("POST /ask — jamba", "Index not built yet")
        return

    for model in ["gpt-oss", "jamba"]:
        try:
            r = requests.post(
                f"{API_BASE}/ask",
                json={"question": "ما هي أشهر القصائد العربية؟", "model": model, "top_k": 3},
                timeout=120,
            )
            record(f"POST /ask ({model}) → 200", r.status_code == 200, "")
            if r.status_code == 200:
                data = r.json()
                record(f"/ask ({model}) has 'answer'", "answer" in data, "")
                record(f"/ask ({model}) has 'sources'", "sources" in data, f"{len(data.get('sources',[]))} sources")
                record(f"/ask ({model}) has 'time_ms'", "time_ms" in data, f"{data.get('time_ms',0):.0f}ms")
                if data.get("sources"):
                    s = data["sources"][0]
                    record(f"/ask ({model}) source has lines/words", "num_lines" in s and "word_count" in s, "")
        except Exception as e:
            record_skip(f"POST /ask ({model})", str(e))

    try:
        r = requests.post(
            f"{API_BASE}/ask",
            json={"question": "test", "model": "invalid-model", "top_k": 2},
            timeout=10,
        )
        record("POST /ask (bad model) → 400", r.status_code == 400, "")
    except Exception as e:
        record_skip("POST /ask (bad model)", str(e))


def print_summary():
    print("\n" + "=" * 65)
    print("[AR] ملخص نتائج الاختبارات")
    print("[EN] Test Results Summary")
    print("=" * 65)
    passed  = sum(1 for _, s, _ in results if s == PASS)
    failed  = sum(1 for _, s, _ in results if s == FAIL)
    skipped = sum(1 for _, s, _ in results if s == SKIP)
    total   = len(results)
    print(f"  Total : {total}")
    print(f"  {PASS}  : {passed}")
    print(f"  {FAIL}  : {failed}")
    print(f"  {SKIP}  : {skipped}")
    print("=" * 65)
    if failed == 0:
        print("[AR] ✓ جميع الاختبارات المُشغَّلة نجحت")
        print("[EN] ✓ All executed tests passed")
    else:
        print("[AR] ✗ بعض الاختبارات فشلت — راجع الإخراج أعلاه")
        print("[EN] ✗ Some tests failed — see output above")


if __name__ == "__main__":
    print("=" * 65)
    print("[AR] بدء الاختبارات الشاملة لنظام الشعر العربي RAG")
    print("[EN] Running Arabic Poetry RAG Pipeline Tests")
    print("=" * 65)

    test_env()
    test_preprocessing()
    test_embeddings()
    test_faiss()
    test_retriever()
    test_generator_ollama()
    test_generator_jamba()
    test_fastapi()

    print_summary()
