import os
import json
import re
from dotenv import load_dotenv

try:
    import pyarabic.araby as araby
    PYARABIC_AVAILABLE = True
except ImportError:
    PYARABIC_AVAILABLE = False
    print("[EN] WARNING: pyarabic not installed. Using basic normalization fallback.")

load_dotenv()

DATA_PATH     = os.getenv("DATA_PATH", "data/poems_raw.json")
CHUNKS_PATH   = os.getenv("CHUNKS_PATH", "data/poems_chunks.json")
CHUNK_SIZE    = 300
CHUNK_OVERLAP = 50


def remove_tashkeel(text: str) -> str:
    if PYARABIC_AVAILABLE:
        return araby.strip_tashkeel(text)
    tashkeel_pattern = re.compile(
        r"[\u064B-\u065F\u0610-\u061A\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
    )
    return tashkeel_pattern.sub("", text)


def normalize_alef(text: str) -> str:
    if PYARABIC_AVAILABLE:
        return araby.normalize_alef(text)
    return re.sub(r"[أإآٱ]", "ا", text)


def normalize_ya(text: str) -> str:
    return text.replace("ى", "ي")


def remove_non_arabic_noise(text: str) -> str:
    text  = re.sub(r"&[a-z]+;", " ", text)
    text  = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text  = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.splitlines()]
    text  = "\n".join(line for line in lines if line)
    return text.strip()


def clean_arabic_text(text: str) -> str:
    text = remove_tashkeel(text)
    text = normalize_alef(text)
    text = normalize_ya(text)
    text = remove_non_arabic_noise(text)
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start  = 0
    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunks.append(text[start:])
            break

        newline_pos = text.rfind("\n", start, end)
        if newline_pos > start + overlap:
            end = newline_pos

        chunks.append(text[start:end].strip())
        start = end - overlap

    return [c for c in chunks if c.strip()]


def chunk_poems(poems: list[dict]) -> list[dict]:
    all_chunks = []
    for poem in poems:
        cleaned     = clean_arabic_text(poem["poem_text"])
        text_chunks = chunk_text(cleaned)

        for idx, chunk in enumerate(text_chunks):
            all_chunks.append({
                "chunk_id":     f"{poem['poem_url']}#chunk{idx}",
                "chunk_index":  idx,
                "total_chunks": len(text_chunks),
                "text":         chunk,
                "word_count":   len(chunk.split()),
                "title":        poem.get("title",     "بلا عنوان"),
                "poet_name":    poem.get("poet_name", "مجهول"),
                "era":          poem.get("era",       "غير محدد"),
                "num_lines":    poem.get("num_lines", "غير محدد"),
                "poem_url":     poem.get("poem_url",  ""),
            })

    return all_chunks


def main():
    print("=" * 60)
    print("[AR] بدء تنظيف وتقطيع النصوص العربية")
    print("[EN] Starting Arabic text cleaning and chunking")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"[EN] ERROR: Raw data file not found at {DATA_PATH}")
        print(f"[AR] خطأ: ملف البيانات الخام غير موجود في {DATA_PATH}")
        print("[EN] Please run scraper/scrape_aldiwan.py first.")
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        poems = json.load(f)

    print(f"[AR] تم تحميل {len(poems)} قصيدة من {DATA_PATH}")
    print(f"[EN] Loaded {len(poems)} poems from {DATA_PATH}\n")

    if poems:
        sample_raw   = poems[0]["poem_text"][:200]
        sample_clean = clean_arabic_text(poems[0]["poem_text"])[:200]
        print("[EN] Before/After Cleaning Example:")
        print(f"  BEFORE: {sample_raw}")
        print(f"  AFTER:  {sample_clean}\n")

    print("[AR] جاري التقطيع...")
    print("[EN] Chunking poems...")
    chunks = chunk_poems(poems)

    print(f"[AR] تم إنشاء {len(chunks)} قطعة نصية من {len(poems)} قصيدة")
    print(f"[EN] Created {len(chunks)} chunks from {len(poems)} poems")
    print(f"[EN] Average chunks per poem: {len(chunks)/max(len(poems),1):.1f}")
    print(f"[EN] Average words per chunk: {sum(c['word_count'] for c in chunks)/max(len(chunks),1):.1f}")

    os.makedirs(os.path.dirname(CHUNKS_PATH) if os.path.dirname(CHUNKS_PATH) else ".", exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\n[AR] تم حفظ القطع في: {CHUNKS_PATH}")
    print(f"[EN] Chunks saved to: {CHUNKS_PATH}")

    if chunks:
        print("\n[EN] Sample chunk:")
        c = chunks[0]
        print(f"  chunk_id  : {c['chunk_id']}")
        print(f"  poet      : {c['poet_name']}")
        print(f"  era       : {c['era']}")
        print(f"  word_count: {c['word_count']}")
        print(f"  text      : {c['text'][:120]}...")


if __name__ == "__main__":
    main()