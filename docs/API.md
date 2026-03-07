# API Documentation — Arabic Poetry RAG

Base URL: `http://localhost:8000`

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Endpoints

### `POST /ask`

Answer a question about Arabic poetry using the RAG pipeline.

**Request body (JSON)**

| Field      | Type    | Required | Default   | Description                              |
| ---------- | ------- | -------- | --------- | ---------------------------------------- |
| `question` | string  | ✅       | —         | Arabic question (min 3 chars)            |
| `model`    | string  | —        | `gpt-oss` | LLM to use: `"gpt-oss"` or `"jamba"`     |
| `top_k`    | integer | —        | `5`       | Number of poem chunks to retrieve (1–10) |

**Response (200 OK)**

```json
{
  "answer": "يُعدّ المتنبي من أعظم شعراء العربية...",
  "sources": [
    {
      "title": "على قدر أهل العزم",
      "poet_name": "المتنبي",
      "era": "العصر العباسي",
      "poem_url": "https://www.aldiwan.net/poem/almutanabbi/...",
      "score": 0.91,
      "text": "عَلى قَدرِ أَهلِ العَزمِ تَأتي العَزائِمُ..."
    }
  ],
  "model_used": "gpt-oss:20b-cloud",
  "time_ms": 3241.5
}
```

**Error responses**

| Status | Meaning                                   |
| ------ | ----------------------------------------- |
| 400    | Invalid `model` value                     |
| 422    | Request validation error (missing fields) |
| 503    | FAISS index not found (run embed step)    |
| 500    | Internal retrieval or generation error    |

**cURL example**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "ما هي أشهر قصائد المتنبي؟", "model": "gpt-oss", "top_k": 5}'
```

**Python requests example**

```python
import requests

resp = requests.post(
    "http://localhost:8000/ask",
    json={
        "question": "ما هي أشهر قصائد المتنبي؟",
        "model": "gpt-oss",
        "top_k": 5,
    }
)
data = resp.json()
print(data["answer"])
for src in data["sources"]:
    print(f"  {src['title']} — {src['poet_name']} ({src['score']:.2f})")
```

---

### `GET /health`

Check service health and model availability.

**Response (200 OK)**

```json
{
  "status": "ok",
  "models_available": ["gpt-oss", "jamba"],
  "index_loaded": true,
  "timestamp": "2025-06-01T12:00:00Z"
}
```

**cURL example**

```bash
curl http://localhost:8000/health
```

**Python requests example**

```python
import requests
print(requests.get("http://localhost:8000/health").json())
```

---

### `GET /stats`

Return statistics about the loaded data and FAISS index.

**Response (200 OK)**

```json
{
  "total_poems": 200,
  "total_chunks": 847,
  "index_size": 847
}
```

**cURL example**

```bash
curl http://localhost:8000/stats
```

**Python requests example**

```python
import requests
print(requests.get("http://localhost:8000/stats").json())
```

---

## Error Codes Table

| HTTP Code | Error Type            | Detail                                           |
| --------- | --------------------- | ------------------------------------------------ |
| 400       | Bad Request           | `model` must be `"gpt-oss"` or `"jamba"`         |
| 422       | Unprocessable Entity  | Missing or invalid request body fields           |
| 500       | Internal Server Error | Retrieval or LLM generation failure              |
| 503       | Service Unavailable   | FAISS index not built — run `embed_and_index.py` |

---

## Complete Python Client Example

```python
import requests

BASE = "http://localhost:8000"

def ask_poetry(question: str, model: str = "gpt-oss", top_k: int = 5) -> dict:
    resp = requests.post(
        f"{BASE}/ask",
        json={"question": question, "model": model, "top_k": top_k},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()

# Single model
result = ask_poetry("ما معنى قصيدة قفا نبك لامرئ القيس؟")
print("Answer:", result["answer"])

# Compare models
gpt = ask_poetry("من هو أمير الشعراء؟", model="gpt-oss")
jmb = ask_poetry("من هو أمير الشعراء؟", model="jamba")
print("GPT-OSS:", gpt["answer"][:200])
print("Jamba  :", jmb["answer"][:200])
```
