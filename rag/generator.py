import os
import time
from dotenv import load_dotenv

load_dotenv()

PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gpt-oss:20b-cloud")
JAMBA_MODEL   = os.getenv("JAMBA_MODEL",   "jamba-mini")
JAMBA_API_KEY = os.getenv("JAMBA_API_KEY", "")


def build_rag_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    context_lines = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_lines.append(
            f"[{i}] عنوان القصيدة: {chunk.get('title', 'مجهول')}\n"
            f"    الشاعر: {chunk.get('poet_name', 'مجهول')} | العصر: {chunk.get('era', 'غير محدد')} | عدد الأبيات: {chunk.get('num_lines', 'غير محدد')}\n"
            f"    مقطع من القصيدة:\n"
            f"    {chunk.get('text', '')}\n"
            f"    المصدر: {chunk.get('poem_url', '')}"
        )

    context_str = "\n\n".join(context_lines) if context_lines else "لا توجد معلومات متاحة."

    prompt = f"""أنت خبير في الشعر العربي. استخدم المقاطع الشعرية التالية للإجابة على السؤال بدقة وأسلوب أدبي رفيع.

──────────────────────────────────────────────
المقاطع الشعرية المسترجعة:
──────────────────────────────────────────────
{context_str}

──────────────────────────────────────────────
سؤال المستخدم: {query}
──────────────────────────────────────────────

تعليمات:
- أجب باللغة العربية الفصحى فقط.
- استند في إجابتك إلى المقاطع المُقدَّمة أعلاه.
- اذكر اسم الشاعر وعنوان القصيدة عند الاستشهاد.
- إذا كانت المقاطع لا تكفي للإجابة، فقل ذلك بوضوح.
- لا تخترع معلومات غير موجودة في النصوص.

الإجابة:"""

    return prompt


def _call_ollama(prompt: str, model: str) -> str:
    from ollama import chat

    response = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    if isinstance(response, dict):
        return response["message"]["content"]
    return response.message.content


def _call_jamba(prompt: str, model: str, api_key: str) -> str:
    from ai21 import AI21Client
    from ai21.models.chat import ChatMessage

    client   = AI21Client(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[ChatMessage(role="user", content=prompt)],
    )
    return response.choices[0].message.content


def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    model_name: str = "gpt-oss",
) -> dict:
    prompt = build_rag_prompt(query, retrieved_chunks)
    start  = time.time()

    if model_name == "gpt-oss":
        model_id = PRIMARY_MODEL
        try:
            answer = _call_ollama(prompt, model_id)
        except Exception as e:
            answer = f"[خطأ في الاتصال بنموذج gpt-oss] {e}"

    elif model_name == "jamba":
        model_id = JAMBA_MODEL
        if not JAMBA_API_KEY:
            answer = "[خطأ: مفتاح JAMBA_API_KEY غير محدد في ملف .env]"
        else:
            try:
                answer = _call_jamba(prompt, model_id, JAMBA_API_KEY)
            except Exception as e:
                answer = f"[خطأ في الاتصال بنموذج Jamba] {e}"
    else:
        answer   = f"[خطأ: نموذج غير معروف '{model_name}'. استخدم 'gpt-oss' أو 'jamba']"
        model_id = model_name

    elapsed_ms = (time.time() - start) * 1000

    return {
        "answer":     answer,
        "model_used": model_id if model_name in ("gpt-oss", "jamba") else model_name,
        "time_ms":    round(elapsed_ms, 2),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("[AR] اختبار توليد الإجابات")
    print("[EN] Testing answer generation")
    print("=" * 60)

    mock_chunks = [
        {
            "title":     "قفا نبك",
            "poet_name": "امرؤ القيس",
            "era":       "الجاهلية",
            "num_lines": "10",
            "poem_url":  "https://www.aldiwan.net/poem/example",
            "text":      "قِفَا نَبْكِ مِن ذِكرَى حَبِيبٍ وَمَنزِلِ\nبِسِقطِ اللِّوى بَينَ الدَّخولِ فَحَومَلِ",
        }
    ]

    query = "ما هي أشهر قصائد امرئ القيس؟"

    for model in ["gpt-oss", "jamba"]:
        print(f"\n[EN] Testing model: {model}")
        result = generate_answer(query, mock_chunks, model_name=model)
        print(f"  Model used : {result['model_used']}")
        print(f"  Time (ms)  : {result['time_ms']}")
        print(f"  Answer     : {result['answer'][:200]}...")