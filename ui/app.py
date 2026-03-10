import os
import sys
import requests as http_requests
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

FASTAPI_HOST  = os.getenv("FASTAPI_HOST",  "0.0.0.0")
FASTAPI_PORT  = int(os.getenv("FASTAPI_PORT",  "8000"))
GRADIO_PORT   = int(os.getenv("GRADIO_PORT",   "7860"))
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "5"))
APP_VER       = os.getenv("APP_VER", "0.1")

API_BASE = f"http://127.0.0.1:{FASTAPI_PORT}"

import gradio as gr

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Tajawal:wght@300;400;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg-primary:    #080c12;
    --bg-secondary:  #0e1420;
    --bg-card:       #131929;
    --bg-glass:      rgba(255,255,255,0.03);
    --accent:        #c9922a;
    --accent2:       #e8b84b;
    --accent-glow:   rgba(201,146,42,0.15);
    --text-main:     #eef2f7;
    --text-dim:      #6b7a99;
    --border:        rgba(255,255,255,0.07);
    --border-accent: rgba(201,146,42,0.3);
    --success:       #2ecc71;
    --error:         #e74c3c;
    --radius:        14px;
}

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg-primary) !important;
    color: var(--text-main) !important;
    font-family: 'Tajawal', sans-serif !important;
    min-height: 100vh;
}

/* ── Animated background ── */
.gradio-container::before {
    content: '';
    position: fixed;
    top: -40%;
    left: -20%;
    width: 70%;
    height: 70%;
    background: radial-gradient(ellipse, rgba(201,146,42,0.06) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
    animation: pulse-bg 8s ease-in-out infinite alternate;
}

@keyframes pulse-bg {
    from { transform: scale(1) translate(0, 0); opacity: 0.6; }
    to   { transform: scale(1.15) translate(4%, 3%); opacity: 1; }
}

/* ── Hero Header ── */
.prag-hero {
    position: relative;
    padding: 52px 32px 44px;
    text-align: center;
    overflow: hidden;
    border-bottom: 1px solid var(--border);
    margin-bottom: 32px;
}

.prag-hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 80% at 50% -10%, rgba(201,146,42,0.12) 0%, transparent 70%);
    pointer-events: none;
}

.prag-wordmark {
    font-family: 'Playfair Display', serif;
    font-size: clamp(4rem, 10vw, 7.5rem);
    font-weight: 900;
    letter-spacing: -0.03em;
    line-height: 1;
    background: linear-gradient(135deg, #e8b84b 0%, #c9922a 40%, #f5d78e 70%, #c9922a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 40px rgba(201,146,42,0.35));
    margin-bottom: 10px;
    display: block;
    animation: fade-in-down 0.8s ease both;
}

.prag-version {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0em;
    color: #ffffff;
    background: rgba(46, 204, 113, 0.2);
    border: 1px solid rgba(46, 204, 113, 0.4);
    padding: 2px 10px;
    border-radius: 20px;
    vertical-align: middle;
    margin-left: 12px;
    display: inline-block;
    opacity: 1;
}

.prag-tagline {
    font-family: 'Tajawal', sans-serif;
    font-size: 1rem;
    font-weight: 300;
    color: var(--text-dim);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    animation: fade-in-up 0.9s ease 0.2s both;
}

.prag-tagline span {
    color: var(--accent2);
    font-weight: 700;
}

.prag-divider {
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    margin: 18px auto 0;
    animation: fade-in-up 1s ease 0.4s both;
}

@keyframes fade-in-down {
    from { opacity: 0; transform: translateY(-20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fade-in-up {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Input area ── */
.gradio-container textarea,
.gradio-container input[type=text] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-main) !important;
    font-family: 'Tajawal', sans-serif !important;
    font-size: 1.05rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    padding: 14px 18px !important;
}
.gradio-container textarea:focus,
.gradio-container input[type=text]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
    outline: none !important;
}

/* ── Labels ── */
.gradio-container label span,
.gradio-container .label-wrap span {
    color: var(--text-dim) !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
}

/* ── Dropdown ── */
.gradio-container select,
.gradio-container .wrap {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-main) !important;
}

/* ── Slider ── */
.gradio-container input[type=range] {
    accent-color: var(--accent) !important;
}

/* ── Buttons ── */
.gradio-container button.primary {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%) !important;
    border: none !important;
    color: #0a0a0a !important;
    font-family: 'Tajawal', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border-radius: var(--radius) !important;
    padding: 12px 28px !important;
    letter-spacing: 0.04em !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 4px 20px rgba(201,146,42,0.3) !important;
}
.gradio-container button.primary:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}
.gradio-container button.secondary {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-dim) !important;
    font-family: 'Tajawal', sans-serif !important;
    border-radius: var(--radius) !important;
    transition: border-color 0.2s !important;
}
.gradio-container button.secondary:hover {
    border-color: var(--accent) !important;
    color: var(--text-main) !important;
}

/* ── Accordion ── */
.gradio-container details {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}
.gradio-container details summary {
    padding: 14px 18px !important;
    color: var(--accent2) !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.04em !important;
}

/* ── Markdown headers ── */
.gradio-container .prose h3,
.gradio-container .prose h4 {
    color: var(--text-main) !important;
    font-family: 'Tajawal', sans-serif !important;
    font-weight: 700 !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 8px !important;
}

/* ── Checkbox ── */
.gradio-container input[type=checkbox] {
    accent-color: var(--accent) !important;
    width: 16px !important;
    height: 16px !important;
}

/* ── Stats bar ── */
.stats-bar {
    display: flex;
    justify-content: center;
    gap: 32px;
    padding: 14px 24px;
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 24px;
    backdrop-filter: blur(8px);
}
.stat-item {
    text-align: center;
}
.stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    color: var(--accent2);
    font-weight: 600;
}
.stat-label {
    font-size: 0.72rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
"""


def _call_api(question: str, model: str, top_k: int) -> dict:
    try:
        resp = http_requests.post(
            f"{API_BASE}/ask",
            json={"question": question, "model": model, "top_k": top_k},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except http_requests.exceptions.ConnectionError:
        return {"error": f"لا يمكن الاتصال بالخادم على {API_BASE} — تأكد من تشغيل: python api/main.py"}
    except http_requests.exceptions.Timeout:
        return {"error": "انتهت مهلة الطلب (120 ثانية). حاول مجدداً."}
    except Exception as e:
        return {"error": f"خطأ: {e}"}


def _get_stats() -> dict:
    try:
        resp = http_requests.get(f"{API_BASE}/stats", timeout=5)
        return resp.json()
    except Exception:
        return {"total_poems": "—", "total_chunks": "—", "index_size": "—"}


def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return "<p style='color:#6b7a99;font-family:Tajawal,sans-serif'>لا توجد مصادر</p>"
    lines = []
    for i, s in enumerate(sources, 1):
        score_pct = int(s.get("score", 0) * 100)
        lines.append(
            f"<div style='border:1px solid rgba(255,255,255,0.07);border-radius:12px;"
            f"padding:16px;margin:10px 0;background:#0e1420;direction:rtl;"
            f"font-family:Tajawal,sans-serif;transition:border-color 0.2s'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'>"
            f"<b style='color:#e8b84b;font-size:1rem'>[{i}] {s.get('title','—')}</b>"
            f"<span style='background:rgba(201,146,42,0.12);color:#c9922a;border-radius:20px;"
            f"padding:2px 10px;font-size:0.75rem;font-family:JetBrains Mono,monospace'>{score_pct}%</span>"
            f"</div>"
            f"<span style='color:#6b7a99;font-size:0.88rem'>{s.get('poet_name','—')} · {s.get('era','—')}</span><br>"
            f"<span style='color:#6b7a99;font-size:0.8rem'>الأبيات: {s.get('num_lines','—')} · الكلمات: {s.get('word_count','—')}</span><br>"
            f"<a href='{s.get('poem_url','')}' target='_blank' "
            f"style='color:#58a6ff;font-size:0.78rem;text-decoration:none'>{s.get('poem_url','')}</a>"
            f"<p style='margin-top:10px;font-size:0.92rem;color:#c8d0e0;line-height:1.7'>"
            f"{s.get('text','')[:200]}…</p>"
            f"</div>"
        )
    return "".join(lines)


def query_single(question: str, model: str, top_k: int):
    if not question.strip():
        return "<p style='color:#e74c3c;direction:rtl;font-family:Tajawal,sans-serif'>الرجاء إدخال سؤال.</p>", "", ""

    data = _call_api(question, model, int(top_k))

    if "error" in data:
        return f"<div style='color:#e74c3c;direction:rtl;padding:16px;font-family:Tajawal,sans-serif'>{data['error']}</div>", "", ""

    answer_html = (
        f"<div style='direction:rtl;text-align:right;background:#0e1420;"
        f"border:1px solid rgba(201,146,42,0.3);border-radius:14px;padding:24px;"
        f"font-size:1.08rem;line-height:2;font-family:Tajawal,sans-serif;color:#eef2f7;"
        f"box-shadow:0 0 30px rgba(201,146,42,0.07)'>"
        f"{data.get('answer','').replace(chr(10),'<br>')}"
        f"</div>"
    )

    sources_html = _format_sources(data.get("sources", []))

    timing_html = (
        f"<div style='margin-top:12px'>"
        f"<span style='background:#0e1420;border:1px solid rgba(46,204,113,0.3);color:#2ecc71;"
        f"border-radius:20px;padding:4px 14px;font-family:JetBrains Mono,monospace;font-size:0.78rem'>"
        f"⏱ {data.get('time_ms',0):.0f} ms</span>&nbsp;"
        f"<span style='background:#0e1420;border:1px solid rgba(201,146,42,0.2);color:#c9922a;"
        f"border-radius:20px;padding:4px 14px;font-family:JetBrains Mono,monospace;font-size:0.78rem'>"
        f"🤖 {data.get('model_used','')}</span>"
        f"</div>"
    )

    return answer_html, sources_html, timing_html


def query_compare(question: str, top_k: int):
    if not question.strip():
        empty = "<p style='color:#e74c3c;direction:rtl;font-family:Tajawal,sans-serif'>الرجاء إدخال سؤال.</p>"
        return empty, "", "", empty, "", ""

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        fut_gpt = ex.submit(_call_api, question, "gpt-oss", int(top_k))
        fut_jmb = ex.submit(_call_api, question, "jamba",   int(top_k))
        gpt_data = fut_gpt.result()
        jmb_data = fut_jmb.result()

    def _render(data):
        if "error" in data:
            return f"<div style='color:#e74c3c;direction:rtl;padding:16px'>{data['error']}</div>", "", ""
        ans = (
            f"<div style='direction:rtl;text-align:right;background:#0e1420;"
            f"border:1px solid rgba(201,146,42,0.25);border-radius:14px;padding:20px;"
            f"font-size:1rem;line-height:1.95;font-family:Tajawal,sans-serif;color:#eef2f7'>"
            f"{data.get('answer','').replace(chr(10),'<br>')}"
            f"</div>"
        )
        src = _format_sources(data.get("sources", []))
        tim = (
            f"<span style='background:#0e1420;border:1px solid rgba(46,204,113,0.3);color:#2ecc71;"
            f"border-radius:20px;padding:3px 12px;font-family:JetBrains Mono,monospace;font-size:0.76rem'>"
            f"⏱ {data.get('time_ms',0):.0f} ms</span>"
        )
        return ans, src, tim

    g_ans, g_src, g_tim = _render(gpt_data)
    j_ans, j_src, j_tim = _render(jmb_data)
    return g_ans, g_src, g_tim, j_ans, j_src, j_tim


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        css=CUSTOM_CSS,
        title="PRAG — Arabic Poetry RAG",
        theme=gr.themes.Base(primary_hue="orange", neutral_hue="slate"),
    ) as demo:

        gr.HTML(f"""
        <div class="prag-hero">
            <span class="prag-wordmark">PRAG <span class="prag-version">v{APP_VER}</span></span>
            <p class="prag-tagline">
                <span>P</span>oetry &nbsp;·&nbsp;
                <span>R</span>etrieval &nbsp;·&nbsp;
                <span>A</span>ugmented &nbsp;·&nbsp;
                <span>G</span>eneration
            </p>
            <p style='color:#6b7a99;font-size:0.88rem;margin-top:10px;font-family:Tajawal,sans-serif'>
                نظام الشعر العربي الذكي
            </p>
            <div class="prag-divider"></div>
        </div>
        """)

        stats = _get_stats()
        gr.HTML(f"""
        <div class="stats-bar">
            <div class="stat-item">
                <div class="stat-value">{stats.get('total_poems','—')}</div>
                <div class="stat-label">القصائد</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">2</div>
                <div class="stat-label">نماذج الذكاء المتاحه</div>
            </div>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=3):
                question_box = gr.Textbox(
                    label="✍️ سؤالك عن الشعر العربي",
                    placeholder="مثال: ما هي أشهر قصائد المتنبي في المدح؟",
                    lines=3,
                    rtl=True,
                )
            with gr.Column(scale=1):
                model_dd = gr.Dropdown(
                    choices=["gpt-oss", "jamba"],
                    value="gpt-oss",
                    label="🤖 النموذج",
                )
                compare_cb = gr.Checkbox(label="⚖️ مقارنة النموذجين", value=False)

        with gr.Row():
            submit_btn = gr.Button("🔍 ابحث وأجب", variant="primary")
            clear_btn  = gr.Button("🗑️ مسح",        variant="secondary")

        with gr.Group(visible=True) as single_group:
            gr.Markdown("### 💬 الإجابة")
            answer_out = gr.HTML(
                value="<div style='color:#6b7a99;text-align:center;padding:40px;"
                      "font-family:Tajawal,sans-serif;font-size:1rem'>ستظهر الإجابة هنا…</div>"
            )
            timing_out = gr.HTML()
            with gr.Accordion("📖 المصادر الشعرية المسترجعة", open=False):
                sources_out = gr.HTML()

        with gr.Group(visible=False) as compare_group:
            gr.Markdown("### ⚖️ مقارنة النموذجين")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🟠 gpt-oss · Transformer")
                    gpt_ans = gr.HTML()
                    gpt_tim = gr.HTML()
                    with gr.Accordion("📖 مصادر gpt-oss", open=False):
                        gpt_src = gr.HTML()
                with gr.Column():
                    gr.Markdown("#### 🔵 Jamba · Mamba Hybrid")
                    jmb_ans = gr.HTML()
                    jmb_tim = gr.HTML()
                    with gr.Accordion("📖 مصادر Jamba", open=False):
                        jmb_src = gr.HTML()

        def toggle_compare(checked):
            return gr.update(visible=not checked), gr.update(visible=checked)

        compare_cb.change(fn=toggle_compare, inputs=[compare_cb], outputs=[single_group, compare_group])

        def on_submit(q, model, compare):
            top_k = 20  # "Unlimited" default
            if compare:
                g_ans, g_sr, g_ti, j_ans, j_sr, j_ti = query_compare(q, top_k)
                return "<div style='color:#6b7a99;font-family:Tajawal,sans-serif'>جارٍ المقارنة…</div>", "", "", g_ans, g_sr, g_ti, j_ans, j_sr, j_ti
            else:
                ans, src, tim = query_single(q, model, top_k)
                return ans, src, tim, "", "", "", "", "", ""

        submit_btn.click(
            fn=on_submit,
            inputs=[question_box, model_dd, compare_cb],
            outputs=[answer_out, sources_out, timing_out, gpt_ans, gpt_src, gpt_tim, jmb_ans, jmb_src, jmb_tim],
        )

        question_box.submit(
            fn=on_submit,
            inputs=[question_box, model_dd, compare_cb],
            outputs=[answer_out, sources_out, timing_out, gpt_ans, gpt_src, gpt_tim, jmb_ans, jmb_src, jmb_tim],
        )

        clear_btn.click(
            fn=lambda: (
                "",
                "<div style='color:#6b7a99;text-align:center;padding:40px;font-family:Tajawal,sans-serif'>ستظهر الإجابة هنا…</div>",
                "", "", "", "", "", "", "", "",
            ),
            outputs=[question_box, answer_out, sources_out, timing_out, gpt_ans, gpt_src, gpt_tim, jmb_ans, jmb_src, jmb_tim],
        )

    return demo


demo = build_ui()

if __name__ == "__main__":
    print("=" * 60)
    print("[AR] تشغيل واجهة PRAG")
    print("[EN] Launching PRAG — Arabic Poetry RAG UI")
    print(f"[EN] API at {API_BASE}")
    print(f"[EN] UI  at http://localhost:{GRADIO_PORT}")
    print("=" * 60)

    demo.launch(
        server_name="0.0.0.0",
        server_port=GRADIO_PORT,
        share=False,
        show_error=True,
    )