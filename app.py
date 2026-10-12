# ZeroGPU integration: MUST be imported before any other module or CUDA initialization!
try:
    import spaces
except ImportError:
    spaces = None

import os
import torch
import gradio as gr
from rag_server import engine

# ---------------------------------------------------------------------------
# Custom CSS matching static/style.css directly
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

:root {
    --font: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --bg: #f8f9fa;
    --surface: #ffffff;
    --border: #e5e7eb;
    --border-hover: #d1d5db;
    --border-focus: #0969da;
    --text-1: #111827;
    --text-2: #374151;
    --text-3: #5b6574;
    --accent: #0969da;
    --accent-hover: #0757b8;
    --accent-bg: #eff6ff;
    --accent-border: #bfdbfe;
    --radius: 10px;
    --radius-sm: 8px;
}

body, .gradio-container {
    background-color: var(--bg) !important;
    font-family: var(--font) !important;
    color: var(--text-1) !important;
    max-width: 840px !important;
    margin: 0 auto !important;
    padding: 0 16px 32px !important;
}

footer {
    display: none !important;
}

/* Demo Notice Banner */
.demo-banner {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 6px 16px;
    background: var(--accent-bg);
    border: 1px solid var(--accent-border);
    border-radius: var(--radius-sm);
    color: var(--text-2);
    font-size: 0.75rem;
    font-weight: 500;
    margin: 12px 0 10px;
}

/* Header */
.header-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 16px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 12px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}

.header-left {
    display: flex;
    align-items: center;
    gap: 10px;
}

.header-title {
    font-size: 0.9375rem;
    font-weight: 600;
    color: var(--text-1);
}

/* Hero / Empty State */
.hero-box {
    text-align: center;
    padding: 24px 16px 12px;
}

.hero-title {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text-1);
    margin-bottom: 6px;
}

.hero-desc {
    font-size: 0.875rem;
    color: var(--text-2);
    max-width: 580px;
    margin: 0 auto 16px;
    line-height: 1.5;
}

/* Starters row */
.starters-row {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px;
    margin-bottom: 16px;
}

.starter-btn {
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    background: var(--surface) !important;
    color: var(--text-2) !important;
    font-size: 0.8125rem !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
    transition: all 0.12s ease !important;
}

.starter-btn:hover {
    border-color: var(--border-hover) !important;
    color: var(--text-1) !important;
    box-shadow: 0 3px 8px rgba(0,0,0,0.06) !important;
    transform: translateY(-1px) !important;
}

/* Documents Shelf */
.kb-shelf-accordion {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--surface) !important;
    margin-bottom: 12px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
}

.privacy-notice {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: #f1f3f5;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text-2);
    font-size: 0.75rem;
    font-weight: 500;
    margin-bottom: 10px;
}

/* Composer Card */
.composer-wrap {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: 0 3px 10px rgba(0,0,0,0.05) !important;
    padding: 8px 12px !important;
    margin-top: 8px !important;
    transition: border-color 0.15s ease !important;
}

.composer-wrap:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent), 0 4px 12px rgba(0,0,0,0.06) !important;
}

/* Primary Send Button */
.send-btn {
    background-color: var(--accent) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    transition: background-color 0.12s ease !important;
}

.send-btn:hover {
    background-color: var(--accent-hover) !important;
}

.composer-footer {
    text-align: center;
    font-size: 0.72rem;
    color: var(--text-3);
    margin-top: 10px;
}
"""

# ---------------------------------------------------------------------------
# Helpers for Document Shelf & Formatting
# ---------------------------------------------------------------------------

def get_kb_badge():
    status = engine.get_status(session_id="default")
    doc_count = status.get("document_count", 0)
    chunk_count = status.get("chunk_count", 0)
    docs = [d.get("name", "doc") for d in status.get("documents", [])]
    return f"**Status:** {doc_count} document(s), {chunk_count} chunks · **Active Files:** {', '.join(docs) if docs else 'None'}"


def format_sources_html(results):
    if not results:
        return ""

    max_score = max((score for _, score in results), default=1.0)
    if max_score <= 0:
        max_score = 1.0

    cards_html = []
    for idx, (chunk, score) in enumerate(results, 1):
        doc_id = chunk.get("doc_id", "Document")
        chunk_id = chunk.get("chunk_id", f"chunk{idx}")
        text = chunk.get("text", "").strip()
        pct = min(100, max(15, int((score / max_score) * 100)))

        card = f"""
        <div style="border: 1px solid #e5e7eb; border-radius: 8px; background: #ffffff; padding: 10px 14px; margin-bottom: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px; border-radius: 50%; background: #eff6ff; color: #0969da; font-size: 11px; font-weight: 600;">{idx}</span>
                    <span style="font-weight: 600; font-size: 13px; color: #111827;">{doc_id}</span>
                    <span style="font-family: monospace; font-size: 11px; color: #5b6574;">{chunk_id}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <div style="width: 44px; height: 4px; border-radius: 2px; background: #e5e7eb; overflow: hidden;">
                        <div style="width: {pct}%; height: 100%; background: #0969da; border-radius: 2px;"></div>
                    </div>
                    <span style="font-family: monospace; font-size: 11px; color: #5b6574;">{score:.4f}</span>
                </div>
            </div>
            <div style="font-size: 13px; color: #374151; line-height: 1.55; background: #f8f9fa; padding: 8px 12px; border-radius: 6px; border-left: 3px solid #0969da;">
                {text}
            </div>
        </div>
        """
        cards_html.append(card)

    return f"""
    <div style="margin-top: 14px;">
        <div style="display: flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 600; color: #5b6574; margin-bottom: 8px;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
            <span>Retrieved Context ({len(results)} chunks with relevance score bars)</span>
        </div>
        {''.join(cards_html)}
    </div>
    """

# ---------------------------------------------------------------------------
# Inference & Streaming Generator Logic
# ---------------------------------------------------------------------------

def ask_assistant_gen(question: str, model_choice: str = "squad_ft", search_method: str = "hybrid", k: int = 3):
    if not question or not question.strip():
        yield "Please enter a question in the box below.", ""
        return

    # 1. Retrieve
    results = engine.retrieve(question, k=int(k), method=search_method, session_id="default")
    if not results:
        yield (
            "No matching context found. Please ensure documents are indexed in the shelf above.",
            "<p style='color:#5b6574;font-size:13px;'>No context chunks retrieved.</p>"
        )
        return

    sources_html = format_sources_html(results)
    context = engine.format_context(results, max_context_tokens=400)
    prompt = engine.format_prompt(model_choice, context, question)

    # 2. Token-by-token streaming
    accumulated_tokens = []
    yield "Thinking & generating answer...", sources_html

    for tok in engine.generate(model_choice, prompt, max_new_tokens=64, temperature=0.0):
        accumulated_tokens.append(tok)
        yield "".join(accumulated_tokens), sources_html


if spaces is not None:
    ask_fn = spaces.GPU(duration=60)(ask_assistant_gen)
else:
    ask_fn = ask_assistant_gen


# ---------------------------------------------------------------------------
# File Management Handlers
# ---------------------------------------------------------------------------

def handle_upload(files):
    if not files:
        return "No files selected.", get_kb_badge()

    session = engine.get_session("default")
    file_tuples = []
    for f in files:
        filepath = f.name if hasattr(f, "name") else str(f)
        fname = os.path.basename(filepath)
        try:
            with open(filepath, "rb") as fp:
                file_tuples.append((fname, fp.read()))
        except Exception as e:
            return f"Error reading {fname}: {e}", get_kb_badge()

    try:
        status = session.add_documents_from_bytes(file_tuples)
        msg = f"✓ Ingested {len(file_tuples)} file(s). Total chunks: {status.get('chunk_count', 0)}"
        return msg, get_kb_badge()
    except Exception as e:
        return f"Ingestion error: {e}", get_kb_badge()


def handle_clear():
    session = engine.get_session("default")
    session.clear_documents()
    return None, "Knowledge base cleared.", get_kb_badge()


# ---------------------------------------------------------------------------
# Gradio Blocks Web Application (Single Centered Column Layout)
# ---------------------------------------------------------------------------

with gr.Blocks(title="Document Assistant — GPT-2 RAG", css=CUSTOM_CSS) as demo:
    # 1. Demo Notice Banner
    gr.HTML(
        """
        <div class="demo-banner">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#0969da" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
            <span>Demo model (GPT-2, 124M). Answers may be unreliable.</span>
        </div>
        """
    )

    # 2. Header Bar
    gr.HTML(
        """
        <div class="header-bar">
            <div class="header-left">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#0969da" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
                <span class="header-title">Document Assistant</span>
            </div>
            <div style="font-size: 12px; color: #5b6574; font-weight: 500;">
                ZeroGPU · 24/7 Cloud
            </div>
        </div>
        """
    )

    # 3. Model Controls Bar
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=[
                ("GPT-2 fine-tuned (SQuAD 124M)", "squad_ft"),
                ("GPT-2 base (OpenAI 124M)", "gpt2_pretrained"),
                ("Qwen 2.5 1.5B (Instruct)", "qwen2.5_instruct"),
            ],
            value="squad_ft",
            label="Active Model",
            scale=3,
        )
        method_dropdown = gr.Dropdown(
            choices=[
                ("Hybrid (BM25 + FAISS)", "hybrid"),
                ("Semantic (FAISS Dense)", "dense"),
                ("Keyword (BM25 Sparse)", "sparse"),
            ],
            value="hybrid",
            label="Search Method",
            scale=2,
        )
        k_slider = gr.Slider(
            minimum=1,
            maximum=5,
            step=1,
            value=3,
            label="Top-k",
            scale=1,
        )

    # 4. Collapsible Document Shelf
    with gr.Accordion(label="📁 Document Ingestion Shelf", open=False, elem_classes=["kb-shelf-accordion"]):
        gr.HTML(
            """
            <div class="privacy-notice">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#5b6574" stroke-width="2"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
                <span>Don't upload anything private. Files are deleted after your session or 30 minutes of inactivity.</span>
            </div>
            """
        )
        kb_status_badge = gr.Markdown(value=get_kb_badge())
        file_input = gr.File(
            label="Drop files here (.txt, .md, .pdf | max 5 MB)",
            file_count="multiple",
            file_types=[".txt", ".md", ".pdf"],
        )
        with gr.Row():
            btn_upload = gr.Button("Ingest & Index Files", variant="primary", scale=2)
            btn_clear_kb = gr.Button("Clear All Documents", scale=1)
        upload_log = gr.Textbox(label="Ingestion Log", lines=1, interactive=False)

    # 5. Hero / Empty State & Clickable Starters
    gr.HTML(
        """
        <div class="hero-box">
            <h1 class="hero-title">Ask questions about your documents.</h1>
            <p class="hero-desc">This assistant uses a custom-built GPT-2 model with hybrid retrieval (BM25 + FAISS) to answer questions grounded in your uploaded context.</p>
        </div>
        """
    )

    with gr.Row(elem_classes=["starters-row"]):
        btn_s1 = gr.Button("How many championships has Faker won?", elem_classes=["starter-btn"])
        btn_s2 = gr.Button("What team did Faker debut with?", elem_classes=["starter-btn"])
        btn_s3 = gr.Button("What playstyle is Faker known for?", elem_classes=["starter-btn"])
        btn_s4 = gr.Button("What is photosynthesis?", elem_classes=["starter-btn"])

    # 6. Assistant Response & Signature Source Cards
    ans_display = gr.Textbox(
        label="Assistant Response",
        lines=4,
        interactive=False,
        placeholder="Answers grounded in retrieved context will appear here...",
    )
    sources_display = gr.HTML(value="")

    # 7. Floating Composer
    with gr.Row(elem_classes=["composer-wrap"]):
        q_box = gr.Textbox(
            placeholder="Ask a question...",
            lines=1,
            max_lines=4,
            show_label=False,
            scale=8,
            container=False,
            autofocus=True,
        )
        btn_send = gr.Button("Send", variant="primary", scale=1, elem_classes=["send-btn"])

    gr.HTML(
        """
        <p class="composer-footer">
            Demo model (GPT-2, 124M) with hybrid retrieval · Running on ZeroGPU (Nvidia A10G) · Answers may be unreliable.
        </p>
        """
    )

    # -----------------------------------------------------------------------
    # Wire Events
    # -----------------------------------------------------------------------

    # Model selector dynamic hot-swap
    model_dropdown.change(
        fn=lambda m: engine.load_model(m),
        inputs=[model_dropdown],
        outputs=None,
    )

    # Send button & Enter key
    btn_send.click(
        fn=ask_fn,
        inputs=[q_box, model_dropdown, method_dropdown, k_slider],
        outputs=[ans_display, sources_display],
    )
    q_box.submit(
        fn=ask_fn,
        inputs=[q_box, model_dropdown, method_dropdown, k_slider],
        outputs=[ans_display, sources_display],
    )

    # Starters (populate question box and stream answer immediately)
    def run_starter_1(m, meth, k):
        q = "How many championships has Faker won?"
        for ans, src in ask_fn(q, m, meth, k):
            yield q, ans, src

    def run_starter_2(m, meth, k):
        q = "What team did Faker debut with?"
        for ans, src in ask_fn(q, m, meth, k):
            yield q, ans, src

    def run_starter_3(m, meth, k):
        q = "What playstyle is Faker known for?"
        for ans, src in ask_fn(q, m, meth, k):
            yield q, ans, src

    def run_starter_4(m, meth, k):
        q = "What is photosynthesis?"
        for ans, src in ask_fn(q, m, meth, k):
            yield q, ans, src

    btn_s1.click(
        fn=run_starter_1,
        inputs=[model_dropdown, method_dropdown, k_slider],
        outputs=[q_box, ans_display, sources_display],
    )
    btn_s2.click(
        fn=run_starter_2,
        inputs=[model_dropdown, method_dropdown, k_slider],
        outputs=[q_box, ans_display, sources_display],
    )
    btn_s3.click(
        fn=run_starter_3,
        inputs=[model_dropdown, method_dropdown, k_slider],
        outputs=[q_box, ans_display, sources_display],
    )
    btn_s4.click(
        fn=run_starter_4,
        inputs=[model_dropdown, method_dropdown, k_slider],
        outputs=[q_box, ans_display, sources_display],
    )

    # File shelf events: auto-upload on drop/selection and manual button click
    file_input.upload(
        fn=handle_upload,
        inputs=file_input,
        outputs=[upload_log, kb_status_badge],
    )
    btn_upload.click(
        fn=handle_upload,
        inputs=file_input,
        outputs=[upload_log, kb_status_badge],
    )
    btn_clear_kb.click(
        fn=handle_clear,
        outputs=[file_input, upload_log, kb_status_badge],
    )

# ---------------------------------------------------------------------------
# Server Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=port)
