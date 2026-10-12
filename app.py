# ZeroGPU integration: MUST be imported before any other module or CUDA initialization!
try:
    import spaces
except ImportError:
    spaces = None

import os
import torch
import gradio as gr
from fastapi import APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from rag_server import app as fastapi_app, engine

# ---------------------------------------------------------------------------
# Backend Query & Generation Logic
# ---------------------------------------------------------------------------

def ask_assistant_gen(question: str, model_choice: str = "squad_ft", k: int = 3):
    if not question or not question.strip():
        yield "Please enter a question in the box above.", ""
        return

    # 1. Retrieve relevant chunks using hybrid BM25 + FAISS retriever
    results = engine.retrieve(question, k=int(k), method="hybrid", session_id="default")
    if not results:
        yield (
            "No matching context found. The knowledge base may be empty. Please upload documents in the 'Knowledge Base' tab.",
            "No context chunks retrieved."
        )
        return

    # Format chunks for inspectable display
    chunks_md_lines = []
    for idx, (chunk, score) in enumerate(results, 1):
        doc_id = chunk.get("doc_id", "Unknown")
        text = chunk.get("text", "").strip()
        chunks_md_lines.append(f"### Chunk {idx} (Score: `{score:.4f}` | Source: `{doc_id}`)\n> {text}\n")
    chunks_display = "\n".join(chunks_md_lines)

    # 2. Build model prompt
    context = engine.format_context(results, max_context_tokens=400)
    prompt = engine.format_prompt(model_choice, context, question)

    # 3. Stream generated answer token by token
    accumulated_tokens = []
    yield "Retrieval complete. Generating answer...", chunks_display

    for tok in engine.generate(model_choice, prompt, max_new_tokens=64, temperature=0.0):
        accumulated_tokens.append(tok)
        yield "".join(accumulated_tokens), chunks_display


# Wrap generation with ZeroGPU decorator if running on Hugging Face Spaces
if spaces is not None:
    ask_fn = spaces.GPU(duration=60)(ask_assistant_gen)
else:
    ask_fn = ask_assistant_gen


# ---------------------------------------------------------------------------
# Knowledge Base Management Logic
# ---------------------------------------------------------------------------

def get_kb_summary():
    status = engine.get_status(session_id="default")
    doc_count = status.get("document_count", 0)
    chunk_count = status.get("chunk_count", 0)
    docs = [d.get("name", "doc") for d in status.get("documents", [])]
    doc_list = ", ".join(docs) if docs else "None"
    return (
        f"**Active Documents**: {doc_count}\n\n"
        f"**Total Indexed Chunks**: {chunk_count}\n\n"
        f"**Indexed Files**: {doc_list}"
    )


def handle_upload(files):
    if not files:
        return "No files selected.", get_kb_summary()

    session = engine.get_session("default")
    file_tuples = []
    for f in files:
        filepath = f.name if hasattr(f, "name") else str(f)
        fname = os.path.basename(filepath)
        try:
            with open(filepath, "rb") as fp:
                file_tuples.append((fname, fp.read()))
        except Exception as e:
            return f"Error reading {fname}: {e}", get_kb_summary()

    try:
        status = session.add_documents_from_bytes(file_tuples)
        msg = f"Successfully ingested {len(file_tuples)} document(s). Total chunks now: {status.get('chunk_count', 0)}"
        return msg, get_kb_summary()
    except Exception as e:
        return f"Ingestion error: {e}", get_kb_summary()


def handle_clear_kb():
    session = engine.get_session("default")
    session.clear_documents()
    return "Knowledge base cleared.", get_kb_summary()


# ---------------------------------------------------------------------------
# Gradio Blocks Web Interface
# ---------------------------------------------------------------------------

theme = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
)

with gr.Blocks(title="Document Assistant - GPT-2 RAG", theme=theme) as demo:
    gr.Markdown(
        """
        # 📑 Document Assistant — GPT-2 RAG Workbench
        Grounded document assistant powered by fine-tuned **GPT-2 124M** (SQuAD) and **hybrid BM25 + FAISS** retrieval.
        * Running natively on **Hugging Face ZeroGPU** (A10G dynamically allocated on demand)
        * [Open Full Enterprise UI](/ui) | [API Status](/api/status)
        """
    )

    with gr.Tabs():
        with gr.Tab("💬 Assistant & Retrieval"):
            with gr.Row():
                with gr.Column(scale=5):
                    q_input = gr.Textbox(
                        label="Question",
                        placeholder="e.g. How many World Championships has Faker won?",
                        lines=2,
                        autofocus=True,
                    )
                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            choices=[
                                ("GPT-2 Fine-Tuned (SQuAD 124M)", "squad_ft"),
                                ("GPT-2 Pretrained (Base 124M)", "gpt2_pretrained"),
                                ("Qwen 2.5 1.5B (Instruct)", "qwen2.5_instruct"),
                            ],
                            value="squad_ft",
                            label="Inference Model",
                            scale=3,
                        )
                        k_slider = gr.Slider(
                            minimum=1,
                            maximum=5,
                            step=1,
                            value=3,
                            label="Top-k Chunks",
                            scale=2,
                        )

                    with gr.Row():
                        btn_submit = gr.Button("Query Assistant", variant="primary", scale=3)
                        btn_clear = gr.Button("Clear", scale=1)

                    examples = gr.Examples(
                        examples=[
                            ["How many World Championships has Faker won?", "squad_ft", 3],
                            ["What is Faker's signature champion?", "squad_ft", 3],
                            ["Which team has Faker played for throughout his entire career?", "squad_ft", 3],
                            ["When was Faker born and what is his real name?", "squad_ft", 3],
                        ],
                        inputs=[q_input, model_dropdown, k_slider],
                        label="Example Questions (Click to load)",
                    )

                with gr.Column(scale=6):
                    ans_output = gr.Textbox(
                        label="Generated Answer",
                        lines=5,
                        interactive=False,
                    )
                    with gr.Accordion("Retrieved Context Chunks (Hybrid BM25 + FAISS)", open=True):
                        chunks_output = gr.Markdown(value="*Retrieved chunks and relevance scores will appear here.*")

            # Events
            btn_submit.click(
                fn=ask_fn,
                inputs=[q_input, model_dropdown, k_slider],
                outputs=[ans_output, chunks_output],
            )
            q_input.submit(
                fn=ask_fn,
                inputs=[q_input, model_dropdown, k_slider],
                outputs=[ans_output, chunks_output],
            )
            btn_clear.click(
                fn=lambda: ("", "", "*Retrieved chunks and relevance scores will appear here.*"),
                outputs=[q_input, ans_output, chunks_output],
            )

        with gr.Tab("📁 Knowledge Base & Ingestion"):
            gr.Markdown(
                """
                ### Manage Document Ingestion
                Upload `.txt`, `.md`, or `.pdf` files to index them into the visitor knowledge base.
                The hybrid retriever will re-chunk (150 tokens) and construct both BM25 sparse and FAISS dense indexes.
                """
            )
            with gr.Row():
                with gr.Column(scale=5):
                    file_input = gr.File(
                        label="Upload Documents (.txt, .md, .pdf)",
                        file_count="multiple",
                        file_types=[".txt", ".md", ".pdf"],
                    )
                    with gr.Row():
                        btn_upload = gr.Button("Ingest & Index Files", variant="primary")
                        btn_clear_kb = gr.Button("Clear Knowledge Base", variant="stop")
                    upload_msg = gr.Textbox(label="Ingestion Log", lines=2, interactive=False)

                with gr.Column(scale=5):
                    kb_status_display = gr.Markdown(value=get_kb_summary())

            btn_upload.click(
                fn=handle_upload,
                inputs=[file_input],
                outputs=[upload_msg, kb_status_display],
            )
            btn_clear_kb.click(
                fn=handle_clear_kb,
                outputs=[upload_msg, kb_status_display],
            )

# ---------------------------------------------------------------------------
# FastAPI API Route Forwarding & Custom UI Mount
# ---------------------------------------------------------------------------

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(STATIC_DIR):
    demo.app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@demo.app.get("/ui", response_class=HTMLResponse)
def serve_custom_ui():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Enterprise UI loading...</h1>", status_code=200)

# Include only /api routes from fastapi_app (avoids root '/' collision with Gradio)
api_router = APIRouter()
for route in fastapi_app.router.routes:
    if hasattr(route, "path") and route.path.startswith("/api"):
        api_router.routes.append(route)
demo.app.include_router(api_router)

# ---------------------------------------------------------------------------
# Server Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=port)
