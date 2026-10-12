# ZeroGPU integration: MUST be imported before any other module or CUDA initialization!
try:
    import spaces
except ImportError:
    spaces = None

import os
import gradio as gr
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from rag_server import app as fastapi_app, engine

# Query function decorated with @spaces.GPU to satisfy Hugging Face ZeroGPU runtime
if spaces is not None:
    @spaces.GPU(duration=60)
    def ask_assistant(question: str) -> str:
        if not question or not question.strip():
            return "Please provide a question."
        results = engine.retrieve(question, k=3, method="hybrid")
        context = engine.format_context(results)
        prompt = engine.format_prompt("gpt2_squad", context, question)
        tokens = []
        for tok in engine.generate("gpt2_squad", prompt):
            tokens.append(tok)
        return "".join(tokens)
else:
    def ask_assistant(question: str) -> str:
        if not question or not question.strip():
            return "Please provide a question."
        results = engine.retrieve(question, k=3, method="hybrid")
        context = engine.format_context(results)
        prompt = engine.format_prompt("gpt2_squad", context, question)
        tokens = []
        for tok in engine.generate("gpt2_squad", prompt):
            tokens.append(tok)
        return "".join(tokens)

# Gradio Interface running natively on ZeroGPU
with gr.Blocks(title="Document Assistant - GPT-2 RAG") as demo:
    gr.Markdown("# 📑 Document Assistant — GPT-2 RAG")
    gr.Markdown(
        """
        Grounded document assistant powered by fine-tuned GPT-2 124M and hybrid BM25 + FAISS retrieval.
        * **Full Enterprise UI**: [Open Standalone UI](/ui)
        * Or ask directly using the ZeroGPU interface below:
        """
    )
    with gr.Row():
        with gr.Column(scale=2):
            q_in = gr.Textbox(
                label="Question",
                placeholder="e.g. How many championships has Faker won?",
                lines=2
            )
            btn_submit = gr.Button("Query Knowledge Base", variant="primary")
        with gr.Column(scale=3):
            ans_out = gr.Textbox(label="Generated Answer", lines=5)

    btn_submit.click(fn=ask_assistant, inputs=q_in, outputs=ans_out)

# Mount all custom FastAPI routes onto the Gradio underlying server
demo.app.include_router(fastapi_app.router)

# Mount static files and the custom enterprise UI at /ui
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=port)
