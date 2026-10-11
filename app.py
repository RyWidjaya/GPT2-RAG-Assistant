"""
Entrypoint for Hugging Face Gradio Space.
Mounts Gradio SDK compatibility onto the FastAPI application and serves
the custom enterprise frontend UI at '/' and '/static' on port 7860.
"""

import os
import uvicorn
import gradio as gr
from rag_server import app, engine

# ZeroGPU integration: ZeroGPU requires a function bound to a Gradio component
# to be decorated with @spaces.GPU so its supervisor registers the GPU event listener.
try:
    import spaces
except ImportError:
    spaces = None

if spaces is not None:
    @spaces.GPU(duration=60)
    def ask_rag(question: str) -> str:
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
    def ask_rag(question: str) -> str:
        if not question or not question.strip():
            return "Please provide a question."
        results = engine.retrieve(question, k=3, method="hybrid")
        context = engine.format_context(results)
        prompt = engine.format_prompt("gpt2_squad", context, question)
        tokens = []
        for tok in engine.generate("gpt2_squad", prompt):
            tokens.append(tok)
        return "".join(tokens)

# Gradio interface satisfying Hugging Face ZeroGPU startup scanner
with gr.Blocks(title="Document Assistant - GPT-2 RAG") as demo:
    gr.Markdown(
        """
        # Document Assistant — GPT-2 RAG
        The full custom enterprise UI is available at: [Open Custom UI](/)
        """
    )
    q_in = gr.Textbox(label="Ask a Question", placeholder="e.g. How many championships has Faker won?", lines=2)
    ans_out = gr.Textbox(label="Answer", lines=4)
    btn_submit = gr.Button("Query RAG Assistant", variant="primary")
    btn_submit.click(fn=ask_rag, inputs=q_in, outputs=ans_out)

# Mount Gradio onto our FastAPI application at '/gradio'
# This leaves the custom enterprise UI at '/' and all API routes untouched!
app = gr.mount_gradio_app(app, demo, path="/gradio", ssr_mode=False)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
