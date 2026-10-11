"""
Entrypoint for Hugging Face Gradio Space.
Mounts Gradio SDK compatibility onto the FastAPI application and serves
the custom enterprise frontend UI at '/' and '/static' on port 7860.
"""

import os
import uvicorn
import gradio as gr
from rag_server import app

# Minimal Gradio block to satisfy Hugging Face Gradio SDK detection
with gr.Blocks(title="Document Assistant - GPT-2 RAG") as demo:
    gr.Markdown(
        """
        # Document Assistant — GPT-2 RAG
        The enterprise UI is live at the root: [Open Full UI](/)
        """
    )

# Mount Gradio onto our FastAPI application at '/gradio'
# This leaves the custom enterprise UI at '/' and all API routes untouched!
app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)

