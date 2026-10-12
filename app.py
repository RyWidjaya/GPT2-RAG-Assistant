# ZeroGPU integration: MUST be imported before any other module or CUDA initialization!
try:
    import spaces
except ImportError:
    class MockSpaces:
        @staticmethod
        def GPU(func=None, duration=60):
            if func is not None:
                return func
            def decorator(f):
                return f
            return decorator
    spaces = MockSpaces()

import os
import sys
import json
import time
import asyncio
from typing import List, Optional

import torch
import gradio as gr
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure default Hugging Face model repository
if "HF_MODEL_REPO" not in os.environ:
    os.environ["HF_MODEL_REPO"] = "XAEOZ/gpt2-squad-rag"

from rag_engine import RAGEngine

# Global Engine instance
engine = RAGEngine()

# Explicit top-level @spaces.GPU function for ZeroGPU static detection & worker lifecycle
@spaces.GPU(duration=60)
def gpu_generate(model_name: str = "squad_ft", prompt: str = "", max_new_tokens: int = 48, temperature: float = 0.0):
    yield from engine.generate(model_name=model_name, prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Pydantic Request Models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    method: Optional[str] = "hybrid"
    k: Optional[int] = 3
    max_context_tokens: Optional[int] = 400
    max_new_tokens: Optional[int] = 48
    temperature: Optional[float] = 0.0
    model: Optional[str] = None
    session_id: Optional[str] = None


class LocalPathRequest(BaseModel):
    path: str
    session_id: Optional[str] = None


class SwitchModelRequest(BaseModel):
    model: str


# ---------------------------------------------------------------------------
# HTML Document Serving
# ---------------------------------------------------------------------------

def get_index_html_content() -> str:
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Document Assistant loading...</h1>"


def get_index_html_response() -> HTMLResponse:
    content = get_index_html_content()
    return HTMLResponse(
        content=content,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Frame-Options": "ALLOWALL",
            "Content-Security-Policy": "frame-ancestors *",
        }
    )


# ---------------------------------------------------------------------------
# API Route Handlers
# ---------------------------------------------------------------------------

async def serve_app():
    return get_index_html_response()


async def serve_root_style():
    css_file = os.path.join(STATIC_DIR, "style.css")
    if os.path.exists(css_file):
        with open(css_file, "r", encoding="utf-8") as f:
            return HTMLResponse(
                content=f.read(),
                media_type="text/css",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
            )
    raise HTTPException(status_code=404, detail="style.css not found")


async def serve_root_app_js():
    js_file = os.path.join(STATIC_DIR, "app.js")
    if os.path.exists(js_file):
        with open(js_file, "r", encoding="utf-8") as f:
            return HTMLResponse(
                content=f.read(),
                media_type="application/javascript",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
            )
    raise HTTPException(status_code=404, detail="app.js not found")


async def get_status(request: Request, x_session_id: Optional[str] = Header(None)):
    """Returns system status, active model, and visitor session inventory."""
    sid = x_session_id or request.query_params.get("session_id") or "default"
    return engine.get_status(session_id=sid)


async def switch_model(req: SwitchModelRequest):
    """Switch active model between squad_ft, gpt2_pretrained, and qwen2.5_instruct."""
    allowed = ["squad_ft", "gpt2_pretrained", "qwen2.5_instruct"]
    if req.model not in allowed:
        raise HTTPException(status_code=400, detail=f"Model must be one of {allowed}")
    success = engine.load_model(req.model)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to load model {req.model}")
    return {"success": True, "current_model": engine.current_model_name}


async def chat(req: ChatRequest, x_session_id: Optional[str] = Header(None)):
    """Synchronous chat response with full context and chunk metrics."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    sid = req.session_id or x_session_id or "default"
    active_model = req.model or engine.current_model_name

    response = engine.answer_question(
        question=req.question,
        k=req.k or 3,
        method=req.method or "hybrid",
        max_context_tokens=req.max_context_tokens or 400,
        max_new_tokens=req.max_new_tokens or 48,
        temperature=req.temperature or 0.0,
        model=active_model,
        session_id=sid,
    )
    return response


async def chat_stream(req: ChatRequest, x_session_id: Optional[str] = Header(None)):
    """Server-Sent Events (SSE) streaming for real-time typewriter generation."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    sid = req.session_id or x_session_id or "default"
    active_model = req.model or engine.current_model_name

    if active_model not in engine.models:
        engine.load_model(active_model)

    async def event_generator():
        t0 = time.time()
        # Retrieve chunks from visitor session
        results = engine.retrieve(req.question, k=req.k or 3, method=req.method or "hybrid", session_id=sid)
        retrieval_ms = round((time.time() - t0) * 1000, 2)
        context = engine.format_context(results, req.max_context_tokens or 400)
        prompt = engine.format_prompt(active_model, context, req.question)

        clean_chunks = [
            {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "text": c["text"],
                "score": round(score, 4)
            }
            for c, score in results
        ]

        # Send initial metadata event
        meta_payload = {
            "retrieved_chunks": clean_chunks,
            "context_used": context,
            "retrieval_ms": retrieval_ms,
            "model": active_model,
            "method": req.method or "hybrid",
            "session_id": sid
        }
        yield f"event: metadata\ndata: {json.dumps(meta_payload)}\n\n"

        # Stream generated tokens (accelerated via ZeroGPU when on HF Space)
        t_gen_start = time.time()
        token_count = 0
        try:
            for token in gpu_generate(
                model_name=active_model,
                prompt=prompt,
                max_new_tokens=req.max_new_tokens or 48,
                temperature=req.temperature or 0.0
            ):
                token_count += 1
                yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
                await asyncio.sleep(0.005)
        except Exception as e:
            print(f"[!] ZeroGPU execution error: {e}. Falling back to standard CPU execution...")
            for token in engine.generate(
                model_name=active_model,
                prompt=prompt,
                max_new_tokens=req.max_new_tokens or 48,
                temperature=req.temperature or 0.0
            ):
                token_count += 1
                yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
                await asyncio.sleep(0.005)

        total_ms = round((time.time() - t0) * 1000, 2)
        done_payload = {
            "token_count": token_count,
            "total_ms": total_ms,
            "generation_ms": round((time.time() - t_gen_start) * 1000, 2)
        }
        yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


async def load_local_folder(req: LocalPathRequest, x_session_id: Optional[str] = Header(None)):
    """Loads all documents from a local folder into visitor's session store."""
    sid = req.session_id or x_session_id or "default"
    session = engine.get_session(sid)
    try:
        status = session.load_directory(req.path)
        return {"success": True, "status": status}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def upload_files(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    x_session_id: Optional[str] = Header(None)
):
    """Handles session document upload (max 5 MB per file, max 5 files, .txt/.md/.pdf only)."""
    sid = session_id or x_session_id or "default"
    session = engine.get_session(sid)

    file_tuples = []
    for f in files:
        fname = f.filename
        content = await f.read()
        file_tuples.append((fname, content))

    try:
        status = session.add_documents_from_bytes(file_tuples)
        return {"success": True, "status": status}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error processing files: {str(e)}")


async def remove_document(
    doc_id: str,
    session_id: Optional[str] = None,
    x_session_id: Optional[str] = Header(None)
):
    """Removes a single document from visitor's session store."""
    sid = session_id or x_session_id or "default"
    status = engine.remove_document(doc_id, session_id=sid)
    return {"success": True, "status": status}


async def clear_context(
    session_id: Optional[str] = None,
    x_session_id: Optional[str] = Header(None)
):
    """Wipes all documents from visitor's session store."""
    sid = session_id or x_session_id or "default"
    status = engine.clear_documents(session_id=sid)
    return {"success": True, "status": status}


def register_custom_routes(target_app):
    """Registers identical application routes and APIs onto a FastAPI instance with full dependency injection."""
    try:
        target_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    except Exception:
        pass

    def add_priority_route(path, endpoint, methods, response_class=None):
        if response_class:
            target_app.add_api_route(path, endpoint, methods=methods, response_class=response_class)
        else:
            target_app.add_api_route(path, endpoint, methods=methods)
        if hasattr(target_app, "router") and hasattr(target_app.router, "routes") and target_app.router.routes:
            new_r = target_app.router.routes.pop()
            target_app.router.routes.insert(0, new_r)

    # UI routes
    add_priority_route("/", serve_app, methods=["GET"], response_class=HTMLResponse)
    add_priority_route("/app", serve_app, methods=["GET"], response_class=HTMLResponse)
    add_priority_route("/index.html", serve_app, methods=["GET"], response_class=HTMLResponse)
    add_priority_route("/style.css", serve_root_style, methods=["GET"])
    add_priority_route("/app.js", serve_root_app_js, methods=["GET"])

    # API endpoints
    add_priority_route("/api/status", get_status, methods=["GET"])
    add_priority_route("/api/model/switch", switch_model, methods=["POST"])
    add_priority_route("/api/chat", chat, methods=["POST"])
    add_priority_route("/api/chat/stream", chat_stream, methods=["POST"])
    add_priority_route("/api/context/load-local", load_local_folder, methods=["POST"])
    add_priority_route("/api/context/upload", upload_files, methods=["POST"])
    add_priority_route("/api/context/document/{doc_id:path}", remove_document, methods=["DELETE"])
    add_priority_route("/api/context/clear", clear_context, methods=["DELETE"])

    try:
        target_app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Gradio Blocks Container (Renders Pure-White Full-Screen Surface)
# ---------------------------------------------------------------------------

GRADIO_FULLSCREEN_CSS = """
html, body {
    margin: 0 !important;
    padding: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    overflow: hidden !important;
    background: #f8f9fa !important;
}
.gradio-container {
    max-width: 100vw !important;
    width: 100vw !important;
    height: 100vh !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    background: #f8f9fa !important;
}
footer, .gradio-container > footer {
    display: none !important;
}
"""

with gr.Blocks(title="Document Assistant — GPT-2 RAG", css=GRADIO_FULLSCREEN_CSS) as demo:
    # Render the full-screen pure-white application iframe pointing to /app
    gr.HTML(
        """
        <iframe
            src="/app"
            style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; border: none; margin: 0; padding: 0; z-index: 999999; background: #f8f9fa;"
            title="Document Assistant"
            allow="clipboard-write"
        ></iframe>
        """
    )

    # Hidden element wired to @spaces.GPU function so ZeroGPU validation discovers it
    dummy_in = gr.Textbox(visible=False)
    dummy_out = gr.Textbox(visible=False)
    dummy_btn = gr.Button(visible=False)
    dummy_btn.click(fn=gpu_generate, inputs=[dummy_in, dummy_in], outputs=[dummy_out])

# Register routes on demo.app ahead of time
register_custom_routes(demo.app)


# ---------------------------------------------------------------------------
# Launch Entrypoint (ZeroGPU Native Gradio Launcher)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.queue()
    server_app, local_url, share_url = demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        prevent_thread_lock=True
    )
    # Re-register routes on the running server_app returned by demo.launch()
    register_custom_routes(server_app)
    print(f"[+] Document Assistant LIVE at {local_url} with all API routes active!")
    demo.block_thread()
