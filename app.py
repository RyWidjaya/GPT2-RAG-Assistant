# ZeroGPU integration: MUST be imported before any other module or CUDA initialization!
try:
    import spaces
except ImportError:
    spaces = None

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
import uvicorn

# Ensure default Hugging Face model repository
if "HF_MODEL_REPO" not in os.environ:
    os.environ["HF_MODEL_REPO"] = "XAEOZ/gpt2-squad-rag"

from rag_engine import RAGEngine

# Global Engine instance
engine = RAGEngine()

# Wrap generation with ZeroGPU if running on Hugging Face Spaces with ZeroGPU
if spaces is not None:
    try:
        engine.generate = spaces.GPU(duration=60)(engine.generate)
        print("[+] ZeroGPU enabled for engine.generate")
    except Exception as e:
        print(f"[!] Warning wrapping engine.generate with spaces.GPU: {e}")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# FastAPI Application (Acts as Root Server)
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Document Assistant GPT-2 RAG",
    description="Full-featured Document Assistant with GPT-2 + Hybrid RAG",
    version="1.1.0"
)

# Enable CORS for cross-origin and iframe embedding
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Request Models
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


def get_index_html_response() -> HTMLResponse:
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = "<h1>Document Assistant loading...</h1>"
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
# Root UI Routes (Pure White Surface - Zero Gradio Widgets, Zero 404s)
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_root():
    return get_index_html_response()


@app.get("/app", response_class=HTMLResponse)
async def serve_app():
    return get_index_html_response()


@app.get("/index.html", response_class=HTMLResponse)
async def serve_index():
    return get_index_html_response()


@app.get("/style.css")
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


@app.get("/app.js")
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


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# API Endpoints (Identical to Localhost Backend)
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def get_status(request: Request, x_session_id: Optional[str] = Header(None)):
    """Returns system status, active model, and visitor session inventory."""
    sid = x_session_id or request.query_params.get("session_id") or "default"
    return engine.get_status(session_id=sid)


@app.post("/api/model/switch")
async def switch_model(req: SwitchModelRequest):
    """Switch active model between squad_ft, gpt2_pretrained, and qwen2.5_instruct."""
    allowed = ["squad_ft", "gpt2_pretrained", "qwen2.5_instruct"]
    if req.model not in allowed:
        raise HTTPException(status_code=400, detail=f"Model must be one of {allowed}")
    success = engine.load_model(req.model)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to load model {req.model}")
    return {"success": True, "current_model": engine.current_model_name}


@app.post("/api/chat")
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


@app.post("/api/chat/stream")
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
            for token in engine.generate(
                model_name=active_model,
                prompt=prompt,
                max_new_tokens=req.max_new_tokens or 48,
                temperature=req.temperature or 0.0
            ):
                token_count += 1
                yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
                await asyncio.sleep(0.005)
        except Exception as e:
            print(f"[!] Generation error: {e}. Falling back to direct model execution...")
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


@app.post("/api/context/load-local")
async def load_local_folder(req: LocalPathRequest, x_session_id: Optional[str] = Header(None)):
    """Loads all documents from a local folder into visitor's session store."""
    sid = req.session_id or x_session_id or "default"
    session = engine.get_session(sid)
    try:
        status = session.load_directory(req.path)
        return {"success": True, "status": status}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/context/upload")
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


@app.delete("/api/context/document/{doc_id:path}")
async def remove_document(
    doc_id: str,
    session_id: Optional[str] = None,
    x_session_id: Optional[str] = Header(None)
):
    """Removes a single document from visitor's session store."""
    sid = session_id or x_session_id or "default"
    status = engine.remove_document(doc_id, session_id=sid)
    return {"success": True, "status": status}


@app.delete("/api/context/clear")
async def clear_context(
    session_id: Optional[str] = None,
    x_session_id: Optional[str] = Header(None)
):
    """Wipes all documents from visitor's session store."""
    sid = session_id or x_session_id or "default"
    status = engine.clear_documents(session_id=sid)
    return {"success": True, "status": status}


# ---------------------------------------------------------------------------
# Mount Gradio Blocks at /gradio (Satisfies Hugging Face Space Gradio SDK)
# ---------------------------------------------------------------------------
with gr.Blocks(title="Document Assistant Backend") as demo:
    gr.Markdown("# Document Assistant Gradio Service Endpoint")

app = gr.mount_gradio_app(app, demo, path="/gradio")


# ---------------------------------------------------------------------------
# Launch Entrypoint (Binds Uvicorn to 0.0.0.0:7860)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
