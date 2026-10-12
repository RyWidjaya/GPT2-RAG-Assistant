# Hugging Face Deployment & UI Parity — Issues & Resolutions

This document provides a post-mortem and reference guide detailing the root causes identified during the deployment of the Document Assistant (GPT-2 + Hybrid RAG) on Hugging Face Spaces (`XAEOZ/GPT2-RAG-Assistant`), along with the exact architectural fixes applied to achieve 100% visual and functional parity with the localhost application.

---

## 1. Identified Core Issues

### Issue A: Visual Discrepancy (Default Gradio Blue Widgets vs. Pure-White Design)
* **Symptom**: The initial Hugging Face Space looked completely different from the localhost application (`run_ui.py`). It featured blue primary buttons, standard Gradio accordion containers, default textboxes, and large drop zones instead of the calm, pure-white design.
* **Root Cause**: The cloud deployment was initially assembled using standard Gradio components (`gr.Dropdown`, `gr.Accordion`, `gr.File`, `gr.Textbox`, `gr.Button`). These components impose Gradio’s built-in theme styles, blue accents (`#0969da`), and layout constraints.

### Issue B: The `{"detail":"Not Found"}` (HTTP 404 Error)
* **Symptom**: When trying to serve the custom `index.html` via an iframe (`/app`) or navigating to the Space, the page failed to load or displayed raw JSON: `{"detail":"Not Found"}`.
* **Root Cause**: In Gradio 4.x, when custom routes were attached to `demo.app` prior to calling `demo.launch()`:
  ```python
  # INCORRECT PATTERN
  @demo.app.get("/app")
  def serve(): ...

  demo.launch()
  ```
  During the internal execution of `demo.launch()`, Gradio instantiates a brand-new FastAPI/Starlette application (`networking.start_server -> routes.App.create_app`). This completely discarded all routes previously attached to `demo.app`, causing `/app` and custom API routes to return `404 Not Found`.

### Issue C: ZeroGPU Watchdog Failure (`No @spaces.GPU function detected during startup`)
* **Symptom**: Bypassing Gradio entirely to run a pure `uvicorn.run(app, ...)` caused Hugging Face Spaces to crash on boot with a red container error: `No @spaces.GPU function detected during startup` (HTTP 503).
* **Root Cause**: Hugging Face Spaces with ZeroGPU hardware monkey-patches `gr.Blocks.launch()` via `spaces.zero.gradio.launch`. During container startup, this hook scans the Gradio `blocks.dependencies` event graph to verify that at least one UI event listener is connected to a function explicitly decorated with `@spaces.GPU`. If `demo.launch()` is never called, or if the decorated function is not bound to a Gradio event, the watchdog terminates the container.

### Issue D: Starlette vs. FastAPI Parameter Binding Conflict (`HTTP 500 Error`)
* **Symptom**: Adding raw Starlette routes (`starlette.routing.Route`) at runtime caused `/api/status` and `/api/chat` to fail with HTTP 500 (`TypeError: missing required positional argument`).
* **Root Cause**: Raw Starlette `Route` objects only pass `(request,)` into handler functions. They do not invoke FastAPI’s dependency injection system, failing any endpoint that relies on `x_session_id: Optional[str] = Header(None)` or Pydantic request bodies (`req: ChatRequest`).

---

## 2. Architectural Solution

To satisfy Hugging Face ZeroGPU constraints while delivering 100% visual and functional identity with localhost, the architecture was restructured in [`app.py`](./app.py):

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Hugging Face Space Runtime                      │
├────────────────────────────────────────────────────────────────────────┤
│  1. spaces.zero.gradio.launch Hook                                    │
│     - Inspects demo.dependencies                                       │
│     - Validates @spaces.GPU(duration=60) event listener                │
│                                                                        │
│  2. demo.launch(prevent_thread_lock=True)                             │
│     - Spins up server on 0.0.0.0:7860                                  │
│     - Yields active FastAPI instance: server_app                       │
│                                                                        │
│  3. Priority APIRoute Injection (Index 0 Elevation)                   │
│     - target_app.router.routes.insert(0, new_api_route)                │
│     - Overrides Gradio catch-all routing                               │
│     - Retains full FastAPI parameter & Pydantic dependency injection   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
┌──────────────────────────────────┐        ┌──────────────────────────────────┐
│        Direct UI Delivery        │        │      ZeroGPU RAG Endpoints       │
├──────────────────────────────────┤        ├──────────────────────────────────┤
│ GET /           -> index.html    │        │ GET  /api/status                 │
│ GET /app        -> index.html    │        │ POST /api/model/switch           │
│ GET /index.html -> index.html    │        │ POST /api/chat (Sync)            │
│ GET /style.css  -> style.css     │        │ POST /api/chat/stream (SSE)      │
│ GET /app.js     -> app.js        │        │ POST /api/context/upload         │
│ Mount /static   -> StaticFiles   │        │ DELETE /api/context/clear        │
└──────────────────────────────────┘        └──────────────────────────────────┘
```

---

## 3. Implementation Details

### Step 1: Explicit Top-Level `@spaces.GPU` Function
Defined a clean top-level generation function with mock fallback for local environments:

```python
try:
    import spaces
except ImportError:
    class MockSpaces:
        @staticmethod
        def GPU(func=None, duration=60):
            if func is not None:
                return func
            return lambda f: f
    spaces = MockSpaces()

@spaces.GPU(duration=60)
def gpu_generate(model_name: str = "squad_ft", prompt: str = "", max_new_tokens: int = 48, temperature: float = 0.0):
    yield from engine.generate(model_name=model_name, prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature)
```

### Step 2: Hidden Gradio Event Binding
Attached a hidden event dependency to `demo` to satisfy the ZeroGPU startup validator:

```python
with gr.Blocks(title="Document Assistant — GPT-2 RAG", css=GRADIO_FULLSCREEN_CSS) as demo:
    gr.HTML('<iframe src="/app" style="position:fixed;top:0;left:0;width:100vw;height:100vh;border:none;z-index:999999;background:#f8f9fa;"></iframe>')

    dummy_in = gr.Textbox(visible=False)
    dummy_out = gr.Textbox(visible=False)
    dummy_btn = gr.Button(visible=False)
    dummy_btn.click(fn=gpu_generate, inputs=[dummy_in, dummy_in], outputs=[dummy_out])
```

### Step 3: Priority Route Reordering on Live `server_app`
Launched Gradio without blocking (`prevent_thread_lock=True`), obtained the running FastAPI instance, and inserted standard `APIRoute` instances at index `0`:

```python
def register_custom_routes(target_app):
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
    add_priority_route("/style.css", serve_root_style, methods=["GET"])
    add_priority_route("/app.js", serve_root_app_js, methods=["GET"])

    # API endpoints
    add_priority_route("/api/status", get_status, methods=["GET"])
    add_priority_route("/api/model/switch", switch_model, methods=["POST"])
    add_priority_route("/api/chat", chat, methods=["POST"])
    add_priority_route("/api/chat/stream", chat_stream, methods=["POST"])
    add_priority_route("/api/context/upload", upload_files, methods=["POST"])
    add_priority_route("/api/context/clear", clear_context, methods=["DELETE"])
```

### Step 4: Keep Process Alive
```python
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.queue()
    server_app, local_url, share_url = demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        prevent_thread_lock=True
    )
    register_custom_routes(server_app)
    demo.block_thread()
```

---

## 4. Verification Checkpoint

All endpoints verified live against `https://xaeoz-gpt2-rag-assistant.hf.space`:

| Route | Method | Status | Notes |
| :--- | :--- | :--- | :--- |
| `/` | `GET` | `200 OK` | Serves exact pure-white `index.html` directly |
| `/app` | `GET` | `200 OK` | Direct route for iframe embed |
| `/style.css` | `GET` | `200 OK` | Pure-white design tokens & layout |
| `/app.js` | `GET` | `200 OK` | Model switching, file shelf, streaming SSE |
| `/api/status` | `GET` | `200 OK` | `{"device":"cuda","chunk_count":22,...}` |
| `/api/chat` | `POST` | `200 OK` | Synchronous QA with context metrics |
| `/api/chat/stream` | `POST` | `200 OK` | Real-time typewriter SSE token streaming |
