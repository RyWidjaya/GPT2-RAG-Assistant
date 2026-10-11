# Neural-RAG // GPT-2 Local UI Studio

A front-end interface blending deep **obsidian black** and **electric blue** cyber aesthetics with your custom RAG pipeline and GPT-2 weights (both Fine-Tuned SQuAD Checkpoint and standard HuggingFace Pretrained weights).

---

## ⚡ Quick Launch (Local Hosting)

You can launch the workstation locally in two simple ways:

### Option 1: Terminal Command
```bash
python run_ui.py
```
This automatically starts the high-performance FastAPI server at `http://127.0.0.1:8000` and opens your default browser.

### Option 2: Windows Batch Launcher
Double-click `run_ui.bat` in File Explorer.

---

## 🎨 Interface Features & Design

### 1. Obsidian Black & Electric Blue Aesthetics
- **Dark Mode Palette**: Deep obsidian backdrop (`#040711`), elevated glassmorphic cards (`#0b1224`), and cyber-luxe borders with subtle glowing neon blue reflections.
- **Accents**: Electric cyan (`#38bdf8`), azure blue (`#0284c7`), and glowing pulse status badges.
- **Micro-Interactions**: Hover glows, active radio indicators, token streaming animations, and custom dark scrollbars.

### 2. Context Ingestion & Folder Drop
- **📁 Drop Context Folder Button**:
  - One-click native OS folder selector (`webkitdirectory`) to select any folder on your machine.
- **Drag-and-Drop Dropzone**:
  - Drag and drop entire directories or multiple files (`.txt`, `.md`, `.pdf`, `.json`, `.csv`, `.py`) directly onto the UI.
  - Automatically traverses nested subfolders recursively and tokenizes them into ~150-token chunks with 30-token overlap.
- **Local Disk Loader**:
  - Enter any local folder path (defaults to `./data`) and click **Index** to index directly from disk.
- **Active Documents Inventory**:
  - View loaded documents, per-document chunk counts, and remove or clear documents dynamically.

### 3. Model Weights Selector
- **GPT-2 SQuAD Fine-Tuned (124M)**:
  - Checkpoint loaded from `checkpoints/gpt2_124M_squad_ft.pt`.
  - Optimized for question-answering over retrieved context.
- **GPT-2 Pretrained Base (124M)**:
  - Standard OpenAI weights loaded via `GPT.from_pretrained('gpt2')`.
  - Hot-swappable in real-time from the left sidebar without restarting the server.

### 4. Advanced Retrieval Engine
- **Hybrid Retrieval (RRF)**:
  - Combines **BM25 Sparse Index** (keyword precision) and **FAISS Dense Index** (`all-MiniLM-L6-v2` cosine similarity) using **Reciprocal Rank Fusion** ($k=60$).
- **Dense Vector Search**:
  - Direct FAISS `IndexFlatIP` cosine similarity search.
- **Sparse Keyword Search**:
  - Direct BM25 Okapi search.
- **Interactive Tuning**:
  - Adjustable Top-K chunks slider (1 to 8).
  - Max Context Tokens slider (100 to 800).
  - Max Generation Tokens slider (16 to 128).
  - Temperature slider (0.0 greedy argmax to creative sampling).

### 5. Chat Interface & RAG Inspector
- **Real-Time Streaming**:
  - Autoregressive token generation streams via Server-Sent Events (SSE).
- **Inspect RAG Context & Chunks Drawer**:
  - Each assistant response includes an interactive collapsible accordion revealing:
    - **Retrieved Chunks**: Document source, unique chunk ID, relevance score badge, and chunk text.
    - **Injected Context**: The exact formatted prompt context fed into GPT-2.
- **Performance Telemetry**:
  - Live retrieval latency (e.g. `16.7ms`), total generation time, tokens generated, and hardware telemetry (`CPU` / `CUDA`).
- **Starter Prompt Chips**:
  - Clickable starter questions for instant demonstration (e.g., *"How many championships has Faker won?"*).

