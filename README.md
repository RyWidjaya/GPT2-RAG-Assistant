---
title: Document Assistant - GPT-2 RAG
emoji: 📑
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Document Assistant — GPT-2 RAG Workbench

> A grounded document assistant built for internal knowledge retrieval. Features a custom GPT-2 architecture fine-tuned on SQuAD, paired with a hybrid sparse (BM25) + dense (FAISS) retrieval pipeline and real-time Server-Sent Events (SSE) token streaming.

🔗 **Live Demo**: https://huggingface.co/spaces/XAEOZ/GPT2-RAG-document-assistant

---

## Architecture Overview

```text
┌────────────────────────────────────────────────────────┐
│               Enterprise Client Web UI                 │
│         (Single-column calm light layout, SSE)         │
└──────────────────────────┬─────────────────────────────┘
                           │ HTTP / EventStream
                           ▼
┌────────────────────────────────────────────────────────┐
│             FastAPI Application (Port 7860)            │
│  - Session isolation (in-memory index per visitor)    │
│  - File validation (.txt, .md, .pdf | ≤ 5MB | ≤ 5 files)│
│  - 30-minute inactivity session pruning               │
└──────────────┬──────────────────────────┬──────────────┘
               │                          │
        Query  ▼                   Prompt ▼
┌──────────────────────────────┐  ┌──────────────────────────────┐
│       Hybrid Retriever       │  │    Unified Model Dispatch    │
│  ┌────────────────────────┐  │  │   generate(model, prompt)    │
│  │ BM25 Sparse Index      │  │  ├──────────────────────────────┤
│  └────────────────────────┘  │  │ • GPT-2 124M SQuAD Fine-Tuned│
│  ┌────────────────────────┐  │  │ • GPT-2 124M Pretrained Base │
│  │ FAISS Dense Index      │  │  │ • Qwen 2.5 1.5B Instruct     │
│  │ (all-MiniLM-L6-v2)     │  │  └──────────────────────────────┘
│  └────────────────────────┘  │
│  ┌────────────────────────┐  │
│  │ Reciprocal Rank Fusion │  │
│  └────────────────────────┘  │
└──────────────────────────────┘
```

### Retrieval & Indexing Pipeline
1. **Document Ingestion**: Multi-page text extraction from `.txt`, `.md`, and `.pdf` files.
2. **Chunking**: 150-token overlapping chunks with 30-token stride via `tiktoken`.
3. **Dual Indexing**:
   - **Sparse Matching**: BM25 Okapi term frequency index for exact keyword matching.
   - **Dense Vectors**: 384-dimensional dense vectors generated via `all-MiniLM-L6-v2` indexed in a normalized FAISS `IndexFlatIP`.
4. **Rank Fusion**: Reciprocal Rank Fusion ($k=60$) dynamically merges sparse and dense rankings into a unified score.

### Generation & Models
* **GPT-2 Fine-Tuned (SQuAD)**: Autoregressively trained on QA pairs using custom `CausalSelfAttention`, `MLP`, and transformer blocks.
* **GPT-2 Base**: Raw OpenAI 124M checkpoint for baseline comparison.
* **Qwen 2.5 1.5B Instruct**: Stronger local instruct-tuned LLM accessed via Hugging Face Transformers.

---

## Session Isolation & Privacy
* Every visitor gets an isolated in-memory index tracked by session ID.
* Upload limits: maximum 5 files, maximum 5 MB per file (`.txt`, `.md`, `.pdf` only).
* Inactive sessions are automatically evicted after 30 minutes of idle time.

---

## Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run local UI launcher
python run_ui.py
```

Open your browser at `http://127.0.0.1:8000`.

---

## Docker Deployment

Build and run locally with Docker:

```bash
docker build -t gpt2-rag-app .
docker run -p 7860:7860 -e HF_MODEL_REPO="<your-username>/gpt2-squad-rag" gpt2-rag-app
```

