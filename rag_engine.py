"""
RAG Engine Manager for GPT-2 Local UI
Integrates:
- GPT-2 Fine-Tuned Checkpoint (SQuAD 124M) & GPT-2 Pretrained Weights (HuggingFace)
- Hybrid Retriever (BM25 Sparse + FAISS Dense + Reciprocal Rank Fusion)
- Multi-format document loading (txt, md, pdf, json, csv, py)
- Drag-and-drop context folder ingestion
- Token-by-token streaming & synchronous generation
"""

import os
import sys
import time
import io
import torch
import tiktoken
from typing import List, Dict, Any, Generator, Tuple, Optional

# Ensure checkpoint unpickler can locate GPTConfig if saved under __main__
from GPT2_Pretrained import GPT, GPTConfig
import __main__
setattr(__main__, 'GPTConfig', GPTConfig)
sys.modules['__main__'].GPTConfig = GPTConfig

from rag.fusion import reciprocal_rank_fusion
from rag.sparse_index import SparseIndex
from rag.chunking import chunk_text

# PyMuPDF for PDF text extraction
try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

# FAISS and SentenceTransformer
from sentence_transformers import SentenceTransformer
import faiss


class CachedDenseIndex:
    """Fast Dense Index that reuses a pre-loaded SentenceTransformer instance."""
    def __init__(self, chunks: List[Dict[str, Any]], embedder: SentenceTransformer):
        self.chunks = chunks
        self.model = embedder
        text_chunks = [c['text'] for c in chunks]

        if len(text_chunks) > 0:
            embedded_chunks = self.model.encode(text_chunks)
            faiss.normalize_L2(embedded_chunks)
            self.index = faiss.IndexFlatIP(embedded_chunks.shape[1])
            self.index.add(embedded_chunks)
        else:
            self.index = None

    def search(self, query: str, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        if self.index is None or len(self.chunks) == 0:
            return []
        k = min(k, len(self.chunks))
        embedded_query = self.model.encode([query])
        faiss.normalize_L2(embedded_query)
        scores, indices = self.index.search(embedded_query, k)
        result = []
        for score, chunk_idx in zip(scores[0], indices[0]):
            if chunk_idx >= 0 and chunk_idx < len(self.chunks):
                result.append((self.chunks[chunk_idx], float(score)))
        return result


class EngineRetriever:
    """Wrapper that manages Sparse (BM25) and Dense (FAISS) indexes."""
    def __init__(self, chunks: List[Dict[str, Any]], embedder: SentenceTransformer):
        self.chunks = chunks
        self.sparse_index = SparseIndex(chunks) if len(chunks) > 0 else None
        self.dense_index = CachedDenseIndex(chunks, embedder) if len(chunks) > 0 else None

    def search_sparse(self, query: str, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        if not self.sparse_index or not self.chunks:
            return []
        k = min(k, len(self.chunks))
        return self.sparse_index.search(query, k)

    def search_dense(self, query: str, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        if not self.dense_index or not self.chunks:
            return []
        return self.dense_index.search(query, k)

    def search_hybrid(self, query: str, k: int = 5, overfetch: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        if not self.chunks:
            return []
        k = min(k, len(self.chunks))
        overfetch = min(max(overfetch, k * 2), len(self.chunks))
        sparse_result = self.search_sparse(query, k=overfetch)
        dense_result = self.search_dense(query, k=overfetch)
        result = reciprocal_rank_fusion([sparse_result, dense_result], k=60)
        return result[:k]


class SessionContext:
    """Manages an isolated in-memory index for a visitor session."""
    MAX_FILES = 5
    MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB
    ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}

    def __init__(self, session_id: str, default_docs: List[Dict[str, Any]], embedder: SentenceTransformer):
        self.session_id = session_id
        self.last_active = time.time()
        self.embedder = embedder
        self.has_custom_docs = False
        self.documents: List[Dict[str, Any]] = []
        self.chunks: List[Dict[str, Any]] = []
        self.retriever: Optional[EngineRetriever] = None

        if default_docs:
            for d in default_docs:
                self.documents.append(dict(d))
            self.rebuild_index()

    def touch(self):
        self.last_active = time.time()

    def rebuild_index(self):
        all_chunks = []
        for doc in self.documents:
            text = doc.get("text", "").strip()
            if not text:
                continue
            pieces = chunk_text(text, chunk_size=150, overlap=30)
            for i, piece in enumerate(pieces):
                all_chunks.append({
                    "chunk_id": f"{doc['id']}_chunk{i}",
                    "doc_id": doc["id"],
                    "text": piece,
                })
        self.chunks = all_chunks
        self.retriever = EngineRetriever(self.chunks, self.embedder) if len(self.chunks) > 0 else None

    def add_documents_from_bytes(self, files: List[Tuple[str, bytes]]) -> Dict[str, Any]:
        """Ingests files into the session index with strict validation."""
        self.touch()

        # If transitioning from default documents, replace them with user's uploads
        if not self.has_custom_docs and files:
            self.documents = []
            self.has_custom_docs = True

        # Check total file count limit
        if len(self.documents) + len(files) > self.MAX_FILES:
            raise ValueError(f"Limit exceeded: maximum {self.MAX_FILES} files allowed per session (currently have {len(self.documents)}).")

        for filename, bdata in files:
            ext = os.path.splitext(filename)[1].lower()

            # Extension validation: .txt, .md, .pdf only
            if ext not in self.ALLOWED_EXTENSIONS:
                raise ValueError(f"Unsupported file type '{ext}' for '{filename}'. Only .txt, .md, and .pdf files are accepted.")

            # File size validation: max 5 MB
            if len(bdata) > self.MAX_FILE_BYTES:
                size_mb = len(bdata) / (1024 * 1024)
                raise ValueError(f"File '{filename}' ({size_mb:.2f} MB) exceeds the 5 MB per-file limit.")

            # Safe extraction
            text = self._extract_text(filename, bdata, ext)
            if not text.strip():
                raise ValueError(f"Could not extract any readable text from '{filename}'.")

            existing = next((d for d in self.documents if d["id"] == filename), None)
            if existing:
                existing["text"] = text
                existing["size"] = len(bdata)
            else:
                self.documents.append({
                    "id": filename,
                    "name": filename,
                    "text": text,
                    "size": len(bdata)
                })

        self.rebuild_index()
        return self.get_status()

    def _extract_text(self, filename: str, content_bytes: bytes, ext: str) -> str:
        """Safely extracts text with full fallback and error handling."""
        if ext == ".pdf":
            if not HAS_FITZ:
                raise ValueError("PDF parsing library (PyMuPDF) is not installed on the server.")
            try:
                doc = fitz.open(stream=content_bytes, filetype="pdf")
                pages = [page.get_text() for page in doc]
                text = "\n".join(pages).strip()
                if not text:
                    raise ValueError(f"The PDF '{filename}' contains no extractable text (it may be a scanned image).")
                return text
            except Exception as e:
                raise ValueError(f"Failed to read PDF '{filename}': {str(e)}")
        else:
            try:
                return content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    return content_bytes.decode("latin-1")
                except Exception as e:
                    raise ValueError(f"Encoding error reading '{filename}': {str(e)}")

    def remove_document(self, doc_id: str) -> Dict[str, Any]:
        self.touch()
        self.documents = [d for d in self.documents if d["id"] != doc_id]
        self.rebuild_index()
        return self.get_status()

    def clear_documents(self) -> Dict[str, Any]:
        self.touch()
        self.documents = []
        self.chunks = []
        self.retriever = None
        self.has_custom_docs = True
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        return {
            "document_count": len(self.documents),
            "chunk_count": len(self.chunks),
            "max_files": self.MAX_FILES,
            "has_custom_docs": self.has_custom_docs,
            "documents": [
                {
                    "id": d["id"],
                    "name": d["name"],
                    "size_bytes": d.get("size", len(d.get("text", ""))),
                    "chunk_count": sum(1 for c in self.chunks if c["doc_id"] == d["id"])
                }
                for d in self.documents
            ]
        }


class RAGEngine:
    SESSION_IDLE_TIMEOUT = 1800  # 30 minutes in seconds

    def __init__(self, checkpoint_path: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = checkpoint_path or os.environ.get("CHECKPOINT_PATH", "checkpoints/gpt2_124M_squad_ft.pt")
        self.enc = tiktoken.get_encoding("gpt2")
        self.eos_id = self.enc.eot_token
        self.max_seq_len = 1024

        # Model storage
        self.models: Dict[str, Any] = {}
        self.current_model_name = "squad_ft"  # squad_ft | gpt2_pretrained | qwen2.5_instruct

        # Embedder singleton for fast re-indexing
        print(f"[*] Initializing SentenceTransformer on {self.device}...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Default documents template
        self.default_documents: List[Dict[str, Any]] = []

        # Session store: session_id -> SessionContext
        self.sessions: Dict[str, SessionContext] = {}

        # Load primary model
        if not self.load_model("squad_ft"):
            print("[!] Primary squad_ft checkpoint not available. Loading gpt2_pretrained as fallback.")
            self.load_model("gpt2_pretrained")

        # Automatically load default data folder if present
        data_dir = os.environ.get("DATA_DIR", "./data")
        if os.path.exists(data_dir):
            print(f"[*] Loading default context template from: {data_dir}...")
            self._load_default_template(data_dir)

    def _load_default_template(self, dir_path: str):
        """Loads default documents template used to initialize new visitor sessions."""
        for root, _, filenames in os.walk(dir_path):
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext in [".txt", ".md", ".pdf"]:
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, "rb") as f:
                            bdata = f.read()
                        text = bdata.decode("utf-8", errors="replace")
                        self.default_documents.append({
                            "id": fname,
                            "name": fname,
                            "text": text,
                            "size": len(bdata)
                        })
                    except Exception as e:
                        print(f"[!] Warning reading default doc {fpath}: {e}")
        print(f"[+] Loaded {len(self.default_documents)} default template document(s).")

    def get_session(self, session_id: Optional[str] = None) -> SessionContext:
        """Retrieves or creates an in-memory session context, pruning inactive sessions."""
        self.cleanup_inactive_sessions()
        sid = (session_id or "default").strip()
        if sid not in self.sessions:
            self.sessions[sid] = SessionContext(sid, self.default_documents, self.embedder)
        else:
            self.sessions[sid].touch()
        return self.sessions[sid]

    def cleanup_inactive_sessions(self):
        """Removes sessions inactive for more than 30 minutes."""
        now = time.time()
        expired = [sid for sid, s in self.sessions.items() if (now - s.last_active) > self.SESSION_IDLE_TIMEOUT]
        for sid in expired:
            print(f"[*] Pruned inactive session after 30 min idle: {sid}")
            del self.sessions[sid]

    def load_model(self, model_name: str) -> bool:
        """Loads or switches active model ('squad_ft', 'gpt2_pretrained', or 'qwen2.5_instruct')."""
        if model_name in self.models:
            self.current_model_name = model_name
            return True

        if model_name == "squad_ft":
            if not os.path.exists(self.checkpoint_path):
                hf_repo = os.environ.get("HF_MODEL_REPO")
                hf_filename = os.environ.get("HF_MODEL_FILENAME", "gpt2_124M_squad_ft.pt")
                if hf_repo:
                    print(f"[*] Downloading '{hf_filename}' from Hugging Face model repository: {hf_repo}...")
                    try:
                        from huggingface_hub import hf_hub_download
                        target_dir = os.path.dirname(self.checkpoint_path) or "checkpoints"
                        os.makedirs(target_dir, exist_ok=True)
                        downloaded = hf_hub_download(
                            repo_id=hf_repo,
                            filename=hf_filename,
                            local_dir=target_dir,
                            token=os.environ.get("HF_TOKEN")
                        )
                        self.checkpoint_path = downloaded
                        print(f"[+] Download complete: {self.checkpoint_path}")
                    except Exception as e:
                        print(f"[!] Failed to download checkpoint from Hugging Face Hub: {e}")
                        return False
                else:
                    print(f"[!] Checkpoint not found at {self.checkpoint_path}. Set HF_MODEL_REPO env var to auto-download.")
                    return False

            print(f"[*] Loading fine-tuned checkpoint from {self.checkpoint_path}...")
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            model = GPT(ckpt['config'])
            model.load_state_dict(ckpt['model'])
            model.to(self.device)
            model.eval()
            self.models["squad_ft"] = model
            self.current_model_name = "squad_ft"
            print("[+] Fine-tuned SQuAD GPT-2 loaded successfully!")
            return True

        elif model_name == "gpt2_pretrained":
            print("[*] Loading HuggingFace GPT-2 124M pretrained weights...")
            model = GPT.from_pretrained("gpt2")
            model.to(self.device)
            model.eval()
            self.models["gpt2_pretrained"] = model
            self.current_model_name = "gpt2_pretrained"
            print("[+] GPT-2 Pretrained weights loaded successfully!")
            return True

        elif model_name == "qwen2.5_instruct":
            model_id = os.environ.get("INSTRUCT_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
            print(f"[*] Loading stronger local instruct model: {model_id}...")
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True
                )
                if not torch.cuda.is_available():
                    model.to("cpu")
                model.eval()
                self.models["qwen2.5_instruct"] = (model, tokenizer)
                self.current_model_name = "qwen2.5_instruct"
                print(f"[+] Instruct model {model_id} loaded successfully!")
                return True
            except Exception as e:
                print(f"[!] Failed to load {model_id}: {e}")
                return False

        return False

    def extract_text_from_file(self, filename: str, content_bytes: bytes) -> str:
        """Extracts text from txt, md, pdf, json, py, csv."""
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".pdf":
            if not HAS_FITZ:
                return "[Error: PyMuPDF (fitz) is not installed to read PDFs]"
            try:
                doc = fitz.open(stream=content_bytes, filetype="pdf")
                pages = [page.get_text() for page in doc]
                return "\n".join(pages)
            except Exception as e:
                return f"[Error parsing PDF {filename}: {str(e)}]"
        else:
            try:
                return content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    return content_bytes.decode("latin-1")
                except Exception as e:
                    return f"[Error decoding text for {filename}: {str(e)}]"

    def rebuild_index(self):
        """Re-chunks all current documents and builds sparse + dense indexes."""
        all_chunks = []
        for doc in self.documents:
            text = doc["text"].strip()
            if not text:
                continue
            pieces = chunk_text(text, chunk_size=150, overlap=30)
            for i, piece in enumerate(pieces):
                all_chunks.append({
                    "chunk_id": f"{doc['id']}_chunk{i}",
                    "doc_id": doc["id"],
                    "text": piece,
                })
        self.chunks = all_chunks
        if len(self.chunks) > 0:
            self.retriever = EngineRetriever(self.chunks, self.embedder)
        else:
            self.retriever = None

    def add_documents_from_bytes(self, files: List[Tuple[str, bytes]]) -> Dict[str, Any]:
        """Ingests files uploaded or dropped via UI."""
        added_count = 0
        for filename, bdata in files:
            # Check supported file types
            ext = os.path.splitext(filename)[1].lower()
            if ext not in [".txt", ".md", ".markdown", ".pdf", ".json", ".csv", ".py", ".log"]:
                continue
            text = self.extract_text_from_file(filename, bdata)
            # Replace existing doc with same id if already present, else append
            existing = next((d for d in self.documents if d["id"] == filename), None)
            if existing:
                existing["text"] = text
                existing["size"] = len(text)
            else:
                self.documents.append({
                    "id": filename,
                    "name": filename,
                    "text": text,
                    "size": len(text)
                })
            added_count += 1

        self.rebuild_index()
        return self.get_status()

    def load_directory(self, dir_path: str) -> Dict[str, Any]:
        """Loads all supported documents from a local directory."""
        if not os.path.isdir(dir_path):
            raise ValueError(f"Directory not found: {dir_path}")

        files_to_add = []
        for root, _, filenames in os.walk(dir_path):
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext in [".txt", ".md", ".markdown", ".pdf", ".json", ".csv", ".py"]:
                    fpath = os.path.join(root, fname)
                    rel_name = os.path.relpath(fpath, dir_path).replace("\\", "/")
                    try:
                        with open(fpath, "rb") as f:
                            files_to_add.append((rel_name, f.read()))
                    except Exception as e:
                        print(f"[!] Warning reading {fpath}: {e}")

        return self.add_documents_from_bytes(files_to_add)

    def remove_document(self, doc_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Removes a document from the visitor's session index."""
        session = self.get_session(session_id)
        return session.remove_document(doc_id)

    def clear_documents(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Clears all documents in the visitor's session index."""
        session = self.get_session(session_id)
        return session.clear_documents()

    def get_status(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Returns engine and visitor session diagnostics."""
        session = self.get_session(session_id)
        sess_status = session.get_status()
        return {
            "device": self.device,
            "current_model": self.current_model_name,
            "available_models": ["squad_ft", "gpt2_pretrained", "qwen2.5_instruct"],
            "session_id": session.session_id,
            "document_count": sess_status["document_count"],
            "chunk_count": sess_status["chunk_count"],
            "max_files": sess_status["max_files"],
            "has_custom_docs": sess_status["has_custom_docs"],
            "documents": sess_status["documents"],
            "embedding_model": "all-MiniLM-L6-v2",
            "has_retriever": session.retriever is not None,
        }

    def retrieve(self, question: str, k: int = 3, method: str = "hybrid", session_id: Optional[str] = None) -> List[Tuple[Dict[str, Any], float]]:
        """Retrieves top chunks from the visitor's session index."""
        session = self.get_session(session_id)
        if not session.retriever or len(session.chunks) == 0:
            return []

        if method == "sparse":
            return session.retriever.search_sparse(question, k=k)
        elif method == "dense":
            return session.retriever.search_dense(question, k=k)
        else:
            return session.retriever.search_hybrid(question, k=k)

    def format_context(self, results: List[Tuple[Dict[str, Any], float]], max_context_tokens: int = 400) -> str:
        """Joins retrieved chunk texts and defensively truncates."""
        chunk_texts = [chunk["text"] for chunk, _score in results]
        context = "\n".join(chunk_texts)
        context_ids = self.enc.encode(context)
        if len(context_ids) > max_context_tokens:
            context_ids = context_ids[:max_context_tokens]
            context = self.enc.decode(context_ids)
        return context

    def format_prompt(self, model_name: str, context: str, question: str) -> str:
        """Tailors prompt structure according to target model family."""
        if model_name == "qwen2.5_instruct":
            return (
                "<|im_start|>system\n"
                "You are a helpful and concise assistant. Answer the user's question directly using only the provided context. If the answer is not in the context, say 'I cannot find that in the documents.'<|im_end|>\n"
                f"<|im_start|>user\nContext:\n{context}\n\nQuestion: {question}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        else:
            # GPT-2 QA prompt structure
            return f"Context: {context}\nQuestion: {question}\nAnswer:"

    def generate(
        self,
        model_name: str,
        prompt: str,
        max_new_tokens: int = 48,
        temperature: float = 0.0
    ) -> Generator[str, None, None]:
        """Unified token generator for all models behind a single function."""
        # Ensure target model is loaded
        if model_name not in self.models:
            success = self.load_model(model_name)
            if not success:
                yield f"[Error: Could not load model '{model_name}']"
                return

        if model_name == "qwen2.5_instruct":
            import threading
            from transformers import TextIteratorStreamer

            model, tokenizer = self.models["qwen2.5_instruct"]
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.05,
                temperature=temperature if temperature > 0.05 else 1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

            thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
            thread.start()

            for token in streamer:
                yield token
            thread.join()

        else:
            # Custom GPT-2 architecture
            model = self.models[model_name]
            model.eval()
            ids = self.enc.encode(prompt)
            idx = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

            with torch.no_grad():
                for _ in range(max_new_tokens):
                    logits, _ = model(idx[:, -self.max_seq_len:])
                    last_logits = logits[0, -1]

                    if temperature > 0.05:
                        probs = torch.softmax(last_logits / temperature, dim=-1)
                        next_id = int(torch.multinomial(probs, num_samples=1).item())
                    else:
                        next_id = int(last_logits.argmax().item())

                    if next_id == self.eos_id:
                        break

                    token_str = self.enc.decode([next_id])
                    yield token_str
                    idx = torch.cat((idx, torch.tensor([[next_id]], device=self.device)), dim=1)

    def answer_question(
        self,
        question: str,
        k: int = 3,
        method: str = "hybrid",
        max_context_tokens: int = 400,
        max_new_tokens: int = 48,
        temperature: float = 0.0,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Full synchronous answer pipeline with per-session retrieval."""
        t_start = time.time()
        active_model = model or self.current_model_name

        # Step 1: Retrieval from session index
        t_ret_start = time.time()
        results = self.retrieve(question, k=k, method=method, session_id=session_id)
        retrieval_ms = round((time.time() - t_ret_start) * 1000, 2)

        context = self.format_context(results, max_context_tokens)
        prompt = self.format_prompt(active_model, context, question)

        # Step 2: Generation via single unified generate() function
        t_gen_start = time.time()
        generated_tokens = []
        for tok in self.generate(active_model, prompt, max_new_tokens, temperature):
            generated_tokens.append(tok)
        answer = "".join(generated_tokens).strip()
        generation_ms = round((time.time() - t_gen_start) * 1000, 2)
        total_ms = round((time.time() - t_start) * 1000, 2)

        # Clean serialized chunk data for json response
        retrieved_clean = [
            {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "text": c["text"],
                "score": round(score, 4)
            }
            for c, score in results
        ]

        return {
            "answer": answer,
            "context_used": context,
            "retrieved_chunks": retrieved_clean,
            "metrics": {
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_ms,
                "token_count": len(generated_tokens),
                "model_used": active_model,
                "method_used": method
            }
        }


