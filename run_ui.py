"""
One-Click Launcher for Document Assistant GPT-2 UI
Runs local FastAPI backend and launches default browser once server is ready.
"""

import sys
import os
import time
import socket
import urllib.request
import webbrowser
import threading
import uvicorn

# Configure UTF-8 encoding for standard output if supported
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure working directory is this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def free_port(port: int = 8000):
    """Frees the specified port if it is occupied by an orphaned process."""
    if not is_port_in_use(port):
        return
    try:
        import subprocess
        output = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True).decode()
        for line in output.strip().splitlines():
            parts = line.split()
            if len(parts) >= 5 and "LISTENING" in parts:
                pid = int(parts[-1])
                if pid != os.getpid():
                    print(f"[*] Port {port} is occupied by previous process (PID {pid}). Freeing port...")
                    os.system(f"taskkill /PID {pid} /F >nul 2>&1")
                    time.sleep(0.5)
    except Exception:
        pass

def find_available_port(start_port: int = 8000) -> int:
    port = start_port
    while is_port_in_use(port) and port < 8100:
        port += 1
    return port

def wait_and_open_browser(url: str):
    """Polls the server until it responds with HTTP 200, then opens the browser."""
    print("[*] Polling server status until ready...")
    for _ in range(60):
        time.sleep(0.8)
        try:
            with urllib.request.urlopen(f"{url}/api/status", timeout=1.5) as res:
                if res.status == 200:
                    print(f"\n[+] Server is LIVE and ready! Opening browser at {url}...\n")
                    webbrowser.open(url)
                    return
        except Exception:
            pass
    # Fallback
    print(f"\n[!] Opening browser at {url}...\n")
    webbrowser.open(url)

if __name__ == "__main__":
    free_port(8000)
    port = find_available_port(8000)
    url = f"http://127.0.0.1:{port}"

    print("=" * 65)
    print("  DOCUMENT ASSISTANT // GPT-2 WORKBENCH (LOCAL HOST)")
    print(f"  * URL:           {url}")
    print("  * Architecture:  Hybrid RAG (BM25 + FAISS Dense)")
    print("  * Models:        GPT-2 124M SQuAD Fine-Tuned & Pretrained Base")
    print("  * Loading models into memory... (please wait ~5-8s)")
    print("=" * 65)

    # Import app (this loads SentenceTransformer and checkpoint)
    from rag_server import app

    # Start browser polling in background thread
    threading.Thread(target=wait_and_open_browser, args=(url,), daemon=True).start()

    # Run Uvicorn directly with the initialized app
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
