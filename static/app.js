/* ═══════════════════════════════════════════════════════════════
   Document Assistant — Client Logic
   ═══════════════════════════════════════════════════════════════ */

function init() {
    // ── Session & State (Safe from Storage Restriction errors) ──
    let sessionId = 'default';
    try {
        sessionId = sessionStorage.getItem('rag_session_id');
        if (!sessionId) {
            sessionId = 'sess_' + Math.random().toString(36).substring(2, 10) + '_' + Date.now();
            sessionStorage.setItem('rag_session_id', sessionId);
        }
    } catch (e) {
        sessionId = 'sess_' + Math.random().toString(36).substring(2, 10) + '_' + Date.now();
    }

    const state = {
        model: 'squad_ft',
        method: 'hybrid',
        k: 3,
        maxCtx: 400,
        maxGen: 48,
        generating: false,
        abort: null,
        kbOpen: true,
        sessionId: sessionId,
    };

    // ── DOM Elements ──
    const $ = id => document.getElementById(id);
    const modelTrigger = $('modelTrigger');
    const modelMenu = $('modelMenu');
    const modelValue = $('modelValue');
    const methodTrigger = $('methodTrigger');
    const methodMenu = $('methodMenu');
    const methodValue = $('methodValue');
    const modelDropdown = $('modelDropdown');
    const methodDropdown = $('methodDropdown');
    const kbShelf = $('kbShelf');
    const kbToggle = $('kbToggle');
    const kbBadge = $('kbBadge');
    const kbTable = $('kbTable');
    const kbDrop = $('kbDrop');
    const btnAddFiles = $('btnAddFiles');
    const btnBrowse = $('btnBrowse');
    const btnClearKb = $('btnClearKb');
    const folderInput = /** @type {HTMLInputElement|null} */ ($('folderInput'));
    const fileInput = /** @type {HTMLInputElement|null} */ ($('fileInput'));
    const localPathInput = /** @type {HTMLInputElement|null} */ ($('localPathInput'));
    const btnIndexPath = $('btnIndexPath');
    const conversation = $('conversation');
    const emptyState = $('emptyState');
    const messages = $('messages');
    const input = /** @type {HTMLTextAreaElement|null} */ ($('input'));
    const btnSend = /** @type {HTMLButtonElement|null} */ ($('btnSend'));
    const btnNewChat = $('btnNewChat');
    const deviceLabel = $('deviceLabel');

    // Inline error helper for document uploads
    function showUploadError(msg) {
        alert(msg);
    }

    // ══════════ DROPDOWNS ══════════
    function setupDropdown(dropdown, trigger, menu, onSelect) {
        if (!dropdown || !trigger || !menu) return;
        trigger.addEventListener('click', evt => {
            evt.preventDefault();
            evt.stopPropagation();
            document.querySelectorAll('.dropdown.open').forEach(d => {
                if (d !== dropdown) d.classList.remove('open');
            });
            dropdown.classList.toggle('open');
        });
        menu.querySelectorAll('.dropdown-item').forEach(item => {
            item.addEventListener('click', evt => {
                evt.preventDefault();
                evt.stopPropagation();
                menu.querySelectorAll('.dropdown-item').forEach(i => i.classList.remove('active'));
                item.classList.add('active');
                dropdown.classList.remove('open');
                onSelect(item);
            });
        });
    }

    setupDropdown(modelDropdown, modelTrigger, modelMenu, async item => {
        const key = item.dataset.model;
        const nameEl = item.querySelector('.item-name');
        const origText = nameEl ? nameEl.textContent : key;
        if (modelValue) modelValue.textContent = origText;

        if (key && key !== state.model) {
            state.model = key;
            if (key === 'qwen2.5_instruct' && modelValue) {
                modelValue.textContent = 'Loading Qwen...';
            }
            await switchModel(key);
            if (modelValue) modelValue.textContent = origText;
        }
    });

    setupDropdown(methodDropdown, methodTrigger, methodMenu, item => {
        state.method = item.dataset.method || 'hybrid';
        const nameEl = item.querySelector('.item-name');
        if (methodValue && nameEl) methodValue.textContent = nameEl.textContent;
    });

    document.addEventListener('click', () => {
        document.querySelectorAll('.dropdown.open').forEach(d => d.classList.remove('open'));
    });

    // ══════════ DOCUMENTS SHELF & VALIDATION ══════════
    if (kbToggle && kbShelf) {
        kbToggle.addEventListener('click', evt => {
            const target = /** @type {HTMLElement|null} */ (evt.target);
            if (target && target.closest('.text-btn')) return;
            state.kbOpen = !state.kbOpen;
            kbShelf.classList.toggle('collapsed', !state.kbOpen);
        });
    }

    if (btnAddFiles && fileInput) {
        btnAddFiles.addEventListener('click', evt => {
            evt.preventDefault();
            evt.stopPropagation();
            fileInput.click();
        });
    }

    if (btnBrowse && fileInput) {
        btnBrowse.addEventListener('click', evt => {
            evt.preventDefault();
            evt.stopPropagation();
            fileInput.click();
        });
    }

    if (folderInput) {
        folderInput.addEventListener('change', async () => {
            const files = Array.from(folderInput.files || []);
            if (files.length) await handleAndUploadFiles(files);
            folderInput.value = '';
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', async () => {
            const files = Array.from(fileInput.files || []);
            if (files.length) await handleAndUploadFiles(files);
            fileInput.value = '';
        });
    }

    // Drag-and-drop
    if (kbDrop && fileInput) {
        kbDrop.addEventListener('click', () => fileInput.click());
        kbDrop.addEventListener('dragover', evt => {
            evt.preventDefault();
            kbDrop.classList.add('dragover');
        });
        kbDrop.addEventListener('dragleave', () => kbDrop.classList.remove('dragover'));
        kbDrop.addEventListener('drop', async evt => {
            evt.preventDefault();
            kbDrop.classList.remove('dragover');

            const dt = evt.dataTransfer;
            if (!dt) return;

            const collected = [];

            async function walk(entry) {
                if (!entry) return;
                if (entry.isFile && typeof entry.file === 'function') {
                    const f = await new Promise(resolve => entry.file(resolve, () => resolve(null)));
                    if (f) collected.push(f);
                } else if (entry.isDirectory && typeof entry.createReader === 'function') {
                    const reader = entry.createReader();
                    const entries = await new Promise(resolve => reader.readEntries(resolve, () => resolve([])));
                    for (const sub of entries) {
                        await walk(sub);
                    }
                }
            }

            if (dt.items && dt.items.length > 0) {
                const promises = [];
                for (let i = 0; i < dt.items.length; i++) {
                    const item = dt.items[i];
                    const entry = item.webkitGetAsEntry ? item.webkitGetAsEntry() : null;
                    if (entry) {
                        promises.push(walk(entry));
                    } else if (item.getAsFile) {
                        const f = item.getAsFile();
                        if (f) collected.push(f);
                    }
                }
                await Promise.all(promises);
            } else if (dt.files && dt.files.length > 0) {
                for (let i = 0; i < dt.files.length; i++) {
                    collected.push(dt.files[i]);
                }
            }

            if (collected.length) {
                await handleAndUploadFiles(collected);
            }
        });
    }

    // File validation: max 5 MB per file, max 5 files total, .txt/.md/.pdf only
    async function handleAndUploadFiles(files) {
        const allowedExts = ['.txt', '.md', '.pdf'];
        const MAX_SIZE = 5 * 1024 * 1024; // 5 MB

        if (files.length > 5) {
            showUploadError('You can upload at most 5 files per session.');
            return;
        }

        const validFiles = [];
        for (const f of files) {
            const ext = '.' + f.name.split('.').pop().toLowerCase();
            if (!allowedExts.includes(ext)) {
                showUploadError(`'${f.name}' was rejected. Only .txt, .md, and .pdf files are accepted.`);
                return;
            }
            if (f.size > MAX_SIZE) {
                showUploadError(`'${f.name}' (${(f.size / (1024 * 1024)).toFixed(1)} MB) exceeds the 5 MB limit.`);
                return;
            }
            validFiles.push(f);
        }

        if (validFiles.length) {
            await uploadFiles(validFiles);
        }
    }

    async function uploadFiles(files) {
        const fd = new FormData();
        files.forEach(f => fd.append('files', f));
        fd.append('session_id', state.sessionId);

        try {
            const res = await fetch('/api/context/upload', {
                method: 'POST',
                headers: { 'X-Session-ID': state.sessionId },
                body: fd
            });
            const data = await res.json();
            if (res.ok && data.success) {
                await sync();
            } else {
                showUploadError(data.detail || 'Failed to upload files.');
            }
        } catch (err) {
            console.error(err);
            showUploadError('Error uploading files to server.');
        }
    }

    // Index local folder button
    if (btnIndexPath && localPathInput) {
        btnIndexPath.addEventListener('click', async evt => {
            evt.preventDefault();
            const p = localPathInput.value.trim();
            if (!p) return;
            const origText = btnIndexPath.textContent;
            btnIndexPath.textContent = 'Indexing...';
            btnIndexPath.disabled = true;
            try {
                const res = await fetch('/api/context/load-local', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'X-Session-ID': state.sessionId },
                    body: JSON.stringify({ path: p, session_id: state.sessionId })
                });
                const data = await res.json();
                if (res.ok && data.success) {
                    await sync();
                } else {
                    alert(data.detail || 'Failed to index folder.');
                }
            } catch (err) {
                console.error(err);
                alert('Error indexing folder: ' + err.message);
            } finally {
                btnIndexPath.textContent = origText;
                btnIndexPath.disabled = false;
            }
        });
    }

    // Clear KB button
    if (btnClearKb) {
        btnClearKb.addEventListener('click', async evt => {
            evt.preventDefault();
            if (!confirm('Remove all documents from your session?')) return;
            try {
                await fetch('/api/context/clear', {
                    method: 'DELETE',
                    headers: { 'X-Session-ID': state.sessionId }
                });
                await sync();
            } catch (err) {
                console.error(err);
            }
        });
    }

    // ══════════ STATUS SYNC ══════════
    async function sync() {
        try {
            const res = await fetch('/api/status', {
                headers: { 'X-Session-ID': state.sessionId }
            });
            if (!res.ok) return;
            const d = await res.json();
            if (deviceLabel && d.device) {
                deviceLabel.textContent = d.device.toUpperCase();
            }
            if (kbBadge) {
                kbBadge.textContent = `${d.document_count} document${d.document_count === 1 ? '' : 's'}, ${d.chunk_count} chunks`;
            }
            renderKbTable(d.documents);
        } catch (err) {
            console.error('Status sync error:', err);
        }
    }

    function renderKbTable(docs) {
        if (!kbTable) return;
        kbTable.innerHTML = '';
        if (!docs || !docs.length) {
            kbTable.innerHTML = '<div style="font-size:.8125rem;color:var(--text-3);padding:4px 0">No documents indexed in your session.</div>';
            return;
        }
        docs.forEach(doc => {
            const row = document.createElement('div');
            row.className = 'kb-row';
            row.innerHTML = `
                <svg class="kb-row-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                    <polyline points="14 2 14 8 20 8"/>
                </svg>
                <span class="kb-row-name">${esc(doc.name)}</span>
                <span class="kb-row-meta">${doc.chunk_count} chunks</span>
                <button class="kb-row-remove" title="Remove" data-id="${esc(doc.id)}">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                </button>
            `;
            const removeBtn = row.querySelector('.kb-row-remove');
            if (removeBtn) {
                removeBtn.addEventListener('click', async evt => {
                    evt.stopPropagation();
                    await fetch(`/api/context/document/${encodeURIComponent(doc.id)}`, {
                        method: 'DELETE',
                        headers: { 'X-Session-ID': state.sessionId }
                    });
                    await sync();
                });
            }
            kbTable.appendChild(row);
        });
    }

    // ══════════ MODEL SWITCH ══════════
    async function switchModel(key) {
        try {
            await fetch('/api/model/switch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: key })
            });
        } catch (err) {
            console.error('Model switch error:', err);
        }
    }

    // ══════════ NEW CHAT ══════════
    if (btnNewChat && messages && emptyState) {
        btnNewChat.addEventListener('click', evt => {
            evt.preventDefault();
            messages.innerHTML = '';
            emptyState.style.display = 'flex';
        });
    }

    // ══════════ STARTERS ══════════
    document.querySelectorAll('.starter').forEach(s => {
        s.addEventListener('click', evt => {
            evt.preventDefault();
            const btn = /** @type {HTMLElement} */ (s);
            const q = btn.dataset.q || btn.textContent.trim();
            if (input && q) {
                input.value = q;
                send();
            }
        });
    });

    // ══════════ COMPOSER ══════════
    if (input) {
        input.addEventListener('input', () => {
            input.style.height = 'auto';
            input.style.height = Math.min(input.scrollHeight, 140) + 'px';
        });
        input.addEventListener('keydown', evt => {
            if (evt.key === 'Enter' && !evt.shiftKey) {
                evt.preventDefault();
                send();
            }
        });
    }
    if (btnSend) {
        btnSend.addEventListener('click', evt => {
            evt.preventDefault();
            send();
        });
    }

    // ══════════ SEND & STREAM (WITH AUTOMATIC FALLBACK) ══════════
    async function send() {
        if (!input || !messages || !emptyState) return;
        const q = input.value.trim();
        if (!q || state.generating) return;

        emptyState.style.display = 'none';
        input.value = '';
        input.style.height = 'auto';

        // Build turn container
        const turn = document.createElement('div');
        turn.className = 'turn';

        // User message
        const userDiv = document.createElement('div');
        userDiv.className = 'user-msg';
        userDiv.innerHTML = `<div class="msg-label">You</div><div class="user-text">${esc(q)}</div>`;
        turn.appendChild(userDiv);

        // Assistant message
        const asstDiv = document.createElement('div');
        asstDiv.className = 'asst-msg';
        const asstNames = {
            squad_ft: 'Document Assistant (GPT-2 Fine-Tuned)',
            gpt2_pretrained: 'GPT-2 Base',
            qwen2.5_instruct: 'Qwen 2.5 1.5B Instruct'
        };
        const asstName = asstNames[state.model] || 'Document Assistant';
        asstDiv.innerHTML = `
            <div class="asst-header">
                <div class="msg-label">${asstName}</div>
                <div class="asst-meta"><span class="asst-latency"></span></div>
            </div>
            <div class="asst-text"><span class="cursor"></span></div>
            <div class="sources"></div>
        `;
        turn.appendChild(asstDiv);
        messages.appendChild(turn);
        scrollDown();

        const textEl = asstDiv.querySelector('.asst-text');
        const latEl = asstDiv.querySelector('.asst-latency');
        const srcEl = asstDiv.querySelector('.sources');

        state.generating = true;
        if (btnSend) btnSend.disabled = true;
        state.abort = new AbortController();

        try {
            // Attempt SSE Stream first
            const res = await fetch('/api/chat/stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Session-ID': state.sessionId
                },
                body: JSON.stringify({
                    question: q,
                    method: state.method,
                    k: state.k,
                    max_context_tokens: state.maxCtx,
                    max_new_tokens: state.maxGen,
                    model: state.model,
                    session_id: state.sessionId,
                }),
                signal: state.abort.signal,
            });

            if (!res.ok || !res.body) {
                throw new Error(`Stream HTTP ${res.status}`);
            }

            const reader = res.body.getReader();
            const dec = new TextDecoder();
            let buf = '';
            let evtType = 'token';
            let full = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                buf += dec.decode(value, { stream: true });
                const lines = buf.split(/\r?\n/);
                buf = lines.pop() || '';

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed) continue;
                    if (trimmed.startsWith('event:')) {
                        evtType = trimmed.slice(6).trim();
                    } else if (trimmed.startsWith('data:')) {
                        const raw = trimmed.slice(5).trim();
                        if (!raw) continue;
                        try {
                            const d = JSON.parse(raw);
                            if (evtType === 'metadata') {
                                if (latEl) latEl.textContent = `${d.retrieval_ms}ms retrieval`;
                                if (srcEl) renderSources(srcEl, d.retrieved_chunks || [], d.method);
                            } else if (evtType === 'token') {
                                full += d.token;
                                if (textEl) {
                                    textEl.innerHTML = esc(full) + '<span class="cursor"></span>';
                                }
                                scrollDown();
                            } else if (evtType === 'done') {
                                if (textEl) textEl.textContent = full || '(No answer generated)';
                                if (latEl) latEl.textContent = `${d.total_ms}ms · ${d.token_count} tokens`;
                            }
                        } catch (parseErr) {
                            console.error('Parse error:', parseErr);
                        }
                    }
                }
            }

            if (textEl) {
                textEl.textContent = full || '(No answer generated)';
            }
        } catch (streamErr) {
            console.warn('Streaming error, falling back to synchronous /api/chat:', streamErr);
            // Fallback to /api/chat
            try {
                const fbRes = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-Session-ID': state.sessionId
                    },
                    body: JSON.stringify({
                        question: q,
                        method: state.method,
                        k: state.k,
                        max_context_tokens: state.maxCtx,
                        max_new_tokens: state.maxGen,
                        model: state.model,
                        session_id: state.sessionId,
                    }),
                });
                const fbData = await fbRes.json();
                if (textEl) textEl.textContent = fbData.answer || '(No answer generated)';
                if (latEl && fbData.total_ms) latEl.textContent = `${fbData.total_ms}ms total`;
                if (srcEl && fbData.retrieved_chunks) renderSources(srcEl, fbData.retrieved_chunks, state.method);
            } catch (fbErr) {
                if (textEl) {
                    textEl.textContent = streamErr.name === 'AbortError' ? '(Cancelled)' : `[Error: ${streamErr.message}]`;
                }
            }
        } finally {
            state.generating = false;
            if (btnSend) btnSend.disabled = false;
            scrollDown();
        }
    }

    // ══════════ SOURCES RENDERING ══════════
    function renderSources(container, chunks, method) {
        if (!chunks || !chunks.length) {
            container.innerHTML = '';
            return;
        }

        const methodLabel = { hybrid: 'Hybrid', dense: 'Semantic', sparse: 'Keyword' }[method] || 'Hybrid';
        const ts = Date.now();
        const ids = chunks.map((_, i) => `src_${ts}_${i}`);
        const maxScore = Math.max(...chunks.map(c => Math.abs(Number(c.score)) || 0.001), 0.001);

        let html = `
            <div class="sources-bar">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                    <polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>
                </svg>
                <span>${chunks.length} source${chunks.length === 1 ? '' : 's'} · ${methodLabel} search</span>
            </div>
            <div class="source-pills">
        `;

        chunks.forEach((c, i) => {
            const scoreNum = Math.abs(Number(c.score)) || 0;
            const pct = Math.min(100, Math.max(15, Math.round((scoreNum / maxScore) * 100)));
            html += `
                <button class="source-pill" data-target="${ids[i]}" type="button">
                    <span class="source-num">${i + 1}</span>
                    <div class="source-pill-text">
                        <div class="source-pill-top">
                            <span class="source-pill-name">${esc(c.doc_id)}</span>
                            <span class="source-pill-id">${esc(c.chunk_id)}</span>
                        </div>
                        <div class="source-score-row">
                            <div class="source-score-track">
                                <div class="source-score-fill" style="width:${pct}%"></div>
                            </div>
                            <span class="source-score-num">${c.score}</span>
                        </div>
                    </div>
                </button>
            `;
        });
        html += '</div>';

        chunks.forEach((c, i) => {
            const scoreNum = Math.abs(Number(c.score)) || 0;
            const pct = Math.min(100, Math.max(15, Math.round((scoreNum / maxScore) * 100)));
            html += `
                <div class="source-detail" id="${ids[i]}">
                    <div class="source-detail-header">
                        <div class="source-detail-title">
                            <span class="source-num">${i + 1}</span>
                            ${esc(c.doc_id)} · ${esc(c.chunk_id)}
                        </div>
                        <div class="source-detail-score-wrap">
                            <div class="source-score-track">
                                <div class="source-score-fill" style="width:${pct}%"></div>
                            </div>
                            <span class="source-detail-score">Score: ${c.score}</span>
                        </div>
                    </div>
                    <div class="source-detail-body">${esc(c.text)}</div>
                </div>
            `;
        });

        container.innerHTML = html;

        // Toggle source details on pill click
        container.querySelectorAll('.source-pill').forEach(pill => {
            pill.addEventListener('click', evt => {
                evt.preventDefault();
                const btn = /** @type {HTMLElement} */ (pill);
                const targetId = btn.dataset.target;
                if (!targetId) return;
                const detail = document.getElementById(targetId);
                if (detail) {
                    detail.classList.toggle('open');
                    scrollDown();
                }
            });
        });
    }

    // ══════════ HELPERS ══════════
    function scrollDown() {
        if (conversation) {
            conversation.scrollTop = conversation.scrollHeight;
        }
    }

    function esc(str) {
        if (!str) return '';
        const d = document.createElement('div');
        d.textContent = str;
        return d.innerHTML;
    }

    // Initialize
    sync();
}

// Ensure execution whether script loads before or after DOMContentLoaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
