import asyncio
import json
import time
import uuid
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from graph.research_graph import research_graph
from memory.store import memory
from observability.logger import logger

app = FastAPI(
    title="Multi-Agent Research Assistant",
    description="LangGraph-powered research system with Supervisor, Searcher, Summarizer, Critic, and Aggregator agents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


APP_HTML = r"""
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Research Assistant</title>
    <style>
        :root {
            --bg: #f4f1ea;
            --panel: rgba(255, 255, 255, 0.78);
            --panel-strong: #ffffff;
            --text: #171717;
            --muted: #5f5b53;
            --line: rgba(23, 23, 23, 0.12);
            --accent: #0f766e;
            --accent-strong: #115e59;
            --accent-soft: rgba(15, 118, 110, 0.12);
            --shadow: 0 20px 60px rgba(23, 23, 23, 0.10);
            --radius: 24px;
        }

        body[data-theme="dark"] {
            --bg: #0b1220;
            --panel: rgba(15, 23, 42, 0.76);
            --panel-strong: #111827;
            --text: #e5e7eb;
            --muted: #9ca3af;
            --line: rgba(148, 163, 184, 0.18);
            --accent: #14b8a6;
            --accent-strong: #0f766e;
            --accent-soft: rgba(20, 184, 166, 0.16);
            --shadow: 0 24px 60px rgba(2, 6, 23, 0.35);
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            min-height: 100vh;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.16), transparent 30%),
                radial-gradient(circle at top right, rgba(217, 119, 6, 0.12), transparent 24%),
                linear-gradient(180deg, #f8f5ee 0%, #f3efe7 50%, #efe9dd 100%);
            transition: background .2s ease, color .2s ease;
        }

        body[data-theme="dark"] {
            background:
                radial-gradient(circle at top left, rgba(20, 184, 166, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(59, 130, 246, 0.12), transparent 22%),
                linear-gradient(180deg, #0f172a 0%, #0b1220 54%, #08111d 100%);
        }

        .wrap {
            width: min(1120px, calc(100% - 32px));
            margin: 0 auto;
            padding: 32px 0 48px;
        }

        .workspace {
            display: grid;
            grid-template-columns: 280px minmax(0, 1fr);
            gap: 24px;
            align-items: start;
        }

        .hero {
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 24px;
            align-items: stretch;
            margin-bottom: 24px;
        }

        .card {
            background: var(--panel);
            backdrop-filter: blur(18px);
            border: 1px solid rgba(255, 255, 255, 0.65);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
        }

        .intro {
            padding: 32px;
            position: relative;
            overflow: hidden;
        }

        .intro::after {
            content: "";
            position: absolute;
            inset: auto -40px -40px auto;
            width: 180px;
            height: 180px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(15, 118, 110, 0.22), transparent 68%);
            pointer-events: none;
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.75);
            border: 1px solid var(--line);
            color: var(--muted);
            font-size: 0.86rem;
            margin-bottom: 18px;
        }

        h1 {
            margin: 0;
            font-size: clamp(2.4rem, 4vw, 4.4rem);
            line-height: 0.96;
            letter-spacing: -0.05em;
            max-width: 12ch;
        }

        .lead {
            margin: 16px 0 0;
            max-width: 60ch;
            color: var(--muted);
            font-size: 1.02rem;
            line-height: 1.65;
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 16px;
        }

        .stat {
            padding: 20px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 140px;
            background: linear-gradient(180deg, rgba(255,255,255,0.86), rgba(255,255,255,0.72));
        }

        .stat strong {
            font-size: 2rem;
            letter-spacing: -0.05em;
        }

        .stat span {
            color: var(--muted);
            line-height: 1.5;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 24px;
        }

        .panel {
            padding: 24px;
        }

        .panel-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 16px;
        }

        .panel-header h2 {
            margin: 0;
            font-size: 1.1rem;
            letter-spacing: -0.03em;
        }

        .panel-header p {
            margin: 4px 0 0;
            color: var(--muted);
            font-size: 0.95rem;
        }

        .form {
            display: grid;
            gap: 16px;
        }

        textarea {
            width: 100%;
            min-height: 150px;
            resize: vertical;
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 18px 20px;
            font: inherit;
            line-height: 1.6;
            background: rgba(255,255,255,0.92);
            color: var(--text);
            outline: none;
            transition: border-color .2s ease, box-shadow .2s ease;
        }

        textarea:focus {
            border-color: rgba(15, 118, 110, 0.45);
            box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.08);
        }

        .row {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
            justify-content: space-between;
        }

        .chips {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }

        .chip {
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.8);
            color: var(--muted);
            border-radius: 999px;
            padding: 9px 14px;
            cursor: pointer;
            font: inherit;
        }

        .toggle {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            color: var(--muted);
            font-size: 0.96rem;
            user-select: none;
        }

        .toggle input { width: 18px; height: 18px; }

        .actions {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
        }

        .btn {
            appearance: none;
            border: 0;
            border-radius: 999px;
            padding: 13px 18px;
            font: inherit;
            font-weight: 650;
            cursor: pointer;
            transition: transform .15s ease, box-shadow .15s ease, background .15s ease;
        }

        .btn:active { transform: translateY(1px); }

        .btn-primary {
            background: linear-gradient(135deg, var(--accent), var(--accent-strong));
            color: white;
            box-shadow: 0 12px 30px rgba(15, 118, 110, 0.24);
        }

        .btn-secondary {
            background: rgba(255,255,255,0.86);
            color: var(--text);
            border: 1px solid var(--line);
        }

        .status {
            color: var(--muted);
            font-size: 0.95rem;
        }

        .result {
            display: grid;
            gap: 18px;
        }

        .answer {
            padding: 22px;
            border-radius: 20px;
            background: linear-gradient(180deg, rgba(15,118,110,0.08), rgba(255,255,255,0.9));
            border: 1px solid rgba(15,118,110,0.12);
            line-height: 1.7;
            white-space: pre-wrap;
        }

        .meta {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
        }

        .meta div {
            padding: 16px;
            border-radius: 18px;
            background: rgba(255,255,255,0.78);
            border: 1px solid var(--line);
        }

        .meta small {
            display: block;
            color: var(--muted);
            margin-bottom: 6px;
        }

        .sources {
            display: grid;
            gap: 10px;
        }

        .source {
            padding: 14px 16px;
            border-radius: 16px;
            background: rgba(255,255,255,0.78);
            border: 1px solid var(--line);
            color: var(--muted);
            word-break: break-word;
        }

        .empty {
            color: var(--muted);
            border: 1px dashed rgba(95, 91, 83, 0.28);
            border-radius: 18px;
            padding: 18px;
            background: rgba(255,255,255,0.55);
        }

        .sidebar {
            position: sticky;
            top: 24px;
            padding: 20px;
            display: grid;
            gap: 16px;
            max-height: calc(100vh - 48px);
            overflow: hidden;
        }

        .sidebar-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 12px;
            flex-wrap: wrap;
        }

        .sidebar-header h2 {
            margin: 0;
            font-size: 1.05rem;
            letter-spacing: -0.03em;
        }

        .sidebar-header p {
            margin: 4px 0 0;
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.5;
        }

        .history-list {
            display: grid;
            gap: 10px;
            overflow: auto;
            padding-right: 4px;
        }

        .history-item {
            width: 100%;
            text-align: left;
            border: 1px solid var(--line);
            border-radius: 18px;
            background: rgba(255,255,255,0.78);
            padding: 14px;
            cursor: pointer;
            font: inherit;
            color: var(--text);
            transition: transform .15s ease, border-color .15s ease, box-shadow .15s ease;
        }

        .history-item:hover {
            transform: translateY(-1px);
            border-color: rgba(15, 118, 110, 0.25);
            box-shadow: 0 12px 24px rgba(23, 23, 23, 0.06);
        }

        .history-item strong {
            display: block;
            font-size: 0.96rem;
            line-height: 1.45;
            margin-bottom: 8px;
        }

        .history-item span {
            display: block;
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.5;
        }

        .history-empty {
            color: var(--muted);
            font-size: 0.94rem;
            line-height: 1.55;
            border: 1px dashed rgba(95, 91, 83, 0.28);
            border-radius: 18px;
            padding: 16px;
            background: rgba(255,255,255,0.5);
        }

        .theme-toggle {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            color: var(--muted);
            font-size: 0.92rem;
            user-select: none;
            white-space: nowrap;
        }

        .theme-toggle input {
            position: absolute;
            opacity: 0;
            pointer-events: none;
        }

        .theme-switch {
            position: relative;
            width: 48px;
            height: 28px;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.28);
            border: 1px solid var(--line);
            flex: 0 0 auto;
            transition: background .2s ease, border-color .2s ease;
        }

        .theme-switch::after {
            content: "";
            position: absolute;
            top: 3px;
            left: 3px;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: var(--panel-strong);
            box-shadow: 0 4px 10px rgba(0,0,0,0.18);
            transition: transform .2s ease, background .2s ease;
        }

        .theme-toggle input:checked + .theme-switch {
            background: rgba(20, 184, 166, 0.32);
            border-color: rgba(20, 184, 166, 0.38);
        }

        .theme-toggle input:checked + .theme-switch::after {
            transform: translateX(20px);
        }

        body[data-theme="dark"] .card,
        body[data-theme="dark"] .btn-secondary,
        body[data-theme="dark"] .chip,
        body[data-theme="dark"] textarea,
        body[data-theme="dark"] .history-item,
        body[data-theme="dark"] .history-empty,
        body[data-theme="dark"] .empty,
        body[data-theme="dark"] .source,
        body[data-theme="dark"] .meta div,
        body[data-theme="dark"] .stat {
            background: rgba(15, 23, 42, 0.96);
            border-color: rgba(148, 163, 184, 0.18);
            color: var(--text);
        }

        body[data-theme="dark"] .lead,
        body[data-theme="dark"] .panel-header p,
        body[data-theme="dark"] .status,
        body[data-theme="dark"] .sidebar-header p,
        body[data-theme="dark"] .history-item span,
        body[data-theme="dark"] .history-empty,
        body[data-theme="dark"] .empty,
        body[data-theme="dark"] .source,
        body[data-theme="dark"] .meta small,
        body[data-theme="dark"] .toggle,
        body[data-theme="dark"] .theme-toggle {
            color: #cbd5e1;
        }

        body[data-theme="dark"] textarea::placeholder {
            color: #94a3b8;
        }

        body[data-theme="dark"] .answer {
            background: rgba(15, 23, 42, 0.98);
            border-color: rgba(20, 184, 166, 0.22);
        }

        body[data-theme="dark"] .btn-primary {
            box-shadow: 0 12px 30px rgba(20, 184, 166, 0.18);
        }

        @media (max-width: 900px) {
            .workspace { grid-template-columns: 1fr; }
            .sidebar { position: static; max-height: none; }
            .hero { grid-template-columns: 1fr; }
            .meta { grid-template-columns: 1fr; }
        }

        @media (max-width: 640px) {
            .wrap { width: min(100% - 20px, 1120px); padding-top: 20px; }
            .intro, .panel { padding: 18px; }
            h1 { max-width: 100%; }
            .stats { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="wrap">
        <div class="workspace">
            <aside class="card panel sidebar">
                <div class="sidebar-header">
                    <div>
                        <h2>Chat history</h2>
                        <p>Browse past research runs and restore them with one click.</p>
                    </div>
                    <div class="actions" style="gap:10px; justify-content:flex-end;">
                        <label class="theme-toggle" for="theme_toggle">
                            <input id="theme_toggle" type="checkbox" />
                            <span class="theme-switch" aria-hidden="true"></span>
                            <span>Dark</span>
                        </label>
                        <button class="btn btn-secondary" id="clear_history" type="button">Clear</button>
                    </div>
                </div>
                <div class="history-list" id="history_list"></div>
            </aside>

            <div class="main-column">
                <section class="hero">
                    <div class="card intro">
                        <div class="eyebrow">Multi-Agent Research Assistant</div>
                        <h1>Ask. Search. Critique. Answer.</h1>
                        <p class="lead">
                            A clean, minimal research workspace built for demos. Enter a question, let the agent graph gather evidence,
                            and review a concise answer with sources and iteration stats.
                        </p>
                    </div>
                    <div class="stats">
                        <div class="card stat">
                            <strong>4</strong>
                            <span>Specialized agent roles working through a LangGraph workflow</span>
                        </div>
                        <div class="card stat">
                            <strong>3</strong>
                            <span>Maximum loop iterations to keep the research bounded and fast</span>
                        </div>
                    </div>
                </section>

                <section class="card panel grid">
                    <div>
                        <div class="panel-header">
                            <div>
                                <h2>Research query</h2>
                                <p>Use a focused question to get a better result.</p>
                            </div>
                        </div>

                        <div class="form">
                            <textarea id="query" placeholder="Example: Is GPT-4o better than Gemini 1.5 Pro for enterprise use?">What are the best vector databases for a small production app?</textarea>

                            <div class="chips" aria-label="Example queries">
                                <button class="chip" type="button" data-query="Is GPT-4o better than Gemini 1.5 Pro for enterprise use?">Enterprise model choice</button>
                                <button class="chip" type="button" data-query="What are the best vector databases for a small production app?">Vector databases</button>
                                <button class="chip" type="button" data-query="What are the main pros and cons of FastAPI versus Flask for APIs?">FastAPI vs Flask</button>
                            </div>

                            <div class="row">
                                <label class="toggle"><input id="use_memory" type="checkbox" checked /> Use memory cache</label>
                                <div class="actions">
                                    <button class="btn btn-secondary" id="clear" type="button">Clear</button>
                                    <button class="btn btn-primary" id="run" type="button">Run research</button>
                                </div>
                            </div>

                            <div class="status" id="status">Ready.</div>
                        </div>
                    </div>

                    <div class="result">
                        <div class="panel-header">
                            <div>
                                <h2>Result</h2>
                                <p>Answer, metadata, and source list.</p>
                            </div>
                        </div>

                        <div id="empty" class="empty">Run a query to see the research answer here.</div>

                        <div id="output" style="display:none; gap:18px;">
                            <div class="meta">
                                <div><small>Request ID</small><strong id="request_id">-</strong></div>
                                <div><small>Iterations</small><strong id="iterations">-</strong></div>
                                <div><small>Claims</small><strong id="claims">-</strong></div>
                            </div>
                            <div class="answer" id="answer"></div>
                            <div>
                                <div class="panel-header" style="margin-bottom:10px;">
                                    <div>
                                        <h2>Sources</h2>
                                        <p>URLs referenced by the research run.</p>
                                    </div>
                                </div>
                                <div class="sources" id="sources"></div>
                            </div>
                        </div>
                    </div>
                </section>
            </div>
        </div>
    </div>

    <script>
        const queryEl = document.getElementById('query');
        const useMemoryEl = document.getElementById('use_memory');
        const statusEl = document.getElementById('status');
        const outputEl = document.getElementById('output');
        const emptyEl = document.getElementById('empty');
        const runBtn = document.getElementById('run');
        const clearBtn = document.getElementById('clear');
        const historyListEl = document.getElementById('history_list');
        const clearHistoryBtn = document.getElementById('clear_history');
        const themeToggleEl = document.getElementById('theme_toggle');
        const HISTORY_KEY = 'research-chat-history';
        const THEME_KEY = 'research-theme';
        const MAX_HISTORY_ITEMS = 12;

        let historyItems = [];

        const setStatus = (text) => { statusEl.textContent = text; };

        const applyTheme = (theme) => {
            document.body.dataset.theme = theme;
            themeToggleEl.checked = theme === 'dark';
        };

        const loadTheme = () => {
            const savedTheme = localStorage.getItem(THEME_KEY);
            if (savedTheme === 'dark' || savedTheme === 'light') {
                return savedTheme;
            }

            return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        };

        const loadHistory = () => {
            try {
                const raw = localStorage.getItem(HISTORY_KEY);
                historyItems = raw ? JSON.parse(raw) : [];
                if (!Array.isArray(historyItems)) {
                    historyItems = [];
                }
            } catch {
                historyItems = [];
            }
        };

        const saveHistory = () => {
            localStorage.setItem(HISTORY_KEY, JSON.stringify(historyItems.slice(0, MAX_HISTORY_ITEMS)));
        };

        const formatTimestamp = (value) => {
            try {
                return new Intl.DateTimeFormat(undefined, {
                    month: 'short',
                    day: 'numeric',
                    hour: 'numeric',
                    minute: '2-digit',
                }).format(new Date(value));
            } catch {
                return 'Recently';
            }
        };

        const shortPreview = (text, limit = 120) => {
            const collapsed = (text || '').replace(/\s+/g, ' ').trim();
            return collapsed.length > limit ? `${collapsed.slice(0, limit).trimEnd()}...` : collapsed;
        };

        const uniqueSourceUrls = (urls) => {
            const seen = new Set();
            const deduped = [];

            (urls || []).forEach((url) => {
                if (!url || seen.has(url)) return;
                seen.add(url);
                deduped.push(url);
            });

            return deduped;
        };

        const renderSourceLinks = (sourceUrls, fallbackText) => {
            const sources = document.getElementById('sources');
            sources.innerHTML = '';

            const urls = uniqueSourceUrls(sourceUrls);
            if (urls.length) {
                urls.forEach((url) => {
                    const link = document.createElement('a');
                    link.className = 'source';
                    link.href = url;
                    link.target = '_blank';
                    link.rel = 'noreferrer';
                    link.textContent = url;
                    sources.appendChild(link);
                });
                return;
            }

            const emptySource = document.createElement('div');
            emptySource.className = 'source';
            emptySource.textContent = fallbackText || 'No source URLs available.';
            sources.appendChild(emptySource);
        };

        const renderSelectedResult = (item) => {
            if (!item) return;

            document.getElementById('request_id').textContent = item.request_id || '-';
            document.getElementById('iterations').textContent = item.iterations ?? '-';
            document.getElementById('claims').textContent = item.total_claims ?? '-';
            document.getElementById('answer').textContent = item.final_answer || 'No answer generated.';

            renderSourceLinks(item.source_urls, 'No source URLs available.');

            emptyEl.style.display = 'none';
            outputEl.style.display = 'grid';
            setStatus('Loaded from chat history.');
        };

        const renderHistory = () => {
            historyListEl.innerHTML = '';

            if (!historyItems.length) {
                const emptyState = document.createElement('div');
                emptyState.className = 'history-empty';
                emptyState.textContent = 'Your recent research runs will appear here after you ask a question.';
                historyListEl.appendChild(emptyState);
                return;
            }

            historyItems.slice(0, MAX_HISTORY_ITEMS).forEach((item) => {
                const button = document.createElement('button');
                button.className = 'history-item';
                button.type = 'button';
                const title = document.createElement('strong');
                title.textContent = item.query;

                const preview = document.createElement('span');
                preview.textContent = shortPreview(item.final_answer || 'No answer stored yet.');

                const timestamp = document.createElement('span');
                timestamp.textContent = formatTimestamp(item.created_at);

                button.append(title, preview, timestamp);
                button.addEventListener('click', () => {
                    queryEl.value = item.query;
                    renderSelectedResult(item);
                });
                historyListEl.appendChild(button);
            });
        };

        const addHistoryItem = (item) => {
            historyItems = [item, ...historyItems.filter((entry) => entry.request_id !== item.request_id)].slice(0, MAX_HISTORY_ITEMS);
            saveHistory();
            renderHistory();
        };

        loadHistory();
        renderHistory();
        applyTheme(loadTheme());

        document.querySelectorAll('[data-query]').forEach((button) => {
            button.addEventListener('click', () => {
                queryEl.value = button.dataset.query;
                queryEl.focus();
            });
        });

        clearBtn.addEventListener('click', () => {
            queryEl.value = '';
            queryEl.focus();
            outputEl.style.display = 'none';
            emptyEl.style.display = 'block';
            setStatus('Cleared.');
        });

        clearHistoryBtn.addEventListener('click', () => {
            historyItems = [];
            saveHistory();
            renderHistory();
            setStatus('Chat history cleared.');
        });

        themeToggleEl.addEventListener('change', () => {
            const theme = themeToggleEl.checked ? 'dark' : 'light';
            localStorage.setItem(THEME_KEY, theme);
            applyTheme(theme);
            setStatus(theme === 'dark' ? 'Dark theme enabled.' : 'Light theme enabled.');
        });

        runBtn.addEventListener('click', async () => {
            const query = queryEl.value.trim();
            if (!query) {
                setStatus('Enter a query first.');
                return;
            }

            runBtn.disabled = true;
            runBtn.textContent = 'Researching...';
            setStatus('Running the graph. This can take a moment.');

            try {
                const response = await fetch('/research', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, use_memory: useMemoryEl.checked }),
                });

                const contentType = response.headers.get('content-type') || '';
                const responseText = await response.text();
                let data = {};

                if (responseText) {
                    try {
                        data = contentType.includes('application/json')
                            ? JSON.parse(responseText)
                            : { detail: responseText };
                    } catch {
                        data = { detail: responseText };
                    }
                }

                if (!response.ok) {
                    const detail = data.detail || 'Research request failed.';
                    throw new Error(detail.startsWith('<!DOCTYPE') ? 'Server returned HTML instead of JSON. Check the API URL or proxy configuration.' : detail);
                }

                document.getElementById('request_id').textContent = data.request_id;
                document.getElementById('iterations').textContent = data.iterations;
                document.getElementById('claims').textContent = data.total_claims;
                document.getElementById('answer').textContent = data.final_answer || 'No answer generated.';

                renderSourceLinks(data.source_urls, 'No source URLs available.');

                emptyEl.style.display = 'none';
                outputEl.style.display = 'grid';
                setStatus(data.from_memory ? 'Answered from memory cache.' : 'Research complete.');

                addHistoryItem({
                    request_id: data.request_id,
                    query,
                    final_answer: data.final_answer || '',
                    source_urls: data.source_urls || [],
                    iterations: data.iterations,
                    total_claims: data.total_claims,
                    created_at: new Date().toISOString(),
                });
            } catch (error) {
                setStatus(error.message || 'Something went wrong.');
            } finally {
                runBtn.disabled = false;
                runBtn.textContent = 'Run research';
            }
        });
    </script>
</body>
</html>
"""


class ResearchRequest(BaseModel):
    query: str
    use_memory: bool = True


class ResearchResponse(BaseModel):
    request_id: str
    query: str
    final_answer: str
    source_urls: list[str]
    iterations: int
    total_claims: int
    elapsed_seconds: float
    from_memory: bool


@app.get("/", response_class=HTMLResponse)
async def homepage():
    return HTMLResponse(APP_HTML)


@app.post("/research", response_model=ResearchResponse)
async def run_research(request: ResearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[api] request_id={request_id} query={request.query!r}")

    if request.use_memory:
        cached = memory.has_recent_answer(request.query)
        if cached:
            logger.info(f"[api] cache hit for request_id={request_id}")
            return ResearchResponse(
                request_id=request_id,
                query=request.query,
                final_answer=cached["answer"],
                source_urls=cached.get("source_urls", []),
                iterations=0,
                total_claims=len(cached.get("claims", [])),
                elapsed_seconds=0.0,
                from_memory=True,
            )

    start = time.perf_counter()

    initial_state = {
        "query": request.query,
        "search_results": [],
        "summary": None,
        "critique": None,
        "iterations": 0,
        "final_answer": "",
        "agent_logs": [],
        "error": None,
    }

    try:
        final_state = await asyncio.to_thread(research_graph.invoke, initial_state)
    except Exception as exc:
        logger.error(f"[api] graph error for request_id={request_id}: {exc}")
        raise HTTPException(status_code=500, detail=f"Research graph failed: {str(exc)}")

    elapsed = round(time.perf_counter() - start, 3)
    summary = final_state.get("summary")
    total_claims = len(summary.claims) if summary else 0
    source_urls = []
    seen_urls = set()
    for result in final_state.get("search_results", []):
        url = getattr(result, "url", "")
        if not url or not url.startswith("http") or url in seen_urls:
            continue
        seen_urls.add(url)
        source_urls.append(url)

    return ResearchResponse(
        request_id=request_id,
        query=request.query,
        final_answer=final_state.get("final_answer", ""),
        source_urls=source_urls,
        iterations=final_state.get("iterations", 0),
        total_claims=total_claims,
        elapsed_seconds=elapsed,
        from_memory=False,
    )


@app.get("/research/stream")
async def stream_research(query: str):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    async def event_generator() -> AsyncGenerator[str, None]:
        initial_state = {
            "query": query,
            "search_results": [],
            "summary": None,
            "critique": None,
            "iterations": 0,
            "final_answer": "",
            "agent_logs": [],
            "error": None,
        }

        for event in research_graph.stream(initial_state):
            for node_name, node_output in event.items():
                payload = {
                    "node": node_name,
                    "data": {
                        "iterations": node_output.get("iterations"),
                        "logs": node_output.get("agent_logs", []),
                    },
                }
                if node_name == "aggregator":
                    payload["data"]["final_answer"] = node_output.get("final_answer", "")

                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(0)

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/memory/search")
async def search_memory(query: str, n: int = 3):
    results = memory.retrieve_similar(query, n_results=n)
    return {"query": query, "results": results}


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
