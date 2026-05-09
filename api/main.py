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
    title="Tarka",
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
    <title>Tarka</title>
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
            display: flex;
            flex-direction: column;
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
            flex: 1;
            min-height: 0;
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

        .history-item.active {
            border-color: rgba(15, 118, 110, 0.5);
            box-shadow: 0 12px 28px rgba(15, 118, 110, 0.12);
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

        .maker-card {
            margin-top: auto;
            padding: 14px;
            border-top: 1px solid var(--line);
            display: grid;
            gap: 12px;
        }

        .maker-top {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .maker-avatar {
            width: 48px;
            height: 48px;
            border-radius: 16px;
            object-fit: cover;
            border: 1px solid var(--line);
            box-shadow: 0 10px 20px rgba(23, 23, 23, 0.08);
            flex: 0 0 auto;
        }

        .maker-copy {
            display: grid;
            gap: 2px;
        }

        .maker-copy strong {
            font-size: 0.98rem;
            letter-spacing: -0.02em;
        }

        .maker-copy span {
            color: var(--muted);
            font-size: 0.88rem;
        }

        .maker-links {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .maker-link {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 10px;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.78);
            color: var(--text);
            text-decoration: none;
            font-size: 0.86rem;
            transition: transform .15s ease, border-color .15s ease, box-shadow .15s ease;
        }

        .maker-link:hover {
            transform: translateY(-1px);
            border-color: rgba(15, 118, 110, 0.25);
            box-shadow: 0 10px 18px rgba(23, 23, 23, 0.06);
        }

        .chat-shell {
            padding: 24px;
            min-height: calc(100vh - 96px);
            display: flex;
            flex-direction: column;
            gap: 18px;
        }

        .chat-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 16px;
        }

        .chat-header-meta {
            display: grid;
            gap: 8px;
            text-align: right;
            color: var(--muted);
            font-size: 0.92rem;
        }

        .chat-header-meta strong {
            color: var(--text);
            display: block;
        }

        .chat-messages {
            flex: 1;
            min-height: 0;
            overflow: auto;
            display: flex;
            flex-direction: column;
            gap: 14px;
            padding-right: 4px;
        }

        .message {
            max-width: min(820px, 92%);
            padding: 16px 18px;
            border-radius: 22px;
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.8);
            box-shadow: 0 10px 30px rgba(23, 23, 23, 0.06);
        }

        .message.user {
            align-self: flex-end;
            background: linear-gradient(135deg, rgba(15, 118, 110, 0.96), rgba(17, 94, 89, 0.98));
            color: white;
            border-color: transparent;
        }

        .message.assistant {
            align-self: flex-start;
            background: rgba(255,255,255,0.86);
        }

        .message-label {
            display: block;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 8px;
            opacity: 0.78;
        }

        .message-content {
            white-space: pre-wrap;
            line-height: 1.7;
        }

        .message-sources {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 12px;
        }

        .message-source {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 8px 12px;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.78);
            color: var(--muted);
            text-decoration: none;
            font-size: 0.88rem;
        }

        .composer {
            border-top: 1px solid var(--line);
            padding-top: 18px;
            display: grid;
            gap: 12px;
        }

        .composer textarea {
            min-height: 102px;
            resize: none;
        }

        .composer-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            flex-wrap: wrap;
        }

        .session-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.74);
            color: var(--muted);
            font-size: 0.9rem;
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

        body[data-theme="dark"] .message,
        body[data-theme="dark"] .message.assistant,
        body[data-theme="dark"] .message-source,
        body[data-theme="dark"] .session-badge {
            background: rgba(15, 23, 42, 0.96);
            border-color: rgba(148, 163, 184, 0.18);
            color: var(--text);
        }

        body[data-theme="dark"] .message.user {
            background: linear-gradient(135deg, rgba(20, 184, 166, 0.92), rgba(13, 148, 136, 0.98));
            color: #f8fafc;
        }

        body[data-theme="dark"] .message-label,
        body[data-theme="dark"] .chat-header-meta,
        body[data-theme="dark"] .session-badge,
        body[data-theme="dark"] .message-source,
        body[data-theme="dark"] .maker-copy span,
        body[data-theme="dark"] .maker-link {
            color: #cbd5e1;
        }

        body[data-theme="dark"] .maker-link,
        body[data-theme="dark"] .maker-avatar {
            background: rgba(15, 23, 42, 0.96);
            border-color: rgba(148, 163, 184, 0.18);
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
                        <h2>Sessions</h2>
                        <p>Each conversation is stored as a separate session.</p>
                    </div>
                    <div class="actions" style="gap:10px; justify-content:flex-end;">
                        <label class="theme-toggle" for="theme_toggle">
                            <input id="theme_toggle" type="checkbox" />
                            <span class="theme-switch" aria-hidden="true"></span>
                            <span>Dark</span>
                        </label>
                        <button class="btn btn-secondary" id="new_session" type="button">New session</button>
                    </div>
                </div>
                <div class="history-list" id="history_list"></div>

                <div class="maker-card" aria-label="Maker information">
                    <div class="maker-top">
                        <img class="maker-avatar" src="https://github.com/Aragog540.png" alt="GitHub profile picture of Swaroop Bhowmik" />
                        <div class="maker-copy">
                            <strong>Made by Swaroop Bhowmik</strong>
                            <span>Built for Tarka</span>
                        </div>
                    </div>
                    <div class="maker-links">
                        <a class="maker-link" href="https://github.com/Aragog540" target="_blank" rel="noreferrer">GitHub</a>
                        <a class="maker-link" href="https://linkedin.com/in/swaroop-bhowmik-8907b52a0/" target="_blank" rel="noreferrer">LinkedIn</a>
                        <a class="maker-link" href="https://www.instagram.com/_.swar.oop._/" target="_blank" rel="noreferrer">Instagram</a>
                    </div>
                </div>
            </aside>

            <div class="main-column">
                <section class="card chat-shell">
                    <div class="chat-header">
                        <div>
                            <div class="eyebrow">Tarka</div>
                            <h1>Research in a conversation.</h1>
                            <p class="lead">
                                Ask follow-ups, keep the thread, and watch answers stream into the session.
                            </p>
                        </div>
                        <div class="chat-header-meta">
                            <span class="session-badge" id="session_badge">Session 1</span>
                            <strong id="status">Ready.</strong>
                            <span>Scroll through the transcript below.</span>
                        </div>
                    </div>

                    <div class="chat-messages" id="chat_messages"></div>

                    <div class="composer">
                        <div class="chips" aria-label="Example queries">
                            <button class="chip" type="button" data-query="Is GPT-4o better than Gemini 1.5 Pro for enterprise use?">Enterprise model choice</button>
                            <button class="chip" type="button" data-query="What are the best vector databases for a small production app?">Vector databases</button>
                            <button class="chip" type="button" data-query="What are the main pros and cons of FastAPI versus Flask for APIs?">FastAPI vs Flask</button>
                        </div>

                        <textarea id="query" placeholder="Ask a follow-up or start a new research session...">What are the best vector databases for a small production app?</textarea>

                        <div class="composer-row">
                            <label class="toggle"><input id="use_memory" type="checkbox" checked /> Use memory cache</label>
                            <div class="actions">
                                <button class="btn btn-secondary" id="clear" type="button">Clear input</button>
                                <button class="btn btn-primary" id="run" type="button">Send</button>
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
        const runBtn = document.getElementById('run');
        const clearBtn = document.getElementById('clear');
        const sessionListEl = document.getElementById('history_list');
        const newSessionBtn = document.getElementById('new_session');
        const sessionBadgeEl = document.getElementById('session_badge');
        const chatMessagesEl = document.getElementById('chat_messages');
        const themeToggleEl = document.getElementById('theme_toggle');
        const HISTORY_KEY = 'tarka-chat-sessions';
        const ACTIVE_SESSION_KEY = 'tarka-active-session';
        const THEME_KEY = 'research-theme';
        const MAX_SESSIONS = 20;
        const MAX_CONTEXT_MESSAGES = 8;

        let sessions = [];
        let activeSessionId = '';
        let activeStream = null;
        let activeAssistantMessageId = null;
        let activeRequestId = null;

        const nowIso = () => new Date().toISOString();
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

        const loadSessions = () => {
            try {
                const raw = localStorage.getItem(HISTORY_KEY);
                sessions = raw ? JSON.parse(raw) : [];
                if (!Array.isArray(sessions)) {
                    sessions = [];
                }
            } catch {
                sessions = [];
            }

            try {
                activeSessionId = localStorage.getItem(ACTIVE_SESSION_KEY) || '';
            } catch {
                activeSessionId = '';
            }

            if (!sessions.length) {
                const starter = createSession('Session 1');
                sessions = [starter];
                activeSessionId = starter.id;
            }

            if (!sessions.some((session) => session.id === activeSessionId)) {
                activeSessionId = sessions[0].id;
            }
        };

        const saveSessions = () => {
            sessions = sessions.slice(0, MAX_SESSIONS);
            localStorage.setItem(HISTORY_KEY, JSON.stringify(sessions));
            localStorage.setItem(ACTIVE_SESSION_KEY, activeSessionId);
        };

        function createSession(title = 'New session') {
            const id = crypto.randomUUID ? crypto.randomUUID() : String(Date.now());
            return {
                id,
                title,
                created_at: nowIso(),
                updated_at: nowIso(),
                messages: [],
            };
        }

        const getActiveSession = () => sessions.find((session) => session.id === activeSessionId) || sessions[0];

        const setActiveSession = (sessionId) => {
            activeSessionId = sessionId;
            const session = getActiveSession();
            if (session) {
                session.updated_at = nowIso();
            }
            saveSessions();
            renderSessions();
            renderMessages();
        };

        const shortPreview = (text, limit = 120) => {
            const collapsed = (text || '').replace(/\s+/g, ' ').trim();
            return collapsed.length > limit ? `${collapsed.slice(0, limit).trimEnd()}...` : collapsed;
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

        const buildConversationContext = (session) => {
            if (!session || !session.messages.length) return '';

            return session.messages
                .filter((message) => message.content)
                .slice(-MAX_CONTEXT_MESSAGES)
                .map((message) => `${message.role === 'user' ? 'User' : 'Assistant'}: ${message.content}`)
                .join('\n');
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

        const renderSourceLinks = (sourceUrls, container) => {
            const urls = uniqueSourceUrls(sourceUrls);
            container.innerHTML = '';

            if (!urls.length) {
                const empty = document.createElement('div');
                empty.className = 'session-badge';
                empty.textContent = 'No source URLs available.';
                container.appendChild(empty);
                return;
            }

            urls.forEach((url) => {
                const link = document.createElement('a');
                link.className = 'message-source';
                link.href = url;
                link.target = '_blank';
                link.rel = 'noreferrer';
                link.textContent = url;
                container.appendChild(link);
            });
        };

        const renderSessions = () => {
            sessionListEl.innerHTML = '';

            if (!sessions.length) {
                const empty = document.createElement('div');
                empty.className = 'history-empty';
                empty.textContent = 'Start a session to keep a running chat log.';
                sessionListEl.appendChild(empty);
                return;
            }

            sessions.slice().sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at)).forEach((session) => {
                const button = document.createElement('button');
                button.className = 'history-item' + (session.id === activeSessionId ? ' active' : '');
                button.type = 'button';

                const title = document.createElement('strong');
                title.textContent = session.title || 'Untitled session';

                const preview = document.createElement('span');
                preview.textContent = `${session.messages.length} message${session.messages.length === 1 ? '' : 's'}`;

                const timestamp = document.createElement('span');
                timestamp.textContent = formatTimestamp(session.updated_at || session.created_at);

                button.append(title, preview, timestamp);
                button.addEventListener('click', () => setActiveSession(session.id));
                sessionListEl.appendChild(button);
            });
        };

        const renderMessages = () => {
            const session = getActiveSession();
            chatMessagesEl.innerHTML = '';

            if (!session || !session.messages.length) {
                const empty = document.createElement('div');
                empty.className = 'history-empty';
                empty.textContent = 'Ask a question to begin a research session. Follow-up questions stay in the same thread.';
                chatMessagesEl.appendChild(empty);
                sessionBadgeEl.textContent = session?.title || 'Session';
                return;
            }

            session.messages.forEach((message) => {
                const bubble = document.createElement('article');
                bubble.className = `message ${message.role}`;

                const label = document.createElement('span');
                label.className = 'message-label';
                label.textContent = message.role === 'user' ? 'You' : 'Tarka';

                const content = document.createElement('div');
                content.className = 'message-content';
                content.textContent = message.content || (message.role === 'assistant' && message.isStreaming ? 'Thinking...' : '');
                message.domContent = content;

                bubble.append(label, content);

                if (message.role === 'assistant' && message.source_urls && message.source_urls.length) {
                    const sources = document.createElement('div');
                    sources.className = 'message-sources';
                    renderSourceLinks(message.source_urls, sources);
                    bubble.appendChild(sources);
                    message.domSources = sources;
                }

                chatMessagesEl.appendChild(bubble);
            });

            chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
            sessionBadgeEl.textContent = session.title || 'Session';
        };

        const updateActiveAssistant = (patch) => {
            const session = getActiveSession();
            if (!session || !activeAssistantMessageId) return;

            const assistant = session.messages.find((message) => message.id === activeAssistantMessageId);
            if (!assistant) return;

            Object.assign(assistant, patch);
            session.updated_at = nowIso();
            saveSessions();
            renderMessages();
            renderSessions();
        };

        const stopActiveStream = () => {
            if (activeStream) {
                activeStream.close();
                activeStream = null;
            }
        };

        const startNewSession = () => {
            stopActiveStream();
            const session = createSession(`Session ${sessions.length + 1}`);
            sessions.unshift(session);
            activeSessionId = session.id;
            activeAssistantMessageId = null;
            activeRequestId = null;
            saveSessions();
            renderSessions();
            renderMessages();
            queryEl.value = '';
            queryEl.focus();
            setStatus('New session created.');
        };

        const streamAnswer = ({ query, context, useMemory }) => new Promise((resolve, reject) => {
            const params = new URLSearchParams({
                query,
                context,
                use_memory: useMemory ? '1' : '0',
            });

            const source = new EventSource(`/research/stream?${params.toString()}`);
            activeStream = source;
            let finished = false;

            const finish = (payload) => {
                finished = true;
                stopActiveStream();
                resolve(payload || {});
            };

            source.onmessage = (event) => {
                if (!event.data || event.data === '[DONE]') {
                    return;
                }

                let payload;
                try {
                    payload = JSON.parse(event.data);
                } catch {
                    return;
                }

                if (payload.type === 'delta') {
                    if (activeAssistantMessageId) {
                        const session = getActiveSession();
                        const assistant = session?.messages.find((message) => message.id === activeAssistantMessageId);
                        if (assistant) {
                            assistant.content = (assistant.content || '') + (payload.data?.delta || '');
                            session.updated_at = nowIso();
                            saveSessions();
                            renderMessages();
                        }
                    }
                    return;
                }

                if (payload.type === 'node') {
                    const nodeName = payload.node || 'research';
                    if (nodeName === 'supervisor') setStatus('Planning the research path...');
                    else if (nodeName === 'searcher') setStatus('Searching sources...');
                    else if (nodeName === 'summarizer') setStatus('Summarizing evidence...');
                    else if (nodeName === 'critic') setStatus('Reviewing gaps...');
                    else if (nodeName === 'aggregator') setStatus('Finishing the answer...');
                    return;
                }

                if (payload.type === 'final') {
                    if (payload.data?.source_urls) {
                        updateActiveAssistant({ source_urls: payload.data.source_urls });
                    }
                    activeRequestId = payload.data?.request_id || activeRequestId;
                    resolve(payload.data || {});
                    finish(payload.data || {});
                    return;
                }

                if (payload.type === 'done') {
                    finish(payload.data || {});
                }
            };

            source.onerror = () => {
                if (finished) return;
                stopActiveStream();
                reject(new Error('Streaming connection failed.'));
            };
        });

        const sendMessage = async () => {
            const query = queryEl.value.trim();
            if (!query) {
                setStatus('Enter a query first.');
                return;
            }

            const session = getActiveSession();
            if (!session) {
                setStatus('No active session available.');
                return;
            }

            stopActiveStream();

            const conversationContext = buildConversationContext(session);
            const userMessage = {
                id: crypto.randomUUID ? crypto.randomUUID() : String(Date.now()),
                role: 'user',
                content: query,
                created_at: nowIso(),
            };
            const assistantMessage = {
                id: crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-assistant`,
                role: 'assistant',
                content: '',
                source_urls: [],
                isStreaming: true,
                created_at: nowIso(),
            };

            session.messages.push(userMessage, assistantMessage);
            if (!session.title || session.title === 'New session' || session.title.startsWith('Session ')) {
                session.title = shortPreview(query, 42);
            }
            session.updated_at = nowIso();
            activeAssistantMessageId = assistantMessage.id;
            activeRequestId = null;
            saveSessions();
            renderSessions();
            renderMessages();

            queryEl.value = '';
            runBtn.disabled = true;
            runBtn.textContent = 'Streaming...';
            setStatus('Streaming the answer into the session.');

            try {
                const finalPayload = await streamAnswer({
                    query,
                    context: conversationContext,
                    useMemory: useMemoryEl.checked,
                });

                updateActiveAssistant({
                    content: finalPayload.final_answer || getActiveSession()?.messages.find((message) => message.id === activeAssistantMessageId)?.content || 'No answer generated.',
                    source_urls: finalPayload.source_urls || getActiveSession()?.messages.find((message) => message.id === activeAssistantMessageId)?.source_urls || [],
                    isStreaming: false,
                });

                session.updated_at = nowIso();
                saveSessions();
                renderSessions();
                renderMessages();

                setStatus(finalPayload.from_memory ? 'Answered from memory cache.' : 'Research complete.');
            } catch (error) {
                updateActiveAssistant({
                    content: error.message || 'Something went wrong while streaming the answer.',
                    isStreaming: false,
                });
                setStatus(error.message || 'Something went wrong.');
            } finally {
                runBtn.disabled = false;
                runBtn.textContent = 'Send';
            }
        };

        loadSessions();
        applyTheme(loadTheme());
        renderSessions();
        renderMessages();

        document.querySelectorAll('[data-query]').forEach((button) => {
            button.addEventListener('click', () => {
                queryEl.value = button.dataset.query;
                queryEl.focus();
            });
        });

        newSessionBtn.addEventListener('click', startNewSession);

        clearBtn.addEventListener('click', () => {
            queryEl.value = '';
            queryEl.focus();
            setStatus('Input cleared.');
        });

        themeToggleEl.addEventListener('change', () => {
            const theme = themeToggleEl.checked ? 'dark' : 'light';
            localStorage.setItem(THEME_KEY, theme);
            applyTheme(theme);
            setStatus(theme === 'dark' ? 'Dark theme enabled.' : 'Light theme enabled.');
        });

        runBtn.addEventListener('click', sendMessage);

        queryEl.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
                event.preventDefault();
                sendMessage();
            }
        });
    </script>
</body>
</html>
"""


class ResearchRequest(BaseModel):
    query: str
    use_memory: bool = True
    conversation_context: str = ""


class ResearchResponse(BaseModel):
    request_id: str
    query: str
    final_answer: str
    source_urls: list[str]
    iterations: int
    total_claims: int
    elapsed_seconds: float
    from_memory: bool


def _source_urls_from_results(results: list) -> list[str]:
    source_urls = []
    seen_urls = set()
    for result in results:
        url = getattr(result, "url", "")
        if not url or not url.startswith("http") or url in seen_urls:
            continue
        seen_urls.add(url)
        source_urls.append(url)
    return source_urls


def _chunk_text(text: str, chunk_size: int = 24) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    current = []
    for word in words:
        current.append(word)
        if len(current) >= chunk_size:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks


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
        "conversation_context": request.conversation_context,
        "search_results": [],
        "summary": None,
        "critique": None,
        "iterations": 0,
        "final_answer": "",
        "source_urls": [],
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
    source_urls = final_state.get("source_urls", []) or _source_urls_from_results(final_state.get("search_results", []))

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
async def stream_research(query: str, context: str = "", use_memory: bool = True):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    async def event_generator() -> AsyncGenerator[str, None]:
        request_id = str(uuid.uuid4())[:8]
        latest_summary = None
        latest_iterations = 0

        if use_memory:
            cached = memory.has_recent_answer(query)
            if cached:
                cached_answer = cached["answer"]
                cached_source_urls = cached.get("source_urls", [])
                for chunk in _chunk_text(cached_answer):
                    yield f"data: {json.dumps({'type': 'delta', 'node': 'assistant', 'data': {'delta': chunk + ' '}})}\n\n"
                    await asyncio.sleep(0)
                yield f"data: {json.dumps({'type': 'final', 'node': 'assistant', 'data': {'request_id': request_id, 'query': query, 'final_answer': cached_answer, 'source_urls': cached_source_urls, 'iterations': 0, 'total_claims': len(cached.get('claims', [])), 'from_memory': True}})}\n\n"
                return

        initial_state = {
            "query": query,
            "conversation_context": context,
            "search_results": [],
            "summary": None,
            "critique": None,
            "iterations": 0,
            "final_answer": "",
            "source_urls": [],
            "agent_logs": [],
            "error": None,
        }

        for event in research_graph.stream(initial_state):
            for node_name, node_output in event.items():
                if node_output.get("iterations") is not None:
                    latest_iterations = node_output.get("iterations", latest_iterations)
                if node_name == "summarizer":
                    latest_summary = node_output.get("summary", latest_summary)

                payload = {
                    "type": "node",
                    "node": node_name,
                    "data": {
                        "iterations": node_output.get("iterations"),
                        "logs": node_output.get("agent_logs", []),
                    },
                }
                if node_name == "aggregator":
                    payload["data"]["final_answer"] = node_output.get("final_answer", "")
                    payload["data"]["source_urls"] = node_output.get("source_urls", [])

                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(0)

                if node_name == "aggregator":
                    final_answer = node_output.get("final_answer", "")
                    source_urls = node_output.get("source_urls", [])
                    for chunk in _chunk_text(final_answer):
                        yield f"data: {json.dumps({'type': 'delta', 'node': 'assistant', 'data': {'delta': chunk + ' '}})}\n\n"
                        await asyncio.sleep(0)
                    total_claims = len(latest_summary.claims) if latest_summary else 0
                    yield f"data: {json.dumps({'type': 'final', 'node': 'assistant', 'data': {'request_id': request_id, 'query': query, 'final_answer': final_answer, 'source_urls': source_urls, 'iterations': latest_iterations, 'total_claims': total_claims, 'from_memory': False}})}\n\n"
                    return

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/memory/search")
async def search_memory(query: str, n: int = 3):
    results = memory.retrieve_similar(query, n_results=n)
    return {"query": query, "results": results}


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
