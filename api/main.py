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

        .memory-select {
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.84);
            color: var(--text);
            border-radius: 999px;
            padding: 8px 12px;
            font: inherit;
            min-width: 170px;
        }

        .message-metrics {
            margin-top: 10px;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .metric-pill {
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.75);
            color: var(--muted);
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 0.8rem;
        }

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
            scrollbar-width: thin;
            scrollbar-color: rgba(148, 163, 184, 0.5) transparent;
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
            display: grid;
            gap: 10px;
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

        .history-item-actions {
            display: flex;
            justify-content: flex-end;
            position: relative;
        }

        .session-menu-trigger {
            appearance: none;
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.78);
            color: var(--muted);
            border-radius: 999px;
            width: 32px;
            height: 32px;
            font: inherit;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            line-height: 1;
        }

        .session-menu-trigger:hover {
            border-color: rgba(15, 118, 110, 0.25);
            color: var(--text);
        }

        .session-menu {
            position: absolute;
            right: 0;
            top: calc(100% + 8px);
            min-width: 148px;
            padding: 8px;
            border-radius: 16px;
            border: 1px solid var(--line);
            background: var(--panel-strong);
            box-shadow: 0 18px 40px rgba(23, 23, 23, 0.14);
            display: none;
            z-index: 10;
        }

        .session-menu.open {
            display: grid;
            gap: 6px;
        }

        .session-menu-item {
            width: 100%;
            text-align: left;
            appearance: none;
            border: 1px solid transparent;
            background: transparent;
            color: var(--text);
            border-radius: 12px;
            padding: 10px 12px;
            cursor: pointer;
            font: inherit;
            font-size: 0.88rem;
        }

        .session-menu-item:hover {
            background: rgba(15, 118, 110, 0.08);
        }

        .session-menu-item.danger {
            color: #b91c1c;
        }

        .session-menu-item.danger:hover {
            background: rgba(239, 68, 68, 0.08);
        }

        .explicit-flag {
            color: #b91c1c;
            background: rgba(185,28,28,0.06);
            border: 1px solid rgba(185,28,28,0.12);
            padding: 8px 10px;
            border-radius: 10px;
            font-size: 0.86rem;
            display: inline-block;
            margin-top: 8px;
        }

        .session-menu-backdrop {
            position: fixed;
            inset: 0;
            background: transparent;
            display: none;
            z-index: 9;
        }

        .session-menu-backdrop.open {
            display: block;
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
            scrollbar-width: thin;
            scrollbar-color: rgba(148, 163, 184, 0.45) transparent;
        }

        .history-list::-webkit-scrollbar,
        .chat-messages::-webkit-scrollbar {
            width: 10px;
        }

        .history-list::-webkit-scrollbar-track,
        .chat-messages::-webkit-scrollbar-track {
            background: transparent;
        }

        .history-list::-webkit-scrollbar-thumb,
        .chat-messages::-webkit-scrollbar-thumb {
            background: rgba(148, 163, 184, 0.4);
            border-radius: 999px;
            border: 2px solid transparent;
            background-clip: padding-box;
        }

        .history-list::-webkit-scrollbar-thumb:hover,
        .chat-messages::-webkit-scrollbar-thumb:hover {
            background: rgba(148, 163, 184, 0.55);
            background-clip: padding-box;
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
            display: none;
        }

        .sources-button {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 8px 12px;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.78);
            color: var(--text);
            text-decoration: none;
            cursor: pointer;
            font: inherit;
            margin-top: 12px;
        }

        .sources-button:hover {
            border-color: rgba(15, 118, 110, 0.25);
        }

        .sources-modal {
            position: fixed;
            inset: 0;
            display: none;
            align-items: center;
            justify-content: center;
            padding: 18px;
            background: rgba(2, 6, 23, 0.56);
            backdrop-filter: blur(12px);
            z-index: 1000;
        }

        .sources-modal.open {
            display: flex;
        }

        .sources-modal-card {
            width: min(680px, 100%);
            max-height: min(72vh, 760px);
            overflow: auto;
            padding: 22px;
            border-radius: 24px;
            background: var(--panel-strong);
            border: 1px solid var(--line);
            box-shadow: 0 30px 80px rgba(2, 6, 23, 0.35);
        }

        .sources-modal-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 16px;
        }

        .sources-modal-header h3 {
            margin: 0;
            font-size: 1.1rem;
        }

        .sources-modal-header p {
            margin: 4px 0 0;
            color: var(--muted);
            font-size: 0.92rem;
        }

        .sources-modal-close {
            appearance: none;
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.86);
            color: var(--text);
            border-radius: 999px;
            width: 36px;
            height: 36px;
            cursor: pointer;
            font: inherit;
        }

        .sources-modal-list {
            display: grid;
            gap: 10px;
        }

        .sources-modal-item {
            display: block;
            padding: 12px 14px;
            border-radius: 16px;
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.8);
            color: var(--text);
            text-decoration: none;
            word-break: break-word;
            font-size: 0.88rem;
        }

        .share-options {
            display: grid;
            gap: 10px;
            margin-top: 8px;
        }

        .share-option-btn {
            appearance: none;
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.84);
            color: var(--text);
            border-radius: 14px;
            padding: 12px 14px;
            font: inherit;
            text-align: left;
            cursor: pointer;
            transition: transform .12s ease, border-color .12s ease, box-shadow .12s ease;
        }

        .share-option-btn:hover {
            transform: translateY(-1px);
            border-color: rgba(15, 118, 110, 0.28);
            box-shadow: 0 12px 24px rgba(23, 23, 23, 0.08);
        }

        .voice-panel {
            display: grid;
            gap: 14px;
            padding: 16px;
            border: 1px solid var(--line);
            border-radius: 20px;
            background: rgba(255,255,255,0.78);
        }

        .voice-panel-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 12px;
            flex-wrap: wrap;
        }

        .voice-panel-header strong {
            display: block;
            font-size: 0.98rem;
            margin-bottom: 2px;
        }

        .voice-panel-header span,
        .voice-note,
        .voice-status {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.5;
        }

        .voice-toolbar,
        .voice-settings-grid,
        .voice-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
        }

        .voice-settings-grid {
            align-items: stretch;
        }

        .voice-mic-btn {
            min-width: 150px;
        }

        .voice-pill,
        .voice-toggle {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.84);
            color: var(--text);
            font: inherit;
            font-size: 0.9rem;
        }

        .voice-pill input,
        .voice-toggle input {
            width: 16px;
            height: 16px;
        }

        .voice-select {
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.86);
            color: var(--text);
            border-radius: 14px;
            padding: 10px 12px;
            font: inherit;
            min-width: 180px;
        }

        .voice-status-box {
            display: grid;
            gap: 8px;
            padding: 12px 14px;
            border-radius: 16px;
            background: rgba(15, 118, 110, 0.08);
            border: 1px solid rgba(15, 118, 110, 0.12);
        }

        .voice-draft {
            display: grid;
            gap: 8px;
            padding: 12px 14px;
            border-radius: 16px;
            background: rgba(255,255,255,0.72);
            border: 1px solid var(--line);
        }

        .voice-draft textarea {
            min-height: 88px;
            resize: vertical;
        }

        .voice-summary {
            display: grid;
            gap: 10px;
            margin-top: 6px;
        }

        .followup-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .followup-chip,
        .speak-claim-btn {
            appearance: none;
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.86);
            color: var(--text);
            border-radius: 999px;
            padding: 8px 12px;
            cursor: pointer;
            font: inherit;
            font-size: 0.86rem;
            transition: transform .12s ease, border-color .12s ease;
        }

        .followup-chip:hover,
        .speak-claim-btn:hover {
            transform: translateY(-1px);
            border-color: rgba(15, 118, 110, 0.28);
        }

        .voice-active {
            border-color: rgba(15, 118, 110, 0.34);
            box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.08);
        }

        body.voice-accessibility {
            font-size: 1.04rem;
        }

        body.voice-accessibility .btn,
        body.voice-accessibility .chip,
        body.voice-accessibility .followup-chip,
        body.voice-accessibility .speak-claim-btn,
        body.voice-accessibility .voice-pill,
        body.voice-accessibility .voice-toggle,
        body.voice-accessibility .voice-select {
            min-height: 46px;
        }

        body.voice-accessibility .message {
            max-width: min(860px, 96%);
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
        body[data-theme="dark"] .memory-select,
        body[data-theme="dark"] textarea,
        body[data-theme="dark"] .history-item,
        body[data-theme="dark"] .history-empty,
        body[data-theme="dark"] .empty,
        body[data-theme="dark"] .source,
        body[data-theme="dark"] .voice-panel,
        body[data-theme="dark"] .voice-status-box,
        body[data-theme="dark"] .voice-draft,
        body[data-theme="dark"] .voice-pill,
        body[data-theme="dark"] .voice-toggle,
        body[data-theme="dark"] .voice-select,
        body[data-theme="dark"] .followup-chip,
        body[data-theme="dark"] .speak-claim-btn,
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

        body[data-theme="dark"] .intro {
            background: rgba(15, 23, 42, 0.96);
        }

        body[data-theme="dark"] .eyebrow {
            background: rgba(20, 184, 166, 0.16);
            border-color: rgba(20, 184, 166, 0.22);
            color: #d1fae5;
        }

        body[data-theme="dark"] h1 {
            color: #f8fafc;
        }

        body[data-theme="dark"] .lead {
            color: #cbd5e1;
        }

        body[data-theme="dark"] .message-label,
        body[data-theme="dark"] .chat-header-meta,
        body[data-theme="dark"] .session-badge,
        body[data-theme="dark"] .sources-button,
        body[data-theme="dark"] .sources-modal-close,
        body[data-theme="dark"] .sources-modal-item,
        body[data-theme="dark"] .share-option-btn,
        body[data-theme="dark"] .session-menu-trigger,
        body[data-theme="dark"] .session-menu,
        body[data-theme="dark"] .session-menu-item,
        body[data-theme="dark"] .maker-copy span,
        body[data-theme="dark"] .maker-link {
            color: #cbd5e1;
        }

        body[data-theme="dark"] .maker-link,
        body[data-theme="dark"] .maker-avatar,
        body[data-theme="dark"] .sources-button,
        body[data-theme="dark"] .sources-modal-close,
        body[data-theme="dark"] .sources-modal-item,
        body[data-theme="dark"] .share-option-btn,
        body[data-theme="dark"] .session-menu-trigger,
        body[data-theme="dark"] .session-menu {
            background: rgba(15, 23, 42, 0.96);
            border-color: rgba(148, 163, 184, 0.18);
        }

        body[data-theme="dark"] .session-menu-item:hover {
            background: rgba(20, 184, 166, 0.12);
        }

        body[data-theme="dark"] .session-menu-item.danger {
            color: #fca5a5;
        }

        body[data-theme="dark"] .session-menu-item.danger:hover {
            background: rgba(239, 68, 68, 0.14);
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
        /* Mobile orientation and touch optimizations */
        .history-list, .chat-messages, .sources-modal-card {
            -webkit-overflow-scrolling: touch;
        }

        @media (orientation: portrait) {
            .workspace { grid-template-columns: 1fr; }
            .sidebar { position: static; max-height: none; overflow: auto; }
            .sidebar { padding: 14px; }
            .wrap { padding-top: 18px; }
            .chat-shell { padding: 16px; min-height: auto; }
            .composer textarea { min-height: 88px; }
            .history-list { max-height: 34vh; }
            .sources-modal-card { width: min(92vw, 680px); max-height: 78vh; }
            .history-item { padding: 12px; border-radius: 14px; }
            .session-menu { left: auto; right: 8px; }
        }

        @media (orientation: landscape) and (max-width: 1024px) {
            .workspace { grid-template-columns: 220px 1fr; gap: 12px; }
            .sidebar { max-height: calc(100vh - 24px); overflow: auto; }
            .chat-shell { padding: 18px; }
            .history-list { max-height: calc(100vh - 160px); }
            h1 { font-size: clamp(1.8rem, 3.2vw, 3.2rem); }
        }
    </style>
</head>
<body>
    <div class="wrap">
        <div class="workspace">
            <aside class="card panel sidebar">
                <div class="sidebar-header">
                    <div>
                        <h2>Tarka AI</h2>
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
                            <span>Building Tarka</span>
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

                        <div class="voice-panel" id="voice_panel">
                            <div class="voice-panel-header">
                                <div>
                                    <strong>Voice mode</strong>
                                    <span>Browser-native voice controls.</span>
                                </div>
                                <div class="voice-status" id="voice_status">Voice idle.</div>
                            </div>

                            <div class="voice-toolbar">
                                <button class="btn btn-primary voice-mic-btn" id="voice_toggle" type="button">Start voice</button>
                                <button class="btn btn-secondary" id="voice_stop" type="button">Stop voice</button>
                                <button class="btn btn-secondary" id="voice_play_answer" type="button">Read latest answer</button>
                            </div>

                            <div class="voice-settings-grid">
                                <label class="voice-toggle"><input id="voice_hands_free" type="checkbox" /> Hands-free</label>
                                <label class="voice-toggle"><input id="voice_auto_send" type="checkbox" /> Auto send transcript</label>
                                <label class="voice-toggle"><input id="voice_speak_answers" type="checkbox" checked /> Speak answers</label>
                                <label class="voice-toggle"><input id="voice_streaming" type="checkbox" checked /> Stream speech</label>
                                <label class="voice-toggle"><input id="voice_barge_in" type="checkbox" checked /> Barge-in</label>
                                <label class="voice-toggle"><input id="voice_source_aware" type="checkbox" checked /> Source-aware speech</label>
                                <label class="voice-toggle"><input id="voice_confidence_cues" type="checkbox" checked /> Confidence cues</label>
                                <label class="voice-toggle"><input id="voice_clarify_first" type="checkbox" checked /> Clarify first</label>
                                <label class="voice-toggle"><input id="voice_memory_commands" type="checkbox" checked /> Voice memory commands</label>
                                <label class="voice-toggle"><input id="voice_accessibility" type="checkbox" /> Accessibility mode</label>
                                <select id="voice_depth" class="voice-select" aria-label="Voice answer depth">
                                    <option value="brief">Brief answer</option>
                                    <option value="balanced" selected>Balanced answer</option>
                                    <option value="deep">Deep dive</option>
                                </select>
                                <select id="voice_lang" class="voice-select" aria-label="Voice language">
                                    <option value="auto" selected>Auto language</option>
                                    <option value="en-US">English (US)</option>
                                    <option value="en-GB">English (UK)</option>
                                    <option value="hi-IN">Hindi (India)</option>
                                </select>
                                <select id="voice_voice" class="voice-select" aria-label="Speech voice"></select>
                            </div>

                            <div class="voice-status-box">
                                <div class="voice-note" id="voice_hint">Tip: click Start voice, speak your query, edit the transcript, then press Send.</div>
                                <div class="voice-note" id="voice_confidence">Transcript confidence: not available until you speak.</div>
                            </div>

                            <div class="voice-draft">
                                <div class="voice-note">Voice transcript draft</div>
                                <textarea id="voice_draft" placeholder="Your transcript will appear here for review before sending..."></textarea>
                            </div>
                        </div>

                        <textarea id="query" placeholder="Ask a follow-up or start a new research session...">What are the best vector databases for a small production app?</textarea>

                        <div class="composer-row">
                                <div class="actions">
                                    <label class="toggle"><input id="use_memory" type="checkbox" checked /> Use memory cache</label>
                                    <select id="memory_mode" class="memory-select" aria-label="Memory mode">
                                        <option value="balanced" selected>Memory mode: Balanced</option>
                                        <option value="prefer_memory">Memory mode: Prefer memory</option>
                                        <option value="search_only">Memory mode: Search only</option>
                                    </select>
                                </div>
                            <div class="actions">
                                    <button class="btn btn-secondary" id="export_chat" type="button">Export</button>
                                    <button class="btn btn-secondary" id="share_chat" type="button">Share</button>
                                <button class="btn btn-secondary" id="clear" type="button">Clear input</button>
                                <button class="btn btn-primary" id="run" type="button">Send</button>
                            </div>
                        </div>
                    </div>
                </section>
            </div>
        </div>
    </div>

    <div class="sources-modal" id="sources_modal" aria-hidden="true">
        <div class="sources-modal-card" role="dialog" aria-modal="true" aria-labelledby="sources_modal_title">
            <div class="sources-modal-header">
                <div>
                    <h3 id="sources_modal_title">Sources</h3>
                    <p>Referenced URLs for this answer.</p>
                </div>
                <button class="sources-modal-close" id="sources_modal_close" type="button" aria-label="Close sources popup">×</button>
            </div>
            <div class="sources-modal-list" id="sources_modal_list"></div>
        </div>
    </div>

    <div class="sources-modal" id="share_modal" aria-hidden="true">
        <div class="sources-modal-card" role="dialog" aria-modal="true" aria-labelledby="share_modal_title">
            <div class="sources-modal-header">
                <div>
                    <h3 id="share_modal_title">Share</h3>
                    <p>Pick a platform. The message is copied first, then redirected.</p>
                </div>
                <button class="sources-modal-close" id="share_modal_close" type="button" aria-label="Close share popup">×</button>
            </div>
            <div class="share-options">
                <button class="share-option-btn" id="share_whatsapp" type="button">Share on WhatsApp</button>
                <button class="share-option-btn" id="share_instagram" type="button">Share on Instagram</button>
                <button class="share-option-btn" id="share_mail" type="button">Share via Email</button>
                <button class="share-option-btn" id="share_copy" type="button">Copy to Clipboard</button>
            </div>
        </div>
    </div>

    <script>
        const queryEl = document.getElementById('query');
        const useMemoryEl = document.getElementById('use_memory');
        const memoryModeEl = document.getElementById('memory_mode');
        const statusEl = document.getElementById('status');
        const runBtn = document.getElementById('run');
        const clearBtn = document.getElementById('clear');
        const exportBtn = document.getElementById('export_chat');
        const shareBtn = document.getElementById('share_chat');
        const sessionListEl = document.getElementById('history_list');
        const newSessionBtn = document.getElementById('new_session');
        const sessionBadgeEl = document.getElementById('session_badge');
        const chatMessagesEl = document.getElementById('chat_messages');
        const sourcesModalEl = document.getElementById('sources_modal');
        const sourcesModalListEl = document.getElementById('sources_modal_list');
        const sourcesModalCloseEl = document.getElementById('sources_modal_close');
        const shareModalEl = document.getElementById('share_modal');
        const shareModalCloseEl = document.getElementById('share_modal_close');
        const shareWhatsAppEl = document.getElementById('share_whatsapp');
        const shareInstagramEl = document.getElementById('share_instagram');
        const shareMailEl = document.getElementById('share_mail');
        const shareCopyEl = document.getElementById('share_copy');
        const themeToggleEl = document.getElementById('theme_toggle');
        const voicePanelEl = document.getElementById('voice_panel');
        const voiceToggleEl = document.getElementById('voice_toggle');
        const voiceStopEl = document.getElementById('voice_stop');
        const voicePlayAnswerEl = document.getElementById('voice_play_answer');
        const voiceHandsFreeEl = document.getElementById('voice_hands_free');
        const voiceAutoSendEl = document.getElementById('voice_auto_send');
        const voiceSpeakAnswersEl = document.getElementById('voice_speak_answers');
        const voiceStreamingEl = document.getElementById('voice_streaming');
        const voiceBargeInEl = document.getElementById('voice_barge_in');
        const voiceSourceAwareEl = document.getElementById('voice_source_aware');
        const voiceConfidenceCuesEl = document.getElementById('voice_confidence_cues');
        const voiceClarifyFirstEl = document.getElementById('voice_clarify_first');
        const voiceMemoryCommandsEl = document.getElementById('voice_memory_commands');
        const voiceAccessibilityEl = document.getElementById('voice_accessibility');
        const voiceDepthEl = document.getElementById('voice_depth');
        const voiceLangEl = document.getElementById('voice_lang');
        const voiceVoiceEl = document.getElementById('voice_voice');
        const voiceDraftEl = document.getElementById('voice_draft');
        const voiceStatusEl = document.getElementById('voice_status');
        const voiceHintEl = document.getElementById('voice_hint');
        const voiceConfidenceEl = document.getElementById('voice_confidence');
        const HISTORY_KEY = 'tarka-chat-sessions';
        const ACTIVE_SESSION_KEY = 'tarka-active-session';
        const THEME_KEY = 'research-theme';
        const VOICE_KEY = 'tarka-voice-settings';
        const MAX_SESSIONS = 20;
        const MAX_CONTEXT_MESSAGES = 8;
        const SpeechRecognitionCtor = window.SpeechRecognition || window.webkitSpeechRecognition || null;

        // List of explicit words to flag in assistant messages. Case-insensitive, matched as whole words.
        const EXPLICIT_WORDS = [
            'fuck', 'shit', 'bitch', 'asshole', 'motherfucker', 'damn', 'crap'
        ];

        const containsExplicit = (text) => {
            if (!text) return false;
            const lower = text.toLowerCase();
            return EXPLICIT_WORDS.some((w) => new RegExp('\\b' + w.replace(/[-\\/\\^$*+?.()|[\\]{}]/g, '\\$&') + '\\b', 'i').test(lower));
        };

        let sessions = [];
        let activeSessionId = '';
        let activeStream = null;
        let activeAssistantMessageId = null;
        let activeRequestId = null;
        let activeShareText = '';
        let speechRecognition = null;
        let speechRecognitionActive = false;
        let speechRecognitionTranscript = '';
        let speechRecognitionInterim = '';
        let speechSynthesisVoice = null;
        let speechQueue = [];
        let speechQueueActive = false;
        let speechStreamBuffer = '';
        let lastVoiceTranscriptConfidence = null;

        const defaultVoiceSettings = {
            handsFree: false,
            autoSend: false,
            speakAnswers: true,
            streamingVoice: true,
            bargeIn: true,
            sourceAware: true,
            confidenceCues: true,
            clarifyFirst: true,
            memoryCommands: true,
            accessibility: false,
            depth: 'balanced',
            lang: 'auto',
            voiceURI: '',
        };

        let voiceSettings = { ...defaultVoiceSettings };

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
                const starter = createSession();
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

        function createSession(title) {
            const id = crypto.randomUUID ? crypto.randomUUID() : String(Date.now());
            // If no explicit title provided, generate a sequential Session #n title
            const nextIndex = sessions && sessions.length ? sessions.length + 1 : 1;
            const finalTitle = title ? String(title) : `Session ${nextIndex}`;
            return {
                id,
                title: finalTitle,
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

        const formatPct = (value) => `${Math.round((Number(value) || 0) * 100)}%`;

        const loadVoiceSettings = () => {
            try {
                const raw = localStorage.getItem(VOICE_KEY);
                if (!raw) return { ...defaultVoiceSettings };
                const parsed = JSON.parse(raw);
                return { ...defaultVoiceSettings, ...(parsed || {}) };
            } catch {
                return { ...defaultVoiceSettings };
            }
        };

        const saveVoiceSettings = () => {
            localStorage.setItem(VOICE_KEY, JSON.stringify(voiceSettings));
        };

        const setVoiceStatus = (text) => {
            voiceStatusEl.textContent = text;
        };

        const applyVoiceAccessibility = () => {
            document.body.classList.toggle('voice-accessibility', !!voiceSettings.accessibility);
        };

        const syncVoiceUI = () => {
            voiceHandsFreeEl.checked = !!voiceSettings.handsFree;
            voiceAutoSendEl.checked = !!voiceSettings.autoSend;
            voiceSpeakAnswersEl.checked = !!voiceSettings.speakAnswers;
            voiceStreamingEl.checked = !!voiceSettings.streamingVoice;
            voiceBargeInEl.checked = !!voiceSettings.bargeIn;
            voiceSourceAwareEl.checked = !!voiceSettings.sourceAware;
            voiceConfidenceCuesEl.checked = !!voiceSettings.confidenceCues;
            voiceClarifyFirstEl.checked = !!voiceSettings.clarifyFirst;
            voiceMemoryCommandsEl.checked = !!voiceSettings.memoryCommands;
            voiceAccessibilityEl.checked = !!voiceSettings.accessibility;
            voiceDepthEl.value = voiceSettings.depth || 'balanced';
            voiceLangEl.value = voiceSettings.lang || 'auto';
            applyVoiceAccessibility();
        };

        const refreshVoiceVoiceList = () => {
            if (!window.speechSynthesis) {
                voiceVoiceEl.innerHTML = '<option value="">Speech synthesis unavailable</option>';
                voiceVoiceEl.disabled = true;
                return;
            }

            const voices = window.speechSynthesis.getVoices();
            const options = ['<option value="">Default voice</option>'];
            voices.forEach((voice) => {
                const label = `${voice.name} (${voice.lang})`;
                options.push(`<option value="${String(voice.voiceURI).replace(/"/g, '&quot;')}">${label}</option>`);
            });
            voiceVoiceEl.innerHTML = options.join('');
            voiceVoiceEl.disabled = voices.length === 0;
            if (voiceSettings.voiceURI) {
                voiceVoiceEl.value = voiceSettings.voiceURI;
            }
        };

        const getRecognitionLanguage = () => {
            if (voiceSettings.lang && voiceSettings.lang !== 'auto') return voiceSettings.lang;
            return navigator.language || 'en-US';
        };

        const chooseSpeechVoice = () => {
            if (!window.speechSynthesis) return null;
            const voices = window.speechSynthesis.getVoices();
            if (!voices.length) return null;

            if (voiceSettings.voiceURI) {
                const selected = voices.find((voice) => voice.voiceURI === voiceSettings.voiceURI);
                if (selected) return selected;
            }

            const targetLang = getRecognitionLanguage().toLowerCase();
            return voices.find((voice) => voice.lang && voice.lang.toLowerCase() === targetLang)
                || voices.find((voice) => voice.lang && voice.lang.toLowerCase().startsWith(targetLang.split('-')[0]))
                || voices[0];
        };

        const cancelVoicePlayback = () => {
            if (window.speechSynthesis) {
                window.speechSynthesis.cancel();
            }
            speechQueue = [];
            speechQueueActive = false;
            speechStreamBuffer = '';
        };

        const enqueueSpeech = (text, { immediate = false } = {}) => {
            const cleaned = (text || '').replace(/\s+/g, ' ').trim();
            if (!cleaned) return;
            speechQueue.push(cleaned);
            if (immediate) {
                cancelVoicePlayback();
                speechQueue.push(cleaned);
            }
            processSpeechQueue();
        };

        const processSpeechQueue = () => {
            if (!window.speechSynthesis || !voiceSettings.speakAnswers || speechQueueActive) return;
            const next = speechQueue.shift();
            if (!next) return;

            const utterance = new SpeechSynthesisUtterance(next);
            utterance.lang = getRecognitionLanguage();
            utterance.rate = voiceSettings.depth === 'brief' ? 1.06 : voiceSettings.depth === 'deep' ? 0.96 : 1.0;
            utterance.pitch = 1;
            utterance.voice = chooseSpeechVoice();
            speechQueueActive = true;

            utterance.onend = () => {
                speechQueueActive = false;
                processSpeechQueue();
                if (!speechQueue.length && voiceSettings.handsFree && !speechRecognitionActive) {
                    startVoiceRecognition({ autoSend: false, restart: true });
                }
            };

            utterance.onerror = () => {
                speechQueueActive = false;
                processSpeechQueue();
            };

            window.speechSynthesis.speak(utterance);
        };

        const flushSpeechStream = (force = false) => {
            if (!voiceSettings.streamingVoice || !voiceSettings.speakAnswers || !speechStreamBuffer.trim()) return;

            const buffer = speechStreamBuffer;
            const sentenceMatch = buffer.match(/^(.+?[.!?](?:\s|$))/s);
            if (sentenceMatch) {
                enqueueSpeech(sentenceMatch[1]);
                speechStreamBuffer = buffer.slice(sentenceMatch[1].length);
                return;
            }

            if (force || buffer.length > 170) {
                const chunk = force ? buffer : buffer.slice(0, 170);
                enqueueSpeech(chunk);
                speechStreamBuffer = force ? '' : buffer.slice(170);
            }
        };

        const speakText = (text, { interrupt = false, force = false } = {}) => {
            if ((!voiceSettings.speakAnswers && !force) || !window.speechSynthesis) return;
            if (interrupt) {
                cancelVoicePlayback();
            }
            enqueueSpeech(text, { immediate: interrupt });
        };

        const buildVoicePromptPrefix = () => {
            const parts = [];
            if (voiceSettings.depth === 'brief') parts.push('Answer briefly in 3 to 5 sentences.');
            if (voiceSettings.depth === 'deep') parts.push('Answer in detail with tradeoffs, caveats, and next steps.');
            if (voiceSettings.sourceAware) parts.push('Be source-aware and mention evidence quality clearly.');
            if (voiceSettings.confidenceCues) parts.push('State confidence when evidence is strong, medium, or weak.');
            if (!parts.length) return '';
            return `Voice response instructions: ${parts.join(' ')}`;
        };

        const looksLikeNeedsClarification = (text) => {
            const cleaned = (text || '').trim();
            const words = cleaned.split(/\s+/).filter(Boolean);
            if (words.length < 6) return true;
            return /\b(something|stuff|thing|whatever|it|this|that)\b/i.test(cleaned) && words.length < 10;
        };

        const applyVoiceCommands = (rawText) => {
            let text = (rawText || '').trim();
            if (!voiceSettings.memoryCommands || !text) {
                return { text, clarified: false };
            }

            const replacements = [
                { pattern: /\b(use fresh search|search only|ignore memory)\b/gi, action: () => { memoryModeEl.value = 'search_only'; } },
                { pattern: /\b(prefer memory|use memory)\b/gi, action: () => { memoryModeEl.value = 'prefer_memory'; } },
                { pattern: /\b(balanced memory|balanced mode)\b/gi, action: () => { memoryModeEl.value = 'balanced'; } },
                { pattern: /\b(brief answer|keep it brief|short answer)\b/gi, action: () => { voiceDepthEl.value = 'brief'; voiceSettings.depth = 'brief'; } },
                { pattern: /\b(deep dive|detailed answer|full answer)\b/gi, action: () => { voiceDepthEl.value = 'deep'; voiceSettings.depth = 'deep'; } },
                { pattern: /\b(auto send|send automatically)\b/gi, action: () => { voiceAutoSendEl.checked = true; } },
                { pattern: /\b(no auto send|don't auto send|do not auto send)\b/gi, action: () => { voiceAutoSendEl.checked = false; } },
            ];

            replacements.forEach(({ pattern, action }) => {
                if (pattern.test(text)) {
                    action();
                    text = text.replace(pattern, ' ');
                }
            });

            text = text.replace(/\s+/g, ' ').trim();
            return { text, clarified: false };
        };

        const updateVoiceSettingsFromUI = () => {
            voiceSettings = {
                ...voiceSettings,
                handsFree: voiceHandsFreeEl.checked,
                autoSend: voiceAutoSendEl.checked,
                speakAnswers: voiceSpeakAnswersEl.checked,
                streamingVoice: voiceStreamingEl.checked,
                bargeIn: voiceBargeInEl.checked,
                sourceAware: voiceSourceAwareEl.checked,
                confidenceCues: voiceConfidenceCuesEl.checked,
                clarifyFirst: voiceClarifyFirstEl.checked,
                memoryCommands: voiceMemoryCommandsEl.checked,
                accessibility: voiceAccessibilityEl.checked,
                depth: voiceDepthEl.value,
                lang: voiceLangEl.value,
                voiceURI: voiceVoiceEl.value,
            };
            saveVoiceSettings();
            syncVoiceUI();
        };

        const initializeVoiceSettings = () => {
            voiceSettings = loadVoiceSettings();
            syncVoiceUI();
            refreshVoiceVoiceList();
            setVoiceStatus(SpeechRecognitionCtor ? 'Voice ready.' : 'Voice is unavailable in this browser.');
            voiceHintEl.textContent = SpeechRecognitionCtor
                ? 'Tip: click Start voice, speak your query, edit the transcript, then press Send.'
                : 'Speech recognition is not available here. You can still use TTS if supported.';
            voiceToggleEl.disabled = !SpeechRecognitionCtor;
            voiceStopEl.disabled = !SpeechRecognitionCtor;
            voicePlayAnswerEl.disabled = !window.speechSynthesis;
        };

        const speakEvidenceForClaim = (claim) => {
            if (!claim) return;
            const lines = [
                `Claim: ${claim.claim || 'No claim text.'}`,
                claim.source ? `Source: ${claim.source}.` : '',
                claim.evidence_snippet ? `Evidence: ${claim.evidence_snippet}` : '',
            ].filter(Boolean);
            speakText(lines.join(' '), { interrupt: true, force: true });
        };

        const buildFollowUpSuggestions = (message) => {
            const suggestions = [];
            if (!message) return suggestions;
            if (message.claims && message.claims.length) {
                const firstClaim = message.claims[0];
                suggestions.push(`Explain the strongest claim: ${firstClaim.claim || 'claim 1'}`);
                suggestions.push(`Show evidence for: ${firstClaim.source || 'claim 1'}`);
            }
            if (message.source_urls && message.source_urls.length) {
                suggestions.push('Compare this answer with an alternative source');
            }
            suggestions.push('Give me a shorter summary');
            return [...new Set(suggestions)].slice(0, 3);
        };

        const getLatestAssistantMessage = () => {
            const session = getActiveSession();
            if (!session) return null;
            return [...session.messages].reverse().find((message) => message.role === 'assistant');
        };

        const buildAnswerSpeech = (payload, { includeAnswer = true } = {}) => {
            const answer = (payload?.final_answer || '').trim();
            const parts = includeAnswer && answer ? [answer] : [];

            if (voiceSettings.sourceAware) {
                const coverage = typeof payload?.evidence_coverage === 'number' ? Math.round(payload.evidence_coverage * 100) : null;
                const confidence = typeof payload?.avg_confidence === 'number' ? Math.round(payload.avg_confidence * 100) : null;
                const evidenceParts = [];
                if (coverage !== null) evidenceParts.push(`Evidence coverage ${coverage} percent.`);
                if (confidence !== null && voiceSettings.confidenceCues) evidenceParts.push(`Average confidence ${confidence} percent.`);
                if (evidenceParts.length) {
                    parts.push(evidenceParts.join(' '));
                }
            }

            const followUps = buildFollowUpSuggestions({
                claims: payload?.claims || [],
                source_urls: payload?.source_urls || [],
            });
            if (followUps.length) {
                parts.push(`Suggested follow-ups: ${followUps.join('; ')}.`);
            }

            return parts.filter(Boolean).join(' ').replace(/\s+/g, ' ').trim();
        };

        const stopVoiceRecognition = () => {
            if (speechRecognition) {
                try {
                    speechRecognition.onstart = null;
                    speechRecognition.onresult = null;
                    speechRecognition.onerror = null;
                    speechRecognition.onend = null;
                    speechRecognition.abort();
                } catch {
                    // ignore
                }
                speechRecognition = null;
            }
            speechRecognitionActive = false;
            voicePanelEl.classList.remove('voice-active');
            voiceToggleEl.textContent = 'Start voice';
            if (!voiceSettings.speakAnswers) {
                setVoiceStatus('Voice idle.');
            }
        };

        function startVoiceRecognition({ autoSend = false } = {}) {
            if (!SpeechRecognitionCtor) {
                setVoiceStatus('Speech recognition is unavailable in this browser.');
                return;
            }

            if (speechRecognitionActive) {
                stopVoiceRecognition();
                return;
            }

            if (voiceSettings.bargeIn) {
                cancelVoicePlayback();
            }

            speechRecognitionTranscript = '';
            speechRecognitionInterim = '';
            lastVoiceTranscriptConfidence = null;
            voiceDraftEl.value = '';
            voiceConfidenceEl.textContent = 'Transcript confidence: waiting for speech.';

            speechRecognition = new SpeechRecognitionCtor();
            speechRecognition.continuous = false;
            speechRecognition.interimResults = true;
            speechRecognition.maxAlternatives = 3;
            speechRecognition.lang = getRecognitionLanguage();

            speechRecognition.onstart = () => {
                speechRecognitionActive = true;
                voicePanelEl.classList.add('voice-active');
                voiceToggleEl.textContent = 'Stop voice';
                setVoiceStatus('Listening for your query...');
            };

            speechRecognition.onresult = (event) => {
                let interim = '';
                let finalText = '';
                let confidence = lastVoiceTranscriptConfidence || 0;

                for (let index = event.resultIndex; index < event.results.length; index += 1) {
                    const result = event.results[index];
                    const transcript = result[0]?.transcript || '';
                    const resultConfidence = typeof result[0]?.confidence === 'number' ? result[0].confidence : 0;
                    confidence = Math.max(confidence, resultConfidence);

                    if (result.isFinal) {
                        finalText += `${transcript} `;
                    } else {
                        interim += `${transcript} `;
                    }
                }

                if (finalText.trim()) {
                    speechRecognitionTranscript = `${speechRecognitionTranscript} ${finalText}`.trim();
                }

                speechRecognitionInterim = interim.trim();
                const displayText = `${speechRecognitionTranscript} ${speechRecognitionInterim}`.replace(/\s+/g, ' ').trim();
                voiceDraftEl.value = displayText;
                queryEl.value = displayText;
                lastVoiceTranscriptConfidence = confidence;
                voiceConfidenceEl.textContent = confidence
                    ? `Transcript confidence: ${Math.round(confidence * 100)}%`
                    : 'Transcript confidence: unavailable.';
            };

            speechRecognition.onerror = (event) => {
                speechRecognitionActive = false;
                voicePanelEl.classList.remove('voice-active');
                voiceToggleEl.textContent = 'Start voice';
                setVoiceStatus(`Voice error: ${event.error || 'unknown error'}.`);
                speechRecognition = null;
            };

            speechRecognition.onend = () => {
                const transcript = `${speechRecognitionTranscript} ${speechRecognitionInterim}`.replace(/\s+/g, ' ').trim();
                speechRecognitionActive = false;
                voicePanelEl.classList.remove('voice-active');
                voiceToggleEl.textContent = 'Start voice';
                speechRecognition = null;

                if (!transcript) {
                    setVoiceStatus('Voice idle.');
                    return;
                }

                const commandResult = applyVoiceCommands(transcript);
                updateVoiceSettingsFromUI();
                const cleanedTranscript = commandResult.text || transcript;
                voiceDraftEl.value = cleanedTranscript;
                queryEl.value = cleanedTranscript;

                if (voiceSettings.clarifyFirst && looksLikeNeedsClarification(cleanedTranscript)) {
                    const clarification = 'I need a bit more detail before I research this. Please add the topic, target audience, or specific comparison you want.';
                    setStatus('Voice captured, but it needs a little more detail.');
                    setVoiceStatus('Need a bit more detail before research.');
                    speakText(clarification, { interrupt: true });
                    return;
                }

                setVoiceStatus(autoSend || voiceSettings.autoSend ? 'Transcript captured. Sending research request...' : 'Transcript captured. Edit it, then press Send.');
                setStatus('Voice transcript ready.');

                if (autoSend || voiceSettings.autoSend) {
                    sendMessage();
                    return;
                }

                if (voiceSettings.handsFree && !speechQueue.length) {
                    setVoiceStatus('Hands-free mode ready for the next command.');
                }
            };

            try {
                speechRecognition.start();
            } catch (error) {
                setVoiceStatus(`Could not start speech recognition: ${error.message || 'unknown error'}.`);
                stopVoiceRecognition();
            }
        }

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

        const openSourcesModal = (payload) => {
            const urls = uniqueSourceUrls(payload?.source_urls || []);
            const claims = Array.isArray(payload?.claims) ? payload.claims : [];
            sourcesModalListEl.innerHTML = '';

            if (claims.length) {
                claims.forEach((claim, idx) => {
                    const card = document.createElement('div');
                    card.className = 'source';

                    const title = document.createElement('strong');
                    title.textContent = `Claim ${idx + 1}: ${claim.claim || 'Untitled claim'}`;
                    card.appendChild(title);

                    const meta = document.createElement('div');
                    meta.className = 'status';
                    meta.textContent = `${claim.source || 'unknown source'} | confidence ${formatPct(claim.confidence_score || 0)}`;
                    card.appendChild(meta);

                    if (claim.evidence_snippet) {
                        const snippet = document.createElement('div');
                        snippet.className = 'message-content';
                        snippet.textContent = `Evidence: ${claim.evidence_snippet}`;
                        card.appendChild(snippet);
                    }

                    const speakClaimBtn = document.createElement('button');
                    speakClaimBtn.type = 'button';
                    speakClaimBtn.className = 'speak-claim-btn';
                    speakClaimBtn.textContent = 'Speak claim';
                    speakClaimBtn.addEventListener('click', () => speakEvidenceForClaim(claim));
                    card.appendChild(speakClaimBtn);

                    if (claim.source_url) {
                        const link = document.createElement('a');
                        link.className = 'sources-modal-item';
                        link.href = claim.source_url;
                        link.target = '_blank';
                        link.rel = 'noreferrer';
                        link.textContent = claim.source_url;
                        card.appendChild(link);
                    }

                    sourcesModalListEl.appendChild(card);
                });
            }

            if (!urls.length) {
                const empty = document.createElement('div');
                empty.className = 'history-empty';
                empty.textContent = 'No source URLs available for this answer.';
                sourcesModalListEl.appendChild(empty);
            } else {
                urls.forEach((url) => {
                    const link = document.createElement('a');
                    link.className = 'sources-modal-item';
                    link.href = url;
                    link.target = '_blank';
                    link.rel = 'noreferrer';
                    link.textContent = url;
                    sourcesModalListEl.appendChild(link);
                });
            }

            sourcesModalEl.classList.add('open');
            sourcesModalEl.setAttribute('aria-hidden', 'false');
        };

        const closeSourcesModal = () => {
            sourcesModalEl.classList.remove('open');
            sourcesModalEl.setAttribute('aria-hidden', 'true');
        };

        const openShareModal = (shareText) => {
            activeShareText = shareText || '';
            shareModalEl.classList.add('open');
            shareModalEl.setAttribute('aria-hidden', 'false');
        };

        const closeShareModal = () => {
            shareModalEl.classList.remove('open');
            shareModalEl.setAttribute('aria-hidden', 'true');
        };

        const buildShareText = (session) => {
            const lastAssistant = [...(session?.messages || [])].reverse().find((m) => m.role === 'assistant');
            return [
                `Tarka session: ${session?.title || 'Session'}`,
                lastAssistant?.content || 'No assistant response yet.',
                ...(lastAssistant?.source_urls || []).slice(0, 5),
            ].join('\n\n');
        };

        const copyShareText = async (text) => {
            if (!text) return false;
            try {
                await navigator.clipboard.writeText(text);
                return true;
            } catch {
                return false;
            }
        };

        const openExternal = (url) => {
            const win = window.open(url, '_blank', 'noopener,noreferrer');
            if (!win) {
                setStatus('Popup blocked by browser. Allow popups and try again.');
                return false;
            }
            return true;
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
                const row = document.createElement('div');
                row.className = 'history-item' + (session.id === activeSessionId ? ' active' : '');
                row.setAttribute('role', 'button');
                row.setAttribute('tabindex', '0');

                const title = document.createElement('strong');
                title.textContent = session.title || 'Untitled session';

                const preview = document.createElement('span');
                preview.textContent = `${session.messages.length} message${session.messages.length === 1 ? '' : 's'}`;

                const timestamp = document.createElement('span');
                timestamp.textContent = formatTimestamp(session.updated_at || session.created_at);

                const menuBackdrop = document.createElement('div');
                menuBackdrop.className = 'session-menu-backdrop';

                const menu = document.createElement('div');
                menu.className = 'session-menu';

                const toggleMenu = (forceOpen = null) => {
                    const isOpen = forceOpen === null ? !menu.classList.contains('open') : forceOpen;
                    menu.classList.toggle('open', isOpen);
                    menuBackdrop.classList.toggle('open', isOpen);
                };

                const renameItem = document.createElement('button');
                renameItem.type = 'button';
                renameItem.className = 'session-menu-item';
                renameItem.textContent = 'Rename';
                renameItem.addEventListener('click', (event) => {
                    event.stopPropagation();
                    toggleMenu(false);
                    const currentTitle = session.title || 'Untitled session';
                    const nextTitle = window.prompt('Rename this session', currentTitle);
                    if (nextTitle === null) return;

                    const trimmedTitle = nextTitle.trim();
                    if (!trimmedTitle) return;

                    session.title = trimmedTitle;
                    session.updated_at = nowIso();
                    saveSessions();
                    renderSessions();
                    renderMessages();
                    setStatus(`Renamed session to "${trimmedTitle}".`);
                });

                const deleteItem = document.createElement('button');
                deleteItem.type = 'button';
                deleteItem.className = 'session-menu-item danger';
                deleteItem.textContent = 'Delete';
                deleteItem.addEventListener('click', (event) => {
                    event.stopPropagation();
                    toggleMenu(false);
                    const confirmDelete = window.confirm(`Delete session \"${session.title || 'Untitled session'}\"?`);
                    if (!confirmDelete) return;

                    sessions = sessions.filter((entry) => entry.id !== session.id);
                    if (!sessions.length) {
                        const starter = createSession();
                        sessions = [starter];
                        activeSessionId = starter.id;
                    } else if (activeSessionId === session.id) {
                        activeSessionId = sessions[0].id;
                    }

                    saveSessions();
                    renderSessions();
                    renderMessages();
                    setStatus('Session deleted.');
                });

                menu.append(renameItem, deleteItem);

                const actions = document.createElement('div');
                actions.className = 'history-item-actions';

                const menuButton = document.createElement('button');
                menuButton.type = 'button';
                menuButton.className = 'session-menu-trigger';
                menuButton.textContent = '⋯';
                menuButton.setAttribute('aria-label', 'Session options');
                menuButton.addEventListener('click', (event) => {
                    event.stopPropagation();
                    toggleMenu();
                });

                actions.append(menuButton, menu, menuBackdrop);
                row.append(title, preview, timestamp, actions);
                row.addEventListener('click', () => setActiveSession(session.id));
                row.addEventListener('keydown', (event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault();
                        setActiveSession(session.id);
                    }
                });
                sessionListEl.appendChild(row);
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

                // Flag explicit content in assistant messages
                if (message.role === 'assistant' && containsExplicit(message.content)) {
                    const flag = document.createElement('div');
                    flag.className = 'explicit-flag';
                    flag.textContent = 'Explicit content flagged';
                    bubble.appendChild(flag);
                }

                if (message.role === 'assistant' && ((message.source_urls && message.source_urls.length) || (message.claims && message.claims.length))) {
                    const sourcesButton = document.createElement('button');
                    sourcesButton.type = 'button';
                    sourcesButton.className = 'sources-button';
                    const claimCount = Array.isArray(message.claims) ? message.claims.length : 0;
                    const sourceCount = uniqueSourceUrls(message.source_urls).length;
                    sourcesButton.textContent = `Evidence (${claimCount} claims, ${sourceCount} URLs)`;
                    sourcesButton.addEventListener('click', () => openSourcesModal({ source_urls: message.source_urls, claims: message.claims }));
                    bubble.appendChild(sourcesButton);
                }

                if (message.role === 'assistant' && (typeof message.evidence_coverage === 'number' || typeof message.avg_confidence === 'number')) {
                    const metrics = document.createElement('div');
                    metrics.className = 'message-metrics';

                    if (typeof message.evidence_coverage === 'number') {
                        const coverage = document.createElement('span');
                        coverage.className = 'metric-pill';
                        coverage.textContent = `Evidence coverage: ${formatPct(message.evidence_coverage)}`;
                        metrics.appendChild(coverage);
                    }

                    if (typeof message.avg_confidence === 'number') {
                        const confidence = document.createElement('span');
                        confidence.className = 'metric-pill';
                        confidence.textContent = `Avg confidence: ${formatPct(message.avg_confidence)}`;
                        metrics.appendChild(confidence);
                    }

                    bubble.appendChild(metrics);
                }

                if (message.role === 'assistant') {
                    const suggestions = buildFollowUpSuggestions(message);
                    if (suggestions.length) {
                        const followUps = document.createElement('div');
                        followUps.className = 'voice-summary';

                        const label = document.createElement('div');
                        label.className = 'voice-note';
                        label.textContent = 'Suggested follow-ups';
                        followUps.appendChild(label);

                        const chips = document.createElement('div');
                        chips.className = 'followup-chips';
                        suggestions.forEach((suggestion) => {
                            const chip = document.createElement('button');
                            chip.type = 'button';
                            chip.className = 'followup-chip';
                            chip.textContent = suggestion;
                            chip.addEventListener('click', () => {
                                queryEl.value = suggestion;
                                queryEl.focus();
                                setStatus('Follow-up loaded into the composer.');
                            });
                            chips.appendChild(chip);
                        });
                        followUps.appendChild(chips);
                        bubble.appendChild(followUps);
                    }
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
                memory_mode: memoryModeEl.value,
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
                    if (voiceSettings.speakAnswers && voiceSettings.streamingVoice) {
                        speechStreamBuffer += payload.data?.delta || '';
                        flushSpeechStream(false);
                    }
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
                    if (voiceSettings.speakAnswers) {
                        if (!voiceSettings.streamingVoice) {
                            speakText(buildAnswerSpeech(payload.data || {}), { interrupt: true });
                        } else {
                            flushSpeechStream(true);
                                const closingSpeech = buildAnswerSpeech(payload.data || {}, { includeAnswer: false });
                                if (closingSpeech) {
                                    enqueueSpeech(closingSpeech);
                                }
                        }
                    }
                    updateActiveAssistant({
                        source_urls: payload.data?.source_urls || [],
                        claims: payload.data?.claims || [],
                        evidence_coverage: payload.data?.evidence_coverage,
                        avg_confidence: payload.data?.avg_confidence,
                    });
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
            const voiceDraft = voiceDraftEl.value.trim();
            const query = (voiceDraft || queryEl.value || '').trim();
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
            const voiceInstruction = buildVoicePromptPrefix();
            const researchQuery = voiceInstruction ? `${query}\n\n${voiceInstruction}` : query;
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
                claims: [],
                evidence_coverage: null,
                avg_confidence: null,
                isStreaming: true,
                created_at: nowIso(),
            };

            session.messages.push(userMessage, assistantMessage);
            // Preserve auto-generated "Session #n" titles; only replace if title is explicitly 'New session' or empty
            if (!session.title || session.title === 'New session') {
                session.title = shortPreview(query, 42);
            }
            session.updated_at = nowIso();
            activeAssistantMessageId = assistantMessage.id;
            activeRequestId = null;
            saveSessions();
            renderSessions();
            renderMessages();

            queryEl.value = '';
            voiceDraftEl.value = '';
            runBtn.disabled = true;
            runBtn.textContent = 'Streaming...';
            setStatus('Streaming the answer into the session.');

            speechStreamBuffer = '';
            if (voiceSettings.bargeIn) {
                cancelVoicePlayback();
            }

            try {
                const finalPayload = await streamAnswer({
                    query: researchQuery,
                    context: conversationContext,
                    useMemory: useMemoryEl.checked,
                });

                updateActiveAssistant({
                    content: finalPayload.final_answer || getActiveSession()?.messages.find((message) => message.id === activeAssistantMessageId)?.content || 'No answer generated.',
                    source_urls: finalPayload.source_urls || getActiveSession()?.messages.find((message) => message.id === activeAssistantMessageId)?.source_urls || [],
                    claims: finalPayload.claims || [],
                    evidence_coverage: typeof finalPayload.evidence_coverage === 'number' ? finalPayload.evidence_coverage : null,
                    avg_confidence: typeof finalPayload.avg_confidence === 'number' ? finalPayload.avg_confidence : null,
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
        initializeVoiceSettings();
        renderSessions();
        renderMessages();

        if (window.speechSynthesis) {
            window.speechSynthesis.onvoiceschanged = () => {
                refreshVoiceVoiceList();
            };
        }

        document.querySelectorAll('[data-query]').forEach((button) => {
            button.addEventListener('click', () => {
                queryEl.value = button.dataset.query;
                queryEl.focus();
            });
        });

        newSessionBtn.addEventListener('click', startNewSession);
        sourcesModalCloseEl.addEventListener('click', closeSourcesModal);
        sourcesModalEl.addEventListener('click', (event) => {
            if (event.target === sourcesModalEl) {
                closeSourcesModal();
            }
        });
        shareModalCloseEl.addEventListener('click', closeShareModal);
        shareModalEl.addEventListener('click', (event) => {
            if (event.target === shareModalEl) {
                closeShareModal();
            }
        });

        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') {
                closeSourcesModal();
                closeShareModal();
            }
        });

        clearBtn.addEventListener('click', () => {
            queryEl.value = '';
            queryEl.focus();
            setStatus('Input cleared.');
        });

        exportBtn.addEventListener('click', () => {
            const session = getActiveSession();
            if (!session || !session.messages.length) {
                setStatus('No messages to export.');
                return;
            }

            const payload = {
                title: session.title,
                exported_at: nowIso(),
                messages: session.messages.map((m) => ({
                    role: m.role,
                    content: m.content,
                    source_urls: m.source_urls || [],
                    claims: m.claims || [],
                    evidence_coverage: m.evidence_coverage,
                    avg_confidence: m.avg_confidence,
                    created_at: m.created_at,
                })),
            };

            const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const anchor = document.createElement('a');
            anchor.href = url;
            anchor.download = `${(session.title || 'session').replace(/[^a-z0-9-_]+/gi, '_').toLowerCase()}.json`;
            document.body.appendChild(anchor);
            anchor.click();
            anchor.remove();
            URL.revokeObjectURL(url);
            setStatus('Session exported as JSON.');
        });

        shareBtn.addEventListener('click', async () => {
            const session = getActiveSession();
            if (!session || !session.messages.length) {
                setStatus('No messages to share.');
                return;
            }

            openShareModal(buildShareText(session));
            setStatus('Choose a platform to share.');
        });

        shareCopyEl.addEventListener('click', async () => {
            const copied = await copyShareText(activeShareText);
            setStatus(copied ? 'Share summary copied to clipboard.' : 'Clipboard access failed.');
            closeShareModal();
        });

        shareWhatsAppEl.addEventListener('click', async () => {
            const copied = await copyShareText(activeShareText);
            const shareUrl = `https://wa.me/?text=${encodeURIComponent(activeShareText)}`;
            if (openExternal(shareUrl)) {
                setStatus(copied ? 'Copied and opened WhatsApp.' : 'Opened WhatsApp. Paste if needed.');
                closeShareModal();
            }
        });

        const syncVoiceDraftToQuery = () => {
            queryEl.value = voiceDraftEl.value;
        };

        voiceToggleEl.addEventListener('click', () => {
            if (speechRecognitionActive) {
                stopVoiceRecognition();
                setVoiceStatus('Voice stopped.');
                return;
            }

            updateVoiceSettingsFromUI();
            startVoiceRecognition({ autoSend: voiceSettings.autoSend });
        });

        voiceStopEl.addEventListener('click', () => {
            stopVoiceRecognition();
            cancelVoicePlayback();
            setVoiceStatus('Voice stopped.');
        });

        voicePlayAnswerEl.addEventListener('click', () => {
            const latestAssistant = getLatestAssistantMessage();
            if (!latestAssistant || !latestAssistant.content) {
                setStatus('No answer available to read yet.');
                return;
            }

            speakText(buildAnswerSpeech({
                final_answer: latestAssistant.content,
                source_urls: latestAssistant.source_urls || [],
                claims: latestAssistant.claims || [],
                evidence_coverage: latestAssistant.evidence_coverage,
                avg_confidence: latestAssistant.avg_confidence,
            }), { interrupt: true, force: true });
            setVoiceStatus('Reading the latest answer.');
        });

        [voiceHandsFreeEl, voiceAutoSendEl, voiceSpeakAnswersEl, voiceStreamingEl, voiceBargeInEl, voiceSourceAwareEl, voiceConfidenceCuesEl, voiceClarifyFirstEl, voiceMemoryCommandsEl, voiceAccessibilityEl, voiceDepthEl, voiceLangEl, voiceVoiceEl].forEach((element) => {
            element.addEventListener('change', () => {
                updateVoiceSettingsFromUI();
                if (element === voiceVoiceEl && voiceSettings.voiceURI) {
                    setVoiceStatus(`Selected voice updated to ${voiceVoiceEl.options[voiceVoiceEl.selectedIndex]?.text || 'custom voice'}.`);
                }
            });
        });

        voiceDraftEl.addEventListener('input', syncVoiceDraftToQuery);

        queryEl.addEventListener('input', () => {
            if (!voiceDraftEl.value.trim()) {
                return;
            }
            if (document.activeElement === queryEl) {
                voiceDraftEl.value = queryEl.value;
            }
        });

        shareInstagramEl.addEventListener('click', async () => {
            const copied = await copyShareText(activeShareText);
            if (openExternal('https://www.instagram.com/')) {
                setStatus(copied ? 'Copied and opened Instagram. Paste in DM or post draft.' : 'Opened Instagram. Paste content manually.');
                closeShareModal();
            }
        });

        shareMailEl.addEventListener('click', async () => {
            const copied = await copyShareText(activeShareText);
            const subject = encodeURIComponent('Tarka research summary');
            const body = encodeURIComponent(activeShareText || 'Shared from Tarka.');
            window.location.href = `mailto:?subject=${subject}&body=${body}`;
            setStatus(copied ? 'Copied and opened email draft.' : 'Opened email draft. Paste if needed.');
            closeShareModal();
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
    memory_mode: str = "balanced"
    conversation_context: str = ""


class ResearchResponse(BaseModel):
    request_id: str
    query: str
    final_answer: str
    source_urls: list[str]
    claims: list[dict]
    iterations: int
    total_claims: int
    evidence_coverage: float
    avg_confidence: float
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

    memory_mode = request.memory_mode if request.memory_mode in {"balanced", "prefer_memory", "search_only"} else "balanced"

    if request.use_memory and memory_mode == "prefer_memory":
        cached = memory.has_recent_answer(request.query)
        if cached:
            logger.info(f"[api] cache hit for request_id={request_id}")
            cached_claims = cached.get("claims", [])
            cached_meta = cached.get("metadata", {}) if isinstance(cached.get("metadata"), dict) else {}
            return ResearchResponse(
                request_id=request_id,
                query=request.query,
                final_answer=cached["answer"],
                source_urls=cached.get("source_urls", []),
                claims=cached_claims,
                iterations=0,
                total_claims=len(cached_claims),
                evidence_coverage=float(cached_meta.get("evidence_coverage", 0.0)),
                avg_confidence=float(cached_meta.get("avg_confidence", 0.0)),
                elapsed_seconds=0.0,
                from_memory=True,
            )

    start = time.perf_counter()

    initial_state = {
        "query": request.query,
        "conversation_context": request.conversation_context,
        "memory_mode": memory_mode if request.use_memory else "search_only",
        "search_results": [],
        "summary": None,
        "critique": None,
        "iterations": 0,
        "final_answer": "",
        "source_urls": [],
        "evidence_coverage": 0.0,
        "avg_confidence": 0.0,
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
    claims = [c.dict() for c in (summary.claims if summary else [])]

    return ResearchResponse(
        request_id=request_id,
        query=request.query,
        final_answer=final_state.get("final_answer", ""),
        source_urls=source_urls,
        claims=claims,
        iterations=final_state.get("iterations", 0),
        total_claims=total_claims,
        evidence_coverage=float(final_state.get("evidence_coverage", 0.0)),
        avg_confidence=float(final_state.get("avg_confidence", 0.0)),
        elapsed_seconds=elapsed,
        from_memory=False,
    )


@app.get("/research/stream")
async def stream_research(query: str, context: str = "", use_memory: bool = True, memory_mode: str = "balanced"):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    async def event_generator() -> AsyncGenerator[str, None]:
        request_id = str(uuid.uuid4())[:8]
        latest_summary = None
        latest_iterations = 0
        resolved_memory_mode = memory_mode if memory_mode in {"balanced", "prefer_memory", "search_only"} else "balanced"

        if use_memory and resolved_memory_mode == "prefer_memory":
            cached = memory.has_recent_answer(query)
            if cached:
                cached_answer = cached["answer"]
                cached_source_urls = cached.get("source_urls", [])
                cached_claims = cached.get("claims", [])
                cached_meta = cached.get("metadata", {}) if isinstance(cached.get("metadata"), dict) else {}
                for chunk in _chunk_text(cached_answer):
                    yield f"data: {json.dumps({'type': 'delta', 'node': 'assistant', 'data': {'delta': chunk + ' '}})}\n\n"
                    await asyncio.sleep(0)
                yield f"data: {json.dumps({'type': 'final', 'node': 'assistant', 'data': {'request_id': request_id, 'query': query, 'final_answer': cached_answer, 'source_urls': cached_source_urls, 'claims': cached_claims, 'iterations': 0, 'total_claims': len(cached_claims), 'evidence_coverage': float(cached_meta.get('evidence_coverage', 0.0)), 'avg_confidence': float(cached_meta.get('avg_confidence', 0.0)), 'from_memory': True}})}\n\n"
                return

        initial_state = {
            "query": query,
            "conversation_context": context,
            "memory_mode": resolved_memory_mode if use_memory else "search_only",
            "search_results": [],
            "summary": None,
            "critique": None,
            "iterations": 0,
            "final_answer": "",
            "source_urls": [],
            "evidence_coverage": 0.0,
            "avg_confidence": 0.0,
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
                    claims = [c.dict() for c in (latest_summary.claims if latest_summary else [])]
                    yield f"data: {json.dumps({'type': 'final', 'node': 'assistant', 'data': {'request_id': request_id, 'query': query, 'final_answer': final_answer, 'source_urls': source_urls, 'claims': claims, 'iterations': latest_iterations, 'total_claims': total_claims, 'evidence_coverage': float(node_output.get('evidence_coverage', 0.0)), 'avg_confidence': float(node_output.get('avg_confidence', 0.0)), 'from_memory': False}})}\n\n"
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
