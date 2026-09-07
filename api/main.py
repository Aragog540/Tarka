import asyncio
import json
import os
import time
import uuid
import sqlite3
import requests
from datetime import datetime, date
from typing import AsyncGenerator, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from graph.research_graph import research_graph
from memory.store import memory
from observability.logger import logger

load_dotenv()

DB_FILE = "tarka.db"
DATABASE_URL = os.getenv("DATABASE_URL")
DB_TYPE = "postgres" if DATABASE_URL else "sqlite"

def get_db_connection():
    if DB_TYPE == "postgres":
        import psycopg2
        return psycopg2.connect(DATABASE_URL)
    else:
        return sqlite3.connect(DB_FILE)

def execute_query(query: str, params: tuple = ()):
    conn = get_db_connection()
    cursor = conn.cursor()
    if DB_TYPE == "postgres":
        query = query.replace("?", "%s")
    cursor.execute(query, params)
    conn.commit()
    conn.close()

def fetch_one(query: str, params: tuple = ()):
    conn = get_db_connection()
    cursor = conn.cursor()
    if DB_TYPE == "postgres":
        query = query.replace("?", "%s")
    cursor.execute(query, params)
    row = cursor.fetchone()
    conn.close()
    return row

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            google_id TEXT PRIMARY KEY,
            email TEXT UNIQUE,
            name TEXT,
            picture TEXT,
            dob TEXT,
            preferred_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_token TEXT PRIMARY KEY,
            google_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            FOREIGN KEY (google_id) REFERENCES users(google_id)
        )
    """)
    conn.commit()
    conn.close()

init_db()


def calculate_age(born_str: str) -> int:
    try:
        born = datetime.strptime(born_str, "%Y-%m-%d").date()
        today = date.today()
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    except Exception:
        return 0


def get_user_from_session(session_token: str) -> Optional[dict]:
    if not session_token:
        return None
    try:
        row = fetch_one("""
            SELECT u.google_id, u.email, u.name, u.picture, u.dob, u.preferred_name
            FROM sessions s
            JOIN users u ON s.google_id = u.google_id
            WHERE s.session_token = ?
        """, (session_token,))
        if row:
            return {
                "google_id": row[0],
                "email": row[1],
                "name": row[2],
                "picture": row[3],
                "dob": row[4],
                "preferred_name": row[5]
            }
    except Exception as e:
        logger.error(f"[db] error getting user from session: {e}")
    return None

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
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Plus+Jakarta+Sans:wght@600;700;800&family=JetBrains+Mono:wght@500;600&family=Nunito:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #f8fafc;
            --bg-grid: rgba(15, 23, 42, 0.02);
            --panel: rgba(255, 255, 255, 0.8);
            --panel-strong: #ffffff;
            --text: #0f172a;
            --muted: #475569;
            --line: rgba(15, 23, 42, 0.08);
            --accent: #4f46e5;
            --accent-strong: #3730a3;
            --accent-soft: rgba(79, 70, 229, 0.06);
            --shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.04), 0 8px 16px -6px rgba(15, 23, 42, 0.04);
            --radius: 16px;
            --radius-inner: 12px;
            
            --status-color: #10b981;

            /* Mochi Light Theme Variables */
            --mochi-bg: #f3dfa7;
            --mochi-bg-input: #FFFFFF;
            --mochi-border-input: #4A8C72;
            --mochi-text: #2C2418;
            --mochi-muted: #7A6A56;
            --mochi-label: #3A6B52;
            --mochi-accent: #c0694c;
            --mochi-btn-bg: #3A6B52;
            --mochi-btn-text: #FFFFFF;
            --mochi-toggle-track: #d3c090;
            --mochi-toggle-thumb: #3A6B52;
            --mochi-input-shadow: rgba(74, 140, 114, 0.15);
        }

        body[data-theme="dark"] {
            --bg: #030712;
            --bg-grid: rgba(255, 255, 255, 0.015);
            --panel: rgba(17, 24, 39, 0.7);
            --panel-strong: #0f172a;
            --text: #f8fafc;
            --muted: #94a3b8;
            --line: rgba(255, 255, 255, 0.06);
            --accent: #818cf8;
            --accent-strong: #6366f1;
            --accent-soft: rgba(99, 102, 241, 0.15);
            --shadow: 0 20px 30px -10px rgba(0, 0, 0, 0.4), 0 10px 15px -5px rgba(0, 0, 0, 0.4);

            /* Mochi Dark Theme Variables */
            --mochi-bg: #36312a;
            --mochi-bg-input: #2A2318;
            --mochi-border-input: #5DCAA5;
            --mochi-text: #F0E8D5;
            --mochi-muted: #A89880;
            --mochi-label: #5DCAA5;
            --mochi-accent: #F0997B;
            --mochi-btn-bg: #5DCAA5;
            --mochi-btn-text: #1C1812;
            --mochi-toggle-track: #4c4436;
            --mochi-toggle-thumb: #5DCAA5;
            --mochi-input-shadow: rgba(93, 202, 165, 0.1);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            min-height: 100vh;
            font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif;
            color: var(--text);
            background-color: var(--bg);
            background-image: 
                radial-gradient(circle at 0% 0%, var(--accent-soft) 0%, transparent 35%),
                radial-gradient(circle at 100% 100%, rgba(6, 182, 212, 0.04) 0%, transparent 35%),
                linear-gradient(var(--bg-grid) 1px, transparent 1px), 
                linear-gradient(90deg, var(--bg-grid) 1px, transparent 1px);
            background-size: 100% 100%, 100% 100%, 20px 20px, 20px 20px;
            transition: background-color .3s ease, color .3s ease;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        h1, h2, h3, h4, strong {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .wrap {
            width: min(1360px, calc(100% - 48px));
            margin: 0 auto;
            padding: 24px 0 40px;
        }

        .workspace {
            display: grid;
            grid-template-columns: 1fr;
            gap: 0;
            align-items: start;
        }

        .card {
            background: var(--panel);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--line);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            transition: background-color .3s ease, border-color .3s ease, box-shadow .3s ease;
        }

        .intro {
            padding: 28px;
            position: relative;
            overflow: hidden;
            border-radius: var(--radius);
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 99px;
            background: var(--accent-soft);
            border: 1px solid rgba(79, 70, 229, 0.15);
            color: var(--accent);
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 14px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        h1 {
            font-size: clamp(2rem, 3.5vw, 3rem);
            line-height: 1.1;
            margin-bottom: 12px;
            background: linear-gradient(135deg, var(--text) 30%, var(--accent) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .lead {
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.6;
            margin-bottom: 0;
        }

        .panel {
            padding: 24px;
        }

        .panel-header {
            margin-bottom: 20px;
        }

        .panel-header h2 {
            font-size: 1.2rem;
            margin-bottom: 4px;
        }

        .panel-header p {
            color: var(--muted);
            font-size: 0.85rem;
            line-height: 1.4;
        }

        textarea {
            width: 100%;
            border: 1px solid var(--line);
            border-radius: var(--radius-inner);
            padding: 14px 16px;
            font-family: inherit;
            font-size: 0.95rem;
            line-height: 1.5;
            background: var(--panel-strong);
            color: var(--text);
            outline: none;
            transition: border-color .2s ease, box-shadow .2s ease;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.01);
        }

        textarea:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.12);
        }

        .chips {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 16px;
            transition: max-height 0.4s cubic-bezier(0.16, 1, 0.3, 1), margin-bottom 0.4s ease, opacity 0.3s ease;
            max-height: 100px;
            overflow: hidden;
            opacity: 1;
        }

        .chips.collapsed {
            max-height: 0;
            margin-bottom: 0;
            padding-top: 0;
            padding-bottom: 0;
            opacity: 0;
            pointer-events: none;
        }

        .chip {
            border: 1px solid var(--line);
            background: var(--panel-strong);
            color: var(--muted);
            border-radius: 99px;
            padding: 8px 14px;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.85rem;
            font-weight: 500;
            transition: all .2s ease;
        }

        .chip:hover {
            border-color: var(--accent);
            color: var(--accent);
            background: var(--accent-soft);
            transform: translateY(-1px);
        }

        .toggle {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            color: var(--muted);
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            user-select: none;
        }

        .toggle input {
            width: 16px;
            height: 16px;
            accent-color: var(--accent);
        }

        .memory-select {
            border: 1px solid var(--line);
            background: var(--panel-strong);
            color: var(--text);
            border-radius: var(--radius-inner);
            padding: 6px 12px;
            font-family: inherit;
            font-size: 0.85rem;
            outline: none;
            cursor: pointer;
        }

        .message-metrics {
            margin-top: 12px;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .metric-pill {
            border: 1px solid var(--line);
            background: var(--accent-soft);
            color: var(--accent);
            border-radius: 99px;
            padding: 4px 10px;
            font-size: 0.75rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }

        .actions {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
        }

        .btn {
            appearance: none;
            border: 1px solid transparent;
            border-radius: 99px;
            padding: 10px 20px;
            font-family: inherit;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all .2s cubic-bezier(0.16, 1, 0.3, 1);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .btn:active {
            transform: scale(0.98);
        }

        .btn-primary {
            background: var(--accent);
            color: #ffffff;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
        }

        .btn-primary:hover {
            background: var(--accent-strong);
            box-shadow: 0 6px 20px rgba(79, 70, 229, 0.3);
            transform: translateY(-1px);
        }

        .btn-secondary {
            background: var(--panel-strong);
            color: var(--text);
            border-color: var(--line);
        }

        .btn-secondary:hover {
            background: var(--bg);
            border-color: var(--accent);
            color: var(--accent);
        }

        .status {
            color: var(--muted);
            font-size: 0.85rem;
            font-weight: 500;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        /* Pulsing Status Dot */
        .status::before {
            content: "";
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: var(--status-color);
            box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
        }

        .status-busy::before {
            --status-color: #f59e0b;
            box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.2);
            animation: pulse-dot 1.2s infinite ease-in-out;
        }

        .status-error::before {
            --status-color: #ef4444;
            box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.2);
        }

        .status-ready::before {
            --status-color: #10b981;
            box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
        }

        @keyframes pulse-dot {
            0% { transform: scale(0.95); opacity: 0.6; }
            50% { transform: scale(1.2); opacity: 1; }
            100% { transform: scale(0.95); opacity: 0.6; }
        }

        .result {
            display: grid;
            gap: 16px;
        }

        .answer {
            padding: 20px;
            border-radius: var(--radius-inner);
            background: var(--accent-soft);
            border: 1px solid rgba(79, 70, 229, 0.1);
            line-height: 1.6;
            font-size: 0.95rem;
            white-space: pre-wrap;
        }

        .sources {
            display: grid;
            gap: 8px;
        }

        .source {
            padding: 12px 14px;
            border-radius: var(--radius-inner);
            background: var(--panel-strong);
            border: 1px solid var(--line);
            color: var(--muted);
            font-size: 0.85rem;
            word-break: break-all;
        }

        .empty {
            color: var(--muted);
            border: 1px dashed var(--line);
            border-radius: var(--radius-inner);
            padding: 24px;
            text-align: center;
            background: rgba(15, 23, 42, 0.01);
            font-size: 0.9rem;
        }

        .sidebar {
            position: fixed;
            top: 0;
            left: -320px;
            width: 300px;
            height: 100vh;
            max-height: 100vh;
            z-index: 1200;
            border-radius: 0 var(--radius) var(--radius) 0;
            transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            background: var(--panel-strong) !important;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 20px;
            box-shadow: 10px 0 30px rgba(0, 0, 0, 0.15);
            overflow: hidden;
        }

        .sidebar.open {
            transform: translateX(320px);
        }

        .drawer-overlay {
            position: fixed;
            inset: 0;
            background: rgba(3, 7, 18, 0.4);
            backdrop-filter: blur(4px);
            -webkit-backdrop-filter: blur(4px);
            opacity: 0;
            pointer-events: none;
            z-index: 1050;
            transition: opacity 0.3s ease;
        }

        .drawer-overlay.open {
            opacity: 1;
            pointer-events: auto;
        }

        .drawer-close {
            appearance: none;
            border: none;
            background: transparent;
            color: var(--text);
            font-size: 1.5rem;
            cursor: pointer;
            line-height: 1;
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            transition: all 0.2s ease;
        }

        .drawer-close:hover {
            background: var(--accent-soft);
            color: var(--accent);
        }

        .header-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 24px;
            padding-left: 52px; /* Avoid overlapping fixed toggle button */
        }

        .header-logo {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-weight: 800;
            font-size: 1.4rem;
            background: linear-gradient(135deg, var(--text) 30%, var(--accent) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .drawer-toggle {
            position: fixed;
            top: 20px;
            left: 20px;
            z-index: 1300;
            width: 40px;
            height: 40px;
            padding: 0;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text);
            background: var(--panel-strong);
            border: 1px solid var(--line);
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }

        .drawer-toggle:hover {
            border-color: var(--accent);
            color: var(--accent);
            background: var(--accent-soft);
            transform: scale(1.05);
        }

        .main-content {
            transition: padding-left 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            width: 100%;
        }

        @media (min-width: 1025px) {
            body.sidebar-open .main-content {
                padding-left: 300px;
            }
            body.sidebar-open .drawer-overlay {
                display: none !important;
            }
            .drawer-close {
                display: none !important;
            }
        }

        .message.assistant.typing .message-content::after {
            content: "▊";
            display: inline-block;
            margin-left: 2px;
            color: var(--accent);
            animation: blink 0.8s steps(2, start) infinite;
        }

        @keyframes blink {
            to { visibility: hidden; }
        }

        .sidebar-header {
            display: flex;
            flex-direction: column;
            gap: 14px;
        }

        .sidebar-header h2 {
            font-size: 1.3rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--text) 30%, var(--accent) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .sidebar-header p {
            color: var(--muted);
            font-size: 0.8rem;
            line-height: 1.4;
        }

        .history-list {
            display: grid;
            grid-template-columns: minmax(0, 1fr);
            gap: 4px;
            align-content: start;
            overflow-y: auto;
            padding-right: 4px;
            padding-bottom: 60px; /* Generous bottom spacing to prevent clipping of last items */
            flex: 1;
            min-height: 0;
            scrollbar-width: thin;
            scrollbar-color: rgba(15, 23, 42, 0.1) transparent;
        }

        body[data-theme="dark"] .history-list {
            scrollbar-color: rgba(255, 255, 255, 0.1) transparent;
        }

        .history-item {
            position: relative;
            width: 100%;
            text-align: left;
            border: 1px solid transparent;
            border-radius: var(--radius-inner);
            background: transparent;
            padding: 8px 12px;
            cursor: pointer;
            font-family: inherit;
            color: var(--text);
            transition: all .15s ease;
            display: flex;
            align-items: center;
            height: 38px;
            box-sizing: border-box;
        }

        .history-item:hover {
            background: var(--accent-soft);
        }

        .history-item.active {
            background: var(--accent-soft);
        }

        .history-item::before {
            content: "";
            position: absolute;
            left: 0;
            top: 20%;
            height: 60%;
            width: 3px;
            background: var(--accent);
            border-radius: 0 4px 4px 0;
            opacity: 0;
            transition: opacity 0.15s ease;
        }

        .history-item.active::before {
            opacity: 1;
        }

        .history-item strong {
            display: block;
            font-size: 0.85rem;
            font-weight: 500;
            line-height: 1.35;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            padding-right: 24px;
            flex: 1;
        }

        .history-item span {
            display: none !important;
        }

        .history-item-actions {
            position: absolute;
            right: 8px;
            top: 50%;
            transform: translateY(-50%);
            opacity: 0;
            transition: opacity 0.15s ease;
            z-index: 10;
        }

        .history-item:hover .history-item-actions,
        .history-item.active .history-item-actions {
            opacity: 1;
        }

        .session-menu-trigger {
            appearance: none;
            border: none;
            background: transparent;
            color: var(--muted);
            font-size: 1.1rem;
            cursor: pointer;
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            transition: all 0.2s ease;
            position: relative;
            z-index: 11;
        }

        .session-menu-trigger:hover {
            background: rgba(15, 23, 42, 0.05);
            color: var(--text);
        }

        body[data-theme="dark"] .session-menu-trigger:hover {
            background: rgba(255, 255, 255, 0.05);
        }

        .session-menu {
            position: absolute;
            right: 26px; /* Position to the left of the trigger button */
            top: 50%;
            transform: translateY(-50%);
            min-width: 100px;
            padding: 4px;
            border-radius: var(--radius-inner);
            border: 1px solid var(--line);
            background: var(--panel-strong);
            box-shadow: var(--shadow);
            display: none;
            z-index: 20;
        }

        .session-menu.open {
            display: grid;
            gap: 2px;
        }

        .session-menu-item {
            width: 100%;
            text-align: left;
            appearance: none;
            border: none;
            background: transparent;
            color: var(--text);
            border-radius: 6px;
            padding: 8px 10px;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.8rem;
            font-weight: 500;
            transition: background 0.15s ease;
        }

        .session-menu-item:hover {
            background: var(--accent-soft);
            color: var(--accent);
        }

        .session-menu-item.danger {
            color: #ef4444;
        }

        .session-menu-item.danger:hover {
            background: rgba(239, 68, 68, 0.08);
            color: #ef4444;
        }

        .explicit-flag {
            color: #ef4444;
            background: rgba(239, 68, 68, 0.06);
            border: 1px solid rgba(239, 68, 68, 0.15);
            padding: 6px 10px;
            border-radius: 8px;
            font-size: 0.75rem;
            font-weight: 600;
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
            font-size: 0.8rem;
            text-align: center;
            border: 1px dashed var(--line);
            border-radius: var(--radius-inner);
            padding: 16px;
            background: rgba(15, 23, 42, 0.005);
        }

        .welcome-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            flex: 1;
            padding: 40px 20px;
            text-align: center;
            animation: welcome-in 0.5s ease-out;
            max-width: 600px;
            margin: auto;
        }

        .welcome-title {
            font-size: 2.2rem;
            font-weight: 600;
            color: var(--text);
            margin: 0 0 12px 0;
            letter-spacing: -0.02em;
        }

        .welcome-subtitle {
            font-size: 1.1rem;
            color: var(--muted);
            margin: 0;
            font-weight: 400;
        }

        @keyframes welcome-in {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .maker-card {
            margin-top: auto;
            padding-top: 16px;
            border-top: 1px solid var(--line);
            display: grid;
            gap: 10px;
        }

        .maker-top {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .maker-avatar {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            object-fit: cover;
            border: 1px solid var(--line);
        }

        .maker-copy {
            display: grid;
            gap: 1px;
        }

        .maker-copy strong {
            font-size: 0.8rem;
            font-weight: 700;
        }

        .maker-copy span {
            color: var(--muted);
            font-size: 0.7rem;
        }

        .maker-links {
            display: flex;
            gap: 6px;
        }

        .maker-link {
            padding: 4px 8px;
            border-radius: 6px;
            border: 1px solid var(--line);
            background: var(--panel-strong);
            color: var(--muted);
            text-decoration: none;
            font-size: 0.7rem;
            font-weight: 600;
            transition: all 0.2s ease;
        }

        .maker-link:hover {
            border-color: var(--accent);
            color: var(--accent);
            background: var(--accent-soft);
        }

        .chat-shell {
            padding: 24px;
            height: calc(100vh - 120px);
            display: flex;
            flex-direction: column;
            gap: 20px;
            overflow: hidden;
        }

        .chat-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 24px;
            border-bottom: 1px solid var(--line);
            padding-bottom: 20px;
            transition: max-height 0.4s cubic-bezier(0.16, 1, 0.3, 1), padding 0.4s ease, margin 0.4s ease, border 0.4s ease, opacity 0.3s ease;
            max-height: 200px;
            overflow: hidden;
            opacity: 1;
        }

        .chat-header.collapsed {
            max-height: 0;
            padding-top: 0;
            padding-bottom: 0;
            margin-top: 0;
            margin-bottom: 0;
            border-bottom: none;
            opacity: 0;
            pointer-events: none;
        }

        .chat-header-meta {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 4px;
            text-align: right;
            color: var(--muted);
            font-size: 0.75rem;
        }

        .chat-header-meta > div {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .chat-header-meta strong {
            color: var(--text);
        }

        .chat-messages {
            flex: 1;
            min-height: 0;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 18px;
            padding-right: 4px;
            scrollbar-width: thin;
            scrollbar-color: rgba(15, 23, 42, 0.1) transparent;
        }

        body[data-theme="dark"] .chat-messages {
            scrollbar-color: rgba(255, 255, 255, 0.1) transparent;
        }

        .message {
            max-width: 85%;
            padding: 14px 18px;
            border-radius: var(--radius-inner);
            line-height: 1.55;
            font-size: 0.95rem;
            animation: message-in 0.3s cubic-bezier(0.16, 1, 0.3, 1) both;
        }

        @keyframes message-in {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            align-self: flex-end;
            background: var(--accent);
            color: #ffffff;
            border-radius: var(--radius-inner) var(--radius-inner) 4px var(--radius-inner);
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.15);
        }

        .message.user .message-label {
            color: rgba(255, 255, 255, 0.8);
        }

        .message.assistant {
            align-self: flex-start;
            background: var(--panel-strong);
            color: var(--text);
            border: 1px solid var(--line);
            border-radius: var(--radius-inner) var(--radius-inner) var(--radius-inner) 4px;
        }

        .message-label {
            display: block;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
            color: var(--muted);
        }

        .message-content {
            white-space: pre-wrap;
        }

        .message-sources {
            display: none;
        }

        .sources-button {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            border-radius: 99px;
            border: 1px solid var(--line);
            background: var(--panel-strong);
            color: var(--muted);
            text-decoration: none;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.75rem;
            font-weight: 600;
            margin-top: 12px;
            transition: all 0.2s ease;
        }

        .sources-button:hover {
            border-color: var(--accent);
            color: var(--accent);
            background: var(--accent-soft);
        }

        .sources-modal {
            position: fixed;
            inset: 0;
            display: none;
            align-items: center;
            justify-content: center;
            padding: 24px;
            background: rgba(3, 7, 18, 0.4);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            z-index: 1000;
            animation: fade-in 0.2s ease-out;
        }

        @keyframes fade-in {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        .sources-modal.open {
            display: flex;
        }

        .sources-modal-card {
            width: min(640px, 100%);
            max-height: min(80vh, 700px);
            overflow-y: auto;
            padding: 28px;
            border-radius: var(--radius);
            background: var(--panel-strong);
            border: 1px solid var(--line);
            box-shadow: var(--shadow);
            display: flex;
            flex-direction: column;
            gap: 20px;
            animation: modal-in 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        }

        @keyframes modal-in {
            from { transform: scale(0.95) translateY(12px); opacity: 0; }
            to { transform: scale(1) translateY(0); opacity: 1; }
        }

        .sources-modal-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 16px;
        }

        .sources-modal-header h3 {
            font-size: 1.25rem;
        }

        .sources-modal-header p {
            color: var(--muted);
            font-size: 0.85rem;
        }

        .sources-modal-close {
            appearance: none;
            border: 1px solid var(--line);
            background: var(--panel-strong);
            color: var(--text);
            border-radius: 50%;
            width: 32px;
            height: 32px;
            cursor: pointer;
            font-size: 1.2rem;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
        }

        .sources-modal-close:hover {
            background: var(--accent-soft);
            color: var(--accent);
            border-color: var(--accent);
        }

        .sources-modal-list {
            display: grid;
            gap: 12px;
            overflow-y: auto;
        }

        .sources-modal-item {
            display: block;
            padding: 10px 14px;
            border-radius: var(--radius-inner);
            border: 1px solid var(--line);
            background: var(--bg);
            color: var(--accent);
            text-decoration: none;
            word-break: break-all;
            font-size: 0.8rem;
            font-weight: 500;
            transition: all 0.2s ease;
        }

        .sources-modal-item:hover {
            border-color: var(--accent);
            background: var(--accent-soft);
        }

        .share-options {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }

        .share-option-btn {
            appearance: none;
            border: 1px solid var(--line);
            background: var(--panel-strong);
            color: var(--text);
            border-radius: var(--radius-inner);
            padding: 12px 14px;
            font-family: inherit;
            font-size: 0.85rem;
            font-weight: 500;
            text-align: center;
            cursor: pointer;
            transition: all .2s ease;
        }

        .share-option-btn:hover {
            border-color: var(--accent);
            color: var(--accent);
            background: var(--accent-soft);
            transform: translateY(-1px);
        }

        .voice-panel {
            display: none;
            gap: 16px;
            padding: 20px;
            border: 1px solid var(--line);
            border-radius: var(--radius-inner);
            background: var(--panel-strong);
            max-height: 200px;
            overflow-y: auto;
        }

        .voice-panel.open {
            display: grid;
        }

        .voice-panel-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 12px;
            border-bottom: 1px solid var(--line);
            padding-bottom: 12px;
        }

        .voice-panel-header strong {
            font-size: 0.95rem;
        }

        .voice-panel-header span,
        .voice-note,
        .voice-status {
            color: var(--muted);
            font-size: 0.8rem;
            line-height: 1.4;
        }

        .voice-toolbar,
        .voice-settings-grid,
        .voice-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
        }

        .voice-settings-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
            gap: 8px;
        }

        .voice-mic-btn {
            min-width: 130px;
        }

        .voice-pill,
        .voice-toggle {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 10px;
            border-radius: 99px;
            border: 1px solid var(--line);
            background: var(--bg);
            color: var(--muted);
            font-family: inherit;
            font-size: 0.75rem;
            font-weight: 500;
            cursor: pointer;
        }

        .voice-toggle input {
            width: 14px;
            height: 14px;
            accent-color: var(--accent);
        }

        .voice-select {
            border: 1px solid var(--line);
            background: var(--bg);
            color: var(--text);
            border-radius: 8px;
            padding: 6px 10px;
            font-family: inherit;
            font-size: 0.75rem;
            outline: none;
            cursor: pointer;
        }

        .voice-status-box {
            display: grid;
            gap: 4px;
            padding: 10px 12px;
            border-radius: 8px;
            background: var(--accent-soft);
            border: 1px solid rgba(79, 70, 229, 0.1);
        }

        .voice-draft {
            display: grid;
            gap: 6px;
            padding: 12px;
            border-radius: 8px;
            background: var(--bg);
            border: 1px solid var(--line);
        }

        .voice-draft textarea {
            min-height: 60px;
            padding: 8px;
            font-size: 0.85rem;
        }

        .voice-summary {
            display: grid;
            gap: 8px;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px dashed var(--line);
        }

        .followup-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }

        .followup-chip,
        .speak-claim-btn {
            appearance: none;
            border: 1px solid var(--line);
            background: var(--panel-strong);
            color: var(--muted);
            border-radius: 99px;
            padding: 6px 12px;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.8rem;
            font-weight: 500;
            transition: all .2s ease;
        }

        .followup-chip:hover,
        .speak-claim-btn:hover {
            border-color: var(--accent);
            color: var(--accent);
            background: var(--accent-soft);
            transform: translateY(-1px);
        }

        .voice-active {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
            animation: pulse-border 1.5s infinite ease-in-out;
        }

        @keyframes pulse-border {
            0% { border-color: rgba(79, 70, 229, 0.3); }
            50% { border-color: rgba(79, 70, 229, 0.8); }
            100% { border-color: rgba(79, 70, 229, 0.3); }
        }

        body.voice-accessibility {
            font-size: 1.05rem;
        }

        .composer {
            border-top: 1px solid var(--line);
            padding-top: 20px;
            display: grid;
            gap: 14px;
        }

        .query-box-container {
            display: flex;
            align-items: center;
            background: var(--panel-strong) !important;
            border: 1px solid var(--line);
            border-radius: 999px; /* Cylindrical style */
            padding: 8px 12px;
            gap: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
            position: relative;
            transition: border-color 0.2s, box-shadow 0.2s;
            width: 100%;
        }

        .query-box-container:focus-within {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.12);
        }

        /* Mode pill styling */
        .mode-pill-selector {
            position: relative;
            flex-shrink: 0;
        }

        .mode-trigger-btn {
            appearance: none;
            border: none;
            background: transparent;
            padding: 0;
            cursor: pointer;
            outline: none;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .mode-icon-circle {
            width: 38px;
            height: 38px;
            background: var(--accent-soft);
            color: var(--accent);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            font-weight: bold;
            border: 1px solid rgba(79, 70, 229, 0.15);
            transition: all 0.2s ease;
        }

        .mode-trigger-btn:hover .mode-icon-circle {
            transform: scale(1.05);
            background: var(--accent);
            color: #ffffff;
            border-color: var(--accent);
        }

        /* Dropdown menu styling */
        .mode-dropdown-menu {
            position: absolute;
            bottom: calc(100% + 12px);
            left: 0;
            background: var(--panel-strong);
            border: 1px solid var(--line);
            border-radius: var(--radius-inner);
            box-shadow: var(--shadow);
            padding: 6px;
            display: none;
            flex-direction: column;
            gap: 4px;
            min-width: 160px;
            z-index: 1000;
            animation: menu-in 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .mode-dropdown-menu.open {
            display: flex;
        }

        @keyframes menu-in {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .mode-dropdown-item {
            appearance: none;
            border: none;
            background: transparent;
            color: var(--text);
            border-radius: 8px;
            padding: 8px 12px;
            font-family: inherit;
            font-size: 0.85rem;
            font-weight: 600;
            text-align: left;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: background 0.15s ease, color 0.15s ease;
            width: 100%;
        }

        .mode-dropdown-item:hover {
            background: var(--accent-soft);
            color: var(--accent);
        }

        /* Textarea input space */
        .query-box-container textarea {
            flex: 1;
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            padding: 8px 4px !important;
            margin: 0;
            resize: none;
            height: 38px;
            min-height: 38px;
            max-height: 120px;
            line-height: 1.4;
            outline: none;
            font-family: inherit;
            font-size: 0.95rem;
            color: var(--text);
            box-sizing: border-box;
            overflow-y: auto;
        }

        /* Send Button styling */
        .send-btn {
            appearance: none;
            border: none;
            background: var(--accent);
            color: #ffffff;
            width: 38px;
            height: 38px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s ease;
            flex-shrink: 0;
            box-shadow: 0 2px 6px rgba(79, 70, 229, 0.2);
            outline: none;
        }

        .send-btn svg {
            width: 18px;
            height: 18px;
            transform: translate(-1px, 1px);
        }

        .send-btn:hover {
            background: var(--accent-strong);
            transform: scale(1.05);
            box-shadow: 0 4px 10px rgba(79, 70, 229, 0.3);
        }

        .send-btn:active {
            transform: scale(0.95);
        }

        .composer-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            flex-wrap: wrap;
        }

        .session-badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 8px;
            border-radius: 6px;
            border: 1px solid var(--line);
            background: var(--panel-strong);
            color: var(--text);
            font-size: 0.75rem;
            font-weight: 600;
        }

        .theme-toggle {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            color: var(--muted);
            font-size: 0.8rem;
            font-weight: 600;
            cursor: pointer;
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
            width: 38px;
            height: 22px;
            border-radius: 99px;
            background: rgba(15, 23, 42, 0.08);
            border: 1px solid var(--line);
            flex: 0 0 auto;
            transition: all .2s ease;
        }

        body[data-theme="dark"] .theme-switch {
            background: rgba(255, 255, 255, 0.08);
        }

        .theme-switch::after {
            content: "";
            position: absolute;
            top: 2px;
            left: 2px;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #ffffff;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            transition: transform .2s cubic-bezier(0.16, 1, 0.3, 1);
        }

        body[data-theme="dark"] .theme-switch::after {
            background: #0f172a;
        }

        .theme-toggle input:checked + .theme-switch {
            background: var(--accent);
            border-color: var(--accent);
        }

        .theme-toggle input:checked + .theme-switch::after {
            transform: translateX(16px);
        }

        @media (max-width: 900px) {
            .workspace { grid-template-columns: 1fr; }
            .chat-shell { height: calc(100vh - 100px); }
        }

        @media (max-width: 640px) {
            .wrap { width: min(100% - 16px, 1360px); padding-top: 16px; }
            .chat-shell, .panel { padding: 16px; }
            .chat-shell { height: calc(100vh - 80px); height: calc(100dvh - 80px); }
            .message { max-width: 95%; }
            .chat-header { flex-direction: column; align-items: flex-start; gap: 12px; }
            .chat-header-meta { align-items: flex-start; text-align: left; }
            .share-options { grid-template-columns: 1fr; }

            /* Mobile optimization for query box */
            .query-box-container {
                display: grid !important;
                grid-template-areas: 
                    "textarea textarea textarea"
                    "mode voice run" !important;
                grid-template-columns: auto 1fr auto !important;
                grid-template-rows: auto auto !important;
                border-radius: 20px !important;
                padding: 12px 16px !important;
                gap: 12px !important;
            }
            .query-box-container textarea {
                grid-area: textarea !important;
                width: 100% !important;
                min-height: 48px !important;
            }
            .mode-pill-selector {
                grid-area: mode !important;
                justify-self: start !important;
            }
            .mic-btn {
                grid-area: voice !important;
                justify-self: start !important;
                display: flex !important;
            }
            .send-btn {
                grid-area: run !important;
                justify-self: end !important;
            }

            /* Increased Touch Targets on Mobile */
            .mode-icon-circle, .mic-btn, .send-btn {
                width: 44px !important;
                height: 44px !important;
                font-size: 1.2rem !important;
            }
            .send-btn svg, .mic-btn svg {
                width: 22px !important;
                height: 22px !important;
            }
            .msg-action-btn {
                width: 36px !important;
                height: 36px !important;
            }
            .msg-action-btn svg {
                width: 18px !important;
                height: 18px !important;
            }
            .followup-chip {
                padding: 10px 16px !important;
                font-size: 0.9rem !important;
            }

            /* Hide unnecessary scroll hint to save space */
            #scroll_hint {
                display: none !important;
            }

            /* Drawer toggle touch target */
            .drawer-toggle {
                width: 44px !important;
                height: 44px !important;
                top: 16px !important;
                left: 16px !important;
            }

            /* Modals spacing */
            .sources-modal {
                padding: 12px !important;
            }
            .sources-modal-card {
                padding: 20px !important;
                max-height: min(90dvh, 700px) !important;
                gap: 16px !important;
                border-radius: var(--radius-inner) !important;
            }
        }

        /* Authentication & Onboarding UI Styles (Mochi Full-Screen Layout) */
        .auth-overlay {
            position: fixed;
            inset: 0;
            z-index: 2000;
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: var(--bg);
            background-image: 
                radial-gradient(circle at 0% 0%, var(--accent-soft) 0%, transparent 35%),
                radial-gradient(circle at 100% 100%, rgba(6, 182, 212, 0.04) 0%, transparent 35%),
                linear-gradient(var(--bg-grid) 1px, transparent 1px), 
                linear-gradient(90deg, var(--bg-grid) 1px, transparent 1px);
            background-size: 100% 100%, 100% 100%, 20px 20px, 20px 20px;
            padding: 40px 24px 24px;
            transition: background 0.3s, color 0.3s;
            overflow-y: auto;
            font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif;
        }

        .auth-container {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 40px;
            width: 100%;
            max-width: 960px;
            margin: auto;
        }

        .auth-form-column {
            flex: 1;
            max-width: 400px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .auth-char-column {
            flex: 0 0 auto;
            display: flex;
            align-items: flex-end;
            justify-content: center;
        }

        .char-wrap {
            position: relative;
            width: 440px;
            height: 510px;
            user-select: none;
            pointer-events: auto;
            cursor: pointer;
        }

        .char-wrap img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        #img-base { z-index: 1; }
        #img-eyes { z-index: 2; transition: transform 0.05s linear; }
        #img-expr { z-index: 3; opacity: 0; transition: opacity 0.15s ease; }
        #img-expr.visible { opacity: 1; }

        .auth-card {
            width: 100%;
            padding: 36px;
            border-radius: var(--radius);
            background: var(--panel);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--line);
            box-shadow: var(--shadow);
            display: flex;
            flex-direction: column;
            gap: 20px;
            text-align: center;
            animation: auth-modal-in 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
            color: var(--text);
        }

        @keyframes auth-modal-in {
            from { transform: scale(0.95) translateY(12px); opacity: 0; }
            to { transform: scale(1) translateY(0); opacity: 1; }
        }

        .auth-logo {
            font-size: 2.6rem;
            font-weight: 800;
            color: var(--accent);
            font-family: 'Plus Jakarta Sans', sans-serif;
            margin-bottom: 2px;
            letter-spacing: -0.02em;
        }

        .auth-title {
            font-size: 1.45rem;
            font-weight: 800;
            color: var(--text);
            margin-bottom: 4px;
        }

        .auth-desc {
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.5;
            margin-bottom: 8px;
        }

        .auth-form-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
            text-align: left;
        }

        .auth-form-group label {
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text);
            margin-left: 16px;
        }

        .auth-input {
            width: 100%;
            border: 1px solid var(--line);
            border-radius: var(--radius-inner);
            padding: 12px 18px;
            font-family: inherit;
            font-size: 0.95rem;
            font-weight: 500;
            background: var(--panel-strong);
            color: var(--text);
            outline: none;
            transition: all 0.2s ease;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
        }

        .auth-input:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-soft), 0 2px 8px rgba(0, 0, 0, 0.02);
        }

        .auth-error {
            color: #ef4444;
            font-size: 0.85rem;
            font-weight: 600;
            background: rgba(239, 68, 68, 0.05);
            border: 1px solid rgba(239, 68, 68, 0.2);
            padding: 12px 16px;
            border-radius: var(--radius-inner);
            display: none;
            text-align: left;
        }

        .dev-login-box {
            border-top: 1px dashed var(--line);
            padding-top: 20px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            margin-top: 12px;
        }

        .dev-login-box p {
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            color: var(--muted);
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }

        .btn-mochi {
            appearance: none;
            border: 1px solid transparent;
            border-radius: 99px;
            padding: 12px 24px;
            font-family: inherit;
            font-weight: 600;
            font-size: 0.95rem;
            cursor: pointer;
            transition: all .2s cubic-bezier(0.16, 1, 0.3, 1);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            background: var(--accent);
            color: #ffffff;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
        }

        .btn-mochi:hover {
            background: var(--accent-strong);
            box-shadow: 0 6px 20px rgba(79, 70, 229, 0.3);
            transform: translateY(-1px);
        }

        .btn-mochi:active {
            transform: scale(0.98);
        }

        .auth-theme-toggle {
            position: absolute;
            top: 24px;
            right: 24px;
            display: flex;
            align-items: center;
            background: var(--panel);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 3px;
            cursor: pointer;
            gap: 2px;
            transition: all 0.3s;
            outline: none;
            z-index: 2010;
        }

        .auth-theme-toggle:hover {
            border-color: var(--accent);
        }

        .auth-theme-toggle .icon {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.3s;
        }

        .auth-theme-toggle .icon img {
            width: 20px;
            height: 20px;
            object-fit: contain;
        }

        .auth-theme-toggle .icon.active {
            background: var(--accent-soft);
        }

        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            15% { transform: translateX(-8px); }
            30% { transform: translateX(8px); }
            45% { transform: translateX(-6px); }
            60% { transform: translateX(6px); }
            75% { transform: translateX(-3px); }
            90% { transform: translateX(3px); }
        }

        .shake {
            animation: shake 0.5s ease;
        }

        @media (max-width: 768px) {
            .auth-container {
                flex-direction: column;
                gap: 20px;
            }
            .auth-char-column {
                order: -1;
                width: 100%;
            }
            .char-wrap {
                width: 200px;
                height: 247px;
            }
            .auth-form-column {
                max-width: 100%;
                width: 100%;
            }
        }

        /* Account profile bar inside sidebar */
        .user-profile-bar {
            margin-top: auto;
            padding-top: 16px;
            border-top: 1px solid var(--line);
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }

        .user-profile-info {
            display: flex;
            align-items: center;
            gap: 10px;
            min-width: 0;
        }

        .user-profile-avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            object-fit: cover;
            border: 1px solid var(--line);
        }

        .user-profile-details {
            display: grid;
            gap: 1px;
            min-width: 0;
        }

        .user-profile-details strong {
            font-size: 0.8rem;
            font-weight: 700;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .user-profile-details span {
            color: var(--muted);
            font-size: 0.7rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .btn-logout {
            padding: 6px 10px;
            font-size: 0.75rem;
            border-radius: 8px;
        }

        /* Mic button inside cylindrical search bar */
        .mic-btn {
            appearance: none;
            border: none;
            background: transparent;
            padding: 0;
            cursor: pointer;
            outline: none;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 38px;
            height: 38px;
            border-radius: 50%;
            color: var(--text-soft);
            transition: all 0.2s ease;
            flex-shrink: 0;
        }
        .mic-btn svg {
            width: 20px;
            height: 20px;
        }
        .mic-btn:hover {
            color: var(--accent);
            background: var(--accent-soft);
            transform: scale(1.05);
        }
        .mic-btn.active {
            color: #ffffff;
            background: var(--accent);
        }

        /* Message Action Buttons under Tarka response */
        .message-actions {
            display: flex;
            gap: 8px;
            margin-top: 10px;
            align-items: center;
        }

        .msg-action-btn {
            appearance: none;
            border: none;
            background: transparent;
            color: var(--text-soft);
            cursor: pointer;
            padding: 6px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.15s ease;
            outline: none;
        }

        .msg-action-btn:hover {
            color: var(--accent);
            background: var(--accent-soft);
        }

        .msg-action-btn svg {
            width: 16px;
            height: 16px;
        }

        #open_settings_btn:hover {
            color: var(--accent);
            background: var(--accent-soft);
        }

        .settings-modal-body {
            padding: 16px 0;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .settings-form-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .settings-form-group label {
            font-weight: 600;
            font-size: 0.9rem;
            color: var(--text);
        }

        .settings-form-group input[type="text"],
        .settings-form-group select {
            width: 100%;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 10px;
            background: var(--panel-strong);
            color: var(--text);
            font-family: inherit;
            outline: none;
            transition: border-color 0.2s;
        }

        .settings-form-group input[type="text"]:focus,
        .settings-form-group select:focus {
            border-color: var(--accent);
        }

        .settings-checkbox-label {
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 600;
            font-size: 0.9rem;
            color: var(--text);
            cursor: pointer;
        }

        .settings-checkbox-label input[type="checkbox"] {
            width: 16px;
            height: 16px;
            cursor: pointer;
        }

        .settings-help-text {
            font-size: 0.8rem;
            color: var(--text-soft);
            margin-top: 2px;
            padding-left: 24px;
        }

    </style>
    <script src="https://accounts.google.com/gsi/client" async defer></script>
</head>
<body>
    <!-- Authentication Overlay (Full-screen Mochi Login Page) -->
    <div class="auth-overlay" id="auth_overlay" style="display: none;">
        <!-- Theme Toggle specifically for the Login Screen -->
        <button class="auth-theme-toggle" id="auth_theme_toggle" type="button" aria-label="Toggle dark mode">
            <span class="icon active" id="auth_sun_icon"><img src="/static/SunToggle.png" alt="Light"></span>
            <span class="icon" id="auth_moon_icon"><img src="/static/MoonToggle.png" alt="Dark"></span>
        </button>

        <div class="auth-container">
            <!-- Left Side: Login Form Columns -->
            <div class="auth-form-column">
                <!-- Step 1: Login -->
                <div class="auth-card" id="login_card">
                    <div class="auth-logo">Tarka</div>
                    <h1 class="auth-title">Welcome to Tarka</h1>
                    <p class="auth-desc">Please sign in with your Google account.</p>
                    
                    <div id="auth_error" class="auth-error"></div>
                    
                    <div style="display: flex; justify-content: center; margin: 10px 0; min-height: 40px;" id="google_signin_wrapper">
                        <div id="g_id_onload"
                             data-client_id="GOOGLE_CLIENT_ID_PLACEHOLDER"
                             data-context="signin"
                             data-ux_mode="popup"
                             data-callback="handleCredentialResponse"
                             data-auto_prompt="false">
                        </div>
                        <div class="g_id_signin"
                             data-type="standard"
                             data-shape="pill"
                             data-theme="filled_blue"
                             data-text="signin_with"
                             data-size="large"
                             data-logo_alignment="left">
                        </div>
                    </div>

                    <div id="mock_login_toggle_wrapper" style="display: none; justify-content: center; margin-top: 14px; margin-bottom: 14px;">
                        <button id="btn_toggle_mock" type="button" style="background: none; border: none; color: var(--mochi-muted); font-size: 0.8rem; text-decoration: underline; cursor: pointer;">
                            Bypass with Mock Login
                        </button>
                    </div>

                    <!-- Dev mode / mock sign-in when GOOGLE_CLIENT_ID is not configured -->
                    <div class="dev-login-box" id="dev_login_box" style="display: none;">
                        <p>Developer Mock Login</p>
                        <div class="auth-form-group">
                            <input type="email" id="mock_email" class="auth-input" placeholder="Sarah.example@gmail.com" />
                        </div>
                        <button class="btn-mochi" id="btn_mock_login" type="button" style="margin-top: 8px;">Mock Sign In</button>
                    </div>
                </div>

                <!-- Step 2: Onboarding -->
                <div class="auth-card" id="onboarding_card" style="display: none;">
                    <div class="auth-logo">Tarka</div>
                    <h1 class="auth-title">Complete Your Profile</h1>
                    <p class="auth-desc">Just a couple of details to personalize your research experience.</p>
                    
                    <div id="onboarding_error" class="auth-error"></div>

                    <form id="onboarding_form" onsubmit="event.preventDefault();">
                        <div class="auth-form-group" style="margin-bottom: 16px;">
                            <label for="onboarding_dob">Date of Birth</label>
                            <input type="date" id="onboarding_dob" class="auth-input" required />
                        </div>
                        <div class="auth-form-group" style="margin-bottom: 24px;">
                            <label for="onboarding_preferred_name">What should Tarka call you?</label>
                            <input type="text" id="onboarding_preferred_name" class="auth-input" placeholder="e.g. Swaroop, Doctor, Captain" required />
                        </div>
                        <button class="btn-mochi" id="btn_submit_onboarding" type="submit">Complete Setup</button>
                    </form>
                </div>
            </div>

            <!-- Right Side: Mochi Character Column -->
            <div class="auth-char-column">
                <div class="char-wrap" id="charWrap">
                    <img id="img-base" src="/static/original.png" alt="character">
                    <img id="img-eyes" src="/static/eyes.png" alt="character eyes">
                    <img id="img-expr" src="" alt="character expression">
                </div>
            </div>
        </div>
    </div>

    <div class="drawer-overlay" id="drawer_overlay"></div>
    
    <button class="drawer-toggle" id="drawer_toggle" type="button" aria-label="Toggle sidebar">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="9" y1="3" x2="9" y2="21"></line></svg>
    </button>

    <aside class="card panel sidebar" id="sidebar_drawer">
        <div class="sidebar-header">
            <div style="display:flex; justify-content:space-between; align-items:center; width:100%; margin-bottom: 8px;">
                <h2 style="padding-left: 48px;">Tarka AI</h2>
                <button class="drawer-close" id="drawer_close" type="button" aria-label="Close sidebar">&times;</button>
            </div>
            <div class="actions" style="gap:10px; justify-content:space-between; width:100%; margin-top: 8px;">
                <label class="theme-toggle" for="theme_toggle">
                    <input id="theme_toggle" type="checkbox" />
                    <span class="theme-switch" aria-hidden="true"></span>
                    <span>Dark</span>
                </label>
                <button class="btn btn-secondary" id="new_session" type="button">New session</button>
            </div>
        </div>
        <div class="history-list" id="history_list"></div>

        <div class="maker-card" aria-label="Maker information" style="margin-bottom: 16px;">
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

        <!-- User Profile info at the bottom -->
        <div class="user-profile-bar" id="user_profile_bar" style="display: none;">
            <div class="user-profile-info">
                <img class="user-profile-avatar" id="user_avatar" src="" alt="User avatar" />
                <div class="user-profile-details">
                    <strong id="user_name">User Name</strong>
                    <span id="user_email">user@example.com</span>
                </div>
            </div>
            <div style="display: flex; gap: 8px; align-items: center;">
                <button class="btn-icon" id="open_settings_btn" type="button" title="Settings" style="background: transparent; border: none; cursor: pointer; color: var(--text-soft); display: flex; align-items: center; justify-content: center; padding: 6px; border-radius: 6px; transition: all 0.2s ease;">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="3"></circle>
                        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
                    </svg>
                </button>
                <button class="btn btn-secondary btn-logout" id="btn_logout" type="button">Logout</button>
            </div>
        </div>
    </aside>

    <div class="main-content">
        <div class="wrap">
            <div class="header-bar">
                <div class="header-logo">Tarka</div>
                <div class="chat-header-meta" id="chat_header_meta">
                    <div>
                        <span class="session-badge" id="session_badge">Session 1</span>
                        <strong id="status" class="status status-ready">Ready.</strong>
                    </div>
                    <span id="scroll_hint">Scroll through the transcript below.</span>
                </div>
            </div>
            <div class="workspace">

            <div class="main-column">
                <section class="card chat-shell">
                    <div class="chat-header" id="chat_header">
                        <div>
                            <div class="eyebrow">Tarka</div>
                            <h1>Research in a conversation.</h1>
                            <p class="lead">
                                Ask follow-ups, keep the thread.
                            </p>
                        </div>
                    </div>

                    <div class="chat-messages" id="chat_messages"></div>

                    <div class="composer">
                        <div class="chips" id="example_chips" aria-label="Example queries">
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

                        <div class="query-box-container">
                            <div class="mode-pill-selector">
                                <button type="button" class="mode-trigger-btn" id="mode_trigger_btn" title="Change Mode">
                                    <span class="mode-icon-circle" id="current_mode_icon">⚡</span>
                                </button>
                                <div class="mode-dropdown-menu" id="mode_dropdown_menu">
                                    <button type="button" class="mode-dropdown-item" data-value="flash">⚡ Flash</button>
                                    <button type="button" class="mode-dropdown-item" data-value="research">🔍 Research</button>
                                    <button type="button" class="mode-dropdown-item" data-value="thesis">🎓 Thesis Mode</button>
                                </div>
                            </div>

                            <button type="button" class="mic-btn" id="voice_mode_btn" title="Voice Mode">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                                    <line x1="12" y1="19" x2="12" y2="23"></line>
                                    <line x1="8" y1="23" x2="16" y2="23"></line>
                                </svg>
                            </button>

                            <textarea id="query" placeholder="Ask a follow-up or start a new research session..."></textarea>

                            <button class="send-btn" id="run" type="button" title="Send">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                                    <line x1="22" y1="2" x2="11" y2="13"></line>
                                    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                                </svg>
                            </button>
                        </div>
                    </div>
                </section>
            </div>
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

    <div class="sources-modal" id="settings_modal" aria-hidden="true">
        <div class="sources-modal-card" role="dialog" aria-modal="true" aria-labelledby="settings_modal_title">
            <div class="sources-modal-header">
                <div>
                    <h3 id="settings_modal_title">Settings</h3>
                    <p>Customize your profile and AI behaviors.</p>
                </div>
                <button class="sources-modal-close" id="settings_modal_close" type="button" aria-label="Close settings popup">×</button>
            </div>
            
            <div class="settings-modal-body">
                <div class="settings-form-group">
                    <label for="settings_preferred_name">Preferred Name</label>
                    <input type="text" id="settings_preferred_name" placeholder="e.g. Swaroop" style="width: 100%; border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: var(--panel-strong); color: var(--text);" />
                </div>
                
                <div class="settings-form-group" style="margin-top: 8px;">
                    <label class="settings-checkbox-label">
                        <input id="use_memory" type="checkbox" checked />
                        Use memory cache
                    </label>
                    <span class="settings-help-text">Enables using context from previous turns saved in storage.</span>
                </div>
                
                <div class="settings-form-group">
                    <label for="memory_mode">Memory Mode</label>
                    <select id="memory_mode" aria-label="Memory mode">
                        <option value="balanced" selected>Balanced</option>
                        <option value="prefer_memory">Prefer memory</option>
                        <option value="search_only">Search only</option>
                    </select>
                </div>
            </div>
            
            <div style="display: flex; justify-content: flex-end; gap: 10px; margin-top: 8px;">
                <button class="btn btn-secondary" id="settings_modal_cancel" type="button">Cancel</button>
                <button class="btn btn-primary" id="settings_modal_save" type="button" style="background: var(--accent); color: #fff;">Save Changes</button>
            </div>
        </div>
    </div>

    <div class="sources-modal" id="mode_onboarding_modal" aria-hidden="true">
        <div class="sources-modal-card" role="dialog" aria-modal="true" aria-labelledby="mode_onboarding_modal_title" style="max-width: 550px;">
            <div class="sources-modal-header">
                <div>
                    <h3 id="mode_onboarding_modal_title">Welcome to Tarka AI!</h3>
                    <p>Select the right mode for your task.</p>
                </div>
                <button class="sources-modal-close" id="mode_onboarding_modal_close" type="button" aria-label="Close onboarding modal">×</button>
            </div>
            
            <div style="padding: 16px 0; display: flex; flex-direction: column; gap: 16px;">
                <p style="font-size: 0.9rem; color: var(--text-soft); margin-bottom: 8px;">
                    Tarka offers three distinct modes designed for different research workflows. You can change them anytime using the mode icon on the search bar.
                </p>
                
                <div style="display: flex; flex-direction: column; gap: 12px;">
                    <div style="background: var(--panel-strong); border: 1px solid var(--line); border-radius: 12px; padding: 12px; display: flex; gap: 12px;">
                        <span style="font-size: 1.5rem; flex-shrink: 0;">⚡</span>
                        <div>
                            <strong style="display: block; font-size: 0.95rem; color: var(--text); margin-bottom: 2px;">Flash (Default)</strong>
                            <span style="font-size: 0.85rem; color: var(--text-soft);">Best for quick, direct, and concise answers in a natural conversational flow. Bypasses deep iterations to save time and credits.</span>
                        </div>
                    </div>
                    
                    <div style="background: var(--panel-strong); border: 1px solid var(--line); border-radius: 12px; padding: 12px; display: flex; gap: 12px;">
                        <span style="font-size: 1.5rem; flex-shrink: 0;">🔍</span>
                        <div>
                            <strong style="display: block; font-size: 0.95rem; color: var(--text); margin-bottom: 2px;">Research</strong>
                            <span style="font-size: 0.85rem; color: var(--text-soft);">Best for structured, multi-dimensional queries. Evaluates claims, verifies evidence, and summarizes by key dimensions.</span>
                        </div>
                    </div>
                    
                    <div style="background: var(--panel-strong); border: 1px solid var(--line); border-radius: 12px; padding: 12px; display: flex; gap: 12px;">
                        <span style="font-size: 1.5rem; flex-shrink: 0;">🎓</span>
                        <div>
                            <strong style="display: block; font-size: 0.95rem; color: var(--text); margin-bottom: 2px;">Thesis Mode</strong>
                            <span style="font-size: 0.85rem; color: var(--text-soft);">Best for academic and scientific writing. Refines your research ideas into a highly technical, structured abstract.</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div style="display: flex; justify-content: flex-end; margin-top: 12px;">
                <button class="btn btn-primary" id="mode_onboarding_modal_btn" type="button" style="background: var(--accent); color: #fff; padding: 10px 20px;">Got it, let's go!</button>
            </div>
        </div>
    </div>

    <script>
        const queryEl = document.getElementById('query');
        const modeTriggerBtn = document.getElementById('mode_trigger_btn');
        const modeDropdownMenu = document.getElementById('mode_dropdown_menu');
        const currentModeIcon = document.getElementById('current_mode_icon');
        const useMemoryEl = document.getElementById('use_memory');
        const memoryModeEl = document.getElementById('memory_mode');
        const statusEl = document.getElementById('status');
        const runBtn = document.getElementById('run');

        let selectedMode = 'flash';

        modeTriggerBtn.addEventListener('click', (event) => {
            event.stopPropagation();
            modeDropdownMenu.classList.toggle('open');
        });

        document.querySelectorAll('.mode-dropdown-item').forEach((item) => {
            item.addEventListener('click', (event) => {
                event.stopPropagation();
                const value = item.dataset.value;
                selectedMode = value;
                
                if (value === 'flash') currentModeIcon.textContent = '⚡';
                else if (value === 'research') currentModeIcon.textContent = '🔍';
                else if (value === 'thesis') currentModeIcon.textContent = '🎓';

                modeDropdownMenu.classList.remove('open');
                setStatus(`Mode changed to ${value.charAt(0).toUpperCase() + value.slice(1)}.`);
            });
        });

        document.addEventListener('click', () => {
            modeDropdownMenu.classList.remove('open');
        });

        const autoResizeQuery = () => {
            queryEl.style.height = '38px';
            queryEl.style.height = (queryEl.scrollHeight) + 'px';
        };
        queryEl.addEventListener('input', autoResizeQuery);
        const sessionListEl = document.getElementById('history_list');
        const newSessionBtn = document.getElementById('new_session');
        const openSettingsBtnEl = document.getElementById('open_settings_btn');
        const settingsModalEl = document.getElementById('settings_modal');
        const settingsModalCloseEl = document.getElementById('settings_modal_close');
        const settingsModalCancelEl = document.getElementById('settings_modal_cancel');
        const settingsModalSaveEl = document.getElementById('settings_modal_save');
        const settingsPreferredNameEl = document.getElementById('settings_preferred_name');
        const modeOnboardingModalEl = document.getElementById('mode_onboarding_modal');
        const modeOnboardingModalCloseEl = document.getElementById('mode_onboarding_modal_close');
        const modeOnboardingModalBtnEl = document.getElementById('mode_onboarding_modal_btn');
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
        const voiceModeBtnEl = document.getElementById('voice_mode_btn');
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
        const drawerToggleEl = document.getElementById('drawer_toggle');
        const drawerCloseEl = document.getElementById('drawer_close');
        const drawerOverlayEl = document.getElementById('drawer_overlay');
        const sidebarDrawerEl = document.getElementById('sidebar_drawer');

        // Authentication DOM selectors
        const authOverlayEl = document.getElementById('auth_overlay');
        const loginCardEl = document.getElementById('login_card');
        const onboardingCardEl = document.getElementById('onboarding_card');
        const authErrorEl = document.getElementById('auth_error');
        const onboardingErrorEl = document.getElementById('onboarding_error');
        const devLoginBoxEl = document.getElementById('dev_login_box');
        const mockEmailEl = document.getElementById('mock_email');
        const btnMockLoginEl = document.getElementById('btn_mock_login');
        const onboardingDobEl = document.getElementById('onboarding_dob');
        const onboardingPreferredNameEl = document.getElementById('onboarding_preferred_name');
        const btnSubmitOnboardingEl = document.getElementById('btn_submit_onboarding');
        const userProfileBarEl = document.getElementById('user_profile_bar');
        const userAvatarEl = document.getElementById('user_avatar');
        const userNameEl = document.getElementById('user_name');
        const userEmailEl = document.getElementById('user_email');
        const btnLogoutEl = document.getElementById('btn_logout');
        const btnToggleMockEl = document.getElementById('btn_toggle_mock');
        const authThemeToggleEl = document.getElementById('auth_theme_toggle');
        const authSunIconEl = document.getElementById('auth_sun_icon');
        const authMoonIconEl = document.getElementById('auth_moon_icon');

        const HISTORY_KEY = 'tarka-chat-sessions';
        const ACTIVE_SESSION_KEY = 'tarka-active-session';
        const THEME_KEY = 'research-theme';
        const VOICE_KEY = 'tarka-voice-settings';
        const SESSION_TOKEN_KEY = 'tarka-session-token';

        let currentUser = null;
        let sessionToken = localStorage.getItem(SESSION_TOKEN_KEY) || '';

        // Mochi Character Variables & State
        const imgEyesEl = document.getElementById('img-eyes');
        const imgExprEl = document.getElementById('img-expr');
        const charWrapEl = document.getElementById('charWrap');
        
        const MOCHI_IMGS = {
            blink:     '/static/blink.png',
            closed:    '/static/closed.png',
            peek:      '/static/peek.png',
            mad:       '/static/mad.png',
            surprised: '/static/surprised.png',
            sleepy:    '/static/sleepy.png'
        };

        let mochiState = 'default';
        let mochiBlinkTimer = null;
        let mochiIdleTimer = null;
        let mochiResetTimer = null;

        const setMochiState = (s) => {
            if (mochiResetTimer && s !== 'blink') {
                clearTimeout(mochiResetTimer);
                mochiResetTimer = null;
            }
            mochiState = s;
            if (s === 'default') {
                if (imgEyesEl) imgEyesEl.style.display = '';
                if (imgExprEl) imgExprEl.classList.remove('visible');
                scheduleMochiBlink();
                startMochiIdleTimer();
            } else {
                if (s !== 'blink') {
                    if (imgEyesEl) imgEyesEl.style.display = 'none';
                }
                if (MOCHI_IMGS[s] && imgExprEl) {
                    imgExprEl.src = MOCHI_IMGS[s];
                    imgExprEl.classList.add('visible');
                }
            }
        };

        const scheduleMochiBlink = () => {
            clearTimeout(mochiBlinkTimer);
            mochiBlinkTimer = setTimeout(doMochiBlink, 3000 + Math.random() * 3000);
        };

        const doMochiBlink = async () => {
            if (mochiState !== 'default') {
                scheduleMochiBlink();
                return;
            }
            await flashMochiBlink();
            if (Math.random() < 0.25) {
                await new Promise(r => setTimeout(r, 60));
                await flashMochiBlink();
            }
            if (mochiState === 'default') scheduleMochiBlink();
        };

        const flashMochiBlink = () => {
            return new Promise(res => {
                if (imgEyesEl) imgEyesEl.style.display = 'none';
                if (imgExprEl) {
                    imgExprEl.src = MOCHI_IMGS.blink;
                    imgExprEl.classList.add('visible');
                }
                setTimeout(() => {
                    if (imgExprEl) imgExprEl.classList.remove('visible');
                    if (mochiState === 'default' && imgEyesEl) imgEyesEl.style.display = '';
                    res();
                }, 140);
            });
        };

        const startMochiIdleTimer = () => {
            clearTimeout(mochiIdleTimer);
            mochiIdleTimer = setTimeout(() => {
                if (mochiState === 'default') setMochiState('sleepy');
            }, 10000);
        };

        const resetMochiIdle = () => {
            if (mochiState === 'sleepy') setMochiState('default');
            startMochiIdleTimer();
        };

        document.addEventListener('mousemove', (e) => {
            resetMochiIdle();
            
            if (mochiState !== 'default' || !charWrapEl || !imgEyesEl) return;
            
            const rect = charWrapEl.getBoundingClientRect();
            if (rect.width === 0) return;
            
            const cx = rect.left + rect.width * 0.5;
            const cy = rect.top + rect.height * 0.42;
            const dx = e.clientX - cx;
            const dy = e.clientY - cy;
            const angle = Math.atan2(dy, dx);
            const dist = Math.min(Math.sqrt(dx*dx + dy*dy), 220);
            const f = dist / 220;
            const rawX = Math.cos(angle) * f * 14;
            const clampedX = rawX < 0 ? Math.max(rawX, -5) : rawX;
            imgEyesEl.style.transform = `translate(${clampedX.toFixed(1)}px,${(Math.sin(angle)*f*10).toFixed(1)}px)`;
        });

        document.addEventListener('keydown', resetMochiIdle);

        if (charWrapEl) {
            charWrapEl.addEventListener('click', () => {
                setMochiState('mad');
                mochiResetTimer = setTimeout(() => setMochiState('default'), 700);
            });
        }

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

        const closeSidebar = () => {
            document.body.classList.remove('sidebar-open');
            localStorage.setItem('tarka-sidebar-open', 'false');
            sidebarDrawerEl.classList.remove('open');
            drawerOverlayEl.classList.remove('open');
        };

        const toggleSidebar = () => {
            const isOpen = document.body.classList.toggle('sidebar-open');
            localStorage.setItem('tarka-sidebar-open', isOpen);
            sidebarDrawerEl.classList.toggle('open', isOpen);
            if (window.innerWidth < 1025) {
                drawerOverlayEl.classList.toggle('open', isOpen);
            } else {
                drawerOverlayEl.classList.remove('open');
            }
        };

        const initSidebar = () => {
            const saved = localStorage.getItem('tarka-sidebar-open');
            const isDesktop = window.innerWidth >= 1025;
            const shouldOpen = saved === null ? isDesktop : (saved === 'true');
            
            if (shouldOpen) {
                document.body.classList.add('sidebar-open');
                sidebarDrawerEl.classList.add('open');
                if (!isDesktop) {
                    drawerOverlayEl.classList.add('open');
                }
            } else {
                document.body.classList.remove('sidebar-open');
                sidebarDrawerEl.classList.remove('open');
                drawerOverlayEl.classList.remove('open');
            }
        };

        // Setup authentication handlers

        const updateAuthUI = (user) => {
            if (user) {
                currentUser = user;
                authOverlayEl.style.display = 'none';
                userProfileBarEl.style.display = 'flex';
                userAvatarEl.src = user.picture || 'https://api.dicebear.com/7.x/bottts/svg?seed=' + user.email;
                userNameEl.textContent = user.preferred_name || user.name;
                userEmailEl.textContent = user.email;

                // Load and render active sessions when signed in
                loadSessions();
                renderSessions();
                renderMessages();
            } else {
                currentUser = null;
                sessionToken = '';
                sessions = [];
                activeSessionId = '';
                localStorage.removeItem(SESSION_TOKEN_KEY);
                userProfileBarEl.style.display = 'none';
                authOverlayEl.style.display = 'flex';
                showLogin();
                setMochiState('default'); // Start Mochi loop!
                
                const clientId = "GOOGLE_CLIENT_ID_PLACEHOLDER";
                if (!clientId || clientId === "GOOGLE_CLIENT_ID_" + "PLACEHOLDER") {
                    document.getElementById('google_signin_wrapper').style.display = 'none';
                    devLoginBoxEl.style.display = 'block';
                    document.getElementById('mock_login_toggle_wrapper').style.display = 'none';
                } else {
                    document.getElementById('google_signin_wrapper').style.display = 'flex';
                    devLoginBoxEl.style.display = 'none';
                    document.getElementById('mock_login_toggle_wrapper').style.display = 'flex';
                }
            }
        };

        const handleAuthResponse = async (payload) => {
            if (payload.session_token) {
                sessionToken = payload.session_token;
                localStorage.setItem(SESSION_TOKEN_KEY, sessionToken);
                
                setMochiState('surprised'); // Mochi happy/surprised face on success!
                setTimeout(() => {
                    if (payload.first_time) {
                        showOnboarding();
                        setMochiState('default');
                    } else {
                        updateAuthUI(payload);
                    }
                }, 1200);
            } else {
                showAuthError("Authentication failed: No session token received.");
            }
        };

        const showAuthError = (msg) => {
            authErrorEl.textContent = msg;
            authErrorEl.style.display = 'block';
            
            // Mochi angry reaction and card shake
            setMochiState('mad');
            loginCardEl.classList.remove('shake');
            void loginCardEl.offsetWidth; // force reflow
            loginCardEl.classList.add('shake');
            mochiResetTimer = setTimeout(() => setMochiState('default'), 2500);
        };

        const showOnboardingError = (msg) => {
            onboardingErrorEl.textContent = msg;
            onboardingErrorEl.style.display = 'block';
            
            // Mochi angry reaction and card shake
            setMochiState('mad');
            onboardingCardEl.classList.remove('shake');
            void onboardingCardEl.offsetWidth; // force reflow
            onboardingCardEl.classList.add('shake');
            mochiResetTimer = setTimeout(() => setMochiState('default'), 2500);
        };

        const showOnboarding = () => {
            loginCardEl.style.display = 'none';
            onboardingCardEl.style.display = 'block';
        };

        const showLogin = () => {
            loginCardEl.style.display = 'block';
            onboardingCardEl.style.display = 'none';
        };

        window.handleCredentialResponse = async (response) => {
            authErrorEl.style.display = 'none';
            try {
                const res = await fetch('/auth/google', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ credential: response.credential })
                });
                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || 'Google sign in failed');
                }
                const data = await res.json();
                await handleAuthResponse(data);
            } catch (err) {
                showAuthError(err.message);
            }
        };

        const performMockLogin = async () => {
            authErrorEl.style.display = 'none';
            const email = mockEmailEl.value.trim();
            if (!email) {
                showAuthError("Please enter an email.");
                return;
            }
            try {
                const res = await fetch('/auth/google', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ mock_email: email })
                });
                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || 'Mock sign in failed');
                }
                const data = await res.json();
                await handleAuthResponse(data);
            } catch (err) {
                showAuthError(err.message);
            }
        };

        const performOnboarding = async () => {
            onboardingErrorEl.style.display = 'none';
            const dob = onboardingDobEl.value;
            const preferredName = onboardingPreferredNameEl.value.trim();
            if (!dob || !preferredName) {
                showOnboardingError("All fields are required.");
                return;
            }
            
            const birthDate = new Date(dob);
            const today = new Date();
            let age = today.getFullYear() - birthDate.getFullYear();
            const m = today.getMonth() - birthDate.getMonth();
            if (m < 0 || (m === 0 && today.getDate() < birthDate.getDate())) {
                age--;
            }
            if (age < 13) {
                showOnboardingError("You must be at least 13 years old to use Tarka.");
                return;
            }

            try {
                const res = await fetch(`/auth/complete-setup?session_token=${encodeURIComponent(sessionToken)}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ dob, preferred_name: preferredName })
                });
                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || 'Setup completion failed');
                }
                const data = await res.json();
                
                setMochiState('surprised'); // Mochi happy/surprised face on success!
                setTimeout(() => {
                    updateAuthUI(data);
                    showModeOnboardingModal();
                }, 1200);
            } catch (err) {
                showOnboardingError(err.message);
            }
        };

        const validateSession = async () => {
            if (!sessionToken) {
                updateAuthUI(null);
                return;
            }
            try {
                const res = await fetch(`/auth/user-profile?session_token=${encodeURIComponent(sessionToken)}`);
                if (!res.ok) {
                    throw new Error("Session expired");
                }
                const user = await res.json();
                updateAuthUI(user);
            } catch {
                updateAuthUI(null);
            }
        };

        const performLogout = () => {
            updateAuthUI(null);
        };

        let sessions = [];
        let activeSessionId = '';
        let activeStream = null;
        let isExecuting = false;
        let activeReject = null;
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

        let typingQueue = [];
        let typingTimer = null;
        let currentTypingMsg = null;
        let currentTargetVal = '';

        const flushTypingQueue = () => {
            if (typingTimer) {
                clearInterval(typingTimer);
                typingTimer = null;
            }
            if (currentTypingMsg) {
                if (currentTypingMsg.domContent) {
                    currentTypingMsg.domContent.textContent = currentTargetVal || currentTypingMsg.content || '';
                    currentTypingMsg.domContent.parentElement.classList.remove('typing');
                }
                currentTypingMsg.content = currentTargetVal || currentTypingMsg.content || '';
            }
            typingQueue = [];
            currentTypingMsg = null;
            currentTargetVal = '';
        };

        const enqueueTextForTyping = (message, newText) => {
            if (currentTypingMsg && currentTypingMsg.id !== message.id) {
                flushTypingQueue();
            }

            if (!currentTypingMsg) {
                currentTypingMsg = message;
                currentTargetVal = message.domContent ? message.domContent.textContent : '';
                if (currentTargetVal === 'Thinking...') {
                    currentTargetVal = '';
                }
            }

            for (let i = 0; i < newText.length; i++) {
                typingQueue.push(newText[i]);
            }

            if (message.domContent) {
                message.domContent.parentElement.classList.add('typing');
            }

            if (!typingTimer) {
                typingTimer = setInterval(() => {
                    if (typingQueue.length > 0) {
                        const speed = typingQueue.length > 30 ? 4 : typingQueue.length > 10 ? 2 : 1;
                        for (let i = 0; i < speed && typingQueue.length > 0; i++) {
                            const char = typingQueue.shift();
                            currentTargetVal += char;
                        }
                        if (message.domContent) {
                            message.domContent.textContent = currentTargetVal;
                        }
                        chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
                    } else if (message.isStreaming === false) {
                        clearInterval(typingTimer);
                        typingTimer = null;
                        if (message.domContent) {
                            message.domContent.parentElement.classList.remove('typing');
                        }
                        message.content = currentTargetVal;
                        currentTypingMsg = null;
                        currentTargetVal = '';
                        saveSessions();
                    }
                }, 12);
            }
        };

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
        const setStatus = (text) => {
            statusEl.textContent = text;
            statusEl.className = 'status';
            const lower = text.toLowerCase();
            if (lower.includes('planning') || lower.includes('searching') || lower.includes('summarizing') || lower.includes('reviewing') || lower.includes('finishing') || lower.includes('streaming')) {
                statusEl.classList.add('status-busy');
            } else if (lower.includes('error') || lower.includes('failed')) {
                statusEl.classList.add('status-error');
            } else if (lower.includes('ready') || lower.includes('complete') || lower.includes('cache')) {
                statusEl.classList.add('status-ready');
            } else {
                statusEl.classList.add('status-idle');
            }
        };



        const applyTheme = (theme) => {
            document.body.dataset.theme = theme;
            themeToggleEl.checked = theme === 'dark';
            if (typeof syncAuthThemeUI === 'function') {
                syncAuthThemeUI(theme);
            }
        };

        const loadTheme = () => {
            const savedTheme = localStorage.getItem(THEME_KEY);
            if (savedTheme === 'dark' || savedTheme === 'light') {
                return savedTheme;
            }

            return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        };

        const getHistoryKey = () => {
            if (currentUser && currentUser.google_id) {
                return HISTORY_KEY + '-' + currentUser.google_id;
            }
            return HISTORY_KEY;
        };

        const getActiveSessionKey = () => {
            if (currentUser && currentUser.google_id) {
                return ACTIVE_SESSION_KEY + '-' + currentUser.google_id;
            }
            return ACTIVE_SESSION_KEY;
        };

        const loadSessions = () => {
            try {
                const raw = localStorage.getItem(getHistoryKey());
                sessions = raw ? JSON.parse(raw) : [];
                if (!Array.isArray(sessions)) {
                    sessions = [];
                }
            } catch {
                sessions = [];
            }

            // Clean up any empty sessions from the loaded history
            sessions = sessions.filter(session => session.messages && session.messages.length > 0);

            // Automatically start a brand new session on page load/login
            const starter = createSession();
            sessions.unshift(starter);
            activeSessionId = starter.id;
        };

        const saveSessions = () => {
            // Only save sessions with messages to localStorage
            const sessionsToSave = sessions.filter(session => session.messages && session.messages.length > 0);
            localStorage.setItem(getHistoryKey(), JSON.stringify(sessionsToSave.slice(0, MAX_SESSIONS)));
            localStorage.setItem(getActiveSessionKey(), activeSessionId);
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
            flushTypingQueue();
            // Discard any empty sessions when switching away, except the one being activated (just in case)
            sessions = sessions.filter(session => session.id === sessionId || (session.messages && session.messages.length > 0));
            activeSessionId = sessionId;
            const session = getActiveSession();
            if (session) {
                session.updated_at = nowIso();
            }
            saveSessions();
            renderSessions();
            renderMessages();
            if (window.innerWidth < 1025) {
                closeSidebar();
            }
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

        const setVoicePanelOpen = (isOpen) => {
            voicePanelEl.classList.toggle('open', isOpen);
            voiceModeBtnEl.classList.toggle('active', isOpen);
            voiceModeBtnEl.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
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
            setVoicePanelOpen(false);
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
                autoResizeQuery();
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
                autoResizeQuery();

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

            const loggedSessions = sessions.filter(session => session.messages && session.messages.length > 0);

            if (!loggedSessions.length) {
                const empty = document.createElement('div');
                empty.className = 'history-empty';
                empty.textContent = 'Start a session to keep a running chat log.';
                sessionListEl.appendChild(empty);
                return;
            }

            loggedSessions.slice().sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at)).forEach((session) => {
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
                menuBackdrop.addEventListener('click', (event) => {
                    event.stopPropagation();
                    toggleMenu(false);
                });

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

            const hasMessages = session && session.messages && session.messages.length > 0;
            const chatHeaderEl = document.getElementById('chat_header');
            if (chatHeaderEl) {
                if (hasMessages) {
                    chatHeaderEl.classList.add('collapsed');
                } else {
                    chatHeaderEl.classList.remove('collapsed');
                }
            }

            const chipsEl = document.getElementById('example_chips');
            if (chipsEl) {
                if (hasMessages) {
                    chipsEl.classList.add('collapsed');
                } else {
                    chipsEl.classList.remove('collapsed');
                }
            }

            if (!session || !session.messages.length) {
                const welcomeContainer = document.createElement('div');
                welcomeContainer.className = 'welcome-container';

                const hour = new Date().getHours();
                const displayName = currentUser ? (currentUser.preferred_name || currentUser.name || '') : '';
                const nameStr = displayName ? displayName : '';

                let welcomeTitle = '';
                let welcomeSubtitle = '';

                if (hour >= 0 && hour < 5) {
                    // Late night (12 am - 5 am)
                    welcomeTitle = nameStr ? `up late ${nameStr}?` : 'up late?';
                    welcomeSubtitle = 'How can i help you?';
                } else if (hour >= 5 && hour < 7) {
                    // Early morning (5 am - 7 am)
                    welcomeTitle = nameStr ? `Damnn, you are up early ${nameStr}` : 'Damnn, you are up early';
                    welcomeSubtitle = 'How can i help you';
                } else {
                    // Non time-bound: rotates every 4 hours
                    const quirkyGreetings = [
                        {
                            title: nameStr ? `${nameStr} , returns !!!` : 'Returned !!!',
                            subtitle: 'What are you upto today'
                        },
                        {
                            title: nameStr ? `Hey ${nameStr}!!` : 'Hey there!!',
                            subtitle: "What's on your agenda today ?"
                        },
                        {
                            title: nameStr ? `Welcome back, ${nameStr}.` : 'Welcome back.',
                            subtitle: 'What can I help you research today?'
                        },
                        {
                            title: nameStr ? `Tarka is ready, ${nameStr}!` : 'Tarka is ready!',
                            subtitle: 'What shall we discover today?'
                        },
                        {
                            title: nameStr ? `Ah, ${nameStr}, prompt engineer extraordinaire!` : 'Ah, prompt engineer extraordinaire!',
                            subtitle: 'What are we looking up today?'
                        },
                        {
                            title: nameStr ? `Back for more knowledge, ${nameStr}?` : 'Back for more knowledge?',
                            subtitle: 'What can I help you research today?'
                        }
                    ];
                    const index = Math.floor(hour / 4) % quirkyGreetings.length;
                    welcomeTitle = quirkyGreetings[index].title;
                    welcomeSubtitle = quirkyGreetings[index].subtitle;
                }

                const title = document.createElement('h1');
                title.className = 'welcome-title';
                title.textContent = welcomeTitle;

                const subtitle = document.createElement('p');
                subtitle.className = 'welcome-subtitle';
                subtitle.textContent = welcomeSubtitle;

                welcomeContainer.append(title, subtitle);
                chatMessagesEl.appendChild(welcomeContainer);

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

                if (message.role === 'assistant' && message.research_mode !== 'flash' && message.research_mode !== 'thesis' && (typeof message.evidence_coverage === 'number' || typeof message.avg_confidence === 'number')) {
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

                if (message.role === 'assistant' && !message.isStreaming) {
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

                if (message.role === 'assistant' && !message.isStreaming && message.content) {
                    const actionRow = document.createElement('div');
                    actionRow.className = 'message-actions';

                    const copyBtn = document.createElement('button');
                    copyBtn.type = 'button';
                    copyBtn.className = 'msg-action-btn';
                    copyBtn.title = 'Copy response to clipboard';
                    copyBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>`;
                    copyBtn.addEventListener('click', () => {
                        navigator.clipboard.writeText(message.content || '').then(() => {
                            setStatus('Answer copied to clipboard.');
                        });
                    });

                    const shareBtn = document.createElement('button');
                    shareBtn.type = 'button';
                    shareBtn.className = 'msg-action-btn';
                    shareBtn.title = 'Share';
                    shareBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="5" r="3"></circle><circle cx="6" cy="12" r="3"></circle><circle cx="18" cy="19" r="3"></circle><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line></svg>`;
                    shareBtn.addEventListener('click', () => {
                        openShareModal(buildShareText(session));
                        setStatus('Choose a platform to share.');
                    });

                    const exportBtn = document.createElement('button');
                    exportBtn.type = 'button';
                    exportBtn.className = 'msg-action-btn';
                    exportBtn.title = 'Export session';
                    exportBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>`;
                    exportBtn.addEventListener('click', () => {
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

                    actionRow.append(copyBtn, shareBtn, exportBtn);
                    bubble.appendChild(actionRow);
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

        const stopExecution = () => {
            stopActiveStream();
            flushTypingQueue();
            if (activeReject) {
                activeReject(new Error('User stopped execution.'));
                activeReject = null;
            }
        };

        const startNewSession = () => {
            flushTypingQueue();
            stopActiveStream();

            // If the current active session is already empty, just keep using it!
            const currentActive = getActiveSession();
            if (currentActive && (!currentActive.messages || currentActive.messages.length === 0)) {
                queryEl.value = '';
                queryEl.focus();
                setStatus('Ready for a new session.');
                return;
            }

            // Discard any other empty sessions from memory
            sessions = sessions.filter(session => session.messages && session.messages.length > 0);

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

        const streamAnswer = ({ query, context, useMemory, researchMode }) => new Promise((resolve, reject) => {
            activeReject = reject;
            const params = new URLSearchParams({
                query,
                context,
                use_memory: useMemory ? '1' : '0',
                memory_mode: memoryModeEl.value,
                session_token: sessionToken,
                research_mode: researchMode,
            });

            const source = new EventSource(`/research/stream?${params.toString()}`);
            activeStream = source;
            let finished = false;

            const finish = (payload) => {
                finished = true;
                activeReject = null;
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
                            enqueueTextForTyping(assistant, payload.data?.delta || '');
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
                activeReject = null;
                stopActiveStream();
                reject(new Error('Streaming connection failed.'));
            };
        });

        const sendMessage = async () => {
            if (isExecuting) {
                stopExecution();
                return;
            }
            flushTypingQueue();
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
                research_mode: selectedMode,
                created_at: nowIso(),
            };

            session.messages.push(userMessage, assistantMessage);
            // If the title is default auto-generated ("Session #n" or "New session" or empty), auto-name it based on the first query
            const isDefaultTitle = !session.title || session.title === 'New session' || /^Session \d+$/.test(session.title);
            if (isDefaultTitle) {
                session.title = shortPreview(query, 30);
                fetch('/research/title', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query })
                })
                .then(res => res.json())
                .then(data => {
                    if (data && data.title) {
                        session.title = data.title;
                        saveSessions();
                        renderSessions();
                    }
                })
                .catch(err => console.error('Error auto-generating session title:', err));
            }
            session.updated_at = nowIso();
            activeAssistantMessageId = assistantMessage.id;
            activeRequestId = null;
            saveSessions();
            renderSessions();
            renderMessages();

            queryEl.value = '';
            queryEl.style.height = '38px';
            voiceDraftEl.value = '';
            
            isExecuting = true;
            runBtn.disabled = false;
            runBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect></svg>';
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
                    researchMode: selectedMode,
                });

                const assistant = session.messages.find((message) => message.id === activeAssistantMessageId);
                if (assistant) {
                    assistant.source_urls = finalPayload.source_urls || assistant.source_urls || [];
                    assistant.claims = finalPayload.claims || [];
                    assistant.evidence_coverage = typeof finalPayload.evidence_coverage === 'number' ? finalPayload.evidence_coverage : null;
                    assistant.avg_confidence = typeof finalPayload.avg_confidence === 'number' ? finalPayload.avg_confidence : null;
                    assistant.isStreaming = false; // Trigger typewriter completion
                    
                    if (!typingTimer) {
                        assistant.content = finalPayload.final_answer || assistant.content || 'No answer generated.';
                        saveSessions();
                        renderMessages();
                        renderSessions();
                    }
                }

                activeRequestId = finalPayload.request_id || activeRequestId;
                setStatus(finalPayload.from_memory ? 'Answered from memory cache.' : 'Research complete.');
            } catch (error) {
                updateActiveAssistant({
                    content: error.message || 'Something went wrong while streaming the answer.',
                    isStreaming: false,
                });
                setStatus(error.message || 'Something went wrong.');
            } finally {
                isExecuting = false;
                runBtn.disabled = false;
                runBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>';
            }
        };

        applyTheme(loadTheme());
        initializeVoiceSettings();
        initSidebar();

        if (window.speechSynthesis) {
            window.speechSynthesis.onvoiceschanged = () => {
                refreshVoiceVoiceList();
            };
        }

        document.querySelectorAll('[data-query]').forEach((button) => {
            button.addEventListener('click', () => {
                queryEl.value = button.dataset.query;
                queryEl.focus();
                autoResizeQuery();
            });
        });

        newSessionBtn.addEventListener('click', startNewSession);
        
        drawerToggleEl.addEventListener('click', toggleSidebar);
        drawerCloseEl.addEventListener('click', closeSidebar);
        drawerOverlayEl.addEventListener('click', closeSidebar);

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
                closeSidebar();
                closeModeOnboardingModal();
                if (settingsModalEl) {
                    settingsModalEl.classList.remove('open');
                    settingsModalEl.setAttribute('aria-hidden', 'true');
                }
            }
        });

        if (openSettingsBtnEl) {
            openSettingsBtnEl.addEventListener('click', () => {
                if (currentUser) {
                    settingsPreferredNameEl.value = currentUser.preferred_name || currentUser.name || '';
                }
                settingsModalEl.classList.add('open');
                settingsModalEl.setAttribute('aria-hidden', 'false');
            });
        }
        if (settingsModalCloseEl) {
            settingsModalCloseEl.addEventListener('click', () => {
                settingsModalEl.classList.remove('open');
                settingsModalEl.setAttribute('aria-hidden', 'true');
            });
        }
        if (settingsModalCancelEl) {
            settingsModalCancelEl.addEventListener('click', () => {
                settingsModalEl.classList.remove('open');
                settingsModalEl.setAttribute('aria-hidden', 'true');
            });
        }
        if (settingsModalEl) {
            settingsModalEl.addEventListener('click', (event) => {
                if (event.target === settingsModalEl) {
                    settingsModalEl.classList.remove('open');
                    settingsModalEl.setAttribute('aria-hidden', 'true');
                }
            });
        }
        if (settingsModalSaveEl) {
            settingsModalSaveEl.addEventListener('click', async () => {
                const newName = settingsPreferredNameEl.value.trim();
                if (!newName) {
                    setStatus('Preferred name cannot be empty.');
                    return;
                }
                if (currentUser) {
                    setStatus('Saving settings...');
                    try {
                        const dob = currentUser.dob || '2000-01-01';
                        const res = await fetch(`/auth/complete-setup?session_token=${encodeURIComponent(sessionToken)}`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ dob, preferred_name: newName })
                        });
                        if (!res.ok) {
                            const err = await res.json();
                            throw new Error(err.detail || 'Failed to update preferred name');
                        }
                        const updatedUser = await res.json();
                        currentUser = updatedUser;
                        userNameEl.textContent = updatedUser.preferred_name || updatedUser.name;
                        setStatus('Settings saved.');
                        settingsModalEl.classList.remove('open');
                        settingsModalEl.setAttribute('aria-hidden', 'true');
                    } catch (err) {
                        console.error('Error saving settings:', err);
                        setStatus('Error saving settings: ' + err.message);
                    }
                } else {
                    setStatus('No user logged in.');
                    settingsModalEl.classList.remove('open');
                    settingsModalEl.setAttribute('aria-hidden', 'true');
                }
            });
        }

        const showModeOnboardingModal = () => {
            if (modeOnboardingModalEl) {
                modeOnboardingModalEl.classList.add('open');
                modeOnboardingModalEl.setAttribute('aria-hidden', 'false');
            }
        };

        const closeModeOnboardingModal = () => {
            if (modeOnboardingModalEl) {
                modeOnboardingModalEl.classList.remove('open');
                modeOnboardingModalEl.setAttribute('aria-hidden', 'true');
            }
        };

        if (modeOnboardingModalCloseEl) {
            modeOnboardingModalCloseEl.addEventListener('click', closeModeOnboardingModal);
        }
        if (modeOnboardingModalBtnEl) {
            modeOnboardingModalBtnEl.addEventListener('click', closeModeOnboardingModal);
        }
        if (modeOnboardingModalEl) {
            modeOnboardingModalEl.addEventListener('click', (event) => {
                if (event.target === modeOnboardingModalEl) {
                    closeModeOnboardingModal();
                }
            });
        }

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
            autoResizeQuery();
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

        voiceModeBtnEl.addEventListener('click', () => {
            const isOpen = !voicePanelEl.classList.contains('open');
            setVoicePanelOpen(isOpen);
            setStatus(isOpen ? 'Voice mode opened.' : 'Voice mode hidden.');
        });

        queryEl.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        });

        // Bind Authentication Event Listeners
        btnMockLoginEl.addEventListener('click', performMockLogin);
        document.getElementById('onboarding_form').addEventListener('submit', (e) => {
            e.preventDefault();
            performOnboarding();
        });
        btnLogoutEl.addEventListener('click', performLogout);
        if (btnToggleMockEl) {
            btnToggleMockEl.addEventListener('click', () => {
                if (devLoginBoxEl.style.display === 'none') {
                    devLoginBoxEl.style.display = 'block';
                    btnToggleMockEl.textContent = 'Hide Mock Login';
                } else {
                    devLoginBoxEl.style.display = 'none';
                    btnToggleMockEl.textContent = 'Bypass with Mock Login';
                }
            });
        }

        // Bind Mochi Specific Input Reaction listeners
        if (mockEmailEl) {
            mockEmailEl.addEventListener('focus', () => {
                if (mochiState === 'default' || mochiState === 'sleepy') setMochiState('peek');
            });
            mockEmailEl.addEventListener('blur', () => {
                if (mochiState === 'peek') setMochiState('default');
            });
        }

        if (onboardingDobEl) {
            onboardingDobEl.addEventListener('focus', () => {
                if (mochiState === 'default' || mochiState === 'sleepy') setMochiState('closed');
            });
            onboardingDobEl.addEventListener('blur', () => {
                if (mochiState === 'closed') setMochiState('default');
            });
        }

        // Bind Theme Toggle for Login screen
        function syncAuthThemeUI(theme) {
            const isDark = theme === 'dark';
            if (authSunIconEl) authSunIconEl.classList.toggle('active', !isDark);
            if (authMoonIconEl) authMoonIconEl.classList.toggle('active', isDark);
        }
        if (authThemeToggleEl) {
            authThemeToggleEl.addEventListener('click', () => {
                const currentTheme = document.body.dataset.theme === 'dark' ? 'light' : 'dark';
                applyTheme(currentTheme);
                localStorage.setItem(THEME_KEY, currentTheme);
                syncAuthThemeUI(currentTheme);
            });
        }

        // Run session validation on page load





        validateSession();
    </script>
</body>
</html>
"""


class GoogleLoginRequest(BaseModel):
    credential: Optional[str] = None
    mock_email: Optional[str] = None


class CompleteSetupRequest(BaseModel):
    dob: str
    preferred_name: str


class ResearchRequest(BaseModel):
    query: str
    use_memory: bool = True
    memory_mode: str = "balanced"
    conversation_context: str = ""
    session_token: Optional[str] = None
    research_mode: str = "flash"


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
    import re
    tokens = re.split(r"(\s+)", text)
    chunks = []
    current = []
    word_count = 0
    for token in tokens:
        if not token:
            continue
        current.append(token)
        if token.strip():
            word_count += 1
            if word_count >= chunk_size:
                chunks.append("".join(current))
                current = []
                word_count = 0
    if current:
        chunks.append("".join(current))
    return chunks


@app.get("/", response_class=HTMLResponse)
async def homepage():
    import os
    client_id = os.getenv("GOOGLE_CLIENT_ID", "")
    html = APP_HTML.replace("GOOGLE_CLIENT_ID_PLACEHOLDER", client_id)
    return HTMLResponse(html)


class TitleRequest(BaseModel):
    query: str

@app.post("/research/title")
async def generate_title(request: TitleRequest):
    query = request.query.strip()
    if not query:
        return {"title": "New Session"}
    try:
        from llm import generate_text
        title = generate_text(
            system_prompt="You are a helper that generates a very brief, concise, and professional title/summary (2 to 4 words) for a chat session based on the user's first query. Do NOT use quotes around the title. Do NOT prefix it. Example: User query: 'How does photosynthesis convert sunlight into energy?' -> Title: 'Understanding Photosynthesis'",
            user_prompt=f"Generate a 2-4 word title for this query: {query}",
            max_tokens=20,
            temperature=0.3
        )
        title = title.strip().replace('"', '').replace("'", "")
        return {"title": title}
    except Exception as e:
        logger.error(f"[title] error generating title: {e}")
        collapsed = " ".join(query.split())
        fallback_title = collapsed[:35] + "..." if len(collapsed) > 35 else collapsed
        return {"title": fallback_title}


@app.post("/research", response_model=ResearchResponse)
async def run_research(request: ResearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    user = get_user_from_session(request.session_token)
    if not user or not user.get("dob") or not user.get("preferred_name"):
        raise HTTPException(status_code=401, detail="Unauthorized: Please complete login and onboarding.")

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
        "preferred_name": user["preferred_name"],
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
        "research_mode": request.research_mode if request.research_mode in {"flash", "research", "thesis"} else "flash",
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
async def stream_research(query: str, context: str = "", use_memory: bool = True, memory_mode: str = "balanced", session_token: str = "", research_mode: str = "flash"):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    user = get_user_from_session(session_token)
    if not user or not user.get("dob") or not user.get("preferred_name"):
        raise HTTPException(status_code=401, detail="Unauthorized: Please complete login and onboarding.")

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
                    yield f"data: {json.dumps({'type': 'delta', 'node': 'assistant', 'data': {'delta': chunk}})}\n\n"
                    await asyncio.sleep(0)
                yield f"data: {json.dumps({'type': 'final', 'node': 'assistant', 'data': {'request_id': request_id, 'query': query, 'final_answer': cached_answer, 'source_urls': cached_source_urls, 'claims': cached_claims, 'iterations': 0, 'total_claims': len(cached_claims), 'evidence_coverage': float(cached_meta.get('evidence_coverage', 0.0)), 'avg_confidence': float(cached_meta.get('avg_confidence', 0.0)), 'from_memory': True}})}\n\n"
                return

        initial_state = {
            "query": query,
            "conversation_context": context,
            "preferred_name": user["preferred_name"],
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
            "research_mode": research_mode if research_mode in {"flash", "research", "thesis"} else "flash",
        }

        try:
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
                            yield f"data: {json.dumps({'type': 'delta', 'node': 'assistant', 'data': {'delta': chunk}})}\n\n"
                            await asyncio.sleep(0)
                        total_claims = len(latest_summary.claims) if latest_summary else 0
                        claims = [c.dict() for c in (latest_summary.claims if latest_summary else [])]
                        yield f"data: {json.dumps({'type': 'final', 'node': 'assistant', 'data': {'request_id': request_id, 'query': query, 'final_answer': final_answer, 'source_urls': source_urls, 'claims': claims, 'iterations': latest_iterations, 'total_claims': total_claims, 'evidence_coverage': float(node_output.get('evidence_coverage', 0.0)), 'avg_confidence': float(node_output.get('avg_confidence', 0.0)), 'from_memory': False}})}\n\n"
                        return

            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"[research/stream] Error during graph streaming: {e}", exc_info=True)
            err_msg = f"Error processing research query: {str(e)}"
            yield f"data: {json.dumps({'type': 'delta', 'node': 'assistant', 'data': {'delta': err_msg}})}\n\n"
            yield f"data: {json.dumps({'type': 'final', 'node': 'assistant', 'data': {'request_id': request_id, 'query': query, 'final_answer': err_msg, 'source_urls': [], 'claims': [], 'iterations': latest_iterations, 'total_claims': 0, 'evidence_coverage': 0.0, 'avg_confidence': 0.0, 'from_memory': False}})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/auth/google")
async def auth_google(req: GoogleLoginRequest):
    google_client_id = os.getenv("GOOGLE_CLIENT_ID", "")
    user_info = None

    if google_client_id and req.credential:
        try:
            resp = requests.get(f"https://oauth2.googleapis.com/tokeninfo?id_token={req.credential}")
            if resp.status_code != 200:
                raise HTTPException(status_code=400, detail="Invalid Google token.")
            data = resp.json()
            if data.get("aud") != google_client_id:
                raise HTTPException(status_code=400, detail="Token audience mismatch.")
            
            user_info = {
                "google_id": data.get("sub"),
                "email": data.get("email"),
                "name": data.get("name"),
                "picture": data.get("picture", f"https://api.dicebear.com/7.x/bottts/svg?seed={data.get('email')}")
            }
        except Exception as e:
            logger.error(f"[auth] google token verification failed: {e}")
            raise HTTPException(status_code=400, detail=f"Google token verification failed: {str(e)}")
            
    elif req.mock_email:
        email = req.mock_email.strip().lower()
        if not email or "@" not in email:
            raise HTTPException(status_code=400, detail="Invalid mock email.")
        name = email.split("@")[0].capitalize()
        user_info = {
            "google_id": f"mock_{email}",
            "email": email,
            "name": name,
            "picture": f"https://api.dicebear.com/7.x/bottts/svg?seed={email}"
        }
    else:
        raise HTTPException(
            status_code=400, 
            detail="Google Client ID is configured but no credential token was provided. "
                   "Or, if you are developing locally, please provide a mock email."
        )

    google_id = user_info["google_id"]
    email = user_info["email"]
    name = user_info["name"]
    picture = user_info["picture"]

    row = fetch_one("SELECT dob, preferred_name FROM users WHERE google_id = ?", (google_id,))
    
    first_time = True
    dob = None
    preferred_name = None
    
    if not row:
        execute_query(
            "INSERT INTO users (google_id, email, name, picture, dob, preferred_name) VALUES (?, ?, ?, ?, NULL, NULL)",
            (google_id, email, name, picture)
        )
    else:
        dob, preferred_name = row
        if dob and preferred_name:
            first_time = False

    session_token = str(uuid.uuid4())
    execute_query(
        "INSERT INTO sessions (session_token, google_id) VALUES (?, ?)",
        (session_token, google_id)
    )

    return {
        "session_token": session_token,
        "first_time": first_time,
        "email": email,
        "name": name,
        "picture": picture,
        "preferred_name": preferred_name
    }


@app.post("/auth/complete-setup")
async def auth_complete_setup(req: CompleteSetupRequest, session_token: str = ""):
    user = get_user_from_session(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    dob = req.dob.strip()
    preferred_name = req.preferred_name.strip()
    
    if not dob or not preferred_name:
        raise HTTPException(status_code=400, detail="DOB and preferred name are required.")
        
    age = calculate_age(dob)
    if age < 13:
        raise HTTPException(status_code=400, detail="You must be at least 13 years old to sign up.")
        
    execute_query(
        "UPDATE users SET dob = ?, preferred_name = ? WHERE google_id = ?",
        (dob, preferred_name, user["google_id"])
    )
    
    user["dob"] = dob
    user["preferred_name"] = preferred_name
    return user


@app.get("/auth/user-profile")
async def auth_user_profile(session_token: str = ""):
    user = get_user_from_session(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid session token.")
    return user


@app.get("/memory/search")
async def search_memory(query: str, n: int = 3):
    results = memory.retrieve_similar(query, n_results=n)
    return {"query": query, "results": results}


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/debug/models")
async def debug_models():
    info = {
        "llm_provider_env": os.getenv("LLM_PROVIDER"),
        "groq_key_set": bool(os.getenv("GROQ_API_KEY")),
        "groq_key_prefix": (os.getenv("GROQ_API_KEY", "")[:8] + "...") if os.getenv("GROQ_API_KEY") else None,
        "anthropic_key_set": bool(os.getenv("ANTHROPIC_API_KEY")),
        "groq_model_env": os.getenv("GROQ_MODEL"),
        "tavily_key_set": bool(os.getenv("TAVILY_API_KEY")),
    }
    if os.getenv("GROQ_API_KEY"):
        try:
            from groq import Groq
            client = Groq(api_key=os.environ["GROQ_API_KEY"])
            models = client.models.list()
            info["available_groq_models"] = [m.id for m in models.data]
        except Exception as e:
            info["groq_models_error"] = str(e)
    return info


@app.get("/static/{filename}")
async def serve_static(filename: str):
    import os
    from fastapi import Response
    filepath = os.path.join("static", filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    ext = os.path.splitext(filename)[1].lower()
    media_type = "image/png"
    if ext == ".svg":
        media_type = "image/svg+xml"
    elif ext in [".jpg", ".jpeg"]:
        media_type = "image/jpeg"
    with open(filepath, "rb") as f:
        return Response(content=f.read(), media_type=media_type)
