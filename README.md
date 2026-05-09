# Tarka

Motivation
Tarka was built to help researchers and engineers rapidly produce evidence-backed answers without spending hours locating and vetting sources. By orchestrating a set of specialized agents (planning, searching, summarizing, critiquing, and aggregating), Tarka automates the repetitive parts of literature review and technical research so humans can focus on interpretation and decision-making.

A production-grade agentic research system built with LangGraph, where specialized agents collaborate to answer complex queries with verifiable, cited responses.

## Architecture

```
User Query
    │
    ▼
Supervisor          ← routes query, generates research plan
    │
    ▼
Searcher            ← Tavily API + ChromaDB memory retrieval
    │
    ▼
Summarizer          ← extracts structured claims {claim, source, confidence}
    │
    ▼
Critic              ← identifies gaps, triggers loop or stops
    │         │
    │ (gaps)  │ (complete)
    ▼         ▼
Searcher   Aggregator  ← composes final answer with citations
               │
               ▼
          Final Answer + stored in ChromaDB
```

The **Critic → Searcher** loop is the core agentic behavior: the system iteratively fills knowledge gaps before composing a final answer. Capped at 3 iterations to prevent runaway loops.

## Stack

| Component       | Tool                           |
|-----------------|--------------------------------|
| Agent framework | LangGraph                      |
| LLM (default)   | Groq llama-3.1-8b-instant      |
| LLM (optional)  | Anthropic Claude Sonnet        |
| Web search      | Tavily API                     |
| Vector DB       | ChromaDB (local/persistent)    |
| API layer       | FastAPI + SSE streaming        |
| Web UI          | HTML5 + localStorage (sidebar, dark theme) |
| Containerization| Docker                         |

## Setup

### 1. Clone and install

```bash
git clone <repo>
cd multi-agent-research
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required:
- `GROQ_API_KEY` — [Get one here](https://console.groq.com) (free tier available; default LLM)
- `TAVILY_API_KEY` — [Get one here](https://tavily.com) (free tier available)

Optional for Anthropic Claude instead of Groq:
- `ANTHROPIC_API_KEY` — [Get one here](https://console.anthropic.com)
- `LLM_PROVIDER=anthropic`
- `ANTHROPIC_MODEL` — defaults to `claude-sonnet-4-20250514`

### 3. Run

**CLI:**
```bash
python run.py "Is GPT-4o better than Gemini 1.5 Pro for enterprise use?"
```

**API server:**
```bash
uvicorn api.main:app --reload --port 8000
```

**Web UI:**
Open `http://localhost:8000/` for the minimalist browser interface. It posts to the same `/research` endpoint used by the API.

**Docker:**
```bash
docker-compose up --build
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/research` | Run full research pipeline |
| `GET` | `/research/stream` | SSE stream of agent events |
| `GET` | `/memory/search?query=...` | Query past research memory |
| `GET` | `/health` | Health check |

### Example request

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the best vector databases in 2024?", "use_memory": true}'
```

### Example response

```json
{
  "request_id": "a3f9b1c2",
  "query": "What are the best vector databases in 2024?",
  "final_answer": "Vector databases are...",
  "source_urls": ["https://example.com/db1", "https://example.com/db2"],
  "iterations": 2,
  "total_claims": 7,
  "elapsed_seconds": 18.4,
  "from_memory": false
}
```

## Key Design Decisions

**Capped iterations** — `MAX_ITERATIONS = 3` in `agents/critic.py` prevents infinite loops, a real production concern.

**Structured Pydantic outputs** — every agent returns validated models (`Summary`, `Critique`, `SearchResult`), not raw strings.

**Persistent memory** — ChromaDB stores past research. Similar queries return cached answers instantly via cosine similarity. Metadata includes `source_urls` for the UI.

**Plain-text answers** — the aggregator emits plain-text prose only (no Markdown), with inline citations as "Source: domain.com".

**Deduplication** — the searcher deduplicates results by URL before passing to the summarizer. The aggregator avoids duplicate source URLs in the side panel.

**Prompt budgeting** — the summarizer caps search results to 5 items and 3000 chars total to stay under Groq's TPM limits (6000 TPM for llama-3.1-8b-instant).

**Web UI features** — chat history sidebar (localStorage), dark theme toggle, and persistent theme preference.

## Project Structure

```
multi-agent-research/
├── agents/
│   ├── supervisor.py       # Query analysis and routing plan
│   ├── searcher.py         # Tavily + memory retrieval
│   ├── summarizer.py       # Structured claim extraction
│   ├── critic.py           # Gap detection + loop control
│   └── aggregator.py       # Final answer composition
├── graph/
│   └── research_graph.py   # LangGraph StateGraph definition
├── state/
│   └── schema.py           # ResearchState + Pydantic models
├── memory/
│   └── store.py            # ChromaDB persistent memory
├── observability/
│   └── logger.py           # Agent call logging decorator
├── api/
│   └── main.py             # FastAPI app + SSE streaming
├── run.py                  # CLI entrypoint
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
