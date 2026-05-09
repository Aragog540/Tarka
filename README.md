# Multi-Agent Research Assistant

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

| Component       | Tool                    |
|-----------------|-------------------------|
| Agent framework | LangGraph               |
| LLM             | Claude Sonnet (Anthropic)|
| Web search      | Tavily API              |
| Vector DB       | ChromaDB (local/persistent)|
| API layer       | FastAPI + SSE streaming  |
| Containerization| Docker                  |

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
- `ANTHROPIC_API_KEY` — [Get one here](https://console.anthropic.com)
- `TAVILY_API_KEY` — [Get one here](https://tavily.com) (free tier available)

Optional for lower-cost web demos:
- `LLM_PROVIDER=groq`
- `GROQ_API_KEY`
- `GROQ_MODEL` — defaults to `llama-3.1-8b-instant`

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
  "final_answer": "...",
  "iterations": 2,
  "total_claims": 7,
  "elapsed_seconds": 18.4,
  "from_memory": false
}
```

## Key Design Decisions

**Capped iterations** — `MAX_ITERATIONS = 3` in `agents/critic.py` prevents infinite loops, a real production concern.

**Structured Pydantic outputs** — every agent returns validated models (`Summary`, `Critique`, `SearchResult`), not raw strings.

**Persistent memory** — ChromaDB stores past research. Similar queries return cached answers instantly via cosine similarity.

**Observability** — every agent call is logged to `logs/agent_runs.jsonl` with timing, inputs, and status.

**Deduplication** — the searcher deduplicates results by URL before passing to the summarizer.

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
