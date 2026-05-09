import asyncio
import json
import time
import uuid
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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


class ResearchRequest(BaseModel):
    query: str
    use_memory: bool = True


class ResearchResponse(BaseModel):
    request_id: str
    query: str
    final_answer: str
    iterations: int
    total_claims: int
    elapsed_seconds: float
    from_memory: bool


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

    return ResearchResponse(
        request_id=request_id,
        query=request.query,
        final_answer=final_state.get("final_answer", ""),
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
