import json
import os
from typing import List

import requests
from anthropic import Anthropic
from state.schema import ResearchState, SearchResult
from observability.logger import log_agent_call, logger
from memory.store import memory

_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
_TAVILY_KEY = os.environ.get("TAVILY_API_KEY", "")
_TAVILY_URL = "https://api.tavily.com/search"

_SUB_QUERY_PROMPT = """You are a search query specialist.

Given a research query and optional context from previous research, generate 3 focused sub-queries to retrieve comprehensive information.

Return ONLY valid JSON:
{
  "sub_queries": ["query1", "query2", "query3"]
}"""


def _generate_sub_queries(query: str, critique_gaps: List[dict] = None) -> List[str]:
    context = ""
    if critique_gaps:
        gap_descriptions = [g["suggested_query"] for g in critique_gaps]
        context = f"\n\nTarget these specific gaps:\n" + "\n".join(f"- {q}" for q in gap_descriptions)

    message = _client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system=_SUB_QUERY_PROMPT,
        messages=[{"role": "user", "content": f"Query: {query}{context}"}],
    )
    raw = json.loads(message.content[0].text.strip())
    return raw["sub_queries"]


def _tavily_search(sub_query: str) -> List[SearchResult]:
    if not _TAVILY_KEY:
        logger.warning("TAVILY_API_KEY not set — returning empty results for query: %s", sub_query)
        return []

    payload = {
        "api_key": _TAVILY_KEY,
        "query": sub_query,
        "search_depth": "advanced",
        "max_results": 3,
        "include_answer": False,
    }
    response = requests.post(_TAVILY_URL, json=payload, timeout=15)
    response.raise_for_status()
    data = response.json()

    return [
        SearchResult(
            source=r.get("url", "").split("/")[2] if r.get("url") else "unknown",
            content=r.get("content", ""),
            url=r.get("url", ""),
            relevance_score=r.get("score", 1.0),
        )
        for r in data.get("results", [])
    ]


def _memory_search(query: str) -> List[SearchResult]:
    similar = memory.retrieve_similar(query, n_results=2)
    results = []
    for item in similar:
        if item["distance"] < 0.4:
            results.append(SearchResult(
                source="research_memory",
                content=f"[Previous research] {item['answer'][:500]}",
                url=f"memory://{item['query'][:50]}",
                relevance_score=round(1 - item["distance"], 3),
            ))
    return results


@log_agent_call("searcher")
def searcher_node(state: ResearchState) -> dict:
    query = state["query"]
    critique = state.get("critique")

    critique_gaps = critique.gaps if critique else []
    sub_queries = _generate_sub_queries(query, [g.dict() for g in critique_gaps] if critique_gaps else None)

    logger.info(f"[searcher] sub_queries={sub_queries}")

    all_results: List[SearchResult] = []
    for sub_q in sub_queries:
        results = _tavily_search(sub_q)
        all_results.extend(results)

    memory_results = _memory_search(query)
    all_results.extend(memory_results)

    unique_urls = set()
    deduped = []
    for r in all_results:
        if r.url not in unique_urls:
            unique_urls.add(r.url)
            deduped.append(r)

    logger.info(f"[searcher] retrieved {len(deduped)} unique results")

    return {
        "search_results": deduped,
        "agent_logs": [{
            "agent": "searcher",
            "sub_queries": sub_queries,
            "results_count": len(deduped),
        }],
    }
