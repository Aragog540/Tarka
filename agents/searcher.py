import json
import os
import re
from hashlib import md5
from typing import List
from urllib.parse import urlsplit, urlunsplit

import requests

from llm import generate_text
from memory.store import memory
from observability.logger import log_agent_call, logger
from state.schema import ResearchState, SearchResult
_TAVILY_KEY = os.environ.get("TAVILY_API_KEY", "")
_TAVILY_URL = "https://api.tavily.com/search"

_SUB_QUERY_PROMPT = """You are a search query specialist.

Given a research query and optional context from previous research, generate 3 focused sub-queries to retrieve comprehensive information.

Return ONLY valid JSON:
{
  "sub_queries": ["query1", "query2", "query3"]
}"""

_MAX_RESULTS_PER_SUB_QUERY = 4
_MAX_RERANKED_RESULTS = 8
_MAX_MEMORY_HINTS = 2
_CHUNK_SIZE = 520
_CHUNK_OVERLAP = 120


def _conversation_context(state: ResearchState) -> str:
    context = state.get("conversation_context", "").strip()
    return f"\n\nConversation context:\n{context}" if context else ""


def _generate_sub_queries(
    query: str,
    critique_gaps: List[dict] = None,
    conversation_context: str = "",
    memory_hints: List[dict] = None,
) -> List[str]:
    context = ""
    if critique_gaps:
        gap_descriptions = [g["suggested_query"] for g in critique_gaps]
        context = f"\n\nTarget these specific gaps:\n" + "\n".join(f"- {q}" for q in gap_descriptions)

    memory_context = ""
    if memory_hints:
        hints = []
        for item in memory_hints[:_MAX_MEMORY_HINTS]:
            hints.append(f"- Prior query: {item.get('query', '')}")
            for claim in (item.get("claims") or [])[:2]:
                claim_text = claim.get("claim", "") if isinstance(claim, dict) else ""
                if claim_text:
                    hints.append(f"  Claim: {claim_text}")
        if hints:
            memory_context = "\n\nPrior memory hints (for query rewriting only, not as evidence):\n" + "\n".join(hints)

    raw = json.loads(generate_text(_SUB_QUERY_PROMPT, f"Query: {query}{conversation_context}{context}{memory_context}", max_tokens=500, json_mode=True))
    return raw.get("sub_queries", [])[:3]


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    parts = urlsplit(url)
    normalized_path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), normalized_path, "", ""))


def _chunk_content(content: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> List[str]:
    text = (content or "").strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = min(len(text), start + chunk_size)
        candidate = text[start:end]
        if end < len(text):
            split_at = candidate.rfind(". ")
            if split_at > int(chunk_size * 0.55):
                end = start + split_at + 1
                candidate = text[start:end]
        candidate = candidate.strip()
        if candidate:
            chunks.append(candidate)
        if end >= len(text):
            break
        start = start + step

    return chunks


def _token_overlap_score(query: str, content: str) -> float:
    query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    if not query_tokens:
        return 0.0
    content_tokens = set(re.findall(r"[a-z0-9]+", content.lower()))
    if not content_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(content_tokens))
    return min(1.0, overlap / max(1, len(query_tokens)))


def _expand_to_chunks(results: List[SearchResult]) -> List[SearchResult]:
    expanded: List[SearchResult] = []
    for result in results:
        chunks = _chunk_content(result.content)
        if not chunks:
            continue
        base_url = _normalize_url(result.url) or result.url
        for i, chunk in enumerate(chunks, 1):
            chunk_hash = md5(f"{base_url}|{chunk}".encode("utf-8")).hexdigest()
            expanded.append(SearchResult(
                source=result.source,
                content=chunk,
                url=result.url,
                relevance_score=result.relevance_score,
                chunk_id=f"{base_url or result.source}#chunk-{i}",
                content_hash=chunk_hash,
                is_memory=result.is_memory,
            ))
    return expanded


def _dedupe_results(results: List[SearchResult]) -> List[SearchResult]:
    deduped: List[SearchResult] = []
    seen = set()
    for result in results:
        key = (
            _normalize_url(result.url),
            result.content_hash or md5(result.content.encode("utf-8")).hexdigest(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


def _rerank_results(query: str, results: List[SearchResult], top_k: int = _MAX_RERANKED_RESULTS) -> List[SearchResult]:
    if not results:
        return []

    seen_source = set()
    scored = []
    for result in results:
        overlap = _token_overlap_score(query, result.content)
        memory_penalty = 0.12 if result.is_memory else 0.0
        source_bonus = 0.04 if result.source not in seen_source else 0.0
        seen_source.add(result.source)
        score = (0.55 * result.relevance_score) + (0.45 * overlap) + source_bonus - memory_penalty
        scored.append((score, result))

    scored.sort(key=lambda x: x[0], reverse=True)
    reranked = [item for _, item in scored[:top_k]]
    return reranked


def _source_from_url(url: str) -> str:
    if not url:
        return "unknown"
    try:
        return url.split("/")[2]
    except IndexError:
        return url


def _tavily_search(sub_query: str) -> List[SearchResult]:
    if not _TAVILY_KEY:
        logger.warning("TAVILY_API_KEY not set — returning empty results for query: %s", sub_query)
        return []

    payload = {
        "api_key": _TAVILY_KEY,
        "query": sub_query,
        "search_depth": "advanced",
        "max_results": _MAX_RESULTS_PER_SUB_QUERY,
        "include_answer": False,
    }
    response = requests.post(_TAVILY_URL, json=payload, timeout=15)
    response.raise_for_status()
    data = response.json()

    return [
        SearchResult(
            source=_source_from_url(r.get("url", "")),
            content=r.get("content", ""),
            url=r.get("url", ""),
            relevance_score=r.get("score", 1.0),
            is_memory=False,
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
                is_memory=True,
            ))
    return results


def _memory_hints(query: str) -> List[dict]:
    similar = memory.retrieve_similar(query, n_results=3)
    hints = []
    for item in similar:
        if item.get("distance", 1.0) <= 0.55:
            hints.append(item)
    return hints


@log_agent_call("searcher")
def searcher_node(state: ResearchState) -> dict:
    query = state["query"]
    critique = state.get("critique")
    memory_mode = state.get("memory_mode", "balanced")
    conversation_context = _conversation_context(state)

    critique_gaps = critique.gaps if critique else []
    hints = _memory_hints(query) if memory_mode in ("balanced", "prefer_memory") else []
    sub_queries = _generate_sub_queries(
        query,
        [g.dict() for g in critique_gaps] if critique_gaps else None,
        conversation_context,
        hints,
    )

    logger.info(f"[searcher] sub_queries={sub_queries}")

    live_results: List[SearchResult] = []
    for sub_q in sub_queries:
        results = _tavily_search(sub_q)
        live_results.extend(results)

    memory_results = _memory_search(query) if memory_mode == "prefer_memory" else []

    expanded_live = _expand_to_chunks(live_results)
    expanded_memory = _expand_to_chunks(memory_results)
    combined_results = _dedupe_results(expanded_live + expanded_memory)
    reranked = _rerank_results(query, combined_results, _MAX_RERANKED_RESULTS)

    logger.info(
        "[searcher] live=%s memory=%s deduped=%s reranked=%s mode=%s",
        len(expanded_live),
        len(expanded_memory),
        len(combined_results),
        len(reranked),
        memory_mode,
    )

    return {
        "search_results": reranked,
        "agent_logs": [{
            "agent": "searcher",
            "sub_queries": sub_queries,
            "results_count": len(reranked),
            "memory_mode": memory_mode,
            "memory_hints_used": len(hints),
        }],
    }
