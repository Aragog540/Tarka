import json

from llm import generate_text
from observability.logger import log_agent_call, logger
from state.schema import Claim, ResearchState, SearchResult, Summary

_SYSTEM_PROMPT = """You are a research summarizer. Extract key factual claims from search results.

Return ONLY valid JSON:
{
  "claims": [
    {
      "claim": "specific factual statement",
      "source": "domain.com",
            "source_url": "https://full-source-url",
      "confidence": "high" | "medium" | "low"
            "confidence_score": 0.0-1.0,
            "evidence_snippet": "short quote-like snippet from the provided content",
            "evidence_chunk_id": "chunk id from input"
    }
  ],
  "raw_summary": "2-3 sentence synthesis of all results"
}

Rules:
- Each claim must be independently verifiable
- Confidence is high if directly stated, medium if inferred, low if speculative
- Prefer specific facts over generalizations
- Evidence snippet must be copied from the source chunk text in the prompt
- Maximum 8 claims"""

_MAX_RESULTS_FOR_PROMPT = 5
_MAX_CHARS_PER_RESULT = 300
_MAX_TOTAL_RESULT_CHARS = 3000


def _conversation_context(state: ResearchState) -> str:
    context = state.get("conversation_context", "").strip()
    return f"\n\nConversation context:\n{context}" if context else ""


def _format_results(results: list[SearchResult]) -> str:
    formatted = []
    total_chars = 0
    for i, r in enumerate(results[:_MAX_RESULTS_FOR_PROMPT], 1):
        remaining_budget = _MAX_TOTAL_RESULT_CHARS - total_chars
        if remaining_budget <= 0:
            break

        content = r.content[: min(_MAX_CHARS_PER_RESULT, remaining_budget)]
        total_chars += len(content)
        chunk_id = r.chunk_id or f"chunk-{i}"
        formatted.append(f"[{i}] Source: {r.source}\nURL: {r.url}\nChunk ID: {chunk_id}\nContent: {content}")
    return "\n\n".join(formatted)


def _confidence_to_score(level: str) -> float:
    return {
        "high": 0.9,
        "medium": 0.6,
        "low": 0.35,
    }.get((level or "").lower(), 0.5)


@log_agent_call("summarizer")
def summarizer_node(state: ResearchState) -> dict:
    query = state["query"]
    results = state["search_results"]
    conversation_context = _conversation_context(state)

    if not results:
        logger.warning("[summarizer] no search results to summarize")
        return {
            "summary": Summary(claims=[], raw_summary="No search results available."),
            "agent_logs": [{"agent": "summarizer", "claims_count": 0}],
        }

    formatted = _format_results(results)

    raw = json.loads(generate_text(_SYSTEM_PROMPT, f"Research query: {query}{conversation_context}\n\nSearch results:\n{formatted}", max_tokens=1500, json_mode=True))

    claims = []
    for item in raw.get("claims", []):
        confidence = item.get("confidence", "medium")
        claims.append(Claim(
            claim=item.get("claim", "").strip(),
            source=item.get("source", "unknown").strip(),
            source_url=item.get("source_url", "").strip(),
            confidence=confidence,
            confidence_score=float(item.get("confidence_score", _confidence_to_score(confidence))),
            evidence_snippet=item.get("evidence_snippet", "").strip()[:280],
            evidence_chunk_id=item.get("evidence_chunk_id", "").strip(),
        ))
    summary = Summary(claims=claims, raw_summary=raw["raw_summary"])

    logger.info(f"[summarizer] extracted {len(claims)} claims")

    return {
        "summary": summary,
        "agent_logs": [{
            "agent": "summarizer",
            "claims_count": len(claims),
            "claim_list": [c.claim for c in claims],
        }],
    }
