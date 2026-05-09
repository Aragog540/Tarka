import json
import os

from anthropic import Anthropic
from state.schema import Claim, ResearchState, Summary, SearchResult
from observability.logger import log_agent_call, logger

_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

_SYSTEM_PROMPT = """You are a research summarizer. Extract key factual claims from search results.

Return ONLY valid JSON:
{
  "claims": [
    {
      "claim": "specific factual statement",
      "source": "domain.com",
      "confidence": "high" | "medium" | "low"
    }
  ],
  "raw_summary": "2-3 sentence synthesis of all results"
}

Rules:
- Each claim must be independently verifiable
- Confidence is high if directly stated, medium if inferred, low if speculative
- Prefer specific facts over generalizations
- Maximum 8 claims"""


def _format_results(results: list[SearchResult]) -> str:
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(f"[{i}] Source: {r.source}\nURL: {r.url}\nContent: {r.content[:800]}")
    return "\n\n".join(formatted)


@log_agent_call("summarizer")
def summarizer_node(state: ResearchState) -> dict:
    query = state["query"]
    results = state["search_results"]

    if not results:
        logger.warning("[summarizer] no search results to summarize")
        return {
            "summary": Summary(claims=[], raw_summary="No search results available."),
            "agent_logs": [{"agent": "summarizer", "claims_count": 0}],
        }

    formatted = _format_results(results)

    message = _client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Research query: {query}\n\nSearch results:\n{formatted}",
        }],
    )

    raw = json.loads(message.content[0].text.strip())

    claims = [Claim(**c) for c in raw["claims"]]
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
