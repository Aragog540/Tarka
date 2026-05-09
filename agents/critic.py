import json

from llm import generate_text
from observability.logger import log_agent_call, logger
from state.schema import Critique, CritiqueGap, ResearchState
MAX_ITERATIONS = 3

_SYSTEM_PROMPT = """You are a research critic. Evaluate the quality and completeness of research claims.

Return ONLY valid JSON:
{
  "gaps": [
    {
      "description": "what is missing or unclear",
      "suggested_query": "specific search query to fill this gap"
    }
  ],
  "verified_claims": ["claim1 that is well-supported", "claim2..."],
  "should_continue": true | false,
  "reasoning": "brief explanation of your decision"
}

Continue (should_continue=true) only if there are 2+ significant gaps that would materially change the answer.
Do NOT continue if gaps are minor or stylistic."""


def _conversation_context(state: ResearchState) -> str:
    context = state.get("conversation_context", "").strip()
    return f"\n\nConversation context:\n{context}" if context else ""


@log_agent_call("critic")
def critic_node(state: ResearchState) -> dict:
    query = state["query"]
    summary = state.get("summary")
    iterations = state.get("iterations", 0)
    conversation_context = _conversation_context(state)

    if iterations >= MAX_ITERATIONS:
        logger.info(f"[critic] max iterations ({MAX_ITERATIONS}) reached — forcing stop")
        return {
            "critique": Critique(
                gaps=[],
                verified_claims=[c.claim for c in (summary.claims if summary else [])],
                should_continue=False,
                reasoning=f"Max iterations ({MAX_ITERATIONS}) reached.",
            ),
            "iterations": iterations,
            "agent_logs": [{
                "agent": "critic",
                "decision": "stop",
                "reason": "max_iterations_reached",
            }],
        }

    if not summary or not summary.claims:
        logger.warning("[critic] no summary to critique")
        return {
            "critique": Critique(
                gaps=[],
                verified_claims=[],
                should_continue=False,
                reasoning="No summary available to critique.",
            ),
            "iterations": iterations + 1,
            "agent_logs": [{"agent": "critic", "decision": "stop", "reason": "no_summary"}],
        }

    claims_text = "\n".join(
        f"- [{c.confidence}] {c.claim} (source: {c.source})"
        for c in summary.claims
    )

    raw = json.loads(generate_text(
        _SYSTEM_PROMPT,
        (
            f"Research query: {query}{conversation_context}\n\n"
            f"Current claims ({len(summary.claims)} total):\n{claims_text}\n\n"
            f"Summary: {summary.raw_summary}\n\n"
            f"This is iteration {iterations + 1} of {MAX_ITERATIONS}."
        ),
        max_tokens=1000,
        json_mode=True,
    ))

    gaps = [CritiqueGap(**g) for g in raw.get("gaps", [])]
    critique = Critique(
        gaps=gaps,
        verified_claims=raw.get("verified_claims", []),
        should_continue=raw.get("should_continue", False),
        reasoning=raw.get("reasoning", ""),
    )

    logger.info(
        f"[critic] gaps={len(gaps)} | should_continue={critique.should_continue} | "
        f"iteration={iterations + 1}/{MAX_ITERATIONS}"
    )

    return {
        "critique": critique,
        "iterations": iterations + 1,
        "agent_logs": [{
            "agent": "critic",
            "gaps_found": len(gaps),
            "decision": "continue" if critique.should_continue else "stop",
            "reasoning": critique.reasoning,
        }],
    }


def should_continue(state: ResearchState) -> str:
    critique = state.get("critique")
    if critique and critique.should_continue:
        return "searcher"
    return "aggregator"
