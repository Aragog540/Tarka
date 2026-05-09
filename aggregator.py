import os

from anthropic import Anthropic
from state.schema import ResearchState
from memory.store import memory
from observability.logger import log_agent_call, logger

_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

_SYSTEM_PROMPT = """You are a research aggregator. Produce a clear, well-structured final answer.

Guidelines:
- Lead with a direct answer to the query
- Organize by key dimensions (e.g., cost, performance, use cases)
- Include inline source citations as [Source: domain.com]
- Flag any remaining uncertainty honestly
- End with a "Sources" section listing all cited URLs
- Do not invent facts — only use what's in the claims"""


@log_agent_call("aggregator")
def aggregator_node(state: ResearchState) -> dict:
    query = state["query"]
    summary = state.get("summary")
    critique = state.get("critique")
    iterations = state.get("iterations", 0)

    claims_text = ""
    if summary and summary.claims:
        claims_text = "\n".join(
            f"- [{c.confidence}] {c.claim} (source: {c.source})"
            for c in summary.claims
        )

    verified_text = ""
    if critique and critique.verified_claims:
        verified_text = "\n".join(f"- {v}" for v in critique.verified_claims)

    urls = {r.source: r.url for r in state.get("search_results", []) if r.url}

    message = _client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Research query: {query}\n\n"
                f"Verified claims:\n{claims_text or 'None available'}\n\n"
                f"Critic-approved claims:\n{verified_text or 'None'}\n\n"
                f"Available source URLs:\n" +
                "\n".join(f"- {domain}: {url}" for domain, url in urls.items()) +
                f"\n\nResearch completed in {iterations} iteration(s)."
            ),
        }],
    )

    final_answer = message.content[0].text.strip()

    if summary and summary.claims:
        memory.store(
            query=query,
            final_answer=final_answer,
            claims=[c.dict() for c in summary.claims],
        )
        logger.info("[aggregator] stored result in memory")

    logger.info(f"[aggregator] final answer generated ({len(final_answer)} chars)")

    return {
        "final_answer": final_answer,
        "agent_logs": [{
            "agent": "aggregator",
            "answer_length": len(final_answer),
            "total_iterations": iterations,
        }],
    }
