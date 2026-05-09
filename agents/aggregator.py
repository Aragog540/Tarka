from llm import generate_text
from memory.store import memory
from observability.logger import log_agent_call, logger
from state.schema import ResearchState

_SYSTEM_PROMPT = """You are a research aggregator. Produce a clear, well-structured final answer.

Guidelines:
- Lead with a direct answer to the query
- Organize by key dimensions (e.g., cost, performance, use cases)
- Use plain text only: no Markdown headings, bold, bullets, numbered lists, tables, or code fences
- Use short labeled paragraphs such as "Direct answer:", "Cost:", and "Sources:"
- Include citations inline as "Source: domain.com" in parentheses
- Flag any remaining uncertainty honestly
- End with a "Sources:" section listing all cited URLs on separate lines
- Do not invent facts — only use what's in the claims"""


def _normalize_plain_text(text: str) -> str:
    lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue

        while line.startswith("#"):
            line = line.lstrip("#").strip()

        if line.startswith(("- ", "* ")):
            line = line[2:].strip()
        elif line[:2].isdigit() and line[2:4] == ". ":
            line = line[4:].strip()

        line = line.replace("**", "").replace("__", "").replace("`", "")
        lines.append(line)

    normalized = "\n".join(lines).strip()
    return normalized


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

    final_answer = generate_text(
        _SYSTEM_PROMPT,
        (
            f"Research query: {query}\n\n"
            f"Verified claims:\n{claims_text or 'None available'}\n\n"
            f"Critic-approved claims:\n{verified_text or 'None'}\n\n"
            f"Available source URLs:\n"
            + "\n".join(f"- {domain}: {url}" for domain, url in urls.items())
            + f"\n\nResearch completed in {iterations} iteration(s)."
        ),
        max_tokens=2000,
    )
    final_answer = _normalize_plain_text(final_answer)

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
