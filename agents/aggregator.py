from llm import generate_text
from memory.store import memory
from observability.logger import log_agent_call, logger
from state.schema import ResearchState

_SYSTEM_PROMPT = """You are a research aggregator. Produce a clear, well-structured final answer.

Guidelines:
- Lead with a direct answer to the query
- Organize by key dimensions (e.g., cost, performance, use cases)
- Use plain text only: no Markdown headings, bold, bullets, numbered lists, tables, or code fences
- Use short labeled paragraphs such as "Direct answer:" and "Cost:"
- Include citations inline as "Source: domain.com" in parentheses
- Flag any remaining uncertainty honestly
- Do not invent facts — only use what's in the claims"""

_FLASH_SYSTEM_PROMPT = """You are a helpful assistant. Write a short, clear, and concise answer in a natural flow based on the verified claims.
Guidelines:
- Write in a natural conversational flow (like a direct response).
- Keep it short and to the point (typically 1-2 paragraphs).
- Do NOT use any structured formatting, no bold text, no headings, no bullet points, no numbered lists, and no labeled paragraphs (like "Direct answer:").
- Do NOT include any prefixes like "Answer:" or "Response:".
- You can include inline citations like (Source: domain.com) if appropriate, but keep the overall text extremely clean and natural.
- Do not invent facts — only use what's in the verified claims."""

_THESIS_SYSTEM_PROMPT = """You are an expert scientific researcher and academic writer.
Your task is to write a highly technical, formal, and sophisticated academic abstract for a research paper based on the user's idea and refined by the verified claims.

Guidelines:
- Structure the abstract professionally: start with background/context, state the problem/idea, describe the methodology or conceptual framework, and highlight the significance or potential implications.
- Use very formal, advanced academic and technical terminology.
- Keep it concise, cohesive, and structured as a single continuous paragraph (standard abstract format).
- Do NOT use markdown formatting (such as bolding, bullet points, headings, or lists) in the output.
- Do not invent facts beyond the provided claims and logical extensions of the user's research idea."""


def _conversation_context(state: ResearchState) -> str:
    context = state.get("conversation_context", "").strip()
    return f"\n\nConversation context:\n{context}" if context else ""


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
    conversation_context = _conversation_context(state)

    claims_text = ""
    if summary and summary.claims:
        claims_text = "\n".join(
            f"- [{c.confidence}] {c.claim} (source: {c.source})"
            for c in summary.claims
        )

    verified_text = ""
    if critique and critique.verified_claims:
        verified_text = "\n".join(f"- {v}" for v in critique.verified_claims)

    source_urls = []
    seen_urls = set()
    for result in state.get("search_results", []):
        url = getattr(result, "url", "")
        if not url or not url.startswith("http") or url in seen_urls:
            continue
        seen_urls.add(url)
        source_urls.append(url)

    claims = summary.claims if summary else []
    total_claims = len(claims)
    claims_with_evidence = len([c for c in claims if c.evidence_snippet and c.source_url])
    evidence_coverage = round((claims_with_evidence / total_claims), 3) if total_claims else 0.0
    avg_confidence = round((sum(c.confidence_score for c in claims) / total_claims), 3) if total_claims else 0.0

    research_mode = state.get("research_mode", "flash")

    if research_mode == "flash":
        system_prompt = _FLASH_SYSTEM_PROMPT
        user_prompt = (
            f"Question/Query: {query}{conversation_context}\n\n"
            f"Verified claims:\n{claims_text or 'None available'}\n"
        )
    elif research_mode == "thesis":
        system_prompt = _THESIS_SYSTEM_PROMPT
        user_prompt = (
            f"Research Idea: {query}{conversation_context}\n\n"
            f"Refinement facts from web search:\n{claims_text or 'None available'}\n"
        )
    else:
        system_prompt = _SYSTEM_PROMPT
        user_prompt = (
            f"Research query: {query}{conversation_context}\n\n"
            f"Verified claims:\n{claims_text or 'None available'}\n\n"
            f"Critic-approved claims:\n{verified_text or 'None'}\n\n"
            f"Available source URLs are tracked separately for the UI and should not be repeated in the answer.\n\n"
            f"Research completed in {iterations} iteration(s)."
        )

    final_answer = generate_text(
        system_prompt,
        user_prompt,
        max_tokens=2000,
    )

    if research_mode == "research":
        final_answer = _normalize_plain_text(final_answer)
    else:
        final_answer = final_answer.strip()

    if summary and summary.claims:
        memory.store(
            query=query,
            final_answer=final_answer,
            claims=[c.dict() for c in summary.claims],
            source_urls=source_urls,
            metadata={
                "evidence_coverage": evidence_coverage,
                "avg_confidence": avg_confidence,
                "iterations": iterations,
                "research_mode": research_mode,
            },
        )
        logger.info("[aggregator] stored result in memory")

    logger.info(f"[aggregator] final answer generated ({len(final_answer)} chars)")

    return {
        "final_answer": final_answer,
        "source_urls": source_urls,
        "evidence_coverage": evidence_coverage,
        "avg_confidence": avg_confidence,
        "agent_logs": [{
            "agent": "aggregator",
            "answer_length": len(final_answer),
            "total_iterations": iterations,
            "evidence_coverage": evidence_coverage,
            "avg_confidence": avg_confidence,
            "research_mode": research_mode,
        }],
    }
