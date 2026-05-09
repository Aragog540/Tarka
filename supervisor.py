import json
import os

from anthropic import Anthropic
from state.schema import ResearchState
from observability.logger import log_agent_call, logger

_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

_SYSTEM_PROMPT = """You are a research supervisor. Given a user query, analyze it and produce a routing plan.

Return ONLY valid JSON with this exact shape:
{
  "complexity": "low" | "medium" | "high",
  "primary_angles": ["angle1", "angle2"],
  "suggested_sub_queries": ["query1", "query2", "query3"],
  "requires_deep_critique": true | false,
  "reasoning": "brief explanation"
}"""


@log_agent_call("supervisor")
def supervisor_node(state: ResearchState) -> dict:
    query = state["query"]

    message = _client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Query: {query}"}],
    )

    raw = message.content[0].text.strip()
    plan = json.loads(raw)

    logger.info(f"[supervisor] complexity={plan['complexity']} | angles={plan['primary_angles']}")

    return {
        "agent_logs": [{
            "agent": "supervisor",
            "plan": plan,
        }],
        "iterations": 0,
    }
