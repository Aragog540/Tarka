import json

from llm import generate_text, safe_json_loads
from observability.logger import log_agent_call, logger
from state.schema import ResearchState

_SYSTEM_PROMPT = """You are a research supervisor. Given a user query, analyze it and produce a routing plan.

Return ONLY valid JSON with this exact shape:
{
  "complexity": "low" | "medium" | "high",
  "primary_angles": ["angle1", "angle2"],
  "suggested_sub_queries": ["query1", "query2", "query3"],
  "requires_deep_critique": true | false
}"""


def _conversation_context(state: ResearchState) -> str:
    context = state.get("conversation_context", "").strip()
    return f"\n\nConversation context:\n{context}" if context else ""


@log_agent_call("supervisor")
def supervisor_node(state: ResearchState) -> dict:
    query = state["query"]
    context = _conversation_context(state)

    raw = generate_text(_SYSTEM_PROMPT, f"Query: {query}{context}", max_tokens=1000, json_mode=True)
    
    fallback_plan = {
        "complexity": "low",
        "primary_angles": [],
        "suggested_sub_queries": [query],
        "requires_deep_critique": False,
    }
    
    plan = safe_json_loads(raw, default_fallback=fallback_plan)

    # Ensure required keys exist and have correct formats
    if not isinstance(plan, dict):
        plan = fallback_plan
    else:
        plan = {
            "complexity": plan.get("complexity", "low"),
            "primary_angles": plan.get("primary_angles", []),
            "suggested_sub_queries": plan.get("suggested_sub_queries") or [query],
            "requires_deep_critique": plan.get("requires_deep_critique", False),
        }

    logger.info(f"[supervisor] complexity={plan['complexity']} | angles={plan['primary_angles']}")

    return {
        "agent_logs": [{
            "agent": "supervisor",
            "plan": plan,
        }],
        "iterations": 0,
    }

