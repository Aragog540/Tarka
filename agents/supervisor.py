import json

from llm import generate_text
from observability.logger import log_agent_call, logger
from state.schema import ResearchState

_SYSTEM_PROMPT = """You are a research supervisor. Given a user query, analyze it and produce a routing plan.



def _conversation_context(state: ResearchState) -> str:
    context = state.get("conversation_context", "").strip()
    return f"\n\nConversation context:\n{context}" if context else ""
Return ONLY valid JSON with this exact shape:
{
  "complexity": "low" | "medium" | "high",
  "primary_angles": ["angle1", "angle2"],
  "suggested_sub_queries": ["query1", "query2", "query3"],
  "requires_deep_critique": true | false,
    raw = generate_text(_SYSTEM_PROMPT, f"Query: {query}{_conversation_context(state)}", max_tokens=1000, json_mode=True)
}"""


@log_agent_call("supervisor")
def supervisor_node(state: ResearchState) -> dict:
    query = state["query"]

    raw = generate_text(_SYSTEM_PROMPT, f"Query: {query}", max_tokens=1000, json_mode=True)
    plan = json.loads(raw)

    logger.info(f"[supervisor] complexity={plan['complexity']} | angles={plan['primary_angles']}")

    return {
        "agent_logs": [{
            "agent": "supervisor",
            "plan": plan,
        }],
        "iterations": 0,
    }
