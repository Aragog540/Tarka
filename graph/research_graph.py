from langgraph.graph import END, StateGraph

from agents.aggregator import aggregator_node
from agents.critic import critic_node, should_continue
from agents.searcher import searcher_node
from agents.supervisor import supervisor_node
from agents.summarizer import summarizer_node
from state.schema import ResearchState


def route_from_supervisor(state: ResearchState) -> str:
    import re
    query = state.get("query", "").strip()
    q = re.sub(r'[^\w\s]', '', query.lower())
    social_words = {
        "hi", "hello", "hey", "yo", "sup", "greetings", "howdy",
        "how are you", "how is it going", "hows it going", "whats up", "what up",
        "good morning", "good afternoon", "good evening", "good night",
        "thanks", "thank you", "bye", "goodbye"
    }
    if q in social_words:
        return "aggregator"
    
    words = q.split()
    if len(words) <= 3 and any(w in social_words for w in words):
        if not any(w in words for w in ["what", "why", "how", "is", "define", "who"]):
            return "aggregator"
            
    return "searcher"


def build_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("searcher", searcher_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("aggregator", aggregator_node)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {"searcher": "searcher", "aggregator": "aggregator"},
    )
    graph.add_edge("searcher", "summarizer")
    graph.add_edge("summarizer", "critic")

    graph.add_conditional_edges(
        "critic",
        should_continue,
        {"searcher": "searcher", "aggregator": "aggregator"},
    )

    graph.add_edge("aggregator", END)

    return graph.compile()


research_graph = build_graph()
