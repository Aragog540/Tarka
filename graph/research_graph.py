from langgraph.graph import END, StateGraph

from agents.aggregator import aggregator_node
from agents.critic import critic_node, should_continue
from agents.searcher import searcher_node
from agents.supervisor import supervisor_node
from agents.summarizer import summarizer_node
from state.schema import ResearchState


def build_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("searcher", searcher_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("aggregator", aggregator_node)

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "searcher")
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
