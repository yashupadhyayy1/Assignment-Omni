from __future__ import annotations

from typing import Any, Dict, Literal, TypedDict

from langgraph.graph import StateGraph, END
from .nodes import weather_node, rag_node


class GraphState(TypedDict, total=False):
    query: str
    route: Literal["weather", "rag"]
    result: str


def router(state: GraphState) -> Literal["weather", "rag"]:
    q = (state.get("query") or "").lower()
    if any(k in q for k in ["weather", "temperature", "rain", "forecast", "climate"]):
        return "weather"
    return "rag"


def build_graph():
    """Build and compile the LangGraph pipeline."""
    graph = StateGraph(GraphState)
    
    # Add nodes
    graph.add_node("weather", weather_node)
    graph.add_node("rag", rag_node)
    
    # Add routing node
    def route_node(state: GraphState) -> GraphState:
        # Ensure state contains 'query' from inputs
        query = state.get("query")
        print(f"route_node: query={query!r}")
        choice = router({"query": query})
        print(f"route_node: route={choice}")
        return {"query": query, "route": choice}
    
    graph.add_node("route", route_node)
    graph.set_entry_point("route")
    
    # Add conditional edges
    def next_edge(state: GraphState):
        route = state.get("route")
        print(f"next_edge: route={route}")
        return route

    graph.add_conditional_edges("route", next_edge, {"weather": "weather", "rag": "rag"})
    graph.add_edge("weather", END)
    graph.add_edge("rag", END)
    print("graph compiled")
    
    # Compile without external checkpointer for simple usage
    return graph.compile()


