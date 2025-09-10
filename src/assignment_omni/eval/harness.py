from __future__ import annotations

from typing import List, Dict, Any
from assignment_omni.graph.pipeline import build_graph
from assignment_omni.config.settings import Settings


def evaluate_weather_queries() -> List[Dict[str, Any]]:
    """Evaluate weather-related queries."""
    graph = build_graph()
    test_queries = [
        "What's the weather like in London?",
        "Temperature in New York",
        "Is it raining in Tokyo?",
        "Weather forecast for Paris"
    ]
    
    results = []
    for query in test_queries:
        try:
            result = graph.invoke({"query": query})
            results.append({
                "query": query,
                "response": result.get("result", ""),
                "route": result.get("route", "unknown"),
                "success": True
            })
        except Exception as e:
            results.append({
                "query": query,
                "response": str(e),
                "route": "error",
                "success": False
            })
    
    return results


def evaluate_rag_queries() -> List[Dict[str, Any]]:
    """Evaluate RAG-related queries (requires PDF to be loaded first)."""
    graph = build_graph()
    test_queries = [
        "What is this document about?",
        "Summarize the main points",
        "What are the key findings?",
        "Tell me about the methodology"
    ]
    
    results = []
    for query in test_queries:
        try:
            result = graph.invoke({"query": query})
            results.append({
                "query": query,
                "response": result.get("result", ""),
                "route": result.get("route", "unknown"),
                "success": True
            })
        except Exception as e:
            results.append({
                "query": query,
                "response": str(e),
                "route": "error",
                "success": False
            })
    
    return results


def run_evaluation() -> None:
    """Run full evaluation and print results."""
    print("=== Weather Query Evaluation ===")
    weather_results = evaluate_weather_queries()
    for result in weather_results:
        print(f"Query: {result['query']}")
        print(f"Route: {result['route']}")
        print(f"Success: {result['success']}")
        print(f"Response: {result['response'][:100]}...")
        print("-" * 50)
    
    print("\n=== RAG Query Evaluation ===")
    rag_results = evaluate_rag_queries()
    for result in rag_results:
        print(f"Query: {result['query']}")
        print(f"Route: {result['route']}")
        print(f"Success: {result['success']}")
        print(f"Response: {result['response'][:100]}...")
        print("-" * 50)


if __name__ == "__main__":
    run_evaluation()
