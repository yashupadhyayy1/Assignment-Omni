from __future__ import annotations

import json
import os
from typing import List, Dict, Any
from datetime import datetime

from assignment_omni.graph.pipeline import build_graph
from assignment_omni.config.settings import Settings
from assignment_omni.eval.langsmith_evaluator import LangSmithEvaluator, EvaluationResult


def get_test_queries() -> List[Dict[str, Any]]:
    """Get comprehensive test queries for evaluation."""
    return [
        # Weather queries
        {"query": "What's the weather like in London?", "expected_route": "weather", "category": "weather"},
        {"query": "Temperature in New York", "expected_route": "weather", "category": "weather"},
        {"query": "Is it raining in Tokyo?", "expected_route": "weather", "category": "weather"},
        {"query": "Weather forecast for Paris", "expected_route": "weather", "category": "weather"},
        {"query": "How's the climate in Sydney?", "expected_route": "weather", "category": "weather"},
        
        # RAG queries
        {"query": "What is this document about?", "expected_route": "rag", "category": "rag"},
        {"query": "Summarize the main points", "expected_route": "rag", "category": "rag"},
        {"query": "What are the key findings?", "expected_route": "rag", "category": "rag"},
        {"query": "Tell me about the methodology", "expected_route": "rag", "category": "rag"},
        {"query": "What information is available in this PDF?", "expected_route": "rag", "category": "rag"},
        
        # Edge cases
        {"query": "Hello, how are you?", "expected_route": "rag", "category": "general"},
        {"query": "What's the temperature?", "expected_route": "weather", "category": "weather"},
        {"query": "Explain quantum computing", "expected_route": "rag", "category": "rag"},
    ]


def run_basic_evaluation() -> List[Dict[str, Any]]:
    """Run basic evaluation without LangSmith."""
    graph = build_graph()
    test_queries = get_test_queries()
    
    results = []
    for test_case in test_queries:
        query = test_case["query"]
        try:
            result = graph.invoke({"query": query})
            results.append({
                "query": query,
                "response": result.get("result", ""),
                "route": result.get("route", "unknown"),
                "expected_route": test_case["expected_route"],
                "category": test_case["category"],
                "success": True,
                "route_correct": result.get("route", "unknown") == test_case["expected_route"]
            })
        except Exception as e:
            results.append({
                "query": query,
                "response": str(e),
                "route": "error",
                "expected_route": test_case["expected_route"],
                "category": test_case["category"],
                "success": False,
                "route_correct": False
            })
    
    return results


def run_langsmith_evaluation() -> Dict[str, Any]:
    """Run comprehensive LangSmith evaluation."""
    evaluator = LangSmithEvaluator()
    graph = build_graph()
    test_queries = get_test_queries()
    
    print("[EVAL] Running LangSmith evaluation...")
    
    # Check if LangSmith is available
    if not evaluator.client:
        print("[WARNING] LangSmith not available. Check LANGSMITH_API_KEY in .env")
        return {"error": "LangSmith not available"}
    
    try:
        # Create dataset
        print("[DATASET] Creating evaluation dataset...")
        dataset_id = evaluator.create_evaluation_dataset(test_queries)
        print(f"[SUCCESS] Dataset created: {dataset_id}")
        
        # Run evaluation
        print("[EVAL] Running evaluation...")
        experiment_name = evaluator.run_evaluation(dataset_id, graph)
        print(f"[SUCCESS] Evaluation completed: {experiment_name}")
        
        # Run comparative analysis
        print("[ANALYSIS] Running comparative analysis...")
        queries = [tc["query"] for tc in test_queries]
        results = evaluator.run_comparative_analysis(queries, graph)
        
        # Generate report
        print("[REPORT] Generating evaluation report...")
        report = evaluator.generate_evaluation_report(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = evaluator.save_evaluation_results(results, f"langsmith_evaluation_{timestamp}.json")
        print(f"[SAVE] Results saved to: {results_file}")
        
        return {
            "experiment_name": experiment_name,
            "dataset_id": dataset_id,
            "results_file": results_file,
            "report": report,
            "success": True
        }
        
    except Exception as e:
        print(f"[ERROR] LangSmith evaluation failed: {e}")
        return {"error": str(e), "success": False}


def print_evaluation_summary(results: List[Dict[str, Any]]) -> None:
    """Print a summary of evaluation results."""
    if not results:
        print("No results to display.")
        return
    
    total_queries = len(results)
    successful_queries = sum(1 for r in results if r["success"])
    correct_routes = sum(1 for r in results if r.get("route_correct", False))
    
    print(f"\n[SUMMARY] EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total queries: {total_queries}")
    print(f"Successful queries: {successful_queries} ({successful_queries/total_queries*100:.1f}%)")
    print(f"Correct routing: {correct_routes} ({correct_routes/total_queries*100:.1f}%)")
    
    # Route distribution
    route_counts = {}
    for result in results:
        route = result["route"]
        route_counts[route] = route_counts.get(route, 0) + 1
    
    print(f"\nRoute distribution:")
    for route, count in route_counts.items():
        print(f"  {route}: {count} queries")
    
    # Category breakdown
    category_counts = {}
    for result in results:
        category = result.get("category", "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
    
    print(f"\nCategory breakdown:")
    for category, count in category_counts.items():
        print(f"  {category}: {count} queries")


def run_evaluation() -> None:
    """Run full evaluation with both basic and LangSmith evaluation."""
    print("[START] Starting Assignment Omni Evaluation")
    print("="*50)
    
    # Run basic evaluation
    print("\n[STEP 1] Running basic evaluation...")
    basic_results = run_basic_evaluation()
    print_evaluation_summary(basic_results)
    
    # Run LangSmith evaluation
    print("\n[STEP 2] Running LangSmith evaluation...")
    langsmith_results = run_langsmith_evaluation()
    
    if langsmith_results.get("success"):
        print("\n[SUCCESS] LangSmith evaluation completed successfully!")
        print(f"Experiment: {langsmith_results['experiment_name']}")
        print(f"Results file: {langsmith_results['results_file']}")
        
        # Print report summary
        report = langsmith_results.get("report", {})
        if "summary" in report:
            summary = report["summary"]
            print(f"\n[REPORT] LangSmith Report Summary:")
            print(f"  Overall average score: {summary.get('overall_avg_score', 0):.2f}")
            print(f"  Min score: {summary.get('min_score', 0):.2f}")
            print(f"  Max score: {summary.get('max_score', 0):.2f}")
    else:
        print(f"\n[ERROR] LangSmith evaluation failed: {langsmith_results.get('error', 'Unknown error')}")
    
    print("\n[COMPLETE] Evaluation completed!")


if __name__ == "__main__":
    run_evaluation()
