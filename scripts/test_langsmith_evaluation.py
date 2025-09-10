#!/usr/bin/env python3
"""
Test script for LangSmith evaluation functionality.
This script can be run to test the evaluation system without full setup.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from assignment_omni.eval.langsmith_evaluator import LangSmithEvaluator
from assignment_omni.graph.pipeline import build_graph


def test_evaluation_system() -> None:
    """Test the evaluation system with sample queries."""
    print("🧪 Testing LangSmith Evaluation System")
    print("=" * 40)
    
    # Initialize evaluator
    evaluator = LangSmithEvaluator()
    
    if not evaluator.client:
        print("⚠️  LangSmith not configured. Running in test mode...")
        print("   Set LANGSMITH_API_KEY in .env to enable full evaluation.")
        return
    
    # Test queries
    test_queries = [
        "What's the weather in London?",
        "Summarize this document",
        "Temperature in New York",
        "What is this PDF about?"
    ]
    
    print(f"\n🔍 Testing with {len(test_queries)} queries...")
    
    # Build graph
    graph = build_graph()
    
    # Run evaluation
    results = evaluator.run_comparative_analysis(test_queries, graph)
    
    # Print results
    print("\n📊 Evaluation Results:")
    print("-" * 40)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Query: {result.query}")
        print(f"   Route: {result.route}")
        print(f"   Overall Score: {result.metrics.get('overall_score', 0):.2f}")
        print(f"   Response: {result.response[:100]}...")
    
    # Generate report
    report = evaluator.generate_evaluation_report(results)
    
    print(f"\n📈 Summary Report:")
    print(f"   Total queries: {report['summary']['total_queries']}")
    print(f"   Average score: {report['summary']['overall_avg_score']:.2f}")
    print(f"   Min score: {report['summary']['min_score']:.2f}")
    print(f"   Max score: {report['summary']['max_score']:.2f}")
    
    # Save results
    results_file = evaluator.save_evaluation_results(results)
    print(f"\n💾 Results saved to: {results_file}")
    
    print("\n✅ Evaluation test completed!")


def test_without_langsmith() -> None:
    """Test evaluation without LangSmith (basic functionality)."""
    print("🧪 Testing Basic Evaluation (No LangSmith)")
    print("=" * 40)
    
    # Test queries
    test_queries = [
        "What's the weather in London?",
        "Summarize this document",
        "Temperature in New York",
        "What is this PDF about?"
    ]
    
    # Build graph
    graph = build_graph()
    
    print(f"\n🔍 Testing with {len(test_queries)} queries...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        try:
            result = graph.invoke({"query": query})
            print(f"   Route: {result.get('route', 'unknown')}")
            print(f"   Response: {result.get('result', 'No response')[:100]}...")
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n✅ Basic evaluation test completed!")


def main() -> None:
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] == "--no-langsmith":
        test_without_langsmith()
    else:
        test_evaluation_system()


if __name__ == "__main__":
    main()
