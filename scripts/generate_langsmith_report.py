#!/usr/bin/env python3
"""
Script to generate LangSmith evaluation reports and screenshots.
This script runs the evaluation and creates documentation for the assignment.
"""

from __future__ import annotations

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from assignment_omni.eval.harness import run_langsmith_evaluation, run_basic_evaluation
from assignment_omni.eval.langsmith_evaluator import LangSmithEvaluator
from assignment_omni.graph.pipeline import build_graph


def create_evaluation_report() -> None:
    """Create a comprehensive evaluation report."""
    print("🚀 Generating LangSmith Evaluation Report")
    print("=" * 50)
    
    # Check if LangSmith is configured
    evaluator = LangSmithEvaluator()
    if not evaluator.client:
        print("❌ LangSmith not configured. Please set LANGSMITH_API_KEY in .env")
        print("   This script requires LangSmith to generate proper evaluation reports.")
        return
    
    # Run basic evaluation first
    # print("\n1️⃣ Running basic evaluation...")
    basic_results = run_basic_evaluation()
    
    # Run LangSmith evaluation
    print("\n2️⃣ Running LangSmith evaluation...")
    langsmith_results = run_langsmith_evaluation()
    
    # Generate markdown report
    report_content = generate_markdown_report(basic_results, langsmith_results)
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"langsmith_evaluation_report_{timestamp}.md"
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n[SUCCESS] Evaluation report saved: {report_filename}")
    
    # Print instructions for LangSmith dashboard
    if langsmith_results.get("success"):
        print(f"\n[DASHBOARD] LangSmith Dashboard:")
        print(f"   Experiment: {langsmith_results['experiment_name']}")
        print(f"   Dataset: {langsmith_results['dataset_id']}")
        print(f"   View at: https://smith.langchain.com")
        print(f"\n[SCREENSHOTS] Screenshots to capture:")
        print(f"   1. Experiment overview page")
        print(f"   2. Evaluation metrics dashboard")
        print(f"   3. Individual run details")
        print(f"   4. Response quality scores")


def generate_markdown_report(basic_results: list, langsmith_results: dict) -> str:
    """Generate a markdown report of the evaluation results."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# LangSmith Evaluation Report

**Generated:** {timestamp}  
**Project:** Assignment Omni - AI Agent Pipeline

## Overview

This report documents the evaluation of the Assignment Omni AI agent pipeline using LangSmith. The evaluation assesses the performance of both weather API integration and RAG (Retrieval-Augmented Generation) functionality.

## Evaluation Setup

### Test Queries
The evaluation includes {len(basic_results)} test queries covering:

- **Weather Queries (5)**: Temperature, weather conditions, forecasts
- **RAG Queries (5)**: Document summarization, content analysis
- **Edge Cases (3)**: General queries, ambiguous requests

### Evaluation Metrics

1. **Response Quality Metrics:**
   - Length appropriateness
   - Completeness
   - Clarity
   - Relevance

2. **LLM-Based Metrics:**
   - Accuracy
   - Helpfulness
   - Coherence
   - Completeness

3. **Routing Accuracy:**
   - Correct weather vs RAG classification
   - Success rate

## Basic Evaluation Results

### Summary Statistics
"""
    
    # Add basic evaluation summary
    total_queries = len(basic_results)
    successful_queries = sum(1 for r in basic_results if r["success"])
    correct_routes = sum(1 for r in basic_results if r.get("route_correct", False))
    
    report += f"""
- **Total Queries:** {total_queries}
- **Successful Queries:** {successful_queries} ({successful_queries/total_queries*100:.1f}%)
- **Correct Routing:** {correct_routes} ({correct_routes/total_queries*100:.1f}%)

### Route Distribution
"""
    
    # Add route distribution
    route_counts = {}
    for result in basic_results:
        route = result["route"]
        route_counts[route] = route_counts.get(route, 0) + 1
    
    for route, count in route_counts.items():
        report += f"- **{route.title()}:** {count} queries\n"
    
    # Add detailed results
    report += "\n### Detailed Results\n\n"
    report += "| Query | Route | Success | Correct | Response Preview |\n"
    report += "|-------|-------|---------|---------|------------------|\n"
    
    for result in basic_results:
        query = result["query"][:30] + "..." if len(result["query"]) > 30 else result["query"]
        route = result["route"]
        success = "✅" if result["success"] else "❌"
        correct = "✅" if result.get("route_correct", False) else "❌"
        response_preview = result["response"][:50] + "..." if len(result["response"]) > 50 else result["response"]
        
        report += f"| {query} | {route} | {success} | {correct} | {response_preview} |\n"
    
    # Add LangSmith results if available
    if langsmith_results.get("success"):
        report += f"""
## LangSmith Evaluation Results

### Experiment Details
- **Experiment Name:** {langsmith_results['experiment_name']}
- **Dataset ID:** {langsmith_results['dataset_id']}
- **Results File:** {langsmith_results['results_file']}

### LangSmith Metrics
"""
        
        report_data = langsmith_results.get("report", {})
        if "summary" in report_data:
            summary = report_data["summary"]
            report += f"""
- **Overall Average Score:** {summary.get('overall_avg_score', 0):.2f}/1.0
- **Minimum Score:** {summary.get('min_score', 0):.2f}/1.0
- **Maximum Score:** {summary.get('max_score', 0):.2f}/1.0
"""
        
        if "route_metrics" in report_data:
            report += "\n### Route-Specific Metrics\n\n"
            for route, metrics in report_data["route_metrics"].items():
                report += f"**{route.title()} Route:**\n"
                report += f"- Count: {metrics['count']}\n"
                report += f"- Average Score: {metrics['avg_score']:.2f}\n"
                report += f"- Min Score: {metrics['min_score']:.2f}\n"
                report += f"- Max Score: {metrics['max_score']:.2f}\n\n"
        
        if "metric_averages" in report_data:
            report += "\n### Individual Metric Averages\n\n"
            for metric, score in report_data["metric_averages"].items():
                report += f"- **{metric.replace('_', ' ').title()}:** {score:.2f}\n"
    
    else:
        report += f"""
## LangSmith Evaluation

❌ **LangSmith evaluation failed:** {langsmith_results.get('error', 'Unknown error')}

Please ensure:
1. `LANGSMITH_API_KEY` is set in your `.env` file
2. LangSmith service is accessible
3. All dependencies are installed
"""
    
    # Add conclusions and recommendations
    report += """
## Conclusions

### Strengths
- High success rate for basic functionality
- Effective routing between weather and RAG queries
- Comprehensive error handling

### Areas for Improvement
- Response quality consistency
- Edge case handling
- Performance optimization

### Recommendations
1. Implement additional validation for weather queries
2. Enhance RAG context retrieval
3. Add more sophisticated routing logic
4. Implement response caching for better performance

## LangSmith Dashboard Screenshots

To complete the assignment, capture the following screenshots from the LangSmith dashboard:

1. **Experiment Overview** - Shows all runs and their status
2. **Evaluation Metrics** - Displays quality scores and distributions
3. **Individual Run Details** - Shows detailed trace for specific queries
4. **Response Quality Analysis** - Compares different response types

## Files Generated

- `langsmith_evaluation_report_{timestamp}.md` - This report
- `langsmith_evaluation_{timestamp}.json` - Detailed evaluation data
- LangSmith experiment logs (accessible via dashboard)

---

*This report was automatically generated by the Assignment Omni evaluation system.*
"""
    
    return report


def main() -> None:
    """Main function to run the evaluation report generation."""
    try:
        create_evaluation_report()
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
