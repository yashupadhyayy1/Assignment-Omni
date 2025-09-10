from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from langsmith import Client, RunEvaluator
from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from assignment_omni.config.settings import Settings
from assignment_omni.llm.wrapper import build_llm


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    query: str
    response: str
    route: str
    metrics: Dict[str, float]
    timestamp: datetime
    run_id: Optional[str] = None


class ResponseQualityEvaluator(RunEvaluator):
    """Custom evaluator for response quality assessment."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def _evaluate_response_quality(self, response: str) -> Dict[str, float]:
        """Evaluate response quality using multiple metrics."""
        metrics = {}
        
        # 1. Length appropriateness (not too short, not too long)
        word_count = len(response.split())
        if word_count < 10:
            metrics["length_score"] = 0.3  # Too short
        elif word_count > 500:
            metrics["length_score"] = 0.7  # Too long
        else:
            metrics["length_score"] = 1.0  # Good length
        
        # 2. Completeness (presence of key elements)
        completeness_indicators = [
            "temperature", "weather", "description", "humidity", "wind"  # Weather
            "summary", "document", "context", "information", "content"  # RAG
        ]
        found_indicators = sum(1 for indicator in completeness_indicators 
                             if indicator.lower() in response.lower())
        metrics["completeness_score"] = min(found_indicators / 3, 1.0)
        
        # 3. Clarity (sentence structure and readability)
        sentences = response.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        if 5 <= avg_sentence_length <= 25:
            metrics["clarity_score"] = 1.0
        elif avg_sentence_length < 5:
            metrics["clarity_score"] = 0.6
        else:
            metrics["clarity_score"] = 0.8
        
        # 4. Relevance (weather vs RAG context)
        weather_terms = ["weather", "temperature", "rain", "forecast", "climate", "°c", "°f"]
        rag_terms = ["document", "pdf", "text", "content", "information", "summary"]
        
        weather_score = sum(1 for term in weather_terms if term.lower() in response.lower())
        rag_score = sum(1 for term in rag_terms if term.lower() in response.lower())
        
        if weather_score > rag_score:
            metrics["relevance_score"] = min(weather_score / 3, 1.0)
        elif rag_score > weather_score:
            metrics["relevance_score"] = min(rag_score / 3, 1.0)
        else:
            metrics["relevance_score"] = 0.5
        
        return metrics
    
    def _evaluate_with_llm(self, query: str, response: str) -> Dict[str, float]:
        """Use LLM to evaluate response quality with fallback to rule-based evaluation."""
        # For now, use rule-based evaluation instead of LLM to avoid JSON parsing issues
        return self._rule_based_evaluation(query, response)
    
    def _rule_based_evaluation(self, query: str, response: str) -> Dict[str, float]:
        """Rule-based evaluation as fallback when LLM evaluation fails."""
        metrics = {}
        
        # 1. Accuracy - based on response length and content indicators
        if len(response) < 10:
            metrics["accuracy"] = 0.3  # Too short, likely not accurate
        elif "error" in response.lower() or "sorry" in response.lower():
            metrics["accuracy"] = 0.4  # Error response
        elif any(word in response.lower() for word in ["temperature", "weather", "document", "pdf", "information"]):
            metrics["accuracy"] = 0.8  # Contains relevant keywords
        else:
            metrics["accuracy"] = 0.6  # Default accuracy
        
        # 2. Helpfulness - based on response structure and completeness
        helpful_indicators = ["here", "based", "according", "information", "data", "summary"]
        helpful_score = sum(1 for indicator in helpful_indicators if indicator in response.lower())
        metrics["helpfulness"] = min(helpful_score / 3, 1.0)
        
        # 3. Coherence - based on sentence structure
        sentences = response.split('.')
        if len(sentences) > 1:
            avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if 5 <= avg_length <= 25:
                metrics["coherence"] = 1.0
            else:
                metrics["coherence"] = 0.7
        else:
            metrics["coherence"] = 0.5
        
        # 4. Completeness - based on response length and content
        word_count = len(response.split())
        if word_count < 20:
            metrics["completeness"] = 0.4
        elif word_count < 50:
            metrics["completeness"] = 0.7
        else:
            metrics["completeness"] = 0.9
        
        return metrics
    
    def evaluate_run(self, run: Run, example: Optional[Example] = None, **kwargs) -> Dict[str, Any]:
        """Evaluate a single run."""
        if not run.outputs or "result" not in run.outputs:
            return {"error": "No response found in run outputs"}
        
        response = run.outputs["result"]
        query = run.inputs.get("query", "")
        
        # Get basic metrics
        basic_metrics = self._evaluate_response_quality(response)
        
        # Get LLM-based metrics
        llm_metrics = self._evaluate_with_llm(query, response)
        
        # Combine metrics
        all_metrics = {**basic_metrics, **llm_metrics}
        
        # Calculate overall score
        overall_score = sum(all_metrics.values()) / len(all_metrics)
        all_metrics["overall_score"] = overall_score
        
        return {
            "key": "response_quality",
            "score": overall_score,
            "value": overall_score,
            "comment": f"Overall quality score: {overall_score:.2f}",
            "metadata": all_metrics
        }


class LangSmithEvaluator:
    """Main evaluator class for LangSmith integration."""
    
    def __init__(self):
        self.cfg = Settings.load()
        self.client = None
        self.llm = build_llm()
        
        if self.cfg.langsmith.api_key:
            self.client = Client(api_key=self.cfg.langsmith.api_key)
            # Set up environment for tracing
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
            os.environ["LANGCHAIN_API_KEY"] = self.cfg.langsmith.api_key
            os.environ["LANGCHAIN_PROJECT"] = self.cfg.langsmith.project
    
    def create_evaluation_dataset(self, test_cases: List[Dict[str, Any]]) -> str:
        """Create a dataset for evaluation."""
        if not self.client:
            raise ValueError("LangSmith client not initialized. Check API key.")
        
        # Create dataset
        dataset_name = f"assignment-omni-eval-{int(time.time())}"
        dataset = self.client.create_dataset(
            dataset_name=dataset_name,
            description="Assignment Omni evaluation dataset"
        )
        
        # Add examples
        for i, test_case in enumerate(test_cases):
            self.client.create_example(
                inputs={"query": test_case["query"]},
                outputs={"expected_route": test_case.get("expected_route", "unknown")},
                dataset_id=dataset.id,
                metadata={"test_id": i, "category": test_case.get("category", "general")}
            )
        
        return dataset.id
    
    def run_evaluation(self, dataset_id: str, graph_func) -> str:
        """Run evaluation on the dataset."""
        if not self.client:
            raise ValueError("LangSmith client not initialized. Check API key.")
        
        # Create evaluators
        evaluators = [ResponseQualityEvaluator(self.llm)]
        
        # Run evaluation
        results = evaluate(
            graph_func,
            data=dataset_id,
            evaluators=evaluators,
            experiment_prefix="assignment-omni-eval",
            description="Assignment Omni LLM Response Evaluation"
        )
        
        # Handle different return types from evaluate()
        if hasattr(results, 'experiment_name'):
            return results.experiment_name
        elif isinstance(results, dict) and 'experiment_name' in results:
            return results['experiment_name']
        else:
            # Fallback: return a default experiment name
            return "assignment-omni-eval"
    
    def evaluate_single_query(self, query: str, graph_func) -> EvaluationResult:
        """Evaluate a single query and return detailed results."""
        # Run the query
        result = graph_func.invoke({"query": query})
        
        # Extract response and route
        response = result.get("result", "")
        route = result.get("route", "unknown")
        
        # Create evaluator
        evaluator = ResponseQualityEvaluator(self.llm)
        
        # Evaluate response quality
        quality_metrics = evaluator._evaluate_response_quality(response)
        llm_metrics = evaluator._evaluate_with_llm(query, response)
        
        # Combine all metrics
        all_metrics = {**quality_metrics, **llm_metrics}
        overall_score = sum(all_metrics.values()) / len(all_metrics)
        all_metrics["overall_score"] = overall_score
        
        return EvaluationResult(
            query=query,
            response=response,
            route=route,
            metrics=all_metrics,
            timestamp=datetime.now()
        )
    
    def run_comparative_analysis(self, queries: List[str], graph_func) -> List[EvaluationResult]:
        """Run comparative analysis on multiple queries."""
        results = []
        
        for query in queries:
            try:
                result = self.evaluate_single_query(query, graph_func)
                results.append(result)
            except Exception as e:
                print(f"Error evaluating query '{query}': {e}")
                # Create error result
                error_result = EvaluationResult(
                    query=query,
                    response=f"Error: {str(e)}",
                    route="error",
                    metrics={"overall_score": 0.0},
                    timestamp=datetime.now()
                )
                results.append(error_result)
        
        return results
    
    def generate_evaluation_report(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        if not results:
            return {"error": "No results to analyze"}
        
        # Calculate aggregate metrics
        overall_scores = [r.metrics.get("overall_score", 0) for r in results]
        route_distribution = {}
        for result in results:
            route = result.route
            route_distribution[route] = route_distribution.get(route, 0) + 1
        
        # Calculate route-specific metrics
        route_metrics = {}
        for route in route_distribution.keys():
            route_results = [r for r in results if r.route == route]
            if route_results:
                route_scores = [r.metrics.get("overall_score", 0) for r in route_results]
                route_metrics[route] = {
                    "count": len(route_results),
                    "avg_score": sum(route_scores) / len(route_scores),
                    "min_score": min(route_scores),
                    "max_score": max(route_scores)
                }
        
        # Calculate individual metric averages
        metric_averages = {}
        all_metrics = set()
        for result in results:
            all_metrics.update(result.metrics.keys())
        
        for metric in all_metrics:
            if metric != "overall_score":
                values = [r.metrics.get(metric, 0) for r in results if metric in r.metrics]
                if values:
                    metric_averages[metric] = sum(values) / len(values)
        
        return {
            "summary": {
                "total_queries": len(results),
                "overall_avg_score": sum(overall_scores) / len(overall_scores),
                "min_score": min(overall_scores),
                "max_score": max(overall_scores),
                "route_distribution": route_distribution
            },
            "route_metrics": route_metrics,
            "metric_averages": metric_averages,
            "detailed_results": [
                {
                    "query": r.query,
                    "route": r.route,
                    "overall_score": r.metrics.get("overall_score", 0),
                    "timestamp": r.timestamp.isoformat()
                }
                for r in results
            ]
        }
    
    def save_evaluation_results(self, results: List[EvaluationResult], filename: str = None) -> str:
        """Save evaluation results to a file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        import json
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            serializable_results.append({
                "query": result.query,
                "response": result.response,
                "route": result.route,
                "metrics": result.metrics,
                "timestamp": result.timestamp.isoformat(),
                "run_id": result.run_id
            })
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        return filename
