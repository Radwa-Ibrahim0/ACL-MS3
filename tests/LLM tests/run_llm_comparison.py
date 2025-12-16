"""
LLM Comparison Runner
=====================

Runs test queries through the full pipeline and collects:
1. The context data (baseline + embedding results) sent to LLMs
2. All 3 LLM responses

Output is saved to JSON for manual accuracy evaluation.
"""

import json
import time
import sys
import os
# Add project root to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from llm_test_queries import BASE_QUERIES, EDGE_CASE_QUERIES, TestQuery, EdgeCaseQuery
from Main.llm_layer import (
    build_retrieval_context,
    build_structured_prompt,
    build_default_model_adapters,
    RetrievalContext
)
from Main.embedding_bge_m3 import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result for a single query with context and all LLM responses."""
    query_id: str
    query_text: str
    difficulty: str  # or category for edge cases
    
    # The data that was sent to all LLMs (same for all 3)
    context_data: Dict[str, Any]
    prompt_sent: str
    
    # Responses from each LLM
    llm_responses: Dict[str, Dict[str, Any]]  # model_name -> {response, time, tokens}


def run_single_query(
    query_id: str,
    query_text: str,
    difficulty: str,
    adapters: List,
    config: Dict[str, str],
    verbose: bool = True
) -> QueryResult:
    """Run a single query through pipeline and get all 3 LLM responses."""
    
    print(f"\n{'='*70}")
    print(f"[{query_id}] {query_text}")
    print('='*70)
    
    # Step 1: Get context from your pipeline (baseline + embedding)
    ctx = build_retrieval_context(query_text, config=config)
    
    # Step 2: Build the prompt (same prompt goes to all 3 LLMs)
    prompt = build_structured_prompt(ctx)
    
    # Step 3: Collect context data for output
    context_data = {
        "user_query": ctx.user_query,
        "intent": ctx.intent,
        "entities": ctx.entities,
        "baseline_description": ctx.baseline_desc,
        "baseline_is_fallback": ctx.baseline_is_fallback,
        "baseline_results": ctx.baseline_results[:30],  # Limit for readability
        "embedding_results": ctx.embedding_results[:30],  # Limit for readability
    }
    
    if verbose:
        print(f"\n📦 CONTEXT DATA:")
        print(f"   Intent: {ctx.intent}")
        print(f"   Entities: {ctx.entities}")
        print(f"   Baseline results: {len(ctx.baseline_results)} rows")
        print(f"   Embedding results: {len(ctx.embedding_results)} rows")
    
    # Step 4: Query all 3 LLMs with the same prompt
    llm_responses = {}
    print(f"\n🤖 LLM RESPONSES:")
    
    for adapter in adapters:
        model_name = adapter.name
        print(f"\n--- {model_name} ---")
        
        try:
            start = time.time()
            response_text, metrics = adapter.generate(prompt)
            elapsed = time.time() - start
            
            llm_responses[model_name] = {
                "response": response_text,
                "response_time_sec": round(elapsed, 2),
                "input_tokens": metrics.input_tokens,
                "output_tokens": metrics.output_tokens,
                "error": None
            }
            
            # Print response (truncated for terminal)
            print(f"⏱️  Time: {elapsed:.2f}s | Tokens: {metrics.input_tokens}/{metrics.output_tokens}")
            display_text = response_text[:600] if response_text else "No response"
            if len(response_text) > 600:
                display_text += "..."
            print(f"Response:\n{display_text}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            llm_responses[model_name] = {
                "response": None,
                "response_time_sec": None,
                "input_tokens": None,
                "output_tokens": None,
                "error": str(e)
            }
    
    return QueryResult(
        query_id=query_id,
        query_text=query_text,
        difficulty=difficulty,
        context_data=context_data,
        prompt_sent=prompt,
        llm_responses=llm_responses
    )


def run_base_queries(adapters: List, config: Dict[str, str]) -> List[QueryResult]:
    """Run all base queries."""
    results = []
    for q in BASE_QUERIES:
        result = run_single_query(q.id, q.query, q.difficulty, adapters, config, verbose=True)
        results.append(result)
    return results


def run_edge_queries(adapters: List, config: Dict[str, str]) -> List[QueryResult]:
    """Run all edge case queries."""
    results = []
    for q in EDGE_CASE_QUERIES:
        result = run_single_query(q.id, q.query, q.category, adapters, config, verbose=True)
        results.append(result)
    return results


def save_results(
    base_results: List[QueryResult],
    edge_results: List[QueryResult],
    output_file: str
):
    """Save all results to a text file in readable JSON format."""
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_base_queries": len(base_results),
        "total_edge_queries": len(edge_results),
        "base_query_results": [],
        "edge_case_results": []
    }
    
    # Format base results
    for r in base_results:
        result_dict = {
            "query_id": r.query_id,
            "query_text": r.query_text,
            "difficulty": r.difficulty,
            "context_data": {
                "intent": r.context_data["intent"],
                "entities": r.context_data["entities"],
                "baseline_results": r.context_data["baseline_results"],
                "embedding_results": r.context_data["embedding_results"]
            },
            "llm_responses": {}
        }
        for model, resp in r.llm_responses.items():
            result_dict["llm_responses"][model] = {
                "response": resp["response"],
                "response_time_sec": resp["response_time_sec"],
                "error": resp["error"]
            }
        output["base_query_results"].append(result_dict)
    
    # Format edge results
    for r in edge_results:
        result_dict = {
            "query_id": r.query_id,
            "query_text": r.query_text,
            "category": r.difficulty,  # This is actually category for edge cases
            "context_data": {
                "intent": r.context_data["intent"],
                "entities": r.context_data["entities"],
                "baseline_results": r.context_data["baseline_results"],
                "embedding_results": r.context_data["embedding_results"]
            },
            "llm_responses": {}
        }
        for model, resp in r.llm_responses.items():
            result_dict["llm_responses"][model] = {
                "response": resp["response"],
                "response_time_sec": resp["response_time_sec"],
                "error": resp["error"]
            }
        output["edge_case_results"].append(result_dict)
    
    # Save to file with nice formatting
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to {output_file}")


def print_summary(base_results: List[QueryResult], edge_results: List[QueryResult]):
    """Print a quick summary of results."""
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    
    print(f"\n📊 Base Queries: {len(base_results)}")
    print(f"🔬 Edge Cases: {len(edge_results)}")
    
    # Get model names from first result
    if base_results:
        models = list(base_results[0].llm_responses.keys())
        print(f"\n🤖 Models tested: {len(models)}")
        for m in models:
            print(f"   - {m}")
    
    # Quick response time summary
    print("\n⏱️  Average Response Times:")
    if base_results and base_results[0].llm_responses:
        for model in base_results[0].llm_responses.keys():
            times = [
                r.llm_responses[model]["response_time_sec"] 
                for r in base_results + edge_results
                if r.llm_responses[model]["response_time_sec"] is not None
            ]
            if times:
                avg = sum(times) / len(times)
                print(f"   {model}: {avg:.2f}s")


def run_all(output_file: str = "llm_test_results.txt", run_edge: bool = True):
    """Run the full evaluation."""
    
    print("=" * 70)
    print("FPL LLM COMPARISON TEST")
    print("=" * 70)
    
    # Load config and build adapters
    config = load_config()
    adapters = build_default_model_adapters(config)
    
    if not adapters:
        print("❌ No LLM adapters available. Check your config.txt for API keys.")
        return None
    
    print(f"\n✅ Loaded {len(adapters)} LLM adapters")
    for a in adapters:
        print(f"   - {a.name}")
    
    # Run base queries
    print(f"\n📊 Running {len(BASE_QUERIES)} base queries...")
    base_results = run_base_queries(adapters, config)
    
    # Run edge cases (optional)
    edge_results = []
    if run_edge:
        print(f"\n🔬 Running {len(EDGE_CASE_QUERIES)} edge case queries...")
        edge_results = run_edge_queries(adapters, config)
    
    # Save results
    save_results(base_results, edge_results, output_file)
    
    # Print summary
    print_summary(base_results, edge_results)
    
    print(f"\n📁 Full results saved to: {output_file}")
    print("\nYou can now evaluate accuracy by comparing each LLM response")
    print("to the context_data that was provided to it.")
    
    return {
        "base_results": base_results,
        "edge_results": edge_results
    }


def run_single_test(query_text: str, output_file: str = None):
    """Run a single custom query for testing."""
    
    config = load_config()
    adapters = build_default_model_adapters(config)
    
    if not adapters:
        print("❌ No LLM adapters available.")
        return None
    
    result = run_single_query("CUSTOM", query_text, "custom", adapters, config)
    
    # Print results
    print("\n" + "=" * 70)
    print(f"Query: {query_text}")
    print("=" * 70)
    
    print("\n📦 CONTEXT DATA PROVIDED TO LLMs:")
    print(f"  Intent: {result.context_data['intent']}")
    print(f"  Entities: {result.context_data['entities']}")
    print(f"  Baseline results: {len(result.context_data['baseline_results'])} rows")
    print(f"  Embedding results: {len(result.context_data['embedding_results'])} rows")
    
    print("\n🤖 LLM RESPONSES:")
    for model, resp in result.llm_responses.items():
        print(f"\n--- {model} ({resp['response_time_sec']}s) ---")
        if resp['response']:
            # Print first 500 chars
            text = resp['response'][:500]
            if len(resp['response']) > 500:
                text += "..."
            print(text)
        else:
            print(f"ERROR: {resp['error']}")
    
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)
        print(f"\n📁 Saved to {output_file}")
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--single":
            # Run single query mode
            query = input("Enter your query: ").strip()
            if query:
                run_single_test(query, "single_test_result.txt")
        elif sys.argv[1] == "--base-only":
            # Run only base queries (no edge cases)
            run_all("llm_test_results.txt", run_edge=False)
        elif sys.argv[1] == "--edge-only":
            # Run only edge case queries
            print("=" * 70)
            print("FPL LLM COMPARISON TEST - EDGE CASES ONLY")
            print("=" * 70)
            config = load_config()
            adapters = build_default_model_adapters(config)
            if adapters:
                edge_results = run_edge_queries(adapters, config)
                save_results([], edge_results, "llm_edge_results.txt")
        else:
            print("Usage:")
            print("  python run_llm_comparison.py           # Run all queries")
            print("  python run_llm_comparison.py --base-only   # Run only base queries")
            print("  python run_llm_comparison.py --edge-only   # Run only edge cases")
            print("  python run_llm_comparison.py --single      # Run single custom query")
    else:
        # Run full evaluation
        run_all("llm_test_results.txt", run_edge=True)
