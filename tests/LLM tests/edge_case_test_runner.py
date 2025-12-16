"""
Edge Case Test Runner for Error Analysis
=========================================

Runs 15 carefully selected edge case queries to analyze error types:
1. Retrieval Failure - KG query returns no/wrong nodes
2. Context Overload/Noise - Too much irrelevant context provided to LLM
3. LLM Hallucination - Answer not supported by retrieved context
4. Poor Synthesis - Retrieved facts correct but LLM fails to synthesize

Output: Detailed analysis text file with categorized errors.
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

from Main.llm_layer import (
    build_retrieval_context,
    build_structured_prompt,
    build_default_model_adapters,
    RetrievalContext
)
from Main.embedding_bge_m3 import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# 15 EDGE CASE QUERIES FOR ERROR ANALYSIS
# ============================================================

@dataclass
class EdgeCaseQuery:
    """An edge case query designed to expose specific error types."""
    id: str
    query: str
    category: str
    expected_error_type: str  # retrieval_failure, context_overload, hallucination, poor_synthesis
    expected_issue: str
    notes: str = ""


EDGE_CASE_QUERIES_15: List[EdgeCaseQuery] = [
    
    # ============== RETRIEVAL FAILURE CASES (4) ==============
    
    EdgeCaseQuery(
        id="ERR_01",
        query="What was Erling Haaland's salary in 2022-23?",
        category="out_of_scope",
        expected_error_type="retrieval_failure",
        expected_issue="Salary data not in KG - should return empty/fallback results",
        notes="Tests if system correctly signals missing data vs hallucinating"
    ),
    
    EdgeCaseQuery(
        id="ERR_02",
        query="How many goals did Hallend score in 2022-23?",
        category="misspelling",
        expected_error_type="retrieval_failure",
        expected_issue="Misspelled 'Haaland' as 'Hallend' - entity resolution failure",
        notes="Tests robustness to typos in player names"
    ),
    
    EdgeCaseQuery(
        id="ERR_03",
        query="What was CR7's performance in 2021-22?",
        category="nickname",
        expected_error_type="retrieval_failure",
        expected_issue="CR7 is Cristiano Ronaldo's nickname - may not resolve",
        notes="Tests handling of player nicknames/aliases"
    ),
    
    EdgeCaseQuery(
        id="ERR_04",
        query="Who scored the most goals in December 2022?",
        category="temporal",
        expected_error_type="retrieval_failure",
        expected_issue="Monthly breakdown requires date filtering not in KG schema",
        notes="Tests handling of unsupported temporal granularity"
    ),
    
    # ============== CONTEXT OVERLOAD/NOISE CASES (3) ==============
    
    EdgeCaseQuery(
        id="ERR_05",
        query="List every single player appearance in the 2022-23 season with all their stats.",
        category="scale",
        expected_error_type="context_overload",
        expected_issue="Would return thousands of rows - context too large for LLM",
        notes="Tests handling of overly broad queries"
    ),
    
    EdgeCaseQuery(
        id="ERR_06",
        query="Tell me about all the midfielders in the Premier League.",
        category="broad_query",
        expected_error_type="context_overload",
        expected_issue="Too many midfielders - noise overwhelms signal",
        notes="Tests if LLM can focus on relevant subset"
    ),
    
    EdgeCaseQuery(
        id="ERR_07",
        query="Who is the best player?",
        category="ambiguous",
        expected_error_type="context_overload",
        expected_issue="No metric defined - embedding returns noisy mix of players",
        notes="Tests handling of undefined criteria"
    ),
    
    # ============== LLM HALLUCINATION CASES (4) ==============
    
    EdgeCaseQuery(
        id="ERR_08",
        query="Who will be the top scorer in the 2024-25 season?",
        category="future_prediction",
        expected_error_type="hallucination",
        expected_issue="Future data not available - LLM may hallucinate prediction",
        notes="Tests if LLM admits knowledge cutoff"
    ),
    
    EdgeCaseQuery(
        id="ERR_09",
        query="How many Champions League goals did Liverpool score?",
        category="out_of_scope",
        expected_error_type="hallucination",
        expected_issue="Only PL FPL data - LLM may hallucinate CL stats",
        notes="Tests if LLM stays within KG scope"
    ),
    
    EdgeCaseQuery(
        id="ERR_10",
        query="What is Harry Kane's nationality and birth date?",
        category="missing_attributes",
        expected_error_type="hallucination",
        expected_issue="Personal info not in FPL KG - LLM may use parametric knowledge",
        notes="Tests if LLM distinguishes context vs world knowledge"
    ),
    
    EdgeCaseQuery(
        id="ERR_11",
        query="Who scored more, Kane or the Egyptian?",
        category="indirect_reference",
        expected_error_type="hallucination",
        expected_issue="'The Egyptian' = Salah but may be misresolved or hallucinated",
        notes="Tests handling of indirect player references"
    ),
    
    # ============== POOR SYNTHESIS CASES (4) ==============
    
    EdgeCaseQuery(
        id="ERR_12",
        query="Which player improved the most between 2021-22 and 2022-23 in terms of total points?",
        category="complex_reasoning",
        expected_error_type="poor_synthesis",
        expected_issue="Requires cross-season delta calculation - synthesis challenge",
        notes="Tests multi-step reasoning with correct data"
    ),
    
    EdgeCaseQuery(
        id="ERR_13",
        query="If I picked Haaland, Salah, and De Bruyne for my FPL team in 2022-23, what would be my combined points?",
        category="hypothetical",
        expected_error_type="poor_synthesis",
        expected_issue="Requires summing 3 players' points - arithmetic synthesis",
        notes="Tests numerical aggregation from multiple results"
    ),
    
    EdgeCaseQuery(
        id="ERR_14",
        query="Compare the total goals scored by defenders vs midfielders in the 2022-23 season.",
        category="group_comparison",
        expected_error_type="poor_synthesis",
        expected_issue="Requires grouping by position and summing - complex synthesis",
        notes="Tests position-level aggregation and comparison"
    ),
    
    EdgeCaseQuery(
        id="ERR_15",
        query="What is the average points per game for forwards who scored at least 10 goals in 2022-23?",
        category="conditional_aggregation",
        expected_error_type="poor_synthesis",
        expected_issue="Multi-step: filter by goals threshold, then calculate PPG average",
        notes="Tests conditional filtering + statistical calculation"
    ),
]


@dataclass
class QueryResult:
    """Result for a single query with context and all LLM responses."""
    query_id: str
    query_text: str
    category: str
    expected_error_type: str
    expected_issue: str
    
    # The data that was sent to all LLMs
    context_data: Dict[str, Any]
    prompt_sent: str
    
    # Responses from each LLM
    llm_responses: Dict[str, Dict[str, Any]]


def analyze_retrieval_quality(context_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the quality of retrieval results."""
    
    baseline_results = context_data.get("baseline_results", [])
    embedding_results = context_data.get("embedding_results", [])
    
    analysis = {
        "baseline_count": len(baseline_results),
        "embedding_count": len(embedding_results),
        "baseline_is_fallback": context_data.get("baseline_is_fallback", False),
        "has_baseline_data": len(baseline_results) > 0,
        "has_embedding_data": len(embedding_results) > 0,
        "retrieval_status": "success"
    }
    
    # Determine retrieval status
    if len(baseline_results) == 0 and len(embedding_results) == 0:
        analysis["retrieval_status"] = "total_failure"
    elif len(baseline_results) == 0:
        analysis["retrieval_status"] = "baseline_failure"
    elif analysis["baseline_is_fallback"]:
        analysis["retrieval_status"] = "fallback_used"
    
    return analysis


def run_single_edge_query(
    query: EdgeCaseQuery,
    adapters: List,
    config: Dict[str, str],
    verbose: bool = True
) -> QueryResult:
    """Run a single edge case query through pipeline."""
    
    print(f"\n{'='*70}")
    print(f"[{query.id}] {query.query}")
    print(f"Category: {query.category} | Expected Error: {query.expected_error_type}")
    print('='*70)
    
    # Step 1: Get context from pipeline
    ctx = build_retrieval_context(query.query, config=config)
    
    # Step 2: Build the prompt
    prompt = build_structured_prompt(ctx)
    
    # Step 3: Collect context data
    context_data = {
        "user_query": ctx.user_query,
        "intent": ctx.intent,
        "entities": ctx.entities,
        "baseline_description": ctx.baseline_desc,
        "baseline_is_fallback": ctx.baseline_is_fallback,
        "baseline_results": ctx.baseline_results[:30],
        "embedding_results": ctx.embedding_results[:30],
    }
    
    if verbose:
        print(f"\n📦 CONTEXT DATA:")
        print(f"   Intent: {ctx.intent}")
        print(f"   Entities: {ctx.entities}")
        print(f"   Baseline results: {len(ctx.baseline_results)} rows")
        print(f"   Embedding results: {len(ctx.embedding_results)} rows")
        print(f"   Baseline fallback: {ctx.baseline_is_fallback}")
    
    # Step 4: Query all LLMs
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
            
            print(f"⏱️  Time: {elapsed:.2f}s | Tokens: {metrics.input_tokens}/{metrics.output_tokens}")
            display_text = response_text[:500] if response_text else "No response"
            if len(response_text) > 500:
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
        query_id=query.id,
        query_text=query.query,
        category=query.category,
        expected_error_type=query.expected_error_type,
        expected_issue=query.expected_issue,
        context_data=context_data,
        prompt_sent=prompt,
        llm_responses=llm_responses
    )


def run_all_edge_queries(adapters: List, config: Dict[str, str]) -> List[QueryResult]:
    """Run all 15 edge case queries."""
    results = []
    for q in EDGE_CASE_QUERIES_15:
        result = run_single_edge_query(q, adapters, config, verbose=True)
        results.append(result)
        
        # Save individual result
        save_single_result(result, f"edge_{result.query_id}_result.txt")
        
    return results


def save_single_result(result: QueryResult, filename: str):
    """Save a single result to file."""
    output = {
        "query_id": result.query_id,
        "query_text": result.query_text,
        "category": result.category,
        "expected_error_type": result.expected_error_type,
        "expected_issue": result.expected_issue,
        "context_data": result.context_data,
        "llm_responses": result.llm_responses
    }
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def generate_error_analysis_report(results: List[QueryResult]) -> str:
    """Generate comprehensive error analysis report."""
    
    report = []
    report.append("=" * 80)
    report.append("FPL RAG SYSTEM - ERROR ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total Edge Case Queries Tested: {len(results)}")
    report.append("")
    
    # ========== ERROR TYPE DEFINITIONS ==========
    report.append("\n" + "=" * 80)
    report.append("ERROR TYPE DEFINITIONS")
    report.append("=" * 80)
    report.append("""
1. RETRIEVAL FAILURE
   - Definition: The Knowledge Graph query returns no nodes, wrong nodes, or 
     falls back to a generic query that doesn't address the user's question.
   - Symptoms: Empty baseline results, mismatched entities, fallback query used
   - Root causes: Entity resolution failure, missing data in KG, unsupported query type
   
2. CONTEXT OVERLOAD/NOISE
   - Definition: Too much irrelevant context is provided to the LLM, diluting 
     the signal and making it hard to extract the relevant answer.
   - Symptoms: Large result sets, low relevance scores, mixed/unrelated players
   - Root causes: Overly broad queries, ambiguous criteria, no filtering

3. LLM HALLUCINATION
   - Definition: The LLM's answer contains information NOT supported by the 
     retrieved context, invented from its parametric knowledge.
   - Symptoms: Facts not in context, confident wrong answers, out-of-scope claims
   - Root causes: LLM ignoring grounding instructions, filling gaps with training data

4. POOR SYNTHESIS
   - Definition: The retrieved facts are correct, but the LLM fails to properly 
     combine, calculate, or reason over them to produce a coherent answer.
   - Symptoms: Wrong calculations, incomplete comparisons, missed logical connections
   - Root causes: Complex multi-step reasoning, numerical aggregation errors
""")
    
    # ========== CATEGORIZED RESULTS ==========
    error_categories = {
        "retrieval_failure": [],
        "context_overload": [],
        "hallucination": [],
        "poor_synthesis": []
    }
    
    for r in results:
        error_categories[r.expected_error_type].append(r)
    
    # ========== RETRIEVAL FAILURE ANALYSIS ==========
    report.append("\n" + "=" * 80)
    report.append("1. RETRIEVAL FAILURE CASES")
    report.append("=" * 80)
    
    for r in error_categories["retrieval_failure"]:
        report.append(f"\n--- [{r.query_id}] {r.category.upper()} ---")
        report.append(f"Query: \"{r.query_text}\"")
        report.append(f"Expected Issue: {r.expected_issue}")
        report.append(f"\nRetrieval Analysis:")
        
        retrieval = analyze_retrieval_quality(r.context_data)
        report.append(f"  - Baseline results: {retrieval['baseline_count']} rows")
        report.append(f"  - Embedding results: {retrieval['embedding_count']} rows")
        report.append(f"  - Fallback used: {retrieval['baseline_is_fallback']}")
        report.append(f"  - Status: {retrieval['retrieval_status']}")
        report.append(f"  - Detected intent: {r.context_data.get('intent', 'N/A')}")
        report.append(f"  - Detected entities: {r.context_data.get('entities', {})}")
        
        report.append(f"\nLLM Responses:")
        for model, resp in r.llm_responses.items():
            if resp["response"]:
                preview = resp["response"][:300].replace("\n", " ")
                if len(resp["response"]) > 300:
                    preview += "..."
                report.append(f"  [{model}]: {preview}")
            else:
                report.append(f"  [{model}]: ERROR - {resp['error']}")
        
        # Observed error analysis
        report.append(f"\nObserved Behavior:")
        if retrieval['retrieval_status'] == 'total_failure':
            report.append("  ❌ CONFIRMED RETRIEVAL FAILURE - No data retrieved")
        elif retrieval['retrieval_status'] == 'baseline_failure':
            report.append("  ⚠️ PARTIAL FAILURE - Only embedding results available")
        elif retrieval['retrieval_status'] == 'fallback_used':
            report.append("  ⚠️ FALLBACK QUERY - Generic results, may not address query")
        else:
            report.append("  ✓ Data retrieved - check if relevant to query")
    
    # ========== CONTEXT OVERLOAD ANALYSIS ==========
    report.append("\n" + "=" * 80)
    report.append("2. CONTEXT OVERLOAD/NOISE CASES")
    report.append("=" * 80)
    
    for r in error_categories["context_overload"]:
        report.append(f"\n--- [{r.query_id}] {r.category.upper()} ---")
        report.append(f"Query: \"{r.query_text}\"")
        report.append(f"Expected Issue: {r.expected_issue}")
        report.append(f"\nContext Analysis:")
        
        baseline = r.context_data.get("baseline_results", [])
        embedding = r.context_data.get("embedding_results", [])
        report.append(f"  - Baseline results: {len(baseline)} rows")
        report.append(f"  - Embedding results: {len(embedding)} rows")
        report.append(f"  - Total context size: {len(baseline) + len(embedding)} rows")
        
        # Check for diversity/noise in results
        if embedding:
            positions = set(e.get("position", "?") for e in embedding)
            report.append(f"  - Position diversity in embeddings: {positions}")
            if len(positions) > 2:
                report.append("  ⚠️ High position diversity suggests noisy results")
        
        report.append(f"\nLLM Responses:")
        for model, resp in r.llm_responses.items():
            if resp["response"]:
                preview = resp["response"][:300].replace("\n", " ")
                if len(resp["response"]) > 300:
                    preview += "..."
                report.append(f"  [{model}]: {preview}")
            else:
                report.append(f"  [{model}]: ERROR - {resp['error']}")
        
        report.append(f"\nObserved Behavior:")
        total_context = len(baseline) + len(embedding)
        if total_context > 50:
            report.append("  ❌ CONTEXT OVERLOAD - Too many results for focused answer")
        elif total_context > 20:
            report.append("  ⚠️ MODERATE NOISE - Results may contain irrelevant data")
        else:
            report.append("  ✓ Context size manageable")
    
    # ========== HALLUCINATION ANALYSIS ==========
    report.append("\n" + "=" * 80)
    report.append("3. LLM HALLUCINATION CASES")
    report.append("=" * 80)
    
    for r in error_categories["hallucination"]:
        report.append(f"\n--- [{r.query_id}] {r.category.upper()} ---")
        report.append(f"Query: \"{r.query_text}\"")
        report.append(f"Expected Issue: {r.expected_issue}")
        report.append(f"\nContext Provided:")
        
        baseline = r.context_data.get("baseline_results", [])
        embedding = r.context_data.get("embedding_results", [])
        report.append(f"  - Baseline results: {len(baseline)} rows")
        report.append(f"  - Embedding results: {len(embedding)} rows")
        
        # Show what data WAS available
        if baseline:
            sample = baseline[:3]
            report.append(f"  - Sample baseline data: {sample}")
        if embedding:
            sample = embedding[:3]
            report.append(f"  - Sample embedding data: {sample}")
        
        report.append(f"\nLLM Responses (CHECK FOR HALLUCINATION):")
        for model, resp in r.llm_responses.items():
            if resp["response"]:
                # Full response for hallucination analysis
                report.append(f"\n  [{model}]:")
                report.append(f"  {resp['response'][:500]}")
                if len(resp["response"]) > 500:
                    report.append("  [... truncated ...]")
            else:
                report.append(f"  [{model}]: ERROR - {resp['error']}")
        
        report.append(f"\nHallucination Check:")
        report.append("  Manual verification needed: Does response contain facts NOT in context?")
    
    # ========== POOR SYNTHESIS ANALYSIS ==========
    report.append("\n" + "=" * 80)
    report.append("4. POOR SYNTHESIS CASES")
    report.append("=" * 80)
    
    for r in error_categories["poor_synthesis"]:
        report.append(f"\n--- [{r.query_id}] {r.category.upper()} ---")
        report.append(f"Query: \"{r.query_text}\"")
        report.append(f"Expected Issue: {r.expected_issue}")
        report.append(f"\nContext Provided:")
        
        baseline = r.context_data.get("baseline_results", [])
        embedding = r.context_data.get("embedding_results", [])
        report.append(f"  - Baseline results: {len(baseline)} rows")
        report.append(f"  - Embedding results: {len(embedding)} rows")
        
        # Show relevant data for synthesis
        if baseline:
            report.append(f"  - Baseline data (first 5):")
            for row in baseline[:5]:
                report.append(f"    {row}")
        
        report.append(f"\nLLM Responses (CHECK SYNTHESIS QUALITY):")
        for model, resp in r.llm_responses.items():
            if resp["response"]:
                report.append(f"\n  [{model}]:")
                report.append(f"  {resp['response'][:600]}")
                if len(resp["response"]) > 600:
                    report.append("  [... truncated ...]")
            else:
                report.append(f"  [{model}]: ERROR - {resp['error']}")
        
        report.append(f"\nSynthesis Quality Check:")
        report.append("  - Did LLM correctly combine/calculate from the data?")
        report.append("  - Did LLM miss any relevant facts?")
        report.append("  - Is the reasoning chain sound?")
    
    # ========== SUMMARY STATISTICS ==========
    report.append("\n" + "=" * 80)
    report.append("SUMMARY STATISTICS")
    report.append("=" * 80)
    
    report.append(f"\nQueries by Expected Error Type:")
    for error_type, queries in error_categories.items():
        report.append(f"  - {error_type}: {len(queries)} queries")
    
    # Count actual retrieval issues
    retrieval_issues = 0
    for r in results:
        retrieval = analyze_retrieval_quality(r.context_data)
        if retrieval['retrieval_status'] != 'success':
            retrieval_issues += 1
    
    report.append(f"\nActual Retrieval Issues Detected: {retrieval_issues}/{len(results)}")
    
    # Response time stats
    all_times = []
    for r in results:
        for model, resp in r.llm_responses.items():
            if resp["response_time_sec"]:
                all_times.append((model, resp["response_time_sec"]))
    
    if all_times:
        report.append(f"\nAverage Response Times:")
        models = set(t[0] for t in all_times)
        for model in models:
            times = [t[1] for t in all_times if t[0] == model]
            avg = sum(times) / len(times)
            report.append(f"  - {model}: {avg:.2f}s")
    
    # ========== RECOMMENDATIONS ==========
    report.append("\n" + "=" * 80)
    report.append("RECOMMENDATIONS FOR IMPROVEMENT")
    report.append("=" * 80)
    report.append("""
1. RETRIEVAL FAILURE MITIGATION:
   - Implement fuzzy matching for player names (Levenshtein distance)
   - Add nickname/alias lookup table (CR7 → Cristiano Ronaldo)
   - Return explicit "data not found" signal instead of empty results
   - Add more robust entity extraction in preprocessing

2. CONTEXT OVERLOAD MITIGATION:
   - Implement result ranking and truncation
   - Add query decomposition for broad queries
   - Use semantic filtering to remove low-relevance results
   - Prompt user for clarification on ambiguous queries

3. HALLUCINATION MITIGATION:
   - Strengthen grounding instructions in prompt
   - Add explicit "only use provided context" constraints
   - Implement post-generation fact verification
   - Use lower temperature settings for factual queries

4. POOR SYNTHESIS MITIGATION:
   - Add chain-of-thought prompting for complex reasoning
   - Pre-compute aggregations in the KG layer
   - Provide calculation examples in the prompt
   - Break complex queries into simpler sub-queries
""")
    
    return "\n".join(report)


def main():
    """Run the full edge case analysis."""
    
    print("=" * 70)
    print("FPL RAG SYSTEM - ERROR ANALYSIS TEST SUITE")
    print("=" * 70)
    
    # Load config and build adapters
    config = load_config()
    adapters = build_default_model_adapters(config)
    
    if not adapters:
        print("❌ No LLM adapters available. Check your config.txt for API keys.")
        return
    
    print(f"\n✅ Loaded {len(adapters)} LLM adapters")
    for a in adapters:
        print(f"   - {a.name}")
    
    print(f"\n🔬 Running {len(EDGE_CASE_QUERIES_15)} edge case queries...")
    print("   This will test 4 error categories:")
    print("   - Retrieval Failure (4 queries)")
    print("   - Context Overload (3 queries)")
    print("   - Hallucination (4 queries)")
    print("   - Poor Synthesis (4 queries)")
    
    # Run all queries
    results = run_all_edge_queries(adapters, config)
    
    # Generate analysis report
    print("\n📊 Generating error analysis report...")
    report = generate_error_analysis_report(results)
    
    # Save report
    report_file = "ERROR_ANALYSIS_REPORT.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n✅ Error analysis report saved to: {report_file}")
    
    # Also save raw results as JSON
    json_output = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": len(results),
        "results": []
    }
    
    for r in results:
        json_output["results"].append({
            "query_id": r.query_id,
            "query_text": r.query_text,
            "category": r.category,
            "expected_error_type": r.expected_error_type,
            "expected_issue": r.expected_issue,
            "context_data": r.context_data,
            "llm_responses": {
                model: {
                    "response": resp["response"],
                    "response_time_sec": resp["response_time_sec"],
                    "error": resp["error"]
                }
                for model, resp in r.llm_responses.items()
            }
        })
    
    with open("edge_case_results.json", "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Raw results saved to: edge_case_results.json")
    
    # Print summary
    print("\n" + "=" * 70)
    print("QUICK SUMMARY")
    print("=" * 70)
    
    error_counts = {
        "retrieval_failure": 0,
        "context_overload": 0,
        "hallucination": 0,
        "poor_synthesis": 0
    }
    
    for r in results:
        error_counts[r.expected_error_type] += 1
        retrieval = analyze_retrieval_quality(r.context_data)
        if retrieval['retrieval_status'] != 'success':
            print(f"  ⚠️ [{r.query_id}] Retrieval issue: {retrieval['retrieval_status']}")
    
    print(f"\nQueries by error category:")
    for cat, count in error_counts.items():
        print(f"  - {cat}: {count}")


if __name__ == "__main__":
    main()
