"""
FPL Graph-RAG Baseline vs Ground Truth Comparison Script
=========================================================
This script compares results from the baseline.py pipeline with the ground truth
queries from truth.txt to evaluate the accuracy of the NLP preprocessing + query building.
"""

import json
import re
import sys
import os
# Add project root to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from datetime import datetime
from neo4j import GraphDatabase
from typing import Dict, List, Any, Set, Tuple
import Main.preprocessing as preprocessing
import Main.baseline as baseline

# Get project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def load_config() -> Dict[str, str]:
    """Load Neo4j credentials from config.txt in project root"""
    config = {}
    config_path = os.path.join(PROJECT_ROOT, "config.txt")
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and "=" in line:
                key, value = line.split("=", 1)
                config[key] = value
    return config


def parse_truth_file(filepath=None) -> List[Dict]:
    """
    Parse the truth.txt file to extract questions and their corresponding Cypher queries.
    """
    if filepath is None:
        filepath = os.path.join(PROJECT_ROOT, "tests", "truth.txt")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    pattern = r'Q(\d+)\.\s*(.+?)\n-+\n(MATCH[\s\S]+?;)'
    matches = re.findall(pattern, content)
    
    questions = []
    for match in matches:
        q_id = int(match[0])
        q_text = match[1].strip()
        cypher_query = match[2].strip()
        
        questions.append({
            "question_id": f"Q{q_id}",
            "question_text": q_text,
            "cypher_query": cypher_query
        })
    
    return questions


def execute_ground_truth_query(driver, query: str) -> List[Dict]:
    """Execute the ground truth Cypher query directly against Neo4j."""
    try:
        with driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]
    except Exception as e:
        return [{"error": str(e)}]


def execute_baseline_pipeline(question_text: str) -> Tuple[List[Dict], Dict]:
    """
    Execute the baseline pipeline: preprocessing -> query execution.
    Returns (results, preprocessing_output)
    """
    try:
        # Step 1: Preprocess the query
        preprocessing_output = preprocessing.process_user_query(question_text)
        
        # Step 2: Execute through baseline
        results = baseline.execute_baseline_query(preprocessing_output)
        
        return results, preprocessing_output
    except Exception as e:
        return [{"error": str(e)}], {"error": str(e)}


def normalize_result(result: Dict) -> Dict:
    """Normalize a result dict for comparison (lowercase keys, handle types)."""
    normalized = {}
    for key, value in result.items():
        # Normalize key
        norm_key = key.lower().replace("_", "").replace(" ", "")
        
        # Normalize value
        if isinstance(value, str):
            normalized[norm_key] = value.lower().strip()
        elif isinstance(value, (int, float)):
            normalized[norm_key] = round(float(value), 2)
        else:
            normalized[norm_key] = str(value).lower().strip()
    
    return normalized


def extract_player_names(results: List[Dict]) -> Set[str]:
    """Extract player names from results for comparison."""
    names = set()
    for r in results:
        for key, value in r.items():
            if 'player' in key.lower() or 'name' in key.lower():
                if isinstance(value, str):
                    names.add(value.lower().strip())
    return names


def extract_key_values(results: List[Dict], key_hints: List[str] = None) -> List[Any]:
    """Extract key values from results for comparison."""
    if key_hints is None:
        key_hints = ['player', 'name', 'team', 'points', 'goals', 'assists']
    
    values = []
    for r in results:
        row_values = {}
        for key, value in r.items():
            for hint in key_hints:
                if hint in key.lower():
                    row_values[hint] = value
                    break
        if row_values:
            values.append(row_values)
    return values


def compare_results(ground_truth: List[Dict], baseline_results: List[Dict]) -> Dict:
    """
    Compare ground truth results with baseline pipeline results.
    Returns comparison metrics.
    """
    comparison = {
        "ground_truth_count": len(ground_truth),
        "baseline_count": len(baseline_results),
        "counts_match": len(ground_truth) == len(baseline_results),
        "player_overlap": 0.0,
        "exact_match": False,
        "partial_match": False,
        "match_details": []
    }
    
    # Check for errors
    if ground_truth and "error" in ground_truth[0]:
        comparison["ground_truth_error"] = ground_truth[0]["error"]
        return comparison
    
    if baseline_results and "error" in baseline_results[0]:
        comparison["baseline_error"] = baseline_results[0]["error"]
        return comparison
    
    # Extract player names for overlap comparison
    gt_players = extract_player_names(ground_truth)
    bl_players = extract_player_names(baseline_results)
    
    if gt_players and bl_players:
        overlap = gt_players.intersection(bl_players)
        union = gt_players.union(bl_players)
        comparison["player_overlap"] = len(overlap) / len(union) if union else 0.0
        comparison["gt_players"] = list(gt_players)[:10]  # First 10 for brevity
        comparison["bl_players"] = list(bl_players)[:10]
        comparison["common_players"] = list(overlap)[:10]
    
    # Check for exact match (same players in same order with same values)
    if len(ground_truth) == len(baseline_results) and len(ground_truth) > 0:
        gt_normalized = [normalize_result(r) for r in ground_truth]
        bl_normalized = [normalize_result(r) for r in baseline_results]
        
        # Check if all players match
        gt_player_list = [r.get('playername', r.get('player', '')) for r in gt_normalized]
        bl_player_list = [r.get('playername', r.get('player', '')) for r in bl_normalized]
        
        if gt_player_list == bl_player_list:
            comparison["exact_match"] = True
        elif set(gt_player_list) == set(bl_player_list):
            comparison["partial_match"] = True  # Same players, different order
    
    # Partial match if significant overlap
    if comparison["player_overlap"] >= 0.8:
        comparison["partial_match"] = True
    
    return comparison


def serialize_for_json(obj):
    """Custom JSON serializer."""
    if isinstance(obj, set):
        return list(obj)
    elif hasattr(obj, '__dict__'):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def run_comparison(truth_file="truth.txt", output_file="comparison_results.json"):
    """
    Main function to run comparison between ground truth and baseline pipeline.
    """
    # Load configuration
    config = load_config()
    uri = config.get("URI", "neo4j://127.0.0.1:7687")
    username = config.get("USERNAME", "neo4j")
    password = config.get("PASSWORD", "")
    
    # Connect to Neo4j for ground truth queries
    print(f"Connecting to Neo4j at {uri}...")
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    try:
        driver.verify_connectivity()
        print("✔ Connected to Neo4j successfully!")
    except Exception as e:
        print(f"✘ Failed to connect to Neo4j: {e}")
        return
    
    # Parse truth file
    print(f"\nParsing {truth_file}...")
    questions = parse_truth_file(truth_file)
    print(f"✔ Found {len(questions)} questions/queries")
    
    # Results container
    comparison_results = {
        "metadata": {
            "comparison_timestamp": datetime.now().isoformat(),
            "truth_file": truth_file,
            "total_questions": len(questions),
            "exact_matches": 0,
            "partial_matches": 0,
            "failures": 0,
            "neo4j_uri": uri
        },
        "comparisons": []
    }
    
    # Process each question
    print("\n" + "=" * 70)
    print("RUNNING COMPARISONS")
    print("=" * 70)
    
    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {q['question_id']}: {q['question_text'][:60]}...")
        
        # 1. Execute ground truth query
        print("  → Running ground truth query...")
        gt_results = execute_ground_truth_query(driver, q['cypher_query'])
        
        # 2. Execute baseline pipeline
        print("  → Running baseline pipeline...")
        bl_results, preprocessing_output = execute_baseline_pipeline(q['question_text'])
        
        # 3. Compare results
        comparison = compare_results(gt_results, bl_results)
        
        # Build comparison record
        comparison_record = {
            "question_id": q['question_id'],
            "question_text": q['question_text'],
            "ground_truth_query": q['cypher_query'],
            "preprocessing_output": {
                "intent": preprocessing_output.get("intent", "ERROR"),
                "entities": preprocessing_output.get("entities", {}),
                "ranking": preprocessing_output.get("ranking"),
                "threshold": preprocessing_output.get("threshold")
            } if isinstance(preprocessing_output, dict) and "error" not in preprocessing_output else {"error": str(preprocessing_output)},
            "ground_truth_results": gt_results[:5],  # First 5 for brevity
            "ground_truth_count": len(gt_results),
            "baseline_results": bl_results[:5] if bl_results else [],
            "baseline_count": len(bl_results) if bl_results else 0,
            "comparison": comparison,
            "successful": comparison.get("exact_match", False) or comparison.get("partial_match", False)
        }
        
        comparison_results["comparisons"].append(comparison_record)
        
        # Update counters
        if comparison.get("exact_match"):
            comparison_results["metadata"]["exact_matches"] += 1
            status = "✔ EXACT MATCH"
        elif comparison.get("partial_match"):
            comparison_results["metadata"]["partial_matches"] += 1
            status = "◐ PARTIAL MATCH"
        else:
            comparison_results["metadata"]["failures"] += 1
            status = "✘ MISMATCH"
        
        print(f"  {status} (GT: {len(gt_results)} rows, BL: {len(bl_results) if bl_results else 0} rows, Overlap: {comparison.get('player_overlap', 0):.1%})")
    
    # Close driver
    driver.close()
    
    # Calculate success rate
    total = comparison_results["metadata"]["total_questions"]
    exact = comparison_results["metadata"]["exact_matches"]
    partial = comparison_results["metadata"]["partial_matches"]
    
    comparison_results["metadata"]["exact_match_rate"] = f"{(exact/total)*100:.2f}%" if total > 0 else "0%"
    comparison_results["metadata"]["partial_match_rate"] = f"{(partial/total)*100:.2f}%" if total > 0 else "0%"
    comparison_results["metadata"]["overall_success_rate"] = f"{((exact+partial)/total)*100:.2f}%" if total > 0 else "0%"
    
    # Save results
    print("\n" + "=" * 70)
    print(f"Saving results to {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False, default=serialize_for_json)
    
    print(f"✔ Results saved to {output_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"Total Questions:       {total}")
    print(f"Exact Matches:         {exact} ({comparison_results['metadata']['exact_match_rate']})")
    print(f"Partial Matches:       {partial} ({comparison_results['metadata']['partial_match_rate']})")
    print(f"Failures/Mismatches:   {comparison_results['metadata']['failures']}")
    print(f"Overall Success Rate:  {comparison_results['metadata']['overall_success_rate']}")
    print("=" * 70)
    
    # Print failed queries for analysis
    if comparison_results["metadata"]["failures"] > 0:
        print("\n⚠ FAILED/MISMATCHED QUERIES:")
        print("-" * 70)
        for comp in comparison_results["comparisons"]:
            if not comp["successful"]:
                print(f"\n{comp['question_id']}: {comp['question_text']}")
                print(f"  GT Count: {comp['ground_truth_count']}, BL Count: {comp['baseline_count']}")
                print(f"  Preprocessing: {comp['preprocessing_output'].get('intent', 'N/A')}")
                print(f"  Entities: {comp['preprocessing_output'].get('entities', {})}")
    
    return comparison_results


if __name__ == "__main__":
    import sys
    
    output_file = sys.argv[1] if len(sys.argv) > 1 else "comparison_results.json"
    run_comparison(output_file=output_file)
