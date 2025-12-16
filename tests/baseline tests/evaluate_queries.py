"""
FPL Graph-RAG Query Evaluation Script
=====================================
This script reads queries from truth.txt, executes them against Neo4j,
and saves the results to a JSON file with success/failure status.
"""

import json
import re
import sys
import os
# Add project root to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from datetime import datetime
from neo4j import GraphDatabase


def load_config():
    """Load Neo4j credentials from config.txt in project root"""
    config = {}
    # Look for config.txt in project root (two levels up from this file)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    config_path = os.path.join(project_root, "config.txt")
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and "=" in line:
                key, value = line.split("=", 1)
                config[key] = value
    return config

# Get project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def parse_truth_file(filepath=None):
    """
    Parse the truth.txt file to extract questions and their corresponding Cypher queries.
    Returns a list of dictionaries with question_id, question_text, and cypher_query.
    """
    if filepath is None:
        filepath = os.path.join(PROJECT_ROOT, "tests", "truth.txt")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Pattern to match Q1, Q2, etc. sections
    # Each question starts with Q followed by a number, then a period, then the question text
    # The query follows after the dashed line
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


def execute_query(driver, query):
    """
    Execute a Cypher query and return the results along with metadata.
    Returns a tuple of (results, error_message, nodes_count, relationships_count)
    """
    try:
        with driver.session() as session:
            result = session.run(query)
            records = [dict(record) for record in result]
            
            # Get summary information
            summary = result.consume()
            
            # Count nodes and relationships in the results
            nodes_fetched = 0
            relationships_fetched = 0
            
            # Try to get counters from summary if available
            if hasattr(summary, 'counters'):
                counters = summary.counters
                nodes_fetched = getattr(counters, 'nodes_created', 0) + getattr(counters, 'nodes_deleted', 0)
                relationships_fetched = getattr(counters, 'relationships_created', 0) + getattr(counters, 'relationships_deleted', 0)
            
            return records, None, len(records), summary
            
    except Exception as e:
        return None, str(e), 0, None


def serialize_result(obj):
    """Custom JSON serializer for Neo4j result objects"""
    if hasattr(obj, '__dict__'):
        return str(obj)
    elif isinstance(obj, (datetime,)):
        return obj.isoformat()
    else:
        return str(obj)


def run_evaluation(truth_file="truth.txt", output_file="evaluation_results.json"):
    """
    Main function to run all queries from truth.txt and save results to JSON.
    """
    # Load configuration
    config = load_config()
    uri = config.get("URI", "neo4j://127.0.0.1:7687")
    username = config.get("USERNAME", "neo4j")
    password = config.get("PASSWORD", "")
    
    # Connect to Neo4j
    print(f"Connecting to Neo4j at {uri}...")
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    # Test connection
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
    evaluation_results = {
        "metadata": {
            "evaluation_timestamp": datetime.now().isoformat(),
            "truth_file": truth_file,
            "total_queries": len(questions),
            "successful_queries": 0,
            "failed_queries": 0,
            "neo4j_uri": uri
        },
        "queries": []
    }
    
    # Execute each query
    print("\nExecuting queries...")
    print("-" * 60)
    
    for i, q in enumerate(questions, 1):
        print(f"Running {q['question_id']}: {q['question_text'][:50]}...")
        
        # Execute the query
        results, error, record_count, summary = execute_query(driver, q['cypher_query'])
        
        # Determine success
        is_successful = error is None
        
        # Build query result object
        query_result = {
            "question_id": q['question_id'],
            "question_text": q['question_text'],
            "expected_query": q['cypher_query'],
            "executed_query": q['cypher_query'],  # Same as expected in this evaluation
            "successful": is_successful,
            "error_message": error,
            "records_returned": record_count,
            "results": []
        }
        
        if is_successful and results:
            # Convert results to serializable format
            serialized_results = []
            for record in results:
                serialized_record = {}
                for key, value in record.items():
                    try:
                        # Try direct JSON serialization
                        json.dumps(value)
                        serialized_record[key] = value
                    except (TypeError, ValueError):
                        # Fall back to string representation
                        serialized_record[key] = serialize_result(value)
                serialized_results.append(serialized_record)
            
            query_result["results"] = serialized_results
            evaluation_results["metadata"]["successful_queries"] += 1
            print(f"  ✔ Success - {record_count} records returned")
        elif is_successful:
            evaluation_results["metadata"]["successful_queries"] += 1
            print(f"  ✔ Success - 0 records returned")
        else:
            evaluation_results["metadata"]["failed_queries"] += 1
            print(f"  ✘ Failed - {error}")
        
        evaluation_results["queries"].append(query_result)
    
    # Close driver
    driver.close()
    
    # Calculate success rate
    total = evaluation_results["metadata"]["total_queries"]
    successful = evaluation_results["metadata"]["successful_queries"]
    evaluation_results["metadata"]["success_rate"] = f"{(successful/total)*100:.2f}%" if total > 0 else "0%"
    
    # Save results to JSON
    print("\n" + "-" * 60)
    print(f"Saving results to {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False, default=serialize_result)
    
    print(f"✔ Results saved to {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Queries:      {total}")
    print(f"Successful:         {successful}")
    print(f"Failed:             {evaluation_results['metadata']['failed_queries']}")
    print(f"Success Rate:       {evaluation_results['metadata']['success_rate']}")
    print("=" * 60)
    
    return evaluation_results


def print_detailed_results(output_file="evaluation_results.json"):
    """
    Print a detailed summary of the evaluation results from the JSON file.
    """
    with open(output_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    print("\n" + "=" * 80)
    print("DETAILED QUERY RESULTS")
    print("=" * 80)
    
    for query in results["queries"]:
        status = "✔ SUCCESS" if query["successful"] else "✘ FAILED"
        print(f"\n{query['question_id']}: {query['question_text']}")
        print(f"Status: {status}")
        print(f"Records: {query['records_returned']}")
        
        if not query["successful"]:
            print(f"Error: {query['error_message']}")
        elif query["results"] and len(query["results"]) > 0:
            print("Sample results (first 3):")
            for j, record in enumerate(query["results"][:3]):
                print(f"  {j+1}. {record}")
        print("-" * 80)


if __name__ == "__main__":
    import sys
    
    # Allow custom output file name via command line
    output_file = sys.argv[1] if len(sys.argv) > 1 else "evaluation_results.json"
    
    # Run evaluation
    run_evaluation(output_file=output_file)
    
    # Optionally print detailed results
    print("\nWould you like to see detailed results? Check the JSON file or run:")
    print(f"  python evaluate_queries.py --details")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--details":
        print_detailed_results()
