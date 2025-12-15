"""
embedding_evaluation.py

Evaluation script to compare MiniLM and BGE-M3 embedding models.
Uses queries that match the exact vocabulary from embedding descriptions.

This script:
1. Generates ground truth from Neo4j database
2. Runs semantic search with both embedding models
3. Calculates evaluation metrics (Precision@K, Recall@K, MRR, NDCG)
4. Outputs a detailed comparison report
"""

import json
import logging
from typing import Dict, List, Any, Set
from datetime import datetime
from neo4j import GraphDatabase
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================
# 1. QUERY DEFINITIONS (matching embedding vocabulary)
# ============================================================

# These queries use EXACT phrases from the embedding descriptions
EVALUATION_QUERIES = [
    {
        "query": "elite premium top-tier season-defining asset FWD",
        "description": "Top forwards by total points (elite tier p90+)",
        "position_filter": "FWD",
        "cypher": """
            MATCH (p:Player)-[:PLAYS_AS]->(pos:Position {name: 'FWD'})
            OPTIONAL MATCH (p)-[r:PLAYED_IN]->(:Fixture)
            WITH p, pos, SUM(r.total_points) AS total_points
            ORDER BY total_points DESC
            LIMIT 10
            RETURN p.player_name AS name
        """
    },
    {
        "query": "elite goalkeeper with elite clean sheet potential and top-tier defensive returns",
        "description": "Goalkeepers with most clean sheets (elite tier)",
        "position_filter": "GK",
        "cypher": """
            MATCH (p:Player)-[:PLAYS_AS]->(pos:Position {name: 'GK'})
            OPTIONAL MATCH (p)-[r:PLAYED_IN]->(:Fixture)
            WITH p, pos, SUM(r.clean_sheets) AS clean_sheets
            ORDER BY clean_sheets DESC
            LIMIT 10
            RETURN p.player_name AS name
        """
    },
    {
        "query": "elite defensive asset with top-tier clean sheets defender",
        "description": "Defenders with most clean sheets (elite tier)",
        "position_filter": "DEF",
        "cypher": """
            MATCH (p:Player)-[:PLAYS_AS]->(pos:Position {name: 'DEF'})
            OPTIONAL MATCH (p)-[r:PLAYED_IN]->(:Fixture)
            WITH p, pos, SUM(r.clean_sheets) AS clean_sheets
            ORDER BY clean_sheets DESC
            LIMIT 10
            RETURN p.player_name AS name
        """
    },
    {
        "query": "goal-scoring midfielder and creative playmaker with elite goals",
        "description": "Midfielders with most goals (elite tier)",
        "position_filter": "MID",
        "cypher": """
            MATCH (p:Player)-[:PLAYS_AS]->(pos:Position {name: 'MID'})
            OPTIONAL MATCH (p)-[r:PLAYED_IN]->(:Fixture)
            WITH p, pos, SUM(r.goals_scored) AS goals
            ORDER BY goals DESC
            LIMIT 10
            RETURN p.player_name AS name
        """
    },
    {
        "query": "rarely plays very low minutes rotation or bench player",
        "description": "Players with minimal minutes (low tier)",
        "position_filter": None,
        "cypher": """
            MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
            OPTIONAL MATCH (p)-[r:PLAYED_IN]->(:Fixture)
            WITH p, pos, SUM(r.minutes) AS minutes
            WHERE minutes > 0 AND minutes < 100
            ORDER BY minutes ASC
            LIMIT 15
            RETURN p.player_name AS name
        """
    }
]


# ============================================================
# 2. GROUND TRUTH GENERATOR
# ============================================================

def load_config():
    """Load Neo4j configuration from config.txt"""
    config = {}
    with open("config.txt", "r") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip()
    return config


def generate_ground_truth(config: Dict[str, str]) -> Dict[str, Dict]:
    """Generate ground truth from Neo4j database."""
    driver = GraphDatabase.driver(
        config["URI"],
        auth=(config["USERNAME"], config["PASSWORD"])
    )
    
    ground_truth = {}
    
    with driver.session() as session:
        for q in EVALUATION_QUERIES:
            result = session.run(q["cypher"])
            players = [record["name"] for record in result]
            
            ground_truth[q["query"]] = {
                "description": q["description"],
                "position_filter": q["position_filter"],
                "relevant_players": players,
                "top_5_expected": players[:5]
            }
            
            logger.info(f"Query: '{q['query'][:50]}...' -> {len(players)} ground truth players")
    
    driver.close()
    return ground_truth


# ============================================================
# 3. EVALUATION METRICS
# ============================================================

def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Calculate Precision@K"""
    if not retrieved:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = sum(1 for p in retrieved_k if p in relevant)
    return hits / k


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Calculate Recall@K"""
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = sum(1 for p in retrieved_k if p in relevant)
    return hits / len(relevant)


def mean_reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """Calculate MRR"""
    for i, player in enumerate(retrieved, 1):
        if player in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Calculate NDCG@K"""
    dcg = 0.0
    for i, player in enumerate(retrieved[:k], 1):
        if player in relevant:
            dcg += 1.0 / np.log2(i + 1)
    
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def overlap_at_k(retrieved: List[str], expected_top_k: List[str], k: int) -> float:
    """Calculate overlap with expected top-K"""
    if not expected_top_k:
        return 0.0
    retrieved_set = set(retrieved[:k])
    expected_set = set(expected_top_k[:k])
    return len(retrieved_set & expected_set) / len(expected_set)


# ============================================================
# 4. MODEL EVALUATION
# ============================================================

def evaluate_model(search_system, ground_truth: Dict, model_name: str, k: int = 5) -> Dict:
    """Evaluate a single embedding model against ground truth."""
    results = {}
    
    for query, truth in ground_truth.items():
        logger.info(f"Evaluating '{query[:50]}...' with {model_name}...")
        
        # Run search
        position = truth.get("position_filter")
        search_results = search_system.search(query, top_k=k, position=position)
        
        retrieved = [r["player"] for r in search_results]
        relevant = set(truth["relevant_players"])
        expected = truth["top_5_expected"]
        
        # Calculate metrics
        results[query] = {
            "retrieved": retrieved,
            "precision_at_k": precision_at_k(retrieved, relevant, k),
            "recall_at_k": recall_at_k(retrieved, relevant, k),
            "mrr": mean_reciprocal_rank(retrieved, relevant),
            "ndcg_at_k": ndcg_at_k(retrieved, relevant, k),
            "overlap": overlap_at_k(retrieved, expected, k)
        }
    
    return results


# ============================================================
# 5. REPORT GENERATION
# ============================================================

def generate_report(
    ground_truth: Dict,
    minilm_results: Dict,
    bge_results: Dict,
    k: int = 5
) -> str:
    """Generate a detailed evaluation report."""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("EMBEDDING MODEL EVALUATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Evaluation K: {k} (Top-{k} results)")
    report_lines.append("")
    
    # Detailed results
    report_lines.append("-" * 80)
    report_lines.append("DETAILED RESULTS BY QUERY")
    report_lines.append("-" * 80)
    
    for query in ground_truth:
        truth = ground_truth[query]
        minilm = minilm_results[query]
        bge = bge_results[query]
        
        report_lines.append(f"\n📋 Query: '{query}'")
        report_lines.append(f"   Position Filter: {truth['position_filter']}")
        report_lines.append(f"   Expected Top-5: {truth['top_5_expected']}")
        
        report_lines.append(f"\n   🔹 MiniLM Results:")
        report_lines.append(f"      Retrieved: {minilm['retrieved']}")
        report_lines.append(f"      Metrics: P@5={minilm['precision_at_k']*100:.2f}%, "
                          f"Overlap={minilm['overlap']*100:.2f}%, MRR={minilm['mrr']:.4f}")
        
        report_lines.append(f"\n   🔸 BGE-M3 Results:")
        report_lines.append(f"      Retrieved: {bge['retrieved']}")
        report_lines.append(f"      Metrics: P@5={bge['precision_at_k']*100:.2f}%, "
                          f"Overlap={bge['overlap']*100:.2f}%, MRR={bge['mrr']:.4f}")
        
        # Determine winner
        minilm_score = minilm['precision_at_k'] + minilm['mrr']
        bge_score = bge['precision_at_k'] + bge['mrr']
        
        if bge_score > minilm_score:
            winner = "BGE-M3"
        elif minilm_score > bge_score:
            winner = "MiniLM"
        else:
            winner = "Tie"
        
        report_lines.append(f"\n   🏆 Query Winner: {winner}")
    
    # Aggregate metrics
    report_lines.append("\n" + "-" * 80)
    report_lines.append("AGGREGATE METRICS COMPARISON")
    report_lines.append("-" * 80)
    
    metrics = ["precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "overlap"]
    
    report_lines.append(f"\n{'Metric':<25} {'MiniLM':<15} {'BGE-M3':<15} {'Winner':<10}")
    report_lines.append("-" * 70)
    
    minilm_wins = 0
    bge_wins = 0
    
    for metric in metrics:
        minilm_avg = np.mean([minilm_results[q][metric] for q in ground_truth])
        bge_avg = np.mean([bge_results[q][metric] for q in ground_truth])
        
        if bge_avg > minilm_avg:
            winner = "BGE-M3"
            bge_wins += 1
        elif minilm_avg > bge_avg:
            winner = "MiniLM"
            minilm_wins += 1
        else:
            winner = "Tie"
        
        report_lines.append(f"mean_{metric:<20} {minilm_avg:<15.4f} {bge_avg:<15.4f} {winner}")
    
    # Final verdict
    report_lines.append("\n" + "=" * 80)
    report_lines.append("FINAL VERDICT")
    report_lines.append("=" * 80)
    report_lines.append(f"\n   Metrics Won - MiniLM: {minilm_wins}, BGE-M3: {bge_wins}")
    
    if bge_wins > minilm_wins:
        overall = "BGE-M3"
    elif minilm_wins > bge_wins:
        overall = "MiniLM"
    else:
        overall = "Tie"
    
    report_lines.append(f"\n   🏆 OVERALL WINNER: {overall}")
    report_lines.append("\n" + "=" * 80)
    
    return "\n".join(report_lines)


def generate_markdown_report(
    ground_truth: Dict,
    minilm_results: Dict,
    bge_results: Dict,
    k: int = 5
) -> str:
    """Generate a markdown evaluation report."""
    
    lines = []
    lines.append("# Embedding Model Evaluation Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Evaluation:** Top-{k} results")
    lines.append("")
    
    lines.append("## Overview")
    lines.append("")
    lines.append("This report compares two embedding models for FPL semantic search:")
    lines.append("- **MiniLM** (all-MiniLM-L6-v2): 384 dimensions, fast inference")
    lines.append("- **BGE-M3** (BAAI/bge-m3): 1024 dimensions, high quality")
    lines.append("")
    
    lines.append("## Evaluation Queries")
    lines.append("")
    lines.append("Queries are designed to match the exact vocabulary used in player embedding descriptions:")
    lines.append("")
    
    for i, query in enumerate(ground_truth, 1):
        truth = ground_truth[query]
        lines.append(f"{i}. **{query}**")
        lines.append(f"   - {truth['description']}")
        lines.append(f"   - Position: {truth['position_filter'] or 'All'}")
        lines.append("")
    
    lines.append("## Results by Query")
    lines.append("")
    
    for query in ground_truth:
        truth = ground_truth[query]
        minilm = minilm_results[query]
        bge = bge_results[query]
        
        lines.append(f"### Query: \"{query}\"")
        lines.append("")
        lines.append(f"**Expected Top-5:** {', '.join(truth['top_5_expected'])}")
        lines.append("")
        
        lines.append("| Model | Retrieved Players | Precision@5 | MRR |")
        lines.append("|-------|------------------|-------------|-----|")
        lines.append(f"| MiniLM | {', '.join(minilm['retrieved'][:3])}... | {minilm['precision_at_k']*100:.1f}% | {minilm['mrr']:.3f} |")
        lines.append(f"| BGE-M3 | {', '.join(bge['retrieved'][:3])}... | {bge['precision_at_k']*100:.1f}% | {bge['mrr']:.3f} |")
        lines.append("")
    
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| Metric | MiniLM | BGE-M3 | Winner |")
    lines.append("|--------|--------|--------|--------|")
    
    metrics = [
        ("Mean Precision@5", "precision_at_k"),
        ("Mean Recall@5", "recall_at_k"),
        ("Mean MRR", "mrr"),
        ("Mean NDCG@5", "ndcg_at_k"),
        ("Mean Overlap", "overlap")
    ]
    
    minilm_wins = 0
    bge_wins = 0
    
    for name, metric in metrics:
        minilm_avg = np.mean([minilm_results[q][metric] for q in ground_truth])
        bge_avg = np.mean([bge_results[q][metric] for q in ground_truth])
        
        if bge_avg > minilm_avg:
            winner = "**BGE-M3** ✅"
            bge_wins += 1
        elif minilm_avg > bge_avg:
            winner = "**MiniLM** ✅"
            minilm_wins += 1
        else:
            winner = "Tie"
        
        lines.append(f"| {name} | {minilm_avg*100:.1f}% | {bge_avg*100:.1f}% | {winner} |")
    
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    
    if bge_wins > minilm_wins:
        lines.append(f"**🏆 Winner: BGE-M3** ({bge_wins}/{len(metrics)} metrics)")
        lines.append("")
        lines.append("BGE-M3 demonstrates superior semantic understanding for FPL player queries,")
        lines.append("particularly when queries match the embedding description vocabulary.")
    elif minilm_wins > bge_wins:
        lines.append(f"**🏆 Winner: MiniLM** ({minilm_wins}/{len(metrics)} metrics)")
    else:
        lines.append("**🤝 Result: Tie**")
    
    lines.append("")
    lines.append("---")
    lines.append("*Report generated by embedding_evaluation.py*")
    
    return "\n".join(lines)


# ============================================================
# 6. MAIN EXECUTION
# ============================================================

def main():
    print("\n" + "=" * 80)
    print("EMBEDDING MODEL EVALUATION: MiniLM vs BGE-M3")
    print("=" * 80)
    
    # Load config
    config = load_config()
    
    # Generate ground truth from Neo4j
    print("\n--- Generating Ground Truth from Neo4j ---")
    ground_truth = generate_ground_truth(config)
    
    # Initialize models
    print("\n--- Initializing MiniLM Model ---")
    from embedding_minilm import SemanticSearchMiniLM
    minilm_search = SemanticSearchMiniLM(config)
    logger.info("✅ MiniLM initialized")
    
    print("\n--- Initializing BGE-M3 Model ---")
    from embedding_bge_m3 import SemanticSearchBGEM3
    bge_search = SemanticSearchBGEM3(config)
    logger.info("✅ BGE-M3 initialized")
    
    # Evaluate models
    k = 5
    
    print("\n--- Evaluating MiniLM ---")
    minilm_results = evaluate_model(minilm_search, ground_truth, "MiniLM", k)
    minilm_search.close()
    
    print("\n--- Evaluating BGE-M3 ---")
    bge_results = evaluate_model(bge_search, ground_truth, "BGE-M3", k)
    bge_search.close()
    
    # Generate and print report
    print("\n--- Generating Report ---")
    report = generate_report(ground_truth, minilm_results, bge_results, k)
    print(report)
    
    # Save results to JSON
    results = {
        "timestamp": datetime.now().isoformat(),
        "k": k,
        "ground_truth": ground_truth,
        "minilm_results": minilm_results,
        "bge_results": bge_results
    }
    
    with open("embedding_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to embedding_evaluation_results.json")
    
    # Generate markdown report
    md_report = generate_markdown_report(ground_truth, minilm_results, bge_results, k)
    with open("EMBEDDING_EVALUATION_REPORT.md", "w", encoding="utf-8") as f:
        f.write(md_report)
    logger.info("Report saved to EMBEDDING_EVALUATION_REPORT.md")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
