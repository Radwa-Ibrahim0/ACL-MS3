"""
evaluate_embeddings.py

Evaluation script to compare MiniLM and BGE-M3 embedding models against ground truth.

This script:
1. Defines ground truth answers for 5 FPL queries based on actual player statistics
2. Runs semantic search with both embedding models
3. Calculates evaluation metrics (Precision@K, Recall@K, MRR, NDCG)
4. Outputs a detailed comparison report
"""

import json
import sys
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================
# 1. GROUND TRUTH DEFINITIONS
# ============================================================

# Ground truth generated from actual database statistics (fpl_two_seasons.csv)
# Queries designed to match embedding description vocabulary for better evaluation

GROUND_TRUTH = {
    "strong striker with above-average goals": {
        "description": "Forwards in the strong tier (p75-p90) for goals - above average but not elite",
        "position_filter": "FWD",
        "relevant_players": [
            # From database: FWD with strong goal output (p75-p90 range)
            # Elite: Haaland (36), Kane (30) - these are p90+
            # Strong (above-average): players with good but not elite goals
            "Ollie Watkins",                         # 26 goals, 306 pts - strong tier
            "Ivan Toney",                            # 20 goals, 182 pts - strong tier
            "Cristiano Ronaldo dos Santos Aveiro",   # 18 goals, 159 pts - strong tier
            "Callum Wilson",                         # 18 goals, 157 pts - strong tier
            "Jamie Vardy",                           # 15 goals, 133 pts - strong tier
            "Aleksandar Mitrović",                   # 14 goals, 107 pts - strong tier
            "Teemu Pukki",                           # 11 goals, 142 pts - strong tier
            "Dominic Calvert-Lewin",                 # Strong tier forward
            "Darwin Núñez Ribeiro",                  # Strong tier forward
            "Gabriel Fernando de Jesus",             # Strong tier forward
        ],
        "top_5_expected": [
            "Ollie Watkins",
            "Ivan Toney",
            "Cristiano Ronaldo dos Santos Aveiro",
            "Callum Wilson",
            "Jamie Vardy"
        ]
    },
    
    "elite defenders with top-tier clean sheets": {
        "description": "Defenders in elite tier (p90+) for clean sheets - top 10%",
        "position_filter": "DEF",
        "relevant_players": [
            # From database: DEF with elite clean sheets (14+ CS = p90)
            "Virgil van Dijk",                # 21 CS - elite
            "João Pedro Cavaco Cancelo",      # 19 CS - elite
            "Trent Alexander-Arnold",         # 18 CS - elite
            "Aymeric Laporte",                # 18 CS - elite
            "Andrew Robertson",               # 17 CS - elite
            "Joel Matip",                     # 17 CS - elite
            "Kieran Trippier",                # 16 CS - elite
            "Eric Dier",                      # 16 CS - elite
            "Benjamin White",                 # 15 CS - elite
            "Antonio Rüdiger",                # 15 CS - elite
        ],
        "top_5_expected": [
            "Virgil van Dijk",
            "João Pedro Cavaco Cancelo",
            "Trent Alexander-Arnold",
            "Aymeric Laporte",
            "Andrew Robertson"
        ]
    },
    
    "elite goalkeepers with high clean sheet potential": {
        "description": "Goalkeepers in elite tier (p90+) for clean sheets - top defensive returns",
        "position_filter": "GK",
        "relevant_players": [
            # From database: GK with elite clean sheets (15+ CS = p90)
            "Alisson Ramses Becker",          # 20 CS, 176 pts - elite
            "Ederson Santana de Moraes",      # 20 CS, 155 pts - elite
            "David De Gea Quintana",          # 17 CS, 161 pts - elite
            "Hugo Lloris",                    # 16 CS, 158 pts - elite
            "Nick Pope",                      # 14 CS, 157 pts - elite
            "Aaron Ramsdale",                 # 14 CS, 143 pts - elite
            "David Raya Martin",              # 12 CS, 166 pts - strong
            "José Malheiro de Sá",            # 11 CS, 146 pts - strong
            "Jordan Pickford",                # 8 CS - strong
            "Bernd Leno",                     # 8 CS - strong
        ],
        "top_5_expected": [
            "Alisson Ramses Becker",
            "Ederson Santana de Moraes",
            "David De Gea Quintana",
            "Hugo Lloris",
            "Nick Pope"
        ]
    },
    
    "players in outstanding form": {
        "description": "Players with elite form rating (p90+) - on a hot streak",
        "position_filter": None,  # All positions
        "relevant_players": [
            # From database: Top players by form rating (elite tier)
            "Erling Haaland",              # form=0.743 - elite
            "Mohamed Salah",               # form=0.732 - elite
            "Heung-Min Son",               # form=0.653 - elite
            "Harry Kane",                  # form=0.636 - elite
            "Trent Alexander-Arnold",      # form=0.567 - elite
            "Martin Ødegaard",             # form=0.551 - elite
            "Gabriel Martinelli Silva",    # form=0.547 - elite
            "Marcus Rashford",             # form=0.529 - elite
            "Bukayo Saka",                 # form=0.527 - elite
            "Jarrod Bowen",                # Strong form - elite
        ],
        "top_5_expected": [
            "Erling Haaland",
            "Mohamed Salah",
            "Heung-Min Son",
            "Harry Kane",
            "Trent Alexander-Arnold"
        ]
    },
    
    "players who rarely plays and have very low minutes": {
        "description": "Players with minimal minutes (low tier <p25) - fringe/bench players",
        "position_filter": None,  # All positions
        "relevant_players": [
            # From database: Players with very low minutes (<500)
            "Tyrese Francois",             # 1 min - low
            "Nathan Redmond",              # 1 min - low
            "Jack Hinshelwood",            # 1 min - low
            "George Abbott",               # 1 min - low
            "Marcus Oliveira Alencar",     # 1 min - low
            "Alex Kral",                   # 1 min - low
            "Conor Coventry",              # 1 min - low
            "David Ozoh",                  # 1 min - low
            "Connor Ronan",                # 1 min - low
            "Ethan Nwaneri",               # 1 min - low
            "Wesley Moraes",               # 1 min - low
            "Daniel Chesters",             # 1 min - low
            "Emil Krafth",                 # 1 min - low
            "Moise Kean",                  # 1 min - low
            "Kasey McAteer",               # 2 min - low
        ],
        "top_5_expected": [
            "Tyrese Francois",
            "Nathan Redmond",
            "Jack Hinshelwood",
            "George Abbott",
            "Marcus Oliveira Alencar"
        ]
    },
    
    "elite premium top-tier season-defining asset FWD": {
        "description": "Elite forwards (p90+) - the absolute best FPL forwards",
        "position_filter": "FWD",
        "relevant_players": [
            # From database: Top FWD by total points (elite tier p90+)
            "Harry Kane",                              # 455 pts, 47 goals - elite
            "Ivan Toney",                              # 321 pts, 32 goals - elite
            "Ollie Watkins",                           # 306 pts, 26 goals - elite
            "Erling Haaland",                          # 272 pts, 36 goals - elite
            "Gabriel Fernando de Jesus",               # 245 pts, 19 goals - elite
            "Callum Wilson",                           # 232 pts, 26 goals - elite
            "Michail Antonio",                         # 224 pts, 15 goals - elite
            "Jamie Vardy",                             # 215 pts, 18 goals - elite
            "Danny Ings",                              # 203 pts, 15 goals - elite
            "Che Adams",                               # 197 pts, 12 goals - elite
        ],
        "top_5_expected": [
            "Harry Kane",
            "Ivan Toney",
            "Ollie Watkins",
            "Erling Haaland",
            "Gabriel Fernando de Jesus"
        ]
    }
}


# ============================================================
# 2. EVALUATION METRICS
# ============================================================

def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Precision@K: fraction of retrieved items that are relevant.
    
    Args:
        retrieved: List of retrieved player names (ordered by rank)
        relevant: Set of relevant (ground truth) player names
        k: Number of top results to consider
    
    Returns:
        Precision score (0.0 to 1.0)
    """
    if k == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_retrieved = sum(1 for p in retrieved_k if p in relevant)
    return relevant_retrieved / k


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Recall@K: fraction of relevant items that were retrieved.
    
    Args:
        retrieved: List of retrieved player names (ordered by rank)
        relevant: Set of relevant (ground truth) player names
        k: Number of top results to consider
    
    Returns:
        Recall score (0.0 to 1.0)
    """
    if len(relevant) == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_retrieved = sum(1 for p in retrieved_k if p in relevant)
    return relevant_retrieved / len(relevant)


def mean_reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR): 1/rank of first relevant item.
    
    Args:
        retrieved: List of retrieved player names (ordered by rank)
        relevant: Set of relevant (ground truth) player names
    
    Returns:
        MRR score (0.0 to 1.0)
    """
    for i, player in enumerate(retrieved):
        if player in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: List[str], expected_order: List[str], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@K).
    
    Measures ranking quality considering position of relevant items.
    
    Args:
        retrieved: List of retrieved player names (ordered by rank)
        expected_order: List of expected players in ideal order
        k: Number of top results to consider
    
    Returns:
        NDCG score (0.0 to 1.0)
    """
    import math
    
    def dcg(ranking: List[str], ideal: List[str], k: int) -> float:
        score = 0.0
        for i, player in enumerate(ranking[:k]):
            if player in ideal:
                # Relevance = position in ideal list (higher = more relevant)
                rel = len(ideal) - ideal.index(player) if player in ideal else 0
                score += rel / math.log2(i + 2)  # +2 because rank starts at 1
        return score
    
    dcg_score = dcg(retrieved, expected_order, k)
    ideal_dcg = dcg(expected_order, expected_order, k)
    
    if ideal_dcg == 0:
        return 0.0
    return dcg_score / ideal_dcg


def overlap_score(retrieved: List[str], expected: List[str], k: int) -> float:
    """
    Calculate simple overlap between retrieved and expected top-K.
    
    Args:
        retrieved: List of retrieved player names
        expected: List of expected player names
        k: Number of top results to consider
    
    Returns:
        Overlap percentage (0.0 to 1.0)
    """
    retrieved_set = set(retrieved[:k])
    expected_set = set(expected[:k])
    
    if len(expected_set) == 0:
        return 0.0
    
    overlap = len(retrieved_set & expected_set)
    return overlap / len(expected_set)


def fuzzy_match(name1: str, name2: str) -> bool:
    """
    Fuzzy match two player names (handles variations).
    
    Args:
        name1: First player name
        name2: Second player name
    
    Returns:
        True if names match (exact or fuzzy)
    """
    # Normalize names
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()
    
    # Exact match
    if n1 == n2:
        return True
    
    # One contains the other (handles "Salah" vs "Mohamed Salah")
    if n1 in n2 or n2 in n1:
        return True
    
    # Check last name match
    parts1 = n1.split()
    parts2 = n2.split()
    if parts1[-1] == parts2[-1]:
        return True
    
    return False


def find_matching_player(player: str, ground_truth_list: List[str]) -> Optional[str]:
    """Find a matching player in the ground truth list."""
    for gt_player in ground_truth_list:
        if fuzzy_match(player, gt_player):
            return gt_player
    return None


# ============================================================
# 3. MODEL EVALUATION
# ============================================================

def load_config() -> Dict[str, str]:
    """Load configuration from config.txt"""
    config: Dict[str, str] = {}
    try:
        with open("config.txt", "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    config[key] = value
    except FileNotFoundError:
        logger.error("config.txt not found")
    return config


def evaluate_model(
    model_name: str,
    search_func,
    queries: Dict[str, Dict],
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Evaluate a single embedding model against ground truth.
    
    Args:
        model_name: Name of the model being evaluated
        search_func: Function that takes (query, top_k, position) and returns results
        queries: Ground truth queries dictionary
        top_k: Number of results to evaluate
    
    Returns:
        Dictionary with evaluation metrics
    """
    results = {
        "model": model_name,
        "queries": {},
        "aggregate": {}
    }
    
    all_precision = []
    all_recall = []
    all_mrr = []
    all_ndcg = []
    all_overlap = []
    
    for query, ground_truth in queries.items():
        logger.info(f"Evaluating '{query}' with {model_name}...")
        
        position = ground_truth.get("position_filter")
        relevant_players = set(ground_truth["relevant_players"])
        expected_top5 = ground_truth["top_5_expected"]
        
        # Run search
        try:
            search_results = search_func(query, top_k=top_k, position=position)
            retrieved = [r["player"] for r in search_results]
        except Exception as e:
            logger.error(f"Search failed for {model_name}: {e}")
            retrieved = []
        
        # Normalize retrieved names for matching
        matched_retrieved = []
        for player in retrieved:
            match = find_matching_player(player, ground_truth["relevant_players"])
            if match:
                matched_retrieved.append(match)
            else:
                matched_retrieved.append(player)  # Keep original if no match
        
        # Calculate metrics
        precision = precision_at_k(matched_retrieved, relevant_players, top_k)
        recall = recall_at_k(matched_retrieved, relevant_players, top_k)
        mrr = mean_reciprocal_rank(matched_retrieved, relevant_players)
        ndcg = ndcg_at_k(matched_retrieved, expected_top5, top_k)
        overlap = overlap_score(matched_retrieved, expected_top5, top_k)
        
        all_precision.append(precision)
        all_recall.append(recall)
        all_mrr.append(mrr)
        all_ndcg.append(ndcg)
        all_overlap.append(overlap)
        
        # Store query results
        results["queries"][query] = {
            "retrieved": retrieved,
            "matched": matched_retrieved,
            "expected": expected_top5,
            "metrics": {
                "precision@k": round(precision, 4),
                "recall@k": round(recall, 4),
                "mrr": round(mrr, 4),
                "ndcg@k": round(ndcg, 4),
                "overlap": round(overlap, 4)
            }
        }
    
    # Calculate aggregate metrics
    results["aggregate"] = {
        "mean_precision@k": round(sum(all_precision) / len(all_precision), 4),
        "mean_recall@k": round(sum(all_recall) / len(all_recall), 4),
        "mean_mrr": round(sum(all_mrr) / len(all_mrr), 4),
        "mean_ndcg@k": round(sum(all_ndcg) / len(all_ndcg), 4),
        "mean_overlap": round(sum(all_overlap) / len(all_overlap), 4)
    }
    
    return results


# ============================================================
# 4. COMPARISON AND REPORTING
# ============================================================

def compare_models(minilm_results: Dict, bge_m3_results: Dict) -> Dict[str, Any]:
    """
    Compare two models and determine winner for each metric.
    
    Args:
        minilm_results: Evaluation results for MiniLM
        bge_m3_results: Evaluation results for BGE-M3
    
    Returns:
        Comparison summary
    """
    comparison = {
        "winner_by_metric": {},
        "winner_by_query": {},
        "overall_winner": None
    }
    
    # Compare aggregate metrics
    metrics = ["mean_precision@k", "mean_recall@k", "mean_mrr", "mean_ndcg@k", "mean_overlap"]
    minilm_wins = 0
    bge_m3_wins = 0
    
    for metric in metrics:
        minilm_score = minilm_results["aggregate"][metric]
        bge_m3_score = bge_m3_results["aggregate"][metric]
        
        if minilm_score > bge_m3_score:
            comparison["winner_by_metric"][metric] = "MiniLM"
            minilm_wins += 1
        elif bge_m3_score > minilm_score:
            comparison["winner_by_metric"][metric] = "BGE-M3"
            bge_m3_wins += 1
        else:
            comparison["winner_by_metric"][metric] = "Tie"
    
    # Compare per query
    for query in minilm_results["queries"]:
        minilm_overlap = minilm_results["queries"][query]["metrics"]["overlap"]
        bge_m3_overlap = bge_m3_results["queries"][query]["metrics"]["overlap"]
        
        if minilm_overlap > bge_m3_overlap:
            comparison["winner_by_query"][query] = "MiniLM"
        elif bge_m3_overlap > minilm_overlap:
            comparison["winner_by_query"][query] = "BGE-M3"
        else:
            comparison["winner_by_query"][query] = "Tie"
    
    # Overall winner
    if minilm_wins > bge_m3_wins:
        comparison["overall_winner"] = "MiniLM"
    elif bge_m3_wins > minilm_wins:
        comparison["overall_winner"] = "BGE-M3"
    else:
        comparison["overall_winner"] = "Tie"
    
    comparison["score"] = {"MiniLM": minilm_wins, "BGE-M3": bge_m3_wins}
    
    return comparison


def print_report(minilm_results: Dict, bge_m3_results: Dict, comparison: Dict):
    """Print a detailed comparison report."""
    
    print("\n" + "=" * 80)
    print("EMBEDDING MODEL EVALUATION REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Evaluation K: 5 (Top-5 results)")
    
    # Per-query results
    print("\n" + "-" * 80)
    print("DETAILED RESULTS BY QUERY")
    print("-" * 80)
    
    for query in GROUND_TRUTH:
        print(f"\n📋 Query: '{query}'")
        print(f"   Position Filter: {GROUND_TRUTH[query].get('position_filter', 'None')}")
        print(f"   Expected Top-5: {GROUND_TRUTH[query]['top_5_expected']}")
        
        print(f"\n   🔹 MiniLM Results:")
        minilm_q = minilm_results["queries"][query]
        print(f"      Retrieved: {minilm_q['retrieved']}")
        print(f"      Metrics: P@5={minilm_q['metrics']['precision@k']:.2%}, "
              f"Overlap={minilm_q['metrics']['overlap']:.2%}, "
              f"MRR={minilm_q['metrics']['mrr']:.4f}")
        
        print(f"\n   🔸 BGE-M3 Results:")
        bge_m3_q = bge_m3_results["queries"][query]
        print(f"      Retrieved: {bge_m3_q['retrieved']}")
        print(f"      Metrics: P@5={bge_m3_q['metrics']['precision@k']:.2%}, "
              f"Overlap={bge_m3_q['metrics']['overlap']:.2%}, "
              f"MRR={bge_m3_q['metrics']['mrr']:.4f}")
        
        winner = comparison["winner_by_query"][query]
        print(f"\n   🏆 Query Winner: {winner}")
    
    # Aggregate comparison
    print("\n" + "-" * 80)
    print("AGGREGATE METRICS COMPARISON")
    print("-" * 80)
    
    print(f"\n{'Metric':<25} {'MiniLM':<15} {'BGE-M3':<15} {'Winner':<15}")
    print("-" * 70)
    
    metrics = ["mean_precision@k", "mean_recall@k", "mean_mrr", "mean_ndcg@k", "mean_overlap"]
    for metric in metrics:
        minilm_val = minilm_results["aggregate"][metric]
        bge_m3_val = bge_m3_results["aggregate"][metric]
        winner = comparison["winner_by_metric"][metric]
        print(f"{metric:<25} {minilm_val:<15.4f} {bge_m3_val:<15.4f} {winner:<15}")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    print(f"\n   Metrics Won - MiniLM: {comparison['score']['MiniLM']}, BGE-M3: {comparison['score']['BGE-M3']}")
    print(f"\n   🏆 OVERALL WINNER: {comparison['overall_winner']}")
    print("\n" + "=" * 80)


def save_results(minilm_results: Dict, bge_m3_results: Dict, comparison: Dict, filename: str = "embedding_evaluation_results.json"):
    """Save evaluation results to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "ground_truth": {k: {"top_5_expected": v["top_5_expected"], "position_filter": v.get("position_filter")} 
                        for k, v in GROUND_TRUTH.items()},
        "minilm_results": minilm_results,
        "bge_m3_results": bge_m3_results,
        "comparison": comparison
    }
    
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Results saved to {filename}")


# ============================================================
# 5. MAIN EXECUTION
# ============================================================

def main():
    """Main evaluation function."""
    print("\n" + "=" * 80)
    print("EMBEDDING MODEL EVALUATION: MiniLM vs BGE-M3")
    print("=" * 80)
    
    config = load_config()
    
    if not config:
        logger.error("Failed to load configuration. Exiting.")
        return
    
    top_k = 5
    
    # Initialize MiniLM
    print("\n--- Initializing MiniLM Model ---")
    try:
        from embedding_minilm import SemanticSearchMiniLM
        minilm_search = SemanticSearchMiniLM(config)
        minilm_available = True
        logger.info("✅ MiniLM initialized")
    except Exception as e:
        logger.error(f"Failed to initialize MiniLM: {e}")
        minilm_available = False
    
    # Initialize BGE-M3
    print("\n--- Initializing BGE-M3 Model ---")
    try:
        from embedding_bge_m3 import SemanticSearchBGEM3
        bge_m3_search = SemanticSearchBGEM3(config)
        bge_m3_available = True
        logger.info("✅ BGE-M3 initialized")
    except Exception as e:
        logger.error(f"Failed to initialize BGE-M3: {e}")
        bge_m3_available = False
    
    if not minilm_available and not bge_m3_available:
        logger.error("No models available. Exiting.")
        return
    
    # Evaluate MiniLM
    minilm_results = None
    if minilm_available:
        print("\n--- Evaluating MiniLM ---")
        minilm_results = evaluate_model(
            "MiniLM",
            minilm_search.search,
            GROUND_TRUTH,
            top_k=top_k
        )
        minilm_search.close()
    
    # Evaluate BGE-M3
    bge_m3_results = None
    if bge_m3_available:
        print("\n--- Evaluating BGE-M3 ---")
        bge_m3_results = evaluate_model(
            "BGE-M3",
            bge_m3_search.search,
            GROUND_TRUTH,
            top_k=top_k
        )
        bge_m3_search.close()
    
    # Compare if both available
    if minilm_results and bge_m3_results:
        print("\n--- Comparing Models ---")
        comparison = compare_models(minilm_results, bge_m3_results)
        
        # Print report
        print_report(minilm_results, bge_m3_results, comparison)
        
        # Save results
        save_results(minilm_results, bge_m3_results, comparison)
    
    elif minilm_results:
        print("\n--- MiniLM Results Only ---")
        print(json.dumps(minilm_results, indent=2))
    
    elif bge_m3_results:
        print("\n--- BGE-M3 Results Only ---")
        print(json.dumps(bge_m3_results, indent=2))
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
