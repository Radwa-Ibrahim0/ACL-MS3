"""
Direct test script for baseline.py - bypasses preprocessing LLM API
Tests the fixes for:
1. LIMIT 10 with ranking, LIMIT 50 without ranking
2. Print only first 10 results in terminal
3. Order by ALL statistics in the order they were written
4. Position filtering (only MID, FWD, etc. - no unknown positions)
"""

import sys
import os
# Add project root to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Main.baseline import execute_baseline_query

print("=" * 70)
print("DIRECT BASELINE TESTING (No LLM API - Simulated Preprocessing)")
print("=" * 70)

# Test cases with simulated preprocessing output
test_cases = [
    # ============================================================
    # Test 1: Multiple statistics - should order by goals_scored first, then assists
    # ============================================================
    {
        "name": "Players ranked by goals AND assists (order matters)",
        "preprocessing": {
            "query": "Best players by goals scored and assists",
            "intent": "top_players_by_stat",
            "entities": {"Player": [], "Team": [], "Position": [], "Statistic": ["goals_scored", "assists"], "Season": [], "Gameweek": []},
            "ranking": "best",
            "threshold": None
        }
    },
    
    # ============================================================
    # Test 2: Multiple statistics reversed - should order by assists first, then goals
    # ============================================================
    {
        "name": "Players ranked by assists AND goals (reversed order)",
        "preprocessing": {
            "query": "Best players by assists and goals scored",
            "intent": "top_players_by_stat",
            "entities": {"Player": [], "Team": [], "Position": [], "Statistic": ["assists", "goals_scored"], "Season": [], "Gameweek": []},
            "ranking": "best",
            "threshold": None
        }
    },
    
    # ============================================================
    # Test 3: With ranking (best) - should LIMIT 10
    # ============================================================
    {
        "name": "Top 10 players by goals (with ranking=best -> LIMIT 10)",
        "preprocessing": {
            "query": "Top 10 players by goals scored",
            "intent": "top_players_by_stat",
            "entities": {"Player": [], "Team": [], "Position": [], "Statistic": ["goals_scored"], "Season": [], "Gameweek": []},
            "ranking": "best",
            "threshold": None
        }
    },
    
    # ============================================================
    # Test 4: Without ranking - should LIMIT 50
    # ============================================================
    {
        "name": "Midfielders >= 70 points (no ranking -> LIMIT 50)",
        "preprocessing": {
            "query": "Midfielders with more than or equal 70 total points",
            "intent": "top_players_by_stat",
            "entities": {"Player": [], "Team": [], "Position": ["MID"], "Statistic": ["total_points"], "Season": [], "Gameweek": []},
            "ranking": None,
            "threshold": {"stat": "total_points", "operator": ">=", "value": 70}
        }
    },
    
    # ============================================================
    # Test 5: Team + Multiple stats
    # ============================================================
    {
        "name": "Arsenal players by total_points and goals_scored",
        "preprocessing": {
            "query": "Arsenal players ranked by points and goals",
            "intent": "top_players_by_stat",
            "entities": {"Player": [], "Team": ["Arsenal"], "Position": [], "Statistic": ["total_points", "goals_scored"], "Season": [], "Gameweek": []},
            "ranking": "best",
            "threshold": None
        }
    },
    
    # ============================================================
    # Test 6: Forwards with 5 assists (no ranking -> LIMIT 50)
    # ============================================================
    {
        "name": "Forwards with exactly 5 assists (no ranking)",
        "preprocessing": {
            "query": "Forwards with 5 assists",
            "intent": "top_players_by_stat",
            "entities": {"Player": [], "Team": [], "Position": ["FWD"], "Statistic": ["assists"], "Season": [], "Gameweek": []},
            "ranking": None,
            "threshold": {"stat": "assists", "operator": "=", "value": 5}
        }
    },
]

for i, test in enumerate(test_cases, 1):
    print(f"\n{'='*70}")
    print(f"🔹 Test {i}: {test['name']}")
    print("-" * 70)
    
    print("\n📥 Simulated Preprocessing Output:")
    print(f"   Intent:    {test['preprocessing']['intent']}")
    print(f"   Entities:  {test['preprocessing']['entities']}")
    print(f"   Ranking:   {test['preprocessing']['ranking']}")
    print(f"   Threshold: {test['preprocessing']['threshold']}")
    
    print("\n" + "-" * 70)
    
    try:
        results = execute_baseline_query(test['preprocessing'])
        print(f"\n✅ Test {i} completed. Retrieved {len(results)} results.")
    except Exception as e:
        print(f"\n❌ Test {i} failed with error: {e}")
    
    print("=" * 70)

print("\n" + "=" * 70)
print("DIRECT TESTING COMPLETE")
print("=" * 70)
