"""
LLM Comparison Test Queries
===========================

Simple test queries for comparing 3 LLMs on the FPL Knowledge Graph RAG system.

- 10 BASE QUERIES: Mix of easy/medium/hard to benchmark accuracy
- 18 EDGE CASE QUERIES: Designed to expose limitations and errors

The evaluation flow:
1. Query → Pipeline (baseline + embedding) → Context Data
2. Context Data + Prompt → LLM1, LLM2, LLM3 → 3 Answers
3. Output: {query, context_data, answers} for manual accuracy evaluation
"""

from dataclasses import dataclass
from typing import List


@dataclass
class TestQuery:
    """A test query with metadata."""
    id: str
    query: str
    difficulty: str  # easy, medium, hard
    category: str    # factual, aggregation, comparison, etc.
    notes: str = ""  # What this query tests


@dataclass
class EdgeCaseQuery:
    """An edge case query to expose limitations."""
    id: str
    query: str
    category: str          # ambiguous, out_of_scope, complex_reasoning, etc.
    expected_issue: str    # What problem this query might expose
    notes: str = ""


# ============================================================
# 10 BASE QUERIES (Easy / Medium / Hard)
# ============================================================

BASE_QUERIES: List[TestQuery] = [
    
    # ---- EASY (3) ----
    
    TestQuery(
        id="BASE_01",
        query="Who was the top scorer in the 2022-23 Premier League season?",
        difficulty="easy",
        category="aggregation",
        notes="Simple top-1 aggregation query"
    ),
    
    TestQuery(
        id="BASE_02",
        query="Which goalkeeper had the most saves in the 2022-23 season?",
        difficulty="easy",
        category="aggregation",
        notes="Position filter + aggregation"
    ),
    
    TestQuery(
        id="BASE_03",
        query="How many total points did Harry Kane score in the 2022-23 season?",
        difficulty="easy",
        category="factual",
        notes="Specific player lookup"
    ),
    
    # ---- MEDIUM (4) ----
    
    TestQuery(
        id="BASE_04",
        query="Who had the most assists in the 2022-23 season and how many did they have?",
        difficulty="medium",
        category="aggregation",
        notes="Requires understanding of assists metric"
    ),
    
    TestQuery(
        id="BASE_05",
        query="List the top 5 players by total FPL points in the 2022-23 season.",
        difficulty="medium",
        category="ranking",
        notes="Multi-result ranking query"
    ),
    
    TestQuery(
        id="BASE_06",
        query="Which teams were promoted to the Premier League for the 2022-23 season?",
        difficulty="medium",
        category="comparison",
        notes="Cross-season team comparison"
    ),
    
    TestQuery(
        id="BASE_07",
        query="How many players received a red card in the 2022-23 season?",
        difficulty="medium",
        category="counting",
        notes="Filtered count query"
    ),
    
    # ---- HARD (3) ----
    
    TestQuery(
        id="BASE_08",
        query="Which player scored the most goals in a single gameweek during the 2022-23 season?",
        difficulty="hard",
        category="multi-hop",
        notes="Requires grouping by player AND gameweek"
    ),
    
    TestQuery(
        id="BASE_09",
        query="Compare Erling Haaland and Harry Kane's total goals in the 2022-23 season. Who scored more and by how many?",
        difficulty="hard",
        category="comparison",
        notes="Direct player comparison with numerical reasoning"
    ),
    
    TestQuery(
        id="BASE_10",
        query="Which defender had the most clean sheets in the 2021-22 season?",
        difficulty="hard",
        category="multi-hop",
        notes="Position filter + aggregation"
    ),
]


# ============================================================
# 10 ADVANCED QUERIES (Thresholds, Gameweeks, Multi-stat, etc.)
# ============================================================

ADVANCED_QUERIES: List[TestQuery] = [
    
    # ---- THRESHOLD QUERIES (2) ----
    
    TestQuery(
        id="ADV_01",
        query="Which players scored more than 15 goals in the 2022-23 season?",
        difficulty="medium",
        category="threshold",
        notes="Tests threshold filter (>15 goals)"
    ),
    
    TestQuery(
        id="ADV_02",
        query="List all midfielders who had at least 10 assists in the 2022-23 season.",
        difficulty="medium",
        category="threshold",
        notes="Tests position + threshold combination"
    ),
    
    # ---- SPECIFIC GAMEWEEK QUERIES (2) ----
    
    TestQuery(
        id="ADV_03",
        query="Who scored the most goals in Gameweek 1 of the 2022-23 season?",
        difficulty="hard",
        category="gameweek",
        notes="Tests specific gameweek filtering"
    ),
    
    TestQuery(
        id="ADV_04",
        query="Which players got the most FPL points in Gameweek 38 of the 2021-22 season?",
        difficulty="hard",
        category="gameweek",
        notes="Tests final gameweek of different season"
    ),
    
    # ---- MULTI-STAT QUERIES (2) ----
    
    TestQuery(
        id="ADV_05",
        query="Which forwards had more than 10 goals AND more than 5 assists in the 2022-23 season?",
        difficulty="hard",
        category="multi_stat",
        notes="Tests AND condition with multiple stats"
    ),
    
    TestQuery(
        id="ADV_06",
        query="Who had the best goals plus assists combined in the 2022-23 season?",
        difficulty="hard",
        category="multi_stat",
        notes="Tests stat combination (G+A)"
    ),
    
    # ---- MULTI-POSITION/TEAM QUERIES (2) ----
    
    TestQuery(
        id="ADV_07",
        query="Compare the total goals scored by defenders vs midfielders in the 2022-23 season.",
        difficulty="hard",
        category="multi_position",
        notes="Tests position group aggregation comparison"
    ),
    
    TestQuery(
        id="ADV_08",
        query="Which Manchester City players had the most assists in the 2022-23 season?",
        difficulty="medium",
        category="team_filter",
        notes="Tests team-specific player filtering"
    ),
    
    # ---- CROSS-SEASON / COMPARATIVE (2) ----
    
    TestQuery(
        id="ADV_09",
        query="Did Mohamed Salah score more goals in 2021-22 or 2022-23?",
        difficulty="hard",
        category="cross_season",
        notes="Tests comparing same player across seasons"
    ),
    
    TestQuery(
        id="ADV_10",
        query="Which goalkeeper had the most clean sheets in the 2022-23 season and how many did they have?",
        difficulty="medium",
        category="aggregation",
        notes="Tests GK position + clean_sheets stat"
    ),
]


# ============================================================
# 18 EDGE CASE QUERIES
# ============================================================

EDGE_CASE_QUERIES: List[EdgeCaseQuery] = [
    
    # ---- AMBIGUOUS (3) ----
    
    EdgeCaseQuery(
        id="EDGE_01",
        query="Who is the best player?",
        category="ambiguous",
        expected_issue="No season specified, no metric defined (points? goals? assists?)"
    ),
    
    EdgeCaseQuery(
        id="EDGE_02",
        query="How did Salah do?",
        category="ambiguous",
        expected_issue="No season, no specific metric, informal language"
    ),
    
    EdgeCaseQuery(
        id="EDGE_03",
        query="Who scored more, Kane or the Egyptian?",
        category="ambiguous",
        expected_issue="Indirect reference ('the Egyptian'), no season specified"
    ),
    
    # ---- OUT OF SCOPE (3) ----
    
    EdgeCaseQuery(
        id="EDGE_04",
        query="What was Erling Haaland's salary in 2022-23?",
        category="out_of_scope",
        expected_issue="Salary data not in the knowledge graph"
    ),
    
    EdgeCaseQuery(
        id="EDGE_05",
        query="Who will be the top scorer in the 2024-25 season?",
        category="out_of_scope",
        expected_issue="Future prediction - data only covers 2021-22 and 2022-23"
    ),
    
    EdgeCaseQuery(
        id="EDGE_06",
        query="How many Champions League goals did Liverpool score?",
        category="out_of_scope",
        expected_issue="Only Premier League FPL data available"
    ),
    
    # ---- COMPLEX REASONING (3) ----
    
    EdgeCaseQuery(
        id="EDGE_07",
        query="Which player improved the most between 2021-22 and 2022-23 in terms of total points?",
        category="complex_reasoning",
        expected_issue="Requires cross-season comparison for same players"
    ),
    
    EdgeCaseQuery(
        id="EDGE_08",
        query="What is the average points per game for forwards who scored at least 10 goals in 2022-23?",
        category="complex_reasoning",
        expected_issue="Multi-step filtering + calculation"
    ),
    
    EdgeCaseQuery(
        id="EDGE_09",
        query="If I picked Haaland, Salah, and De Bruyne for my FPL team in 2022-23, what would be my combined points?",
        category="complex_reasoning",
        expected_issue="Hypothetical team calculation"
    ),
    
    # ---- ENTITY RESOLUTION (2) ----
    
    EdgeCaseQuery(
        id="EDGE_10",
        query="How many goals did Bruno score in 2022-23?",
        category="entity_resolution",
        expected_issue="Multiple players named Bruno (Fernandes, Guimarães)"
    ),
    
    EdgeCaseQuery(
        id="EDGE_11",
        query="What are Man City's total goals in 2022-23?",
        category="entity_resolution",
        expected_issue="Team-level aggregation (need to sum all Man City players)"
    ),
    
    # ---- TEMPORAL (2) ----
    
    EdgeCaseQuery(
        id="EDGE_12",
        query="Who scored the most goals in December 2022?",
        category="temporal",
        expected_issue="Monthly breakdown requires date filtering"
    ),
    
    EdgeCaseQuery(
        id="EDGE_13",
        query="Which newly promoted team performed best in 2022-23?",
        category="temporal",
        expected_issue="Requires identifying promoted teams + comparing performance"
    ),
    
    # ---- NONSENSICAL (2) ----
    
    EdgeCaseQuery(
        id="EDGE_14",
        query="What is the square root of Mohamed Salah's assists divided by the color blue?",
        category="nonsensical",
        expected_issue="Mathematically meaningless query"
    ),
    
    EdgeCaseQuery(
        id="EDGE_15",
        query="Rank all players by their zodiac sign compatibility with winning.",
        category="nonsensical",
        expected_issue="No zodiac data, concept is meaningless for FPL"
    ),
    
    # ---- SCALE (1) ----
    
    EdgeCaseQuery(
        id="EDGE_16",
        query="List every single player appearance in the 2022-23 season with all their stats.",
        category="scale",
        expected_issue="Would return thousands of rows - impractical"
    ),
    
    # ---- MISSPELLING / NICKNAME (2) ----
    
    EdgeCaseQuery(
        id="EDGE_17",
        query="How many goals did Hallend score?",
        category="misspelling",
        expected_issue="'Hallend' is a misspelling of 'Haaland'"
    ),
    
    EdgeCaseQuery(
        id="EDGE_18",
        query="What was CR7's performance in 2021-22?",
        category="nickname",
        expected_issue="CR7 is Cristiano Ronaldo's nickname"
    ),
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_all_base_queries() -> List[TestQuery]:
    """Return all base test queries."""
    return BASE_QUERIES


def get_all_edge_queries() -> List[EdgeCaseQuery]:
    """Return all edge case queries."""
    return EDGE_CASE_QUERIES


def get_queries_by_difficulty(difficulty: str) -> List[TestQuery]:
    """Filter base queries by difficulty level."""
    return [q for q in BASE_QUERIES if q.difficulty == difficulty]


def get_edge_queries_by_category(category: str) -> List[EdgeCaseQuery]:
    """Filter edge queries by category."""
    return [q for q in EDGE_CASE_QUERIES if q.category == category]


def print_test_summary():
    """Print a summary of all test queries."""
    print("=" * 60)
    print("FPL LLM TEST SUITE")
    print("=" * 60)
    
    print(f"\n📊 BASE QUERIES: {len(BASE_QUERIES)}")
    for diff in ["easy", "medium", "hard"]:
        count = len(get_queries_by_difficulty(diff))
        print(f"   - {diff.capitalize()}: {count}")
    
    print(f"\n🔬 EDGE CASE QUERIES: {len(EDGE_CASE_QUERIES)}")
    categories = set(q.category for q in EDGE_CASE_QUERIES)
    for cat in sorted(categories):
        count = len(get_edge_queries_by_category(cat))
        print(f"   - {cat}: {count}")


if __name__ == "__main__":
    print_test_summary()
    
    print("\n" + "=" * 60)
    print("BASE QUERIES:")
    print("=" * 60)
    for q in BASE_QUERIES:
        print(f"\n[{q.id}] ({q.difficulty.upper()})")
        print(f"  Q: {q.query}")
    
    print("\n" + "=" * 60)
    print("EDGE CASE QUERIES:")
    print("=" * 60)
    for q in EDGE_CASE_QUERIES:
        print(f"\n[{q.id}] ({q.category})")
        print(f"  Q: {q.query}")
        print(f"  Issue: {q.expected_issue}")
