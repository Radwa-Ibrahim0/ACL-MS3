"""
baseline_retrieval.py

Baseline graph retrieval layer for the FPL theme.

This module uses:
  - intent (string) from preprocessing.get_intent / process_user_query
  - entities (dict) from preprocessing.extract_entities / process_user_query

to:
  1. Select an appropriate Cypher query template (baseline, no embeddings).
  2. Fill in parameters based on extracted entities.
  3. Execute the Cypher query against Neo4j.
"""

import sys
import logging
from typing import Dict, List, Any, Tuple, Optional
from neo4j import GraphDatabase

# ------------------------------------------------------------
# Logging setup - use StreamHandler with flush for immediate output
# ------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove existing handlers to avoid duplicates
if logger.handlers:
    logger.handlers.clear()

# Create handler that flushes immediately
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
handler.flush = lambda: sys.stdout.flush()
logger.addHandler(handler)

# Import preprocessing module for process_user_query
try:
    from preprocessing import process_user_query
except ImportError:
    process_user_query = None
    logger.warning("Could not import process_user_query from preprocessing module.")


# ------------------------------------------------------------
# Config loader (same format as in preprocessing / KG)
# ------------------------------------------------------------

def load_config(path: str = "config.txt") -> Dict[str, str]:
    config: Dict[str, str] = {}
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    config[k.strip()] = v.strip()
    except FileNotFoundError:
        logger.warning("config.txt not found in baseline_retrieval. Using empty config.")
    return config


# Global config for current season/GW defaults
_CONFIG = load_config()
CURRENT_SEASON = _CONFIG.get("CURRENT_SEASON", "2022-23")
try:
    CURRENT_GW = int(_CONFIG.get("CURRENT_GW", "1"))
except ValueError:
    CURRENT_GW = 1


# ------------------------------------------------------------
# Neo4j helper
# ------------------------------------------------------------

class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Connected to Neo4j at {uri}")

    def close(self):
        if self._driver is not None:
            self._driver.close()
            logger.info("Neo4j connection closed.")

    def run_query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        params = params or {}
        logger.debug(f"Executing Cypher query:\n{cypher}")
        logger.debug(f"With parameters: {params}")
        with self._driver.session() as session:
            result = session.run(cypher, **params)
            records = [record.data() for record in result]
            logger.debug(f"Query returned {len(records)} records")
            return records


# ------------------------------------------------------------
# Baseline query builder
# ------------------------------------------------------------

def _first(lst: List[Any], default: Any = None) -> Any:
    return lst[0] if lst else default


def _correct_intent_based_on_entities(intent: str, entities: Dict[str, List[Any]], query: str = "") -> str:
    """
    Correct the intent based on entity presence when the original intent 
    doesn't match the extracted entities well.
    
    This helps when LLM classification fails or is unavailable.
    """
    has_player = bool(entities.get("Player"))
    has_team = bool(entities.get("Team"))
    has_position = bool(entities.get("Position"))
    has_gw = bool(entities.get("Gameweek"))
    has_season = bool(entities.get("Season"))
    has_stat = bool(entities.get("Statistic"))
    
    query_lower = query.lower() if query else ""
    
    # If intent is player_performance but we have a team and NO players,
    # this might be a team analysis or fixture query (but not if asking for player stats)
    if intent == "player performance" and has_team and not has_player:
        # If asking for player stats from a team, keep as player performance
        player_stat_keywords = ["top", "most", "best", "highest", "leading", "scorer", "points", "goals", "assists", "who", "which player", "players"]
        if has_stat or any(kw in query_lower for kw in player_stat_keywords):
            logger.info(f"Intent kept as 'player performance' (asking for player stats from team)")
            return "player performance"
        
        # Check for fixture-related keywords
        fixture_keywords = ["fixture", "match", "matches", "games", "upcoming", "schedule", "vs", "playing"]
        if any(kw in query_lower for kw in fixture_keywords):
            logger.info(f"Intent corrected: 'player performance' -> 'fixture query' (team without players, fixture keywords)")
            return "fixture query"
        
        # Check for team analysis keywords
        team_keywords = ["happened", "result", "how did"]
        if any(kw in query_lower for kw in team_keywords):
            logger.info(f"Intent corrected: 'player performance' -> 'team analysis' (team without players, analysis keywords)")
            return "team analysis"
        
        # Default: team + GW = team analysis
        if has_gw:
            logger.info(f"Intent corrected: 'player performance' -> 'team analysis' (team + gameweek)")
            return "team analysis"
        
        # Fallback for team-only queries (no stat, no keywords) -> team analysis
        logger.info(f"Intent corrected: 'player performance' -> 'team analysis' (team without players)")
        return "team analysis"
    
    # If intent is player_performance but we only have position (no players),
    # and query asks for recommendations
    if intent == "player performance" and has_position and not has_player:
        recommend_keywords = ["recommend", "suggest", "pick", "should", "consider", "buy", "transfer"]
        if any(kw in query_lower for kw in recommend_keywords):
            logger.info(f"Intent corrected: 'player performance' -> 'recommendation' (position + recommend keywords)")
            return "recommendation"
    
    return intent


# CASE expression to pick the requested statistic from r:PLAYED_IN.
# We can't do r[$stat] in Cypher, so we enumerate the supported fields.
STAT_VALUE_CASE = """
CASE $stat
    WHEN 'goals_scored'     THEN r.goals_scored
    WHEN 'assists'          THEN r.assists
    WHEN 'total_points'     THEN r.total_points
    WHEN 'bonus'            THEN r.bonus
    WHEN 'clean_sheets'     THEN r.clean_sheets
    WHEN 'goals_conceded'   THEN r.goals_conceded
    WHEN 'own_goals'        THEN r.own_goals
    WHEN 'penalties_saved'  THEN r.penalties_saved
    WHEN 'penalties_missed' THEN r.penalties_missed
    WHEN 'yellow_cards'     THEN r.yellow_cards
    WHEN 'red_cards'        THEN r.red_cards
    WHEN 'saves'            THEN r.saves
    WHEN 'bps'              THEN r.bps
    WHEN 'influence'        THEN r.influence
    WHEN 'creativity'       THEN r.creativity
    WHEN 'threat'           THEN r.threat
    WHEN 'ict_index'        THEN r.ict_index
    WHEN 'form'             THEN r.form
    ELSE 0
END
"""


def _stat_alias(stat: str) -> str:
    """Map raw stat field to a more readable aggregate column name."""
    mapping = {
        "goals_scored": "total_goals",
        "assists": "total_assists",
        "total_points": "total_points_value",
        "bonus": "total_bonus",
        "clean_sheets": "total_clean_sheets",
        "goals_conceded": "total_goals_conceded",
        "own_goals": "total_own_goals",
        "penalties_saved": "total_penalties_saved",
        "penalties_missed": "total_penalties_missed",
        "yellow_cards": "total_yellow_cards",
        "red_cards": "total_red_cards",
        "saves": "total_saves",
        "bps": "total_bps",
        "influence": "total_influence",
        "creativity": "total_creativity",
        "threat": "total_threat",
        "ict_index": "total_ict_index",
        "form": "form_score",
    }
    return mapping.get(stat, "stat_value")


def _build_player_name_condition(players: List[str], player_var: str = "p") -> str:
    """
    Build a Cypher condition for matching player names.
    Uses CONTAINS to handle partial name matches (e.g., 'Saka' matches 'Bukayo Saka').
    
    For multiple players, creates: 
        ANY(name IN $players WHERE p.player_name CONTAINS name)
    For single player:
        p.player_name CONTAINS $player
    """
    if len(players) > 1:
        return f"ANY(name IN $players WHERE {player_var}.player_name CONTAINS name)"
    elif len(players) == 1:
        return f"{player_var}.player_name CONTAINS $player"
    return "true"


def build_baseline_query(intent: str, entities: Dict[str, List[Any]], query: str = "") -> Tuple[str, Dict[str, Any], str]:
    """
    Given an intent and extracted entities, choose an appropriate Cypher
    query template (baseline experiment) and fill its parameters.

    Returns:
        (cypher_query, params_dict, description_string)
    """
    # Correct intent based on entities if needed
    original_intent = intent
    intent = _correct_intent_based_on_entities(intent, entities, query)
    if intent != original_intent:
        logger.info(f"Intent was corrected from '{original_intent}' to '{intent}'")
    
    # Ensure expected keys exist
    for key in ["Player", "Team", "Position", "Statistic", "Season", "Gameweek"]:
        entities.setdefault(key, [])

    players   = entities["Player"]
    teams     = entities["Team"]
    positions = entities["Position"]
    stats     = entities["Statistic"]
    seasons   = entities["Season"]
    gws       = entities["Gameweek"]

    # Defaults to keep things usable
    season = _first(seasons)
    gw     = _first(gws)
    stat   = _first(stats, "total_points")  # default to total_points if no stat requested
    stat_alias = _stat_alias(stat)

    # If we have a gameweek but no season, use the current season from config
    if gw is not None and season is None:
        season = CURRENT_SEASON
        logger.info(f"No season specified with GW {gw}, using CURRENT_SEASON: {season}")

    # If we have a team but no season, default to current season for better results
    if teams and season is None:
        season = CURRENT_SEASON
        logger.info(f"No season specified with team query, using CURRENT_SEASON: {season}")

    logger.info(f"Building query for intent='{intent}', players={players}, teams={teams}, "
                f"positions={positions}, stat={stat}, season={season}, gw={gw}")

    # --------------------------------------------------------
    # Intent: player performance
    # --------------------------------------------------------
    if intent == "player performance":
        # Case 1: compare multiple players by a stat in a specific gameweek
        if len(players) >= 2 and season and gw is not None:
            player_condition = _build_player_name_condition(players)
            cypher = f"""
            MATCH (s:Season {{season_name: $season}})
                  -[:HAS_GW]->(gw:Gameweek {{GW_number: $gw}})
                  -[:HAS_FIXTURE]->(f:Fixture)
            MATCH (p:Player)-[r:PLAYED_IN]->(f)
            WHERE {player_condition}
                 WITH p, {STAT_VALUE_CASE} AS stat_value
                 RETURN p.player_name AS player,
                     SUM(stat_value) AS {stat_alias}
                 ORDER BY {stat_alias} DESC
            """
            params = {
                "season": season,
                "gw": gw,
                "players": players,
                "stat": stat,
            }
            desc = "Compare multiple players by a chosen statistic in a specific gameweek."
            logger.info(f"Selected query: {desc}")
            return cypher, params, desc

        # Case 2: compare multiple players by a stat across a season
        if len(players) >= 2 and season:
            player_condition = _build_player_name_condition(players)
            cypher = f"""
            MATCH (s:Season {{season_name: $season}})
                  -[:HAS_GW]->(gw:Gameweek)
                  -[:HAS_FIXTURE]->(f:Fixture)
            MATCH (p:Player)-[r:PLAYED_IN]->(f)
            WHERE {player_condition}
                 WITH p, {STAT_VALUE_CASE} AS stat_value
                 RETURN p.player_name AS player,
                     SUM(stat_value) AS {stat_alias}
                 ORDER BY {stat_alias} DESC
            """
            params = {
                "season": season,
                "players": players,
                "stat": stat,
            }
            desc = "Compare multiple players by a chosen statistic in a given season."
            logger.info(f"Selected query: {desc}")
            return cypher, params, desc

        # Case 3: single player, stat in a specific gameweek
        if len(players) == 1 and season and gw is not None:
            player_condition = _build_player_name_condition(players)
            cypher = f"""
            MATCH (s:Season {{season_name: $season}})
                  -[:HAS_GW]->(gw:Gameweek {{GW_number: $gw}})
                  -[:HAS_FIXTURE]->(f:Fixture)
            MATCH (p:Player)-[r:PLAYED_IN]->(f)
            WHERE {player_condition}
            OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
                 WITH p, pos, gw, r, {STAT_VALUE_CASE} AS stat_value
            RETURN p.player_name AS player,
                   COALESCE(pos.name, 'N/A') AS position,
                   gw.GW_number AS gw,
                     stat_value AS {stat_alias},
                   r.total_points AS total_points,
                   r.minutes AS minutes
            """
            params = {
                "season": season,
                "gw": gw,
                "player": players[0],
                "stat": stat,
            }
            desc = "Retrieve a single player's performance in a specific gameweek."
            logger.info(f"Selected query: {desc}")
            return cypher, params, desc

        # Case 4: single player, stat over a season
        if len(players) == 1 and season:
            player_condition = _build_player_name_condition(players)
            cypher = f"""
            MATCH (s:Season {{season_name: $season}})
                  -[:HAS_GW]->(gw:Gameweek)
                  -[:HAS_FIXTURE]->(f:Fixture)
            MATCH (p:Player)-[r:PLAYED_IN]->(f)
            WHERE {player_condition}
            OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
                 WITH p, pos, {STAT_VALUE_CASE} AS stat_value,
                 SUM(r.total_points) AS total_points,
                 SUM(r.minutes) AS total_minutes
            RETURN p.player_name AS player,
                   COALESCE(pos.name, 'N/A') AS position,
                     SUM(stat_value) AS {stat_alias},
                   total_points,
                   total_minutes
            """
            params = {
                "season": season,
                "player": players[0],
                "stat": stat,
            }
            desc = "Aggregate a single player's chosen statistic over a season."
            logger.info(f"Selected query: {desc}")
            return cypher, params, desc

        # Case 5: top players by stat in a specific gameweek (filtered by position)
        if season and gw is not None:
            cypher = f"""
            MATCH (s:Season {{season_name: $season}})
                  -[:HAS_GW]->(gw:Gameweek {{GW_number: $gw}})
                  -[:HAS_FIXTURE]->(f:Fixture)
            MATCH (p:Player)-[r:PLAYED_IN]->(f)
            OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
                 WITH p, pos, gw, {STAT_VALUE_CASE} AS stat_value
            WHERE ($positions IS NULL OR size($positions) = 0 OR pos.name IN $positions)
            RETURN p.player_name AS player,
                   COALESCE(pos.name, 'N/A') AS position,
                   gw.GW_number AS gw,
                     SUM(stat_value) AS {stat_alias}
                 ORDER BY {stat_alias} DESC
            LIMIT 20
            """
            params = {
                "season": season,
                "gw": gw,
                "positions": positions,
                "stat": stat,
            }
            desc = "Top players by chosen statistic in a specific gameweek (optionally filtered by position)."
            logger.info(f"Selected query: {desc}")
            return cypher, params, desc

        # Case 6: top players from a specific team by stat in a season
        # NOTE: Due to data model limitations (no Player-Team relationship), 
        # this returns ALL players in fixtures involving the team (both sides)
        if teams and season:
            cypher = f"""
            MATCH (t:Team)
            WHERE toLower(t.name) CONTAINS toLower($team)
            WITH t
            MATCH (s:Season {{season_name: $season}})
                  -[:HAS_GW]->(gw:Gameweek)
                  -[:HAS_FIXTURE]->(f:Fixture)
            WHERE (f)-[:HAS_HOME_TEAM]->(t) OR (f)-[:HAS_AWAY_TEAM]->(t)
            MATCH (p:Player)-[r:PLAYED_IN]->(f)
            OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
                 WITH p, pos, t, {STAT_VALUE_CASE} AS stat_value
            WHERE ($positions IS NULL OR size($positions) = 0 OR pos.name IN $positions)
            RETURN p.player_name AS player,
                   COALESCE(pos.name, 'N/A') AS position,
                   t.name AS team,
                     SUM(stat_value) AS {stat_alias}
                 ORDER BY {stat_alias} DESC
            LIMIT 20
            """
            params = {
                "team": teams[0],
                "season": season,
                "positions": positions,
                "stat": stat,
            }
            desc = f"Top players in fixtures involving {teams[0]} by chosen statistic in {season}."
            logger.info(f"Selected query: {desc}")
            return cypher, params, desc
        
        # Case 7: top players from a specific team by stat (all time)
        # NOTE: Due to data model limitations, this returns ALL players in fixtures involving the team
        if teams:
            cypher = f"""
            MATCH (t:Team)
            WHERE toLower(t.name) CONTAINS toLower($team)
            WITH t
            MATCH (f:Fixture)
            WHERE (f)-[:HAS_HOME_TEAM]->(t) OR (f)-[:HAS_AWAY_TEAM]->(t)
            MATCH (p:Player)-[r:PLAYED_IN]->(f)
            OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
                 WITH p, pos, t, {STAT_VALUE_CASE} AS stat_value
            WHERE ($positions IS NULL OR size($positions) = 0 OR pos.name IN $positions)
            RETURN p.player_name AS player,
                   COALESCE(pos.name, 'N/A') AS position,
                   t.name AS team,
                     SUM(stat_value) AS {stat_alias}
                 ORDER BY {stat_alias} DESC
            LIMIT 20
            """
            params = {
                "team": teams[0],
                "positions": positions,
                "stat": stat,
            }
            desc = f"Top players in fixtures involving {teams[0]} by chosen statistic (all time)."
            logger.info(f"Selected query: {desc}")
            return cypher, params, desc

        # Case 8a: top players by stat across ALL seasons (no season, position-filtered, no specific team or players)
        if not season and not teams and not players:
            cypher = f"""
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
            OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
                 WITH p, pos, {STAT_VALUE_CASE} AS stat_value
            WHERE ($positions IS NULL OR size($positions) = 0 OR pos.name IN $positions)
            RETURN p.player_name AS player,
                   COALESCE(pos.name, 'N/A') AS position,
                     SUM(stat_value) AS {stat_alias}
                 ORDER BY {stat_alias} DESC
            LIMIT 20
            """
            params = {
                "positions": positions,
                "stat": stat,
            }
            desc = "League-wide player performance by chosen statistic across all seasons, optionally filtered by position."
            logger.info(f"Selected query: {desc}")
            return cypher, params, desc

        # Case 8: top players by stat across a season (filtered by position, no specific team)
        if season:
            cypher = f"""
            MATCH (s:Season {{season_name: $season}})
                  -[:HAS_GW]->(gw:Gameweek)
                  -[:HAS_FIXTURE]->(f:Fixture)
            MATCH (p:Player)-[r:PLAYED_IN]->(f)
            OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
                 WITH p, pos, {STAT_VALUE_CASE} AS stat_value
            WHERE ($positions IS NULL OR size($positions) = 0 OR pos.name IN $positions)
            RETURN p.player_name AS player,
                   COALESCE(pos.name, 'N/A') AS position,
                     SUM(stat_value) AS {stat_alias}
                 ORDER BY {stat_alias} DESC
            LIMIT 20
            """
            params = {
                "season": season,
                "positions": positions,
                "stat": stat,
            }
            desc = "Top players by chosen statistic in a season (optionally filtered by position)."
            logger.info(f"Selected query: {desc}")
            return cypher, params, desc

    # --------------------------------------------------------
    # Intent: statistics (league-wide leaders)
    # --------------------------------------------------------
    if intent == "statistics":
        # Case 1: Top players by stat in a specific gameweek
        if season and gw is not None:
            cypher = f"""
            MATCH (s:Season {{season_name: $season}})
                  -[:HAS_GW]->(gw:Gameweek {{GW_number: $gw}})
                  -[:HAS_FIXTURE]->(f:Fixture)
            MATCH (p:Player)-[r:PLAYED_IN]->(f)
            OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
                 WITH p, pos, {STAT_VALUE_CASE} AS stat_value
            WHERE ($positions IS NULL OR size($positions) = 0 OR pos.name IN $positions)
            RETURN p.player_name AS player,
                   COALESCE(pos.name, 'N/A') AS position,
                     SUM(stat_value) AS {stat_alias}
                 ORDER BY {stat_alias} DESC
            LIMIT 20
            """
            params = {
                "season": season,
                "gw": gw,
                "positions": positions,
                "stat": stat,
            }
            desc = "Top players by a chosen statistic in a specific gameweek."
            logger.info(f"Selected query: {desc}")
            return cypher, params, desc

        # Case 2: Top players by stat across a season
        if season:
            cypher = f"""
            MATCH (s:Season {{season_name: $season}})
                  -[:HAS_GW]->(gw:Gameweek)
                  -[:HAS_FIXTURE]->(f:Fixture)
            MATCH (p:Player)-[r:PLAYED_IN]->(f)
            OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
            WITH p, pos, {STAT_VALUE_CASE} AS stat_value
            WHERE ($positions IS NULL OR size($positions) = 0 OR pos.name IN $positions)
            RETURN p.player_name AS player,
                   COALESCE(pos.name, 'N/A') AS position,
                   SUM(stat_value) AS {stat_alias}
            ORDER BY {stat_alias} DESC
            LIMIT 20
            """
            params = {
                "season": season,
                "positions": positions,
                "stat": stat,
            }
            desc = "Top players by a chosen statistic in a season, optionally filtered by position."
            logger.info(f"Selected query: {desc}")
            return cypher, params, desc

        # Case 3: Top players by stat across ALL seasons (position-filtered, no season specified)
        if not season:
            cypher = f"""
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
            OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
            WITH p, pos, {STAT_VALUE_CASE} AS stat_value
            WHERE ($positions IS NULL OR size($positions) = 0 OR pos.name IN $positions)
            RETURN p.player_name AS player,
                   COALESCE(pos.name, 'N/A') AS position,
                   SUM(stat_value) AS {stat_alias}
            ORDER BY {stat_alias} DESC
            LIMIT 20
            """
            params = {
                "positions": positions,
                "stat": stat,
            }
            desc = "Top players by a chosen statistic across all seasons, optionally filtered by position."
            logger.info(f"Selected query: {desc}")
            return cypher, params, desc

    # --------------------------------------------------------
    # Intent: recommendations (best players by form/points)
    # --------------------------------------------------------
    if intent == "recommendation":
        # Case 1: Recommend players based on recent form up to a specific gameweek
        if season and gw is not None:
            cypher = """
            MATCH (s:Season {season_name: $season})
                  -[:HAS_GW]->(gw:Gameweek)
                  -[:HAS_FIXTURE]->(f:Fixture)
            WHERE gw.GW_number <= $gw
            MATCH (p:Player)-[r:PLAYED_IN]->(f)
            OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
            WITH p, pos,
                 AVG(r.form) AS avg_form,
                 SUM(r.total_points) AS total_points,
                 MAX(gw.GW_number) AS latest_gw
            WHERE ($positions IS NULL OR size($positions) = 0 OR pos.name IN $positions)
            RETURN p.player_name AS player,
                   COALESCE(pos.name, 'N/A') AS position,
                   avg_form,
                   total_points,
                   latest_gw
            ORDER BY avg_form DESC, total_points DESC
            LIMIT 20
            """
            params = {
                "season": season,
                "gw": gw,
                "positions": positions,
            }
            desc = "Recommend players by average form up to a specific gameweek, optionally filtered by position."
            logger.info(f"Selected query: {desc}")
            return cypher, params, desc

        # Case 2: Recommend players by form across a full season
        if season:
            cypher = """
            MATCH (s:Season {season_name: $season})
                  -[:HAS_GW]->(gw:Gameweek)
                  -[:HAS_FIXTURE]->(f:Fixture)
            MATCH (p:Player)-[r:PLAYED_IN]->(f)
            OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
            WITH p, pos,
                 AVG(r.form) AS avg_form,
                 SUM(r.total_points) AS total_points
            WHERE ($positions IS NULL OR size($positions) = 0 OR pos.name IN $positions)
            RETURN p.player_name AS player,
                   COALESCE(pos.name, 'N/A') AS position,
                   avg_form,
                   total_points
            ORDER BY avg_form DESC, total_points DESC
            LIMIT 20
            """
            params = {
                "season": season,
                "positions": positions,
            }
            desc = "Recommend players by average form (and total points) in a season, optionally filtered by position."
            logger.info(f"Selected query: {desc}")
            return cypher, params, desc

           # Case 3: Recommend players by form across ALL seasons (no season detected)
        cypher = """
           MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
           OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
           WITH p, pos,
               AVG(r.form) AS avg_form,
               SUM(r.total_points) AS total_points
           WHERE ($positions IS NULL OR size($positions) = 0 OR pos.name IN $positions)
           RETURN p.player_name AS player,
                COALESCE(pos.name, 'N/A') AS position,
                avg_form,
                total_points
           ORDER BY avg_form DESC, total_points DESC
           LIMIT 20
           """
        
        params = {"positions": positions}
        desc = "Recommend players by average form and total points across all seasons, optionally filtered by position."
        logger.info(f"Selected query: {desc}")
        return cypher, params, desc

    # --------------------------------------------------------
    # Intent: fixture query
    # --------------------------------------------------------
    if intent == "fixture query":
        # Case 1: fixtures for a given team in a specific gameweek
        if teams and season and gw is not None:
            cypher = """
            MATCH (s:Season {season_name: $season})
                  -[:HAS_GW]->(gw:Gameweek {GW_number: $gw})
                  -[:HAS_FIXTURE]->(f:Fixture)
            MATCH (t:Team {name: $team})
            WHERE (f)-[:HAS_HOME_TEAM]->(t) OR (f)-[:HAS_AWAY_TEAM]->(t)
            WITH gw, f, t
            OPTIONAL MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
            OPTIONAL MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
            WITH gw, f, t,
                 CASE WHEN home.name = t.name THEN away.name ELSE home.name END AS opponent,
                 home.name AS home_team,
                 away.name AS away_team
            RETURN gw.GW_number AS gw,
                   f.fixture_number AS fixture_number,
                   t.name AS team,
                   opponent AS opponent,
                   home_team,
                   away_team
            ORDER BY gw, fixture_number
            """
            params = {
                "season": season,
                "gw": gw,
                "team": teams[0],
            }
            desc = "Fixtures for a given team in a specific gameweek."
            logger.info(f"Selected query: {desc}")
            return cypher, params, desc

        # Case 2: all fixtures in a specific gameweek (league-wide)
        if season and gw is not None:
            cypher = """
            MATCH (s:Season {season_name: $season})
                  -[:HAS_GW]->(gw:Gameweek {GW_number: $gw})
                  -[:HAS_FIXTURE]->(f:Fixture)
            OPTIONAL MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
            OPTIONAL MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
            RETURN gw.GW_number AS gw,
                   f.fixture_number AS fixture_number,
                   home.name AS home_team,
                   away.name AS away_team
            ORDER BY fixture_number
            """
            params = {
                "season": season,
                "gw": gw,
            }
            desc = "All fixtures in a given gameweek."
            logger.info(f"Selected query: {desc}")
            return cypher, params, desc

        # Case 3: all fixtures for a team across a season
        if teams and season:
            cypher = """
            MATCH (s:Season {season_name: $season})
                  -[:HAS_GW]->(gw:Gameweek)
                  -[:HAS_FIXTURE]->(f:Fixture)
            MATCH (t:Team {name: $team})
            WHERE (f)-[:HAS_HOME_TEAM]->(t) OR (f)-[:HAS_AWAY_TEAM]->(t)
            OPTIONAL MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
            OPTIONAL MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
            RETURN gw.GW_number AS gw,
                   f.fixture_number AS fixture_number,
                   home.name AS home_team,
                   away.name AS away_team
            ORDER BY gw, fixture_number
            """
            params = {
                "season": season,
                "team": teams[0],
            }
            desc = "All fixtures for a given team across a season."
            logger.info(f"Selected query: {desc}")
            return cypher, params, desc
        
        # Case 4: team fixtures across all seasons (no season specified)
        if teams:
            cypher = """
            MATCH (gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
            MATCH (t:Team {name: $team})
            WHERE (f)-[:HAS_HOME_TEAM]->(t) OR (f)-[:HAS_AWAY_TEAM]->(t)
            OPTIONAL MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
            OPTIONAL MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
            RETURN gw.season AS season,
                   gw.GW_number AS gw,
                   f.fixture_number AS fixture_number,
                   home.name AS home_team,
                   away.name AS away_team
            ORDER BY season, gw, fixture_number
            """
            params = {"team": teams[0]}
            desc = "All fixtures for a given team across all seasons."
            logger.info(f"Selected query: {desc}")
            return cypher, params, desc

    # --------------------------------------------------------
    # Intent: team analysis
    # --------------------------------------------------------
    if intent == "team analysis":
        # Case 1: Team performance in a SPECIFIC GAMEWEEK - show player stats from that GW
        if teams and season and gw is not None:
            cypher = f"""
            MATCH (s:Season {{season_name: $season}})
                  -[:HAS_GW]->(gw:Gameweek {{GW_number: $gw}})
                  -[:HAS_FIXTURE]->(f:Fixture)
            MATCH (t:Team {{name: $team}})
            WHERE (f)-[:HAS_HOME_TEAM]->(t) OR (f)-[:HAS_AWAY_TEAM]->(t)
            OPTIONAL MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
            OPTIONAL MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
            OPTIONAL MATCH (p:Player)-[r:PLAYED_IN]->(f)
            WITH gw, f, t, home, away, p, r
            RETURN gw.GW_number AS gw,
                   f.fixture_number AS fixture_number,
                   home.name AS home_team,
                   away.name AS away_team,
                   COLLECT(DISTINCT {{
                       player: p.player_name,
                       goals: r.goals_scored,
                       assists: r.assists,
                       points: r.total_points,
                       minutes: r.minutes
                   }}) AS player_performances
            ORDER BY fixture_number
            """
            params = {
                "season": season,
                "gw": gw,
                "team": teams[0],
            }
            desc = "Team analysis for a specific gameweek - fixture details and player performances."
            logger.info(f"Selected query: {desc}")
            return cypher, params, desc

        # Case 2: Team season overview - aggregate stats across the season
        if teams and season:
            cypher = """
            MATCH (s:Season {season_name: $season})
                  -[:HAS_GW]->(gw:Gameweek)
                  -[:HAS_FIXTURE]->(f:Fixture)
            MATCH (t:Team {name: $team})
            WHERE (f)-[:HAS_HOME_TEAM]->(t) OR (f)-[:HAS_AWAY_TEAM]->(t)
            OPTIONAL MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
            OPTIONAL MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
            RETURN gw.GW_number AS gw,
                   f.fixture_number AS fixture_number,
                   home.name AS home_team,
                   away.name AS away_team
            ORDER BY gw, fixture_number
            """
            params = {
                "season": season,
                "team": teams[0],
            }
            desc = "List all fixtures a team is involved in during a season (basis for team analysis)."
            logger.info(f"Selected query: {desc}")
            return cypher, params, desc

        # Case 3: Team fixtures across all seasons
        if teams:
            cypher = """
            MATCH (gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
            MATCH (t:Team {name: $team})
            WHERE (f)-[:HAS_HOME_TEAM]->(t) OR (f)-[:HAS_AWAY_TEAM]->(t)
            OPTIONAL MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
            OPTIONAL MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
            RETURN gw.season AS season,
                   gw.GW_number AS gw,
                   f.fixture_number AS fixture_number,
                   home.name AS home_team,
                   away.name AS away_team
            ORDER BY season, gw, fixture_number
            """
            params = {"team": teams[0]}
            desc = "List all fixtures a team is involved in across all seasons."
            logger.info(f"Selected query: {desc}")
            return cypher, params, desc

    # --------------------------------------------------------
    # Default fallback if nothing matched
    # --------------------------------------------------------
    logger.warning(f"No specific query matched for intent='{intent}', using enriched fallback")

    # We build a richer, more generic player statistics view so the LLM
    # has plenty of structured signal later, while still trying to stay
    # close to the user's entities (season / team / position / statistic).

    # Choose a concrete stat column for convenience but also return
    # multiple commonly-used stats in the SELECT.
    stat_for_order = stat or "total_points"

    base_match = """
    MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
    OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
    OPTIONAL MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
    OPTIONAL MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
    OPTIONAL MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)
    OPTIONAL MATCH (gw)<-[:HAS_GW]-(s:Season)
    """

    where_clauses: List[str] = []
    params: Dict[str, Any] = {"stat": stat_for_order}

    # If the user mentioned a season, restrict to that season
    if season is not None:
        where_clauses.append("s.season_name = $season")
        params["season"] = season

    # If the user mentioned gameweeks, restrict to that subset
    if gws:
        where_clauses.append("gw.GW_number IN $gws")
        params["gws"] = gws

    # If the user mentioned a team, restrict to fixtures involving that team
    if teams:
        where_clauses.append("(home.name = $team OR away.name = $team)")
        params["team"] = teams[0]

    # If the user mentioned positions, restrict to those
    if positions:
        where_clauses.append("pos.name IN $positions")
        params["positions"] = positions

    where_block = ""
    if where_clauses:
        where_block = "WHERE " + " AND ".join(where_clauses)

    cypher = f"""
    {base_match}
    {where_block}
    WITH p, pos, home, away, gw, s,
         SUM(r.total_points)      AS total_points,
         SUM(r.goals_scored)      AS goals_scored,
         SUM(r.assists)           AS assists,
         SUM(r.clean_sheets)      AS clean_sheets,
         SUM(r.saves)             AS saves,
         SUM(r.minutes)           AS minutes,
         AVG(r.form)              AS avg_form,
         SUM({STAT_VALUE_CASE})   AS stat_value
    RETURN p.player_name                AS player,
           COALESCE(pos.name, 'N/A')    AS position,
           s.season_name                AS season,
           gw.GW_number                 AS gw,
           home.name                    AS home_team,
           away.name                    AS away_team,
           total_points,
           goals_scored,
           assists,
           clean_sheets,
           saves,
           minutes,
           avg_form,
           stat_value
    ORDER BY stat_value DESC, total_points DESC
    LIMIT 40
    """

    desc = (
        "Fallback: enriched player statistics view filtered by any "
        "season, gameweek, team and position entities extracted from the "
        "query."
    )
    logger.info(f"Selected query: {desc}")
    return cypher, params, desc


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

class BaselineRetriever:
    """
    High-level wrapper that ties together:
      - preprocessing (optional, via process_user_query)
      - baseline query building
      - Neo4j execution
    """

    def __init__(self, config_path: str = "config.txt"):
        self.config = load_config(config_path)
        uri = self.config.get("URI")
        user = self.config.get("USERNAME")
        pwd = self.config.get("PASSWORD")
        if not uri or not user or not pwd:
            raise ValueError("URI / USERNAME / PASSWORD must be set in config.txt for baseline retrieval.")
        self.client = Neo4jClient(uri, user, pwd)

    def close(self):
        self.client.close()

    def run_from_intent_entities(self, intent: str, entities: Dict[str, List[Any]], query: str = "") -> Dict[str, Any]:
        """
        Main entry when you already have intent + entities from preprocessing.
        Returns a dict with:
          - intent
          - entities
          - cypher
          - params
          - description
          - results (list of dict rows)
        """
        cypher, params, desc = build_baseline_query(intent, entities, query)
        results = self.client.run_query(cypher, params)
        return {
            "intent": intent,
            "entities": entities,
            "cypher": cypher,
            "params": params,
            "description": desc,
            "results": results,
        }

    def run_from_raw_query(self, user_text: str) -> Dict[str, Any]:
        """
        Convenience entry point if you want to go directly from raw user text.
        This requires that process_user_query is importable from your
        preprocessing module.
        """
        if process_user_query is None:
            raise ImportError(
                "process_user_query could not be imported. "
                "Import it from your preprocessing module or call "
                "run_from_intent_entities directly."
            )
        pre = process_user_query(user_text)
        # Pass the original query for intent correction
        return self.run_from_intent_entities(pre["intent"], pre["entities"], user_text)


if __name__ == "__main__":
    # Simple manual test (requires a running Neo4j with the FPL KG loaded)
    br = BaselineRetriever()

    test_questions = [
        # "Compare Saliba and Gabriel clean sheets in 2023-24.",
        # "How many goals did Saka score this season?",
        # "Top midfielders by total points this season.",
        # "Suggest forwards in good form this season.",
        # "What are Man City's fixtures next GW?",
        # "List all fixtures for Spurs in 2022-23.",
        # "Give me the best defenders by points last season.",
        # "Which defenders should I consider picking up for the upcoming gameweek?",
        # "Show me Brighton's matches from last season.",
        # "Who were the top-scoring forwards two seasons ago?",
        # "Is Watkins in better form this season than last year?",
        # "What happened in Spurs' previous gameweek?",
        # "List all midfielders playing for Man United this year.",
        # "Among Liverpool players, who generated the most total points in 2022/23?",
        # "Does Ake or Stones have better clean-sheet numbers this season?",
        # "Who's delivering the most assists for Arsenal lately?",
        # "Show me the upcoming fixtures for Newcastle.",
        # "How many clean sheets did United keep two seasons ago?",
        # "Who scored more points last season, Salah or KDB?",
        # "Show me Brightin's fixtures in 2 gameweeks from now.",
        # "Which Tottenham defenders have been in good form this season?",
        # "Show me the forwards with the most goals and assists scored across all seasons.",
        # "List the defenders with the highest number of clean sheets across all seasons.",
        # "Which midfielders have the most assists overall?",
        # "Which goalkeepers have made the most saves across the seasons?",
        # "Who are the players currently showing the best form?",
        "Show me the forwards with the most goals and assists across all seasons.",
        "Which defenders have the most clean sheets and total points?",
        "Top forwards by goals scored.",
        "Who are the best attacking players in terms of goals and assists combined?",
    ]

    max_rows = 10

    for q in test_questions:
        print("\n" + "=" * 80, flush=True)
        print(f"Question: {q}", flush=True)
        try:
            result = br.run_from_raw_query(q)
            print(f"Intent: {result['intent']}", flush=True)
            print(f"Entities: {result['entities']}", flush=True)
            print(f"Description: {result['description']}", flush=True)

            rows = result["results"]
            print(f"Rows returned: {len(rows)}", flush=True)
            print(f"First {min(max_rows, len(rows))} unique results:", flush=True)

            seen = set()
            shown = 0
            for row in rows:
                key = (
                    row.get("player"),
                    row.get("position"),
                    row.get("season"),
                    row.get("gw"),
                )
                if key in seen:
                    continue
                seen.add(key)
                print(f"   {row}", flush=True)
                shown += 1
                if shown >= max_rows:
                    break
        except Exception as e:
            print(f"Error: {e}", flush=True)
        sys.stdout.flush()

    br.close()
