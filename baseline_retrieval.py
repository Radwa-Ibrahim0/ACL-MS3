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
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Any, Tuple, Optional, Callable
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


QueryTuple = Tuple[str, Dict[str, Any], str]


@dataclass
class QueryContext:
    intent: str
    entities: Dict[str, List[Any]]
    query_text: str = ""
    ranking: Optional[str] = None
    threshold: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        for key in ["Player", "Team", "Position", "Statistic", "Season", "Gameweek"]:
            self.entities.setdefault(key, [])

        self.players: List[str] = [str(p) for p in self.entities["Player"] if p]
        self.teams: List[str] = [str(t) for t in self.entities["Team"] if t]
        self.positions: List[str] = [str(pos) for pos in self.entities["Position"] if pos]
        self.stats: List[str] = [str(stat) for stat in self.entities["Statistic"] if stat]
        self.seasons: List[str] = [str(season) for season in self.entities["Season"] if season]

        gw_values: List[int] = []
        for gw in self.entities["Gameweek"]:
            try:
                gw_values.append(int(gw))
            except (TypeError, ValueError):
                continue
        self.gameweeks = gw_values

        self.default_season = _CONFIG.get("CURRENT_SEASON", CURRENT_SEASON)
        try:
            self.default_gameweek = int(_CONFIG.get("CURRENT_GW", str(CURRENT_GW)))
        except ValueError:
            self.default_gameweek = CURRENT_GW

    def primary_season(self) -> str:
        return self.seasons[0] if self.seasons else self.default_season

    def season_list(self) -> List[str]:
        return self.seasons or [self.default_season]

    def primary_gameweek(self) -> int:
        return self.gameweeks[0] if self.gameweeks else self.default_gameweek

    def gameweek_list(self) -> List[int]:
        return self.gameweeks or [self.primary_gameweek()]

    def primary_stat(self) -> str:
        return self.stats[0] if self.stats else "total_points"

    def stat_list(self) -> List[str]:
        if self.stats:
            return list(dict.fromkeys(self.stats))
        return ["total_points"]

    def ranking_or(self, fallback: Optional[str] = None) -> str:
        value = (self.ranking or fallback or "best").lower()
        if value not in {"best", "worst"}:
            value = "best"
        return value

    def has_players(self, minimum: int = 1) -> bool:
        return len(self.players) >= minimum


def _query_has_keywords(text: str, keywords: List[str]) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(kw in lowered for kw in keywords)


def _mentions_all_seasons(text: str) -> bool:
    return _query_has_keywords(
        text,
        [
            "across all seasons",
            "across seasons",
            "all seasons",
            "overall",
            "ever",
            "career",
        ],
    )


def _select_player_query_key(ctx: QueryContext) -> Optional[str]:
    num_players = len(ctx.players)
    num_stats = len(ctx.stats)
    has_multiple_stats = num_stats > 1
    has_threshold = ctx.threshold is not None
    ranking = (ctx.ranking or "").lower() if ctx.ranking else None
    multi_seasons = len(ctx.seasons) > 1
    multi_gameweeks = len(ctx.gameweeks) > 1
    query_mentions_all_seasons = _mentions_all_seasons(ctx.query_text)

    if num_players >= 2:
        if has_multiple_stats:
            return "CompareMultiplePlayersByMultipleStats"
        if multi_seasons:
            return "ComparePlayersAcrossSeasons"
        if multi_gameweeks:
            return "ComparePlayersAcrossGameweeks"
        return "CompareMultiplePlayersBySingleStat"

    if num_players == 1:
        if multi_seasons:
            return "CompareSinglePlayerAcrossSeasons"
        if multi_gameweeks:
            return "GetPlayerPerformanceAcrossGameweeks"
        if ctx.gameweeks:
            return "GetPlayerPerformanceByGameweek"
        if ctx.seasons:
            return "GetPlayerSeasonPerformance"
        if query_mentions_all_seasons:
            return "GetPlayerCareerPerformance"
        return "GetPlayerSeasonPerformance"

    if has_threshold:
        return "RecommendPlayersByStatThreshold"

    if has_multiple_stats:
        if multi_gameweeks:
            return "GetPlayerMultipleStatsAcrossGameweeks"
        if multi_seasons or query_mentions_all_seasons:
            return "GetPlayerMultipleStatsAcrossSeasons"
        return "GetPlayerMultipleStatsBySeason"

    if ranking == "worst":
        return "GetWorstPlayersByStat"
    if ranking == "best":
        return "GetTopPlayersByStat"

    if multi_gameweeks:
        return "GetPlayerPerformanceAcrossGameweeks"
    if ctx.gameweeks:
        return "GetPlayerPerformanceByGameweek"
    if multi_seasons or query_mentions_all_seasons:
        return "GetPlayerPerformanceAcrossSeasons"

    return "GetPlayerSeasonPerformance"


def _select_recommendation_query_key(ctx: QueryContext) -> Optional[str]:
    text = ctx.query_text or ""
    if _query_has_keywords(text, ["bench", "drop", "avoid"]):
        return "RecommendPlayersToBench"
    if _query_has_keywords(text, ["next gw", "next gameweek", "upcoming", "future gw", "coming week"]):
        return "RecommendPlayersForNextGameweek"

    recommendation_keywords = [
        "recommend",
        "suggest",
        "should i",
        "who should i",
        "pick",
        "consider",
        "buy",
        "transfer",
        "targets",
        "options",
        "looking for",
    ]

    if not _query_has_keywords(text, recommendation_keywords):
        return _select_player_query_key(ctx)

    if ctx.threshold:
        return "RecommendPlayersByStatThreshold"
    if len(ctx.stats) > 1:
        return "RecommendPlayersByMultipleStats"
    return "RecommendPlayersByStat"


def _select_team_query_key(ctx: QueryContext) -> Optional[str]:
    num_teams = len(ctx.teams)
    multi_seasons = len(ctx.seasons) > 1
    multi_gameweeks = len(ctx.gameweeks) > 1

    if num_teams >= 2:
        if len(ctx.stats) > 1:
            return "CompareTeamsByMultipleStats"
        return "CompareTeamsBySingleStat"

    if num_teams == 1:
        if multi_gameweeks:
            return "GetTeamPerformanceAcrossGameweeks"
        if ctx.gameweeks:
            return "GetTeamPerformanceByGameweek"
        if multi_seasons:
            return "GetTeamPerformanceAcrossSeasons"
        if ctx.seasons:
            return "GetTeamPerformanceBySeason"
        return "GetTeamPerformanceBySeason"

    ranking = (ctx.ranking or "").lower() if ctx.ranking else None
    if ranking == "worst":
        return "GetWorstTeamsByStat"
    return "GetTopTeamsByStat"


def _select_fixture_query_key(ctx: QueryContext) -> Optional[str]:
    num_teams = len(ctx.teams)
    multi_gameweeks = len(ctx.gameweeks) > 1
    query = ctx.query_text.lower()

    if num_teams >= 2:
        if _query_has_keywords(query, ["compare", "versus", "vs", "head-to-head", "head to head"]):
            return "CompareFixtures"
        return "GetSpecificFixturesByTwoTeams"

    if num_teams == 1:
        if _query_has_keywords(query, ["next", "upcoming", "future"]):
            return "GetUpcomingFixturesForTeam"
        if _query_has_keywords(query, ["previous", "past", "last", "recent"]):
            return "GetPastFixturesForTeam"
        if multi_gameweeks:
            return "GetFixturesAcrossGameweeks"
        if ctx.gameweeks:
            return "GetFixturesByGameweek"
        if len(ctx.seasons) > 1:
            return "GetAllFixtureDetailsAcrossSeasons"
        return "GetFixturesByTeam"

    if multi_gameweeks:
        return "GetFixturesAcrossGameweeks"
    if ctx.gameweeks:
        return "GetFixturesByGameweek"
    if len(ctx.seasons) > 1:
        return "GetAllFixtureDetailsAcrossSeasons"
    return "GetAllFixtureDetailsBySeason"


def _select_team_winning_query_key(ctx: QueryContext) -> Optional[str]:
    if ctx.gameweeks:
        return "GetNumberOfWinsAndLosesAndDrawsForTeamTillCurrentGameweek"
    return "GetNumberOfWinsAndLosesAndDrawsForTeamBySeason"


def _select_comparison_query_key(ctx: QueryContext) -> Optional[str]:
    if len(ctx.players) >= 2:
        if len(ctx.stats) > 1:
            return "CompareMultiplePlayersByMultipleStats"
        return "CompareMultiplePlayersBySingleStat"
    if len(ctx.teams) >= 2:
        if len(ctx.stats) > 1:
            return "CompareTeamsByMultipleStats"
        return "CompareTeamsBySingleStat"
    return _select_player_query_key(ctx)


def _resolve_canonical_intent(ctx: QueryContext) -> Optional[str]:
    canonical = (ctx.intent or "").upper()
    if canonical == "PLAYER-RELATED":
        return _select_player_query_key(ctx)
    if canonical == "RECOMMENDATION":
        return _select_recommendation_query_key(ctx)
    if canonical == "TEAM-RELATED":
        return _select_team_query_key(ctx)
    if canonical == "FIXTURE-RELATED":
        return _select_fixture_query_key(ctx)
    if canonical == "TEAM WINNING":
        return _select_team_winning_query_key(ctx)
    if canonical == "COMPARISON":
        return _select_comparison_query_key(ctx)
    return None


@dataclass(frozen=True)
class PlayerQueryOptions:
    scope: str
    group_by: Tuple[str, ...]
    description: str
    multi_stats: bool = False
    combined_stats: bool = False
    require_players: bool = False
    allow_threshold: bool = False
    include_avg_form: bool = False
    limit: int = 40
    ranking_override: Optional[str] = None
    order_field: Optional[str] = None


def _player_opts(
    scope: str,
    group_by: Tuple[str, ...],
    description: str,
    **kwargs: Any,
) -> PlayerQueryOptions:
    return PlayerQueryOptions(scope=scope, group_by=group_by, description=description, **kwargs)


ALLOWED_THRESHOLD_OPERATORS = {">", ">=", "<", "<=", "=", "!="}

GROUP_FIELD_MAP = {
    "player": "p",
    "position": "pos",
    "season": "s",
    "gameweek": "gw",
}

RETURN_FIELD_MAP = {
    "player": "p.player_name AS player",
    "position": "COALESCE(pos.name, 'N/A') AS position",
    "season": "s.season_name AS season",
    "gameweek": "gw.GW_number AS gameweek",
}


def _scope_filters(scope: str, ctx: QueryContext, params: Dict[str, Any]) -> List[str]:
    filters: List[str] = []

    if scope == "gameweek":
        params["season"] = ctx.primary_season()
        params["gameweek"] = ctx.primary_gameweek()
        filters.append("s.season_name = $season")
        filters.append("gw.GW_number = $gameweek")
    elif scope == "gameweek_range":
        params["season"] = ctx.primary_season()
        params["gameweeks"] = ctx.gameweek_list()
        filters.append("s.season_name = $season")
        filters.append("gw.GW_number IN $gameweeks")
    elif scope == "season":
        params["season"] = ctx.primary_season()
        filters.append("s.season_name = $season")
    elif scope == "seasons":
        if ctx.seasons:
            params["seasons"] = ctx.season_list()
            filters.append("s.season_name IN $seasons")
    elif scope == "season_optional":
        if ctx.seasons:
            params["seasons"] = ctx.season_list()
            filters.append("s.season_name IN $seasons")
    elif scope == "gameweek_optional":
        if ctx.gameweeks:
            params["season"] = ctx.primary_season()
            params["gameweeks"] = ctx.gameweek_list()
            filters.append("s.season_name = $season")
            filters.append("gw.GW_number IN $gameweeks")
    elif scope == "career":
        pass

    return filters


def _build_threshold_clause(
    threshold: Optional[Dict[str, Any]],
    alias_map: Dict[str, str],
    param_name: str = "threshold_value",
) -> Tuple[Optional[str], Dict[str, Any]]:
    if not threshold:
        return None, {}

    stat_key = str(threshold.get("stat", ""))
    operator = str(threshold.get("operator", ">=")).strip()
    if operator not in ALLOWED_THRESHOLD_OPERATORS:
        operator = ">="

    try:
        value = float(threshold.get("value"))
    except (TypeError, ValueError):
        return None, {}

    alias = alias_map.get(stat_key)
    if not alias:
        alias = alias_map.get(_stat_alias(stat_key))
    if not alias:
        return None, {}

    return f"{alias} {operator} ${param_name}", {param_name: value}


def _order_direction(ranking: str) -> str:
    return "ASC" if ranking == "worst" else "DESC"


def _player_filter_clause(ctx: QueryContext, params: Dict[str, Any], player_var: str = "p") -> Optional[str]:
    condition = _build_player_name_condition(ctx.players, player_var)
    if condition == "true":
        return None
    if len(ctx.players) > 1:
        params["players"] = ctx.players
    else:
        params["player"] = ctx.players[0]
    return condition


def _positions_filter_clause(ctx: QueryContext, params: Dict[str, Any]) -> Optional[str]:
    """Return a position filter clause only when positions were provided."""
    if not ctx.positions:
        return None
    params["positions"] = ctx.positions
    return "pos.name IN $positions"


def _team_fixture_filter_clause(ctx: QueryContext, params: Dict[str, Any]) -> Optional[str]:
    if not ctx.teams:
        return None
    params["teams"] = ctx.teams
    return "(home.name IN $teams OR away.name IN $teams)"


def _multi_stat_aggregations(stats: List[str]) -> Tuple[List[str], Dict[str, str]]:
    aggregations: List[str] = []
    alias_map: Dict[str, str] = {}
    seen: set[str] = set()
    for raw_stat in stats:
        stat = str(raw_stat)
        if stat in seen:
            continue
        seen.add(stat)
        alias = _stat_alias(stat)
        aggregations.append(f"SUM(r.{stat}) AS {alias}")
        alias_map[stat] = alias
    return aggregations, alias_map


def _player_stat_query(ctx: QueryContext, opts: PlayerQueryOptions) -> QueryTuple:
    if opts.require_players and not ctx.has_players():
        raise ValueError("Player intent requires at least one recognized player entity.")

    params: Dict[str, Any] = {}
    filters = _scope_filters(opts.scope, ctx, params)

    where_clauses = list(filters)

    player_clause = _player_filter_clause(ctx, params)
    if player_clause:
        where_clauses.append(player_clause)

    # Position filter needs to be applied right after position MATCH, before OPTIONAL MATCH
    position_clause = _positions_filter_clause(ctx, params)
    
    team_clause = _team_fixture_filter_clause(ctx, params)
    if team_clause:
        where_clauses.append(team_clause)

    # Build position match with inline WHERE if position filter exists
    if ctx.positions:
        position_match_block = f"""MATCH (p)-[:PLAYS_AS]->(pos:Position)
    WHERE {position_clause}"""
    else:
        position_match_block = "OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)"

    match_block = f"""
    MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
    MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)
    MATCH (gw)<-[:HAS_GW]-(s:Season)
    {position_match_block}
    OPTIONAL MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
    OPTIONAL MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
    """

    # Build WHERE block for remaining clauses (not position)
    where_block = ""
    if where_clauses:
        where_block = "WHERE " + " AND ".join(where_clauses)

    group_vars: List[str] = []
    return_fields: List[str] = []
    for key in opts.group_by:
        var = GROUP_FIELD_MAP.get(key)
        column = RETURN_FIELD_MAP.get(key)
        if not var or not column:
            continue
        if var not in group_vars:
            group_vars.append(var)
        return_fields.append(column)

    if not group_vars:
        group_vars.append("p")
        return_fields.append(RETURN_FIELD_MAP["player"])
        return_fields.append(RETURN_FIELD_MAP["position"])

    aggregations: List[str] = []
    value_aliases: List[str] = []
    alias_map: Dict[str, str] = {}

    if opts.multi_stats:
        stats = ctx.stat_list()
        multi_aggs, multi_map = _multi_stat_aggregations(stats)
        aggregations.extend(multi_aggs)
        alias_map.update(multi_map)
        value_aliases.extend(multi_map.values())
    else:
        primary_stat = ctx.primary_stat()
        primary_alias = _stat_alias(primary_stat)
        aggregations.append(f"SUM({STAT_VALUE_CASE}) AS {primary_alias}")
        alias_map[primary_stat] = primary_alias
        value_aliases.append(primary_alias)
        params["stat"] = primary_stat

    if "total_points" not in value_aliases:
        aggregations.append("SUM(r.total_points) AS total_points")
        alias_map.setdefault("total_points", "total_points")
        value_aliases.append("total_points")

    if "minutes" not in value_aliases:
        aggregations.append("SUM(r.minutes) AS minutes")
        value_aliases.append("minutes")

    if opts.include_avg_form:
        aggregations.append("AVG(r.form) AS avg_form")
        alias_map.setdefault("form", "avg_form")
        value_aliases.append("avg_form")

    aggregations.append("COUNT(DISTINCT f) AS fixtures_played")
    value_aliases.append("fixtures_played")

    if opts.combined_stats:
        stats_for_combo = ctx.stat_list()
        combo_terms = [f"SUM(r.{stat})" for stat in stats_for_combo]
        if combo_terms:
            aggregations.append(f"({' + '.join(combo_terms)}) AS combined_metric")
            alias_map["combined_metric"] = "combined_metric"
            value_aliases.append("combined_metric")

    with_components = group_vars + aggregations
    with_clause = "WITH " + ",\n     ".join(with_components)

    threshold_clause = None
    threshold_params: Dict[str, Any] = {}
    if opts.allow_threshold:
        threshold_clause, threshold_params = _build_threshold_clause(ctx.threshold, alias_map)
        params.update(threshold_params)

    post_with_block = ""
    if threshold_clause:
        post_with_block = f"WHERE {threshold_clause}"

    for alias in value_aliases:
        if alias not in return_fields:
            return_fields.append(alias)

    order_field = opts.order_field
    if not order_field:
        if opts.multi_stats and alias_map:
            order_field = next(iter(alias_map.values()))
        else:
            order_field = value_aliases[0]

    ranking = _order_direction(ctx.ranking_or(opts.ranking_override))
    limit_clause = f"LIMIT {opts.limit}" if opts.limit > 0 else ""

    query = "\n".join(
        part
        for part in [
            match_block.strip(),
            where_block,
            with_clause,
            post_with_block,
            "RETURN " + ", ".join(return_fields),
            f"ORDER BY {order_field} {ranking}",
            limit_clause,
        ]
        if part
    )

    return query, params, opts.description


TEAM_STAT_EXPRESSIONS = {
    "goals_scored": ("SUM(goals_for)", "total_goals_for"),
    "goals_conceded": ("SUM(goals_against)", "total_goals_against"),
    "clean_sheets": ("SUM(CASE WHEN goals_against = 0 THEN 1 ELSE 0 END)", "total_clean_sheets"),
    "total_points": (
        "SUM(CASE WHEN goals_for > goals_against THEN 3 WHEN goals_for = goals_against THEN 1 ELSE 0 END)",
        "team_points_total",
    ),
    "wins": ("SUM(CASE WHEN goals_for > goals_against THEN 1 ELSE 0 END)", "wins"),
    "losses": ("SUM(CASE WHEN goals_for < goals_against THEN 1 ELSE 0 END)", "losses"),
    "draws": ("SUM(CASE WHEN goals_for = goals_against THEN 1 ELSE 0 END)", "draws"),
}


def _team_stat_projection(stat: str) -> Tuple[str, str]:
    stat_key = stat if stat in TEAM_STAT_EXPRESSIONS else "total_points"
    return TEAM_STAT_EXPRESSIONS.get(stat_key, TEAM_STAT_EXPRESSIONS["total_points"])


def _team_stat_query(
    ctx: QueryContext,
    *,
    scope: str,
    description: str,
    require_team: bool = True,
    multi_stats: bool = False,
    include_season: bool = False,
    ranking_override: Optional[str] = None,
    limit: int = 20,
) -> QueryTuple:
    if require_team and not ctx.teams:
        raise ValueError("Team intent requires at least one recognized club name.")

    params: Dict[str, Any] = {}
    filters = _scope_filters(scope, ctx, params)
    where_clauses = list(filters)

    if ctx.teams:
        params["teams"] = ctx.teams
        where_clauses.append(
            "ANY(teamName IN $teams WHERE toLower(team.name) CONTAINS toLower(teamName))"
        )

    match_block = """
    MATCH (s:Season)-[:HAS_GW]->(gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
    MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
    MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
    WITH s, gw, f,
         [{team: home, goals_for: COALESCE(f.team_h_score, 0), goals_against: COALESCE(f.team_a_score, 0)},
          {team: away, goals_for: COALESCE(f.team_a_score, 0), goals_against: COALESCE(f.team_h_score, 0)}] AS team_rows
    UNWIND team_rows AS row
    WITH s, gw, f, row.team AS team, row.goals_for AS goals_for, row.goals_against AS goals_against
    """

    where_block = ""
    if where_clauses:
        where_block = "WHERE " + " AND ".join(where_clauses)

    group_fields = ["team"]
    return_fields = ["team.name AS team"]
    if include_season:
        group_fields.append("s")
        return_fields.append("s.season_name AS season")

    stats_to_use = ctx.stat_list() if multi_stats else [ctx.primary_stat()]
    stats_to_use = list(dict.fromkeys(stats_to_use))
    if not stats_to_use:
        stats_to_use = ["total_points"]

    aggregations: List[str] = []
    value_aliases: List[str] = []
    primary_alias: Optional[str] = None

    for stat in stats_to_use:
        expr, alias = _team_stat_projection(stat)
        aggregations.append(f"{expr} AS {alias}")
        value_aliases.append(alias)
        if primary_alias is None:
            primary_alias = alias

    aggregations.append("COUNT(DISTINCT gw) AS matches_played")
    value_aliases.append("matches_played")

    with_clause = "WITH " + ", \n     ".join(group_fields + aggregations)

    for alias in value_aliases:
        if alias not in return_fields:
            return_fields.append(alias)

    order_field = primary_alias or value_aliases[0]
    ranking = _order_direction(ctx.ranking_or(ranking_override))
    limit_clause = f"LIMIT {limit}" if limit > 0 else ""

    query = "\n".join(
        part
        for part in [
            match_block.strip(),
            where_block,
            with_clause,
            "RETURN " + ", ".join(return_fields),
            f"ORDER BY {order_field} {ranking}",
            limit_clause,
        ]
        if part
    )

    return query, params, description


def _team_record_query(ctx: QueryContext, *, upto_current_gw: bool, description: str) -> QueryTuple:
    if not ctx.teams:
        raise ValueError("Team record intent requires a team name.")

    params: Dict[str, Any] = {
        "team": ctx.teams[0],
        "season": ctx.primary_season(),
    }
    filters = ["s.season_name = $season", "((f)-[:HAS_HOME_TEAM]->(t) OR (f)-[:HAS_AWAY_TEAM]->(t))"]

    if upto_current_gw:
        params["cutoff_gw"] = ctx.primary_gameweek()
        filters.append("gw.GW_number <= $cutoff_gw")

    where_clause = " AND ".join(filters)

    query = f"""
    MATCH (t:Team)
    WHERE toLower(t.name) CONTAINS toLower($team)
    MATCH (s:Season)-[:HAS_GW]->(gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
    WHERE {where_clause}
    OPTIONAL MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
    OPTIONAL MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
    WITH t, s, gw, f, home, away,
         CASE WHEN home.name = t.name THEN COALESCE(f.team_h_score, 0) ELSE COALESCE(f.team_a_score, 0) END AS goals_for,
         CASE WHEN home.name = t.name THEN COALESCE(f.team_a_score, 0) ELSE COALESCE(f.team_h_score, 0) END AS goals_against
    RETURN t.name AS team,
           s.season_name AS season,
           SUM(CASE WHEN goals_for > goals_against THEN 1 ELSE 0 END) AS wins,
           SUM(CASE WHEN goals_for = goals_against THEN 1 ELSE 0 END) AS draws,
           SUM(CASE WHEN goals_for < goals_against THEN 1 ELSE 0 END) AS losses,
           COUNT(DISTINCT gw) AS matches_played
    """

    return query, params, description


def _recommend_players_next_gw(ctx: QueryContext) -> QueryTuple:
    season = ctx.primary_season()
    target_gw = ctx.primary_gameweek()
    if not ctx.gameweeks:
        target_gw = min(ctx.default_gameweek + 1, 38)
    start_gw = max(1, target_gw - 3)
    if start_gw >= target_gw:
        start_gw = max(1, target_gw - 1)

    params = {
        "season": season,
        "start_gw": start_gw,
        "target_gw": target_gw,
    }

    where_clauses = ["gw.GW_number >= $start_gw", "gw.GW_number < $target_gw"]
    
    if ctx.positions:
        params["positions"] = ctx.positions
        where_clauses.append("pos.name IN $positions")

    where_block = "WHERE " + " AND ".join(where_clauses)

    query = f"""
    MATCH (s:Season {{season_name: $season}})-[:HAS_GW]->(gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
    MATCH (p:Player)-[r:PLAYED_IN]->(f)
    MATCH (p)-[:PLAYS_AS]->(pos:Position)
    {where_block}
    WITH p, pos,
         AVG(r.form) AS avg_form,
         SUM(r.total_points) AS total_points,
         MAX(gw.GW_number) AS latest_gw
    RETURN p.player_name AS player,
           COALESCE(pos.name, 'N/A') AS position,
           avg_form,
           total_points,
           latest_gw AS last_gameweek_considered
    ORDER BY avg_form DESC, total_points DESC
    LIMIT 20
    """

    desc = "Recommend players in form heading into the next gameweek based on the last three rounds."
    return query, params, desc


def _recommend_players_to_bench(ctx: QueryContext) -> QueryTuple:
    season = ctx.primary_season()
    params = {"season": season}

    where_clauses: List[str] = []
    
    if ctx.positions:
        params["positions"] = ctx.positions
        where_clauses.append("pos.name IN $positions")

    # Use MATCH instead of OPTIONAL MATCH when position filter is active
    position_match = "MATCH (p)-[:PLAYS_AS]->(pos:Position)" if ctx.positions else "OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)"
    where_block = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    query = f"""
    MATCH (s:Season {{season_name: $season}})-[:HAS_GW]->(gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
    MATCH (p:Player)-[r:PLAYED_IN]->(f)
    {position_match}
    {where_block}
    WITH p, pos,
         AVG(r.form) AS avg_form,
         AVG(r.minutes) AS avg_minutes,
         SUM(r.total_points) AS total_points,
         COUNT(DISTINCT gw) AS fixtures_played
    RETURN p.player_name AS player,
           COALESCE(pos.name, 'N/A') AS position,
           avg_form,
           avg_minutes,
           total_points,
           fixtures_played
    ORDER BY avg_form ASC, avg_minutes ASC, total_points ASC
    LIMIT 20
    """

    desc = "Flag players with low form and minutes who are strong candidates to bench."
    return query, params, desc


def _build_dynamic_fallback(ctx: QueryContext) -> QueryTuple:
    params: Dict[str, Any] = {"stat": ctx.primary_stat()}
    filters: List[str] = []

    if ctx.seasons:
        params["seasons"] = ctx.season_list()
        filters.append("s.season_name IN $seasons")
    if ctx.gameweeks:
        params["gws"] = ctx.gameweek_list()
        filters.append("gw.GW_number IN $gws")
    if ctx.teams:
        params["teams"] = ctx.teams
        filters.append("(home.name IN $teams OR away.name IN $teams)")

    player_clause = _player_filter_clause(ctx, params)
    if player_clause:
        filters.append(player_clause)

    # Position filter needs to be applied right after position MATCH, before OPTIONAL MATCH
    position_filter_inline = ""
    if ctx.positions:
        params["positions"] = ctx.positions
        position_filter_inline = "WHERE pos.name IN $positions"

    where_block = "WHERE " + " AND ".join(filters) if filters else ""

    alias_map = {
        ctx.primary_stat(): "stat_value",
        "goals_scored": "goals_scored",
        "assists": "assists",
        "clean_sheets": "clean_sheets",
        "saves": "saves",
        "total_points": "total_points",
        "form": "avg_form",
    }

    threshold_clause, threshold_params = _build_threshold_clause(ctx.threshold, alias_map)
    params.update(threshold_params)

    post_with = f"WHERE {threshold_clause}" if threshold_clause else ""

    ranking = _order_direction(ctx.ranking_or(None))

    # Build position match with inline WHERE if position filter exists
    if ctx.positions:
        position_match_block = f"""MATCH (p)-[:PLAYS_AS]->(pos:Position)
    {position_filter_inline}"""
    else:
        position_match_block = "OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)"

    query = f"""
    MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
    {position_match_block}
    OPTIONAL MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
    OPTIONAL MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
    OPTIONAL MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)
    OPTIONAL MATCH (gw)<-[:HAS_GW]-(s:Season)
    {where_block}
    WITH p, pos, s, gw, home, away,
         SUM(r.total_points) AS total_points,
         SUM(r.goals_scored) AS goals_scored,
         SUM(r.assists) AS assists,
         SUM(r.clean_sheets) AS clean_sheets,
         SUM(r.saves) AS saves,
         SUM({STAT_VALUE_CASE}) AS stat_value,
         AVG(r.form) AS avg_form,
         SUM(r.minutes) AS minutes
    {post_with}
    RETURN p.player_name AS player,
           COALESCE(pos.name, 'N/A') AS position,
           s.season_name AS season,
           gw.GW_number AS gameweek,
           home.name AS home_team,
           away.name AS away_team,
           stat_value,
           total_points,
           goals_scored,
           assists,
           clean_sheets,
           saves,
           minutes,
           avg_form
    ORDER BY stat_value {ranking}, total_points {ranking}
    LIMIT 40
    """

    desc = "Fallback player statistics view leveraging all available filters from preprocessing."
    return query, params, desc


PLAYER_QUERY_CONFIG: Dict[str, PlayerQueryOptions] = {
    "GetPlayerPerformanceByGameweek": _player_opts(
        scope="gameweek",
        group_by=("player", "position", "season", "gameweek"),
        description="Player performance for a specific gameweek.",
    ),
    "GetPlayerPerformanceAcrossGameweeks": _player_opts(
        scope="gameweek_range",
        group_by=("player", "position", "season", "gameweek"),
        description="Player performance across multiple gameweeks.",
        limit=200,
    ),
    "GetPlayerSeasonPerformance": _player_opts(
        scope="season",
        group_by=("player", "position", "season"),
        description="Aggregate a player's season performance.",
    ),
    "GetPlayerPerformanceAcrossSeasons": _player_opts(
        scope="seasons",
        group_by=("player", "position", "season"),
        description="Compare a player's performance across seasons.",
        limit=120,
    ),
    "GetPlayerMultipleStatsByGameweek": _player_opts(
        scope="gameweek",
        group_by=("player", "position", "season", "gameweek"),
        description="Multiple stats for each player in a gameweek.",
        multi_stats=True,
    ),
    "GetPlayerMultipleStatsBySeason": _player_opts(
        scope="season",
        group_by=("player", "position", "season"),
        description="Multiple stats for players across a season.",
        multi_stats=True,
    ),
    "GetPlayerMultipleStatsAcrossGameweeks": _player_opts(
        scope="gameweek_range",
        group_by=("player", "position", "season", "gameweek"),
        description="Multiple stats across several gameweeks.",
        multi_stats=True,
        limit=200,
    ),
    "GetPlayerMultipleStatsAcrossSeasons": _player_opts(
        scope="seasons",
        group_by=("player", "position", "season"),
        description="Multiple stats for players across seasons.",
        multi_stats=True,
        limit=120,
    ),
    "GetPlayerCombinedStats": _player_opts(
        scope="season_optional",
        group_by=("player", "position", "season"),
        description="Combined scoring metric derived from multiple stats.",
        multi_stats=True,
        combined_stats=True,
    ),
    "GetPlayerCareerPerformance": _player_opts(
        scope="career",
        group_by=("player", "position"),
        description="Career-long player aggregates across all seasons.",
        limit=150,
    ),
    "GetTopPlayersByStat": _player_opts(
        scope="season_optional",
        group_by=("player", "position"),
        description="League leaderboard for the requested stat.",
        ranking_override="best",
        limit=20,
    ),
    "GetWorstPlayersByStat": _player_opts(
        scope="season_optional",
        group_by=("player", "position"),
        description="Bottom performers for the requested stat.",
        ranking_override="worst",
        limit=20,
    ),
    "ComparePlayersBySingleStat": _player_opts(
        scope="season_optional",
        group_by=("player", "position", "season"),
        description="Compare named players by a single stat.",
        require_players=True,
        limit=100,
    ),
    "ComparePlayersByMultipleStats": _player_opts(
        scope="season_optional",
        group_by=("player", "position", "season"),
        description="Compare players across several stats.",
        multi_stats=True,
        require_players=True,
        limit=100,
    ),
    "ComparePlayersAcrossGameweeks": _player_opts(
        scope="gameweek_range",
        group_by=("player", "position", "season", "gameweek"),
        description="Compare players week-by-week over a range.",
        require_players=True,
        limit=250,
    ),
    "ComparePlayersAcrossSeasons": _player_opts(
        scope="seasons",
        group_by=("player", "position", "season"),
        description="Compare players across multiple seasons.",
        require_players=True,
        limit=150,
    ),
    "CompareMultiplePlayersBySingleStat": _player_opts(
        scope="season_optional",
        group_by=("player", "position", "season"),
        description="Multi-player comparison for a single stat.",
        require_players=True,
        limit=150,
    ),
    "CompareMultiplePlayersByMultipleStats": _player_opts(
        scope="season_optional",
        group_by=("player", "position", "season"),
        description="Multi-player comparison across several stats.",
        multi_stats=True,
        require_players=True,
        limit=150,
    ),
    "CompareSinglePlayerAcrossSeasons": _player_opts(
        scope="seasons",
        group_by=("player", "position", "season"),
        description="Single player trajectory across seasons.",
        require_players=True,
        limit=80,
    ),
    "CompareStatToPlayer": _player_opts(
        scope="season_optional",
        group_by=("player", "position", "season"),
        description="Compare other players to the referenced player's stat totals.",
        require_players=True,
        limit=120,
    ),
    "RecommendPlayersByStat": _player_opts(
        scope="season_optional",
        group_by=("player", "position", "season"),
        description="Recommend high-performing players for the requested stat.",
        include_avg_form=True,
        limit=30,
    ),
    "RecommendPlayersByStatThreshold": _player_opts(
        scope="season_optional",
        group_by=("player", "position", "season"),
        description="Players filtered by a stat threshold.",
        include_avg_form=True,
        allow_threshold=True,
        limit=50,
    ),
    "RecommendPlayersByMultipleStats": _player_opts(
        scope="season_optional",
        group_by=("player", "position", "season"),
        description="Recommend players using multiple requested statistics.",
        multi_stats=True,
        include_avg_form=True,
        limit=40,
    ),
}


INTENT_DISPATCH: Dict[str, Callable[[QueryContext], QueryTuple]] = {
    name: partial(_player_stat_query, opts=opts)
    for name, opts in PLAYER_QUERY_CONFIG.items()
}

INTENT_DISPATCH.update(
    {
        "GetTeamPerformanceByGameweek": lambda ctx: _team_fixture_details(
            ctx,
            scope="gameweek",
            description="Team fixtures and outcomes for a specific gameweek.",
        ),
        "GetTeamPerformanceBySeason": lambda ctx: _team_fixture_details(
            ctx,
            scope="season",
            description="All fixtures for the team across a season.",
        ),
        "GetTeamPerformanceAcrossGameweeks": lambda ctx: _team_fixture_details(
            ctx,
            scope="gameweek_range",
            description="Team fixtures filtered to selected gameweeks.",
        ),
        "GetTeamPerformanceAcrossSeasons": lambda ctx: _team_fixture_details(
            ctx,
            scope="seasons",
            description="Team fixture history across seasons.",
        ),
        "GetTeamMultipleStats": lambda ctx: _team_stat_query(
            ctx,
            scope="season",
            description="Team stat breakdown for the selected season.",
            multi_stats=True,
            include_season=True,
        ),
        "GetTeamSingleStats": lambda ctx: _team_stat_query(
            ctx,
            scope="season",
            description="Single team stat summary for the selected season.",
            include_season=True,
        ),
        "GetTopTeamsByStat": lambda ctx: _team_stat_query(
            ctx,
            scope="season_optional",
            description="League leaderboard for team statistics.",
            require_team=False,
            ranking_override="best",
            limit=20,
        ),
        "GetWorstTeamsByStat": lambda ctx: _team_stat_query(
            ctx,
            scope="season_optional",
            description="Lowest-ranked teams for the requested statistic.",
            require_team=False,
            ranking_override="worst",
            limit=20,
        ),
        "GetAllFixtureDetailsBySeason": lambda ctx: _fixture_list_query(
            ctx,
            scope="season",
            description="All fixtures and teams for the requested season.",
        ),
        "GetAllFixtureDetailsAcrossSeasons": lambda ctx: _fixture_list_query(
            ctx,
            scope="seasons",
            description="Fixture overview across multiple seasons.",
        ),
        "GetFixturesByGameweek": lambda ctx: _fixture_list_query(
            ctx,
            scope="gameweek",
            description="All fixtures scheduled in a specific gameweek.",
        ),
        "GetFixturesAcrossGameweeks": lambda ctx: _fixture_list_query(
            ctx,
            scope="gameweek_range",
            description="Fixtures across multiple specified gameweeks.",
        ),
        "GetFixturesByTeam": lambda ctx: _fixture_list_query(
            ctx,
            scope="season_optional",
            description="Team-specific fixtures.",
            require_team=True,
        ),
        "GetSpecificFixturesByTwoTeams": lambda ctx: _fixture_list_query(
            ctx,
            scope="season_optional",
            description="Head-to-head fixtures between two clubs.",
            dual_team=True,
        ),
        "GetUpcomingFixturesForTeam": lambda ctx: _fixture_list_query(
            ctx,
            scope="season_optional",
            description="Upcoming fixtures for the specified team.",
            require_team=True,
            upcoming=True,
            limit=20,
        ),
        "GetPastFixturesForTeam": lambda ctx: _fixture_list_query(
            ctx,
            scope="season_optional",
            description="Recently completed fixtures for the specified team.",
            require_team=True,
            past=True,
            limit=30,
        ),
        "GetNumberOfWinsAndLosesAndDrawsForTeamBySeason": lambda ctx: _team_record_query(
            ctx,
            upto_current_gw=False,
            description="Season win/draw/loss breakdown for a team.",
        ),
        "GetNumberOfWinsAndLosesAndDrawsForTeamTillCurrentGameweek": lambda ctx: _team_record_query(
            ctx,
            upto_current_gw=True,
            description="Team record up to the referenced gameweek.",
        ),
        "RecommendPlayersForNextGameweek": _recommend_players_next_gw,
        "RecommendPlayersToBench": _recommend_players_to_bench,
        "CompareFixtures": lambda ctx: _fixture_list_query(
            ctx,
            scope="season_optional",
            description="Compare fixtures between two clubs.",
            dual_team=True,
        ),
        "CompareTeamsBySingleStat": lambda ctx: _team_stat_query(
            ctx,
            scope="season_optional",
            description="Compare named teams by a single stat.",
            include_season=True,
        ),
        "CompareTeamsByMultipleStats": lambda ctx: _team_stat_query(
            ctx,
            scope="season_optional",
            description="Compare teams across multiple stats.",
            multi_stats=True,
            include_season=True,
        ),
        "CompareTeamsAcrossSeasons": lambda ctx: _team_stat_query(
            ctx,
            scope="seasons",
            description="Compare team stats season by season.",
            include_season=True,
        ),
        "CompareSingleTeamAcrossSeasons": lambda ctx: _team_stat_query(
            ctx,
            scope="seasons",
            description="Single team stat trend across seasons.",
            include_season=True,
        ),
        "CompareTeamsByGameweek": lambda ctx: _team_stat_query(
            ctx,
            scope="gameweek_range",
            description="Team comparison limited to selected gameweeks.",
            include_season=True,
        ),
    }
)


def _team_fixture_details(ctx: QueryContext, *, scope: str, description: str) -> QueryTuple:
    if not ctx.teams:
        raise ValueError("Team fixture intent requires at least one team name.")

    params: Dict[str, Any] = {"team": ctx.teams[0]}
    filters = _scope_filters(scope, ctx, params)

    filter_parts = ["((f)-[:HAS_HOME_TEAM]->(t) OR (f)-[:HAS_AWAY_TEAM]->(t))"] + filters
    where_clause = " AND ".join(filter_parts) if filter_parts else "true"

    query = f"""
    MATCH (t:Team)
    WHERE toLower(t.name) CONTAINS toLower($team)
    MATCH (s:Season)-[:HAS_GW]->(gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
    WHERE {where_clause}
    OPTIONAL MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
    OPTIONAL MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
    WITH t, s, gw, f, home, away,
         CASE WHEN home.name = t.name THEN away.name ELSE home.name END AS opponent,
         COALESCE(f.team_h_score, 0) AS home_score,
         COALESCE(f.team_a_score, 0) AS away_score
    RETURN s.season_name AS season,
           gw.GW_number AS gameweek,
           f.fixture_number AS fixture_number,
           home.name AS home_team,
           away.name AS away_team,
           t.name AS focus_team,
           opponent AS opponent,
           home_score,
           away_score,
           f.kickoff_time AS kickoff_time
    ORDER BY season, gameweek, fixture_number
    """

    return query, params, description


def _fixture_list_query(
    ctx: QueryContext,
    *,
    scope: str,
    description: str,
    require_team: bool = False,
    dual_team: bool = False,
    upcoming: bool = False,
    past: bool = False,
    limit: int = 0,
) -> QueryTuple:
    if require_team and not ctx.teams:
        raise ValueError("This fixture intent requires at least one team name.")
    if dual_team and len(ctx.teams) < 2:
        raise ValueError("Head-to-head fixture intents require two teams.")

    params: Dict[str, Any] = {}
    filters = _scope_filters(scope, ctx, params)
    where_clauses = list(filters)

    if dual_team:
        team_a, team_b = ctx.teams[:2]
        params["team_a"] = team_a
        params["team_b"] = team_b
        where_clauses.append(
            "((home.name CONTAINS $team_a AND away.name CONTAINS $team_b) OR (home.name CONTAINS $team_b AND away.name CONTAINS $team_a))"
        )
    elif ctx.teams:
        params["teams"] = ctx.teams
        where_clauses.append("(home.name IN $teams OR away.name IN $teams)")

    if upcoming:
        where_clauses.append("datetime(f.kickoff_time) >= datetime()")
    if past:
        where_clauses.append("datetime(f.kickoff_time) < datetime()")

    where_block = ""
    if where_clauses:
        where_block = "WHERE " + " AND ".join(where_clauses)

    query = """
    MATCH (s:Season)-[:HAS_GW]->(gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
    OPTIONAL MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
    OPTIONAL MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
    {where_block}
    RETURN s.season_name AS season,
           gw.GW_number AS gameweek,
           f.fixture_number AS fixture_number,
           home.name AS home_team,
           away.name AS away_team,
           f.kickoff_time AS kickoff_time
    ORDER BY season, gameweek, fixture_number
    {limit_clause}
    """.format(
        where_block=where_block,
        limit_clause=f"LIMIT {limit}" if limit > 0 else "",
    )

    return query, params, description


def build_baseline_query(
    intent: str,
    entities: Dict[str, List[Any]],
    query: str = "",
    ranking: Optional[str] = None,
    threshold: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any], str]:
    """Resolve the Cypher template for the supplied intent/entities."""

    safe_entities: Dict[str, List[Any]] = {}
    for key, value in (entities or {}).items():
        if isinstance(value, list):
            safe_entities[key] = list(value)
        elif value is None:
            safe_entities[key] = []
        else:
            safe_entities[key] = [value]
    ctx = QueryContext(
        intent=intent,
        entities=safe_entities,
        query_text=query,
        ranking=ranking,
        threshold=threshold,
    )

    handler = INTENT_DISPATCH.get(intent)
    effective_intent = intent

    if handler is None:
        resolved_intent = _resolve_canonical_intent(ctx)
        if resolved_intent:
            handler = INTENT_DISPATCH.get(resolved_intent)
            if handler:
                logger.info(
                    f"Canonical intent '{ctx.intent}' mapped to handler '{resolved_intent}'."
                )
                effective_intent = resolved_intent

    if handler is not None:
        try:
            ctx.intent = effective_intent
            return handler(ctx)
        except ValueError as exc:
            logger.warning(f"{intent} handler requirements not met: {exc}. Falling back.")

    logger.warning(f"No handler registered for intent='{intent}'. Using fallback query.")
    return _build_dynamic_fallback(ctx)


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

    def run_from_intent_entities(
        self,
        intent: str,
        entities: Dict[str, List[Any]],
        query: str = "",
        ranking: Optional[str] = None,
        threshold: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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
        cypher, params, desc = build_baseline_query(intent, entities, query, ranking, threshold)
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
        # Pass the original query plus ranking/threshold metadata
        return self.run_from_intent_entities(
            pre["intent"],
            pre["entities"],
            user_text,
            ranking=pre.get("ranking"),
            threshold=pre.get("threshold"),
        )


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
        "Who are the best attacking players in terms of goals and threat combined?",
        "Which players scored more than 5 goals up till this gameweek?",
        "which players have the least points so far?",
        "Show me Brightin's fixtures in 2 gameweeks from now.",
        "Which Tottenham defenders have been in good form this season?",
        "Give me players with same number of goals as Haaland.",
        "i want players with 0 cleansheets",
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
