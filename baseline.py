"""
baseline.py - Baseline Cypher query system for FPL Knowledge Graph
Connects to Neo4j and executes basic queries based on preprocessing output.
"""

from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional
import json



# ============================================================
# 1. CONFIGURATION
# ============================================================

def load_config() -> Dict[str, str]:
    """Load Neo4j credentials from config.txt"""
    config: Dict[str, str] = {}
    try:
        with open("config.txt", "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    config[k.strip()] = v.strip()
    except FileNotFoundError:
        print("⚠ config.txt not found. Cannot connect to Neo4j.")
    return config


# ============================================================
# 2. NEO4J CONNECTION
# ============================================================

class Neo4jConnection:
    """Handles Neo4j database connection"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        if self.driver:
            self.driver.close()
    
    def execute_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Execute a Cypher query and return results"""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]


# ============================================================
# 3. CYPHER QUERY BUILDER
# ============================================================

class BaselineQueryBuilder:
    """Builds Cypher queries based on preprocessing output"""
    
    def __init__(self, connection: Neo4jConnection):
        self.conn = connection
    
    def query_top_players_by_statistic(
        self, 
        entities: Dict[str, List], 
        ranking: Optional[str] = None,
        threshold: Optional[Dict] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Query 1: Get top players by a single statistic
        
        Expected input from preprocessing:
        - entities: {"Statistic": ["<any_single_stat>"]} - ONLY ONE statistic
        - ranking: MUST be "best"
        - threshold: Optional {"stat": "<stat_name>", "operator": ">", "value": N}
        - Other entities (Player, Team, Position, Season, Gameweek) MUST be empty
        
        Returns: List of players with their total for that statistic, or empty if conditions not met
        """
        
        # Get statistics
        statistics = entities.get("Statistic", [])
        
        # Check: Must have exactly ONE statistic
        if len(statistics) != 1:
            print("⚠ Query 1 requires exactly ONE statistic.")
            return []
        
        # Check: No other entities allowed
        if (entities.get("Player", []) or 
            entities.get("Team", []) or 
            entities.get("Position", []) or 
            entities.get("Season", []) or 
            entities.get("Gameweek", [])):
            print("⚠ Query 1 only works with a single statistic.")
            print("   Other entities (Team, Player, Position, Season, Gameweek) must be empty.")
            return []
        
        # Check: Ranking must be "best"
        if ranking != "best":
            print("⚠ Query 1 only works with ranking='best'.")
            return []
        
        # Get the statistic name
        stat_name = statistics[0]
        
        # Get current season from config
        config = load_config()
        current_season = config.get("CURRENT_SEASON", "2022-23")
        
        # Build the base query dynamically with season filter
        cypher = f"""
        MATCH (s:Season {{season_name: $current_season}})-[:HAS_GW]->(gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
        MATCH (p:Player)-[r:PLAYED_IN]->(f)
        OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
        WITH p, pos, SUM(r.{stat_name}) AS total_stat
        WHERE total_stat > 0
        """
        
        # Add threshold filter if provided
        params = {"current_season": current_season}
        if threshold and threshold.get("stat") == stat_name:
            operator = threshold.get("operator", ">")
            value = threshold.get("value", 0)
            params["threshold_value"] = value
            
            # Handle both symbol and text operators
            if operator in [">", "GT"]:
                cypher += "AND total_stat > $threshold_value\n"
            elif operator in [">=", "GE"]:
                cypher += "AND total_stat >= $threshold_value\n"
            elif operator in ["<", "LT"]:
                cypher += "AND total_stat < $threshold_value\n"
            elif operator in ["<=", "LE"]:
                cypher += "AND total_stat <= $threshold_value\n"
            elif operator in ["=", "EQ"]:
                cypher += "AND total_stat = $threshold_value\n"
        
        # Add ordering and limit (always DESC for "best")
        cypher += f"""
        RETURN p.player_name AS player_name, pos.name AS position, total_stat
        ORDER BY total_stat DESC
        LIMIT {limit}
        """
        
        print(f"\n📊 Executing Query 1: Top players by {stat_name} (best) in season {current_season}")
        print(f"Cypher Query:\n{cypher}")
        if params:
            print(f"Parameters: {params}")
        
        results = self.conn.execute_query(cypher, params)
        
        # Format output
        print(f"\n✅ Results ({len(results)} players):")
        for i, record in enumerate(results, 1):
            position = record.get('position') or 'Unknown position'
            print(f"{i}. {record['player_name']} ({position}): {record['total_stat']} {stat_name}")
        
        return results
    
    def query_worst_players_by_statistic(
        self, 
        entities: Dict[str, List], 
        ranking: Optional[str] = None,
        threshold: Optional[Dict] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Query 2: Get worst players by a single statistic
        
        Expected input from preprocessing:
        - entities: {"Statistic": ["<any_single_stat>"]} - ONLY ONE statistic
        - ranking: MUST be "worst"
        - threshold: Optional {"stat": "<stat_name>", "operator": ">", "value": N}
        - Other entities (Player, Team, Position, Season, Gameweek) MUST be empty
        
        Returns: List of players with their total for that statistic, or empty if conditions not met
        """
        
        # Get statistics
        statistics = entities.get("Statistic", [])
        
        # Check: Must have exactly ONE statistic
        if len(statistics) != 1:
            print("⚠ Query 2 requires exactly ONE statistic.")
            return []
        
        # Check: No other entities allowed
        if (entities.get("Player", []) or 
            entities.get("Team", []) or 
            entities.get("Position", []) or 
            entities.get("Season", []) or 
            entities.get("Gameweek", [])):
            print("⚠ Query 2 only works with a single statistic.")
            print("   Other entities (Team, Player, Position, Season, Gameweek) must be empty.")
            return []
        
        # Check: Ranking must be "worst"
        if ranking != "worst":
            print("⚠ Query 2 only works with ranking='worst'.")
            return []
        
        # Get the statistic name
        stat_name = statistics[0]
        
        # Get current season from config
        config = load_config()
        current_season = config.get("CURRENT_SEASON", "2022-23")
        
        # Build the base query dynamically with season filter
        cypher = f"""
        MATCH (s:Season {{season_name: $current_season}})-[:HAS_GW]->(gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
        MATCH (p:Player)-[r:PLAYED_IN]->(f)
        OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
        WITH p, pos, SUM(r.{stat_name}) AS total_stat
        WHERE total_stat > 0
        """
        
        # Add threshold filter if provided
        params = {"current_season": current_season}
        if threshold and threshold.get("stat") == stat_name:
            operator = threshold.get("operator", ">")
            value = threshold.get("value", 0)
            params["threshold_value"] = value
            
            # Handle both symbol and text operators
            if operator in [">", "GT"]:
                cypher += "AND total_stat > $threshold_value\n"
            elif operator in [">=", "GE"]:
                cypher += "AND total_stat >= $threshold_value\n"
            elif operator in ["<", "LT"]:
                cypher += "AND total_stat < $threshold_value\n"
            elif operator in ["<=", "LE"]:
                cypher += "AND total_stat <= $threshold_value\n"
            elif operator in ["=", "EQ"]:
                cypher += "AND total_stat = $threshold_value\n"
        
        # Add ordering and limit (ASC for "worst")
        cypher += f"""
        RETURN p.player_name AS player_name, pos.name AS position, total_stat
        ORDER BY total_stat ASC
        LIMIT {limit}
        """
        
        print(f"\n📊 Executing Query 2: Worst players by {stat_name} (worst) in season {current_season}")
        print(f"Cypher Query:\n{cypher}")
        if params:
            print(f"Parameters: {params}")
        
        results = self.conn.execute_query(cypher, params)
        
        # Format output
        print(f"\n✅ Results ({len(results)} players):")
        for i, record in enumerate(results, 1):
            position = record.get('position') or 'Unknown position'
            print(f"{i}. {record['player_name']} ({position}): {record['total_stat']} {stat_name}")
        
        return results
    
    def query_top_players_by_stat_and_position(
        self, 
        entities: Dict[str, List], 
        ranking: Optional[str] = None,
        threshold: Optional[Dict] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Query 3: Get top players by statistic filtered by position
        
        Expected input from preprocessing:
        - entities: {"Statistic": ["<stat>"], "Position": ["<position>"]}
        - ranking: MUST be "best"
        - Other entities (Player, Team, Season, Gameweek) MUST be empty
        """
        
        statistics = entities.get("Statistic", [])
        positions = entities.get("Position", [])
        
        # Check: Must have exactly ONE statistic and ONE position
        if len(statistics) != 1 or len(positions) != 1:
            print("⚠ Query 3 requires exactly ONE statistic and ONE position.")
            return []
        
        # Check: No other entities allowed
        if (entities.get("Player", []) or 
            entities.get("Team", []) or 
            entities.get("Season", []) or 
            entities.get("Gameweek", [])):
            print("⚠ Query 3 only works with statistic + position.")
            print("   Other entities must be empty.")
            return []
        
        # Check: Ranking must be "best"
        if ranking != "best":
            print("⚠ Query 3 only works with ranking='best'.")
            return []
        
        stat_name = statistics[0]
        position = positions[0]
        
        # Get current season from config
        config = load_config()
        current_season = config.get("CURRENT_SEASON", "2022-23")
        
        cypher = f"""
        MATCH (s:Season {{season_name: $current_season}})-[:HAS_GW]->(gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
        MATCH (p:Player)-[:PLAYS_AS]->(pos:Position {{name: $position}})
        MATCH (p)-[r:PLAYED_IN]->(f)
        WITH p, pos, SUM(r.{stat_name}) AS total_stat
        WHERE total_stat > 0
        """
        
        params = {"position": position, "current_season": current_season}
        
        # Add threshold filter if provided
        if threshold and threshold.get("stat") == stat_name:
            operator = threshold.get("operator", ">")
            value = threshold.get("value", 0)
            params["threshold_value"] = value
            
            if operator in [">", "GT"]:
                cypher += "AND total_stat > $threshold_value\n"
            elif operator in [">=", "GE"]:
                cypher += "AND total_stat >= $threshold_value\n"
            elif operator in ["<", "LT"]:
                cypher += "AND total_stat < $threshold_value\n"
            elif operator in ["<=", "LE"]:
                cypher += "AND total_stat <= $threshold_value\n"
            elif operator in ["=", "EQ"]:
                cypher += "AND total_stat = $threshold_value\n"
        
        cypher += f"""
        RETURN p.player_name AS player_name, pos.name AS position, total_stat
        ORDER BY total_stat DESC
        LIMIT {limit}
        """
        
        print(f"\n📊 Executing Query 3: Top {position} players by {stat_name} in season {current_season}")
        print(f"Cypher Query:\n{cypher}")
        print(f"Parameters: {params}")
        
        results = self.conn.execute_query(cypher, params)
        
        print(f"\n✅ Results ({len(results)} players):")
        for i, record in enumerate(results, 1):
            position = record.get('position') or 'Unknown position'
            print(f"{i}. {record['player_name']} ({position}): {record['total_stat']} {stat_name}")
        
        return results
    
    def query_worst_players_by_stat_and_position(
        self, 
        entities: Dict[str, List], 
        ranking: Optional[str] = None,
        threshold: Optional[Dict] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Query 4: Get worst players by statistic filtered by position
        
        Expected input from preprocessing:
        - entities: {"Statistic": ["<stat>"], "Position": ["<position>"]}
        - ranking: MUST be "worst"
        - Other entities (Player, Team, Season, Gameweek) MUST be empty
        """
        
        statistics = entities.get("Statistic", [])
        positions = entities.get("Position", [])
        
        # Check: Must have exactly ONE statistic and ONE position
        if len(statistics) != 1 or len(positions) != 1:
            print("⚠ Query 4 requires exactly ONE statistic and ONE position.")
            return []
        
        # Check: No other entities allowed
        if (entities.get("Player", []) or 
            entities.get("Team", []) or 
            entities.get("Season", []) or 
            entities.get("Gameweek", [])):
            print("⚠ Query 4 only works with statistic + position.")
            print("   Other entities must be empty.")
            return []
        
        # Check: Ranking must be "worst"
        if ranking != "worst":
            print("⚠ Query 4 only works with ranking='worst'.")
            return []
        
        stat_name = statistics[0]
        position = positions[0]
        
        # Get current season from config
        config = load_config()
        current_season = config.get("CURRENT_SEASON", "2022-23")
        
        cypher = f"""
        MATCH (s:Season {{season_name: $current_season}})-[:HAS_GW]->(gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
        MATCH (p:Player)-[:PLAYS_AS]->(pos:Position {{name: $position}})
        MATCH (p)-[r:PLAYED_IN]->(f)
        WITH p, pos, SUM(r.{stat_name}) AS total_stat
        WHERE total_stat > 0
        """
        
        params = {"position": position, "current_season": current_season}
        
        # Add threshold filter if provided
        if threshold and threshold.get("stat") == stat_name:
            operator = threshold.get("operator", ">")
            value = threshold.get("value", 0)
            params["threshold_value"] = value
            
            if operator in [">", "GT"]:
                cypher += "AND total_stat > $threshold_value\n"
            elif operator in [">=", "GE"]:
                cypher += "AND total_stat >= $threshold_value\n"
            elif operator in ["<", "LT"]:
                cypher += "AND total_stat < $threshold_value\n"
            elif operator in ["<=", "LE"]:
                cypher += "AND total_stat <= $threshold_value\n"
            elif operator in ["=", "EQ"]:
                cypher += "AND total_stat = $threshold_value\n"
        
        cypher += f"""
        RETURN p.player_name AS player_name, pos.name AS position, total_stat
        ORDER BY total_stat ASC
        LIMIT {limit}
        """
        
        print(f"\n📊 Executing Query 4: Worst {position} players by {stat_name} (current season)")
        print(f"Cypher Query:\n{cypher}")
        print(f"Parameters: {params}")
        
        results = self.conn.execute_query(cypher, params)
        
        print(f"\n✅ Results ({len(results)} players):")
        for i, record in enumerate(results, 1):
            position = record.get('position') or 'Unknown position'
            print(f"{i}. {record['player_name']} ({position}): {record['total_stat']} {stat_name}")
        
        return results
    
    def query_player_performance(
        self, 
        entities: Dict[str, List], 
        ranking: Optional[str] = None,
        threshold: Optional[Dict] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Query 8: Get all statistics for a specific player
        
        Expected input from preprocessing:
        - entities: {"Player": ["<player_name>"]}
        - No statistic required
        - Other entities (Team, Position, Season, Gameweek) can be empty
        """
        
        players = entities.get("Player", [])
        statistics = entities.get("Statistic", [])
        
        # Check: Must have exactly ONE player and NO statistics
        if len(players) != 1:
            print("⚠ Query 8 requires exactly ONE player.")
            return []
        
        if statistics:
            print("⚠ Query 8 should not have statistics specified.")
            return []
        
        player_name = players[0]
        
        cypher = """
        MATCH (p:Player)
        WHERE p.player_name CONTAINS $player_name
        OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
        RETURN 
            p.player_name AS player_name,
            pos.name AS position,
            SUM(r.goals_scored) AS total_goals,
            SUM(r.assists) AS total_assists,
            SUM(r.total_points) AS total_points,
            SUM(r.minutes) AS total_minutes,
            SUM(r.clean_sheets) AS total_clean_sheets,
            SUM(r.goals_conceded) AS total_goals_conceded,
            SUM(r.saves) AS total_saves,
            SUM(r.yellow_cards) AS total_yellow_cards,
            SUM(r.red_cards) AS total_red_cards,
            COUNT(f) AS matches_played
        """
        
        params = {"player_name": player_name}
        
        print(f"\n📊 Executing Query 8: Performance stats for players containing '{player_name}'")
        print(f"Cypher Query:\n{cypher}")
        print(f"Parameters: {params}")
        
        results = self.conn.execute_query(cypher, params)
        
        if results:
            for r in results:
                position = r.get('position') or 'Unknown position'
                print(f"\n✅ Results for {r['player_name']} ({position}):")
                print(f"  Matches Played: {r['matches_played']}")
                print(f"  Goals: {r['total_goals']}")
                print(f"  Assists: {r['total_assists']}")
                print(f"  Total Points: {r['total_points']}")
                print(f"  Minutes: {r['total_minutes']}")
                print(f"  Clean Sheets: {r['total_clean_sheets']}")
                print(f"  Goals Conceded: {r['total_goals_conceded']}")
                print(f"  Saves: {r['total_saves']}")
                print(f"  Yellow Cards: {r['total_yellow_cards']}")
                print(f"  Red Cards: {r['total_red_cards']}")
        else:
            print(f"\n⚠ No data found for player containing: {player_name}")
        
        return results
    
    def query_team_fixtures(
        self, 
        entities: Dict[str, List], 
        ranking: Optional[str] = None,
        threshold: Optional[Dict] = None,
        limit: int = 20
    ) -> List[Dict]:
        """
        Query 9: Get all fixtures for a specific team
        
        Expected input from preprocessing:
        - entities: {"Team": ["<team_name>"]}
        - No statistic, no player
        """
        
        teams = entities.get("Team", [])
        players = entities.get("Player", [])
        statistics = entities.get("Statistic", [])
        
        # Check: Must have exactly ONE team, NO players, NO statistics
        if len(teams) != 1:
            print("⚠ Query 9 requires exactly ONE team.")
            return []
        
        if players or statistics:
            print("⚠ Query 9 should not have players or statistics specified.")
            return []
        
        team_name = teams[0]
        
        # Get current season from config
        config = load_config()
        current_season = config.get("CURRENT_SEASON", "2022-23")
        
        cypher = """
        MATCH (s:Season {season_name: $current_season})-[:HAS_GW]->(gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
        MATCH (t:Team {name: $team_name})
        WHERE (f)-[:HAS_HOME_TEAM]->(t) OR (f)-[:HAS_AWAY_TEAM]->(t)
        MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
        MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
        RETURN 
            s.season_name AS season,
            gw.GW_number AS gameweek,
            f.fixture_number AS fixture_id,
            home.name AS home_team,
            away.name AS away_team,
            f.kickoff_time AS kickoff_time
        ORDER BY gw.GW_number
        LIMIT $limit
        """
        
        params = {"team_name": team_name, "limit": limit, "current_season": current_season}
        
        print(f"\n📊 Executing Query 9: Fixtures for {team_name} in season {current_season}")
        print(f"Cypher Query:\n{cypher}")
        print(f"Parameters: {params}")
        
        results = self.conn.execute_query(cypher, params)
        
        print(f"\n✅ Results ({len(results)} fixtures):")
        for i, record in enumerate(results, 1):
            print(f"{i}. GW{record['gameweek']} ({record['season']}): {record['home_team']} vs {record['away_team']}")
        
        return results
    
    def query_gameweek_top_performers(
        self, 
        entities: Dict[str, List], 
        ranking: Optional[str] = None,
        threshold: Optional[Dict] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Query 10: Get top players in a specific gameweek by a statistic
        
        Expected input from preprocessing:
        - entities: {"Gameweek": [<number>], "Statistic": ["<stat>"]}
        - ranking: MUST be "best"
        - Other entities (Player, Team, Position, Season) MUST be empty
        """
        
        gameweeks = entities.get("Gameweek", [])
        statistics = entities.get("Statistic", [])
        
        # Check: Must have exactly ONE gameweek and ONE statistic
        if len(gameweeks) != 1 or len(statistics) != 1:
            print("⚠ Query 10 requires exactly ONE gameweek and ONE statistic.")
            return []
        
        # Check: No other entities allowed
        if (entities.get("Player", []) or 
            entities.get("Team", []) or 
            entities.get("Position", []) or 
            entities.get("Season", [])):
            print("⚠ Query 10 only works with gameweek + statistic.")
            print("   Other entities must be empty.")
            return []
        
        # Check: Ranking must be "best"
        if ranking != "best":
            print("⚠ Query 10 only works with ranking='best'.")
            return []
        
        gameweek = gameweeks[0]
        stat_name = statistics[0]
        
        # Get current season from config
        config = load_config()
        current_season = config.get("CURRENT_SEASON", "2022-23")
        
        cypher = f"""
        MATCH (s:Season {{season_name: $current_season}})-[:HAS_GW]->(gw:Gameweek {{GW_number: $gameweek}})-[:HAS_FIXTURE]->(f:Fixture)
        MATCH (p:Player)-[r:PLAYED_IN]->(f)
        WHERE r.{stat_name} > 0
        OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
        RETURN 
            p.player_name AS player_name,
            pos.name AS position,
            r.{stat_name} AS stat_value
        ORDER BY stat_value DESC
        LIMIT {limit}
        """
        
        params = {"gameweek": gameweek, "current_season": current_season}
        
        # Add threshold filter if provided
        if threshold and threshold.get("stat") == stat_name:
            operator = threshold.get("operator", ">")
            value = threshold.get("value", 0)
            params["threshold_value"] = value
            
            # Update WHERE clause in cypher
            cypher = cypher.replace(
                f"WHERE r.{stat_name} > 0",
                f"WHERE r.{stat_name} > 0 AND r.{stat_name} {self._get_operator_symbol(operator)} $threshold_value"
            )
        
        print(f"\n📊 Executing Query 10: Top players in GW{gameweek} by {stat_name} (season {current_season})")
        print(f"Cypher Query:\n{cypher}")
        print(f"Parameters: {params}")
        
        results = self.conn.execute_query(cypher, params)
        
        print(f"\n✅ Results ({len(results)} players):")
        for i, record in enumerate(results, 1):
            position = record.get('position') or 'Unknown position'
            print(f"{i}. {record['player_name']} ({position}): {record['stat_value']} {stat_name}")
        
        return results
    
    def _get_operator_symbol(self, operator: str) -> str:
        """Helper to convert operator text to symbol"""
        mapping = {
            "GT": ">", ">": ">",
            "GE": ">=", ">=": ">=",
            "LT": "<", "<": "<",
            "LE": "<=", "<=": "<=",
            "EQ": "=", "=": "="
        }
        return mapping.get(operator, ">")
    
    def query_compare_two_players(
        self, 
        entities: Dict[str, List], 
        ranking: Optional[str] = None,
        threshold: Optional[Dict] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Query 16: Compare statistics of two players
        
        Expected input from preprocessing:
        - entities: {"Player": ["player1", "player2"]} - Exactly 2 players
        - No statistics, no season (auto filter by current season)
        """
        
        players = entities.get("Player", [])
        statistics = entities.get("Statistic", [])
        seasons = entities.get("Season", [])
        
        # Check: Must have exactly TWO players
        if len(players) != 2:
            print("⚠ Query 16 requires exactly TWO players.")
            return []
        
        # Check: No season entity (auto filter by current season)
        if seasons:
            print("⚠ Query 16 should not have season specified (auto-filtered by current season).")
            return []
        
        player1 = players[0]
        player2 = players[1]
        
        # Get current season from config
        config = load_config()
        current_season = config.get("CURRENT_SEASON", "2022-23")
        
        cypher = """
           MATCH (s:Season {season_name: $current_season})-[:HAS_GW]->(gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
           MATCH (p:Player)
           WHERE p.player_name CONTAINS $player1 OR p.player_name CONTAINS $player2
           MATCH (p)-[r:PLAYED_IN]->(f)
           OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
           RETURN p.player_name AS player_name,
               pos.name AS position,
               SUM(r.goals_scored) AS total_goals,
               SUM(r.assists) AS total_assists,
               SUM(r.total_points) AS total_points,
               SUM(r.minutes) AS total_minutes,
               SUM(r.clean_sheets) AS total_clean_sheets,
               SUM(r.goals_conceded) AS total_goals_conceded,
               SUM(r.saves) AS total_saves,
               COUNT(f) AS matches_played
           ORDER BY player_name
           """
        
        params = {"player1": player1, "player2": player2, "current_season": current_season}
        
        print(f"\n📊 Executing Query 16: Comparing {player1} vs {player2} (season {current_season})")
        print(f"Cypher Query:\n{cypher}")
        print(f"Parameters: {params}")
        
        results = self.conn.execute_query(cypher, params)
        
        if results:
            print(f"\n✅ Comparison Results:")
            for r in results:
                position = r.get('position') or 'Unknown position'
                print(f"\n{r['player_name']} ({position}):")
                print(f"  Matches: {r['matches_played']}")
                print(f"  Goals: {r['total_goals']}")
                print(f"  Assists: {r['total_assists']}")
                print(f"  Points: {r['total_points']}")
                print(f"  Minutes: {r['total_minutes']}")
                print(f"  Clean Sheets: {r['total_clean_sheets']}")
        else:
            print(f"\n⚠ No data found for comparison")
        
        return results
    
    def query_gameweek_fixtures(
        self, 
        entities: Dict[str, List], 
        ranking: Optional[str] = None,
        threshold: Optional[Dict] = None,
        limit: int = 20
    ) -> List[Dict]:
        """
        Query 17: Get all fixtures in a specific gameweek
        
        Expected input from preprocessing:
        - entities: {"Gameweek": ["GW_number"]} - Exactly 1 gameweek
        - No season (auto filter by current season from config.txt)
        """
        
        gameweeks = entities.get("Gameweek", [])
        seasons = entities.get("Season", [])
        
        # Check: Must have exactly ONE gameweek
        if len(gameweeks) != 1:
            print("⚠ Query 17 requires exactly ONE gameweek.")
            return []
        
        # Check: No season entity (auto filter by current season)
        if seasons:
            print("⚠ Query 17 should not have season specified (auto-filtered by current season).")
            return []
        
        gameweek_num = gameweeks[0]
        
        # Get current season from config
        config = load_config()
        current_season = config.get("CURRENT_SEASON", "2022-23")
        
        cypher = """
        MATCH (s:Season {season_name: $current_season})-[:HAS_GW]->(gw:Gameweek {GW_number: $gameweek})-[:HAS_FIXTURE]->(f:Fixture)
        MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
        MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
        RETURN 
            gw.GW_number AS gameweek,
            f.fixture_number AS fixture_id,
            home.name AS home_team,
            away.name AS away_team,
            f.kickoff_time AS kickoff_time
        ORDER BY f.fixture_number
        """
        
        params = {"gameweek": int(gameweek_num), "current_season": current_season}
        
        print(f"\n📊 Executing Query 17: Fixtures in Gameweek {gameweek_num} (season {current_season})")
        print(f"Cypher Query:\n{cypher}")
        print(f"Parameters: {params}")
        
        results = self.conn.execute_query(cypher, params)
        
        print(f"\n✅ Results ({len(results)} fixtures):")
        for i, record in enumerate(results, 1):
            print(f"{i}. {record['home_team']} vs {record['away_team']}")
        
        return results
    
    def query_head_to_head_fixtures(
        self, 
        entities: Dict[str, List], 
        ranking: Optional[str] = None,
        threshold: Optional[Dict] = None,
        limit: int = 20
    ) -> List[Dict]:
        """
        Query 23: Get head-to-head fixtures between two teams
        
        Expected input from preprocessing:
        - entities: {"Team": ["team1", "team2"]} - Exactly 2 teams
        - No players, no season (auto filter by current season)
        """
        
        teams = entities.get("Team", [])
        players = entities.get("Player", [])
        seasons = entities.get("Season", [])
        statistics = entities.get("Statistic", [])
        
        # Check: Must have exactly TWO teams
        if len(teams) != 2:
            print("⚠ Query 23 requires exactly TWO teams.")
            return []
        
        # Check: No season entity (auto filter by current season)
        if seasons:
            print("⚠ Query 23 should not have season specified (auto-filtered by current season).")
            return []
        
        # Check: No players or statistics
        if players or statistics:
            print("⚠ Query 23 should not have players or statistics specified.")
            return []
        
        team1 = teams[0]
        team2 = teams[1]
        
        # Get current season from config
        config = load_config()
        current_season = config.get("CURRENT_SEASON", "2022-23")
        
        cypher = """
        MATCH (s:Season {season_name: $current_season})-[:HAS_GW]->(gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
        MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
        MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
        WHERE (home.name = $team1 AND away.name = $team2) OR (home.name = $team2 AND away.name = $team1)
        RETURN 
            s.season_name AS season,
            gw.GW_number AS gameweek,
            f.fixture_number AS fixture_id,
            home.name AS home_team,
            away.name AS away_team,
            f.kickoff_time AS kickoff_time
        ORDER BY gw.GW_number
        """
        
        params = {"team1": team1, "team2": team2, "current_season": current_season}
        
        print(f"\n📊 Executing Query 23: {team1} vs {team2} head-to-head fixtures (season {current_season})")
        print(f"Cypher Query:\n{cypher}")
        print(f"Parameters: {params}")
        
        results = self.conn.execute_query(cypher, params)
        
        print(f"\n✅ Results ({len(results)} fixtures):")
        for i, record in enumerate(results, 1):
            print(f"{i}. GW{record['gameweek']}: {record['home_team']} vs {record['away_team']}")
        
        return results

    def query_dynamic_fallback(
        self, 
        entities: Dict[str, List], 
        ranking: Optional[str] = None,
        threshold: Optional[Dict] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Query 11 (Fallback): Dynamic query that handles any combination of entities
        
        This is the fallback query when no base query (1-10) matches.
        It dynamically builds a Cypher query based on whatever entities are present.
        
        Supported entities:
        - Season: Filter by specific season(s)
        - Gameweek: Filter by specific gameweek(s)
        - Player: Filter by specific player(s)
        - Team: Filter by specific team(s) - infers player's team from fixtures
        - Position: Filter by specific position(s)
        - Statistic: Aggregate and return statistic(s)
        
        Returns: List of players matching all criteria with their statistics
        """
        
        # Extract all entities
        seasons = entities.get("Season", [])
        gameweeks = entities.get("Gameweek", [])
        players = entities.get("Player", [])
        teams = entities.get("Team", [])
        positions = entities.get("Position", [])
        statistics = entities.get("Statistic", [])
        
        # Get current season from config (default if no season specified)
        config = load_config()
        current_season = config.get("CURRENT_SEASON", "2022-23")
        
        # Determine season filtering mode:
        # - "all" or "ALL" in seasons list means query all seasons
        # - Multiple seasons means filter by those specific seasons
        # - Single season means filter by that season
        # - No season + specific player = all seasons (for historical stats)
        # - No season + no player = current season only
        
        use_all_seasons = False
        seasons_to_use = []
        
        # Check for "all" marker in seasons
        if any(s.lower() == "all" for s in seasons):
            use_all_seasons = True
        elif len(seasons) > 1:
            # Multiple specific seasons
            seasons_to_use = seasons
        elif len(seasons) == 1:
            # Single specific season
            seasons_to_use = [seasons[0]]
        elif players and not gameweeks:
            # No season specified but querying specific player(s) without gameweek
            # = aggregate across all seasons for historical stats
            use_all_seasons = True
        else:
            # Default to current season
            seasons_to_use = [current_season]
        
        # Build parameters dictionary
        params = {}
        
        # If team filtering is needed, use a different query structure
        # that infers the player's team from their fixtures
        if teams:
            # Pass season info to team filter method
            return self._query_dynamic_with_team_filter(
                entities, seasons_to_use[0] if seasons_to_use else None, teams, gameweeks, players, positions, 
                statistics, ranking, threshold, limit, use_all_seasons, seasons_to_use
            )
        
        # Start building the Cypher query (no team filter)
        if use_all_seasons:
            # Query across ALL seasons
            cypher = """
        MATCH (s:Season)-[:HAS_GW]->(gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
        MATCH (p:Player)-[r:PLAYED_IN]->(f)
        MATCH (p)-[:PLAYS_AS]->(pos:Position)
        """
        elif len(seasons_to_use) > 1:
            # Query across MULTIPLE specific seasons
            params["seasons"] = seasons_to_use
            cypher = """
        MATCH (s:Season)-[:HAS_GW]->(gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
        WHERE s.season_name IN $seasons
        MATCH (p:Player)-[r:PLAYED_IN]->(f)
        MATCH (p)-[:PLAYS_AS]->(pos:Position)
        """
        else:
            # Query single season
            params["season"] = seasons_to_use[0]
            cypher = """
        MATCH (s:Season {season_name: $season})-[:HAS_GW]->(gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
        MATCH (p:Player)-[r:PLAYED_IN]->(f)
        MATCH (p)-[:PLAYS_AS]->(pos:Position)
        """
        
        # Build WHERE clauses dynamically
        where_clauses = []
        
        # Filter by gameweeks
        if gameweeks:
            gw_numbers = [int(gw) for gw in gameweeks]
            params["gameweeks"] = gw_numbers
            where_clauses.append("gw.GW_number IN $gameweeks")
        
        # Filter by specific players (case-insensitive substring match, e.g. "salah" matches
        # any player whose name contains "Salah")
        if players:
            params["players"] = [name.lower() for name in players]
            where_clauses.append("ANY(name IN $players WHERE toLower(p.player_name) CONTAINS name)")
        
        # Filter by positions - map common names to DB format
        if positions:
            position_map = {
                "Forward": "FWD", "FWD": "FWD", "Striker": "FWD",
                "Midfielder": "MID", "MID": "MID", "Midfield": "MID",
                "Defender": "DEF", "DEF": "DEF", "Defense": "DEF",
                "Goalkeeper": "GK", "GK": "GK", "Goalie": "GK"
            }
            mapped_positions = [position_map.get(p, p) for p in positions]
            params["positions"] = mapped_positions
            where_clauses.append("pos.name IN $positions")
        
        # Add WHERE clause if any filters exist
        if where_clauses:
            cypher += "WHERE " + " AND ".join(where_clauses) + "\n"
        
        # Build aggregation based on statistics or default stats
        if statistics:
            # Build the WITH clause with aggregations
            stats_with = ", ".join([f"SUM(r.{stat}) AS total_{stat}" for stat in statistics])
            # Build the RETURN clause using the aliases from WITH
            stats_return = ", ".join([f"total_{stat}" for stat in statistics])
            
            # When querying all seasons, aggregate by player_name to combine across seasons
            if use_all_seasons or len(seasons_to_use) > 1:
                cypher += f"""
        WITH p.player_name AS player_name, pos.name AS position, {stats_with}
        """
            else:
                cypher += f"""
        WITH p, pos, {stats_with}
        """
            
            # Apply threshold if provided
            if threshold:
                thresh_stat = threshold.get("stat")
                thresh_op = threshold.get("operator", ">")
                thresh_val = threshold.get("value", 0)
                params["threshold_value"] = thresh_val
                
                # Map operators
                op_map = {">": ">", "GT": ">", ">=": ">=", "GE": ">=", 
                          "<": "<", "LT": "<", "<=": "<=", "LE": "<=", 
                          "=": "=", "EQ": "="}
                op = op_map.get(thresh_op, ">")
                
                if thresh_stat in statistics:
                    cypher += f"WHERE total_{thresh_stat} {op} $threshold_value\n"
            
            # Determine ordering - order by ALL statistics in the order they were specified
            order_dir = "DESC" if ranking == "best" else ("ASC" if ranking == "worst" else "DESC")
            order_by_clauses = ", ".join([f"total_{stat} {order_dir}" for stat in statistics])
            
            # Use the limit parameter passed in (defaults to 10)
            limit_val = limit
            
            # Return clause depends on whether we're aggregating by player name or node
            if use_all_seasons or len(seasons_to_use) > 1:
                cypher += f"""
        RETURN player_name, position, {stats_return}
        ORDER BY {order_by_clauses}
        LIMIT {limit_val}
        """
            else:
                cypher += f"""
        RETURN p.player_name AS player_name, pos.name AS position, {stats_return}
        ORDER BY {order_by_clauses}
        LIMIT {limit_val}
        """
        else:
            # Default: return common stats
            # When querying all seasons, aggregate by player_name to combine across seasons
            if use_all_seasons or len(seasons_to_use) > 1:
                cypher += """
        WITH p.player_name AS player_name, pos.name AS position, 
             SUM(r.total_points) AS total_points,
             SUM(r.goals_scored) AS goals_scored,
             SUM(r.assists) AS assists,
             SUM(r.minutes) AS minutes,
             COUNT(r) AS matches_played
        """
            else:
                cypher += """
        WITH p, pos, 
             SUM(r.total_points) AS total_points,
             SUM(r.goals_scored) AS goals_scored,
             SUM(r.assists) AS assists,
             SUM(r.minutes) AS minutes,
             COUNT(r) AS matches_played
        """
            
            # Apply threshold if provided
            if threshold:
                thresh_stat = threshold.get("stat")
                thresh_op = threshold.get("operator", ">")
                thresh_val = threshold.get("value", 0)
                params["threshold_value"] = thresh_val
                
                op_map = {">": ">", "GT": ">", ">=": ">=", "GE": ">=", 
                          "<": "<", "LT": "<", "<=": "<=", "LE": "<=", 
                          "=": "=", "EQ": "="}
                op = op_map.get(thresh_op, ">")
                
                if thresh_stat == "total_points":
                    cypher += f"WHERE total_points {op} $threshold_value\n"
                elif thresh_stat == "goals_scored":
                    cypher += f"WHERE goals_scored {op} $threshold_value\n"
                elif thresh_stat == "assists":
                    cypher += f"WHERE assists {op} $threshold_value\n"
            
            order_dir = "DESC" if ranking == "best" else ("ASC" if ranking == "worst" else "DESC")
            
            # Use the limit parameter passed in (defaults to 10)
            limit_val = limit
            
            # Return clause depends on whether we're aggregating by player name or node
            if use_all_seasons or len(seasons_to_use) > 1:
                cypher += f"""
        RETURN player_name, position, 
               total_points, goals_scored, assists, minutes, matches_played
        ORDER BY total_points {order_dir}
        LIMIT {limit_val}
        """
            else:
                cypher += f"""
        RETURN p.player_name AS player_name, pos.name AS position, 
               total_points, goals_scored, assists, minutes, matches_played
        ORDER BY total_points {order_dir}
        LIMIT {limit_val}
        """
        
        # Build description of what we're querying
        desc_parts = []
        if players:
            desc_parts.append(f"players: {', '.join(players)}")
        if teams:
            desc_parts.append(f"teams: {', '.join(teams)}")
        if positions:
            desc_parts.append(f"positions: {', '.join(positions)}")
        if gameweeks:
            desc_parts.append(f"gameweeks: {', '.join(map(str, gameweeks))}")
        if statistics:
            desc_parts.append(f"statistics: {', '.join(statistics)}")
        if threshold:
            desc_parts.append(f"threshold: {threshold}")
        if ranking:
            desc_parts.append(f"ranking: {ranking}")
        
        desc = " | ".join(desc_parts) if desc_parts else "all players"
        
        # Determine season description for logging
        if use_all_seasons:
            season_desc = "ALL SEASONS"
        elif len(seasons_to_use) > 1:
            season_desc = f"Seasons: {', '.join(seasons_to_use)}"
        else:
            season_desc = f"Season: {seasons_to_use[0]}"
        
        print(f"\n📊 Executing Query 11 (Fallback): Dynamic query")
        print(f"   Filters: {desc}")
        print(f"   {season_desc}")
        print(f"Cypher Query:\n{cypher}")
        print(f"Parameters: {params}")
        
        results = self.conn.execute_query(cypher, params)
        
        # Format output - filter out results with no position
        filtered_results = [r for r in results if r.get('position')]
        
        print(f"\n✅ Results ({len(filtered_results)} players):")
        # Print only first 10 results in terminal
        for i, record in enumerate(filtered_results[:10], 1):
            position = record.get('position')
            player_name = record.get('player_name')
            
            # Build stats string
            stats_str = []
            for key, val in record.items():
                if key not in ['player_name', 'position'] and val is not None:
                    stats_str.append(f"{key}: {val}")
            
            print(f"{i}. {player_name} ({position}): {', '.join(stats_str)}")
        
        if len(filtered_results) > 10:
            print(f"   ... and {len(filtered_results) - 10} more results")
        
        return filtered_results

    def _query_dynamic_with_team_filter(
        self,
        entities: Dict[str, List],
        season_to_use: Optional[str],
        teams: List[str],
        gameweeks: List[str],
        players_filter: List[str],
        positions: List[str],
        statistics: List[str],
        ranking: Optional[str],
        threshold: Optional[Dict],
        limit: int,
        use_all_seasons: bool = False,
        seasons_to_use: List[str] = None
    ) -> List[Dict]:
        """
        Helper method for query_dynamic_fallback when team filtering is needed.
        
        Strategy: A player is considered to be ON a team if they played in most
        of that team's fixtures (>= 15 out of 38, or whatever threshold makes sense).
        This distinguishes team players from opponents who only face them 1-2 times.
        
        Note: When filtering by gameweek, we show ALL players in that fixture.
        """
        
        if seasons_to_use is None:
            seasons_to_use = []
        
        params = {"teams": teams}
        
        # For gameweek filtering, we need a different approach - show fixture details
        if gameweeks:
            return self._query_team_gameweek_players(
                season_to_use, teams, gameweeks, players_filter, positions,
                statistics, ranking, threshold, limit
            )
        
        # For full season team queries, use match count to identify team players
        # Build season filtering
        if use_all_seasons:
            cypher = """
        // Find fixtures involving target team(s) - ALL SEASONS
        MATCH (s:Season)-[:HAS_GW]->(gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
        MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
        MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
        WHERE home.name IN $teams OR away.name IN $teams
        """
        elif len(seasons_to_use) > 1:
            params["seasons"] = seasons_to_use
            cypher = """
        // Find fixtures involving target team(s) - MULTIPLE SEASONS
        MATCH (s:Season)-[:HAS_GW]->(gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
        WHERE s.season_name IN $seasons
        MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
        MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
        WHERE home.name IN $teams OR away.name IN $teams
        """
        else:
            params["season"] = season_to_use
            cypher = """
        // Find fixtures involving target team(s)
        MATCH (s:Season {season_name: $season})-[:HAS_GW]->(gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
        MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
        MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
        WHERE home.name IN $teams OR away.name IN $teams
        """
        
        cypher += """
        // Find players who played in these fixtures
        MATCH (p:Player)-[r:PLAYED_IN]->(f)
        MATCH (p)-[:PLAYS_AS]->(pos:Position)
        
        // Determine the target team for this fixture
        WITH p, pos, r, gw,
             CASE WHEN home.name IN $teams THEN home.name ELSE away.name END AS target_team
        """
        
        # Build WHERE clauses
        where_clauses = []
        
        # Filter by specific players (case-insensitive substring match)
        if players_filter:
            params["players"] = [name.lower() for name in players_filter]
            where_clauses.append("ANY(name IN $players WHERE toLower(p.player_name) CONTAINS name)")
        
        # Filter by positions - handle common position name variations
        if positions:
            # Map common names to DB format
            position_map = {
                "Forward": "FWD", "FWD": "FWD", "Striker": "FWD",
                "Midfielder": "MID", "MID": "MID", "Midfield": "MID",
                "Defender": "DEF", "DEF": "DEF", "Defense": "DEF",
                "Goalkeeper": "GK", "GK": "GK", "Goalie": "GK"
            }
            mapped_positions = [position_map.get(p, p) for p in positions]
            params["positions"] = mapped_positions
            where_clauses.append("pos.name IN $positions")
        
        if where_clauses:
            cypher += "WHERE " + " AND ".join(where_clauses) + "\n"
        
        # For full season, require at least 10 matches to filter to actual team players
        min_matches = 10
        
        # Build aggregation - group by player and collect their team
        if statistics:
            stats_with = ", ".join([f"SUM(r.{stat}) AS total_{stat}" for stat in statistics])
            stats_return = ", ".join([f"total_{stat}" for stat in statistics])
            
            cypher += f"""
        WITH p, pos, target_team, {stats_with}, COUNT(r) AS matches_in_team_fixtures
        // Only include players who played in many of this team's fixtures (team players)
        WHERE matches_in_team_fixtures >= {min_matches}
        """
            
            # Apply threshold if provided
            if threshold:
                thresh_stat = threshold.get("stat")
                thresh_op = threshold.get("operator", ">")
                thresh_val = threshold.get("value", 0)
                params["threshold_value"] = thresh_val
                
                op_map = {">": ">", "GT": ">", ">=": ">=", "GE": ">=", 
                          "<": "<", "LT": "<", "<=": "<=", "LE": "<=", 
                          "=": "=", "EQ": "="}
                op = op_map.get(thresh_op, ">")
                
                if thresh_stat in statistics:
                    cypher += f"AND total_{thresh_stat} {op} $threshold_value\n"
            
            # Order by ALL statistics in the order they were specified
            order_dir = "DESC" if ranking == "best" else ("ASC" if ranking == "worst" else "DESC")
            order_by_clauses = ", ".join([f"total_{stat} {order_dir}" for stat in statistics])
            
            # Use the limit parameter
            limit_val = limit
            
            cypher += f"""
        RETURN p.player_name AS player_name, pos.name AS position, target_team AS team, 
               {stats_return}, matches_in_team_fixtures AS matches
        ORDER BY {order_by_clauses}
        LIMIT {limit_val}
        """
        else:
            cypher += f"""
        WITH p, pos, target_team,
             SUM(r.total_points) AS total_points,
             SUM(r.goals_scored) AS goals_scored,
             SUM(r.assists) AS assists,
             SUM(r.minutes) AS minutes,
             COUNT(r) AS matches_in_team_fixtures
        // Only include players who played in many of this team's fixtures (team players)
        WHERE matches_in_team_fixtures >= {min_matches}
        """
            
            # Apply threshold if provided
            if threshold:
                thresh_stat = threshold.get("stat")
                thresh_op = threshold.get("operator", ">")
                thresh_val = threshold.get("value", 0)
                params["threshold_value"] = thresh_val
                
                op_map = {">": ">", "GT": ">", ">=": ">=", "GE": ">=", 
                          "<": "<", "LT": "<", "<=": "<=", "LE": "<=", 
                          "=": "=", "EQ": "="}
                op = op_map.get(thresh_op, ">")
                
                if thresh_stat == "total_points":
                    cypher += f"AND total_points {op} $threshold_value\n"
                elif thresh_stat == "goals_scored":
                    cypher += f"AND goals_scored {op} $threshold_value\n"
                elif thresh_stat == "assists":
                    cypher += f"AND assists {op} $threshold_value\n"
            
            order_dir = "DESC" if ranking == "best" else ("ASC" if ranking == "worst" else "DESC")
            
            # Use the limit parameter
            limit_val = limit
            
            cypher += f"""
        RETURN p.player_name AS player_name, pos.name AS position, target_team AS team,
               total_points, goals_scored, assists, minutes, matches_in_team_fixtures AS matches
        ORDER BY total_points {order_dir}
        LIMIT {limit_val}
        """
        
        # Build description
        desc_parts = [f"teams: {', '.join(teams)}"]
        if players_filter:
            desc_parts.append(f"players: {', '.join(players_filter)}")
        if positions:
            desc_parts.append(f"positions: {', '.join(positions)}")
        if statistics:
            desc_parts.append(f"statistics: {', '.join(statistics)}")
        if threshold:
            desc_parts.append(f"threshold: {threshold}")
        if ranking:
            desc_parts.append(f"ranking: {ranking}")
        
        # Determine season description for logging
        if use_all_seasons:
            season_desc = "ALL SEASONS"
        elif len(seasons_to_use) > 1:
            season_desc = f"Seasons: {', '.join(seasons_to_use)}"
        else:
            season_desc = f"Season: {season_to_use}"
        
        print(f"\n📊 Executing Query 11 (Fallback): Dynamic query with team filter")
        print(f"   Filters: {' | '.join(desc_parts)}")
        print(f"   {season_desc}")
        print(f"   Note: Shows players who played frequently in {', '.join(teams)} fixtures")
        print(f"Cypher Query:\n{cypher}")
        print(f"Parameters: {params}")
        
        results = self.conn.execute_query(cypher, params)
        
        # Format output - filter out results with no position
        filtered_results = [r for r in results if r.get('position')]
        
        print(f"\n✅ Results ({len(filtered_results)} players):")
        # Print only first 10 results in terminal
        for i, record in enumerate(filtered_results[:10], 1):
            position = record.get('position')
            player_name = record.get('player_name')
            team = record.get('team') or 'Unknown team'
            
            stats_str = []
            for key, val in record.items():
                if key not in ['player_name', 'position', 'team'] and val is not None:
                    stats_str.append(f"{key}: {val}")
            
            print(f"{i}. {player_name} ({position}, {team}): {', '.join(stats_str)}")
        
        if len(filtered_results) > 10:
            print(f"   ... and {len(filtered_results) - 10} more results")
        
        return filtered_results

    def _query_team_gameweek_players(
        self,
        season_to_use: str,
        teams: List[str],
        gameweeks: List[str],
        players_filter: List[str],
        positions: List[str],
        statistics: List[str],
        ranking: Optional[str],
        threshold: Optional[Dict],
        limit: int
    ) -> List[Dict]:
        """
        Query players who played in a specific gameweek's fixture involving a team.
        
        Note: This returns ALL players who played in the fixture, including both
        the target team and their opponent. This is useful for analyzing specific
        matchups or gameweek performances.
        """
        
        params = {
            "season": season_to_use, 
            "teams": teams,
            "gameweeks": [int(gw) for gw in gameweeks]
        }
        
        # Build query to get players from specific gameweek fixtures
        cypher = """
        // Find the specific fixture(s) involving the target team in the specified gameweek(s)
        MATCH (s:Season {season_name: $season})-[:HAS_GW]->(gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
        WHERE gw.GW_number IN $gameweeks
        MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
        MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
        WHERE home.name IN $teams OR away.name IN $teams
        
        // Find all players who played in this fixture
        MATCH (p:Player)-[r:PLAYED_IN]->(f)
        MATCH (p)-[:PLAYS_AS]->(pos:Position)
        
        // Include both home and away team info
        WITH p, pos, r, gw, home, away,
             CASE WHEN home.name IN $teams THEN home.name ELSE away.name END AS target_team,
             home.name + ' vs ' + away.name AS fixture_name
        """
        
        where_clauses = []
        
        if players_filter:
            params["players"] = [name.lower() for name in players_filter]
            where_clauses.append("ANY(name IN $players WHERE toLower(p.player_name) CONTAINS name)")
        
        if positions:
            position_map = {
                "Forward": "FWD", "FWD": "FWD", "Striker": "FWD",
                "Midfielder": "MID", "MID": "MID", "Midfield": "MID",
                "Defender": "DEF", "DEF": "DEF", "Defense": "DEF",
                "Goalkeeper": "GK", "GK": "GK", "Goalie": "GK"
            }
            mapped_positions = [position_map.get(p, p) for p in positions]
            params["positions"] = mapped_positions
            where_clauses.append("pos.name IN $positions")
        
        if where_clauses:
            cypher += "WHERE " + " AND ".join(where_clauses) + "\n"
        
        # Build return clause based on statistics
        if statistics:
            stats_select = ", ".join([f"r.{stat} AS {stat}" for stat in statistics])
            
            cypher += f"""
        RETURN p.player_name AS player_name, pos.name AS position,
               gw.GW_number AS gameweek, fixture_name, target_team AS team,
               {stats_select}, r.minutes AS minutes
        """
            
            if threshold:
                thresh_stat = threshold.get("stat")
                thresh_op = threshold.get("operator", ">")
                thresh_val = threshold.get("value", 0)
                params["threshold_value"] = thresh_val
                
                op_map = {">": ">", "GT": ">", ">=": ">=", "GE": ">=", 
                          "<": "<", "LT": "<", "<=": "<=", "LE": "<=", 
                          "=": "=", "EQ": "="}
                op = op_map.get(thresh_op, ">")
                
                if thresh_stat in statistics:
                    cypher = cypher.rstrip() + f"\nWHERE r.{thresh_stat} {op} $threshold_value\n"
            
            # Order by ALL statistics in the order they were specified
            order_dir = "DESC" if ranking == "best" else ("ASC" if ranking == "worst" else "DESC")
            order_by_clauses = ", ".join([f"r.{stat} {order_dir}" for stat in statistics])
            cypher += f"ORDER BY {order_by_clauses}\n"
        else:
            cypher += """
        RETURN p.player_name AS player_name, pos.name AS position,
               gw.GW_number AS gameweek, fixture_name, target_team AS team,
               r.total_points AS total_points, r.goals_scored AS goals_scored,
               r.assists AS assists, r.minutes AS minutes
        """
            
            if threshold:
                thresh_stat = threshold.get("stat")
                thresh_op = threshold.get("operator", ">")
                thresh_val = threshold.get("value", 0)
                params["threshold_value"] = thresh_val
                
                op_map = {">": ">", "GT": ">", ">=": ">=", "GE": ">=", 
                          "<": "<", "LT": "<", "<=": "<=", "LE": "<=", 
                          "=": "=", "EQ": "="}
                op = op_map.get(thresh_op, ">")
                
                if thresh_stat in ["total_points", "goals_scored", "assists", "minutes"]:
                    cypher = cypher.rstrip() + f"\nWHERE r.{thresh_stat} {op} $threshold_value\n"
            
            order_dir = "DESC" if ranking == "best" else ("ASC" if ranking == "worst" else "DESC")
            cypher += f"ORDER BY r.total_points {order_dir}\n"
        
        # LIMIT 10 if ranking specified, else LIMIT 50
        limit_val = 10 if ranking else 50
        cypher += f"LIMIT {limit_val}"
        
        # Build description
        desc_parts = [f"teams: {', '.join(teams)}", f"gameweeks: {', '.join(map(str, gameweeks))}"]
        if players_filter:
            desc_parts.append(f"players: {', '.join(players_filter)}")
        if positions:
            desc_parts.append(f"positions: {', '.join(positions)}")
        if statistics:
            desc_parts.append(f"statistics: {', '.join(statistics)}")
        if threshold:
            desc_parts.append(f"threshold: {threshold}")
        if ranking:
            desc_parts.append(f"ranking: {ranking}")
        
        print(f"\n📊 Executing Query 11 (Fallback): Gameweek fixture players")
        print(f"   Filters: {' | '.join(desc_parts)}")
        print(f"   Season: {season_to_use}")
        print(f"   Note: Shows ALL players in {', '.join(teams)} fixtures for GW {', '.join(map(str, gameweeks))}")
        print(f"Cypher Query:\n{cypher}")
        print(f"Parameters: {params}")
        
        results = self.conn.execute_query(cypher, params)
        
        # Filter out results with no position
        filtered_results = [r for r in results if r.get('position')]
        
        print(f"\n✅ Results ({len(filtered_results)} players):")
        # Print only first 10 results in terminal
        for i, record in enumerate(filtered_results[:10], 1):
            position = record.get('position')
            player_name = record.get('player_name')
            fixture = record.get('fixture_name') or ''
            gw = record.get('gameweek', '')
            
            stats_str = []
            for key, val in record.items():
                if key not in ['player_name', 'position', 'team', 'fixture_name', 'gameweek'] and val is not None:
                    stats_str.append(f"{key}: {val}")
            
            print(f"{i}. {player_name} ({position}) in GW{gw} [{fixture}]: {', '.join(stats_str)}")
        
        if len(filtered_results) > 10:
            print(f"   ... and {len(filtered_results) - 10} more results")
        
        return filtered_results


# ============================================================
# 4. MAIN EXECUTION
# ============================================================

def execute_baseline_query(preprocessing_output: Dict[str, Any]) -> List[Dict]:
    """
    Execute baseline query based on preprocessing output
    
    Args:
        preprocessing_output: Output from preprocessing.process_user_query()
            {
                "query": str,
                "intent": str,
                "entities": {...},
                "ranking": str | None,
                "threshold": dict | None,
                "limit": int | None
            }
    
    Returns:
        List of query results
    """
    
    # Load config and connect to Neo4j
    config = load_config()
    uri = config.get("URI")
    user = config.get("USERNAME")
    password = config.get("PASSWORD")
    
    conn = Neo4jConnection(uri, user, password)
    builder = BaselineQueryBuilder(conn)
    
    try:
        entities = preprocessing_output.get("entities", {})
        ranking = preprocessing_output.get("ranking")
        threshold = preprocessing_output.get("threshold")
        limit = preprocessing_output.get("limit")  # Extract limit from preprocessing
        
        # Determine effective limit:
        # - If explicit limit provided, use it
        # - If threshold provided without limit, return all matches (use high limit)
        # - Otherwise use default of 10
        if limit is not None:
            effective_limit = limit
        elif threshold is not None:
            effective_limit = 500  # Return all matching when threshold specified
        else:
            effective_limit = 10  # Default limit
        
        # Extract entity counts
        statistics = entities.get("Statistic", [])
        players = entities.get("Player", [])
        teams = entities.get("Team", [])
        positions = entities.get("Position", [])
        seasons = entities.get("Season", [])
        gameweeks = entities.get("Gameweek", [])
        
        # Get intent for disambiguation
        intent = preprocessing_output.get("intent", "")
        
        # Query 16: Compare two players (exactly 2 players, nothing else, no threshold)
        if (
            len(players) == 2
            and len(teams) == 0
            and len(statistics) == 0
            and len(positions) == 0
            and len(seasons) == 0
            and len(gameweeks) == 0
            and intent == "COMPARISON"
            and threshold is None
        ):
            return builder.query_compare_two_players(entities, ranking, threshold)

        # Query 23: Head-to-head fixtures (exactly 2 teams, fixture intent, nothing else, no threshold)
        if (
            len(teams) == 2
            and len(players) == 0
            and len(statistics) == 0
            and len(positions) == 0
            and len(seasons) == 0
            and len(gameweeks) == 0
            and intent == "FIXTURE-RELATED"
            and threshold is None
        ):
            return builder.query_head_to_head_fixtures(entities, ranking, threshold)

        # Query 17: Gameweek fixtures (exactly 1 gameweek, nothing else, no threshold)
        if (
            len(gameweeks) == 1
            and len(teams) == 0
            and len(players) == 0
            and len(statistics) == 0
            and len(positions) == 0
            and len(seasons) == 0
            and intent == "FIXTURE-RELATED"
            and threshold is None
        ):
            return builder.query_gameweek_fixtures(entities, ranking, threshold)

        # Query 8: Single player performance (exactly 1 player, nothing else, no threshold)
        if (
            len(players) == 1
            and len(statistics) == 0
            and len(teams) == 0
            and len(positions) == 0
            and len(seasons) == 0
            and len(gameweeks) == 0
            and intent == "PLAYER-RELATED"
            and threshold is None
        ):
            return builder.query_player_performance(entities, ranking, threshold)

        # Query 9: Team fixtures (exactly 1 team, nothing else, no threshold)
        if (
            len(teams) == 1
            and len(players) == 0
            and len(statistics) == 0
            and len(positions) == 0
            and len(seasons) == 0
            and len(gameweeks) == 0
            and intent == "FIXTURE-RELATED"
            and threshold is None
        ):
            return builder.query_team_fixtures(entities, ranking, threshold)

        # Query 10: Gameweek top performers (1 gameweek, 1 statistic, nothing else, no threshold)
        if (
            len(gameweeks) == 1
            and len(statistics) == 1
            and len(players) == 0
            and len(teams) == 0
            and len(positions) == 0
            and len(seasons) == 0
            and intent == "PLAYER-RELATED"
            and ranking == "best"
            and threshold is None
        ):
            return builder.query_gameweek_top_performers(entities, ranking, threshold, effective_limit)

        # Query 3: Top players by stat + position (1 stat, 1 position, nothing else, no threshold)
        if (
            len(statistics) == 1
            and len(positions) == 1
            and len(players) == 0
            and len(teams) == 0
            and len(seasons) == 0
            and len(gameweeks) == 0
            and intent == "PLAYER-RELATED"
            and ranking == "best"
            and threshold is None
        ):
            return builder.query_top_players_by_stat_and_position(entities, ranking, threshold, effective_limit)

        # Query 4: Worst players by stat + position (1 stat, 1 position, nothing else, no threshold)
        if (
            len(statistics) == 1
            and len(positions) == 1
            and len(players) == 0
            and len(teams) == 0
            and len(seasons) == 0
            and len(gameweeks) == 0
            and intent == "PLAYER-RELATED"
            and ranking == "worst"
            and threshold is None
        ):
            return builder.query_worst_players_by_stat_and_position(entities, ranking, threshold, effective_limit)

        # Query 1: Top players by statistic (1 stat only, no other entities, no threshold)
        if (
            len(statistics) == 1
            and len(players) == 0
            and len(teams) == 0
            and len(positions) == 0
            and len(seasons) == 0
            and len(gameweeks) == 0
            and intent == "PLAYER-RELATED"
            and ranking == "best"
            and threshold is None
        ):
            return builder.query_top_players_by_statistic(entities, ranking, threshold, effective_limit)

        # Query 2: Worst players by statistic (1 stat only, no other entities, no threshold)
        if (
            len(statistics) == 1
            and len(players) == 0
            and len(teams) == 0
            and len(positions) == 0
            and len(seasons) == 0
            and len(gameweeks) == 0
            and intent == "PLAYER-RELATED"
            and ranking == "worst"
            and threshold is None
        ):
            return builder.query_worst_players_by_statistic(entities, ranking, threshold, effective_limit)
        
        # Query 11 (Fallback): Dynamic query for any combination not matching base queries
        print("⚠ No base query matched. Using dynamic fallback query (Query 11)...")
        return builder.query_dynamic_fallback(entities, ranking, threshold, effective_limit)
            
    finally:
        conn.close()


# ============================================================
# 5. TESTING
# ============================================================

if __name__ == "__main__":
    import preprocessing
    
    print("=" * 70)
    print("BASELINE QUERY TESTING - Including Fallback Query 11")
    print("=" * 70)
    
    # Natural language test queries - processed through preprocessing.py
    test_cases = [
        # ============================================================
        # BASE QUERY TESTS (Queries 1-10, 16, 17, 23)
        # ============================================================
        
        # # Query 1: Top players by single statistic (various stats)
        # "Who are the best players by goals scored?",
        # "Top players by assists this season",
        # "Best players by total points",
        # "Players with the most clean sheets",
        # "Who has the most bonus points?",
        
        # # Query 2: Worst players by single statistic
        # "Worst players by total points",
        # "Who are the worst players by goals scored?",
        # "Players with the least assists",
        
        # # Query 3: Top players by stat + position
        # "Best forwards by goals scored",
        # "Top midfielders by assists",
        # "Best defenders by clean sheets",
        # "Top goalkeepers by saves",
        
        # # Query 4: Worst players by stat + position
        # "Worst midfielders by assists",
        # "Worst forwards by total points",
        # "Worst defenders by goals conceded",
        
        # # Query 8: Player performance (single player stats)
        # "How did Salah perform this season?",
        # "Show me Haaland's stats",
        # "What are Saka's numbers this season?",
        # "Tell me about Kevin De Bruyne's performance",
        
        # # Query 9: Team fixtures
        # "Show me Arsenal fixtures",
        # "What are Liverpool's matches?",
        # "Chelsea fixtures this season",
        # "Man City's schedule",
        
        # # Query 10: Gameweek top performers
        # "Best players by total points in gameweek 15",
        # "Top scorers in gameweek 20",
        # "Who performed best in GW 10?",
        
        # # Query 16: Compare two players
        # "Compare Salah and Haaland",
        # "Saka vs Martinelli",
        # "Compare De Bruyne and Bruno Fernandes",
        
        # # Query 17: Gameweek fixtures
        # "What fixtures are in gameweek 10?",
        # "Show me GW 5 matches",
        # "Gameweek 20 fixtures",
        
        # # Query 23: Head to head
        # "Arsenal vs Liverpool head to head",
        # "Man City vs Chelsea history",
        # "Spurs vs Man United fixtures",
        
        # ============================================================
        # FALLBACK QUERY TESTS (Query 11 - Dynamic)
        # These have extra entities that don't fit base queries
        # ============================================================
        
        # Multiple statistics (fallback - more than 1 stat)
        # "Best players by goals and assists combined",
        # "Top players by goals scored and total points",
        # "Players ranked by assists and bonus points",
        
        # # Team + Statistic (fallback - team filter on stats)
        # # "Top Arsenal players by total points",
        # # "Best Chelsea players by goals scored",
        # # "Liverpool players with most assists",
        # "Man United top scorers",
        
        # Position + Team (fallback - position + team combo)
        # "Best forwards from Man City",
        # "Top defenders from Arsenal",
        # "Chelsea midfielders ranked by points",
        # "Liverpool goalkeepers by saves",
        
        # Gameweek + Team (fallback - gw + team combo)
        # "Liverpool players in gameweek 10",
        # "How did Arsenal players do in GW 15?",
        # "Man City performance in gameweek 20",
        
        # Statistic with threshold (fallback - uses threshold)
        # "Players with less than 10 goals",
        # # "Midfielders with more than or equal 70 total points",
        # # "Forwards with 5 assists",
        
        # # Team + Threshold (fallback - team + threshold)
        # # "Chelsea players with more than 5 goals",
        # # "Arsenal players with at least 50 points",
        # # "Liverpool players with more than 3 assists",
        
        # # Complex combinations (fallback - multiple filters)
        # # "Best forwards from Arsenal by goals scored",
        # # "Top Man City midfielders by assists",
        # # "Chelsea defenders with clean sheets",
        
        # # ============================================================
        # # EDGE CASES & VARIATIONS
        # # ============================================================
        
        # # Different phrasing
        # # "Who scored the most goals?",
        # # "Which defenders have the most clean sheets?",
        # # "Show me the top assist providers",
        
        # # Specific team queries with stats
        # # "Newcastle players ranked by total points",
        # # "Brighton forwards by goals",
        # # "West Ham midfielders with assists",
        
        # # Different gameweeks
        # # "Best players in gameweek 1 and gameweek 2",
        # # "Top performers in GW 38", #limitation
        # # "Gameweek 25 results",#limitation

        # # #diiferent seasons
        # # "Top players by goals scored in the 2022/2023 and 2021/2022 season",
        # # "Compare Salah points this season and last season",
        # # "Compare Salah and KDB points this gameweek and last gameweek",
        
        # # ============================================================
        # # NEW TEST CASES - General queries (no specific positions)
        # # ============================================================
        
        # # Top players by various stats (Query 1)
        # "Top 10 players by goals scored",
        # "Best players by assists",
        # "Players with most total points this season",
        # "Who has the highest bonus points?",
        
        # # Worst players (Query 2)
        # "Worst players by minutes played",
        # "Players with fewest goals",
        
        # # Team-based queries (no position filter)
        # "Arsenal players by total points",
        # "Liverpool top scorers",
        # "Man City players with most assists",
        # "Chelsea best performers",
        
        # # Threshold queries (no position)
        # "Players with more than 20 goals",
        # "Players with at least 10 assists",
        # "Players with more than 200 total points",
        
        # # Team + Threshold (no position)
        # "Arsenal players with more than 10 goals",
        # "Liverpool players with at least 150 points",
        
        # # Gameweek queries (no position)
        # "Best players in gameweek 10",
        # "Top performers in gameweek 25",
        
        # # Multiple stats (no position)
        # "Players ranked by goals and assists",
        # "Best players by total points and bonus",
        "how many points did salah score in gameweek 10",
        "Top 15 players by ICT index in the 2022-23 season"
    ]
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"🔹 Test {i}: '{query}'")
        print("-" * 70)
        
        try:
            # Process through preprocessing layer (uses LLM API)
            preprocessing_result = preprocessing.process_user_query(query)
            
            print("\n📥 Preprocessing Output:")
            print(f"   Intent:    {preprocessing_result['intent']}")
            print(f"   Entities:  {preprocessing_result['entities']}")
            print(f"   Ranking:   {preprocessing_result['ranking']}")
            print(f"   Threshold: {preprocessing_result['threshold']}")
            
            # Execute baseline query
            print("\n" + "-" * 70)
            results = execute_baseline_query(preprocessing_result)
            
            print(f"\n✅ Test {i} completed. Retrieved {len(results)} results.")
            
        except Exception as e:
            print(f"\n❌ Test {i} failed with error: {e}")
        
        print("=" * 70)
