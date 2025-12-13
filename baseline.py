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
        WITH p, SUM(r.{stat_name}) AS total_stat
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
        RETURN p.player_name AS player_name, total_stat
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
            print(f"{i}. {record['player_name']}: {record['total_stat']} {stat_name}")
        
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
        WITH p, SUM(r.{stat_name}) AS total_stat
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
        RETURN p.player_name AS player_name, total_stat
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
            print(f"{i}. {record['player_name']}: {record['total_stat']} {stat_name}")
        
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
        WITH p, SUM(r.{stat_name}) AS total_stat
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
        RETURN p.player_name AS player_name, total_stat
        ORDER BY total_stat DESC
        LIMIT {limit}
        """
        
        print(f"\n📊 Executing Query 3: Top {position} players by {stat_name} in season {current_season}")
        print(f"Cypher Query:\n{cypher}")
        print(f"Parameters: {params}")
        
        results = self.conn.execute_query(cypher, params)
        
        print(f"\n✅ Results ({len(results)} players):")
        for i, record in enumerate(results, 1):
            print(f"{i}. {record['player_name']}: {record['total_stat']} {stat_name}")
        
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
        WITH p.player_name AS player_name, SUM(r.{stat_name}) AS total_stat
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
        RETURN player_name, total_stat
        ORDER BY total_stat ASC
        LIMIT {limit}
        """
        
        print(f"\n📊 Executing Query 4: Worst {position} players by {stat_name} (current season)")
        print(f"Cypher Query:\n{cypher}")
        print(f"Parameters: {params}")
        
        results = self.conn.execute_query(cypher, params)
        
        print(f"\n✅ Results ({len(results)} players):")
        for i, record in enumerate(results, 1):
            print(f"{i}. {record['player_name']}: {record['total_stat']} {stat_name}")
        
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
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
        RETURN 
            p.player_name AS player_name,
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
                print(f"\n✅ Results for {r['player_name']}:")
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
        RETURN 
            p.player_name AS player_name,
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
            print(f"{i}. {record['player_name']}: {record['stat_value']} {stat_name}")
        
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
        RETURN p.player_name AS player_name,
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
                print(f"\n{r['player_name']}:")
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
                "threshold": dict | None
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
        
        # Extract entity counts
        statistics = entities.get("Statistic", [])
        players = entities.get("Player", [])
        teams = entities.get("Team", [])
        positions = entities.get("Position", [])
        seasons = entities.get("Season", [])
        gameweeks = entities.get("Gameweek", [])
        
        # Get intent for disambiguation
        intent = preprocessing_output.get("intent", "")
        
        # Query 16: Compare two players (2 players, 0 teams, 0 statistics)
        if len(players) == 2 and len(teams) == 0 and len(statistics) == 0:
            return builder.query_compare_two_players(entities, ranking, threshold)
        
        # Query 23: Head-to-head fixtures (2 teams, intent is FIXTURE-RELATED)
        if len(teams) == 2 and len(players) == 0 and intent == "FIXTURE-RELATED":
            return builder.query_head_to_head_fixtures(entities, ranking, threshold)
        
        # Query 17: Gameweek fixtures (1 gameweek, no teams/players/stats)
        if len(gameweeks) == 1 and len(teams) == 0 and len(players) == 0 and len(statistics) == 0:
            return builder.query_gameweek_fixtures(entities, ranking, threshold)
        
        # Query 8: Single player performance (1 player, 0 statistics)
        if len(players) == 1 and len(statistics) == 0:
            return builder.query_player_performance(entities, ranking, threshold)
        
        # Query 9: Team fixtures (1 team, 0 players, 0 statistics)
        if len(teams) == 1 and len(players) == 0 and len(statistics) == 0:
            return builder.query_team_fixtures(entities, ranking, threshold)
        
        # Query 10: Gameweek top performers (1 gameweek, 1 statistic, ranking=best)
        if len(gameweeks) == 1 and len(statistics) == 1 and ranking == "best":
            return builder.query_gameweek_top_performers(entities, ranking, threshold)
        
        # Query 3: Top players by stat + position (1 stat, 1 position, ranking=best)
        if len(statistics) == 1 and len(positions) == 1 and ranking == "best":
            if not players and not teams and not seasons and not gameweeks:
                return builder.query_top_players_by_stat_and_position(entities, ranking, threshold)
        
        # Query 4: Worst players by stat + position (1 stat, 1 position, ranking=worst)
        if len(statistics) == 1 and len(positions) == 1 and ranking == "worst":
            if not players and not teams and not seasons and not gameweeks:
                return builder.query_worst_players_by_stat_and_position(entities, ranking, threshold)
        
        # Query 1: Top players by statistic (1 stat, ranking=best, no other entities)
        if len(statistics) == 1 and ranking == "best":
            if not players and not teams and not positions and not seasons and not gameweeks:
                return builder.query_top_players_by_statistic(entities, ranking, threshold)
        
        # Query 2: Worst players by statistic (1 stat, ranking=worst, no other entities)
        if len(statistics) == 1 and ranking == "worst":
            if not players and not teams and not positions and not seasons and not gameweeks:
                return builder.query_worst_players_by_statistic(entities, ranking, threshold)
        
        print("⚠ No matching query found for the given entities")
        return []
            
    finally:
        conn.close()


# ============================================================
# 5. TESTING
# ============================================================

if __name__ == "__main__":
    import preprocessing
    
    print("=" * 70)
    print("BASELINE QUERY TESTING - Queries 1, 2, 3, 4, 8, 9, 10, 16, 17, 23")
    print("=" * 70)
    
    test_cases = [
        "Liverpool vs Newcastle fixtures",    # Query 23: Head-to-head
    ]
    
    for test_query in test_cases:
        print(f"\n🔹 Test Query: '{test_query}'")
        print("-" * 70)
        
        # Get preprocessing output
        preprocessing_result = preprocessing.process_user_query(test_query)
        
        print("\n📥 Preprocessing Output:")
        print(f"Intent:    {preprocessing_result['intent']}")
        print(f"Entities:  {preprocessing_result['entities']}")
        print(f"Ranking:   {preprocessing_result['ranking']}")
        print(f"Threshold: {preprocessing_result['threshold']}")
        
        # Execute baseline query
        print("\n" + "=" * 70)
        results = execute_baseline_query(preprocessing_result)
        print("=" * 70)
        
        print(f"\n✅ Query completed. Retrieved {len(results)} results.")
        print("\n" + "=" * 70)

