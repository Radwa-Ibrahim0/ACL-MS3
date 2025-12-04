import csv
from neo4j import GraphDatabase

# -----------------------------
# Load config.txt credentials
# -----------------------------
def load_config():
    config = {}
    with open("config.txt", "r") as f:
        for line in f:
            key, value = line.strip().split("=", 1)
            config[key] = value
    return config


# -----------------------------
# Helpers to cast values
# -----------------------------
def to_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default


def to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


# -----------------------------
# Create KG class
# -----------------------------
class FPL_KG:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    # -------------------------
    # Constraint Creation
    # -------------------------
    def create_constraints(self):
        constraints = [

            # Seasons
            """CREATE CONSTRAINT season_unique IF NOT EXISTS
               FOR (s:Season)
               REQUIRE s.season_name IS UNIQUE""",

            # Gameweeks
            """CREATE CONSTRAINT gameweek_unique IF NOT EXISTS
               FOR (gw:Gameweek)
               REQUIRE (gw.season, gw.GW_number) IS UNIQUE""",

            # Fixtures
            """CREATE CONSTRAINT fixture_unique IF NOT EXISTS
               FOR (f:Fixture)
               REQUIRE (f.season, f.fixture_number) IS UNIQUE""",

            # Teams
            """CREATE CONSTRAINT team_unique IF NOT EXISTS
               FOR (t:Team)
               REQUIRE t.name IS UNIQUE""",

            # Players
            """CREATE CONSTRAINT player_unique IF NOT EXISTS
               FOR (p:Player)
               REQUIRE (p.player_name, p.player_element) IS UNIQUE""",

            # Positions
            """CREATE CONSTRAINT position_unique IF NOT EXISTS
               FOR (pos:Position)
               REQUIRE pos.name IS UNIQUE"""
        ]

        with self.driver.session() as session:
            for c in constraints:
                session.run(c)
        print("✔ All constraints created.")

    # -------------------------
    # Insert Row Into Graph
    # -------------------------
    def insert_row(self, row):

        query = """
        MERGE (s:Season {season_name: $season})

        MERGE (gw:Gameweek {season: $season, GW_number: $GW})

        MERGE (f:Fixture {season: $season, fixture_number: $fixture})
        SET f.kickoff_time = $kickoff_time

        MERGE (t_home:Team {name: $home_team})
        MERGE (t_away:Team {name: $away_team})

        MERGE (p:Player {player_name: $name, player_element: $element})

        MERGE (pos:Position {name: $position})

        MERGE (s)-[:HAS_GW]->(gw)
        MERGE (gw)-[:HAS_FIXTURE]->(f)
        MERGE (f)-[:HAS_HOME_TEAM]->(t_home)
        MERGE (f)-[:HAS_AWAY_TEAM]->(t_away)
        MERGE (p)-[:PLAYS_AS]->(pos)

        MERGE (p)-[r:PLAYED_IN]->(f)
        SET r.minutes          = $minutes,
            r.goals_scored     = $goals_scored,
            r.assists          = $assists,
            r.total_points     = $total_points,
            r.bonus            = $bonus,
            r.clean_sheets     = $clean_sheets,
            r.goals_conceded   = $goals_conceded,
            r.own_goals        = $own_goals,
            r.penalties_saved  = $penalties_saved,
            r.penalties_missed = $penalties_missed,
            r.yellow_cards     = $yellow_cards,
            r.red_cards        = $red_cards,
            r.saves            = $saves,
            r.bps              = $bps,
            r.influence        = $influence,
            r.creativity       = $creativity,
            r.threat           = $threat,
            r.ict_index        = $ict_index,
            r.form             = $form
        """

        # Build params with proper numeric casting
        params = {
            # identifiers / strings
            "season": row["season"],
            "GW": to_int(row["GW"]),
            "fixture": to_int(row["fixture"]),
            "kickoff_time": row["kickoff_time"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "name": row["name"],
            "element": to_int(row["element"]),
            "position": row["position"],

            # relationship numeric properties
            "minutes": to_int(row["minutes"]),
            "goals_scored": to_int(row["goals_scored"]),
            "assists": to_int(row["assists"]),
            "total_points": to_int(row["total_points"]),
            "bonus": to_int(row["bonus"]),
            "clean_sheets": to_int(row["clean_sheets"]),
            "goals_conceded": to_int(row["goals_conceded"]),
            "own_goals": to_int(row["own_goals"]),
            "penalties_saved": to_int(row["penalties_saved"]),
            "penalties_missed": to_int(row["penalties_missed"]),
            "yellow_cards": to_int(row["yellow_cards"]),
            "red_cards": to_int(row["red_cards"]),
            "saves": to_int(row["saves"]),
            "bps": to_int(row["bps"]),
            "influence": to_float(row["influence"]),
            "creativity": to_float(row["creativity"]),
            "threat": to_float(row["threat"]),
            "ict_index": to_float(row["ict_index"]),
            "form": to_float(row["form"]),
        }

        with self.driver.session() as session:
            session.run(query, params)

    # -------------------------
    # Load CSV
    # -------------------------
    def load_csv(self, path):
        print("Loading CSV:", path)

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.insert_row(row)

        print("✔ Finished loading CSV.")

# -----------------------------
# Run Script
# -----------------------------
if __name__ == "__main__":
    config = load_config()

    kg = FPL_KG(
        uri=config["URI"],
        username=config["USERNAME"],
        password=config["PASSWORD"]
    )

    kg.create_constraints()
    kg.load_csv("")

    kg.close()
    print("✔ Knowledge graph created successfully.")
