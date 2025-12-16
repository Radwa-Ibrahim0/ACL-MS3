import spacy
from spacy.pipeline import EntityRuler
from neo4j import GraphDatabase
import requests
import sys
import re
import time
import json
from typing import Dict, List, Any, Tuple, Optional

# Global debug flag for verbose internal logging
DEBUG = False

# ============================================================
# 1. CONFIGURATION
# ============================================================

def load_config() -> Dict[str, str]:
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
        print("⚠ config.txt not found. Using empty config.")
    return config


# ============================================================
# 2. GLOBALS & CANONICAL DICTIONARIES
# ============================================================

NLP = None
ENTITY_RULER = None
CACHED_PLAYERS: List[str] = []
CACHED_TEAMS: List[str] = []
DATA_LOADED = False
LLM_CLIENT: Optional[Dict[str, str]] = None
CONFIG = load_config()

# Terms that should never be treated as player names even if spaCy flags them
PLAYER_FALSE_POSITIVES = {"far"}

# Canonical position codes must match Position.name in Neo4j
POS_CANON: Dict[str, str] = {
    # Goalkeepers
    "gk": "GK",
    "goalkeeper": "GK",
    "goalkeepers": "GK",
    "goalie": "GK",
    "goalies": "GK",
    "keeper": "GK",
    "keepers": "GK",
    "netminder": "GK",
    "netminders": "GK",
    "shot stopper": "GK",
    "shot stoppers": "GK",
    "custodian": "GK",
    "custodians": "GK",
    "between the sticks": "GK",
    "between the posts": "GK",
    "number 1": "GK",
    "no 1": "GK",
    "no1": "GK",

    # Defenders
    "def": "DEF",
    "defs": "DEF",
    "defender": "DEF",
    "defenders": "DEF",
    "defence": "DEF",
    "defense": "DEF",
    "centre back": "DEF",
    "center back": "DEF",
    "center backs": "DEF",
    "center-backs": "DEF",
    "centre-back": "DEF",
    "center-back": "DEF",
    "cb": "DEF",
    "fullback": "DEF",
    "fullbacks": "DEF",
    "left back": "DEF",
    "left-back": "DEF",
    "right-back": "DEF",
    "right backs": "DEF",
    "left backs": "DEF",
    "right back": "DEF",
    "lb": "DEF",
    "rb": "DEF",
    "wingback": "DEF",
    "wingbacks": "DEF",
    "wb": "DEF",
    "sweeper": "DEF",
    "sweepers": "DEF",

    # Midfielders
    "mid": "MID",
    "mids": "MID",
    "midfielder": "MID",
    "midfielders": "MID",
    "midfield": "MID",
    "cam": "MID",
    "am": "MID",
    "cm": "MID",
    "cdm": "MID",
    "dm": "MID",
    "winger": "MID",
    "wingers": "MID",
    "left winger": "MID",
    "right winger": "MID",
    "lw": "MID",
    "rw": "MID",

    # Forwards
    "fwd": "FWD",
    "fwds": "FWD",
    "forward": "FWD",
    "forwards": "FWD",
    "striker": "FWD",
    "strikers": "FWD",
    "st": "FWD",
    "cf": "FWD",
    "center forward": "FWD",
    "centre forward": "FWD",
}

# Canonical statistics must match relationship properties on [:PLAYED_IN]
STAT_CANON: Dict[str, str] = {
    # Goals / Assists / Points
    "goal": "goals_scored",
    "goals": "goals_scored",
    "goals scored": "goals_scored",
    "scored": "goals_scored",
    "scoring": "goals_scored",
    "netted": "goals_scored",
    "nets": "goals_scored",
    "strike": "goals_scored",
    "strikes": "goals_scored",
    "striking": "goals_scored",
    "hat-trick": "goals_scored",
    "brace": "goals_scored",

    "assist": "assists",
    "assists": "assists",
    "assisting": "assists",
    "chances created": "assists",
    "chance created": "assists",
    "helper": "assists",
    "helpers": "assists",
    "set up": "assists",
    "setups": "assists",
    "set up goal": "assists",
    "set up goals": "assists",

    "points": "total_points",
    "total points": "total_points",
    "fpl points": "total_points",
    "fantasy points": "total_points",
    "returns": "total_points",

    # Bonus & BPS
    "bonus": "bonus",
    "bonus points": "bonus",
    "bps": "bps",
    "bonus point system": "bps",

    # Defensive
    "clean sheet": "clean_sheets",
    "clean sheets": "clean_sheets",
    "clean-sheets": "clean_sheets",
    "clean-sheet": "clean_sheets",
    "cs": "clean_sheets",
    "cleanies": "clean_sheets",

    "goals conceded": "goals_conceded",
    "conceded": "goals_conceded",

    # Penalties
    "penalties saved": "penalties_saved",
    "penalty saved": "penalties_saved",
    "penalty saves": "penalties_saved",
    "penalties missed": "penalties_missed",
    "penalty missed": "penalties_missed",
    "penalty misses": "penalties_missed",

    # Cards
    "yellow card": "yellow_cards",
    "yellow cards": "yellow_cards",
    "yellows": "yellow_cards",
    "red card": "red_cards",
    "red cards": "red_cards",
    "reds": "red_cards",
    "card": "yellow_cards",
    "cards": "yellow_cards",

    # GK
    "saves": "saves",
    "save": "saves",

    # ICT metrics
    "ict index": "ict_index",
    "ict": "ict_index",
    "influence": "influence",
    "creativity": "creativity",
    "threat": "threat",

    # Form
    "form": "form",
    "recent form": "form",
    "good form": "form",
}


# Intent category mapping for easier lookup
# Four canonical intents - aligned with baseline.py query routing
PLAYER_INTENTS = {"PLAYER-RELATED"}
TEAM_INTENTS = {"TEAM-RELATED"}
FIXTURE_INTENTS = {"FIXTURE-RELATED"}
COMPARISON_INTENTS = {"COMPARISON"}

INTENT_LABELS = (
    PLAYER_INTENTS
    | TEAM_INTENTS
    | FIXTURE_INTENTS
    | COMPARISON_INTENTS
)

INTENT_CATEGORIES = {
    "player": PLAYER_INTENTS,
    "team": TEAM_INTENTS,
    "fixture": FIXTURE_INTENTS,
    "comparison": COMPARISON_INTENTS,
}

CURRENT_SEASON = CONFIG.get("CURRENT_SEASON", "2023-24")
try:
    CURRENT_GW = int(CONFIG.get("CURRENT_GW", "1"))
except ValueError:
    CURRENT_GW = 1


# ============================================================
# 3. UTILS
# ============================================================

def unique(seq: List[Any]) -> List[Any]:
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


# ============================================================
# 4. LOAD PLAYERS & TEAMS FROM NEO4J
# ============================================================

def load_knowledge_graph_data(config: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """
    Load Player.player_name and Team.name lists from Neo4j.
    Falls back to mock data if connection/config not available.
    """
    global CACHED_PLAYERS, CACHED_TEAMS, DATA_LOADED

    if DATA_LOADED and CACHED_PLAYERS and CACHED_TEAMS:
        return CACHED_PLAYERS, CACHED_TEAMS

    uri = config.get("URI")
    user = config.get("USERNAME")
    pwd = config.get("PASSWORD")

    if not uri or not user or not pwd:
        print("⚠ No Neo4j config. Using mock data.")
        CACHED_PLAYERS = ["Erling Haaland", "Mohamed Salah", "Bukayo Saka", "Cole Palmer"]
        CACHED_TEAMS = ["Arsenal", "Liverpool", "Manchester City", "Chelsea"]
        DATA_LOADED = True
        return CACHED_PLAYERS, CACHED_TEAMS

    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, pwd))
        driver.verify_connectivity()
        with driver.session() as session:
            print("Connected to Neo4j. Loading players & teams...")
            result_p = session.run("MATCH (p:Player) RETURN DISTINCT p.player_name AS name")
            result_t = session.run("MATCH (t:Team) RETURN DISTINCT t.name AS name")
            CACHED_PLAYERS = [r["name"] for r in result_p if r["name"]]
            CACHED_TEAMS = [r["name"] for r in result_t if r["name"]]
            print(f"✅ Loaded {len(CACHED_PLAYERS)} players, {len(CACHED_TEAMS)} teams from KG.")
    except Exception as e:
        print(f"❌ Database error: {e}")
        if not CACHED_PLAYERS and not CACHED_TEAMS:
            print("⚠ Falling back to mock data.")
            CACHED_PLAYERS = ["Erling Haaland", "Mohamed Salah", "Bukayo Saka", "Cole Palmer"]
            CACHED_TEAMS = ["Arsenal", "Liverpool", "Manchester City", "Chelsea"]
    finally:
        if driver:
            driver.close()
    DATA_LOADED = True
    return CACHED_PLAYERS, CACHED_TEAMS


# ============================================================
# 5. BUILD NLP PIPELINE (spaCy + EntityRuler)
# ============================================================

def build_nlp(players: List[str], teams: List[str]):
    """
    Build spaCy NLP pipeline and EntityRuler patterns from KG vocab.
    """
    global NLP, ENTITY_RULER
    if NLP is not None:
        return NLP

    print("Loading spaCy model...")
    NLP = spacy.load("en_core_web_sm")
    ENTITY_RULER = NLP.add_pipe("entity_ruler", before="ner")
    patterns = []

    #--- Player patterns ---
    seen_single_tokens = set()

    for p in players:
        name_tokens = p.split()

        # Full name pattern
        patterns.append({
            "label": "PLAYER",
            "pattern": [{"LOWER": t.lower()} for t in name_tokens],
            "id": f"PLAYER{p}"
        })

        # Single-token patterns for each name part (Haaland, Saka, Gabriel, etc.)
        for tok in name_tokens:
            tok_lower = tok.lower()
            if len(tok_lower) <= 2:
                continue  # skip very short tokens like "Di"
            if tok_lower in seen_single_tokens:
                continue  # don't add the same token pattern many times

            patterns.append({
                "label": "PLAYER",
                "pattern": [{"LOWER": tok_lower}],
                "id": f"PLAYERTOKEN{tok}"
            })
            seen_single_tokens.add(tok_lower)

    # --- Team patterns ---
    for t in teams:
        team_tokens = t.split()
        patterns.append({
            "label": "TEAM",
            "pattern": [{"LOWER": tok.lower()} for tok in team_tokens],
            "id": f"TEAM_{t}"
        })

    # --- Position patterns ---
    position_terms = set(POS_CANON.keys())
    for term in position_terms:
        tokens = term.split()
        patterns.append({
            "label": "POSITION",
            "pattern": [{"LOWER": tok} for tok in tokens],
            "id": f"POS_{term}"
        })

    # --- Statistic patterns ---
    stat_terms = set(STAT_CANON.keys())
    for term in stat_terms:
        tokens = term.split()
        patterns.append({
            "label": "STATISTIC",
            "pattern": [{"LOWER": tok} for tok in tokens],
            "id": f"STAT_{term}"
        })

    ENTITY_RULER.add_patterns(patterns)
    print(f"✅ EntityRuler patterns added: {len(patterns)}")
    return NLP


# ============================================================
# 6. ENTITY EXTRACTION (aligned with KG)
# ============================================================

def previous_season_from_str(season_str: str) -> str:
    """
    Given a season string like "2023-24" or "'2023-24",
    return the previous season string in the same style,
    e.g. "2022-23" or "'2022-23".
    """
    # Preserve leading apostrophe style if present
    prefix = "'" if season_str.strip().startswith("'") else ""

    m = re.search(r"(20\d{2})[-/](\d{2})", season_str)
    if not m:
        return season_str  # fallback if pattern unexpected

    start_year = int(m.group(1))        # e.g. 2023
    prev_start = start_year - 1         # e.g. 2022
    prev_end2 = str(start_year)[-2:]    # e.g. "23"

    return f"{prefix}{prev_start}-{prev_end2}"


def parse_season_strings(text: str) -> List[str]:
    """
    Extract season identifiers from user text.

    Supports numeric formats like:
      2022-23, '2022-23, 2022/23
      2022-2023, '2022-2023, 2022/2023
      2022  (single year → 2022-23)

    Also supports relative phrases like:
      "this season", "current season", "ongoing season"
      "last season", "previous season", "prior season", "past season"

    Normalizes everything into the same style as your KG, using
    the CURRENT_SEASON value from config as reference
    (so make sure CURRENT_SEASON is in the same format your KG uses).
    """
    seasons: List[str] = []

    # --- CASE 1: Explicit short formats like 2022-23 or '2022-23 or 2022/23 ---
    for m in re.finditer(r"\b'?(20\d{2})[-/](\d{2})\b", text):
        start = m.group(1)   # 2022
        end2 = m.group(2)    # 23
        seasons.append(f"{start}-{end2}")

    # --- CASE 2: Full year formats like 2022-2023 or '2022-2023 or 2022/2023 ---
    for m in re.finditer(r"\b'?(20\d{2})[-/](20\d{2})\b", text):
        start = m.group(1)   # 2022
        end_full = m.group(2)  # 2023
        end2 = end_full[-2:]   # "23"
        seasons.append(f"{start}-{end2}")

    # --- CASE 3: Single year like "2022" or "'2022" ---
    # (This will also fire on "2022-2023", but unique() will dedupe.)
    for m in re.finditer(r"\b'?(20\d{2})\b", text):
        year = m.group(1)  # e.g. 2022
        next_year_two_digits = str(int(year) + 1)[-2:]  # "23"
        seasons.append(f"{year}-{next_year_two_digits}")

    # --- CASE 4: Phrases like "this season", "current season", "ongoing season" ---
    lower = text.lower()
    global CURRENT_SEASON

    # Ensure CURRENT_SEASON is set; fallback is OK
    current_season_str = CURRENT_SEASON

    if re.search(r"\b(this|current|ongoing|present)\s+season\b", lower):
        seasons.append(current_season_str)

    # --- CASE 5: Phrases like "last season", "previous season", "prior season", "past season" ---
    if re.search(r"\b(last|previous|prior|past)\s+season\b", lower):
        seasons.append(previous_season_from_str(current_season_str))

    # --- CASE 6: Phrases like "so far", "until now", "up till now" imply current season ---
    if re.search(r"\b(so\s+far|until\s+now|till\s+now|up\s+to\s+now)\b", lower):
        if current_season_str not in seasons:
            seasons.append(current_season_str)

    return unique(seasons)


def parse_gameweeks(text: str) -> List[int]:
    """
    Extract explicit GW numbers and handle relative phrases:

    'GW 12', 'gameweek 5'
    'this/current/ongoing/present gw/gameweek'
    'last/previous/prior/past gw/gameweek'
    'next/upcoming/following gw/gameweek'
    'up till this gameweek', 'up to this gameweek', 'so far', 'until now', 'up till now', 'up to now'-> [1, 2, ..., CURRENT_GW]
    """
    gws: List[int] = []

    # Explicit: GW 12, gw12, gameweek 12
    for m in re.finditer(r"(?i)\b(?:gw|gameweek)\s*([0-9]{1,2})\b", text):
        try:
            gws.append(int(m.group(1)))
        except ValueError:
            pass

    lower = text.lower()

    # Check for "up till/until this gameweek", "so far", "until now" patterns FIRST
    # These return a range from 1 to CURRENT_GW
    range_patterns = [
        r"\b(up\s+till?|until|up\s+to)\s+(this|current)?\s*(gw|gameweek)\b",
        r"\bso\s+far\b",
        r"\buntil\s+now\b",
        r"\btill\s+now\b",
        r"\bup\s+to\s+now\b",
        r"\bthrough(out)?\s+(this|current)?\s*(gw|gameweek|season)?\b",
    ]
    
    for pattern in range_patterns:
        if re.search(pattern, lower):
            # Return all gameweeks from 1 to CURRENT_GW
            gws = list(range(1, CURRENT_GW + 1))
            return unique(gws)

    # Relative: this/current/ongoing/present gameweek
    if re.search(r"\b(this|current|ongoing|present)\s+(gw|gameweek)\b", lower):
        gws.append(CURRENT_GW)

    # Relative: last/previous/prior/past gameweek
    if re.search(r"\b(last|previous|prior|past)\s+(gw|gameweek)\b", lower):
        gws.append(max(1, CURRENT_GW - 1))

    # Relative: next/upcoming/following gameweek
    if re.search(r"\b(next|upcoming|following)\s+(gw|gameweek)\b", lower):
        gws.append(CURRENT_GW + 1)

    return unique(gws)


def extract_entities(query: str, nlp) -> Dict[str, List[Any]]:
    """
    Extract entities aligned with your KG:
      - Player (Player.player_name)
      - Team (Team.name)
      - Position (Position.name: GK/DEF/MID/FWD)
      - Statistic (relationship props: goals_scored, total_points, etc.)
      - Season (Season.season_name)
      - Gameweek (Gameweek.GW_number)
    """
    doc = nlp(query)

    entities: Dict[str, List[Any]] = {
        "Player": [],
        "Team": [],
        "Position": [],
        "Statistic": [],
        "Season": [],
        "Gameweek": [],
    }

    for ent in doc.ents:
        text_raw = ent.text
        text_norm = text_raw.lower().strip()

        if ent.label_ == "PLAYER":
            if text_norm in PLAYER_FALSE_POSITIVES:
                continue
            if text_raw not in entities["Player"]:
                entities["Player"].append(text_raw)

        elif ent.label_ == "TEAM":
            if text_raw not in entities["Team"]:
                entities["Team"].append(text_raw)

        elif ent.label_ == "POSITION":
            canon = POS_CANON.get(text_norm)
            if canon and canon not in entities["Position"]:
                entities["Position"].append(canon)

        elif ent.label_ == "STATISTIC":
            canon = STAT_CANON.get(text_norm)
            if canon and canon not in entities["Statistic"]:
                entities["Statistic"].append(canon)

    # Seasons and GWs via regex
    seasons = parse_season_strings(query)
    gws = parse_gameweeks(query)

    for s in seasons:
        if s not in entities["Season"]:
            entities["Season"].append(s)

    for gw in gws:
        if gw not in entities["Gameweek"]:
            entities["Gameweek"].append(gw)

    # Final de-dup (just in case)
    for key in entities:
        entities[key] = unique(entities[key])

    return entities


# ============================================================
# 6.b RANKING AND THRESHOLD EXTRACTION (NEW)
# ============================================================

def extract_limit(query: str) -> Optional[int]:
    """
    Extract limit/count parameter from query.
    
    Returns:
        int - the number of results requested (e.g., "top 10" -> 10)
        None - if no specific limit is mentioned, or if threshold implies all results
    
    Examples:
        "Top 10 players" -> 10
        "Top 15 players by ICT" -> 15
        "Bottom 5 players" -> 5
        "Players with at least 150 points" -> None (return all matching)
    """
    query_lower = query.lower()
    
    # Patterns for extracting limits
    limit_patterns = [
        r"top\s+(\d+)",
        r"bottom\s+(\d+)",
        r"best\s+(\d+)",
        r"worst\s+(\d+)",
        r"first\s+(\d+)",
        r"last\s+(\d+)",
        r"(\d+)\s+best",
        r"(\d+)\s+worst",
        r"(\d+)\s+top",
        r"show\s+(\d+)",
        r"list\s+(\d+)",
        r"get\s+(\d+)",
        r"find\s+(\d+)",
    ]
    
    for pattern in limit_patterns:
        match = re.search(pattern, query_lower)
        if match:
            return int(match.group(1))
    
    return None


def extract_ranking(query: str) -> Optional[str]:
    """
    Extract ranking parameter from query.
    
    Returns:
        "best" - if query uses: best, top, most, highest
        "worst" - if query uses: worst, least, bottom, lowest
        None - otherwise
    """
    query_lower = query.lower()
    
    best_keywords = ["best", "top", "most", "highest", "leading", "greatest", "maximum"]
    worst_keywords = ["worst", "least", "bottom", "lowest", "fewest", "minimum", "poorest"]
    
    # Check for worst first (more specific in some contexts)
    if any(kw in query_lower for kw in worst_keywords):
        return "worst"
    
    if any(kw in query_lower for kw in best_keywords):
        return "best"
    
    return None


def extract_threshold(query: str) -> Optional[Dict[str, Any]]:
    """
    Extract threshold parameter from query.
    
    Returns a dict with:
        - stat: which stat the threshold applies to
        - operator: one of "LT", "LE", "GT", "GE"
        - value: numeric value
    
    Or None if no threshold detected.
    """
    query_lower = query.lower()
    
    # Map natural language operators to codes
    operator_patterns = [
        # Greater than
        (r"more than\s+(\d+(?:\.\d+)?)", "GT"),
        (r"greater than\s+(\d+(?:\.\d+)?)", "GT"),
        (r"above\s+(\d+(?:\.\d+)?)", "GT"),
        (r"over\s+(\d+(?:\.\d+)?)", "GT"),
        (r"exceeds?\s+(\d+(?:\.\d+)?)", "GT"),
        (r">\s*(\d+(?:\.\d+)?)", "GT"),
        
        # Greater than or equal
        (r"at least\s+(\d+(?:\.\d+)?)", "GE"),
        (r"minimum of\s+(\d+(?:\.\d+)?)", "GE"),
        (r"minimum\s+(\d+(?:\.\d+)?)", "GE"),
        (r"no less than\s+(\d+(?:\.\d+)?)", "GE"),
        (r">=\s*(\d+(?:\.\d+)?)", "GE"),
        
        # Less than
        (r"less than\s+(\d+(?:\.\d+)?)", "LT"),
        (r"fewer than\s+(\d+(?:\.\d+)?)", "LT"),
        (r"below\s+(\d+(?:\.\d+)?)", "LT"),
        (r"under\s+(\d+(?:\.\d+)?)", "LT"),
        (r"<\s*(\d+(?:\.\d+)?)", "LT"),
        
        # Less than or equal
        (r"at most\s+(\d+(?:\.\d+)?)", "LE"),
        (r"maximum of\s+(\d+(?:\.\d+)?)", "LE"),
        (r"maximum\s+(\d+(?:\.\d+)?)", "LE"),
        (r"no more than\s+(\d+(?:\.\d+)?)", "LE"),
        (r"<=\s*(\d+(?:\.\d+)?)", "LE"),
    ]
    
    for pattern, operator in operator_patterns:
        match = re.search(pattern, query_lower)
        if match:
            value = float(match.group(1))
            # Try to convert to int if it's a whole number
            if value == int(value):
                value = int(value)
            
            # Try to identify which stat the threshold applies to
            stat = identify_threshold_stat(query_lower)
            
            return {
                "stat": stat,
                "operator": operator,
                "value": value
            }
    
    return None


def identify_threshold_stat(query_lower: str) -> str:
    """
    Identify which stat a threshold applies to based on query context.
    Returns the canonical stat name.
    """
    # Check for stat keywords near the threshold
    stat_priority = [
        ("goal", "goals_scored"),
        ("assist", "assists"),
        ("point", "total_points"),
        ("clean sheet", "clean_sheets"),
        ("save", "saves"),
        ("bonus", "bonus"),
        ("form", "form"),
        ("threat", "threat"),
        ("creativity", "creativity"),
        ("influence", "influence"),
        ("ict", "ict_index"),
        ("yellow", "yellow_cards"),
        ("red", "red_cards"),
        ("minute", "minutes"),
        ("concede", "goals_conceded"),
    ]
    
    for keyword, stat in stat_priority:
        if keyword in query_lower:
            return stat
    
    # Default to total_points
    return "total_points"


def rule_based_intent(query: str, raw_entities: Dict[str, List[Any]]) -> str:
    """Lightweight fallback intent classifier when the LLM is unavailable.
    
    Returns one of 4 intents aligned with baseline.py:
    - PLAYER-RELATED: Player stats, performance queries, rankings
    - TEAM-RELATED: Team-based queries (players from a team, team stats)
    - FIXTURE-RELATED: Fixture/match/gameweek queries, head-to-head between teams
    - COMPARISON: Compare two players
    """
    q = query.lower()
    players = raw_entities.get("Player", []) or []
    teams = raw_entities.get("Team", []) or []
    gameweeks = raw_entities.get("Gameweek", []) or []

    def has_any(keywords: List[str]) -> bool:
        return any(kw in q for kw in keywords)

    # COMPARISON: Compare two players (Query 16 in baseline.py)
    comparison_keywords = ["compare", "comparison", "vs", "versus", " v ", " v."]
    if len(players) == 2 and has_any(comparison_keywords):
        return "COMPARISON"
    
    # Also trigger comparison if 2 players detected with comparison-like phrasing
    if len(players) == 2 and has_any(["than", "better", "worse", "difference", "against"]):
        return "COMPARISON"

    # FIXTURE-RELATED: fixture/match/gameweek queries, head-to-head between teams
    fixture_keywords = [
        "fixture",
        "fixtures",
        "match",
        "matches",
        "schedule",
        "playing against",
        "play against",
        "next game",
        "upcoming game",
        "head to head",
        "head-to-head",
        "h2h",
    ]
    
    # If 2 teams and fixture-related keywords -> FIXTURE-RELATED (for head-to-head)
    if len(teams) == 2 and has_any(fixture_keywords + ["vs", "versus"]):
        return "FIXTURE-RELATED"
    
    # If gameweek mentioned with fixture keywords -> FIXTURE-RELATED
    if gameweeks and has_any(["fixture", "fixtures", "match", "matches", "schedule"]):
        return "FIXTURE-RELATED"
    
    # TEAM-RELATED: Team-focused queries without specific players
    if teams and not players:
        return "TEAM-RELATED"
    
    # Default: PLAYER-RELATED (player stats, rankings)
    return "PLAYER-RELATED"

# ============================================================
# 7. INTENT CLASSIFICATION (LLM → normalized label)
# ============================================================


def init_llm_client(config: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Initialise an OpenRouter client using tngtech/deepseek-r1t2-chimera:free.

    Expects OPENROUTER_API_KEY in config.txt.
    Returns a small dict describing the client, or None if not configured.
    """
    global LLM_CLIENT, CONFIG

    if not CONFIG:
        CONFIG = load_config()

    if LLM_CLIENT is not None:
        return LLM_CLIENT

    merged_config: Dict[str, str] = {}
    merged_config.update(CONFIG)
    merged_config.update(config or {})

    api_key = merged_config.get("OPENROUTER_API_KEY")
    if not api_key:
        print("⚠ No OPENROUTER_API_KEY in config. LLM features will be disabled.")
        LLM_CLIENT = None
        return LLM_CLIENT

    LLM_CLIENT = {
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "api_key": api_key,
        "model": "tngtech/deepseek-r1t2-chimera:free",
    }
    return LLM_CLIENT


def call_openrouter(client: Dict[str, str], prompt: str) -> str:
    """Call OpenRouter chat completions with the given prompt and return text."""
    headers = {
        "Authorization": f"Bearer {client['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": client["model"],
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }

    resp = requests.post(client["base_url"], headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Unexpected OpenRouter response format: {data}") from e

def get_intent_and_entities_llm(
    query: str,
    raw_entities: Dict[str, List[Any]],
    teams: List[str],
    current_season: str,
    current_gw: int,
    config: Dict[str, str],
) -> Tuple[str, Dict[str, List[Any]], Optional[str], Optional[Dict[str, Any]]]:
    """Single LLM call to infer intent, refine entities, and extract ranking/threshold.

    Returns (intent, entities, ranking, threshold). On any error or missing config, 
    falls back to rule-based intent and raw_entities with rule-based ranking/threshold.
    """
    # Extract ranking and threshold using rule-based methods first
    ranking = extract_ranking(query)
    threshold = extract_threshold(query)

    allowed_positions = ["GK", "DEF", "MID", "FWD"]
    allowed_stats = sorted(set(STAT_CANON.values()))
    fallback_intent = rule_based_intent(query, raw_entities)

    client = init_llm_client(config)
    if not client:
        if DEBUG:
            print("LLM client unavailable; returning rule-based intent and entities.")
        return fallback_intent, raw_entities, ranking, threshold

    system_instruction = f"""
You are an FPL (Fantasy Premier League) assistant.

You must do FOUR tasks in ONE response:

1) INTENT CLASSIFICATION
   Choose exactly ONE intent from this list:
   - PLAYER-RELATED: Queries about player statistics, performance, rankings (single player or general)
   - TEAM-RELATED: Queries focused on a specific team's players or team-level statistics
   - FIXTURE-RELATED: Queries about fixtures, matches, schedules, head-to-head between teams, gameweek fixtures
   - COMPARISON: Queries that compare two specific players (e.g., "Compare Salah and Haaland")

2) ENTITY REFINEMENT
   Based on the query AND the provided raw_entities and teams list:
     - Interpret relative season phrases (this season, last season, two seasons ago)
       using CURRENT_SEASON = '{current_season}'.
     - Interpret relative gameweek phrases using CURRENT_GW = {current_gw}.
     - Map team names to the closest valid team name from the teams list.
     - Use ONLY these position codes: {allowed_positions}.
     - Use ONLY these statistic names: {allowed_stats}.

3) RANKING PARAMETER
   Determine if the query asks for best or worst ranking:
     - best if query uses: best, top, most, highest, leading, greatest or if the query implies so
     - worst if query uses: worst, least, bottom, lowest, fewest or if the query implies so
     - null otherwise

4) THRESHOLD PARAMETER
   If the query specifies a numeric threshold condition:
     - Extract the stat, operator, and value
     - Operators: GT (greater than), GE (>=), LT (less than), LE (<=), EQ (=)
     - Example: players with more than 3 goals -> stat=goals_scored, operator=GT, value=3
     - Example: players with 50 points -> stat=total_points, operator=EQ, value=50
     - Return null if no threshold detected
     
Your output MUST be a single JSON object with exactly these keys:
  intent: string (PLAYER-RELATED, TEAM-RELATED, FIXTURE-RELATED, or COMPARISON)
  entities: object with Player, Team, Position, Statistic, Season, Gameweek arrays
  ranking: best or worst or null
  threshold: object with stat, operator, value or null

Season rules:
- Seasons use the format 'YYYY-YY' (e.g., '2022-23', '2020-21').
- Seasons could only be (2021-22) or (2022-23) only nothing else.
- CURRENT_SEASON = '{current_season}'.
- Compute relative references like but not limited to:
    - 'this season', 'current season'  -> CURRENT_SEASON
    - 'last season', 'previous season' -> the season immediately before CURRENT_SEASON
    - 'two seasons ago'                -> two seasons before CURRENT_SEASON
  Example: If CURRENT_SEASON = '2022-23':
    - this season        -> '2022-23'
    - last season        -> '2021-22'
    - two seasons ago    -> (not applicable, leave empty)
    - all seasons        -> ['2021-22', '2022-23']

Gameweek rules:
- Gameweeks are positive integers.
- CURRENT_GW = {current_gw}.
- 'this gameweek', 'this gw', 'current gw'      -> CURRENT_GW
- 'last gw', 'previous gw', 'last gameweek'     -> CURRENT_GW - 1
- 'next gw', 'upcoming gw', 'next gameweek'     -> CURRENT_GW + 1
- 'up till X gameweek'                          -> all GWs from 1 to X
- 'so far' or 'up till this gameweek'           -> all GWs from 1 to CURRENT_GW
- but not limited to the above examples.

PLAYER RESOLUTION RULES:
1. If the user mentions a player *not detected by spaCy*, you must add that player.
2. Users may use nicknames, abbreviations, initials, or short forms (e.g., "KDB", "CR7", "Mo", "Saka", "Gabby", "Licha").
   - You must resolve these to the **correct full player name** from the KG.
   - Then you must add **only the last name** to the final Player list.
     - Example: "KDB" -> "Kevin De Bruyne" -> add **"De Bruyne"**  
     - Example: "Mo" -> "Mohamed Salah" -> add **"Salah"**
3. If the user already mentions a last name (e.g., “Salah”), do NOT change it.
   - Keep the extracted player exactly as it appears.
4. Never include the short form, nickname, or initials in the final extraction.
   - Only the canonical last name derived from the KG.
5. Remove duplicates from the final Player list.

Your output MUST be a single JSON object with exactly these keys:
  "intent": string (PLAYER-RELATED, TEAM-RELATED, FIXTURE-RELATED, or COMPARISON)
  "entities": {{
      "Player":   list of strings,
      "Team":     list of strings,
      "Position": list of strings (subset of {allowed_positions}),
      "Statistic":list of strings (subset of {allowed_stats}),
      "Season":   list of strings (each 'YYYY-YY'),
      "Gameweek": list of integers
  }},
  "ranking": {{
    "stat": string (subset of {allowed_stats}),
    "value": "best" | "worst"
  }} | null,
  "threshold": {{
        "stat": string (subset of {allowed_stats}),
        "operator": "GT/LT/GE/LE/EQ",
        "value": number
  }} | null

Do NOT include extra keys.
Do NOT include explanations or comments.
Only output valid JSON.
"""

    payload = {
        "query": query,
        "raw_entities": raw_entities,
        "teams": teams,
    }

    user_prompt = system_instruction + "\n\nINPUT:\n" + json.dumps(payload)

    try:
        raw = call_openrouter(client, user_prompt)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(json)?", "", raw).strip()
            if raw.endswith("```"):
                raw = raw[:-3].strip()
        data = json.loads(raw)
    except Exception as e:
        print(f"⚠ Combined LLM intent+entity error: {e}. Falling back to rule-based.")
        return fallback_intent, raw_entities, ranking, threshold

    # Extract intent
    intent = data.get("intent") or fallback_intent
    if intent not in INTENT_LABELS:
        intent = fallback_intent

    # Extract entities, fall back to raw_entities per key if missing
    llm_entities = data.get("entities", {}) or {}
    final_entities: Dict[str, List[Any]] = {}
    for key in ["Player", "Team", "Position", "Statistic", "Season", "Gameweek"]:
        val = llm_entities.get(key)
        if val is None or (isinstance(val, list) and not val):
            val = raw_entities.get(key, [])
        if val is None:
            val = []
        if not isinstance(val, list):
            val = [val]
        cleaned = [item for item in val if item not in (None, "")]
        final_entities[key] = unique(cleaned)

    # Extract ranking from LLM response (override rule-based if present)
    llm_ranking = data.get("ranking")
    if llm_ranking in ["best", "worst"]:
        ranking = llm_ranking
    
    # Extract threshold from LLM response (override rule-based if present)
    llm_threshold = data.get("threshold")
    if llm_threshold and isinstance(llm_threshold, dict):
        if all(k in llm_threshold for k in ["stat", "operator", "value"]):
            threshold = llm_threshold

    if DEBUG:
        print(f"🔍 Raw entities:   {raw_entities}")
        print(f"🤖 LLM-combined: intent={intent}, entities={final_entities}, ranking={ranking}, threshold={threshold}")

    return intent, final_entities, ranking, threshold


# ============================================================
# 8. TOP-LEVEL FUNCTION
# ============================================================

def process_user_query(query: str) -> Dict[str, Any]:
    """
    End-to-end preprocessing:
      - ensures KG vocab is loaded
      - builds NLP pipeline
      - extracts entities (spaCy + rules)
      - refines entities with LLM (to fill gaps / normalize)
      - classifies intent (with entity-aware fallback)
      - extracts ranking, threshold, and limit parameters

    Returns:
      {
        "query": <original text>,
        "intent": <normalized intent>,
        "entities": {
           "Player": [...],
           "Team": [...],
           "Position": [...],   # GK/DEF/MID/FWD
           "Statistic": [...],  # goals_scored, total_points, ...
           "Season": [...],     # "2023-24"
           "Gameweek": [...],   # integers
        },
        "ranking": "best" | "worst" | None,
        "threshold": {"stat": str, "operator": str, "value": number} | None,
        "limit": int | None  # Number of results to return, None means use default or all
      }
    """
    global NLP, CONFIG

    if not CONFIG:
        CONFIG = load_config()

    players, teams = load_knowledge_graph_data(CONFIG)
    nlp = build_nlp(players, teams)

    # 1) spaCy + rule-based extraction
    raw_entities = extract_entities(query, nlp)

    # 2) Combined LLM call for intent + refined entities + ranking + threshold (with robust fallback)
    current_season = CONFIG.get("CURRENT_SEASON", CURRENT_SEASON)
    try:
        current_gw = int(CONFIG.get("CURRENT_GW", str(CURRENT_GW)))
    except ValueError:
        current_gw = CURRENT_GW

    intent, entities, ranking, threshold = get_intent_and_entities_llm(
        query=query,
        raw_entities=raw_entities,
        teams=teams,
        current_season=current_season,
        current_gw=current_gw,
        config=CONFIG,
    )

    # 3) Extract limit from query (rule-based)
    limit = extract_limit(query)
    
    # If threshold is present and no explicit limit, return all matches (None means no limit)
    # If threshold is present WITH explicit limit, use that limit
    # If no threshold and no limit, baseline will use default (10)

    return {
        "query": query,
        "intent": intent,
        "entities": entities,
        "ranking": ranking,
        "threshold": threshold,
        "limit": limit,
    }


# ============================================================
# 9. LOCAL TESTING
# ============================================================

if __name__ == "__main__":
    print("Running local tests for preprocessing...\n")

    test_cases = [
        # "Who is the best striker for goals scored this season?",
        # "Compare Saliba and Gabriel clean sheets in 2023-24.",
        # "How many goals did Saka score two seasons ago?",
        # "What happened in Man United's previous gameweek?",
        #"Suggest midfielders in good form this season.",
        # "Show me Spurs fixtures next gw.",
        # "Which defenders should I consider picking up for the upcoming gameweek?",
        # "Top midfielders by total points this season.",
        "Which Tottenham defenders have been in good form this season?",
         #"Which players scored more than 5 goals up till this gameweek?",
        # "which players have the least points so far?",
        # "Show me Brightin's fixtures in 2 gameweeks from now.",
        # "Which Tottenham defenders have been in good form this season?",
        # "Give me players with same number of goals as Haaland.",
        # "i want players with 0 cleansheets",
        # "Show me the forwards with the most goals and assists across all seasons.",
        # "Which defenders have the most clean sheets and total points?",
        # "Top forwards by goals scored.",
        # "Who are the best attacking players in terms of goals and threat combined?",
        # "Which players scored more than 5 goals up till this gameweek?",
        # "which players have the least points so far?",
        # "How many goals did far score?",
        # "Top 15 players by ICT index in the 2022-23 season",
        # "Show me Brightin's fixtures in 2 gameweeks from now.",
        # "Which Tottenham defenders have been in good form this season?",
        # "Give me players with same number of goals as Haaland.",
        # "i want players with 0 cleansheets",
    ]

    for q in test_cases:
        res = process_user_query(q)
        print(f"Input:    {res['query']}")
        print(f"Intent:   {res['intent']}")
        print(f"Entities: {res['entities']}")
        print(f"Ranking:  {res['ranking']}")
        print(f"Threshold:{res['threshold']}")
        print(f"Limit:    {res['limit']}")
        print("-" * 60)

        # For local testing you can comment this out to speed things up
        # or keep it if you are worried about API rate limits.
        # time.sleep(1)
