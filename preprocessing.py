import spacy
from spacy.pipeline import EntityRuler
from neo4j import GraphDatabase
import google.generativeai as genai
import sys
import re
import time
from typing import Dict, List, Any, Tuple, Optional

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
GEMINI_MODEL = None
CONFIG = load_config()

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

INTENT_LABELS = {
    "player performance",
    "team analysis",
    "fixture query",
    "recommendation",
    "statistics",
    "trivia"
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

    return unique(seasons)

def parse_gameweeks(text: str) -> List[int]:
    """
    Extract explicit GW numbers and handle relative phrases:
        
    'GW 12', 'gameweek 5'
    'this/current/ongoing/present gw/gameweek'
    'last/previous/prior/past gw/gameweek'
    'next/upcoming/following gw/gameweek'
    """
    gws: List[int] = []

    # Explicit: GW 12, gw12, gameweek 12
    for m in re.finditer(r"(?i)\b(?:gw|gameweek)\s*([0-9]{1,2})\b", text):
        try:
            gws.append(int(m.group(1)))
        except ValueError:
            pass

    lower = text.lower()

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
# 7. INTENT CLASSIFICATION (LLM → normalized label)
# ============================================================

def init_gemini_model(config: Dict[str, str]):
    global GEMINI_MODEL
    if GEMINI_MODEL is not None:
        return GEMINI_MODEL

    api_key = config.get("KEY")
    if not api_key:
        print("⚠ No GEMINI_API_KEY in config. Intent will default to 'player performance'.")
        GEMINI_MODEL = None
        return GEMINI_MODEL

    genai.configure(api_key=api_key)
    GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash")
    return GEMINI_MODEL


def normalize_intent(raw_text: str) -> str:
    """
    Normalize the raw text from LLM into one of INTENT_LABELS.
    """
    if not raw_text:
        return "player performance"

    text = raw_text.strip().lower()
    text = text.splitlines()[0]
    text = re.sub(r"^intent\s*:\s*", "", text)
    text = text.replace("the intent is", "").strip(": ").strip()

    # Exact or substring match
    for intent in INTENT_LABELS:
        if intent in text:
            return intent

    # Heuristics
    if "recommend" in text or "suggest" in text or "pick" in text:
        return "recommendation"
    if "fixture" in text or "gw" in text or "gameweek" in text:
        return "fixture query"
    if "team" in text or "defence" in text or "defense" in text:
        return "team analysis"
    if "stat" in text or "stats" in text:
        return "statistics"
    if "trivia" in text or "quiz" in text:
        return "trivia"

    return "player performance"


def rule_based_intent(query: str, entities: Dict[str, List[Any]]) -> str:
    """
    Rule-based fallback intent classifier using query keywords and extracted entities.
    Used when LLM is unavailable or fails.
    """
    query_lower = query.lower()
    
    has_player = bool(entities.get("Player"))
    has_team = bool(entities.get("Team"))
    has_position = bool(entities.get("Position"))
    has_stat = bool(entities.get("Statistic"))
    has_season = bool(entities.get("Season"))
    has_gw = bool(entities.get("Gameweek"))
    
    # Fixture-related keywords
    fixture_keywords = [
        "fixture", "fixtures", "match", "matches", "game", "games",
        "playing", "plays", "vs", "versus", "against", "opponent",
        "upcoming", "next match", "schedule"
    ]
    
    # Team analysis keywords
    team_keywords = [
        "happened", "what happened", "how did", "performance",
        "results", "result", "scored", "conceded", "won", "lost", "drew",
        "show me", "list"
    ]
    
    # Recommendation keywords
    recommend_keywords = [
        "recommend", "suggest", "pick", "should i", "consider",
        "buy", "transfer", "best", "top", "good", "form",
        "who should", "which", "differential"
    ]
    
    # Statistics keywords  
    stat_keywords = [
        "most", "highest", "top", "leading", "best", "worst",
        "total", "average", "compare", "comparison", "leader", "leaders"
    ]
    
    # Check for fixture queries
    if any(kw in query_lower for kw in fixture_keywords):
        if has_team:
            return "fixture query"
    
    # Check for team analysis - team mentioned with analysis-type questions
    if has_team and not has_player:
        # Team + gameweek usually means "what happened in that GW"
        if has_gw and any(kw in query_lower for kw in team_keywords):
            return "team analysis"
        # Team + season with "show", "list", "matches" = fixture/team analysis
        if has_season and any(kw in query_lower for kw in ["show", "list", "matches", "fixtures", "games"]):
            return "fixture query"
        # Generic team questions
        if any(kw in query_lower for kw in team_keywords):
            return "team analysis"
    
    # Check for recommendations
    if any(kw in query_lower for kw in recommend_keywords):
        # "which defenders should I pick" = recommendation
        if has_position and not has_player:
            return "recommendation"
        # "best players" type queries
        if "best" in query_lower or "top" in query_lower:
            if has_position:
                return "recommendation"
    
    # Check for statistics/comparison queries
    if has_player and len(entities.get("Player", [])) >= 2:
        # Comparing multiple players
        return "player performance"
    
    if any(kw in query_lower for kw in stat_keywords):
        if has_stat or has_position:
            return "statistics"
    
    # Player-specific queries
    if has_player:
        return "player performance"
    
    # Default based on what entities we have
    if has_team:
        return "team analysis"
    if has_position:
        return "statistics"
    
    return "player performance"


def get_intent(query: str, config: Dict[str, str], entities: Optional[Dict[str, List[Any]]] = None) -> str:
    """
    Get intent using LLM with rule-based fallback.
    If entities are provided, uses them for better fallback classification.
    """
    model = init_gemini_model(config)
    
    # If no model available, use rule-based fallback
    if model is None:
        if entities:
            return rule_based_intent(query, entities)
        return "player performance"

    prompt = (
        "You are an intent classifier for a Fantasy Premier League assistant.\n"
        "Given the user query, choose exactly ONE of these intents:\n"
        "  - player performance: Questions about specific player stats, scores, or comparisons\n"
        "  - team analysis: Questions about a team's overall performance, results, or what happened in a match\n"
        "  - fixture query: Questions about upcoming matches, schedules, or fixture lists\n"
        "  - recommendation: Questions asking for suggestions on which players to pick or transfer\n"
        "  - statistics: General league-wide statistics or leaderboards\n"
        "  - trivia: Fun facts or historical trivia questions\n\n"
        "Respond with ONLY the intent string, nothing else.\n\n"
        f"User query: {query}"
    )

    try:
        resp = model.generate_content(prompt)
        raw = resp.text if hasattr(resp, "text") else str(resp)
        return normalize_intent(raw)
    except Exception as e:
        print(f"⚠ Intent classification error: {e}")
        # Use rule-based fallback when LLM fails
        if entities:
            fallback_intent = rule_based_intent(query, entities)
            print(f"⚠ Using rule-based fallback intent: {fallback_intent}")
            return fallback_intent
        return "player performance"


# ============================================================
# 8. TOP-LEVEL FUNCTION
# ============================================================

def process_user_query(query: str) -> Dict[str, Any]:
    """
    End-to-end preprocessing:
      - ensures KG vocab is loaded
      - builds NLP pipeline
      - extracts entities
      - classifies intent (with entity-aware fallback)

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
        }
      }
    """
    global NLP, CONFIG

    if not CONFIG:
        CONFIG = load_config()

    players, teams = load_knowledge_graph_data(CONFIG)
    nlp = build_nlp(players, teams)

    # Extract entities FIRST so we can use them for intent classification fallback
    entities = extract_entities(query, nlp)
    
    # Pass entities to get_intent for better fallback classification
    intent = get_intent(query, CONFIG, entities)

    return {
        "query": query,
        "intent": intent,
        "entities": entities,
    }


# ============================================================
# 9. LOCAL TESTING
# ============================================================

if __name__ == "__main__":
    print("Running local tests for preprocessing...\n")

    test_cases = [
        "Who is the best striker for goals scored this season?",
        "Compare Saliba and Gabriel clean sheets in 2023-24.",
        # "Compare Saliba and Mohamed Salah clean sheets in 2023.",
        # "Give me the fixture difficulty for Manchester City next GW.",
        # "Suggest midfielders in good form for my team.",
        # "How many assists did Saka get last season?",
        # "How many goals did Harry score in gameweek 10?",
        # "Show me stats for Arsenal center backs in the last gameweek.",
        # "Which cheap midfielders in good form should I buy next GW?",
        # "Give me some FPL trivia about Liverpool defenders.",
        # "Give me all players playing as defenders in Liverpool and in Arsenal.",
        # "Give me all players playing as defenders and mids.",
    ]

    for q in test_cases:
        res = process_user_query(q)
        print(f"Input:  {res['query']}")
        print(f"Intent: {res['intent']}")
        print(f"Entities: {res['entities']}")
        print("-" * 60)
        time.sleep(1)  # be nice to the API
