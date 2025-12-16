"""
embedding_minilm.py

Feature Vector Embeddings using Sentence-Transformers (all-MiniLM-L6-v2)

This module implements:
  - 1c: Feature Vector Embeddings for FPL Knowledge Graph nodes
  - 2b: Vector Similarity Search for semantic query matching

Model: all-MiniLM-L6-v2
  - Embedding dimension: 384
  - Fast and lightweight
  - Good for semantic similarity tasks
"""

import sys
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from neo4j import GraphDatabase

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Run: pip install sentence-transformers")


# ============================================================
# 1. CONFIGURATION
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


# ============================================================
# 2. EMBEDDING MODEL CLASS
# ============================================================

class MiniLMEmbedder:
    """
    Feature Vector Embedding using Sentence-Transformers all-MiniLM-L6-v2
    
    This model creates 384-dimensional embeddings that capture semantic meaning
    of text. We use it to embed:
      - Player descriptions (name, position, stats summary)
      - Team descriptions
      - User queries for semantic search
    """
    
    MODEL_NAME = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    
    def __init__(self):
        """Initialize the embedding model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
        
        logger.info(f"Loading SentenceTransformer model: {self.MODEL_NAME}")
        self.model = SentenceTransformer(self.MODEL_NAME)
        logger.info(f"✅ Model loaded. Embedding dimension: {self.EMBEDDING_DIM}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing).
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings.tolist()
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))


# ============================================================
# 3. POSITION-SPECIFIC THRESHOLDS (Empirical Percentiles)
# ============================================================

# Thresholds derived from fpl_two_seasons.csv percentile analysis
# Structure: {position: {metric: {p25, p50, p75, p90}}}
POSITION_THRESHOLDS = {
    "GK": {
        "total_points": {"p25": 0, "p50": 0, "p75": 48, "p90": 164.5},
        "minutes": {"p25": 0, "p50": 0, "p75": 1462.5, "p90": 4032},
        "goals_per90": {"p25": 0, "p50": 0, "p75": 0, "p90": 0},
        "assists_per90": {"p25": 0, "p50": 0, "p75": 0, "p90": 0.01},
        "form": {"p25": 0, "p50": 0, "p75": 0.08, "p90": 0.31},
        "clean_sheets": {"p25": 0, "p50": 0, "p75": 3.25, "p90": 15},
        "saves": {"p25": 0, "p50": 0, "p75": 49.5, "p90": 132.8},
        "bonus": {"p25": 0, "p50": 0, "p75": 5, "p90": 16},
    },
    "DEF": {
        "total_points": {"p25": 0, "p50": 29.5, "p75": 90, "p90": 151},
        "minutes": {"p25": 0, "p50": 1103.5, "p75": 2485.25, "p90": 4054.5},
        "goals_per90": {"p25": 0, "p50": 0, "p75": 0.06, "p90": 0.10},
        "assists_per90": {"p25": 0, "p50": 0.03, "p75": 0.09, "p90": 0.19},
        "form": {"p25": 0, "p50": 0.06, "p75": 0.17, "p90": 0.25},
        "clean_sheets": {"p25": 0, "p50": 2, "p75": 7, "p90": 14},
        "saves": {"p25": 0, "p50": 0, "p75": 0, "p90": 0},
        "bonus": {"p25": 0, "p50": 1, "p75": 6, "p90": 12},
    },
    "MID": {
        "total_points": {"p25": 1, "p50": 27, "p75": 83, "p90": 175},
        "minutes": {"p25": 3, "p50": 653, "p75": 2196, "p90": 3858.6},
        "goals_per90": {"p25": 0, "p50": 0.06, "p75": 0.18, "p90": 0.35},
        "assists_per90": {"p25": 0, "p50": 0.09, "p75": 0.18, "p90": 0.30},
        "form": {"p25": 0, "p50": 0.06, "p75": 0.18, "p90": 0.29},
        "clean_sheets": {"p25": 0, "p50": 1, "p75": 7, "p90": 14},
        "saves": {"p25": 0, "p50": 0, "p75": 0, "p90": 0},
        "bonus": {"p25": 0, "p50": 0, "p75": 5, "p90": 14},
    },
    "FWD": {
        "total_points": {"p25": 0, "p50": 31, "p75": 87, "p90": 166.3},
        "minutes": {"p25": 0, "p50": 588, "p75": 1867.25, "p90": 3259.2},
        "goals_per90": {"p25": 0, "p50": 0.24, "p75": 0.35, "p90": 0.56},
        "assists_per90": {"p25": 0, "p50": 0.11, "p75": 0.24, "p90": 0.34},
        "form": {"p25": 0, "p50": 0.07, "p75": 0.19, "p90": 0.27},
        "clean_sheets": {"p25": 0, "p50": 0, "p75": 6, "p90": 10},
        "saves": {"p25": 0, "p50": 0, "p75": 0, "p90": 0},
        "bonus": {"p25": 0, "p50": 1, "p75": 9.75, "p90": 18.1},
    },
}

# Fallback thresholds for unknown positions (overall percentiles)
DEFAULT_THRESHOLDS = {
    "total_points": {"p25": 0, "p50": 25, "p75": 85, "p90": 166},
    "minutes": {"p25": 0, "p50": 649, "p75": 2254, "p90": 3754.6},
    "goals_per90": {"p25": 0, "p50": 0.03, "p75": 0.14, "p90": 0.31},
    "assists_per90": {"p25": 0, "p50": 0.05, "p75": 0.15, "p90": 0.27},
    "form": {"p25": 0, "p50": 0.06, "p75": 0.18, "p90": 0.27},
    "clean_sheets": {"p25": 0, "p50": 1, "p75": 7, "p90": 13},
    "saves": {"p25": 0, "p50": 0, "p75": 0, "p90": 0},
    "bonus": {"p25": 0, "p50": 0, "p75": 6, "p90": 14},
}


def get_tier_label(value: float, thresholds: Dict[str, float]) -> str:
    """
    Classify a value into a tier based on percentile thresholds.
    
    Returns: 'elite' (>=p90), 'strong' (p75-p90), 'average' (p50-p75), 
             'below_average' (p25-p50), or 'low' (<p25)
    """
    if value >= thresholds["p90"]:
        return "elite"
    elif value >= thresholds["p75"]:
        return "strong"
    elif value >= thresholds["p50"]:
        return "average"
    elif value >= thresholds["p25"]:
        return "below_average"
    else:
        return "low"


# ============================================================
# 4. PLAYER FEATURE VECTOR GENERATION
# ============================================================

def create_player_description(player_data: Dict[str, Any]) -> str:
    """
    Create a rich, discriminative text description of a player for embedding.
    
    Uses position-specific percentile thresholds derived from empirical data
    (fpl_two_seasons.csv) to classify players into tiers:
      - elite (≥p90): Top 10% performers
      - strong (p75-p90): Above average performers  
      - average (p50-p75): Middle tier
      - below_average (p25-p50): Below median
      - low (<p25): Bottom quartile
    """
    name = player_data.get("name", "Unknown")
    position = player_data.get("position", "Unknown")

    # Aggregate stats (handle None values)
    total_points = player_data.get("total_points", 0) or 0
    goals = player_data.get("goals_scored", 0) or 0
    assists = player_data.get("assists", 0) or 0
    clean_sheets = player_data.get("clean_sheets", 0) or 0
    minutes = player_data.get("minutes", 0) or 0
    form = player_data.get("form", 0) or 0
    bonus = player_data.get("bonus", 0) or 0
    saves = player_data.get("saves", 0) or 0

    # --- Derived metrics ---
    games_90 = minutes / 90.0 if minutes > 0 else 0.0
    goals_per90 = (goals / games_90) if games_90 > 0 else 0.0
    assists_per90 = (assists / games_90) if games_90 > 0 else 0.0
    points_per90 = (total_points / games_90) if games_90 > 0 else 0.0

    # Get position-specific thresholds (fallback to default if unknown position)
    thresholds = POSITION_THRESHOLDS.get(position, DEFAULT_THRESHOLDS)

    # --- Performance tier based on total points (position-specific) ---
    points_tier = get_tier_label(total_points, thresholds["total_points"])
    tier_descriptions = {
        "elite": "elite premium top-tier, season-defining asset",
        "strong": "high-performing premium option, very strong pick",
        "average": "reliable starter, solid mid-tier option",
        "below_average": "decent budget or rotation option",
        "low": "low-impact budget option, fringe or bench player"
    }
    tier = tier_descriptions[points_tier]

    # --- Minutes / nailedness description (position-specific) ---
    minutes_tier = get_tier_label(minutes, thresholds["minutes"])
    minutes_descriptions = {
        "elite": "nailed-on regular starter who plays almost every match",
        "strong": "mostly regular starter with good minutes",
        "average": "rotation risk but with a fair amount of minutes",
        "below_average": "rotation or bench player with limited minutes",
        "low": "rarely plays, very low minutes"
    }
    minutes_desc = minutes_descriptions[minutes_tier]

    # --- Scoring ability (position-specific) ---
    goals_tier = get_tier_label(goals_per90, thresholds["goals_per90"])
    scoring_descriptions = {
        "elite": "explosive goal threat with elite goals per 90 minutes for the position",
        "strong": "strong and consistent goal threat above average for the position",
        "average": "moderate goal threat, scores at an average rate for the position",
        "below_average": "light goal threat, scores below average for the position",
        "low": "minimal goal threat, rarely scores"
    }
    scoring = scoring_descriptions[goals_tier]

    # --- Assist / creativity ability (position-specific) ---
    assists_tier = get_tier_label(assists_per90, thresholds["assists_per90"])
    creativity_descriptions = {
        "elite": "elite creative playmaker with top-tier assists per 90 for the position",
        "strong": "very good creative player with above-average assist output",
        "average": "some creative output with average assists for the position",
        "below_average": "limited creativity, below-average assist potential",
        "low": "offers almost no assist potential"
    }
    creativity = creativity_descriptions[assists_tier]

    # --- Form description (position-specific) ---
    form_tier = get_tier_label(form, thresholds["form"])
    form_descriptions = {
        "elite": "currently in outstanding form and on a hot streak",
        "strong": "currently in good and reliable form",
        "average": "currently in average, steady form",
        "below_average": "currently in below-average form",
        "low": "no recent form data or not playing recently"
    }
    form_desc = form_descriptions[form_tier]

    # --- Bonus points tier (position-specific) ---
    bonus_tier = get_tier_label(bonus, thresholds["bonus"])
    bonus_descriptions = {
        "elite": "elite bonus point magnet, consistently earns extra points",
        "strong": "strong bonus point earner",
        "average": "average bonus point accumulation",
        "below_average": "below-average bonus points",
        "low": "rarely earns bonus points"
    }
    bonus_desc = bonus_descriptions[bonus_tier]

    # --- Position-specific description ---
    if position == "GK":
        cs_tier = get_tier_label(clean_sheets, thresholds["clean_sheets"])
        cs_descriptions = {
            "elite": "elite clean sheet potential, top-tier defensive returns",
            "strong": "good clean sheet potential, above average for goalkeepers",
            "average": "average clean sheet potential",
            "below_average": "below-average clean sheet numbers",
            "low": "very low clean sheet potential"
        }
        cs_desc = cs_descriptions[cs_tier]

        saves_tier = get_tier_label(saves, thresholds["saves"])
        saves_descriptions = {
            "elite": "elite shot stopper with very high save volume",
            "strong": "good shot stopper with above-average saves",
            "average": "average save numbers",
            "below_average": "below-average save volume",
            "low": "low save volume, limited shot-stopping opportunities"
        }
        save_desc = saves_descriptions[saves_tier]

        pos_desc = (
            f"goalkeeper combining {cs_desc} and {save_desc}. "
            f"Kept {clean_sheets} clean sheets and made {saves} saves."
        )

    elif position == "DEF":
        cs_tier = get_tier_label(clean_sheets, thresholds["clean_sheets"])
        cs_descriptions = {
            "elite": "elite defensive asset with top-tier clean sheets",
            "strong": "strong defensive asset with good clean sheet numbers",
            "average": "average clean sheet potential for a defender",
            "below_average": "below-average clean sheet numbers",
            "low": "low clean sheet potential"
        }
        cs_desc = cs_descriptions[cs_tier]

        # Attacking contribution based on goals + assists tiers
        attacking_contributions = goals + assists
        if goals_tier in ["elite", "strong"] and assists_tier in ["elite", "strong"]:
            attacking_desc = "offers exceptional attacking threat from defence"
        elif goals_tier in ["elite", "strong"] or assists_tier in ["elite", "strong"]:
            attacking_desc = "offers strong attacking contribution from defence"
        elif goals_tier == "average" or assists_tier == "average":
            attacking_desc = "offers some attacking contribution from defence"
        elif attacking_contributions > 0:
            attacking_desc = "offers limited attacking threat from defence"
        else:
            attacking_desc = "mainly offers defensive returns only"

        pos_desc = (
            f"defender with {cs_desc}, {attacking_desc}. "
            f"Recorded {clean_sheets} clean sheets, {goals} goals and {assists} assists."
        )

    elif position == "MID":
        role = []
        if goals_tier in ["elite", "strong"]:
            role.append("goal-scoring midfielder")
        if assists_tier in ["elite", "strong"]:
            role.append("creative playmaker")
        if not role:
            if goals_tier == "average" or assists_tier == "average":
                role.append("box-to-box midfielder with some attacking output")
            else:
                role.append("defensive or supporting midfielder")

        role_str = " and ".join(role)

        pos_desc = (
            f"{role_str} contributing {goals} goals and {assists} assists. "
            f"Often involved in attacking phases and chance creation."
        )

    elif position == "FWD":
        if goals_tier == "elite":
            fwd_desc = "elite prolific striker, top-tier goal scorer"
        elif goals_tier == "strong":
            fwd_desc = "strong striker with above-average goals"
        elif goals_tier == "average":
            fwd_desc = "decent forward with average goal output"
        elif goals_tier == "below_average":
            fwd_desc = "below-average goal scorer for a forward"
        else:
            fwd_desc = "low-scoring forward, minimal goal threat"

        pos_desc = (
            f"{fwd_desc}, scoring {goals} goals and providing {assists} assists. "
            f"Mainly focused on finishing chances and attacking in the box."
        )

    else:
        pos_desc = (
            f"football player with {goals} goals and {assists} assists, "
            f"role not clearly specified."
        )

    # --- Final description string ---
    description = (
        f"{name} is a {tier} {position} in Fantasy Premier League. "
        f"{minutes_desc}. "
        f"{pos_desc} "
        f"{scoring}. {creativity}. "
        f"Has accumulated {total_points} total FPL points with {bonus} bonus points ({bonus_desc}), "
        f"averaging approximately {points_per90:.2f} points per 90 minutes. "
        f"{form_desc}."
    )

    return description

   


def create_team_description(team_data: Dict[str, Any]) -> str:
    """
    Create a text description of a team for embedding.
    
    Args:
        team_data: Dictionary containing team information
        
    Returns:
        Text description suitable for embedding
    """
    name = team_data.get("name", "Unknown")
    total_goals = team_data.get("total_goals", 0)
    total_points = team_data.get("total_points", 0)
    
    description = (
        f"{name} is a Premier League team. "
        f"Total goals scored by players: {total_goals}. "
        f"Total FPL points from players: {total_points}."
    )
    
    return description


# ============================================================
# 5. NEO4J INTEGRATION - STORE & RETRIEVE EMBEDDINGS
# ============================================================

class EmbeddingStore:
    """
    Manages storage and retrieval of embeddings in Neo4j.
    
    Neo4j 5.x supports native vector indexes for efficient similarity search.
    For older versions, we store embeddings as list properties.
    """
    
    def __init__(self, uri: str, username: str, password: str):
        """Initialize connection to Neo4j"""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        logger.info(f"Connected to Neo4j at {uri}")
    
    def close(self):
        """Close the database connection"""
        self.driver.close()
        logger.info("Neo4j connection closed")
    
    def create_vector_index(self, index_name: str = "player_embedding_minilm"):
        """
        Create a vector index for similarity search (Neo4j 5.11+).
        
        For older Neo4j versions, this will be skipped and we'll use
        brute-force similarity computation.
        """
        query = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (p:Player)
        ON p.embedding_minilm
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: 384,
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        try:
            with self.driver.session() as session:
                session.run(query)
            logger.info(f"✅ Vector index '{index_name}' created or already exists")
        except Exception as e:
            logger.warning(f"Could not create vector index (may need Neo4j 5.11+): {e}")
    
    def store_player_embedding(self, player_name: str, embedding: List[float]):
        """
        Store embedding vector for a player node.
        
        Args:
            player_name: Name of the player
            embedding: Embedding vector (384 dimensions)
        """
        query = """
        MATCH (p:Player)
        WHERE toLower(p.player_name) CONTAINS toLower($name)
        SET p.embedding_minilm = $embedding
        RETURN p.player_name AS name
        """
        with self.driver.session() as session:
            result = session.run(query, name=player_name, embedding=embedding)
            records = list(result)
            if records:
                logger.debug(f"Stored embedding for: {records[0]['name']}")
            else:
                logger.warning(f"Player not found: {player_name}")
    
    def store_player_embeddings_batch(self, embeddings_data: List[Dict[str, Any]]):
        """
        Store embeddings for multiple players efficiently.
        
        Args:
            embeddings_data: List of dicts with 'name' and 'embedding' keys
        """
        query = """
        UNWIND $batch AS item
        MATCH (p:Player {player_name: item.name})
        SET p.embedding_minilm = item.embedding
        """
        with self.driver.session() as session:
            session.run(query, batch=embeddings_data)
        logger.info(f"Stored {len(embeddings_data)} player embeddings")
    
    def get_all_players_for_embedding(self) -> List[Dict[str, Any]]:
        """
        Retrieve all players with their aggregated stats for embedding generation.
        
        Returns:
            List of player data dictionaries
        """
        query = """
        MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
        OPTIONAL MATCH (p)-[r:PLAYED_IN]->(:Fixture)
        WITH p, pos, 
             SUM(r.total_points) AS total_points,
             SUM(r.goals_scored) AS goals_scored,
             SUM(r.assists) AS assists,
             SUM(r.clean_sheets) AS clean_sheets,
             SUM(r.minutes) AS minutes,
             SUM(r.bonus) AS bonus,
             SUM(r.saves) AS saves,
             AVG(r.form) AS avg_form
        RETURN DISTINCT p.player_name AS name,
               pos.name AS position,
               total_points,
               goals_scored,
               assists,
               clean_sheets,
               minutes,
               bonus,
               saves,
               avg_form AS form
        ORDER BY total_points DESC
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]
    
    def vector_similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        position_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find players most similar to the query embedding.
        
        Uses Neo4j vector index if available, otherwise brute-force.
        
        Args:
            query_embedding: Query vector (384 dimensions)
            top_k: Number of results to return
            position_filter: Optional position to filter by (GK/DEF/MID/FWD)
            
        Returns:
            List of similar players with similarity scores
        """
        # Try native vector search first (Neo4j 5.11+)
        try:
            if position_filter:
                query = """
                CALL db.index.vector.queryNodes('player_embedding_minilm', $k, $embedding)
                YIELD node, score
                MATCH (node)-[:PLAYS_AS]->(pos:Position)
                WHERE pos.name = $position
                RETURN node.player_name AS player,
                       pos.name AS position,
                       score
                LIMIT $k
                """
                params = {"k": top_k * 2, "embedding": query_embedding, "position": position_filter}
            else:
                query = """
                CALL db.index.vector.queryNodes('player_embedding_minilm', $k, $embedding)
                YIELD node, score
                MATCH (node)-[:PLAYS_AS]->(pos:Position)
                RETURN node.player_name AS player,
                       pos.name AS position,
                       score
                """
                params = {"k": top_k, "embedding": query_embedding}
            
            with self.driver.session() as session:
                result = session.run(query, **params)
                results = [dict(r) for r in result]
                # Deduplicate by player name (keep first/highest score)
                seen = set()
                unique_results = []
                for r in results:
                    if r['player'] not in seen:
                        seen.add(r['player'])
                        unique_results.append(r)
                return unique_results[:top_k]
                
        except Exception as e:
            logger.warning(f"Vector index not available, using brute-force: {e}")
            return self._brute_force_similarity(query_embedding, top_k, position_filter)
    
    def _brute_force_similarity(
        self,
        query_embedding: List[float],
        top_k: int,
        position_filter: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Brute-force similarity search when vector index is not available.
        """
        if position_filter:
            query = """
            MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
            WHERE p.embedding_minilm IS NOT NULL AND pos.name = $position
            RETURN DISTINCT p.player_name AS player,
                   pos.name AS position,
                   p.embedding_minilm AS embedding
            """
            params = {"position": position_filter}
        else:
            query = """
            MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
            WHERE p.embedding_minilm IS NOT NULL
            RETURN DISTINCT p.player_name AS player,
                   pos.name AS position,
                   p.embedding_minilm AS embedding
            """
            params = {}
        
        with self.driver.session() as session:
            result = session.run(query, **params)
            records = list(result)
        
        # Compute similarities
        query_vec = np.array(query_embedding)
        similarities = []
        
        for record in records:
            player_vec = np.array(record["embedding"])
            similarity = float(np.dot(query_vec, player_vec) / 
                             (np.linalg.norm(query_vec) * np.linalg.norm(player_vec)))
            similarities.append({
                "player": record["player"],
                "position": record["position"],
                "score": similarity
            })
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x["score"], reverse=True)
        return similarities[:top_k]


# ============================================================
# 6. SEMANTIC QUERY SEARCH
# ============================================================

class SemanticSearchMiniLM:
    """
    Semantic search for FPL queries using MiniLM embeddings.
    
    This enables natural language queries like:
      - "fast attacking midfielder with good assists"
      - "reliable goalkeeper with many clean sheets"
      - "high-scoring forward"
    """
    
    def __init__(self, config: Dict[str, str]):
        """Initialize the semantic search system"""
        self.embedder = MiniLMEmbedder()
        self.store = EmbeddingStore(
            uri=config["URI"],
            username=config["USERNAME"],
            password=config["PASSWORD"]
        )
    
    def close(self):
        """Clean up resources"""
        self.store.close()
    
    def generate_and_store_all_embeddings(self):
        """
        Generate embeddings for all players and store in Neo4j.
        
        This should be run once to populate the database with embeddings.
        """
        logger.info("Fetching all players from Neo4j...")
        players = self.store.get_all_players_for_embedding()
        logger.info(f"Found {len(players)} players")
        
        # Generate descriptions
        logger.info("Generating player descriptions...")
        descriptions = []
        player_names = []
        
        for player in players:
            desc = create_player_description(player)
            descriptions.append(desc)
            player_names.append(player["name"])
        
        # Generate embeddings in batch
        logger.info("Generating embeddings (this may take a moment)...")
        embeddings = self.embedder.embed_texts(descriptions)
        
        # Prepare batch data
        batch_data = [
            {"name": name, "embedding": emb}
            for name, emb in zip(player_names, embeddings)
        ]
        
        # Store in Neo4j
        logger.info("Storing embeddings in Neo4j...")
        self.store.store_player_embeddings_batch(batch_data)
        
        # Create vector index
        self.store.create_vector_index()
        
        logger.info("✅ All embeddings generated and stored!")
        return len(players)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        position: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for players matching a natural language query.
        
        Args:
            query: Natural language query (e.g., "high-scoring midfielder")
            top_k: Number of results to return
            position: Optional position filter (GK/DEF/MID/FWD)
            
        Returns:
            List of matching players with similarity scores
        """
        # Embed the query
        query_embedding = self.embedder.embed_text(query)
        
        # Search for similar players
        results = self.store.vector_similarity_search(
            query_embedding,
            top_k=top_k,
            position_filter=position
        )
        
        return results


# ============================================================
# 7. MAIN - TESTING
# ============================================================

if __name__ == "__main__":
    config = load_config()
    
    print("=" * 70)
    print("MiniLM Embedding System for FPL Knowledge Graph")
    print("Model: all-MiniLM-L6-v2 (384 dimensions)")
    print("=" * 70)
    
    # Initialize system
    search_system = SemanticSearchMiniLM(config)
    
    # Check if embeddings need to be generated
    print("\nOptions:")
    print("1. Generate embeddings for all players")
    print("2. Run semantic search demo")
    print("3. Both (generate then run demo)")
    print("4. Run custom semantic search")
    
    choice = input("\nEnter choice (1/2/3/4): ").strip()
    
    if choice in ["1", "3"]:
        print("\n--- Generating Embeddings ---")
        num_players = search_system.generate_and_store_all_embeddings()
        print(f"Generated embeddings for {num_players} players")
    
    if choice in ["2", "3"]:
        print("\n--- Semantic Search Demo ---")
        sample_queries: List[Tuple[str, Optional[str]]] = [
            ("high scoring forward with lots of goals", "FWD"),
            ("reliable defender with clean sheets", "DEF"),
            ("creative midfielder with assists", "MID"),
            ("goalkeeper with many saves", "GK"),
            ("player in excellent form", None)
        ]
        
        for query_text, pos_filter in sample_queries:
            suffix = f" [Position: {pos_filter}]" if pos_filter else ""
            print(f"\n🔍 Query: '{query_text}'{suffix}")
            results = search_system.search(query_text, top_k=5, position=pos_filter)
            
            if not results:
                print("   No results found")
                continue
            
            for idx, result in enumerate(results, 1):
                print(
                    f"   {idx}. {result['player']} ({result['position']}) - "
                    f"Score: {result['score']:.4f}"
                )
    
    if choice == "4":
        print("\n--- Custom Semantic Search ---")
        while True:
            query_text = input("Enter your query (blank to exit): ").strip()
            if not query_text:
                break
            pos_input = input(
                "Optional position filter (GK/DEF/MID/FWD or blank): "
            ).strip().upper()
            position_filter = pos_input if pos_input in {"GK", "DEF", "MID", "FWD"} else None
            results = search_system.search(query_text, top_k=5, position=position_filter)
            if not results:
                print("   No results found")
                continue
            for idx, result in enumerate(results, 1):
                print(
                    f"   {idx}. {result['player']} ({result['position']}) - "
                    f"Score: {result['score']:.4f}"
                )
    
    search_system.close()
    print("\n✅ Done!")
