"""
embedding_mpnet.py

Feature Vector Embeddings using Sentence-Transformers all-mpnet-base-v2

This module implements:
  - 1c: Feature Vector Embeddings for FPL Knowledge Graph nodes
  - 2b: Vector Similarity Search for semantic query matching

Model: all-mpnet-base-v2
  - Embedding dimension: 768
  - Higher quality than MiniLM, but slower
  - FREE (runs locally)
  
Comparison with MiniLM:
  - MiniLM (384 dims): Fast, lightweight, good quality
  - MPNet (768 dims): Slower, better semantic understanding, still free
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

# Sentence-Transformers all-mpnet-base-v2
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")


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
# 2. EMBEDDING MODEL CLASS - MPNet
# ============================================================

class MPNetEmbedder:
    """
    Feature Vector Embedding using Sentence-Transformers all-mpnet-base-v2
    
    This is a FREE model that creates 768-dimensional embeddings.
    Higher quality than MiniLM but slower.
    
    Comparison:
      - MiniLM (384 dims): Fast, lightweight, good quality
      - MPNet (768 dims): Slower, better quality, still free
    """
    
    MODEL_NAME = "all-mpnet-base-v2"
    EMBEDDING_DIM = 768
    
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
        """Generate embedding for a single text string."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (batch processing)."""
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings.tolist()


# ============================================================
# 3. UNIFIED EMBEDDER (Uses MPNet)
# ============================================================

class UnifiedEmbedder:
    """
    Unified embedding interface using MPNet.
    
    Uses all-mpnet-base-v2 for high quality free embeddings.
    """
    
    def __init__(self, config: Dict[str, str]):
        """
        Initialize the MPNet embedder.
        
        Args:
            config: Configuration dictionary
        """
        self.embedder = MPNetEmbedder()
        self.model_name = "mpnet"
        self.embedding_dim = MPNetEmbedder.EMBEDDING_DIM
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.embedder.embed_text(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return self.embedder.embed_texts(texts)


# ============================================================
# 5. PLAYER FEATURE VECTOR GENERATION
# ============================================================

def create_player_description(player_data: Dict[str, Any]) -> str:
    """
    Create a rich, discriminative text description of a player for embedding.

    The goal is to:
      - Separate elite, good, average, and weak options
      - Capture role-specific strengths (goals, assists, clean sheets, saves)
      - Encode form, minutes (nailed vs rotation risk), and style hints
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

    # --- Performance tier based on total points ---
    if total_points >= 220:
        tier = "elite premium top tier, season-defining asset"
    elif total_points >= 170:
        tier = "high-performing premium option, very strong pick"
    elif total_points >= 120:
        tier = "reliable starter, strong mid-tier option"
    elif total_points >= 60:
        tier = "decent budget or rotation option"
    else:
        tier = "low-impact budget option, fringe or bench player"

    # --- Minutes / nailedness description ---
    if minutes >= 2800:
        minutes_desc = "nailed-on regular starter who plays almost every match"
    elif minutes >= 2000:
        minutes_desc = "mostly regular starter with good minutes"
    elif minutes >= 1200:
        minutes_desc = "rotation risk but with a fair amount of minutes"
    elif minutes >= 500:
        minutes_desc = "rotation / bench player with limited minutes"
    else:
        minutes_desc = "rarely plays, very low minutes"

    # --- Scoring ability ---
    if goals_per90 >= 0.7:
        scoring = "explosive goal threat with extremely high goals per 90 minutes"
    elif goals_per90 >= 0.4:
        scoring = "strong and consistent goal threat"
    elif goals_per90 >= 0.2:
        scoring = "moderate goal threat, scores occasionally"
    elif goals > 0:
        scoring = "light goal threat, scores rarely"
    else:
        scoring = "almost no goal threat"

    # --- Assist / creativity ability ---
    if assists_per90 >= 0.4:
        creativity = "elite creative playmaker with many assists per 90 minutes"
    elif assists_per90 >= 0.25:
        creativity = "very good creative player with regular assists"
    elif assists_per90 >= 0.1:
        creativity = "some creative output with occasional assists"
    elif assists > 0:
        creativity = "limited creativity, few assists"
    else:
        creativity = "offers almost no assist potential"

    # --- Form description (you can adjust thresholds to your data) ---
    if form >= 7.0:
        form_desc = "currently in outstanding form and on a hot streak"
    elif form >= 5.0:
        form_desc = "currently in good and reliable form"
    elif form >= 3.0:
        form_desc = "currently in average, mixed form"
    elif form > 0:
        form_desc = "currently in poor form and underperforming"
    else:
        form_desc = "no recent form data or not playing recently"

    # --- Position-specific description ---
    if position == "GK":
        if clean_sheets >= 14:
            cs_desc = "elite clean sheet potential"
        elif clean_sheets >= 9:
            cs_desc = "good clean sheet potential"
        elif clean_sheets >= 4:
            cs_desc = "some clean sheet potential"
        else:
            cs_desc = "very low clean sheet potential"

        if saves >= 120:
            save_desc = "high-volume shot stopper with many saves"
        elif saves >= 80:
            save_desc = "good shot stopper with a solid number of saves"
        elif saves >= 40:
            save_desc = "average number of saves"
        else:
            save_desc = "low save volume"

        pos_desc = (
            f"goalkeeper combining {cs_desc} and {save_desc}. "
            f"Kept {clean_sheets} clean sheets and made {saves} saves."
        )

    elif position == "DEF":
        if clean_sheets >= 16:
            cs_desc = "elite defensive asset with many clean sheets"
        elif clean_sheets >= 10:
            cs_desc = "strong defensive asset with good clean sheet numbers"
        elif clean_sheets >= 5:
            cs_desc = "decent clean sheet potential"
        else:
            cs_desc = "low clean sheet potential"

        attacking_desc = ""
        if goals >= 5 and assists >= 5:
            attacking_desc = "offers very strong attacking threat from defence"
        elif goals + assists >= 5:
            attacking_desc = "offers useful attacking contribution from defence"
        elif goals + assists > 0:
            attacking_desc = "offers a small bit of attacking threat"
        else:
            attacking_desc = "mainly offers defensive returns only"

        pos_desc = (
            f"defender with {cs_desc}, {attacking_desc}. "
            f"Recorded {clean_sheets} clean sheets, {goals} goals and {assists} assists."
        )

    elif position == "MID":
        role = []
        if goals >= 10:
            role.append("goal-scoring midfielder")
        if assists >= 10:
            role.append("elite creative playmaker")
        if not role:
            role.append("box-to-box or supporting midfielder")

        role_str = " and ".join(role)

        pos_desc = (
            f"{role_str} contributing {goals} goals and {assists} assists. "
            f"Often involved in attacking phases and chance creation."
        )

    elif position == "FWD":
        if goals >= 20:
            fwd_desc = "elite prolific striker and primary goal scorer"
        elif goals >= 12:
            fwd_desc = "very strong striker with regular goals"
        elif goals >= 6:
            fwd_desc = "decent forward with some goals"
        elif goals > 0:
            fwd_desc = "low-scoring forward"
        else:
            fwd_desc = "forward with almost no goals"

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
        f"Has accumulated {total_points} total FPL points with {bonus} bonus points, "
        f"averaging approximately {points_per90:.2f} points per 90 minutes. "
        f"{form_desc}."
    )

    return description


# ============================================================
# 6. NEO4J INTEGRATION - STORE & RETRIEVE EMBEDDINGS
# ============================================================

class EmbeddingStoreMPNet:
    """
    Manages storage and retrieval of MPNet embeddings in Neo4j.
    
    Uses a different property name (embedding_mpnet) than MiniLM to allow
    both embedding types to coexist for comparison.
    """
    
    def __init__(self, uri: str, username: str, password: str, embedding_dim: int = 768):
        """Initialize connection to Neo4j"""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.embedding_dim = embedding_dim
        logger.info(f"Connected to Neo4j at {uri}")
    
    def close(self):
        """Close the database connection"""
        self.driver.close()
        logger.info("Neo4j connection closed")
    
    def create_vector_index(self, index_name: str = "player_embedding_mpnet"):
        """Create a vector index for similarity search (Neo4j 5.11+)."""
        query = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (p:Player)
        ON p.embedding_mpnet
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {self.embedding_dim},
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
    
    def store_player_embeddings_batch(self, embeddings_data: List[Dict[str, Any]]):
        """Store embeddings for multiple players efficiently."""
        query = """
        UNWIND $batch AS item
        MATCH (p:Player {player_name: item.name})
        SET p.embedding_mpnet = item.embedding
        """
        with self.driver.session() as session:
            session.run(query, batch=embeddings_data)
        logger.info(f"Stored {len(embeddings_data)} player embeddings")
    
    def get_all_players_for_embedding(self) -> List[Dict[str, Any]]:
        """Retrieve all players with their aggregated stats for embedding generation."""
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
        """Find players most similar to the query embedding."""
        try:
            if position_filter:
                query = """
                CALL db.index.vector.queryNodes('player_embedding_mpnet', $k, $embedding)
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
                CALL db.index.vector.queryNodes('player_embedding_mpnet', $k, $embedding)
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
        """Brute-force similarity search when vector index is not available."""
        if position_filter:
            query = """
            MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
            WHERE p.embedding_mpnet IS NOT NULL AND pos.name = $position
            RETURN DISTINCT p.player_name AS player,
                   pos.name AS position,
                   p.embedding_mpnet AS embedding
            """
            params = {"position": position_filter}
        else:
            query = """
            MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
            WHERE p.embedding_mpnet IS NOT NULL
            RETURN DISTINCT p.player_name AS player,
                   pos.name AS position,
                   p.embedding_mpnet AS embedding
            """
            params = {}
        
        with self.driver.session() as session:
            result = session.run(query, **params)
            records = list(result)
        
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
        
        similarities.sort(key=lambda x: x["score"], reverse=True)
        return similarities[:top_k]


# ============================================================
# 7. SEMANTIC QUERY SEARCH
# ============================================================

class SemanticSearchMPNet:
    """
    Semantic search for FPL queries using MPNet embeddings.
    
    This is the second embedding model for comparison with MiniLM.
    """
    
    def __init__(self, config: Dict[str, str]):
        """Initialize the semantic search system"""
        self.embedder = UnifiedEmbedder(config)
        self.store = EmbeddingStoreMPNet(
            uri=config["URI"],
            username=config["USERNAME"],
            password=config["PASSWORD"],
            embedding_dim=self.embedder.embedding_dim
        )
        logger.info(f"Using embedding model: {self.embedder.model_name}")
    
    def close(self):
        """Clean up resources"""
        self.store.close()
    
    def generate_and_store_all_embeddings(self):
        """Generate embeddings for all players and store in Neo4j."""
        logger.info("Fetching all players from Neo4j...")
        players = self.store.get_all_players_for_embedding()
        logger.info(f"Found {len(players)} players")
        
        logger.info("Generating player descriptions...")
        descriptions = []
        player_names = []
        
        for player in players:
            desc = create_player_description(player)
            descriptions.append(desc)
            player_names.append(player["name"])
        
        logger.info(f"Generating embeddings using {self.embedder.model_name}...")
        embeddings = self.embedder.embed_texts(descriptions)
        
        batch_data = [
            {"name": name, "embedding": emb}
            for name, emb in zip(player_names, embeddings)
        ]
        
        logger.info("Storing embeddings in Neo4j...")
        self.store.store_player_embeddings_batch(batch_data)
        self.store.create_vector_index()
        
        logger.info("✅ All embeddings generated and stored!")
        return len(players)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        position: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for players matching a natural language query."""
        query_embedding = self.embedder.embed_text(query)
        results = self.store.vector_similarity_search(
            query_embedding,
            top_k=top_k,
            position_filter=position
        )
        return results


# ============================================================
# 8. MODEL COMPARISON UTILITIES
# ============================================================

def compare_models(
    config: Dict[str, str],
    query: str,
    position: Optional[str] = None,
    top_k: int = 5
):
    """
    Compare search results between MiniLM and MPNet models.
    
    Args:
        config: Configuration dictionary
        query: Natural language query to evaluate
        position: Optional position filter (GK/DEF/MID/FWD)
        top_k: Number of results to display per model
    """
    print(f"\n{'='*70}")
    suffix = f" (Position: {position})" if position else ""
    print(f"Model Comparison for query: '{query}'{suffix}")
    print("="*70)
    
    # Try to import MiniLM search
    try:
        from embedding_minilm import SemanticSearchMiniLM
        
        print("\n--- MiniLM (384 dimensions) ---")
        minilm_search = SemanticSearchMiniLM(config)
        minilm_results = minilm_search.search(query, top_k=top_k, position=position)
        
        for i, r in enumerate(minilm_results, 1):
            print(f"  {i}. {r['player']} ({r['position']}) - Score: {r['score']:.4f}")
        
        minilm_search.close()
    except Exception as e:
        print(f"MiniLM search not available: {e}")
    
    # MPNet/OpenAI search
    print(f"\n--- MPNet (768 dimensions) ---")
    mpnet_search = SemanticSearchMPNet(config)
    mpnet_results = mpnet_search.search(query, top_k=top_k, position=position)
    
    for i, r in enumerate(mpnet_results, 1):
        print(f"  {i}. {r['player']} ({r['position']}) - Score: {r['score']:.4f}")
    
    mpnet_search.close()
    
    print("\n" + "="*70)


# ============================================================
# 9. MAIN - TESTING
# ============================================================

if __name__ == "__main__":
    config = load_config()
    
    print("=" * 70)
    print("MPNet Embedding System for FPL Knowledge Graph")
    print("Model: all-mpnet-base-v2 (768 dimensions)")
    print("=" * 70)
    
    # Initialize system
    search_system = SemanticSearchMPNet(config)
    
    print(f"\nActive model: {search_system.embedder.model_name}")
    print(f"Embedding dimensions: {search_system.embedder.embedding_dim}")
    
    print("\nOptions:")
    print("1. Generate embeddings for all players")
    print("2. Run semantic search demo")
    print("3. Both (generate then run demo)")
    print("4. Run custom semantic search")
    print("5. Compare with MiniLM model")
    
    choice = input("\nEnter choice (1/2/3/4/5): ").strip()
    
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
    
    if choice == "5":
        print("\n--- Model Comparison ---")
        test_query = input("Enter a search query: ").strip()
        if test_query:
            pos_input = input(
                "Optional position filter for comparison (GK/DEF/MID/FWD or blank): "
            ).strip().upper()
            position_filter = pos_input if pos_input in {"GK", "DEF", "MID", "FWD"} else None
            compare_models(config, test_query, position_filter)
    
    search_system.close()
    print("\n✅ Done!")
