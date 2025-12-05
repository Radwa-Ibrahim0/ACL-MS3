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
    Create a rich text description of a player for embedding.
    
    Uses more detailed descriptions than MiniLM version for higher
    quality semantic matching.
    """
    name = player_data.get("name", "Unknown")
    position = player_data.get("position", "Unknown")
    
    # Aggregate stats
    total_points = player_data.get("total_points", 0) or 0
    goals = player_data.get("goals_scored", 0) or 0
    assists = player_data.get("assists", 0) or 0
    clean_sheets = player_data.get("clean_sheets", 0) or 0
    minutes = player_data.get("minutes", 0) or 0
    form = player_data.get("form", 0) or 0
    bonus = player_data.get("bonus", 0) or 0
    
    # Detailed position descriptions
    position_descriptions = {
        "GK": "goalkeeper who plays in goal, responsible for saves and clean sheets",
        "DEF": "defender who plays in defense, responsible for clean sheets and stopping attacks",
        "MID": "midfielder who plays in midfield, responsible for creativity, assists and goals",
        "FWD": "forward striker who plays in attack, responsible for scoring goals"
    }
    pos_desc = position_descriptions.get(position, "football player")
    
    # Performance tier based on points
    if total_points > 150:
        tier = "elite premium"
    elif total_points > 100:
        tier = "high-performing"
    elif total_points > 50:
        tier = "solid mid-tier"
    else:
        tier = "budget"
    
    # Form description
    if form > 6:
        form_desc = "in excellent form"
    elif form > 4:
        form_desc = "in good form"
    elif form > 2:
        form_desc = "in average form"
    else:
        form_desc = "struggling for form"
    
    # Build comprehensive description
    description = (
        f"{name} is a {tier} {pos_desc}. "
        f"Has scored {goals} goals and provided {assists} assists. "
        f"Accumulated {total_points} total FPL points with {bonus} bonus points. "
        f"Has {clean_sheets} clean sheets and played {minutes} minutes. "
        f"Currently {form_desc} with form rating of {form}."
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
             AVG(r.form) AS avg_form
        RETURN p.player_name AS name,
               pos.name AS position,
               total_points,
               goals_scored,
               assists,
               clean_sheets,
               minutes,
               bonus,
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
                return results[:top_k]
                
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
            RETURN p.player_name AS player,
                   pos.name AS position,
                   p.embedding_mpnet AS embedding
            """
            params = {"position": position_filter}
        else:
            query = """
            MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
            WHERE p.embedding_mpnet IS NOT NULL
            RETURN p.player_name AS player,
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

def compare_models(config: Dict[str, str], query: str, top_k: int = 5):
    """
    Compare search results between MiniLM and MPNet/OpenAI models.
    
    This helps evaluate which model produces better results for your use case.
    """
    print(f"\n{'='*70}")
    print(f"Model Comparison for query: '{query}'")
    print("="*70)
    
    # Try to import MiniLM search
    try:
        from embedding_minilm import SemanticSearchMiniLM
        
        print("\n--- MiniLM (384 dimensions) ---")
        minilm_search = SemanticSearchMiniLM(config)
        minilm_results = minilm_search.search(query, top_k=top_k)
        
        for i, r in enumerate(minilm_results, 1):
            print(f"  {i}. {r['player']} ({r['position']}) - Score: {r['score']:.4f}")
        
        minilm_search.close()
    except Exception as e:
        print(f"MiniLM search not available: {e}")
    
    # MPNet/OpenAI search
    print(f"\n--- MPNet (768 dimensions) ---")
    mpnet_search = SemanticSearchMPNet(config)
    mpnet_results = mpnet_search.search(query, top_k=top_k)
    
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
    print("2. Run semantic search queries")
    print("3. Both (generate then search)")
    print("4. Compare with MiniLM model")
    
    choice = input("\nEnter choice (1/2/3/4): ").strip()
    
    if choice in ["1", "3"]:
        print("\n--- Generating Embeddings ---")
        num_players = search_system.generate_and_store_all_embeddings()
        print(f"Generated embeddings for {num_players} players")
    
    if choice in ["2", "3"]:
        print("\n--- Semantic Search Demo ---")
        
        test_queries = [
            "high scoring forward with lots of goals",
            "reliable defender with clean sheets",
            "creative midfielder with assists",
            "goalkeeper with many saves",
            "player in excellent form"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            results = search_system.search(query, top_k=5)
            
            for i, r in enumerate(results, 1):
                print(f"   {i}. {r['player']} ({r['position']}) - Score: {r['score']:.4f}")
    
    if choice == "4":
        print("\n--- Model Comparison ---")
        test_query = input("Enter a search query: ").strip()
        if test_query:
            compare_models(config, test_query)
    
    search_system.close()
    print("\n✅ Done!")
