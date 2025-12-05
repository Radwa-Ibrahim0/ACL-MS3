"""
Test script for embedding models
"""
import sys

print("=" * 60)
print("Testing Embedding Models for FPL Knowledge Graph")
print("=" * 60)

# Load config
def load_config():
    config = {}
    with open("config.txt", "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                config[key] = value
    return config

config = load_config()

# Test 1: MiniLM
print("\n[1] Testing MiniLM Embeddings (384 dimensions)...")
try:
    from embedding_minilm import MiniLMEmbedder, SemanticSearchMiniLM
    
    embedder = MiniLMEmbedder()
    test_text = "Mohamed Salah is a high scoring forward with many goals"
    embedding = embedder.embed_text(test_text)
    
    print(f"    ✅ MiniLM loaded successfully!")
    print(f"    - Model: {MiniLMEmbedder.MODEL_NAME}")
    print(f"    - Embedding dimension: {len(embedding)}")
    print(f"    - Sample embedding (first 5 values): {embedding[:5]}")
    
except Exception as e:
    print(f"    ❌ MiniLM test failed: {e}")

# Test 2: MPNet
print("\n[2] Testing MPNet Embeddings (768 dimensions)...")
try:
    from embedding_mpnet import MPNetEmbedder, SemanticSearchMPNet
    
    embedder = MPNetEmbedder()
    test_text = "Mohamed Salah is a high scoring forward with many goals"
    embedding = embedder.embed_text(test_text)
    
    print(f"    ✅ MPNet loaded successfully!")
    print(f"    - Model: {MPNetEmbedder.MODEL_NAME}")
    print(f"    - Embedding dimension: {len(embedding)}")
    print(f"    - Sample embedding (first 5 values): {embedding[:5]}")
    
except Exception as e:
    print(f"    ❌ MPNet test failed: {e}")

# Test 3: Compare embeddings
print("\n[3] Comparing embedding quality...")
try:
    from embedding_minilm import MiniLMEmbedder
    from embedding_mpnet import MPNetEmbedder
    import numpy as np
    
    minilm = MiniLMEmbedder()
    mpnet = MPNetEmbedder()
    
    queries = [
        "high scoring striker",
        "defender with clean sheets",
        "creative midfielder",
    ]
    
    print("    Embedding different queries:")
    for query in queries:
        minilm_emb = minilm.embed_text(query)
        mpnet_emb = mpnet.embed_text(query)
        print(f"    - '{query}'")
        print(f"      MiniLM: {len(minilm_emb)} dims, norm={np.linalg.norm(minilm_emb):.4f}")
        print(f"      MPNet:  {len(mpnet_emb)} dims, norm={np.linalg.norm(mpnet_emb):.4f}")
    
    print("\n    ✅ Both models working correctly!")
    
except Exception as e:
    print(f"    ❌ Comparison failed: {e}")

# Test 4: Neo4j Connection
print("\n[4] Testing Neo4j connection...")
try:
    from neo4j import GraphDatabase
    
    driver = GraphDatabase.driver(
        config["URI"],
        auth=(config["USERNAME"], config["PASSWORD"])
    )
    
    with driver.session() as session:
        result = session.run("MATCH (p:Player) RETURN count(p) AS count")
        count = result.single()["count"]
        print(f"    ✅ Neo4j connected! Found {count} players in database.")
    
    driver.close()
    
except Exception as e:
    print(f"    ❌ Neo4j test failed: {e}")

print("\n" + "=" * 60)
print("Testing Complete!")
print("=" * 60)
