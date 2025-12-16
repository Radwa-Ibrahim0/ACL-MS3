# Main Module - FPL Graph-RAG System

This folder contains the core implementation of the Fantasy Premier League (FPL) Graph-RAG system. The system allows users to query an FPL Knowledge Graph using natural language and receive AI-generated responses.

## Architecture Overview

```
User Query → preprocessing.py → baseline.py → embedding_*.py → llm_layer.py → Response
                   ↓                ↓               ↓               ↓
              NLP + NER       Cypher Queries   Vector Search    LLM Generation
```

---

## Files

### 1. `app.py` - Streamlit Web Interface
**Purpose:** Main entry point for the user interface.

**Key Features:**
- Football-themed Streamlit UI for querying the FPL Knowledge Graph
- Multiple retrieval modes: Baseline, Embedding, and Hybrid
- Side-by-side LLM comparison (3 models)
- Displays executed Cypher queries and retrieved context
- Configurable settings (embedding model, LLM selection)

**Run Command:**
```bash
streamlit run Main/app.py
```

---

### 2. `preprocessing.py` - NLP Query Preprocessing
**Purpose:** Parses natural language queries to extract intent and entities.

**Key Components:**
- **spaCy NER Pipeline:** Uses EntityRuler with patterns for FPL entities (players, teams, statistics)
- **LLM-based Intent Detection:** Calls OpenRouter API to classify query intent
- **Entity Extraction:** Identifies Players, Teams, Positions, Seasons, Gameweeks, Statistics

**Main Function:**
```python
process_user_query(query: str) -> Dict
# Returns: {
#   "query": str,
#   "intent": str,  # PLAYER-RELATED, FIXTURE-RELATED, COMPARISON, etc.
#   "entities": {"Player": [], "Team": [], "Statistic": [], ...},
#   "ranking": str | None,  # "best" or "worst"
#   "threshold": dict | None  # {"stat": "goals_scored", "operator": ">=", "value": 5}
# }
```

**Supported Intents:**
- `PLAYER-RELATED` - Questions about player stats/performance
- `FIXTURE-RELATED` - Questions about matches/fixtures
- `COMPARISON` - Compare two players or teams
- `TEAM-RELATED` - Questions about team performance

---

### 3. `baseline.py` - Cypher Query Builder
**Purpose:** Executes structured Cypher queries against the Neo4j Knowledge Graph.

**Key Components:**
- **Neo4jConnection:** Handles database connectivity
- **BaselineQueryBuilder:** Contains 11+ query templates for different query types

**Query Types:**
| Query | Description | Trigger Conditions |
|-------|-------------|-------------------|
| Query | Top players by statistic (season-wide) | 1 stat, ranking="best" |
| Query | Worst players by statistic | 1 stat, ranking="worst" |
| Query | Top players by stat + position | 1 stat, 1 position, ranking="best" |
| Query | Worst players by stat + position | 1 stat, 1 position, ranking="worst" |
| Query | Gameweek top performers | 1 gameweek, 1 stat, ranking="best" |
| Query | Dynamic fallback | Any combination with threshold |
| Query | Compare two players | 2 players, intent=COMPARISON |
| Query | Head-to-head fixtures | 2 teams, intent=FIXTURE-RELATED |
| Query | Gameweek fixtures | 1 gameweek only |
| Query | Player performance | 1 player only |
| Query | Team fixtures | 1 team only |

**Main Function:**
```python
execute_baseline_query(preprocessing_output: Dict) -> Dict
# Returns: {
#   "results": List[Dict],
#   "cypher_query": str,  # The actual executed Cypher query
#   "parameters": Dict,
#   "description": str
# }
```

---

### 4. `embedding_bge_m3.py` - BGE-M3 Semantic Search
**Purpose:** Vector similarity search using BGE-M3 embeddings (1024 dimensions).

**Key Features:**
- **Model:** BAAI/bge-m3 (superior multilingual and semantic understanding)
- **Embedding Dimension:** 1024
- Generates text descriptions for KG nodes (players, teams, fixtures)
- Stores embeddings in Neo4j as node properties
- Cosine similarity search for semantic matching

**Main Class:**
```python
class SemanticSearchBGEM3:
    def search(query: str, top_k: int = 10) -> List[Dict]
    def generate_all_embeddings()  # Populates KG with embeddings
```

---

### 5. `embedding_minilm.py` - MiniLM Semantic Search
**Purpose:** Lightweight vector similarity search using MiniLM embeddings (384 dimensions).

**Key Features:**
- **Model:** all-MiniLM-L6-v2 (fast and lightweight)
- **Embedding Dimension:** 384
- Same API as BGE-M3 for easy swapping
- Faster inference, lower memory usage

**Main Class:**
```python
class SemanticSearchMiniLM:
    def search(query: str, top_k: int = 10) -> List[Dict]
    def generate_all_embeddings()
```

---

### 6. `llm_layer.py` - LLM Response Generation
**Purpose:** Orchestrates LLM calls to generate natural language responses from retrieved context.

**Key Components:**
- **RetrievalContext:** Dataclass holding merged baseline + embedding results
- **OpenRouterAdapter:** API adapter for OpenRouter-hosted LLMs
- **Prompt Builder:** Constructs structured prompts with context injection

**Supported LLMs (via OpenRouter):**
- `mistralai/mistral-7b-instruct:free` - Fast, good quality
- `openrouter/gpt-oss-20b:free` - Larger model, better reasoning
- `google/gemma-3-27b-it:free` - Google's instruction-tuned model

**Main Functions:**
```python
build_retrieval_context(query: str, baseline_result: Dict, embedding_results: List) -> RetrievalContext

build_structured_prompt(context: RetrievalContext) -> str

class OpenRouterAdapter:
    def generate(prompt: str) -> ModelAnswer
```

---

## Data Flow

1. **User Input:** Natural language query (e.g., "Top 10 midfielders by goals")

2. **Preprocessing (`preprocessing.py`):**
   - spaCy NER extracts entities: `{"Position": ["MID"], "Statistic": ["goals_scored"]}`
   - LLM classifies intent: `"PLAYER-RELATED"`
   - Determines ranking: `"best"`

3. **Baseline Query (`baseline.py`):**
   - Routes to `query_top_players_by_stat_and_position()`
   - Executes Cypher: `MATCH (p:Player)-[:PLAYED_IN]->... ORDER BY goals_scored DESC LIMIT 10`
   - Returns structured results + the actual Cypher query

4. **Embedding Search (`embedding_*.py`):**
   - Encodes query to vector
   - Finds semantically similar nodes via cosine similarity
   - Returns top-k matches with similarity scores

5. **LLM Generation (`llm_layer.py`):**
   - Merges baseline + embedding results into context
   - Builds prompt with instructions
   - Calls LLM API for natural language response

---



**spaCy Model:**
```bash
python -m spacy download en_core_web_sm
```
