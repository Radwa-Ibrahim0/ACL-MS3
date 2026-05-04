# ACL-MS3: FPL Graph-RAG Assistant

ACL-MS3 is an AI-powered Fantasy Premier League (FPL) assistant that answers natural-language questions over an FPL knowledge graph. It combines structured Neo4j/Cypher retrieval, semantic vector search, and LLM-based response generation through a Streamlit web interface.

## Features

- Natural-language FPL questions, such as “Top 10 midfielders by goals” or player/team comparisons.
- Neo4j knowledge graph for seasons, gameweeks, fixtures, teams, players, positions, and player match statistics.
- Baseline graph retrieval using Cypher query templates.
- Semantic search using BGE-M3 or MiniLM embeddings.
- Hybrid retrieval combining structured graph results with embedding results.
- LLM response generation through OpenRouter.
- Streamlit UI with model comparison, retrieval settings, Cypher visibility, and raw context inspection.

## Project Structure

```text
ACL-MS3/
├── Main/
│   ├── app.py                 # Streamlit web application
│   ├── baseline.py            # Neo4j connection and Cypher query builder
│   ├── preprocessing.py       # Intent detection and entity extraction
│   ├── embedding_bge_m3.py    # BGE-M3 semantic search implementation
│   ├── embedding_minilm.py    # MiniLM semantic search implementation
│   ├── llm_layer.py           # Retrieval-context construction and LLM adapters
│   └── README.md              # Detailed Main module documentation
├── milestone2/
│   ├── Create_kg.py           # Loads FPL CSV data into Neo4j
│   └── fpl_two_seasons.csv    # FPL dataset
├── tests/
├── image/
├── config.txt                 # Local configuration; do not commit real secrets
└── README.md
```

## System Architecture

```text
User Question
     ↓
preprocessing.py
     ↓
Intent + Entity Extraction
     ↓
baseline.py ────────────────┐
     ↓                       │
Cypher / Neo4j Results       │
                             ├── llm_layer.py → Grounded FPL Answer
embedding_bge_m3.py or       │
embedding_minilm.py ─────────┘
     ↓
Semantic Search Results
```

## Prerequisites

- Python 3.10+
- Neo4j running locally or remotely
- OpenRouter API key
- spaCy English model: `en_core_web_sm`
- FPL CSV data loaded into Neo4j

## Installation

Clone the repository:

```bash
git clone https://github.com/Radwa-Ibrahim0/ACL-MS3.git
cd ACL-MS3
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install streamlit neo4j spacy requests sentence-transformers numpy pandas google-generativeai
python -m spacy download en_core_web_sm
```

> ```bash
> pip install -r requirements.txt
> ```

## Configuration

Create a `config.txt` file in the project root:

```ini
URI=neo4j://127.0.0.1:7687
USERNAME=neo4j
PASSWORD=your_neo4j_password
CURRENT_SEASON=2022-23
CURRENT_GW=10
OPENROUTER_API_KEY=your_openrouter_api_key
```

Do not commit real API keys or passwords. Add `config.txt` to `.gitignore` and keep a separate `config.example.txt` file with placeholder values.

## Build the Knowledge Graph

Make sure Neo4j is running, then load the FPL CSV data:

```bash
python milestone2/Create_kg.py
```

This script creates constraints and loads seasons, gameweeks, fixtures, teams, players, positions, and player performance statistics into Neo4j.

## Generate Embeddings

The app supports BGE-M3 and MiniLM embeddings. If embeddings are not already stored in Neo4j, generate them before using embedding or hybrid retrieval.

Example for BGE-M3:

```python
from Main.embedding_bge_m3 import SemanticSearchBGEM3, load_config

config = load_config()
search = SemanticSearchBGEM3(config)
search.generate_all_embeddings()
search.close()
```

For the lighter option, use `SemanticSearchMiniLM` from `Main.embedding_minilm`.

## Run the App

```bash
streamlit run Main/app.py
```

Then open the local Streamlit URL shown in the terminal.

## Example Questions

- `Top 10 players by total points`
- `Best midfielders by goals scored`
- `Compare Mohamed Salah and Bukayo Saka`
- `Show gameweek 10 fixtures`
- `Which forwards have the best form?`

## Retrieval Modes

The Streamlit sidebar supports:

- **Baseline Only**: Structured Cypher queries over Neo4j.
- **Embeddings Only**: Vector similarity search.
- **Hybrid (Both)**: Combines graph retrieval and semantic search.

## Models

The LLM layer calls OpenRouter-hosted models for answer generation. The app can run one selected model or compare multiple models side by side.

Embedding options:

- **BGE-M3**: Higher-quality semantic search with 1024-dimensional embeddings.
- **MiniLM**: Faster, lighter semantic search with 384-dimensional embeddings.

## Troubleshooting

### `config.txt not found`

Create `config.txt` in the project root using the template above.

### Neo4j connection errors

Check that:

- Neo4j is running.
- `URI`, `USERNAME`, and `PASSWORD` are correct.
- The database contains the expected FPL graph data.

### spaCy model error

Install the required model:

```bash
python -m spacy download en_core_web_sm
```

### Embedding model download is slow

The first run may download Sentence Transformers models from Hugging Face. Later runs should be faster because models are cached locally.

## Security Notes

This project uses external API keys and database credentials. Keep them out of version control.

Recommended actions:

- Rotate any API keys that were previously committed.
- Add `config.txt` to `.gitignore`.
- Create `config.example.txt` with placeholder values.
- Prefer environment variables for production deployments.

## Contributors

<a href="https://github.com/HazemMansour1">
  <img src="https://github.com/HazemMansour1.png" width="60" height="60" style="border-radius: 50%;" alt="HazemMansour1"/>
</a>
<a href="https://github.com/Radwa-Ibrahim0">
  <img src="https://github.com/Radwa-Ibrahim0.png" width="60" height="60" style="border-radius: 50%;" alt="Radwa-Ibrahim0"/>
</a>
<a href="https://github.com/amy847">
  <img src="https://github.com/amy847.png" width="60" height="60" style="border-radius: 50%;" alt="amy847"/>
</a>

- [HazemMansour1](https://github.com/HazemMansour1) — Hazem Mansour
- [Radwa-Ibrahim0](https://github.com/Radwa-Ibrahim0) — Radwa Ibrahim
- [amy847](https://github.com/amy847) — manuella ehab
