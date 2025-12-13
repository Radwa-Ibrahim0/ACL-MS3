# FPL Graph-RAG Assistant - Streamlit UI

A beautiful, football fantasy themed user interface for the FPL Graph-RAG system.

## 🚀 Quick Start

```bash
# Make sure you're in the project directory
cd ACL-MS3

# Run the Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## ✨ Features

### 1. 💬 Query Interface
- Enter natural language questions about Fantasy Premier League
- Get AI-powered answers using Knowledge Graph context
- Example queries provided for quick testing

### 2. 📊 Retrieved Context Display
- **Baseline Results**: See data retrieved from Neo4j using Cypher queries
- **Embedding Results**: View semantic search results using BGE-M3 embeddings
- **Player Recommendations**: Get ranked player suggestions with explanations

### 3. 🔐 Cypher Query Transparency
- View the actual Cypher queries executed against the Knowledge Graph
- Understand how the system retrieves information
- Great for debugging and learning

### 4. 🤖 Model Selection
- **Single Model Mode**: Use one LLM at a time
  - GPT-OSS-20B (Free)
  - Mistral-7B-Instruct (Free)
  - Gemma-3-27B (Free)
- **Compare Models Mode**: Run the same query on multiple models and compare responses side-by-side

### 5. 🔍 Retrieval Method Selection
- **Hybrid (Both)**: Uses both baseline Cypher queries and embedding-based semantic search
- **Baseline Only**: Traditional structured queries using Neo4j
- **Embeddings Only**: Pure semantic similarity search using BGE-M3

### 6. ⚙️ Advanced Settings
- Adjust number of results retrieved
- Toggle Cypher query display
- Toggle raw context display

## 🎨 Theme

The UI features an official Fantasy Premier League inspired design:
- Purple (#37003c) - FPL primary color
- Green (#00ff87) - FPL accent color
- Gradient backgrounds and card styling
- Position-specific badges for players

## 📋 Example Queries

Try these queries to test the system:

1. **Top Scorers**: "Who are the top goal scorers?"
2. **Position Specific**: "Best midfielders for assists"
3. **Player Info**: "Tell me about Mohamed Salah"
4. **Defensive**: "Top defenders with clean sheets"
5. **Comparison**: "Compare Kane and Haaland"
6. **Budget Options**: "Best budget forwards"
7. **Captain Pick**: "Who should I captain this week?"

## 🛠️ Requirements

- Python 3.8+
- Streamlit (`pip install streamlit`)
- Neo4j database running with FPL data
- Valid API keys in `config.txt`

## 📁 File Structure

```
ACL-MS3/
├── app.py                 # Main Streamlit application
├── config.txt             # Configuration (Neo4j credentials, API keys)
├── preprocessing.py       # Query preprocessing and NER
├── baseline.py            # Cypher query generation and execution
├── embedding_bge_m3.py    # BGE-M3 embedding search
├── llm_layer.py           # LLM integration layer
└── Readmes/
    └── UI_README.md       # This file
```

## 🔧 Configuration

Ensure your `config.txt` has the following:

```
URI=neo4j://127.0.0.1:7687
USERNAME=neo4j
PASSWORD=your_password
OPENROUTER_API_KEY=your_api_key
CURRENT_SEASON=2022-23
```

## 🎯 Usage Tips

1. **Start simple**: Try basic queries first like "top goal scorers"
2. **Use examples**: Click on example queries from the sidebar
3. **Check context**: Enable "Show Raw Context" to see what data the LLM receives
4. **Compare models**: Use comparison mode to see how different LLMs interpret the same data
5. **Experiment with retrieval**: Try different retrieval methods to see which works best for your query

## 🐛 Troubleshooting

- **"Error generating response"**: Check your API keys in config.txt
- **Empty results**: Ensure Neo4j is running with FPL data loaded
- **Slow responses**: Embedding search loads models on first use - subsequent queries are faster

## 📄 License

Part of the ACL-MS3 Project
