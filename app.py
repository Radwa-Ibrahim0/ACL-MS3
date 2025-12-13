"""
Fantasy Premier League Graph-RAG Assistant - Streamlit UI

A football fantasy themed interface for the Graph-RAG system that allows users to:
- Query the FPL Knowledge Graph
- View retrieved context from the KG
- See Cypher queries executed
- Compare different LLM models
- Switch between retrieval methods (baseline, embeddings, hybrid)
- Get player recommendations
"""

import streamlit as st
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple

# Import the Graph-RAG system components
from preprocessing import process_user_query
from baseline import execute_baseline_query, load_config, Neo4jConnection, BaselineQueryBuilder
from embedding_bge_m3 import SemanticSearchBGEM3, load_config as load_config_embed
from llm_layer import (
    build_retrieval_context,
    build_structured_prompt,
    OpenRouterAdapter,
    RetrievalContext,
    ModelAnswer,
    ModelMetrics,
)

# Configure logging
logging.basicConfig(level=logging.WARNING)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FPL Graph-RAG Assistant",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS - MINIMALISTIC FOOTBALL THEME
# ============================================================

st.markdown("""
<style>
    /* Main theme colors - Premier League inspired */
    :root {
        --primary-purple: #37003c;
        --accent-green: #00ff87;
        --dark-green: #02894e;
        --text-light: #ffffff;
        --text-muted: #8a8a8a;
        --bg-card: #f7f7f7;
        --border-light: #e5e5e5;
    }
    
    /* Clean header */
    .main-header {
        background: #37003c;
        padding: 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        color: #00ff87 !important;
        font-size: 2rem;
        margin: 0;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.95rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Minimalistic card styling */
    .info-card {
        background: #ffffff;
        border-left: 3px solid #00ff87;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    .context-card {
        background: #ffffff;
        border: 1px solid #e5e5e5;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    
    .result-card {
        background: #ffffff;
        border: 2px solid #37003c;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Clean badge design */
    .player-badge {
        display: inline-block;
        background: #37003c;
        color: #00ff87;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-weight: 500;
        font-size: 0.85rem;
        margin: 0.15rem;
    }
    
    .stat-badge {
        display: inline-block;
        background: #f0f0f0;
        color: #37003c;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-weight: 500;
        font-size: 0.85rem;
        margin: 0.15rem;
        border: 1px solid #e5e5e5;
    }
    
    /* Clean sidebar */
    section[data-testid="stSidebar"] {
        background: #fafafa;
        border-right: 1px solid #e5e5e5;
    }
    
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stCheckbox label {
        color: #37003c !important;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    /* Clean button styling */
    .stButton > button {
        background: #37003c;
        color: #00ff87;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 4px;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: #4a0050;
        box-shadow: 0 2px 8px rgba(55, 0, 60, 0.2);
    }
    
    /* Info boxes */
    .stSuccess {
        background-color: rgba(0, 255, 135, 0.08) !important;
        border-color: #00ff87 !important;
    }
    
    .stInfo {
        background-color: rgba(55, 0, 60, 0.05) !important;
        border-color: #37003c !important;
    }
    
    /* Table styling */
    .dataframe {
        border: 1px solid #e5e5e5 !important;
    }
    
    .dataframe th {
        background: #37003c !important;
        color: #00ff87 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #8a8a8a;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid #e5e5e5;
        font-size: 0.85rem;
    }
    
    /* Section headers */
    h3 {
        color: #37003c !important;
        font-weight: 600 !important;
        font-size: 1.3rem !important;
    }
    
    h4 {
        color: #37003c !important;
        font-weight: 600 !important;
    }
    
    h5 {
        color: #37003c !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# INITIALIZATION & CACHING
# ============================================================

@st.cache_resource
def load_configuration():
    """Load configuration from config.txt"""
    return load_config()


@st.cache_resource
def get_available_models(_config: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """Get available LLM models with their configurations"""
    api_key = _config.get("OPENROUTER_API_KEY", "")
    
    models = {
        "GPT-OSS-20B (Free)": {
            "name": "GPT-OSS-20B",
            "model": "openai/gpt-oss-20b:free",
            "api_key": api_key,
            "description": "Fast and efficient for general queries"
        },
        "Mistral-7B-Instruct (Free)": {
            "name": "Mistral-7B",
            "model": "mistralai/mistral-7b-instruct:free",
            "api_key": api_key,
            "description": "Good balance of speed and quality"
        },
        "Gemma-3-27B (Free)": {
            "name": "Gemma-3-27B",
            "model": "google/gemma-3-27b-it:free",
            "api_key": api_key,
            "description": "High quality responses, more detailed"
        },
    }
    
    return models


# ============================================================
# RETRIEVAL FUNCTIONS
# ============================================================

def run_preprocessing(query: str) -> Dict[str, Any]:
    """Run preprocessing on user query"""
    return process_user_query(query)


def run_baseline_retrieval(preprocessing_output: Dict[str, Any]) -> Tuple[List[Dict], str]:
    """Run baseline Cypher query retrieval"""
    results = execute_baseline_query(preprocessing_output)
    
    # Build a description of what was queried
    entities = preprocessing_output.get("entities", {})
    intent = preprocessing_output.get("intent", "")
    ranking = preprocessing_output.get("ranking", "")
    
    desc = f"Intent: {intent}"
    if ranking:
        desc += f", Ranking: {ranking}"
    
    return results, desc


def run_embedding_retrieval(query: str, config: Dict[str, str], top_k: int = 10, position: Optional[str] = None) -> List[Dict]:
    """Run embedding-based semantic search"""
    try:
        semantic = SemanticSearchBGEM3(config)
        results = semantic.search(query, top_k=top_k, position=position)
        semantic.close()
        return results
    except Exception as e:
        st.error(f"Embedding search error: {e}")
        return []


def generate_cypher_preview(preprocessing_output: Dict[str, Any]) -> str:
    """Generate a preview of the Cypher query that would be executed"""
    entities = preprocessing_output.get("entities", {})
    intent = preprocessing_output.get("intent", "")
    ranking = preprocessing_output.get("ranking", "")
    
    # Build example Cypher based on entities
    stats = entities.get("Statistic", [])
    players = entities.get("Player", [])
    teams = entities.get("Team", [])
    positions = entities.get("Position", [])
    
    if players and len(players) == 1:
        cypher = f"""
MATCH (p:Player)
WHERE toLower(p.player_name) CONTAINS toLower('{players[0]}')
MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
RETURN p.player_name AS player,
       SUM(r.total_points) AS total_points,
       SUM(r.goals_scored) AS goals,
       SUM(r.assists) AS assists
"""
    elif stats and ranking == "best":
        stat = stats[0] if stats else "total_points"
        pos_filter = f"MATCH (p)-[:PLAYS_AS]->(pos:Position {{name: '{positions[0]}'}})" if positions else ""
        cypher = f"""
MATCH (s:Season {{season_name: '2022-23'}})-[:HAS_GW]->(gw:Gameweek)
      -[:HAS_FIXTURE]->(f:Fixture)
MATCH (p:Player)-[r:PLAYED_IN]->(f)
{pos_filter}
WITH p, SUM(r.{stat}) AS total_stat
WHERE total_stat > 0
RETURN p.player_name AS player, total_stat
ORDER BY total_stat DESC
LIMIT 10
"""
    elif teams:
        team = teams[0]
        cypher = f"""
MATCH (t:Team {{name: '{team}'}})<-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]-(f:Fixture)
MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)
RETURN gw.GW_number AS gameweek,
       f.fixture_number AS fixture
ORDER BY gw.GW_number
LIMIT 10
"""
    else:
        cypher = """
// General fallback query
MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
RETURN p.player_name AS player,
       SUM(r.total_points) AS total_points
ORDER BY total_points DESC
LIMIT 10
"""
    
    return cypher.strip()


def call_llm(prompt: str, model_config: Dict[str, str]) -> Tuple[str, ModelMetrics]:
    """Call the selected LLM model"""
    adapter = OpenRouterAdapter(
        name=model_config["name"],
        model=model_config["model"],
        api_key=model_config["api_key"],
    )
    return adapter.generate(prompt)


# ============================================================
# UI COMPONENTS
# ============================================================

def render_header():
    """Render the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>FPL Graph-RAG Assistant</h1>
        <p>AI-powered Fantasy Premier League advisor using Knowledge Graph technology</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with settings"""
    config = load_configuration()
    models = get_available_models(config)
    
    with st.sidebar:
        st.markdown("### Settings")
        st.markdown("---")
        
        # Model selection
        st.markdown("#### LLM Model")
        
        # Mode: Single or Compare
        model_mode = st.radio(
            "Mode",
            options=["Single Model", "Compare Models"],
            help="Choose single model or compare multiple models"
        )
        
        if model_mode == "Single Model":
            selected_model = st.selectbox(
                "Select Model",
                options=list(models.keys()),
                help="Choose which AI model to use for generating answers"
            )
            st.markdown(f"*{models[selected_model]['description']}*")
            compare_models_list = []
        else:
            selected_model = list(models.keys())[0]  # Default
            compare_models_list = st.multiselect(
                "Select Models to Compare",
                options=list(models.keys()),
                default=list(models.keys())[:2],
                help="Choose multiple models to compare their responses"
            )
        
        st.markdown("---")
        
        # Retrieval method selection
        st.markdown("#### Retrieval Method")
        retrieval_method = st.radio(
            "Select Method",
            options=["Hybrid (Both)", "Baseline Only", "Embeddings Only"],
            help="Choose how to retrieve information from the Knowledge Graph"
        )
        
        st.markdown("---")
        
        # Advanced settings
        st.markdown("#### Advanced")
        top_k = st.slider(
            "Results per method",
            min_value=5,
            max_value=20,
            value=10,
            help="Number of results to retrieve"
        )
        
        show_cypher = st.checkbox(
            "Show Cypher Queries",
            value=True,
            help="Display the Cypher queries executed"
        )
        
        show_context = st.checkbox(
            "Show Raw Context",
            value=True,
            help="Display the raw KG context before LLM processing"
        )
        
        st.markdown("---")
        
        # Example queries
        st.markdown("#### Example Queries")
        example_queries = [
            "Who are the top goal scorers?",
            "Best midfielders for assists",
            "Tell me about Mohamed Salah",
            "Top defenders with clean sheets",
            "Compare Kane and Haaland",
            "Best budget forwards",
            "Who should I captain this week?"
        ]
        
        selected_example = st.selectbox(
            "Try an example",
            options=[""] + example_queries,
            help="Select an example query to try"
        )
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #888; font-size: 0.8em;'>
            Built for FPL managers<br>
            Powered by Neo4j & BGE-M3
        </div>
        """, unsafe_allow_html=True)
        
    return {
        "model": selected_model,
        "model_config": models[selected_model],
        "model_mode": model_mode,
        "compare_models": compare_models_list,
        "all_models": models,
        "retrieval_method": retrieval_method,
        "top_k": top_k,
        "show_cypher": show_cypher,
        "show_context": show_context,
        "example_query": selected_example
    }


def render_entity_badges(entities: Dict[str, List]):
    """Render entity badges"""
    html = ""
    
    for entity_type, values in entities.items():
        if values:
            for val in values:
                if entity_type == "Player":
                    html += f'<span class="player-badge">{val}</span> '
                elif entity_type == "Team":
                    html += f'<span class="stat-badge">{val}</span> '
                elif entity_type == "Position":
                    html += f'<span class="stat-badge">{val}</span> '
                elif entity_type == "Statistic":
                    html += f'<span class="stat-badge">{val}</span> '
                elif entity_type == "Gameweek":
                    html += f'<span class="stat-badge">GW{val}</span> '
                else:
                    html += f'<span class="stat-badge">{val}</span> '
    
    if html:
        st.markdown(html, unsafe_allow_html=True)


def render_baseline_results(results: List[Dict], desc: str):
    """Render baseline retrieval results"""
    st.markdown("##### Baseline Graph Results")
    st.caption(desc)
    
    if results:
        # Convert to display format
        display_data = []
        for i, r in enumerate(results[:10], 1):
            row = {"#": i}
            row.update(r)
            display_data.append(row)
        
        st.dataframe(display_data, use_container_width=True)
    else:
        st.info("No baseline results returned for this query.")


def render_embedding_results(results: List[Dict]):
    """Render embedding search results"""
    st.markdown("##### Semantic Search Results")
    
    if results:
        display_data = []
        for i, r in enumerate(results[:10], 1):
            display_data.append({
                "#": i,
                "Player": r.get("player", "N/A"),
                "Position": r.get("position", "N/A"),
                "Points": r.get("total_points", "N/A"),
                "Score": f"{r.get('score', 0):.4f}"
            })
        
        st.dataframe(display_data, use_container_width=True)
    else:
        st.info("No embedding results returned for this query.")


def render_cypher_query(cypher: str):
    """Render Cypher query in a styled box"""
    st.markdown("##### Cypher Query Preview")
    st.code(cypher, language="cypher")


def render_llm_response(answer: str, metrics: ModelMetrics):
    """Render the LLM response with metrics"""
    st.markdown("""
    <div class="result-card">
        <h4 style="color: #37003c; margin-top: 0;">AI Assistant Response</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(answer)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Response Time", f"{metrics.response_time_sec:.2f}s")
    with col2:
        if metrics.input_tokens:
            st.metric("Input Tokens", metrics.input_tokens)
    with col3:
        if metrics.output_tokens:
            st.metric("Output Tokens", metrics.output_tokens)


def render_recommendations(results: List[Dict], query_type: str = "general"):
    """Render player recommendations with explanations"""
    st.markdown("##### Player Recommendations")
    
    if not results:
        st.info("No specific recommendations available for this query.")
        return
    
    for i, player in enumerate(results[:5], 1):
        with st.expander(f"#{i} {player.get('player', player.get('player_name', 'Unknown'))}"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                position = player.get('position', 'N/A')
                if position == 'GK':
                    st.markdown("🧤 **Goalkeeper**")
                elif position == 'DEF':
                    st.markdown("🛡️ **Defender**")
                elif position == 'MID':
                    st.markdown("⚡ **Midfielder**")
                elif position == 'FWD':
                    st.markdown("⚽ **Forward**")
            
            with col2:
                if 'total_points' in player:
                    st.write(f"**Total Points:** {player['total_points']}")
                if 'total_stat' in player:
                    st.write(f"**Stat Value:** {player['total_stat']}")
                if 'score' in player:
                    st.write(f"**Similarity Score:** {player['score']:.4f}")
                if 'goals_scored' in player:
                    st.write(f"**Goals:** {player['goals_scored']}")
                if 'assists' in player:
                    st.write(f"**Assists:** {player['assists']}")


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    """Main application entry point"""
    render_header()
    settings = render_sidebar()
    config = load_configuration()
    
    # Handle example query selection
    if settings["example_query"]:
        st.session_state["query_input"] = settings["example_query"]
    
    # Main query input
    st.markdown("### Ask Your FPL Question")
    
    query = st.text_input(
        "Enter your question about Fantasy Premier League",
        value=st.session_state.get("query_input", ""),
        placeholder="e.g., Who are the best midfielders for goals this season?",
        key="main_query"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        submit_btn = st.button("Get Answer", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("Clear", use_container_width=True)
    
    if clear_btn:
        st.session_state["query_input"] = ""
        st.rerun()
    
    if submit_btn and query:
        st.markdown("---")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Preprocessing
        status_text.text("Processing query...")
        progress_bar.progress(10)
        
        preprocessing_output = run_preprocessing(query)
        
        # Display preprocessing results
        st.markdown("### Query Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Intent:** `{preprocessing_output.get('intent', 'Unknown')}`")
            if preprocessing_output.get('ranking'):
                st.markdown(f"**Ranking:** `{preprocessing_output.get('ranking')}`")
        
        with col2:
            st.markdown("**Detected Entities:**")
            render_entity_badges(preprocessing_output.get('entities', {}))
        
        progress_bar.progress(25)
        
        # Step 2: Retrieval
        status_text.text("Retrieving from Knowledge Graph...")
        
        baseline_results = []
        embedding_results = []
        cypher_preview = ""
        baseline_desc = ""
        
        retrieval_method = settings["retrieval_method"]
        
        if retrieval_method in ["Hybrid (Both)", "Baseline Only"]:
            baseline_results, baseline_desc = run_baseline_retrieval(preprocessing_output)
            cypher_preview = generate_cypher_preview(preprocessing_output)
        
        progress_bar.progress(50)
        
        if retrieval_method in ["Hybrid (Both)", "Embeddings Only"]:
            positions = preprocessing_output.get('entities', {}).get('Position', [])
            pos_filter = positions[0] if positions else None
            embedding_results = run_embedding_retrieval(
                query, config, top_k=settings["top_k"], position=pos_filter
            )
        
        progress_bar.progress(70)
        
        # Display retrieval results
        if settings["show_context"]:
            st.markdown("### Retrieved Context")
            
            if settings["show_cypher"] and cypher_preview:
                render_cypher_query(cypher_preview)
            
            context_tabs = st.tabs(["Baseline Results", "Embedding Results", "Recommendations"])
            
            with context_tabs[0]:
                if retrieval_method != "Embeddings Only":
                    render_baseline_results(baseline_results, baseline_desc if retrieval_method != "Embeddings Only" else "")
                else:
                    st.info("Baseline retrieval disabled. Switch to Hybrid or Baseline Only to see these results.")
            
            with context_tabs[1]:
                if retrieval_method != "Baseline Only":
                    render_embedding_results(embedding_results)
                else:
                    st.info("Embedding retrieval disabled. Switch to Hybrid or Embeddings Only to see these results.")
            
            with context_tabs[2]:
                # Show recommendations from both sources
                all_results = baseline_results + embedding_results
                render_recommendations(all_results)
        
        progress_bar.progress(85)
        
        # Step 3: LLM Generation
        status_text.text("Generating response with AI...")
        
        # Build context for LLM
        context = RetrievalContext(
            user_query=query,
            intent=preprocessing_output.get('intent', ''),
            entities=preprocessing_output.get('entities', {}),
            baseline_desc=baseline_desc if retrieval_method != "Embeddings Only" else "",
            baseline_results=baseline_results if retrieval_method != "Embeddings Only" else [],
            baseline_is_fallback=False,
            embedding_results=embedding_results if retrieval_method != "Baseline Only" else [],
        )
        
        prompt = build_structured_prompt(context)
        
        # Check if we're in comparison mode
        if settings.get("model_mode") == "Compare Models" and settings.get("compare_models"):
            # Compare multiple models
            progress_bar.progress(90)
            status_text.text("Comparing multiple models...")
            
            st.markdown("### Model Comparison")
            
            comparison_results = []
            model_cols = st.columns(len(settings["compare_models"]))
            
            for idx, model_name in enumerate(settings["compare_models"]):
                model_config = settings["all_models"][model_name]
                try:
                    answer, metrics = call_llm(prompt, model_config)
                    comparison_results.append({
                        "model": model_name,
                        "answer": answer,
                        "metrics": metrics,
                        "success": True
                    })
                except Exception as e:
                    comparison_results.append({
                        "model": model_name,
                        "answer": f"Error: {str(e)}",
                        "metrics": None,
                        "success": False
                    })
            
            progress_bar.progress(100)
            status_text.text("Complete!")
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()
            
            # Display comparison results side by side
            for idx, result in enumerate(comparison_results):
                with model_cols[idx]:
                    st.markdown(f"#### {result['model']}")
                    if result['success']:
                        st.markdown(result['answer'])
                        if result['metrics']:
                            st.caption(f"Response time: {result['metrics'].response_time_sec:.2f}s")
                    else:
                        st.error(result['answer'])
            
            # Show metrics comparison table
            st.markdown("#### Performance Metrics")
            metrics_data = []
            for result in comparison_results:
                if result['success'] and result['metrics']:
                    metrics_data.append({
                        "Model": result['model'],
                        "Response Time (s)": f"{result['metrics'].response_time_sec:.2f}",
                        "Input Tokens": result['metrics'].input_tokens or "N/A",
                        "Output Tokens": result['metrics'].output_tokens or "N/A"
                    })
            if metrics_data:
                st.dataframe(metrics_data, use_container_width=True)
        
        else:
            # Single model mode
            try:
                answer, metrics = call_llm(prompt, settings["model_config"])
                
                progress_bar.progress(100)
                status_text.text("Complete!")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()
                
                # Display LLM response
                st.markdown("### Answer")
                render_llm_response(answer, metrics)
                
            except Exception as e:
                progress_bar.progress(100)
                status_text.empty()
                st.error(f"Error generating response: {str(e)}")
                st.info("Try selecting a different model or checking your API configuration.")
        
        # Show full context (collapsible)
        if settings["show_context"]:
            with st.expander("View Full LLM Prompt"):
                st.text(prompt)
    
    elif submit_btn:
        st.warning("Please enter a question to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>FPL Graph-RAG Assistant | Built with Streamlit, Neo4j, and BGE-M3</p>
        <p>Using Knowledge Graph + Retrieval Augmented Generation for intelligent FPL advice</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
