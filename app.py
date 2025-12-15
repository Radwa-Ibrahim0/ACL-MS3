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
import base64
import os
from typing import Dict, List, Any, Optional, Tuple

# Import the Graph-RAG system components
from preprocessing import process_user_query
from baseline import execute_baseline_query, load_config, Neo4jConnection, BaselineQueryBuilder
from embedding_bge_m3 import SemanticSearchBGEM3, load_config as load_config_embed
from embedding_minilm import SemanticSearchMiniLM
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
    page_title="FPL Assistant",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS - FPL INSPIRED THEME
# ============================================================

st.markdown("""
<style>
    :root {
        --fpl-purple: #37003c;
        --fpl-magenta: #963cff;
        --fpl-cyan: #04f5ff;
        --fpl-green: #00ff87;
        --fpl-white: #ffffff;
        --fpl-gray-100: #f8f9fa;
        --fpl-gray-200: #e9ecef;
        --fpl-gray-300: #dee2e6;
        --fpl-gray-400: #adb5bd;
        --fpl-gray-600: #6c757d;
    }

    html, body, [class*="css"] {
        font-family: 'Poppins', -apple-system, BlinkMacSystemFont, sans-serif;
        background: var(--fpl-gray-100);
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* Remove white space around banner and content */
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 1200px;
    }

    /* Reduce spacing between elements */
    .main .block-container > div {
        margin-bottom: 0 !important;
    }

    /* Remove form border/box */
    [data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
        background: transparent !important;
    }

    /* Tighter spacing for headings */
    h3 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Banner taller with increased height and padding */
    .fpl-header {
        background: linear-gradient(135deg, #37003c 0%, #963cff 50%, #00ff87 100%);
        padding: 0;
        border-radius: 12px;
        margin-bottom: 1rem;
        overflow: hidden;
        position: relative;
        display: flex;
        align-items: stretch;
        min-height: 220px;
    }

    .fpl-header-content {
        padding: 2.5rem 2.5rem;
        position: relative;
        z-index: 2;
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    /* Photo moved to the left with adjusted positioning */
    .fpl-header-image {
        position: relative;
        width: 340px;
        min-height: 100%;
        display: flex;
        align-items: flex-end;
        justify-content: flex-start;
        overflow: hidden;
        padding-left: 10px;
    }

    .fpl-header-image img {
        height: 180px;
        width: auto;
        object-fit: contain;
        object-position: bottom left;
    }

    .fpl-header-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        padding: 8px 16px;
        border-radius: 20px;
        margin-bottom: 0.75rem;
    }

    .fpl-header-badge span {
        color: var(--fpl-green);
        font-size: 0.9rem;
        font-weight: 500;
    }

    .fpl-header h1 {
        color: #ffffff !important;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.5px;
    }

    .fpl-header p {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.05rem;
        margin: 0;
        font-weight: 400;
    }

    .info-card {
        background: var(--fpl-white);
        border-left: 3px solid var(--fpl-green);
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(55, 0, 60, 0.06);
    }

    .context-card {
        background: var(--fpl-white);
        border: 1px solid var(--fpl-gray-300);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    .result-card {
        background: var(--fpl-white);
        border: 2px solid var(--fpl-magenta);
        padding: 1.25rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 8px 24px rgba(55, 0, 60, 0.06);
    }

    .player-badge {
        display: inline-block;
        background: var(--fpl-purple);
        color: var(--fpl-green);
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-weight: 500;
        font-size: 0.85rem;
        margin: 0.15rem;
    }

    .stat-badge {
        display: inline-block;
        background: var(--fpl-gray-100);
        color: var(--fpl-purple);
        padding: 0.25rem 0.75rem;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.85rem;
        margin: 0.15rem;
        border: 1px solid var(--fpl-gray-300);
    }

    .position-badge {
        display: inline-block;
        background: var(--fpl-magenta);
        color: var(--fpl-white);
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
    }

    /* Entity badge with category label */
    .entity-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        margin: 0.15rem;
    }

    .entity-category {
        background: var(--fpl-gray-200);
        color: var(--fpl-gray-600);
        padding: 0.2rem 0.5rem;
        border-radius: 4px 0 0 4px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
    }

    .entity-value {
        background: var(--fpl-purple);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 0 4px 4px 0;
        font-size: 0.8rem;
        font-weight: 500;
    }

    /* Non-copiable intent text */
    .intent-display {
        background: var(--fpl-gray-200);
        color: var(--fpl-purple);
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
        user-select: none;
        -webkit-user-select: none;
        -moz-user-select: none;
        -ms-user-select: none;
        display: inline-block;
    }

    .threshold-display {
        background: var(--fpl-gray-200);
        color: var(--fpl-purple);
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }

    /* Ultra-compact sidebar with no scroll */
    section[data-testid="stSidebar"] {
        background: var(--fpl-gray-100);
        border-right: 1px solid var(--fpl-gray-300);
        box-shadow: none;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding: 0.5rem 0.5rem 0.5rem;
    }

    /* Remove top padding from sidebar */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-top: 0.5rem !important;
    }

    /* Bigger Settings title - no extra margin */
    .sidebar-title {
        color: var(--fpl-purple);
        font-size: 1.25rem;
        font-weight: 700;
        margin: 0 0 0.35rem 0;
        padding: 0;
        line-height: 1.2;
    }

    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: var(--fpl-purple) !important;
        margin: 0;
        font-weight: 600;
        font-size: 0.8rem;
    }

    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] .stRadio label {
        color: var(--fpl-purple) !important;
        font-weight: 500;
        margin-bottom: 0 !important;
        font-size: 0.8rem;
    }

    /* Even tighter spacing for sidebar components */
    section[data-testid="stSidebar"] [data-testid="stSelectbox"],
    section[data-testid="stSidebar"] [data-testid="stSlider"],
    section[data-testid="stSidebar"] [data-testid="stCheckbox"],
    section[data-testid="stSidebar"] [data-testid="stRadio"],
    section[data-testid="stSidebar"] [data-testid="stMultiselect"],
    section[data-testid="stSidebar"] [data-testid="stExpander"] {
        margin-bottom: 0.1rem;
    }

    section[data-testid="stSidebar"] .stCheckbox,
    section[data-testid="stSidebar"] .stRadio {
        margin-bottom: 0;
    }
    
    section[data-testid="stSidebar"] .stRadio > div {
        gap: 0.1rem;
    }

    section[data-testid="stSidebar"] .stSelectbox > div > div {
        padding-top: 0;
        padding-bottom: 0;
    }

    /* Expander styling for collapsible sections */
    section[data-testid="stSidebar"] [data-testid="stExpander"] {
        border: none;
        background: transparent;
    }

    section[data-testid="stSidebar"] [data-testid="stExpander"] summary {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--fpl-purple);
        padding: 0.25rem 0;
    }

    section[data-testid="stSidebar"] [data-testid="stExpander"] > div {
        padding: 0;
    }

    /* ========== PURPLE RADIO BUTTONS & CHECKBOXES ========== */
    /* Native HTML input accent */
    input[type="radio"],
    input[type="checkbox"] {
        accent-color: #37003c !important;
    }

    /* BaseWeb Radio - stable sizing */
    [data-baseweb="radio"] {
        align-items: center !important;
    }

    [data-baseweb="radio"] > div:first-child {
        border-color: #37003c !important;
        background-color: transparent !important;
        width: 20px !important;
        height: 20px !important;
        min-width: 20px !important;
        min-height: 20px !important;
    }

    /* BaseWeb Radio - inner fill when checked */
    [data-baseweb="radio"] > div:first-child > div {
        background-color: #37003c !important;
        width: 10px !important;
        height: 10px !important;
    }

    /* ========== CHECKBOX PURPLE OVERRIDE ========== */
    /* Target the checkbox container */
    .stCheckbox > label > div:first-child {
        background-color: transparent !important;
        border-color: #37003c !important;
    }

    /* Checked state */
    .stCheckbox > label > div:first-child[data-checked="true"],
    .stCheckbox [aria-checked="true"] > div:first-child {
        background-color: #37003c !important;
        border-color: #37003c !important;
    }

    /* BaseWeb Checkbox styling */
    div[data-baseweb="checkbox"] > div:first-child {
        border-color: #37003c !important;
    }

    div[data-baseweb="checkbox"][aria-checked="true"] > div:first-child {
        background-color: #37003c !important;
        border-color: #37003c !important;
    }

    /* Override any inline styles on checkbox */
    .stCheckbox div[role="checkbox"] > div:first-child {
        background-color: inherit;
        border-color: #37003c !important;
    }

    .stCheckbox div[role="checkbox"][aria-checked="true"] > div:first-child {
        background-color: #37003c !important;
    }

    /* Checkbox checkmark SVG */
    [data-baseweb="checkbox"] svg,
    .stCheckbox svg {
        fill: white !important;
        stroke: white !important;
    }

    /* ========== FORCE SIDEBAR TO STAY OPEN ========== */
    /* Hide collapse button */
    [data-testid="collapsedControl"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }

    /* Hide sidebar toggle button */
    button[kind="header"],
    [data-testid="stSidebarCollapseButton"],
    section[data-testid="stSidebar"] button[aria-label="Close sidebar"],
    section[data-testid="stSidebar"] button[aria-expanded] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Force sidebar to always be visible */
    section[data-testid="stSidebar"] {
        transform: none !important;
        width: 300px !important;
        min-width: 300px !important;
    }

    section[data-testid="stSidebar"][aria-expanded="false"] {
        transform: none !important;
        margin-left: 0 !important;
    }

    section[data-testid="stSidebar"] hr {
        margin: 0.15rem 0 !important;
        border-color: var(--fpl-gray-300) !important;
    }

    /* Tabs without orange */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--fpl-gray-200);
        padding: 4px;
        border-radius: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 6px;
        color: var(--fpl-gray-600);
        padding: 8px 16px;
        font-weight: 500;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: var(--fpl-white);
        color: var(--fpl-purple);
    }

    .stTabs [aria-selected="true"] {
        background: var(--fpl-white) !important;
        color: var(--fpl-purple) !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background-color: transparent !important;
    }

    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* Submit button purple styling */
    button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {
        background: var(--fpl-purple) !important;
        color: var(--fpl-white) !important;
        font-weight: 600 !important;
        border: none !important;
        height: 42px !important;
        border-radius: 6px !important;
        box-shadow: none !important;
    }

    button[kind="primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        background: #4a0050 !important;
    }

    /* Arrow button styling */
    .arrow-btn button {
        background: var(--fpl-purple) !important;
        color: var(--fpl-white) !important;
        font-weight: 600;
        border: none !important;
        padding: 0 !important;
        border-radius: 6px !important;
        box-shadow: none !important;
        height: 42px !important;
        min-height: 42px !important;
        width: 42px !important;
        min-width: 42px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 1.25rem !important;
    }

    .arrow-btn button:hover {
        background: #4a0050 !important;
    }

    div[data-testid="stFormSubmitButton"] button {
        background: var(--fpl-purple);
        color: var(--fpl-white);
        font-weight: 600;
        border: none;
        height: 42px;
        border-radius: 6px;
        padding: 0 1rem;
        box-shadow: none;
        min-height: 42px;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background: #4a0050;
        transform: none;
        box-shadow: none;
    }

    /* Form input styling */
    .stTextInput > div > div > input {
        height: 42px;
        border-radius: 6px;
        border: 1px solid var(--fpl-gray-300);
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--fpl-purple);
        box-shadow: 0 0 0 1px var(--fpl-purple);
    }

    .stSuccess {
        background-color: rgba(0, 255, 135, 0.1) !important;
        border-color: var(--fpl-green) !important;
    }

    .stInfo {
        background-color: rgba(55, 0, 60, 0.06) !important;
        border-color: var(--fpl-purple) !important;
    }

    .dataframe {
        border: 1px solid var(--fpl-gray-300) !important;
    }

    .dataframe th {
        background: var(--fpl-purple) !important;
        color: var(--fpl-green) !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }

    /* Example query pills - beautiful gradient style */
    .query-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 0.5rem;
        margin-top: 0.5rem;
    }

    .query-pill {
        background: linear-gradient(135deg, #37003c 0%, #963cff 100%);
        color: white !important;
        padding: 10px 20px;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 2px 8px rgba(55, 0, 60, 0.3);
        text-decoration: none;
    }

    .query-pill:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(150, 60, 255, 0.4);
        background: linear-gradient(135deg, #4a0050 0%, #a855f7 100%);
    }

    /* Toggle buttons for settings */
    .toggle-container {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }

    .toggle-btn {
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 2px solid var(--fpl-gray-300);
        background: var(--fpl-white);
        color: var(--fpl-gray-600);
    }

    .toggle-btn.active {
        background: var(--fpl-purple);
        color: white;
        border-color: var(--fpl-purple);
    }

    .toggle-btn:hover:not(.active) {
        border-color: var(--fpl-purple);
        color: var(--fpl-purple);
    }

    /* Segmented control / pill selector for sidebar options */
    .segment-container {
        display: flex;
        background: var(--fpl-gray-200);
        border-radius: 8px;
        padding: 3px;
        gap: 3px;
    }

    .segment-btn {
        flex: 1;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        border: none;
        background: transparent;
        color: var(--fpl-gray-600);
        text-align: center;
    }

    .segment-btn.active {
        background: var(--fpl-purple);
        color: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }

    /* Pill button styling for example queries - extra rounded */
    .query-pills button {
        background: linear-gradient(135deg, #37003c 0%, #963cff 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 8px 16px !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        box-shadow: 0 2px 8px rgba(55, 0, 60, 0.3) !important;
        transition: all 0.3s ease !important;
    }

    .query-pills button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(150, 60, 255, 0.4) !important;
        background: linear-gradient(135deg, #4a0050 0%, #a855f7 100%) !important;
    }

    /* Input and submit in same row - no extra box */
    .input-row {
        display: flex;
        gap: 8px;
        align-items: center;
    }

    /* Full prompt code block styling with black background */
    .full-prompt-box {
        background: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Fira Code', 'Consolas', 'Monaco', monospace;
        font-size: 0.85rem;
        line-height: 1.5;
        white-space: pre-wrap;
        word-wrap: break-word;
        overflow-x: auto;
        border: 1px solid #333;
    }

    .footer {
        text-align: center;
        color: var(--fpl-gray-400);
        padding: 1rem;
        margin-top: 1rem;
        border-top: 1px solid var(--fpl-gray-300);
        font-size: 0.8rem;
    }

    h3, h4, h5 {
        color: var(--fpl-purple) !important;
        font-weight: 600 !important;
    }

    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0.25rem;
    }

    /* Select box focus */
    .stSelectbox > div > div {
        border-color: var(--fpl-gray-300);
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: var(--fpl-purple) !important;
        box-shadow: 0 0 0 1px var(--fpl-purple) !important;
    }

    /* Multiselect tags */
    [data-baseweb="tag"] {
        background-color: var(--fpl-purple) !important;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--fpl-purple), var(--fpl-green)) !important;
    }

    /* Compact intent and entities display */
    .query-analysis-container {
        background: var(--fpl-white);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid var(--fpl-gray-300);
        margin: 0.5rem 0 1rem 0;
    }

    .intent-section strong {
        color: var(--fpl-purple);
        font-weight: 600;
        display: block;
        margin-bottom: 0.3rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# INITIALIZATION & CACHING
# ============================================================

def load_configuration():
    """Load configuration from config.txt (no caching - always fresh)"""
    return load_config()


def get_available_models(config: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """Get available LLM models with their configurations (no caching - always fresh API key)"""
    api_key = config.get("OPENROUTER_API_KEY", "")
    
    models = {
        "Mistral-7B-Instruct": {
            "name": "Mistral-7B",
            "model": "mistralai/mistral-7b-instruct:free",
            "api_key": api_key,
            "description": "Good balance of speed and quality"
        },
        "GPT-OSS-20B": {
            "name": "GPT-OSS-20B",
            "model": "openai/gpt-oss-20b:free",
            "api_key": api_key,
            "description": "Fast and efficient for general queries"
        },
        "Gemma-3-27B": {
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


def run_embedding_retrieval(query: str, config: Dict[str, str], top_k: int = 10, position: Optional[str] = None, model: str = "BGE-M3") -> List[Dict]:
    """Run embedding-based semantic search with selected model"""
    try:
        if model == "MiniLM":
            semantic = SemanticSearchMiniLM(config)
        else:  # Default to BGE-M3
            semantic = SemanticSearchBGEM3(config)
        
        results = semantic.search(query, top_k=top_k, position=position)
        semantic.close()
        return results
    except Exception as e:
        st.error(f"Embedding search error ({model}): {e}")
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

def get_base64_image(image_path: str) -> str:
    """Load image and convert to base64 for HTML embedding"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        return ""


def render_header():
    """Render the FPL-style hero banner with player image"""
    # Get the directory where app.py is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "player-comp-4-2x-D7jkZCyT.png")
    
    # Convert image to base64 for HTML embedding
    img_base64 = get_base64_image(image_path)
    
    # Build image tag - use base64 if available, otherwise hide image section
    if img_base64:
        image_html = f'<img src="data:image/png;base64,{img_base64}" alt="Premier League Players" />'
    else:
        image_html = ''
    
    st.markdown(f"""
    <div class="fpl-header">
        <div class="fpl-header-content">
            <div class="fpl-header-badge">
                <span>⚽</span>
                <span>Fantasy Premier League</span>
            </div>
            <h1>FPL Assistant</h1>
            <p>AI-powered Fantasy Football advisor using Knowledge Graph technology</p>
        </div>
        <div class="fpl-header-image">
            {image_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with dropdown and toggle settings"""
    config = load_configuration()
    models = get_available_models(config)
    
    with st.sidebar:
        st.markdown('<p class="sidebar-title">⚙️ Settings</p>', unsafe_allow_html=True)
        
        with st.expander("🤖 LLM Model", expanded=True):
            # Use selectbox instead of radio for model mode
            model_mode = st.selectbox(
                "Mode",
                options=["Single Model", "Compare Models"],
                help="Choose single model or compare multiple models",
                label_visibility="collapsed"
            )
            
            if model_mode == "Single Model":
                selected_model = st.selectbox(
                    "Select Model",
                    options=list(models.keys()),
                    help="Choose which AI model to use",
                    label_visibility="collapsed",
                    key="model_select"
                )
                st.caption(f"_{models[selected_model]['description']}_")
                compare_models_list = []
            else:
                selected_model = list(models.keys())[0]
                compare_models_list = st.multiselect(
                    "Models to Compare",
                    options=list(models.keys()),
                    default=list(models.keys())[:2],
                    label_visibility="collapsed"
                )
        
        with st.expander("🔍 Retrieval Method", expanded=True):
            # Use selectbox instead of radio
            retrieval_method = st.selectbox(
                "Method",
                options=["Hybrid (Both)", "Baseline Only", "Embeddings Only"],
                help="How to retrieve from Knowledge Graph",
                label_visibility="collapsed"
            )
            
            # Embedding model selector
            embedding_model = st.selectbox(
                "Embedding Model",
                options=["BGE-M3", "MiniLM"],
                help="BGE-M3: Higher quality (1024 dims) | MiniLM: Faster (384 dims)",
                label_visibility="visible"
            )
            st.caption(f"_{'🎯 High quality, slower' if embedding_model == 'BGE-M3' else '⚡ Fast inference, lighter'}_")
        
        with st.expander("🔧 Display Options", expanded=False):
            # Use multiselect styled as pills for display options
            display_options = st.multiselect(
                "Show in results",
                options=["📝 Cypher Queries", "📄 Raw Context"],
                default=["📝 Cypher Queries", "📄 Raw Context"],
                label_visibility="collapsed"
            )
            show_cypher = "📝 Cypher Queries" in display_options
            show_context = "📄 Raw Context" in display_options
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: var(--fpl-gray-400); font-size: 0.7em;'>
            Built for FPL managers<br>
            Powered by Neo4j & Embeddings
        </div>
        """, unsafe_allow_html=True)
        
    return {
        "model": selected_model,
        "model_config": models[selected_model],
        "model_mode": model_mode,
        "compare_models": compare_models_list,
        "all_models": models,
        "retrieval_method": retrieval_method,
        "embedding_model": embedding_model,
        "top_k": 10,
        "show_cypher": show_cypher,
        "show_context": show_context,
    }


def render_entity_badges(entities: Dict[str, List]):
    """Render entity badges with category labels - unified purple theme"""
    html = ""
    
    for entity_type, values in entities.items():
        if values:
            for val in values:
                display_val = f"GW{val}" if entity_type == "Gameweek" else val
                html += f'''<span class="entity-badge">
                    <span class="entity-category">{entity_type}</span>
                    <span class="entity-value">{display_val}</span>
                </span>'''
    
    if html:
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.caption("None detected")


def render_baseline_results(results: List[Dict], desc: str):
    """Render baseline retrieval results"""
    st.markdown("##### 🔍 Baseline Graph Results")
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


def render_embedding_results(results: List[Dict], model_name: str = "BGE-M3"):
    """Render embedding search results"""
    st.markdown(f"##### 🧠 Semantic Search Results ({model_name})")
    
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
    st.markdown("##### 📝 Cypher Query Preview")
    st.code(cypher, language="cypher")


def render_llm_response(answer: str, metrics: ModelMetrics):
    """Render the LLM response with metrics"""
    # Header styled container
    st.markdown("""
    <div class="result-card">
        <h4 style="color: #37003c; margin-top: 0;">🤖 AI Assistant Response</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Use st.markdown for the answer so markdown formatting works
    st.markdown(answer)
    
    # Metrics in collapsible section
    with st.expander("⚡ Performance Metrics", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⏱️ Response Time", f"{metrics.response_time_sec:.2f}s")
        with col2:
            if metrics.input_tokens:
                st.metric("📥 Input Tokens", metrics.input_tokens)
        with col3:
            if metrics.output_tokens:
                st.metric("📤 Output Tokens", metrics.output_tokens)


def render_recommendations(results: List[Dict], query_type: str = "general"):
    """Render player recommendations with explanations"""
    st.markdown("##### ⭐ Player Recommendations")
    
    if not results:
        st.info("No specific recommendations available for this query.")
        return
    
    for i, player in enumerate(results[:5], 1):
        with st.expander(f"#{i} {player.get('player', player.get('player_name', 'Unknown'))}"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                position = player.get('position', 'N/A')
                if position == 'GK':
                    st.markdown("Goalkeeper")
                elif position == 'DEF':
                    st.markdown("Defender")
                elif position == 'MID':
                    st.markdown("Midfielder")
                elif position == 'FWD':
                    st.markdown("Forward")
            
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
    
    st.markdown("### 💬 Ask Your FPL Question")
    
    # Initialize session state
    if "query_input" not in st.session_state:
        st.session_state["query_input"] = ""
    
    # Get current query from state
    current_query = st.session_state.get("query_input", "")

    # Wrap input + submit in a form so pressing Enter runs the query
    with st.form("query_form"):
        col_input, col_btn = st.columns([10, 1])
        with col_input:
            query = st.text_input(
                "Enter your question",
                value=current_query,
                placeholder="e.g., Who are the best midfielders for goals?",
                key="main_query",
                label_visibility="collapsed"
            )
        with col_btn:
            submit_btn = st.form_submit_button("→", use_container_width=True)

    # Update session state with the current query value
    if query != current_query:
        st.session_state["query_input"] = query

    if submit_btn and query:
        st.markdown("---")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Preprocessing
        status_text.text("Processing query...")
        progress_bar.progress(10)
        
        preprocessing_output = run_preprocessing(query)
        
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
                query, config, top_k=settings["top_k"], position=pos_filter,
                model=settings["embedding_model"]
            )
        
        progress_bar.progress(70)
        
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
        
        progress_bar.progress(85)
        
        # Step 3: LLM Generation
        status_text.text("Generating response with AI...")
        
        # Check if we're in comparison mode
        if settings.get("model_mode") == "Compare Models" and settings.get("compare_models"):
            # Compare multiple models
            progress_bar.progress(90)
            status_text.text("Comparing multiple models...")
            
            comparison_results = []
            
            for model_name in settings["compare_models"]:
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
            
            # DISPLAY AI RESPONSE FIRST
            st.markdown("### 🎯 Answer")
            
            model_cols = st.columns(len(comparison_results))
            for idx, result in enumerate(comparison_results):
                with model_cols[idx]:
                    st.markdown(f"#### {result['model']}")
                    if result['success']:
                        st.markdown(result['answer'])
                        if result['metrics']:
                            st.caption(f"Response time: {result['metrics'].response_time_sec:.2f}s")
                    else:
                        st.error(result['answer'])
            
            # Show metrics comparison table in collapsible
            with st.expander("⚡ Performance Metrics", expanded=False):
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
            
            # Each section individually collapsible
            with st.expander("🔎 Query Analysis", expanded=False):
                col1, col2, col3, col4 = st.columns([1.2, 1, 1, 2])
                with col1:
                    st.markdown("**🎯 Intent**")
                    intent_val = preprocessing_output.get('intent', 'Unknown')
                    st.markdown(f'<span class="intent-display">{intent_val}</span>', unsafe_allow_html=True)
                with col2:
                    st.markdown("**📊 Ranking**")
                    ranking_val = preprocessing_output.get('ranking', '—')
                    st.markdown(f'<span class="intent-display">{ranking_val if ranking_val else "—"}</span>', unsafe_allow_html=True)
                with col3:
                    st.markdown("**🔢 Threshold**")
                    threshold_val = preprocessing_output.get('threshold', '—')
                    st.markdown(f'<span class="threshold-display">{threshold_val if threshold_val else "—"}</span>', unsafe_allow_html=True)
                with col4:
                    st.markdown("**🏷️ Entities**")
                    render_entity_badges(preprocessing_output.get('entities', {}))
            
            if settings["show_cypher"] and cypher_preview:
                with st.expander("📝 Cypher Query Preview", expanded=False):
                    render_cypher_query(cypher_preview)
            
            # Combined context in ONE collapsible with tabs
            with st.expander("📚 Retrieved Context & Recommendations", expanded=False):
                context_tabs = st.tabs(["Baseline Results", "Embedding Results", "Recommendations"])
                
                with context_tabs[0]:
                    if retrieval_method != "Embeddings Only":
                        render_baseline_results(baseline_results, baseline_desc)
                    else:
                        st.info("Baseline retrieval disabled. Switch to Hybrid or Baseline Only to see these results.")
                
                with context_tabs[1]:
                    if retrieval_method != "Baseline Only":
                        render_embedding_results(embedding_results)
                    else:
                        st.info("Embedding retrieval disabled. Switch to Hybrid or Embeddings Only to see these results.")
                
                with context_tabs[2]:
                    all_results = baseline_results + embedding_results
                    render_recommendations(all_results)
            
            if settings["show_context"]:
                with st.expander("💬 Full LLM Prompt", expanded=False):
                    st.markdown(f'<div class="full-prompt-box">{prompt}</div>', unsafe_allow_html=True)
        
        else:
            # Single model mode
            try:
                answer, metrics = call_llm(prompt, settings["model_config"])
                
                progress_bar.progress(100)
                status_text.text("Complete!")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()
                
                # DISPLAY AI RESPONSE FIRST
                st.markdown("### 🎯 Answer")
                render_llm_response(answer, metrics)
                
                # Each section individually collapsible
                with st.expander("🔎 Query Analysis", expanded=False):
                    col1, col2, col3, col4 = st.columns([1.2, 1, 1, 2])
                    with col1:
                        st.markdown("**🎯 Intent**")
                        intent_val = preprocessing_output.get('intent', 'Unknown')
                        st.markdown(f'<span class="intent-display">{intent_val}</span>', unsafe_allow_html=True)
                    with col2:
                        st.markdown("**📊 Ranking**")
                        ranking_val = preprocessing_output.get('ranking', '—')
                        st.markdown(f'<span class="intent-display">{ranking_val if ranking_val else "—"}</span>', unsafe_allow_html=True)
                    with col3:
                        st.markdown("**🔢 Threshold**")
                        threshold_val = preprocessing_output.get('threshold', '—')
                        st.markdown(f'<span class="threshold-display">{threshold_val if threshold_val else "—"}</span>', unsafe_allow_html=True)
                    with col4:
                        st.markdown("**🏷️ Entities**")
                        render_entity_badges(preprocessing_output.get('entities', {}))
                
                if settings["show_cypher"] and cypher_preview:
                    with st.expander("📝 Cypher Query Preview", expanded=False):
                        render_cypher_query(cypher_preview)
                
                # Combined context in ONE collapsible with tabs
                with st.expander("📚 Retrieved Context & Recommendations", expanded=False):
                    context_tabs = st.tabs(["Baseline Results", "Embedding Results", "Recommendations"])
                    
                    with context_tabs[0]:
                        if retrieval_method != "Embeddings Only":
                            render_baseline_results(baseline_results, baseline_desc)
                        else:
                            st.info("Baseline retrieval disabled. Switch to Hybrid or Baseline Only to see these results.")
                    
                    with context_tabs[1]:
                        if retrieval_method != "Baseline Only":
                            render_embedding_results(embedding_results)
                        else:
                            st.info("Embedding retrieval disabled. Switch to Hybrid or Embeddings Only to see these results.")
                    
                    with context_tabs[2]:
                        all_results = baseline_results + embedding_results
                        render_recommendations(all_results)
                
                if settings["show_context"]:
                    with st.expander("💬 Full LLM Prompt", expanded=False):
                        st.markdown(f'<div class="full-prompt-box">{prompt}</div>', unsafe_allow_html=True)
                
            except Exception as e:
                progress_bar.progress(100)
                status_text.empty()
                st.error(f"Error generating response: {str(e)}")
                st.info("Try selecting a different model or checking your API configuration.")
    
    elif submit_btn:
        st.warning("Please enter a question to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>FPL Assistant | Built with Streamlit, Neo4j, and BGE-M3</p>
        <p>Using Knowledge Graph + Retrieval Augmented Generation for intelligent FPL advice</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()