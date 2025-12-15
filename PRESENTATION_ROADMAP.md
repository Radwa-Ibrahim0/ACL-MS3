# FPL Graph-RAG Assistant - Presentation Roadmap

## 📋 System Pipeline Overview

### 1️⃣ **Preprocessing Layer** (`preprocessing.py`)
**Input:** Raw user query (natural language)

**Technical Implementation - Hybrid Approach:**

#### **Stage 1: Rule-Based Extraction**
- **Pattern Matching** using regex for:
  - Player names (from predefined list loaded from `rule.txt`)
  - Team names (Premier League teams)
  - Positions (GK, DEF, MID, FWD variants)
  - Statistics keywords (goals, assists, clean sheets, etc.)
  - Gameweek numbers (e.g., "GW5", "gameweek 10")
- **Keyword Detection** for ranking terms:
  - Best/top/highest → "best"
  - Worst/bottom/lowest → "worst"
- **Numeric Extraction** for thresholds ("top 10", "more than 5")

**Why Rule-Based First?**
- Fast and deterministic
- No API costs
- Handles exact matches perfectly
- Provides structured input for LLM

#### **Stage 2: LLM-Based Intent Classification**
- **Input:** User query + rule-based extracted entities
- **LLM Model:** Uses OpenRouter API (same as main LLM layer)
- **Prompt Engineering:** Few-shot learning with examples:
  - "Who scored the most goals?" → `statistics_ranking`
  - "Tell me about Salah" → `player_info`
  - "Compare Kane and Haaland" → `comparison`
  - "Arsenal's fixtures" → `team_info`
- **Output:** Intent classification from predefined categories

**Why Add LLM?**
- Handles ambiguous queries
- Understands context beyond keywords
- Adapts to natural language variations
- Can infer intent even without exact matches

#### **Combined Output:**
- **Entities** - From rule-based extraction (high precision)
- **Intent** - From LLM classification (high recall)
- **Ranking & Threshold** - From rule-based patterns

**Output:** Structured query object containing:
```json
{
  "intent": "statistics_ranking",
  "entities": {
    "Player": ["Salah"],
    "Position": ["MID"],
    "Statistic": ["goals_scored"]
  },
  "ranking": "best",
  "threshold": 10
}
```

---

### 2️⃣ **Retrieval Layer**

#### **Option A: Baseline Retrieval** (`baseline.py`)
- **Method:** Rule-based Cypher query generation
- **Technical Implementation:**
  1. **Query Builder Class** (`BaselineQueryBuilder`):
     - Maps intent → Cypher template
     - Injects entities into WHERE clauses
     - Handles ranking with ORDER BY (ASC/DESC)
     - Applies thresholds with LIMIT
  2. **Neo4j Connection** (`Neo4jConnection`):
     - Bolt protocol connection
     - Credentials from `config.txt`
     - Session management with auto-close
  3. **Query Execution:**
     - Parameterized queries to prevent injection
     - Returns list of dictionaries
     - Error handling with fallback queries

**Example Cypher Generation:**
```cypher
// Intent: statistics_ranking, Entity: goals_scored, Ranking: best
MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
WITH p, SUM(r.goals_scored) AS total_goals
WHERE total_goals > 0
RETURN p.player_name AS player, total_goals
ORDER BY total_goals DESC
LIMIT 10
```

- **Strengths:** Precise, deterministic, fast (<100ms)
- **Use Case:** Structured queries with clear entities

#### **Option B: Semantic Retrieval** (Embedding-Based Search)

**Embedding Model Comparison:**

| Metric | BGE-M3 | MiniLM |
|--------|--------|--------|
| **Model** | `BAAI/bge-m3` | `sentence-transformers/all-MiniLM-L6-v2` |
| **Dimensions** | 1024 | 384 |
| **Context Quality** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Good |
| **Embedding Time** | ~150ms/query | ~50ms/query |
| **Model Size** | ~2.3GB | ~90MB |
| **Semantic Accuracy** | Higher (better for complex queries) | Lower (simpler queries) |
| **Memory Usage** | ~3GB RAM | ~500MB RAM |
| **Default in UI** | ✅ Yes | No |

**Why BGE-M3 is Default:**
- Superior semantic understanding
- Better handles FPL-specific terminology
- Multi-lingual capability (future-proof)
- State-of-the-art performance on retrieval benchmarks

**Why Keep MiniLM as Option:**
- 3x faster inference
- Lower resource requirements
- Good enough for simple queries
- User can choose based on performance needs

**Technical Implementation** (`embedding_bge_m3.py` & `embedding_minilm.py`):
1. **Query Embedding:**
   - Load pre-trained model from HuggingFace
   - Tokenize user query
   - Generate dense vector representation
2. **Player Embeddings (Pre-computed):**
   - Each player has embedding based on:
     - Player name
     - Position
     - Aggregated statistics (goals, assists, etc.)
     - Team affiliation
   - Stored in Neo4j as node properties
3. **Similarity Search:**
   - Cosine similarity between query and player vectors
   - Optional position filter (WHERE clause)
   - Returns top-k (default: 10) with similarity scores
4. **Neo4j Vector Index:**
   - Uses Neo4j's vector search capabilities
   - Pre-indexed for fast retrieval (<200ms)

**UI Selection:**
- Users can toggle between BGE-M3 and MiniLM in sidebar
- BGE-M3 pre-selected as default
- Trade-off: Quality vs. Speed

- **Strengths:** Handles fuzzy queries, captures semantic meaning, no entity extraction needed
- **Use Case:** Natural language queries without clear entities ("Players good at scoring")

#### **Option C: Hybrid Retrieval** (Default)
- Combines both baseline and semantic results
- Provides comprehensive coverage
- Merges structured and unstructured retrieval

---

### 3️⃣ **LLM Layer** (`llm_layer.py`)
**Input:** Retrieved context + user query

**Technical Implementation:**

#### **Step 1: Context Aggregation** (`build_retrieval_context`)
- **RetrievalContext Class:**
  ```python
  - user_query: str
  - intent: str  
  - entities: Dict[str, List]
  - baseline_results: List[Dict]  # From Cypher
  - embedding_results: List[Dict]  # From semantic search
  - baseline_desc: str
  ```
- **Merging Strategy:**
  - Deduplicates players across baseline and embeddings
  - Prioritizes baseline for exact matches
  - Adds embedding results for context enrichment
  - Formats as structured text blocks

#### **Step 2: Prompt Engineering** (`build_structured_prompt`)
- **Prompt Template Structure:**
  ```
  SYSTEM ROLE: You are an FPL expert assistant...
  
  USER QUERY: {user_query}
  
  DETECTED INTENT: {intent}
  ENTITIES: {formatted_entities}
  
  KNOWLEDGE GRAPH DATA:
  Baseline Results: {baseline_results}
  Semantic Results: {embedding_results}
  
  INSTRUCTIONS:
  - Answer based on provided data
  - Use FPL terminology
  - Be concise but informative
  - Include statistics to support claims
  ```
- **Token Management:**
  - Truncates context if >3000 tokens
  - Prioritizes baseline over embeddings if trimming needed

#### **Step 3: LLM API Call** (`OpenRouterAdapter`)
- **API Configuration:**
  - Endpoint: `https://openrouter.ai/api/v1/chat/completions`
  - Authentication: Bearer token from `config.txt`
  - Headers: `HTTP-Referer`, `X-Title` for OpenRouter tracking
- **Request Parameters:**
  ```python
  {
    "model": "mistralai/mistral-7b-instruct:free",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.7,  # Balanced creativity
    "max_tokens": 500    # Limit response length
  }
  ```
- **Error Handling:**
  - Retry logic (up to 2 retries)
  - Timeout after 30 seconds
  - Fallback error messages

#### **Step 4: Response Processing**
- **ModelAnswer Class:**
  - Extracts answer from JSON response
  - Cleans markdown formatting
  - Validates output
- **ModelMetrics Class:**
  - `response_time_sec`: Total API call duration
  - `input_tokens`: Prompt token count (from API)
  - `output_tokens`: Response token count (from API)
  - Used for performance comparison

**Supported Models (All Free Tier):**

| Model | Parameters | Speed | Quality | Best For |
|-------|-----------|-------|---------|----------|
| **Mistral-7B-Instruct** | 7B | ⚡⚡⚡ Fast | ⭐⭐⭐⭐ Good | General queries, balanced |
| **GPT-OSS-20B** | 20B | ⚡⚡⚡⚡ Very Fast | ⭐⭐⭐ Decent | Quick answers, simple questions |
| **Gemma-3-27B** | 27B | ⚡⚡ Slower | ⭐⭐⭐⭐⭐ Excellent | Complex analysis, detailed responses |

**Why Multiple Models?**
- Different strengths for different query types
- A/B testing capabilities
- Redundancy if one model is down
- Educational: compare LLM behaviors

---

### 4️⃣ **UI Layer** (`app.py`)
**Platform:** Streamlit web application (Python framework)

**Technical Architecture:**
- **Framework:** Streamlit 1.x
- **Rendering:** Server-side Python, reactive updates
- **State Management:** `st.session_state` for persistence
- **CSS Styling:** Inline markdown with custom CSS classes
- **Deployment:** Local (can deploy to Streamlit Cloud)

**Rendering Pipeline (Step-by-Step):**

1. **Initialize Configuration:**
   ```python
   load_configuration()  # Reads config.txt
   get_available_models()  # Loads API keys
   ```

2. **Render Header & Sidebar:**
   - Load base64-encoded player image
   - Build FPL-themed banner with gradient
   - Render sidebar with dropdown settings

3. **User Input Handling:**
   - `st.form()` for Enter key submission
   - Session state updates on pill clicks
   - Input validation (non-empty check)

4. **Query Processing (When Submit):**
   ```python
   progress_bar.progress(10)
   preprocessing_output = run_preprocessing(query)  # Stage 1
   
   progress_bar.progress(50)
   baseline_results = run_baseline_retrieval()      # Stage 2A
   embedding_results = run_embedding_retrieval()    # Stage 2B
   
   progress_bar.progress(85)
   prompt = build_structured_prompt(context)        # Stage 3A
   answer, metrics = call_llm(prompt)               # Stage 3B
   
   progress_bar.progress(100)
   ```

5. **Results Display (Priority Order):**
   - **First:** AI answer in styled card (most important)
   - **Second:** Collapsible expanders for:
     - Query Analysis (intent, entities, etc.)
     - Cypher Preview (generated query)
     - Retrieved Context (3 tabs: baseline, embeddings, recommendations)
     - Full Prompt (debugging view)

6. **Model Comparison Mode:**
   - Parallel columns (`st.columns(len(models))`)
   - Sequential API calls per model
   - Side-by-side answer display
   - Metrics comparison table

**UI Component Hierarchy:**
```
Streamlit App
├── Custom CSS (FPL theme)
├── Header (render_header)
├── Sidebar (render_sidebar)
│   ├── LLM Model Settings
│   ├── Retrieval Method
│   └── Display Options
├── Main Content
│   ├── Example Query Pills
│   ├── Input Form
│   ├── Progress Indicators
│   └── Results Display
│       ├── AI Answer (render_llm_response)
│       ├── Query Analysis (render_entity_badges)
│       ├── Cypher Preview (render_cypher_query)
│       ├── Context Tabs
│       │   ├── Baseline (render_baseline_results)
│       │   ├── Embeddings (render_embedding_results)
│       │   └── Recommendations (render_recommendations)
│       └── Full Prompt
└── Footer
```

**Key Technical Features:**
- **Reactivity:** Auto-reruns on user interaction
- **Caching:** `@st.cache_data` for config loading (disabled for fresh API keys)
- **Form Handling:** Prevents multiple submissions
- **Error Boundaries:** Try-catch blocks with user-friendly messages

---

## ✨ UI Features

### 🎨 **Design & Aesthetics**
- **FPL-Themed Color Palette:**
  - Purple (#37003c) - Primary brand color
  - Magenta (#963cff) - Accents
  - Green (#00ff87) - Highlights
  - Gradient backgrounds for visual appeal
- **Hero Banner** with FPL branding and player imagery
- **Responsive Layout** with fixed sidebar
- **Custom CSS** for professional appearance

### 🔍 **Query Input**
- **Main Input Box** with autocomplete placeholder
- **Example Query Pills** (5 pre-configured queries):
  - ⚽ Top Scorers
  - 🎯 Best Midfielders
  - 👑 About Salah
  - 🧤 Clean Sheets
  - ⚔️ Kane vs Haaland
- **Session State Management** - Persists queries across reruns
- **Form Submission** - Enter key triggers search

### ⚙️ **Sidebar Settings (Always Visible)**

#### 🤖 **LLM Model Selection**
- **Single Model Mode:**
  - Dropdown to select one model
  - Displays model description
- **Compare Models Mode:**
  - Multi-select for side-by-side comparison
  - Default: compares first 2 models
  - Shows responses in parallel columns

#### 🔍 **Retrieval Method**
- **Hybrid (Both)** - Default, uses baseline + embeddings
- **Baseline Only** - Pure Cypher query retrieval
- **Embeddings Only** - Pure semantic search

#### 🧠 **Embedding Model Selection**
- **BGE-M3** (Default):
  - Best context quality
  - 1024 dimensions
  - ~150ms query time
  - Recommended for accuracy
- **MiniLM** (Alternative):
  - Faster inference (~50ms)
  - 384 dimensions
  - Lower memory usage
  - Good for speed priority
- **Toggle in Sidebar** - User can switch between models
- **Trade-off Indicator** - Shows quality vs. speed

#### 🔧 **Display Options**
- Toggle visibility of:
  - 📝 Cypher Queries
  - 📄 Raw Context
- Controls what debug info appears in results

### 📊 **Results Display**

#### 🎯 **AI Answer** (Top Priority)
- Displayed first, prominent styling
- Markdown formatting support
- Collapsible performance metrics:
  - Response time (seconds)
  - Input tokens
  - Output tokens

#### 🔎 **Query Analysis** (Collapsible)
4-column breakdown:
- **Intent** - Detected query type (non-copiable badge)
- **Ranking** - Best/worst/top/etc.
- **Threshold** - Numeric limits
- **Entities** - Color-coded badges by type:
  - Player names
  - Team names
  - Positions
  - Statistics
  - Gameweeks

#### 📝 **Cypher Query Preview** (Collapsible)
- Shows generated Cypher query
- Syntax highlighting
- Helps users understand graph traversal

#### 📚 **Retrieved Context** (Collapsible with 3 Tabs)
1. **Baseline Results Tab:**
   - Table view of Cypher query results
   - Shows top 10 results
   - Displays all returned fields
   
2. **Embedding Results Tab:**
   - Semantic search results
   - Shows player, position, points, similarity score
   - Top 10 most relevant players

3. **Recommendations Tab:**
   - Top 5 player recommendations
   - Expandable cards per player
   - Shows position and key statistics

#### 💬 **Full LLM Prompt** (Collapsible)
- Black code-style background
- Shows exact prompt sent to LLM
- Useful for debugging and transparency

### 🔄 **Model Comparison Mode**
- **Side-by-side columns** (one per selected model)
- **Parallel generation** for fair comparison
- **Metrics comparison table:**
  - Response times
  - Token usage
  - Model names
- **Error handling** per model (shows which failed)

### 📈 **Progress Tracking**
- **Progress bar** (0-100%)
- **Status text** updates:
  - "Processing query..."
  - "Retrieving from Knowledge Graph..."
  - "Generating response with AI..."
  - "Complete!"
- Auto-hides after completion

### 🎯 **Interactive Elements**
- **Clickable example queries** - Auto-fill input
- **Expanders** - Collapse/expand sections
- **Tabs** - Organize context types
- **Hover effects** - Button animations
- **Form validation** - Warns if query empty

---

## ⚠️ UI Limitations

### 🔴 **Technical Limitations**

1. **Model Availability:**
   - Relies on OpenRouter API uptime
   - Free tier models only
   - No offline mode
   - API rate limiting possible

2. **Retrieval Constraints:**
   - Baseline requires well-formed entities
   - Embedding search limited to player-level data
   - No team-level or gameweek-level embeddings
   - Top-k fixed at 10 for embeddings

3. **Performance:**
   - Blocking UI during LLM generation
   - No streaming responses
   - Model comparison can be slow (sequential calls)
   - Progress bar doesn't show actual LLM progress

4. **Error Handling:**
   - Generic error messages for LLM failures
   - No retry mechanism
   - API key errors not user-friendly
   - Neo4j connection errors halt entire flow

### 🟡 **Functional Limitations**

5. **Query Scope:**
   - Limited to 2022-23 season data
   - No multi-season comparisons
   - No gameweek-specific filtering in UI
   - Can't specify date ranges

6. **Recommendations:**
   - No personalized recommendations
   - No team budget constraints
   - No formation validation
   - No injury/suspension data

7. **Comparison Mode:**
   - Max 3 models (UI constraint)
   - No customizable parameters per model
   - Can't save comparison results
   - No A/B testing metrics

8. **Context Display:**
   - Cypher preview is static (not actual executed query)
   - Can't export results
   - No history/bookmarks
   - Can't share queries via URL

### 🟢 **UX Limitations**

9. **Interactivity:**
   - Sidebar cannot be collapsed (forced open)
   - No dark mode toggle
   - No keyboard shortcuts
   - No voice input

10. **Responsiveness:**
    - Desktop-optimized only
    - Mobile layout not fully tested
    - Fixed 300px sidebar width
    - Banner may overflow on small screens

11. **Customization:**
    - No user preferences persistence
    - Can't customize top-k value in UI
    - Can't adjust similarity thresholds
    - No advanced query builder

12. **Data Visualization:**
    - Tables only (no charts/graphs)
    - No player comparison visualizations
    - No trend analysis over gameweeks
    - No formation diagrams

### 🔵 **Future Enhancement Gaps**

13. **Missing Features:**
    - No user authentication
    - No saved queries/favorites
    - No query history
    - No export to PDF/CSV
    - No multi-language support
    - No chat-like conversation history
    - No real-time data updates
    - No integration with official FPL API

---

## 🎤 Presentation Flow Recommendation

### **Slide 1: Introduction**
- Project title and objectives
- Problem: Making FPL decisions with complex data

### **Slide 2: System Architecture**
- High-level diagram showing 4 layers
- Data flow from query → preprocessing → retrieval → LLM → UI

### **Slide 3: Preprocessing Layer**
- Show example query transformation
- Highlight entity extraction and intent classification

### **Slide 4: Knowledge Graph**
- Neo4j structure (nodes: Player, Team, Position, Fixture, etc.)
- Relationships (PLAYED_IN, HAS_HOME_TEAM, etc.)

### **Slide 5: Retrieval Options**
- Compare baseline vs. embeddings vs. hybrid
- Use case for each

### **Slide 6: LLM Integration**
- Show prompt engineering approach
- Model options and comparison capability

### **Slide 7: UI Demo**
- Live demonstration or screenshots
- Highlight key features

### **Slide 8: Features Showcase**
- List 10-12 top features with icons
- Show example query flow

### **Slide 9: Limitations & Future Work**
- Be transparent about constraints
- Show roadmap for improvements

### **Slide 10: Conclusion**
- Impact and learnings
- Q&A

---

## 📸 Screenshot Recommendations

1. **Full UI Overview** - Show complete interface
2. **Query Analysis** - Expanded entities breakdown
3. **Model Comparison** - Side-by-side results
4. **Recommendations** - Player cards expanded
5. **Cypher Preview** - Graph query example
6. **Prompt View** - Full LLM prompt display

---

## 🎯 Key Talking Points

✅ **Strengths to Emphasize:**
- Hybrid retrieval combining structured + unstructured approaches
- Transparent AI with full prompt visibility
- Multiple LLM models for comparison
- FPL-specific domain expertise in prompting
- User-friendly interface with clear visual hierarchy

⚠️ **Honest About Limitations:**
- Single-season data scope
- Free-tier API dependencies
- Desktop-first design
- No real-time FPL integration

🚀 **Future Vision:**
- Multi-season historical analysis
- Live FPL API integration
- Personalized team recommendations
- Mobile app version
- Advanced data visualizations

---

**Good luck with your presentation! 🎉**
