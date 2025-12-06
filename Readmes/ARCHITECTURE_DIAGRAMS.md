# 📊 LLM Layer Architecture Diagrams

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                               │
│              "Show me top scoring forwards"                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────▼────────────┐
                │   FPLRAGSystem          │
                │   (Main Orchestrator)   │
                └────────────┬────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼──────┐    ┌───────▼──────┐    ┌───────▼──────┐
│   Baseline   │    │  Embedding   │    │     LLM      │
│  Retrieval   │    │   Search     │    │   Models     │
│  (Cypher)    │    │  (BGE-M3)    │    │ (3 models)   │
└───────┬──────┘    └───────┬──────┘    └───────┬──────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────▼────────┐
                    │ Result Combiner │
                    │  (Smart Merge)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Prompt Builder  │
                    │   (Structured)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Evaluator     │
                    │ (Metrics Track) │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │     REPORT      │
                    │  JSON + Summary │
                    └─────────────────┘
```

---

## Data Flow - Detailed

```
1. USER QUERY
   │
   ├─► "Show me top scoring forwards"
   │
   └─► Query Processing
       │
       ├─► Intent: player performance
       ├─► Entities: Position=FWD, Stat=goals
       └─► Season: current

2. BASELINE RETRIEVAL
   │
   ├─► Build Cypher Query
   │   MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
   │   WHERE p.position = 'FWD'
   │   RETURN p.name, SUM(r.goals_scored) as goals
   │   ORDER BY goals DESC
   │
   ├─► Execute on Neo4j
   │
   └─► Results:
       ├─► Erling Haaland: 36 goals
       ├─► Harry Kane: 30 goals
       └─► [8 more players...]

3. EMBEDDING SEARCH
   │
   ├─► Embed Query
   │   "top scoring forwards" → [1024-dim vector]
   │
   ├─► Similarity Search in Neo4j
   │   Compare with player embeddings
   │
   └─► Results:
       ├─► Erling Haaland (0.8543 similarity)
       ├─► Harry Kane (0.8321 similarity)
       └─► [3 more players...]

4. RESULT COMBINATION
   │
   ├─► Check Baseline Quality
   │   Is fallback? NO
   │   Has results? YES (10 players)
   │
   ├─► Check Embedding Quality
   │   Has results? YES (5 players)
   │
   ├─► Decision: Use baseline as PRIMARY
   │   ├─► Baseline: specific query, exact stat
   │   └─► Embedding: supplementary context
   │
   └─► Combined Context:
       PRIMARY SOURCE: Baseline
       ├─► 1. Haaland: 36 goals
       ├─► 2. Kane: 30 goals
       └─► [8 more...]
       
       SUPPLEMENTARY: Embeddings
       ├─► Haaland (0.8543)
       ├─► Kane (0.8321)
       └─► [3 more...]

5. PROMPT BUILDING
   │
   ├─► PERSONA
   │   "You are an expert FPL assistant..."
   │
   ├─► CONTEXT
   │   **Baseline Results:**
   │   1. Haaland: 36 goals
   │   2. Kane: 30 goals
   │   [...]
   │
   │   **Semantic Matches:**
   │   1. Haaland (0.8543)
   │   [...]
   │
   └─► TASK
       "Answer: 'Show me top scoring forwards'"
       [7 instructions to prevent hallucination]

6. LLM GENERATION (3 models in parallel)
   │
   ├─► Gemini 2.5 Flash
   │   ├─► Time: 1.2s
   │   ├─► Tokens: 387
   │   └─► Response: "Based on the data, Erling Haaland..."
   │
   ├─► Llama 3 8B
   │   ├─► Time: 2.3s
   │   ├─► Tokens: 412
   │   └─► Response: "The top scoring forwards are..."
   │
   └─► Mistral 7B
       ├─► Time: 1.8s
       ├─► Tokens: 395
       └─► Response: "According to the statistics..."

7. EVALUATION
   │
   ├─► Quantitative (Automatic)
   │   ├─► Response times: [1.2s, 2.3s, 1.8s]
   │   ├─► Token counts: [387, 412, 395]
   │   └─► Costs: [$0, $0, $0]
   │
   └─► Qualitative (Template Generated)
       For each response, score:
       ├─► Relevance (1-5): __
       ├─► Correctness (1-5): __
       ├─► Naturalness (1-5): __
       └─► Completeness (1-5): __

8. REPORT GENERATION
   │
   ├─► JSON Report
   │   {
   │     "timestamp": "2025-12-06...",
   │     "total_queries": 1,
   │     "model_statistics": {...},
   │     "all_results": [...]
   │   }
   │
   └─► Summary Report
       ============================
       QUANTITATIVE METRICS
       ============================
       Gemini: 1.2s avg, 387 tokens
       Llama: 2.3s avg, 412 tokens
       Mistral: 1.8s avg, 395 tokens
       
       ============================
       QUALITATIVE EVALUATION
       ============================
       [Side-by-side comparison]
```

---

## Result Combination Logic

```
┌─────────────────────┐
│ Baseline Results    │
│ + Description       │
└──────────┬──────────┘
           │
           ├─► Check: Is "Fallback" in description?
           │
           ├─► YES (Fallback detected)
           │   │
           │   ├─► Baseline couldn't find specific match
           │   ├─► Query was too vague or entity extraction failed
           │   │
           │   └─► DECISION:
           │       ├─► Primary Source = EMBEDDING
           │       └─► Embedding likely more relevant
           │
           └─► NO (Specific query)
               │
               ├─► Baseline found exact match
               ├─► Query had clear intent + entities
               │
               └─► DECISION:
                   ├─► Primary Source = BASELINE
                   ├─► Supplementary = EMBEDDING (top 3)
                   └─► Best of both worlds

Example 1: "Show me forwards with most goals"
  Baseline: ✅ Specific query → FWD + goals_scored
  Decision: PRIMARY = Baseline, SUPPLEMENTARY = Embeddings

Example 2: "Show me elite attacking players"
  Baseline: ⚠️ Vague query → Uses fallback (top 20 by points)
  Decision: PRIMARY = Embeddings (understands "elite attacking")

Example 3: "Who is the best player?"
  Baseline: ⚠️ Too vague → Fallback
  Embedding: ✅ Can match similar concepts
  Decision: PRIMARY = Embeddings
```

---

## Structured Prompt Anatomy

```
┌─────────────────────────────────────────────────────────────┐
│                         PROMPT                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ **PERSONA:**                                                 │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ You are an expert Fantasy Premier League assistant   │   │
│ │ with deep knowledge of player statistics, team       │   │
│ │ performance, and strategic advice.                   │   │
│ │                                                       │   │
│ │ Purpose: Establishes LLM's role and expertise       │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                              │
│ **CONTEXT:**                                                 │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ **Structured Query Results (Primary):**              │   │
│ │ Query Intent: player performance                     │   │
│ │ Query Description: Top players by goals for FWD      │   │
│ │                                                       │   │
│ │ Results:                                             │   │
│ │ 1. Erling Haaland, goals: 36                        │   │
│ │ 2. Harry Kane, goals: 30                            │   │
│ │ [8 more players...]                                  │   │
│ │                                                       │   │
│ │ **Supplementary Semantic Matches:**                  │   │
│ │ 1. Erling Haaland (FWD) - Score: 0.8543            │   │
│ │ 2. Harry Kane (FWD) - Score: 0.8321                │   │
│ │ [3 more players...]                                  │   │
│ │                                                       │   │
│ │ Purpose: All retrieved knowledge graph information   │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                              │
│ **TASK:**                                                    │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ Answer the user's question:                          │   │
│ │ "Show me the forwards with most goals"               │   │
│ │                                                       │   │
│ │ Instructions:                                        │   │
│ │ 1. Use ONLY the information in CONTEXT              │   │
│ │ 2. If context has relevant data, provide answer     │   │
│ │ 3. If insufficient, clearly state that              │   │
│ │ 4. Be concise but informative                       │   │
│ │ 5. Do NOT make up information                       │   │
│ │ 6. Use specific numbers from context                │   │
│ │ 7. Format naturally and conversationally            │   │
│ │                                                       │   │
│ │ Purpose: Clear question and anti-hallucination rules│   │
│ └──────────────────────────────────────────────────────┘   │
│                                                              │
│ Your answer:                                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Result: LLM generates grounded, accurate response
```

---

## Model Comparison Flow

```
Same Prompt → Multiple Models → Compare Results

┌─────────────────┐
│  Same Prompt    │
│  (Structured)   │
└────────┬────────┘
         │
         ├──────────────┬──────────────┬──────────────┐
         │              │              │              │
    ┌────▼─────┐   ┌───▼──────┐  ┌───▼──────┐      │
    │ Gemini   │   │  Llama   │  │ Mistral  │      │
    │ 2.5 Flash│   │  3 8B    │  │ 7B Inst  │      │
    └────┬─────┘   └───┬──────┘  └───┬──────┘      │
         │             │             │              │
    ┌────▼─────┐  ┌───▼──────┐  ┌───▼──────┐      │
    │Response A│  │Response B│  │Response C│      │
    │Time: 1.2s│  │Time: 2.3s│  │Time: 1.8s│      │
    │Tok: 387  │  │Tok: 412  │  │Tok: 395  │      │
    └────┬─────┘  └───┬──────┘  └───┬──────┘      │
         │            │             │              │
         └────────────┴─────────────┴──────────────┘
                      │
              ┌───────▼────────┐
              │   Evaluator    │
              │  • Compare     │
              │  • Measure     │
              │  • Report      │
              └───────┬────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
    ┌────▼────────┐      ┌────────▼─────┐
    │Quantitative │      │  Qualitative  │
    │  (Auto)     │      │   (Human)     │
    ├─────────────┤      ├───────────────┤
    │• Time       │      │• Relevance    │
    │• Tokens     │      │• Correctness  │
    │• Cost       │      │• Naturalness  │
    │• Errors     │      │• Completeness │
    └─────────────┘      └───────────────┘
```

---

## Error Handling & Fallback

```
User Query
    │
    ├─► Try Baseline
    │   │
    │   ├─► Success? → Return results
    │   │
    │   └─► Error/Fallback?
    │       │
    │       ├─► Log: "Baseline used fallback"
    │       └─► Flag: baseline_is_fallback = True
    │
    ├─► Try Embedding
    │   │
    │   ├─► Success? → Return results
    │   │
    │   └─► Error?
    │       │
    │       ├─► Log: "Embedding search failed"
    │       └─► Return empty list
    │
    ├─► Combine Results
    │   │
    │   ├─► If baseline_is_fallback AND embedding_has_results
    │   │   └─► Primary = Embedding (more reliable)
    │   │
    │   ├─► If baseline_has_results AND NOT fallback
    │   │   └─► Primary = Baseline (specific match)
    │   │
    │   └─► If neither has results
    │       └─► Primary = "none", combined_data = []
    │
    ├─► Build Prompt
    │   │
    │   ├─► If no results:
    │   │   └─► Context: "No results found..."
    │   │
    │   └─► If has results:
    │       └─► Context: [formatted results]
    │
    └─► LLM Generation
        │
        ├─► For each model:
        │   │
        │   ├─► Try generate()
        │   │
        │   ├─► Success?
        │   │   └─► Return {response, tokens, time, cost}
        │   │
        │   └─► Error?
        │       └─► Return {error: msg, time, tokens: 0}
        │
        └─► Continue with all models regardless of individual failures
```

---

## Performance Optimization

```
Parallel Processing:
┌─────────────┐
│   Query     │
└──────┬──────┘
       │
   ┌───┴───┐
   │ FORK  │ (Parallel)
   └───┬───┘
       │
   ┌───┼────────┐
   │   │        │
   ▼   ▼        ▼
 Base Embed   LLMs
   │   │        │
   │   │    ┌───┼───┐
   │   │    │   │   │
   │   │    ▼   ▼   ▼
   │   │    G   L   M
   │   │    │   │   │
   └───┼────┴───┴───┘
       │
    ┌──▼──┐
    │JOIN │
    └──┬──┘
       │
       ▼
    Report

Total Time ≈ Max(baseline, embedding, max(LLMs))
Not: baseline + embedding + LLM1 + LLM2 + LLM3

For typical query:
  Baseline: 0.5s
  Embedding: 0.8s
  Gemini: 1.2s
  Llama: 2.3s
  Mistral: 1.8s

Sequential: 0.5 + 0.8 + 1.2 + 2.3 + 1.8 = 6.6s
Parallel: Max(0.5, 0.8, 2.3) ≈ 3.0s
Speedup: 2.2x faster!
```

---

## Memory & Storage

```
Component               Memory Usage        Storage
────────────────────────────────────────────────────
Baseline Retriever      ~50 MB             0 MB
Embedding Search        ~2 GB (model)      Variable (Neo4j)
BGE-M3 Model           ~2 GB              ~1.5 GB (cached)
Gemini Model           API call           0 MB
Llama Model            API call           0 MB  
Mistral Model          API call           0 MB

Total System Memory:    ~2.1 GB
Total Storage:          ~1.5 GB (model cache)

Per-Query Memory:       ~10-50 MB (temp results)
Per-Query Storage:      ~1-5 KB (JSON report)
```

---

## Complete Example Trace

```
INPUT: "Who scored the most goals this season?"

1. PREPROCESSING (in baseline_retrieval)
   Intent: "player performance"
   Entities: {Statistic: ["goals_scored"], Season: ["2022-23"]}

2. BASELINE RETRIEVAL
   Query Type: Top players by statistic
   Cypher: MATCH (p:Player)-[r:PLAYED_IN]->...
          WHERE season = "2022-23"
          WITH p, SUM(r.goals_scored) as total
          RETURN p.player_name, total
          ORDER BY total DESC LIMIT 20
   
   Results: [
     {"player": "Erling Haaland", "total": 36},
     {"player": "Harry Kane", "total": 30},
     ...
   ]
   
   Is Fallback: NO (specific query matched)

3. EMBEDDING SEARCH
   Query Vector: embed("Who scored the most goals this season?")
   Top 5 Similar Players:
     1. Haaland (0.8543)
     2. Kane (0.8321)
     3. Salah (0.8198)
     4. Rashford (0.8034)
     5. Watkins (0.7912)

4. RESULT COMBINATION
   Baseline: ✅ Has results, NOT fallback
   Embedding: ✅ Has results
   
   Decision: PRIMARY = Baseline
             SUPPLEMENTARY = Embedding (top 3)
   
   Combined Data:
     PRIMARY: 20 players from baseline (with exact goal counts)
     SUPPLEMENTARY: 3 players from embedding (with relevance scores)

5. PROMPT CONSTRUCTION
   [Full structured prompt as shown earlier]

6. LLM GENERATION

   → Gemini 2.5 Flash:
     Start: 10:30:45.123
     Response: "Based on the data provided, Erling Haaland scored 
                the most goals this season with 36 goals, followed 
                by Harry Kane with 30 goals."
     End: 10:30:46.356
     Time: 1.233s
     Tokens: {prompt: 245, completion: 42, total: 287}
   
   → Llama 3 8B:
     Start: 10:30:45.124
     Response: "According to the statistics, the top scorer this 
                season is Erling Haaland with 36 goals. Harry Kane 
                is second with 30 goals."
     End: 10:30:47.489
     Time: 2.365s
     Tokens: {prompt: 245, completion: 38, total: 283}
   
   → Mistral 7B:
     Start: 10:30:45.125
     Response: "The player who scored the most goals this season 
                is Erling Haaland with a total of 36 goals."
     End: 10:30:46.934
     Time: 1.809s
     Tokens: {prompt: 245, completion: 28, total: 273}

7. EVALUATION
   
   Quantitative:
     Gemini:  1.233s, 287 tokens, $0.000
     Llama:   2.365s, 283 tokens, $0.000
     Mistral: 1.809s, 273 tokens, $0.000
   
   Qualitative Template Generated:
     [Side-by-side for human scoring]

8. REPORT
   
   JSON: Full data saved
   Summary: Human-readable comparison
   Console: Real-time output shown

OUTPUT: Complete comparison of 3 models answering the query
```

---

These diagrams visualize the complete LLM Layer implementation! 🎨
