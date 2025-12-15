# LLM Test Query Suite Documentation

## Overview

This test suite compares 3 LLMs on the FPL Knowledge Graph RAG system. 

**Key principle**: All 3 LLMs receive the **same data** (from baseline + embedding) and the **same prompt**. You evaluate how accurately each LLM interprets and responds based on the provided context.

### Flow:
```
Query → Your Pipeline (baseline + embedding) → Context Data
                                                    ↓
                                            Same Prompt → LLM1 → Response 1
                                                       → LLM2 → Response 2  
                                                       → LLM3 → Response 3
```

### Output:
For each query, you get:
- `context_data`: The baseline + embedding results provided to LLMs
- `prompt_sent`: The exact prompt all 3 LLMs received
- `llm_responses`: All 3 responses with timing info

You manually compare each response to the context_data to evaluate accuracy.

---

## Test Queries

### BASE QUERIES (10) - For Accuracy Benchmarking

| ID | Difficulty | Query |
|----|------------|-------|
| BASE_01 | Easy | Who was the top scorer in the 2022-23 Premier League season? |
| BASE_02 | Easy | Which goalkeeper had the most saves in the 2022-23 season? |
| BASE_03 | Easy | How many total points did Harry Kane score in the 2022-23 season? |
| BASE_04 | Medium | Who had the most assists in the 2022-23 season and how many did they have? |
| BASE_05 | Medium | List the top 5 players by total FPL points in the 2022-23 season. |
| BASE_06 | Medium | Which teams were promoted to the Premier League for the 2022-23 season? |
| BASE_07 | Medium | How many players received a red card in the 2022-23 season? |
| BASE_08 | Hard | Which player scored the most goals in a single gameweek during the 2022-23 season? |
| BASE_09 | Hard | Compare Erling Haaland and Harry Kane's total goals in the 2022-23 season. Who scored more and by how many? |
| BASE_10 | Hard | Which defender had the most clean sheets in the 2021-22 season? |

### EDGE CASE QUERIES (18) - For Error Analysis

| ID | Category | Query | Expected Issue |
|----|----------|-------|----------------|
| EDGE_01 | ambiguous | Who is the best player? | No season/metric specified |
| EDGE_02 | ambiguous | How did Salah do? | No season/metric, informal |
| EDGE_03 | ambiguous | Who scored more, Kane or the Egyptian? | Indirect reference |
| EDGE_04 | out_of_scope | What was Erling Haaland's salary in 2022-23? | No salary data |
| EDGE_05 | out_of_scope | Who will be the top scorer in the 2024-25 season? | Future prediction |
| EDGE_06 | out_of_scope | How many Champions League goals did Liverpool score? | Wrong competition |
| EDGE_07 | complex_reasoning | Which player improved the most between 2021-22 and 2022-23? | Cross-season comparison |
| EDGE_08 | complex_reasoning | Average points per game for forwards with 10+ goals in 2022-23? | Multi-step calculation |
| EDGE_09 | complex_reasoning | Combined points for Haaland, Salah, De Bruyne in 2022-23? | Hypothetical sum |
| EDGE_10 | entity_resolution | How many goals did Bruno score in 2022-23? | Multiple Brunos |
| EDGE_11 | entity_resolution | What are Man City's total goals in 2022-23? | Team aggregation |
| EDGE_12 | temporal | Who scored the most goals in December 2022? | Date filtering |
| EDGE_13 | temporal | Which newly promoted team performed best in 2022-23? | Domain knowledge |
| EDGE_14 | nonsensical | Square root of Salah's assists divided by blue? | Meaningless |
| EDGE_15 | nonsensical | Rank players by zodiac sign compatibility? | Invalid concept |
| EDGE_16 | scale | List every player appearance with all stats? | Too many rows |
| EDGE_17 | misspelling | How many goals did Hallend score? | Typo: Haaland |
| EDGE_18 | nickname | What was CR7's performance in 2021-22? | Nickname resolution |

---

## Usage

### Run Full Evaluation
```bash
python run_llm_comparison.py
```

This will:
1. Run all 10 base queries + 18 edge cases through your pipeline
2. Query all 3 LLMs with the same context data
3. Save results to `llm_test_results.json`

### Run Single Query Test
```bash
python run_llm_comparison.py --single
```

### Output JSON Structure
```json
{
  "query_id": "BASE_01",
  "query_text": "Who was the top scorer...",
  "context_data": {
    "intent": "...",
    "entities": {...},
    "baseline_results": [...],
    "embedding_results": [...]
  },
  "prompt_sent": "CONTEXT BEGIN...",
  "llm_responses": {
    "Model1": {"response": "...", "response_time_sec": 2.5},
    "Model2": {"response": "...", "response_time_sec": 3.1},
    "Model3": {"response": "...", "response_time_sec": 2.8}
  }
}
```

### Manual Evaluation
For each query result:
1. Look at `context_data` - this is what the LLMs were given
2. Look at each LLM's `response`
3. Score how accurately each LLM interpreted the provided data

---

## Files

| File | Description |
|------|-------------|
| `llm_test_queries.py` | Test query definitions (just questions) |
| `run_llm_comparison.py` | Runs queries through pipeline, collects 3 LLM answers |
| `llm_test_results.json` | Output file with all results for evaluation |
| `TEST_QUERIES_README.md` | This documentation |
