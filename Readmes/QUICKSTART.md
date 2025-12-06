# 🚀 Quick Start Guide - LLM Layer

This guide will get you up and running with the LLM Layer in 5 minutes.

---

## 📋 Prerequisites

✅ You already have:
- Neo4j database with FPL data
- `baseline_retrieval.py` working
- `embedding_bge_m3.py` working
- Python packages: neo4j, sentence-transformers

✅ You need to install:
```powershell
pip install google-generativeai huggingface_hub
```

---

## 🔑 Step 1: Get API Keys (2 minutes)

### Gemini API Key (Google)
1. Go to: https://aistudio.google.com/app/apikey
2. Click "Get API Key" or "Create API Key"
3. Copy the key (starts with "AI...")

### HuggingFace Token
1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: "LLM Testing"
4. Type: "Read"
5. Copy the token (starts with "hf_...")

---

## ⚙️ Step 2: Set Environment Variables (1 minute)

Open PowerShell and run:

```powershell
# Set Gemini API key
$env:GEMINI_API_KEY = "YOUR-GEMINI-KEY-HERE"

# Set HuggingFace token
$env:HUGGINGFACE_API_KEY = "YOUR-HF-TOKEN-HERE"
```

**Note:** These are temporary. To make them permanent, add to your system environment variables.

---

## 🎯 Step 3: Run the Test Suite (2 minutes)

```powershell
cd C:\Users\mansa\OneDrive\Desktop\ACL-MS3
python llm_layer.py
```

This will:
- Test 3 models on 5 queries
- Show real-time results
- Generate comparison reports
- Create evaluation template

---

## 📊 Step 4: Check Results

After running, you'll have:

1. **Console Output** - Real-time progress and summaries
2. **llm_comparison_report.json** - Complete data
3. **llm_comparison_report_summary.txt** - Human evaluation template

---

## 🔍 Alternative: Run Example Script

For a more guided experience:

```powershell
python example_llm_usage.py
```

This shows:
- Basic query usage
- Position filtering
- Multiple queries
- Report generation

---

## 🎓 Step 5: Evaluate Models

1. Open `llm_comparison_report_summary.txt`
2. Read each model's response
3. Fill in the scores (1-5) for:
   - Relevance
   - Correctness
   - Naturalness
   - Completeness
4. Compare quantitative metrics (time, tokens)

---

## 🧪 Test Your Own Queries

### Option A: Modify the test queries

Edit `llm_layer.py` line 828:

```python
test_queries = [
    "Your custom query here",
    "Another query",
]
```

### Option B: Use in your own code

```python
from llm_layer import FPLRAGSystem

# Initialize
rag = FPLRAGSystem(
    gemini_api_key="your-key",
    huggingface_api_key="your-key"
)

# Query
result = rag.query("Show me top scoring midfielders")

# See responses
for eval_result in result['llm_evaluations']:
    print(f"{eval_result['model']}: {eval_result['response']}")

# Cleanup
rag.close()
```

---

## 🐛 Troubleshooting

### "ImportError: No module named 'google.generativeai'"
```powershell
pip install google-generativeai
```

### "ImportError: No module named 'huggingface_hub'"
```powershell
pip install huggingface_hub
```

### "API key not found"
Make sure you set the environment variables in the same PowerShell session where you run the script.

### "Gemini API error: 429"
You've hit the free tier rate limit (1500 requests/day). Wait a bit or use only HuggingFace models.

### "HuggingFace API error: Model is loading"
Some models take 20-30 seconds to load on first request. Wait and try again.

### "baseline_retrieval import error"
Make sure you're in the correct directory:
```powershell
cd C:\Users\mansa\OneDrive\Desktop\ACL-MS3
```

---

## 📝 What Each File Does

| File | Purpose |
|------|---------|
| `llm_layer.py` | Main implementation (run this for testing) |
| `example_llm_usage.py` | Usage examples and patterns |
| `LLM_LAYER_README.md` | Comprehensive documentation |
| `MILESTONE3_SUMMARY.md` | Requirements satisfaction explanation |
| `QUICKSTART.md` | This file - quick setup guide |

---

## 🎯 Expected First Run

```
====================================================================
FPL RAG SYSTEM - LLM Layer (Milestone 3)
====================================================================

🔧 Initializing RAG system...
✅ RAG System initialized with 3 models

📝 Testing with 5 queries...
====================================================================

[Query 1/5]
======================================================================
Processing query: Show me the forwards with the most goals scored...
======================================================================

[1/4] Running baseline retrieval...
  Intent: player performance
  Results: 10 rows
  Description: Top players by goals_scored for position FWD...

[2/4] Running embedding search...
  Results: 5 semantic matches

[3/4] Combining results...
  Primary source: baseline

[4/4] Generating LLM responses...

Testing Gemini 2.5 Flash...
Response: Based on the provided data, the forwards with the most...
Time: 1.234s
Tokens: 387
Cost: $0.000000

Testing Llama 3 8B...
[similar output]

Testing Mistral 7B Instruct...
[similar output]

✅ Query completed

[Continues for remaining queries...]

📊 Generating comparison report...
✅ Saved detailed report to llm_comparison_report.json
✅ Saved human evaluation template to llm_comparison_report_summary.txt

✅ Evaluation complete!
```

---

## ✅ Success Checklist

After running, verify:

- [ ] 3 models initialized (Gemini, Llama, Mistral)
- [ ] 5 queries processed successfully
- [ ] Console shows responses from all models
- [ ] `llm_comparison_report.json` exists
- [ ] `llm_comparison_report_summary.txt` exists
- [ ] No errors in console output

---

## 🎉 Next Steps

1. ✅ Test with default queries
2. ✅ Review generated reports
3. ✅ Fill in qualitative scores
4. ✅ Test with your own queries
5. ✅ Compare model performance
6. ✅ Choose best model for your needs

---

## 💡 Pro Tips

1. **Start with Gemini only** (faster, fewer API calls):
   - Only set `GEMINI_API_KEY`
   - Skip HuggingFace key

2. **Use fewer queries for testing**:
   - Edit `test_queries` list in `llm_layer.py`
   - Reduce from 5 to 2-3 queries

3. **Check your baseline quality**:
   - If baseline uses fallback often, embeddings will be prioritized
   - This is good! It means the system is working intelligently

4. **Free tier limits**:
   - Gemini: 1500 requests/day
   - HuggingFace: Generous free tier
   - You can test extensively without cost

---

## 📚 Learn More

- **Full documentation:** Read `LLM_LAYER_README.md`
- **Requirements explanation:** Read `MILESTONE3_SUMMARY.md`
- **Code examples:** Check `example_llm_usage.py`
- **API docs:** 
  - Gemini: https://ai.google.dev/docs
  - HuggingFace: https://huggingface.co/docs/huggingface_hub

---

## 🆘 Need Help?

Common issues and solutions are in the Troubleshooting section above.

For detailed architecture and design decisions, see `LLM_LAYER_README.md`.

---

**Ready? Let's go! 🚀**

```powershell
python llm_layer.py
```
