"""
example_llm_usage.py

Example script demonstrating how to use the LLM Layer for FPL queries.

This shows:
1. Basic query processing
2. Accessing different result components
3. Comparing model outputs
4. Generating evaluation reports
"""

import os
from llm_layer import FPLRAGSystem

def main():
    print("="*70)
    print("FPL RAG System - Example Usage")
    print("="*70)
    
    # Option 1: Get API keys from environment variables
    gemini_key = os.getenv("GEMINI_API_KEY")
    hf_key = os.getenv("HUGGINGFACE_API_KEY")
    
    # Option 2: Set API keys directly (for testing)
    # gemini_key = "your-gemini-api-key-here"
    # hf_key = "your-huggingface-api-key-here"
    
    if not gemini_key:
        print("\n⚠️  Set GEMINI_API_KEY environment variable or edit this script")
        gemini_key = input("Enter Gemini API key (or press Enter to skip): ").strip() or None
    
    if not hf_key:
        print("\n⚠️  Set HUGGINGFACE_API_KEY environment variable or edit this script")
        hf_key = input("Enter HuggingFace API key (or press Enter to skip): ").strip() or None
    
    if not gemini_key and not hf_key:
        print("\n❌ At least one API key is required")
        return
    
    # Initialize the RAG system
    print("\n🔧 Initializing RAG system...")
    rag = FPLRAGSystem(
        gemini_api_key=gemini_key,
        huggingface_api_key=hf_key
    )
    
    # Example 1: Simple query
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple Query")
    print("="*70)
    
    query1 = "Which forwards scored the most goals last season?"
    print(f"\nQuery: {query1}")
    
    result1 = rag.query(query1)
    
    # Access different components
    print(f"\nBaseline Intent: {result1['baseline_results']['intent']}")
    print(f"Baseline Results: {len(result1['baseline_results']['results'])} rows")
    print(f"Embedding Results: {len(result1['embedding_results'])} matches")
    print(f"Primary Source: {result1['combined_results']['primary_source']}")
    
    # See model responses
    print("\nModel Responses:")
    for eval_result in result1['llm_evaluations']:
        print(f"\n{eval_result['model']}:")
        print(f"  Response: {eval_result['response'][:150]}...")
        print(f"  Time: {eval_result['time']:.3f}s")
        print(f"  Tokens: {eval_result['tokens']['total']}")
    
    # Example 2: Query with position filter
    print("\n" + "="*70)
    print("EXAMPLE 2: Query with Position Filter")
    print("="*70)
    
    query2 = "Show me elite defenders with many clean sheets"
    print(f"\nQuery: {query2}")
    print("Position Filter: DEF")
    
    result2 = rag.query(query2, position_filter="DEF")
    
    print(f"\nEmbedding Results (filtered to DEF):")
    for i, res in enumerate(result2['embedding_results'][:3], 1):
        print(f"  {i}. {res['player']} ({res['position']}) - Score: {res['score']:.4f}")
    
    # Example 3: Multiple queries for comparison
    print("\n" + "="*70)
    print("EXAMPLE 3: Multiple Queries for Model Comparison")
    print("="*70)
    
    test_queries = [
        "Who are the top 3 midfielders by total points?",
        "Which goalkeepers made the most saves?",
        "Show me players in excellent form"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] {query}")
        result = rag.query(query)
        print(f"  ✅ Processed by {len(result['llm_evaluations'])} models")
    
    # Generate comparison report
    print("\n" + "="*70)
    print("GENERATING COMPARISON REPORT")
    print("="*70)
    
    rag.generate_report("example_comparison_report.json")
    
    print("\n✅ Reports generated:")
    print("  📄 example_comparison_report.json")
    print("  📄 example_comparison_report_summary.txt")
    
    # Cleanup
    rag.close()
    
    print("\n" + "="*70)
    print("EXAMPLE COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Check the generated reports")
    print("2. Fill in the qualitative evaluation template")
    print("3. Compare model performance")
    print("4. Adjust queries or filters as needed")

if __name__ == "__main__":
    main()
