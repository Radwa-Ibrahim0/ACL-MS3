"""
llm_layer.py

LLM Layer for FPL Knowledge Graph RAG System (Milestone 3)

This module implements:
  a. Combine KG results from baseline and embeddings
  b. Use structured prompt: context, persona, task
  c. Compare three models: Gemini 2.5 Flash, Llama 3 8B, Mistral 7B
  d. Quantitative and qualitative evaluation

Models tested:
  - Model A: Gemini 2.5 Flash (Google)
  - Model B: Llama 3 8B (HuggingFace)
  - Model C: Mistral 7B Instruct (HuggingFace)

Metrics:
  - Quantitative: accuracy, response time, token usage, cost
  - Qualitative: answer quality, relevance, naturalness, correctness
"""

import sys
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import os

from jax import config

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)

# Import our existing modules
try:
    from baseline_retrieval import BaselineRetriever
    from embedding_bge_m3 import SemanticSearchBGEM3, load_config
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    MODULES_AVAILABLE = False

# LLM libraries
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed. Run: pip install google-generativeai")

try:
    from huggingface_hub import InferenceClient
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    logger.warning("huggingface_hub not installed. Run: pip install huggingface_hub")

# ============================================================
# 1. CONFIGURATION
# ============================================================

def load_config() -> Dict[str, str]:
    config: Dict[str, str] = {}
    try:
        with open("config.txt", "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    config[k.strip()] = v.strip()
    except FileNotFoundError:
        print("⚠ config.txt not found. Using empty config.")
    return config

CONFIG = load_config()


# ============================================================
# 1. RESULT COMBINATION - Merge baseline and embedding results
# ============================================================

class ResultCombiner:
    """
    Combines results from baseline (Cypher queries) and embedding-based retrieval.
    
    This provides both structured (exact matches) and semantic (meaning-based) information.
    """
    
    @staticmethod
    def combine_results(
        baseline_results: Dict[str, Any],
        embedding_results: List[Dict[str, Any]],
        query: str
    ) -> Dict[str, Any]:
        """
        Merge baseline and embedding results into unified context.
        
        Args:
            baseline_results: Results from baseline_retrieval.py
            embedding_results: Results from embedding_bge_m3.py semantic search
            query: Original user query
            
        Returns:
            Combined context with structured and semantic information
        """
        combined = {
            "query": query,
            "baseline": {
                "intent": baseline_results.get("intent", "unknown"),
                "description": baseline_results.get("description", ""),
                "results": baseline_results.get("results", []),
                "is_fallback": "Fallback" in baseline_results.get("description", "")
            },
            "embedding": {
                "results": embedding_results,
                "count": len(embedding_results)
            },
            "combined_data": []
        }
        
        # Determine which source is more reliable
        baseline_is_fallback = combined["baseline"]["is_fallback"]
        baseline_has_results = len(combined["baseline"]["results"]) > 0
        embedding_has_results = len(embedding_results) > 0
        
        # Priority logic: 
        # - If baseline is fallback, prefer embedding results
        # - If baseline has specific results, prefer baseline but include embedding for context
        # - If both have results, merge them
        
        if baseline_is_fallback and embedding_has_results:
            logger.info("Baseline used fallback - prioritizing embedding results")
            combined["primary_source"] = "embedding"
            combined["combined_data"] = ResultCombiner._format_embedding_results(embedding_results)
        elif baseline_has_results and not baseline_is_fallback:
            logger.info("Baseline has specific results - using baseline as primary")
            combined["primary_source"] = "baseline"
            combined["combined_data"] = ResultCombiner._format_baseline_results(
                baseline_results.get("results", [])
            )
            # Add embedding as supplementary context if available
            if embedding_has_results:
                combined["supplementary_data"] = ResultCombiner._format_embedding_results(
                    embedding_results[:3]  # Top 3 semantic matches
                )
        elif embedding_has_results:
            logger.info("Only embedding has results")
            combined["primary_source"] = "embedding"
            combined["combined_data"] = ResultCombiner._format_embedding_results(embedding_results)
        elif baseline_has_results:
            logger.info("Only baseline has results")
            combined["primary_source"] = "baseline"
            combined["combined_data"] = ResultCombiner._format_baseline_results(
                baseline_results.get("results", [])
            )
        else:
            logger.warning("No results from either source")
            combined["primary_source"] = "none"
            combined["combined_data"] = []
        
        return combined
    
    @staticmethod
    def _format_baseline_results(results: List[Dict[str, Any]]) -> List[str]:
        """Format baseline results into readable strings."""
        formatted = []
        for i, row in enumerate(results[:10], 1):  # Limit to top 10
            # Convert result dict to readable format
            parts = []
            for key, value in row.items():
                if value is not None:
                    parts.append(f"{key}: {value}")
            formatted.append(f"{i}. " + ", ".join(parts))
        return formatted
    
    @staticmethod
    def _format_embedding_results(results: List[Dict[str, Any]]) -> List[str]:
        """Format embedding results into readable strings."""
        formatted = []
        for i, result in enumerate(results[:10], 1):  # Limit to top 10
            player = result.get("player", "Unknown")
            position = result.get("position", "N/A")
            score = result.get("score", 0.0)
            formatted.append(
                f"{i}. {player} ({position}) - Relevance Score: {score:.4f}"
            )
        return formatted


# ============================================================
# 2. STRUCTURED PROMPT BUILDER - Context, Persona, Task
# ============================================================

class PromptBuilder:
    """
    Builds structured prompts with three components:
    - Context: Retrieved KG information
    - Persona: Assistant's role definition
    - Task: Clear instructions for the LLM
    """
    
    # Persona definitions for FPL domain
    PERSONA = """You are an expert Fantasy Premier League (FPL) assistant with deep knowledge of player statistics, team performance, and strategic advice. You provide accurate, data-driven insights based on the knowledge graph information provided."""
    
    @staticmethod
    def build_prompt(combined_results: Dict[str, Any]) -> str:
        """
        Build structured prompt with Context, Persona, and Task.
        
        Args:
            combined_results: Combined results from ResultCombiner
            
        Returns:
            Structured prompt string
        """
        query = combined_results["query"]
        primary_source = combined_results.get("primary_source", "none")
        combined_data = combined_results.get("combined_data", [])
        supplementary_data = combined_results.get("supplementary_data", [])
        baseline_info = combined_results.get("baseline", {})
        
        # Build context section
        context_parts = []
        
        if primary_source == "baseline":
            context_parts.append("**Structured Query Results (Primary):**")
            context_parts.append(f"Query Intent: {baseline_info.get('intent', 'unknown')}")
            context_parts.append(f"Query Description: {baseline_info.get('description', '')}")
            if combined_data:
                context_parts.append("\nResults:")
                context_parts.extend(combined_data)
            else:
                context_parts.append("No specific results found.")
            
            if supplementary_data:
                context_parts.append("\n**Supplementary Semantic Matches:**")
                context_parts.extend(supplementary_data)
        
        elif primary_source == "embedding":
            context_parts.append("**Semantic Search Results (Primary):**")
            if baseline_info.get("is_fallback"):
                context_parts.append("Note: Structured query did not find specific matches, using semantic search.")
            if combined_data:
                context_parts.append("\nRelevant Players:")
                context_parts.extend(combined_data)
            else:
                context_parts.append("No semantic matches found.")
        
        else:
            context_parts.append("**No Results Found:**")
            context_parts.append("The query did not return any results from the knowledge graph.")
        
        context = "\n".join(context_parts)
        
        # Build full structured prompt
        prompt = f"""**PERSONA:**
{PromptBuilder.PERSONA}

**CONTEXT:**
{context}

**TASK:**
Answer the user's question: "{query}"

Instructions:
1. Use ONLY the information provided in the CONTEXT section above
2. If the context contains relevant data, provide a clear, accurate answer
3. If the context is insufficient or empty, clearly state that you don't have enough information
4. Be concise but informative
5. Do not make up or hallucinate information not present in the context
6. If comparing players or teams, use the specific numbers from the context
7. Format your answer in a natural, conversational way

Your answer:"""
        
        return prompt


# ============================================================
# 3. LLM MODEL INTERFACES - Gemini, Llama, Mistral
# ============================================================

class LLMModel:
    """Base class for LLM models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def generate(self, prompt: str) -> Dict[str, Any]:
        """
        Generate response from LLM.
        
        Returns:
            Dict with 'response', 'tokens', 'time', 'cost'
        """
        raise NotImplementedError


class GeminiModel(LLMModel):
    """Google Gemini 2.5 Flash model."""
    
    def __init__(self, api_key: str):
        super().__init__("Gemini 2.5 Flash")
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        logger.info(f"Initialized {self.model_name}")
    
    def generate(self, prompt: str) -> Dict[str, Any]:
        """Generate response from Gemini."""
        start_time = time.time()
        
        try:
            response = self.model.generate_content(prompt)
            elapsed_time = time.time() - start_time
            
            # Extract response text
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            # Token counting (approximate for Gemini)
            # Gemini doesn't provide direct token counts in free tier
            prompt_tokens = len(prompt.split()) * 1.3  # Rough estimate
            completion_tokens = len(response_text.split()) * 1.3
            total_tokens = prompt_tokens + completion_tokens
            
            # Cost estimation for Gemini 2.0 Flash
            # Free tier: 1500 requests per day, 1M tokens per minute
            # Paid tier (if used): very low cost
            cost = 0.0  # Free tier
            
            return {
                "response": response_text,
                "tokens": {
                    "prompt": int(prompt_tokens),
                    "completion": int(completion_tokens),
                    "total": int(total_tokens)
                },
                "time": elapsed_time,
                "cost": cost,
                "model": self.model_name
            }
        
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return {
                "response": f"Error: {str(e)}",
                "tokens": {"prompt": 0, "completion": 0, "total": 0},
                "time": time.time() - start_time,
                "cost": 0.0,
                "model": self.model_name,
                "error": str(e)
            }


class LlamaModel(LLMModel):
    """Llama 3 8B from HuggingFace."""
    
    def __init__(self, api_key: str):
        super().__init__("Llama 3 8B")
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("huggingface_hub not installed")
        
        self.client = InferenceClient(api_key=api_key)
        self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        logger.info(f"Initialized {self.model_name}")
    
    def generate(self, prompt: str) -> Dict[str, Any]:
        """Generate response from Llama."""
        start_time = time.time()
        
        try:
            messages = [{"role": "user", "content": prompt}]
            
            response = self.client.chat_completion(
                messages=messages,
                model=self.model_id,
                max_tokens=500,
                temperature=0.7
            )
            
            elapsed_time = time.time() - start_time
            
            # Extract response
            response_text = response.choices[0].message.content
            
            # Token counting (HuggingFace provides usage stats)
            usage = response.usage if hasattr(response, 'usage') else None
            if usage:
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
                total_tokens = usage.total_tokens
            else:
                # Fallback estimation
                prompt_tokens = len(prompt.split()) * 1.3
                completion_tokens = len(response_text.split()) * 1.3
                total_tokens = prompt_tokens + completion_tokens
            
            # Cost: HuggingFace Inference API has free tier
            cost = 0.0  # Free tier
            
            return {
                "response": response_text,
                "tokens": {
                    "prompt": int(prompt_tokens),
                    "completion": int(completion_tokens),
                    "total": int(total_tokens)
                },
                "time": elapsed_time,
                "cost": cost,
                "model": self.model_name
            }
        
        except Exception as e:
            logger.error(f"Llama generation error: {e}")
            return {
                "response": f"Error: {str(e)}",
                "tokens": {"prompt": 0, "completion": 0, "total": 0},
                "time": time.time() - start_time,
                "cost": 0.0,
                "model": self.model_name,
                "error": str(e)
            }


class MistralModel(LLMModel):
    """Mistral 7B Instruct from HuggingFace."""
    
    def __init__(self, api_key: str):
        super().__init__("Mistral 7B Instruct")
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("huggingface_hub not installed")
        
        self.client = InferenceClient(api_key=api_key)
        self.model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        logger.info(f"Initialized {self.model_name}")
    
    def generate(self, prompt: str) -> Dict[str, Any]:
        """Generate response from Mistral."""
        start_time = time.time()
        
        try:
            messages = [{"role": "user", "content": prompt}]
            
            response = self.client.chat_completion(
                messages=messages,
                model=self.model_id,
                max_tokens=500,
                temperature=0.7
            )
            
            elapsed_time = time.time() - start_time
            
            # Extract response
            response_text = response.choices[0].message.content
            
            # Token counting
            usage = response.usage if hasattr(response, 'usage') else None
            if usage:
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
                total_tokens = usage.total_tokens
            else:
                # Fallback estimation
                prompt_tokens = len(prompt.split()) * 1.3
                completion_tokens = len(response_text.split()) * 1.3
                total_tokens = prompt_tokens + completion_tokens
            
            # Cost: HuggingFace Inference API has free tier
            cost = 0.0  # Free tier
            
            return {
                "response": response_text,
                "tokens": {
                    "prompt": int(prompt_tokens),
                    "completion": int(completion_tokens),
                    "total": int(total_tokens)
                },
                "time": elapsed_time,
                "cost": cost,
                "model": self.model_name
            }
        
        except Exception as e:
            logger.error(f"Mistral generation error: {e}")
            return {
                "response": f"Error: {str(e)}",
                "tokens": {"prompt": 0, "completion": 0, "total": 0},
                "time": time.time() - start_time,
                "cost": 0.0,
                "model": self.model_name,
                "error": str(e)
            }


# ============================================================
# 4. MODEL COMPARISON & EVALUATION
# ============================================================

class ModelEvaluator:
    """
    Evaluates and compares LLM models using quantitative and qualitative metrics.
    
    Quantitative metrics:
    - Response time
    - Token usage
    - Cost
    
    Qualitative metrics (requires human evaluation):
    - Answer quality
    - Relevance
    - Naturalness
    - Correctness
    """
    
    def __init__(self, models: List[LLMModel]):
        self.models = models
        self.results = []
    
    def evaluate_query(
        self,
        query: str,
        combined_results: Dict[str, Any],
        expected_answer: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate all models on a single query.
        
        Args:
            query: User query
            combined_results: Combined baseline and embedding results
            expected_answer: Optional expected answer for accuracy evaluation
            
        Returns:
            List of evaluation results for each model
        """
        prompt = PromptBuilder.build_prompt(combined_results)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Evaluating query: {query}")
        logger.info(f"{'='*70}")
        
        evaluations = []
        
        for model in self.models:
            logger.info(f"\nTesting {model.model_name}...")
            
            result = model.generate(prompt)
            
            # Add query info
            result["query"] = query
            result["primary_source"] = combined_results.get("primary_source", "unknown")
            
            # Store result
            evaluations.append(result)
            self.results.append(result)
            
            # Display result
            logger.info(f"Response: {result['response'][:200]}...")
            logger.info(f"Time: {result['time']:.3f}s")
            logger.info(f"Tokens: {result['tokens']['total']}")
            logger.info(f"Cost: ${result['cost']:.6f}")
        
        return evaluations
    
    def generate_comparison_report(self, output_file: str = "llm_comparison_report.json"):
        """
        Generate comprehensive comparison report.
        
        Saves both JSON data and human-readable summary.
        """
        if not self.results:
            logger.warning("No results to report")
            return
        
        # Calculate aggregate statistics
        model_stats = {}
        for result in self.results:
            model_name = result["model"]
            if model_name not in model_stats:
                model_stats[model_name] = {
                    "model": model_name,
                    "queries": 0,
                    "total_time": 0.0,
                    "total_tokens": 0,
                    "total_cost": 0.0,
                    "errors": 0,
                    "responses": []
                }
            
            stats = model_stats[model_name]
            stats["queries"] += 1
            stats["total_time"] += result["time"]
            stats["total_tokens"] += result["tokens"]["total"]
            stats["total_cost"] += result["cost"]
            if "error" in result:
                stats["errors"] += 1
            stats["responses"].append({
                "query": result["query"],
                "response": result["response"],
                "time": result["time"],
                "tokens": result["tokens"],
                "primary_source": result.get("primary_source", "unknown")
            })
        
        # Calculate averages
        for model_name, stats in model_stats.items():
            n = stats["queries"]
            stats["avg_time"] = stats["total_time"] / n if n > 0 else 0
            stats["avg_tokens"] = stats["total_tokens"] / n if n > 0 else 0
            stats["avg_cost"] = stats["total_cost"] / n if n > 0 else 0
        
        # Create report
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(self.results),
            "models_compared": len(model_stats),
            "model_statistics": model_stats,
            "all_results": self.results
        }
        
        # Save JSON report
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"\n✅ Saved detailed report to {output_file}")
        
        # Generate human-readable summary
        summary_file = output_file.replace(".json", "_summary.txt")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("="*70 + "\n")
            f.write("LLM MODEL COMPARISON REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Queries: {len(self.results)}\n")
            f.write(f"Models Compared: {len(model_stats)}\n\n")
            
            f.write("="*70 + "\n")
            f.write("QUANTITATIVE METRICS (Averages)\n")
            f.write("="*70 + "\n\n")
            
            for model_name, stats in model_stats.items():
                f.write(f"Model: {model_name}\n")
                f.write(f"  Queries: {stats['queries']}\n")
                f.write(f"  Avg Response Time: {stats['avg_time']:.3f}s\n")
                f.write(f"  Avg Tokens: {stats['avg_tokens']:.1f}\n")
                f.write(f"  Total Cost: ${stats['total_cost']:.6f}\n")
                f.write(f"  Errors: {stats['errors']}\n\n")
            
            f.write("="*70 + "\n")
            f.write("QUALITATIVE EVALUATION TEMPLATE\n")
            f.write("="*70 + "\n\n")
            f.write("For each query below, please rate the model responses on:\n")
            f.write("1. Relevance (1-5): How well does it address the query?\n")
            f.write("2. Correctness (1-5): Is the information accurate?\n")
            f.write("3. Naturalness (1-5): How natural/fluent is the language?\n")
            f.write("4. Completeness (1-5): Does it provide sufficient detail?\n\n")
            
            # Group by query for easy comparison
            queries = {}
            for result in self.results:
                q = result["query"]
                if q not in queries:
                    queries[q] = []
                queries[q].append(result)
            
            for i, (query, responses) in enumerate(queries.items(), 1):
                f.write(f"\n{'='*70}\n")
                f.write(f"Query {i}: {query}\n")
                f.write(f"Primary Source: {responses[0].get('primary_source', 'unknown')}\n")
                f.write(f"{'='*70}\n\n")
                
                for result in responses:
                    f.write(f"--- {result['model']} ---\n")
                    f.write(f"Response: {result['response']}\n")
                    f.write(f"Time: {result['time']:.3f}s | Tokens: {result['tokens']['total']}\n")
                    f.write(f"Relevance: __ | Correctness: __ | Naturalness: __ | Completeness: __\n\n")
        
        logger.info(f"✅ Saved human evaluation template to {summary_file}")
        
        # Print summary to console
        print("\n" + "="*70)
        print("QUANTITATIVE COMPARISON SUMMARY")
        print("="*70)
        for model_name, stats in model_stats.items():
            print(f"\n{model_name}:")
            print(f"  Average Response Time: {stats['avg_time']:.3f}s")
            print(f"  Average Tokens: {stats['avg_tokens']:.1f}")
            print(f"  Total Cost: ${stats['total_cost']:.6f}")
            print(f"  Errors: {stats['errors']}")


# ============================================================
# 5. MAIN RAG SYSTEM - Complete Pipeline
# ============================================================

class FPLRAGSystem:
    """
    Complete RAG system for FPL queries.
    
    Pipeline:
    1. Process query with baseline retrieval
    2. Process query with embedding search
    3. Combine results
    4. Build structured prompt
    5. Generate response with LLM(s)
    """
    
    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        huggingface_api_key: Optional[str] = None,
        config_path: str = "config.txt"
    ):
        """Initialize RAG system with API keys."""
        if not MODULES_AVAILABLE:
            raise ImportError("Required modules not available")
        
        # Load config
        self.config = load_config()
        
        # Initialize retrievers
        self.baseline_retriever = BaselineRetriever(config_path)
        self.embedding_search = SemanticSearchBGEM3(self.config)
        
        # Initialize LLM models
        self.models = []
        
        if gemini_api_key and GEMINI_AVAILABLE:
            try:
                self.models.append(GeminiModel(gemini_api_key))
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
        
        if huggingface_api_key and HUGGINGFACE_AVAILABLE:
            try:
                self.models.append(LlamaModel(huggingface_api_key))
            except Exception as e:
                logger.error(f"Failed to initialize Llama: {e}")
            
            try:
                self.models.append(MistralModel(huggingface_api_key))
            except Exception as e:
                logger.error(f"Failed to initialize Mistral: {e}")
        
        if not self.models:
            logger.warning("No LLM models initialized")
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(self.models) if self.models else None
        
        logger.info(f"✅ RAG System initialized with {len(self.models)} models")
    
    def query(
        self,
        user_query: str,
        position_filter: Optional[str] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Process a user query through the complete RAG pipeline.
        
        Args:
            user_query: User's question
            position_filter: Optional position filter for embedding search
            top_k: Number of embedding results to retrieve
            
        Returns:
            Dict with combined results and LLM responses
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing query: {user_query}")
        logger.info(f"{'='*70}")
        
        # Step 1: Baseline retrieval
        logger.info("\n[1/4] Running baseline retrieval...")
        baseline_results = self.baseline_retriever.run_from_raw_query(user_query)
        logger.info(f"  Intent: {baseline_results['intent']}")
        logger.info(f"  Results: {len(baseline_results['results'])} rows")
        logger.info(f"  Description: {baseline_results['description']}")
        
        # Step 2: Embedding search
        logger.info("\n[2/4] Running embedding search...")
        embedding_results = self.embedding_search.search(
            user_query,
            top_k=top_k,
            position=position_filter
        )
        logger.info(f"  Results: {len(embedding_results)} semantic matches")
        
        # Step 3: Combine results
        logger.info("\n[3/4] Combining results...")
        combined_results = ResultCombiner.combine_results(
            baseline_results,
            embedding_results,
            user_query
        )
        logger.info(f"  Primary source: {combined_results['primary_source']}")
        
        # Step 4: Generate LLM responses
        logger.info("\n[4/4] Generating LLM responses...")
        if self.evaluator:
            evaluations = self.evaluator.evaluate_query(user_query, combined_results)
        else:
            evaluations = []
            logger.warning("No models available for evaluation")
        
        return {
            "query": user_query,
            "baseline_results": baseline_results,
            "embedding_results": embedding_results,
            "combined_results": combined_results,
            "llm_evaluations": evaluations
        }
    
    def close(self):
        """Clean up resources."""
        self.baseline_retriever.close()
        self.embedding_search.close()
        logger.info("✅ RAG System closed")
    
    def generate_report(self, output_file: str = "llm_comparison_report.json"):
        """Generate comparison report."""
        if self.evaluator:
            self.evaluator.generate_comparison_report(output_file)
        else:
            logger.warning("No evaluator available")


# ============================================================
# 6. MAIN - Testing and Evaluation
# ============================================================

def main():
    """Main testing function."""
    print("="*70)
    print("FPL RAG SYSTEM - LLM Layer (Milestone 3)")
    print("="*70)
    print("\nThis system compares three LLM models:")
    print("  - Gemini 2.5 Flash (Google)")
    print("  - Llama 3 8B (HuggingFace)")
    print("  - Mistral 7B Instruct (HuggingFace)")
    print("\nMetrics: Response time, token usage, cost, quality")
    print("="*70)
    
    # Get API keys from environment or user input
    gemini_key = CONFIG.get("GEMINI_API_KEY")
    hf_key = CONFIG.get("HUGGINGFACE_API_KEY")
    
    if not gemini_key:
        print("\n⚠️  Gemini API key not found in environment.")
        gemini_key = input("Enter Gemini API key (or press Enter to skip): ").strip()
        if not gemini_key:
            gemini_key = None
    
    if not hf_key:
        print("\n⚠️  HuggingFace API key not found in environment.")
        hf_key = input("Enter HuggingFace API key (or press Enter to skip): ").strip()
        if not hf_key:
            hf_key = None
    
    if not gemini_key and not hf_key:
        print("\n❌ No API keys provided. Cannot test models.")
        return
    
    # Initialize RAG system
    print("\n🔧 Initializing RAG system...")
    rag = FPLRAGSystem(gemini_api_key=gemini_key, huggingface_api_key=hf_key)
    
    # Test queries
    test_queries = [
        "Show me the forwards with the most goals scored across all seasons.",
        "List the defenders with the highest number of clean sheets across all seasons.",
        "Which midfielders have the most assists overall?",
        "Which goalkeepers have made the most saves across the seasons?",
        "Who are the players currently showing the best form?",
    ]
    
    print(f"\n📝 Testing with {len(test_queries)} queries...")
    print("="*70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Query {i}/{len(test_queries)}]")
        try:
            result = rag.query(query)
            print(f"✅ Query completed")
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"Query failed: {e}", exc_info=True)
    
    # Generate report
    print("\n📊 Generating comparison report...")
    rag.generate_report("llm_comparison_report.json")
    
    # Cleanup
    rag.close()
    
    print("\n✅ Evaluation complete!")
    print("📄 Check 'llm_comparison_report.json' for detailed results")
    print("📄 Check 'llm_comparison_report_summary.txt' for human evaluation template")


if __name__ == "__main__":
    main()
