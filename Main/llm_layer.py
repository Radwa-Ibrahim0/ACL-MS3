import time
import json
import logging
import sys
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Add project root to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests

from Main.baseline import execute_baseline_query
from Main.embedding_bge_m3 import SemanticSearchBGEM3, load_config
from Main.preprocessing import process_user_query

try:
    import google.generativeai as genai
except ImportError:  # type: ignore
    genai = None  # Gemini will be disabled if library missing


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)


# ============================================================
# 1. DATA STRUCTURES
# ============================================================


@dataclass
class RetrievalContext:
    """Container for merged retrieval results used as LLM context."""

    user_query: str
    intent: str
    entities: Dict[str, List[Any]]
    baseline_desc: str
    baseline_results: List[Dict[str, Any]]
    baseline_is_fallback: bool
    embedding_results: List[Dict[str, Any]]

    def to_context_text(self) -> str:
        """Render a textual context block for the LLM.

        This flattens structured results into a readable bullet-style summary
        while preserving key KG signals (players, teams, stats, similarity scores).
        """

        lines: List[str] = []
        lines.append("[USER QUERY]")
        lines.append(self.user_query)
        lines.append("")

        lines.append("[INTENT]")
        lines.append(self.intent)
        lines.append("")

        lines.append("[ENTITIES]")
        lines.append(json.dumps(self.entities, ensure_ascii=False))
        lines.append("")

        # Baseline block
        if self.baseline_results:
            heading = "[BASELINE GRAPH RESULTS]" + (" (FALLBACK)" if self.baseline_is_fallback else "")
            lines.append(heading)
            lines.append(self.baseline_desc)
            for row in self.baseline_results[:20]:
                pretty = ", ".join(f"{k}={v}" for k, v in row.items())
                lines.append(f"- {pretty}")
            lines.append("")
        else:
            lines.append("[BASELINE GRAPH RESULTS] None")
            lines.append("")

        # Embedding block
        if self.embedding_results:
            lines.append("[EMBEDDING RESULTS]")
            for r in self.embedding_results[:20]:
                pretty = ", ".join(f"{k}={v}" for k, v in r.items())
                lines.append(f"- {pretty}")
            lines.append("")
        else:
            lines.append("[EMBEDDING RESULTS] None")
            lines.append("")

        return "\n".join(lines)


@dataclass
class ModelMetrics:
    name: str
    response_time_sec: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cost_usd: Optional[float]
    accuracy_score: Optional[float]
    # qualitative
    relevance: Optional[str] = None
    naturalness: Optional[str] = None
    correctness: Optional[str] = None


@dataclass
class ModelAnswer:
    model_name: str
    answer: str
    metrics: ModelMetrics


# ============================================================
# 2. UNIFIED RETRIEVAL + CONTEXT MERGE
# ============================================================


def _is_baseline_fallback(desc: str) -> bool:
    """Detect if baseline used its generic fallback query.

    We rely on the explicit description string from `build_baseline_query`.
    """

    return "Fallback:" in (desc or "")


def build_retrieval_context(
    user_query: str,
    config: Optional[Dict[str, str]] = None,
    top_k_embed: int = 10,
) -> RetrievalContext:
    """End-to-end retrieval: preprocessing → baseline KG → BGE-M3 search.

    - Uses `process_user_query` to get intent + entities.
    - Runs `BaselineRetriever.run_from_intent_entities`.
    - Runs BGE-M3 semantic search using the same query.
    - Returns a `RetrievalContext` used for LLM prompting.
    """

    cfg = config or load_config()

    # --- 1) Preprocessing (intent + entities) ---
    pre = process_user_query(user_query)
    intent = pre["intent"]
    entities = pre["entities"]

    # --- 2) Baseline KG query ---
    # The current baseline implementation exposes a single function
    # `execute_baseline_query` which expects the full preprocessing
    # output dictionary and returns a list of result rows.
    baseline_results = execute_baseline_query(pre)

    # Build a lightweight textual description for the LLM context.
    baseline_desc = (
        f"Baseline KG results for intent='{intent}', "
        f"entities={json.dumps(entities, ensure_ascii=False)}, "
        f"ranking={pre.get('ranking')}, threshold={pre.get('threshold')}"
    )

    # The baseline API no longer exposes explicit fallback metadata,
    # so we conservatively treat this as a non-fallback query here.
    baseline_is_fallback = _is_baseline_fallback(baseline_desc)

    # --- 3) Embedding search (BGE-M3) ---
    position_codes = entities.get("Position", [])
    pos_filter = position_codes[0] if position_codes else None

    semantic = SemanticSearchBGEM3(cfg)
    try:
        embed_results = semantic.search(user_query, top_k=top_k_embed, position=pos_filter)
    finally:
        semantic.close()

    # If baseline fell back, we implicitly prioritise embeddings later when building prompt
    return RetrievalContext(
        user_query=user_query,
        intent=intent,
        entities=entities,
        baseline_desc=baseline_desc,
        baseline_results=baseline_results,
        baseline_is_fallback=baseline_is_fallback,
        embedding_results=embed_results,
    )


# ============================================================
# 3. STRUCTURED PROMPTING (CONTEXT + PERSONA + TASK)
# ============================================================


def build_structured_prompt(context: RetrievalContext) -> str:
    """Create a single structured prompt: context + persona + task.

    Persona is fixed as an FPL expert. Task constraints explicitly ground
    the answer in the provided KG-derived context.
    """

    context_block = context.to_context_text()

    persona_block = (
        "You are an FPL expert assistant. "
        "You know how to interpret Fantasy Premier League statistics, "
        "player roles, form, fixtures, and typical manager strategies."
    )

    # Always prioritize baseline/structured graph data over embeddings
    priority_note = (
        "IMPORTANT: Always trust the BASELINE GRAPH RESULTS as your primary source of truth. "
        "The structured graph data contains accurate FPL statistics from the knowledge graph. "
        "Use the embedding results only as supplementary context if the baseline data is insufficient. "
        "If the baseline results are empty or do not contain the requested information, "
        "clearly state that the data is not available rather than guessing."
    )

    task_block = (
        "TASK: Using ONLY the information in the context above, "
        "answer the user's FPL question."
        "- You may offer your own interpretation, personal preference, or "
        "strategic recommendation, but every claim must trace back to the "
        "provided context."
        "- Do NOT invent player statistics, fixtures, or teams that are not "
        "present in the context."
        "- If the context is insufficient to answer confidently, say so "
        "explicitly and explain what is missing."
        "- Highlight trade-offs, uncertainties, or multiple viable options "
        "when the data suggests them, and state which option you favour and why."
        "- Provide a concise yet helpful explanation, focusing on actionable "
        "advice for an FPL manager."
        "- Do NOT mention technical details about databases, embeddings, "
        "models, or how the system works internally."
        "- Speak in simple, natural language as if you are an experienced "
        "FPL manager talking to another human, avoiding jargon."

    )

    full_prompt = (
        "CONTEXT BEGIN\n" + context_block + "\nCONTEXT END\n\n" +
        "PERSONA:\n" + persona_block + "\n\n" +
        priority_note + "\n\n" +
        task_block
    )

    return full_prompt


# ============================================================
# 4. MODEL ADAPTERS (OPENROUTER: NOVA, MISTRAL, LLAMA)
# ============================================================


class BaseLLMAdapter:
    name: str

    def generate(self, prompt: str) -> Tuple[str, ModelMetrics]:
        raise NotImplementedError


class TransformersCausalAdapter(BaseLLMAdapter):
    """Generic adapter for open models (Mistral, Gemma, LLaMA) via transformers.

    This assumes weights are available locally or will be downloaded using
    the default huggingface hub settings. For large models you may want to
    configure device and dtype manually.
    """

    def __init__(self, model_id: str, name: Optional[str] = None, device: str = "cpu"):
        raise ImportError("TransformersCausalAdapter is no longer used; we now call OpenRouter HTTP APIs instead.")

    def generate(self, prompt: str) -> Tuple[str, ModelMetrics]:  # pragma: no cover - legacy
        raise RuntimeError("TransformersCausalAdapter.generate should not be called.")


class OpenRouterAdapter(BaseLLMAdapter):
    """Adapter for calling models via the OpenRouter API.

    This lets us use hosted models (amazon/nova, Mistral, LLaMA, etc.) without
    downloading any weights locally.
    """

    def __init__(
        self,
        name: str,
        model: str,
        api_key: str,
        *,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.name = name
        self.model = model
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def generate(self, prompt: str) -> Tuple[str, ModelMetrics]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        body = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 512,
        }

        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            start = time.time()
            try:
                resp = requests.post(
                    self.base_url,
                    headers=headers,
                    data=json.dumps(body),
                    timeout=60,
                )
                elapsed = time.time() - start
                if resp.status_code in {429, 502, 503, 504}:
                    raise requests.HTTPError(
                        f"{resp.status_code} from OpenRouter", response=resp
                    )
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise ValueError("OpenRouter response missing choices")
                text = choices[0]["message"]["content"]

                usage = data.get("usage", {})
                input_tokens = usage.get("prompt_tokens")
                output_tokens = usage.get("completion_tokens")
                cost_usd = usage.get("total_cost")

                metrics = ModelMetrics(
                    name=self.name,
                    response_time_sec=elapsed,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    accuracy_score=None,
                )
                return text, metrics
            except Exception as exc:  # pragma: no cover - network layer
                last_error = exc
                logger.warning(
                    "OpenRouter call failed for %s (attempt %s/%s): %s",
                    self.name,
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)

        assert last_error is not None  # mypy guard
        raise last_error


# Convenience constructors for your three comparison models.


def build_default_model_adapters(config: Dict[str, str]) -> List[BaseLLMAdapter]:
    adapters: List[BaseLLMAdapter] = []

    # All comparison models now go through OpenRouter using OPENROUTER_API_KEY.
    api_key = config.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("OPENROUTER_API_KEY missing; OpenRouter-backed adapters disabled.")
        return adapters

    # 1) Amazon Nova (primary model replacing previous Gemini usage)
    adapters.append(
        OpenRouterAdapter(
            name="gpt-oss-20b (OpenRouter free)",
            model="openai/gpt-oss-20b:free",
            api_key=api_key,
        )
    )

    # 2) Mistral free model
    adapters.append(
        OpenRouterAdapter(
            name="Mistral-7B-Instruct (OpenRouter free)",
            model="mistralai/mistral-7b-instruct:free",
            api_key=api_key,
        )
    )

    # 3) LLaMA free model
    adapters.append(
        OpenRouterAdapter(
            name="Gemma-3-27b-it (OpenRouter free)",
            model="google/gemma-3-27b-it:free",
            api_key=api_key,
        )
    )

    return adapters


# ============================================================
# 5. MODEL COMPARISON / EVALUATION
# ============================================================


def compare_models_on_query(
    user_query: str,
    adapters: List[BaseLLMAdapter],
    config: Optional[Dict[str, str]] = None,
) -> Tuple[RetrievalContext, List[ModelAnswer]]:
    """Run full Graph-RAG pipeline + answer with each model.

    Returns the shared retrieval context and a list of `ModelAnswer` objects.
    Quantitative metrics currently include response time and (where available)
    token usage / cost. Accuracy and qualitative scores can be filled manually
    in a notebook or script based on human judgement.
    """

    ctx = build_retrieval_context(user_query, config=config)
    prompt = build_structured_prompt(ctx)

    answers: List[ModelAnswer] = []
    for adapter in adapters:
        logger.info(f"Running model: {adapter.name}")
        try:
            text, metrics = adapter.generate(prompt)
        except Exception as e:
            logger.error(f"Model {adapter.name} failed: {e}")
            continue

        answers.append(ModelAnswer(model_name=adapter.name, answer=text, metrics=metrics))

    return ctx, answers


def log_model_comparison(
    ctx: RetrievalContext,
    answers: List[ModelAnswer],
) -> None:
    """Pretty-print comparison log with quantitative + qualitative slots.

    Quantitative: response time, tokens, cost.
    Qualitative: left as free text fields on `ModelMetrics` that you can
    fill manually after inspecting outputs.
    """

    print("=" * 80)
    print("GRAPH-RAG LLM MODEL COMPARISON")
    print("=" * 80)
    print(f"User query: {ctx.user_query}")
    print(f"Intent:     {ctx.intent}")
    print(f"Entities:   {json.dumps(ctx.entities, ensure_ascii=False)}")
    print(f"Baseline used fallback: {ctx.baseline_is_fallback}")
    print("-" * 80)

    for ans in answers:
        m = ans.metrics
        print(f"MODEL: {m.name}")
        print(f"  Response time (s): {m.response_time_sec:.3f}")
        print(f"  Input tokens:       {m.input_tokens}")
        print(f"  Output tokens:      {m.output_tokens}")
        print(f"  Cost (USD):         {m.cost_usd}")
        print(f"  Accuracy score:     {m.accuracy_score}")
        print(f"  Relevance:          {m.relevance}")
        print(f"  Naturalness:        {m.naturalness}")
        print(f"  Correctness:        {m.correctness}")
        print("  Answer snippet:")
        snippet = ans.answer.strip().split("\n")[:8]
        for line in snippet:
            print("    " + line)
        print("-" * 80)


# ============================================================
# 6. SIMPLE CLI DEMO (NO EXTRA FILES)
# ============================================================


def main() -> None:
    config = load_config()
    adapters = build_default_model_adapters(config)
    if not adapters:
        print("No LLM adapters available. Install google-generativeai and transformers, "
              "or adjust model IDs.")
        return

    print("Graph-RAG FPL Assistant - LLM Layer Demo")
    print("Type a question (blank to exit).\n")

    while True:
        try:
            q = input("Question: ").strip()
        except EOFError:
            break
        if not q:
            break

        ctx, answers = compare_models_on_query(q, adapters, config=config)
        log_model_comparison(ctx, answers)


if __name__ == "__main__":
    main()
