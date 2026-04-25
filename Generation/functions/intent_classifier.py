"""
intent_classifier.py
--------------------
Classifies user LCA queries and extracts structured query parameters in a single LLM call.

The extracted IntentParams guide downstream pipeline routing (BW2 computation vs. fast RAG
response) and are stored in MongoDB for analytics and session continuity.
"""

import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class IntentParams:
    """Structured parameters extracted from a user LCA query."""
    intent: str                       # 'computation' | 'literature' | 'general'
    product: str                      # primary product/material being assessed
    amount: float                     # functional unit amount (e.g. 1.0)
    unit: str                         # functional unit unit string (e.g. 'tonne')
    system_boundary: str              # 'cradle-to-gate' | 'cradle-to-grave' | 'gate-to-gate'
    run_mc: bool                      # whether Monte Carlo analysis was explicitly requested
    n_simulations: int                # MC simulation count; capped at 500
    region: Optional[str]             # production region if mentioned (e.g. 'China')
    comparison_product: Optional[str] # second product for comparison queries (reserved for future)

    def to_dict(self) -> dict:
        return asdict(self)


# Sensible defaults used when LLM parsing fails
_DEFAULTS = IntentParams(
    intent='computation',
    product='Not Specified',
    amount=1.0,
    unit='tonne',
    system_boundary='cradle-to-gate',
    run_mc=False,
    n_simulations=25,
    region=None,
    comparison_product=None,
)


# ---------------------------------------------------------------------------
# LLM extraction prompt
# ---------------------------------------------------------------------------

# Note: {{ and }} are escaped braces for LangChain's template parser; {var} are template slots.
_EXTRACTION_PROMPT = """\
You are an expert LCA query parser. Extract structured parameters from the user query below \
and return a single valid JSON object with NO additional text or markdown fences.

User Query: {question}
Pre-filled Product (may be empty): {product}

Fields to extract:
- "intent": one of:
    "computation"  - user wants LCA calculation, carbon footprint, or environmental impact score
    "literature"   - user wants paper summaries, literature review, or study references
    "general"      - conceptual/explanatory question; no LCA calculation needed
- "product": primary product or material. Use pre-filled value if consistent with the query; \
otherwise extract from query. Default "Not Specified".
- "amount": numeric functional unit quantity (e.g. 1, 100). Default 1.0.
- "unit": unit string (e.g. "tonne", "kilogram", "kWh", "unit"). Default "tonne".
- "system_boundary": one of "cradle-to-gate", "cradle-to-grave", "gate-to-gate". Default "cradle-to-gate".
- "run_mc": true only if user explicitly mentions uncertainty analysis, Monte Carlo, or probabilistic analysis.
- "n_simulations": integer MC count if user specifies (e.g. "run 100 simulations"). Default 25. Max 500.
- "region": country or region of production if explicitly mentioned (e.g. "China", "Europe"). null otherwise.
- "comparison_product": second product name if this is a comparison query. null otherwise.

Return only this JSON, no extra text:
{{
  "intent": "computation",
  "product": "Not Specified",
  "amount": 1.0,
  "unit": "tonne",
  "system_boundary": "cradle-to-gate",
  "run_mc": false,
  "n_simulations": 25,
  "region": null,
  "comparison_product": null
}}"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_intent(question: str, product: str, gpt5_client) -> IntentParams:
    """
    Classify query intent and extract LCA parameters in one LLM call.
    Returns IntentParams with sensible defaults on any parse error (fail-safe).
    """
    chain = ChatPromptTemplate.from_template(_EXTRACTION_PROMPT) | gpt5_client | StrOutputParser()
    raw = ""
    try:
        try:
            raw = chain.invoke({"question": question, "product": product or "Not Specified"})
        except:
            # Rerun once to mitigate occasional LLM generation failures
            raw = chain.invoke({"question": question, "product": product or "Not Specified"})

        # Strip markdown code fences if the model wraps its response
        stripped = raw.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("```")[1]
            if stripped.startswith("json"):
                stripped = stripped[4:]

        data = json.loads(stripped.strip())

        return IntentParams(
            intent=str(data.get("intent", _DEFAULTS.intent)),
            product=str(data.get("product") or product or _DEFAULTS.product),
            amount=float(data.get("amount", _DEFAULTS.amount)),
            unit=str(data.get("unit", _DEFAULTS.unit)),
            system_boundary=str(data.get("system_boundary", _DEFAULTS.system_boundary)),
            run_mc=bool(data.get("run_mc", _DEFAULTS.run_mc)),
            n_simulations=min(int(data.get("n_simulations", _DEFAULTS.n_simulations)), 500),
            region=data.get("region") or None,
            comparison_product=data.get("comparison_product") or None,
        )

    except Exception as exc:
        logging.warning(
            f"[intent_classifier] Parsing failed — using defaults. "
            f"Error: {exc}  |  Raw: {raw!r}"
        )
        return IntentParams(
            intent=_DEFAULTS.intent,
            product=product or _DEFAULTS.product,
            amount=_DEFAULTS.amount,
            unit=_DEFAULTS.unit,
            system_boundary=_DEFAULTS.system_boundary,
            run_mc=_DEFAULTS.run_mc,
            n_simulations=_DEFAULTS.n_simulations,
            region=_DEFAULTS.region,
            comparison_product=_DEFAULTS.comparison_product,
        )
