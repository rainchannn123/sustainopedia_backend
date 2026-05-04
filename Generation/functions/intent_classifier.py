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
from typing import Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class IntentParams:
    """Structured parameters extracted from a user LCA query."""
    intent: str                        # 'computation' | 'literature' | 'general'
    product: str                       # primary product/material being assessed
    amount: float                      # functional unit amount (e.g. 1.0)
    unit: str                          # functional unit unit string (e.g. 'tonne')
    system_boundary: str               # 'cradle-to-gate' | 'cradle-to-grave' | 'gate-to-gate'
    run_mc: bool                       # whether Monte Carlo analysis was explicitly requested
    n_simulations: int                 # MC simulation count; capped at 500
    region: Optional[str]              # production region if mentioned (e.g. 'China')
    comparison_product: Optional[str]  # second product for comparison queries
    region_mode: Optional[str]         # 'country' | 'region'
    materials: Optional[str]           # user-provided material composition notes
    distribution: Optional[str]        # user-provided distribution details
    use_phase: Optional[str]           # user-provided use phase details
    end_of_life: Optional[str]         # user-provided end-of-life details
    additional_notes: Optional[str]    # any extra constraints or notes

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
    region_mode='region',
    materials=None,
    distribution=None,
    use_phase=None,
    end_of_life=None,
    additional_notes=None,
)


# ---------------------------------------------------------------------------
# LLM extraction prompt
# ---------------------------------------------------------------------------

# Note: {{ and }} are escaped braces for LangChain's template parser; {var} are template slots.
_EXTRACTION_PROMPT = """\
You are an expert LCA query parser. Extract structured parameters from the user query below \
and return a single valid JSON object with NO additional text or markdown fences.

If the query already contains explicit key-value labels (for example "Amount:", "Unit:", \
"System boundary:"), prioritize those explicit values.

User Query: {question}
Pre-filled Product (may be empty): {product}

Fields to extract:
- "intent": one of "computation", "literature", "general". Default "computation".
- "product": primary product or material. Use pre-filled value if consistent with the query; \
otherwise extract or guess from query.
- "amount": numeric functional unit quantity (e.g. 1, 100).
- "unit": unit string (e.g. "tonne", "kilogram", "kWh", "unit").
- "system_boundary": one of "cradle-to-gate", "cradle-to-grave", "gate-to-gate". Default "cradle-to-gate".
- "run_mc": true only if user explicitly mentions uncertainty analysis, Monte Carlo, or probabilistic analysis.
- "n_simulations": integer MC count if user specifies (e.g. "run 100 simulations"). Min 1. Max 10000. if "run_mc" == false, set to 1.
- "region": country or region of production if explicitly mentioned (e.g. "China", "Europe"). "GLO" otherwise.
- "comparison_product": second product name if this is a comparison query. null otherwise.
- "region_mode": "country" or "region" if stated. Default "country".
- "materials": material composition or ingredient notes if provided. null otherwise.
- "distribution": transport/distribution description if provided. null otherwise.
- "use_phase": use phase description (lifespan, in-use energy/water) if provided. null otherwise.
- "end_of_life": disposal/recycling description if provided. null otherwise.
- "additional_notes": extra constraints or notes if provided. null otherwise.

For your output, you must strictly follow on information given by user query to fill in the JSON object. Example of expected output JSON format, no extra text:
{{
  "intent": "computation",
  "product": "Apple",
  "amount": 1.0,
  "unit": "tonne",
  "system_boundary": "cradle-to-gate",
  "run_mc": false,
  "n_simulations": 25,
  "region": "GLO",
  "comparison_product": null,
  "region_mode": "country",
  "materials": "100 percentage apple flesh, no peel",
  "distribution": "1000 km truck transport",
  "use_phase": "2 years",
  "end_of_life": "landfill",
  "additional_notes": null
}}"""


def _to_nullable_str(value: Any) -> Optional[str]:
    text = str(value).strip() if value is not None else ''
    if not text:
        return None
    lowered = text.lower()
    if lowered in {'null', 'none', 'n/a', 'na', 'not specified', 'no comparison', 'not needed'}:
        return None
    return text


def _to_positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
        return parsed if parsed > 0 else default
    except Exception:
        return default


def _to_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    lowered = str(value).strip().lower()
    if lowered in {'true', '1', 'yes', 'y'}:
        return True
    if lowered in {'false', '0', 'no', 'n'}:
        return False
    return default


def _clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return default
    return max(low, min(high, parsed))


def _normalize_intent(value: Any) -> str:
    text = str(value or '').strip().lower()
    if 'literature' in text or 'paper' in text or 'study' in text:
        return 'literature'
    if 'general' in text:
        return 'general'
    return 'computation'


def _normalize_system_boundary(value: Any) -> str:
    text = str(value or '').strip().lower().replace('_', '-').replace(' ', '-')
    if text in {'cradle-to-gate', 'cradle-gate'}:
        return 'cradle-to-gate'
    if text in {'cradle-to-grave', 'cradle-grave'}:
        return 'cradle-to-grave'
    if text in {'gate-to-gate', 'gate-gate'}:
        return 'gate-to-gate'
    return _DEFAULTS.system_boundary


def _normalize_region_mode(value: Any) -> str:
    text = str(value or '').strip().lower()
    if text == 'country':
        return 'country'
    return 'region'


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
            intent=_normalize_intent(data.get("intent", _DEFAULTS.intent)),
            product=str(data.get("product") or product or _DEFAULTS.product),
            amount=_to_positive_float(data.get("amount", _DEFAULTS.amount), _DEFAULTS.amount),
            unit=str(data.get("unit", _DEFAULTS.unit)),
            system_boundary=_normalize_system_boundary(data.get("system_boundary", _DEFAULTS.system_boundary)),
            run_mc=_to_bool(data.get("run_mc", _DEFAULTS.run_mc), _DEFAULTS.run_mc),
            n_simulations=_clamp_int(data.get("n_simulations", _DEFAULTS.n_simulations), _DEFAULTS.n_simulations, 1, 500),
            region=_to_nullable_str(data.get("region")),
            comparison_product=_to_nullable_str(data.get("comparison_product")),
            region_mode=_normalize_region_mode(data.get("region_mode", _DEFAULTS.region_mode)),
            materials=_to_nullable_str(data.get("materials")),
            distribution=_to_nullable_str(data.get("distribution")),
            use_phase=_to_nullable_str(data.get("use_phase")),
            end_of_life=_to_nullable_str(data.get("end_of_life")),
            additional_notes=_to_nullable_str(data.get("additional_notes")),
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
            region_mode=_DEFAULTS.region_mode,
            materials=_DEFAULTS.materials,
            distribution=_DEFAULTS.distribution,
            use_phase=_DEFAULTS.use_phase,
            end_of_life=_DEFAULTS.end_of_life,
            additional_notes=_DEFAULTS.additional_notes,
        )
