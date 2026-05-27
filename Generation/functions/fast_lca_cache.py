"""fast_lca_cache.py — MongoDB-backed result cache for the fast-response LCA pipeline.

Stores per-product LCA computation results so repeated queries with the same or
similar intent bypass the BW2 computation engine and are served from cache.
Similarity is measured via cosine distance between Azure OpenAI text embeddings
of the full user query; a configurable threshold gates cache hits.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_SIMILARITY_THRESHOLD = 0.85   # cosine similarity required for a cache hit (full-query embeddings)
_COLLECTION_NAME = "fast_lca_cache"


def _cosine_similarity(a: list, b: list) -> float:
    """Return the cosine similarity between two embedding vectors."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a, norm_b = np.linalg.norm(va), np.linalg.norm(vb)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


# ── Mass-scaling helpers ───────────────────────────────────────────────────────

_TO_KG: dict = {
    "kg": 1.0,
    "kilogram": 1.0,
    "kilograms": 1.0,
    "g": 0.001,
    "gram": 0.001,
    "grams": 0.001,
    "tonne": 1_000.0,
    "ton": 1_000.0,
    "t": 1_000.0,
    "mt": 1_000.0,
    "lb": 0.453_592_37,
    "lbs": 0.453_592_37,
    "pound": 0.453_592_37,
    "pounds": 0.453_592_37,
}


def _normalize_to_kg(amount: float, unit: str) -> float:
    """Convert an (amount, unit) pair to an equivalent mass in kilograms."""
    return amount * _TO_KG.get(unit.lower().strip(), 1.0)


# Matches the leading numeric amount in an amount_location string, e.g. "10.0 tonne, CN".
_AMOUNT_LOC_RE = re.compile(r'^(\d+(?:\.\d+)?)\s+(.+)$')


def _scale_amount_location(s: str, factor: float) -> str:
    """Scale the leading numeric quantity in an amount_location display string.

    Format: "<amount> <unit>, <location>"  e.g. "1.0 tonne, CN"
    Discrete countable items (unit / piece / pcs) are left unchanged because
    a product that contains 1 motor still contains 1 motor regardless of the
    requested functional unit mass.
    """
    if not s:
        return s
    m = _AMOUNT_LOC_RE.match(s.strip())
    if not m:
        return s
    num_str, rest = m.group(1), m.group(2)
    # Skip countable units — these are not mass-proportional.
    if rest.lower().lstrip().startswith(("unit", "piece", "pcs", "item")):
        return s
    scaled_num = round(float(num_str) * factor, 4)
    return f"{scaled_num} {rest}"


def _scale_lca_result(
    cached_doc: dict,
    scale_factor: float,
    req_amount: float,
    req_unit: str,
) -> dict:
    """
    Return a new result dict with all numeric LCA values multiplied by *scale_factor*.

    Scaled fields inside *processed_json* (JSON string) and *lcia_table* (dict):
    - processes[*].amount_location  — input activity amount display string
    - processes[*].mean_impact      — per-process GWP impact
    - processes[*].sd               — standard deviation (numeric string)
    - processes[*].percentile       — "p5 / p95" string
    - totalMeanImpact               — overall total
    - stageImpacts values

    The original cached answer text is returned unchanged.
    """
    # ── Scale processed_json ─────────────────────────────────────────────────
    # Field names in processed_json come from generate_json() / result_data_processing():
    #   mean_impact  (float)          — primary per-process impact shown in the table
    #   sd           (numeric string) — standard deviation, e.g. "0.0523"
    #   percentile   (string)         — "p5 / p95", e.g. "0.41 / 0.63"
    #   totalMeanImpact (float)       — top-level sum shown as the overall total
    scaled_json_str = cached_doc.get("processed_json")
    if scaled_json_str:
        try:
            data = json.loads(scaled_json_str)
            for p in data.get("processes", []):
                # Input activity amount display string e.g. "1.0 tonne, CN"
                if p.get("amount_location"):
                    p["amount_location"] = _scale_amount_location(p["amount_location"], scale_factor)
                # Primary per-process impact field used by the frontend table
                if isinstance(p.get("mean_impact"), (int, float)):
                    p["mean_impact"] = round(p["mean_impact"] * scale_factor, 4)
                # SD stored as a numeric string
                if p.get("sd") not in ("", None, "N/A"):
                    try:
                        p["sd"] = str(round(float(p["sd"]) * scale_factor, 4))
                    except (ValueError, TypeError):
                        pass
                # Percentile stored as "p5 / p95" string
                if p.get("percentile") not in ("", None, "N/A"):
                    try:
                        parts = str(p["percentile"]).split(" / ")
                        if len(parts) == 2:
                            p["percentile"] = (
                                f"{round(float(parts[0]) * scale_factor, 4):.4g}"
                                f" / "
                                f"{round(float(parts[1]) * scale_factor, 4):.4g}"
                            )
                    except (ValueError, TypeError):
                        pass
            if "totalMeanImpact" in data:
                data["totalMeanImpact"] = round(data["totalMeanImpact"] * scale_factor, 4)
            for k in list(data.get("stageImpacts", {})):
                v = data["stageImpacts"][k]
                if isinstance(v, (int, float)):
                    data["stageImpacts"][k] = round(v * scale_factor, 4)
            scaled_json_str = json.dumps(data)
        except Exception:
            pass  # keep original string on parse failure

    # ── Scale lcia_table ─────────────────────────────────────────────────────
    # lcia_table mirrors lcia_data from result_data_processing() — same field names.
    scaled_table = cached_doc.get("lcia_table")
    if scaled_table:
        try:
            scaled_table = copy.deepcopy(scaled_table)
            for p in scaled_table.get("processes", []):
                if p.get("amount_location"):
                    p["amount_location"] = _scale_amount_location(p["amount_location"], scale_factor)
                if isinstance(p.get("mean_impact"), (int, float)):
                    p["mean_impact"] = round(p["mean_impact"] * scale_factor, 4)
                if p.get("sd") not in ("", None, "N/A"):
                    try:
                        p["sd"] = str(round(float(p["sd"]) * scale_factor, 4))
                    except (ValueError, TypeError):
                        pass
                if p.get("percentile") not in ("", None, "N/A"):
                    try:
                        parts = str(p["percentile"]).split(" / ")
                        if len(parts) == 2:
                            p["percentile"] = (
                                f"{round(float(parts[0]) * scale_factor, 4):.4g}"
                                f" / "
                                f"{round(float(parts[1]) * scale_factor, 4):.4g}"
                            )
                    except (ValueError, TypeError):
                        pass
            if "totalMeanImpact" in scaled_table:
                scaled_table["totalMeanImpact"] = round(
                    scaled_table["totalMeanImpact"] * scale_factor, 4
                )
            for k in list(scaled_table.get("stageImpacts", {})):
                v = scaled_table["stageImpacts"][k]
                if isinstance(v, (int, float)):
                    scaled_table["stageImpacts"][k] = round(v * scale_factor, 4)
        except Exception:
            scaled_table = cached_doc.get("lcia_table")  # keep original on failure

    return {
        "answer":         cached_doc.get("answer"),
        "markdown":       None,
        "processed_json": scaled_json_str,
        "lcia_table":     scaled_table,
        "intent_params":  cached_doc.get("intent_params", {}),
    }


class _FastLcaCache:
    """Thread-safe MongoDB wrapper for fast-LCA result caching (lazy-connects)."""

    def __init__(self) -> None:
        self._col = None  # lazy-initialized on first use

    def _get_collection(self):
        if self._col is not None:
            return self._col
        try:
            from pymongo import MongoClient
            from pymongo.uri_parser import parse_uri

            uri = os.getenv("MONGODB_URI")
            if not uri:
                logger.warning("FastLcaCache: MONGODB_URI not set — cache disabled")
                return None

            # Use pymongo's own URI parser to extract the database name.
            # Falls back to "test" (Mongoose's default when no DB is in the URI),
            # ensuring Python and Node.js target the same database.
            parsed = parse_uri(uri)
            db_name = parsed.get("database") or "test"

            client = MongoClient(uri, serverSelectionTimeoutMS=5_000)
            # Verify connectivity before storing the collection reference,
            # mirroring the Node.js .then(() => console.log('MongoDB connected')) check.
            client.admin.command("ping")

            self._col = client[db_name][_COLLECTION_NAME]
            self._col.create_index("product_name")
            logger.info("FastLcaCache: connected to '%s.%s'", db_name, _COLLECTION_NAME)
        except Exception as exc:
            logger.warning("FastLcaCache: connection failed — cache disabled: %s", exc)
            self._col = None
        return self._col

    def lookup(
        self,
        product_name: str,
        embedding: list,
        req_amount: float = 1.0,
        req_unit: str = "tonne",
    ) -> Optional[dict]:
        """Return the best cached result scaled to (req_amount, req_unit), or None."""
        col = self._get_collection()
        if col is None:
            return None
        try:
            projection = {
                "_id": 0,
                "product_name": 1,
                "embedding": 1,
                "answer": 1,
                "processed_json": 1,
                "lcia_table": 1,
                "intent_params": 1,
                "mass_amount": 1,
                "mass_unit": 1,
            }
            best_score, best_doc = 0.0, None
            for doc in col.find({}, projection):
                stored = doc.get("embedding")
                if not stored:
                    continue
                score = _cosine_similarity(embedding, stored)
                if score > best_score:
                    best_score, best_doc = score, doc

            if best_score >= _SIMILARITY_THRESHOLD and best_doc is not None:
                logger.info(
                    "FastLcaCache HIT: '%s' → '%s' (score=%.3f)",
                    product_name,
                    best_doc.get("product_name", "?"),
                    best_score,
                )
                cached_kg = _normalize_to_kg(best_doc.get("mass_amount", 1.0), best_doc.get("mass_unit", "tonne"))
                req_kg = _normalize_to_kg(req_amount, req_unit)
                scale_factor = req_kg / cached_kg if cached_kg != 0.0 else 1.0
                if abs(scale_factor - 1.0) > 1e-9:
                    return _scale_lca_result(best_doc, scale_factor, req_amount, req_unit)
                return best_doc

            logger.info("FastLcaCache MISS: '%s' (best=%.3f)", product_name, best_score)
            return None
        except Exception as exc:
            logger.warning("FastLcaCache.lookup error: %s", exc)
            return None

    def store(
        self,
        product_name: str,
        embedding: list,
        answer: str,
        processed_json: Optional[str],
        lcia_table: Optional[dict],
        intent_params: dict,
        mass_amount: float = 1.0,
        mass_unit: str = "tonne",
    ) -> None:
        """Insert or update a cached LCA result, including the computed functional unit."""
        col = self._get_collection()
        if col is None:
            return
        try:
            col.update_one(
                {"product_name": product_name},
                {
                    "$set": {
                        "product_name":   product_name,
                        "embedding":      embedding,
                        "answer":         answer,
                        "processed_json": processed_json,
                        "lcia_table":     lcia_table,
                        "intent_params":  intent_params,
                        "mass_amount":    mass_amount,
                        "mass_unit":      mass_unit,
                        "cached_at":      datetime.now(timezone.utc).isoformat(),
                    }
                },
                upsert=True,
            )
            logger.info("FastLcaCache: stored result for '%s'", product_name)
        except Exception as exc:
            logger.warning("FastLcaCache.store error: %s", exc)


# Module-level singleton
_cache = _FastLcaCache()


def cache_lookup(
    product_name: str,
    embedding: list,
    req_amount: float = 1.0,
    req_unit: str = "tonne",
) -> Optional[dict]:
    """
    Look up a cached LCA result by full-query embedding similarity.

    *embedding* must be the vector of the complete user query (not just the
    product name) so that queries for the same product under different
    scenarios do not collide.  Returns a result dict scaled to
    (req_amount, req_unit); if the cached functional unit differs, all impact
    values are linearly scaled before returning.
    """
    return _cache.lookup(product_name, embedding, req_amount, req_unit)


def cache_store(
    product_name: str,
    embedding: list,
    answer: str,
    processed_json: Optional[str],
    lcia_table: Optional[dict],
    intent_params: dict,
    mass_amount: float = 1.0,
    mass_unit: str = "tonne",
) -> None:
    """Insert or update a cached LCA result.

    *embedding* must be the vector of the complete user query so that future
    similarity lookups match on query intent, not just product name.
    """
    _cache.store(
        product_name, embedding, answer, processed_json, lcia_table,
        intent_params, mass_amount, mass_unit,
    )
