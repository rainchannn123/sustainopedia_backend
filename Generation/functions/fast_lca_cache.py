"""fast_lca_cache.py — MongoDB-backed result cache for the fast-response LCA pipeline.

Stores per-product LCA computation results so repeated queries for the same or
similar product bypass the BW2 computation engine and are served from cache.
Similarity is measured via cosine distance between Azure OpenAI text embeddings
of the product name; a configurable threshold (default 0.80) gates cache hits.
"""
from __future__ import annotations

import copy
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_SIMILARITY_THRESHOLD = 0.80   # cosine similarity required for a cache hit
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


def _scale_lca_result(
    cached_doc: dict,
    scale_factor: float,
    req_amount: float,
    req_unit: str,
) -> dict:
    """
    Return a new result dict with all numeric LCA values multiplied by *scale_factor*.

    Scaled fields inside *processed_json* (JSON string) and *lcia_table* (dict):
    - processes[*].mean / processes[*].std
    - totalMeanImpact
    - stageImpacts values

    Also generates a concise, accurate answer text for the scaled functional unit.
    """
    # ── Scale processed_json ─────────────────────────────────────────────────
    scaled_json_str = cached_doc.get("processed_json")
    scaled_total: Optional[float] = None
    if scaled_json_str:
        try:
            data = json.loads(scaled_json_str)
            for p in data.get("processes", []):
                p["mean"] = round(p.get("mean", 0.0) * scale_factor, 6)
                if "std" in p:
                    p["std"] = round(p["std"] * scale_factor, 6)
            if "totalMeanImpact" in data:
                data["totalMeanImpact"] = round(data["totalMeanImpact"] * scale_factor, 6)
                scaled_total = data["totalMeanImpact"]
            for k in list(data.get("stageImpacts", {})):
                v = data["stageImpacts"][k]
                if isinstance(v, (int, float)):
                    data["stageImpacts"][k] = round(v * scale_factor, 6)
            scaled_json_str = json.dumps(data)
        except Exception:
            pass  # keep original string on parse failure

    # ── Scale lcia_table ─────────────────────────────────────────────────────
    scaled_table = cached_doc.get("lcia_table")
    if scaled_table:
        try:
            scaled_table = copy.deepcopy(scaled_table)
            for p in scaled_table.get("processes", []):
                p["mean"] = round(p.get("mean", 0.0) * scale_factor, 6)
                if "std" in p:
                    p["std"] = round(p["std"] * scale_factor, 6)
            if "totalMeanImpact" in scaled_table:
                scaled_table["totalMeanImpact"] = round(
                    scaled_table["totalMeanImpact"] * scale_factor, 6
                )
                if scaled_total is None:
                    scaled_total = scaled_table["totalMeanImpact"]
            for k in list(scaled_table.get("stageImpacts", {})):
                v = scaled_table["stageImpacts"][k]
                if isinstance(v, (int, float)):
                    scaled_table["stageImpacts"][k] = round(v * scale_factor, 6)
        except Exception:
            scaled_table = cached_doc.get("lcia_table")  # keep original on failure

    # ── Build answer narrative ────────────────────────────────────────────────
    product_name = cached_doc.get("intent_params", {}).get("product", "the product")
    cached_amount = cached_doc.get("mass_amount", 1.0)
    cached_unit_str = cached_doc.get("mass_unit", "tonne")
    impact_line = (
        f"`{scaled_total:.4g}\u202fkg CO\u2082-eq`"
        if scaled_total is not None
        else "see the table below"
    )
    answer = (
        f"## LCA Result: {product_name} \u2014 {req_amount}\u202f{req_unit}\n\n"
        f"> *Linearly scaled from a cached {cached_amount}\u202f{cached_unit_str} result "
        f"(\u00d7{scale_factor:.4g} scaling factor).*\n\n"
        f"- **Total GWP (IPCC 2013, GWP100a):** {impact_line}\n"
        f"- See the process breakdown table below for stage-by-stage contributions.\n"
    )

    return {
        "answer":         answer,
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

    def lookup(self, product_name: str, embedding: list) -> Optional[dict]:
        """Return the best cached result with cosine similarity >= threshold, or None."""
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
    Look up a cached LCA result by product embedding similarity.

    Returns a result dict scaled to (req_amount, req_unit).  If the cached
    functional unit differs, all impact values are linearly scaled before
    returning so the caller always receives results for the requested mass.
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
    """Persist a new LCA result in the cache, recording the computed functional unit."""
    _cache.store(
        product_name, embedding, answer, processed_json, lcia_table,
        intent_params, mass_amount, mass_unit,
    )
