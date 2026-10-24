import os
import importlib
import sys
import logging
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LCA_SOURCE_DIR = REPO_ROOT / "bw2_sources" / "Big Mushroom" / "Small Mushroom"
ECO_DIR = Path(os.getenv("BRIGHTWAY_DATA_DIR") or REPO_ROOT / "brightway_databases" / "brightway2_ecoinvent")

logger = logging.getLogger(__name__)


def _ensure_lca_source_path():
    source_path = str(LCA_SOURCE_DIR)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)


def _refresh_engine_modules():
    # Unified_LCA_Engine reads ENGINE_TYPE at import time; force a fresh import per run.
    for module_name in ["Unified_LCA_Engine", "lca_calculator"]:
        if module_name in sys.modules:
            sys.modules.pop(module_name, None)


def _initialize_ecoinvent_if_needed():
    try:
        from .prebuild_bw2db import preinstall_ecoinvent_database

        data_dir = preinstall_ecoinvent_database()
        if data_dir:
            data_dir = Path(data_dir)
            os.environ["BRIGHTWAY2_DIR"] = str(data_dir)
            from bw2data import databases, projects

            if "ecoinvent" not in projects:
                raise RuntimeError("Restored Brightway snapshot has no 'ecoinvent' project")
            projects.set_current("ecoinvent")
            if "ecoinvent" not in databases:
                raise RuntimeError("Restored 'ecoinvent' project has no 'ecoinvent' database")
            return True, data_dir
        return False, "Could not prepare the Brightway cache; check server logs"
    except Exception as exc:
        return False, str(exc)


def connect_and_calculate_lca(engine_arg, json_path, background=False):
    if not os.path.isfile(json_path):
        return {"error": "json_not_found", "message": f"JSON file not found: {json_path}"}

    if engine_arg != "ecoinvent":
        return {
            "error": "unsupported_engine",
            "message": "Only the Ecoinvent engine is enabled in the current Azure runtime.",
        }

    _ensure_lca_source_path()
    os.environ["ENGINE_TYPE"] = "ecoinvent"

    ok_db, data_dir = _initialize_ecoinvent_if_needed()
    if not ok_db:
        return {"error": "db_not_initialized", "message": data_dir}

    global ECO_DIR
    ECO_DIR = data_dir
    os.environ["BRIGHTWAY2_DIR"] = str(ECO_DIR)

    try:
        _refresh_engine_modules()
        calculator_module = importlib.import_module("lca_calculator")
        run_fn = getattr(calculator_module, "run_ecoinvent_calculation")
        result = run_fn(json_path)

        if result is None:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": "Ecoinvent calculation failed in lca_calculator",
            }

        return {
            "returncode": 0,
            "stdout": "Ecoinvent calculation completed successfully",
            "stderr": "",
            "result": result,
        }
    except Exception as exc:
        logger.exception("Ecoinvent calculation failed")
        return {"returncode": -1, "stdout": "", "stderr": str(exc)}


def start_lca_calculation(choice, json_path, background=False):
    if choice == "1":
        return {
            "error": "unsupported_choice",
            "message": "Tiangong execution is disabled in the current single-environment deployment.",
        }
    elif choice == "2":
        return connect_and_calculate_lca('ecoinvent', json_path, background)
    elif choice == "3":
        out2 = connect_and_calculate_lca('ecoinvent', json_path, False)
        return {"tiangong": {"error": "unsupported_choice", "message": "Tiangong execution is disabled in the current single-environment deployment."}, "ecoinvent": out2}
    else:
        return {"error": "invalid_choice", "message": "choice must be '1', '2' or '3'"}


# ─── Lightweight MongoDB reference-product lookup for template validation ──

_bw2_ref_product_index: dict = None  # {name_lower: {name, reference_product, unit, location}}
_bw2_ref_product_snapshot_id = None
_bw2_ref_product_index_lock = threading.Lock()


def _build_bw2_ref_product_index(snapshot_id: str) -> dict:
    """
    Build an in-memory name→metadata index from MongoDB activity metadata.
    Cached at module level after the first successful build.
    Returns empty dict on any failure so callers degrade gracefully.
    """
    try:
        from .lca_activity_store import iter_activity_metadata

        index: dict = {}
        for act in iter_activity_metadata(snapshot_id):
            name = act.get("name", "")
            if name:
                index.setdefault(name.lower(), {
                    "name": name,
                    "reference_product": act.get("reference_product", "") or "",
                    "unit": act.get("unit", "") or "",
                    "location": act.get("location", "") or "",
                })

        logger.info("MongoDB reference-product index built: %d activities", len(index))
        return index
    except Exception as exc:
        logger.warning(
            "Could not build MongoDB reference-product index "
            "(template validation will use LLM-only mode): %s", exc,
            exc_info=True,
        )
        return {}


def _get_bw2_ref_product_index() -> dict:
    global _bw2_ref_product_index, _bw2_ref_product_snapshot_id
    try:
        from .lca_activity_store import get_active_activity_snapshot

        snapshot_id = get_active_activity_snapshot()["snapshot_id"]
        if _bw2_ref_product_snapshot_id == snapshot_id and _bw2_ref_product_index:
            return _bw2_ref_product_index

        with _bw2_ref_product_index_lock:
            if _bw2_ref_product_snapshot_id != snapshot_id or not _bw2_ref_product_index:
                index = _build_bw2_ref_product_index(snapshot_id)
                if not index:
                    _bw2_ref_product_index = None
                    _bw2_ref_product_snapshot_id = None
                    return None
                _bw2_ref_product_index = index
                _bw2_ref_product_snapshot_id = snapshot_id
        return _bw2_ref_product_index
    except Exception as exc:
        logger.warning("Could not refresh MongoDB activity metadata index: %s", exc, exc_info=True)
        return None


def lookup_activity_ref_products_batch(activity_names: list, n_close: int = 5) -> dict:
    """
    Return reference_product metadata for a list of ecoinvent activity names.

    Uses a MongoDB activity index (built lazily on first call, then cached for
    the lifetime of the process). Falls back to an empty dict when unavailable.

    Parameters
    ----------
    activity_names : list[str]
        Activity names to look up (verbatim strings from the LLM-generated template).
    n_close : int
        Maximum number of difflib fuzzy candidates to return per name.

    Returns
    -------
    dict
        {activity_name: {
            "reference_product": str,   # from the closest BW2 match
            "unit":             str,
            "close_matches":    [{"name": ..., "reference_product": ..., "unit": ...}, ...]
        }}
        Missing entries mean no match was found.  Returns {} if BW2 unavailable.
    """
    index = _get_bw2_ref_product_index()
    if not index:
        return {}

    from difflib import get_close_matches as _gcm

    all_names_lower = list(index.keys())

    results: dict = {}
    for name in activity_names:
        if not name:
            continue
        name_lower = name.lower()

        # Exact match — highest confidence, short-circuit fuzzy search
        if name_lower in index:
            entry = index[name_lower]
            results[name] = {
                "reference_product": entry["reference_product"],
                "unit": entry["unit"],
                "close_matches": [{
                    "name": entry["name"],
                    "reference_product": entry["reference_product"],
                    "unit": entry["unit"],
                }],
            }
            continue

        # Fuzzy match via difflib (no model required)
        close = _gcm(name_lower, all_names_lower, n=n_close, cutoff=0.45)
        close_entries = [
            {
                "name": index[c]["name"],
                "reference_product": index[c]["reference_product"],
                "unit": index[c]["unit"],
            }
            for c in close
        ]
        results[name] = {
            "reference_product": close_entries[0]["reference_product"] if close_entries else "",
            "unit": close_entries[0]["unit"] if close_entries else "",
            "close_matches": close_entries,
        }

    return results
