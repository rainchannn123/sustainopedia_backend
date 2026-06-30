import os
import importlib
import sys
import logging
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LCA_SOURCE_DIR = REPO_ROOT / "bw2_sources" / "Big Mushroom" / "Small Mushroom"
ECO_DIR = REPO_ROOT / "brightway_databases" / "brightway2_ecoinvent"

logger = logging.getLogger(__name__)


def _ensure_lca_source_path():
    source_path = str(LCA_SOURCE_DIR)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)


def _ensure_ecoinvent_directory():
    try:
        ECO_DIR.mkdir(parents=True, exist_ok=True)
        return True, f"Ecoinvent directory ready: {ECO_DIR}"
    except Exception as exc:
        return False, str(exc)


def _refresh_engine_modules():
    # Unified_LCA_Engine reads ENGINE_TYPE at import time; force a fresh import per run.
    for module_name in ["Unified_LCA_Engine", "lca_calculator"]:
        if module_name in sys.modules:
            sys.modules.pop(module_name, None)


def _initialize_ecoinvent_if_needed():
    ok_db, msg_db = _ensure_ecoinvent_directory()
    if not ok_db:
        return False, msg_db

    if ECO_DIR.exists() and any(ECO_DIR.iterdir()):
        return True, f"Ecoinvent database directory already present: {ECO_DIR}"

    try:
        init_module = importlib.import_module("init_database")
        init_fn = getattr(init_module, "init_databases")
        init_result = init_fn()
        if init_result:
            return True, "Ecoinvent database initialized successfully"
        return False, "Ecoinvent database initialization returned False"
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
    os.environ["BRIGHTWAY2_DIR"] = str(ECO_DIR)

    # ok_db, msg_db = _initialize_ecoinvent_if_needed()
    # if not ok_db:
    #     return {"error": "db_not_initialized", "message": msg_db}

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


# ─── Lightweight BW2 reference-product lookup for pre-calculation validation ──

_bw2_ref_product_index: dict = None  # {name_lower: {name, reference_product, unit, location}}


def _build_bw2_ref_product_index() -> dict:
    """
    Build an in-memory name→metadata index from the BW2 ecoinvent database.
    Uses bw2data directly (no SentenceTransformer — lightweight).
    Cached at module level after the first successful build.
    Returns empty dict on any failure so callers degrade gracefully.
    """
    try:
        _ensure_lca_source_path()
        os.environ["BRIGHTWAY2_DIR"] = str(ECO_DIR)

        # Import bw2data inside the function so that BRIGHTWAY2_DIR is set first.
        import importlib as _il
        bw2data_mod = _il.import_module("bw2data")

        # bw2data.ProjectManager reads BRIGHTWAY2_DIR only during __init__.
        # If the module was already cached in sys.modules (e.g., loaded by a
        # previous LCA calculation with a different BRIGHTWAY2_DIR), the
        # projects._base_data_dir attribute will be wrong and set_current()
        # will fail to find the "Ecoinvent" project.
        # Fix: override _base_data_dir directly on the existing singleton, then
        # re-open the projects registry database so has_project() queries the
        # right file.
        current_base = getattr(bw2data_mod.projects, "_base_data_dir", None)
        if current_base != ECO_DIR:
            logger.info(
                "bw2data projects._base_data_dir was '%s'; overriding to '%s'",
                current_base, ECO_DIR,
            )
            bw2data_mod.projects._base_data_dir = ECO_DIR
            # Re-point the projects registry SQLite file to the new location
            projects_db_path = ECO_DIR / "projects.db"
            try:
                bw2data_mod.projects.db.database.init(str(projects_db_path))
            except Exception as db_exc:
                logger.debug(
                    "projects.db reinit skipped (will proceed anyway): %s", db_exc
                )

        projects = bw2data_mod.projects
        Database = bw2data_mod.Database

        projects.set_current("ecoinvent")
        db = Database("ecoinvent")

        index: dict = {}
        for act in db:
            name = act.get("name", "") or ""
            if name:
                index[name.lower()] = {
                    "name": name,
                    "reference_product": act.get("reference product", "") or "",
                    "unit": act.get("unit", "") or "",
                    "location": act.get("location", "") or "",
                }

        logger.info("BW2 reference-product index built: %d activities", len(index))
        return index
    except Exception as exc:
        logger.warning(
            "Could not build BW2 reference-product index "
            "(template validation will use LLM-only mode): %s", exc,
            exc_info=True,
        )
        return {}


def lookup_activity_ref_products_batch(activity_names: list, n_close: int = 5) -> dict:
    """
    Return reference_product metadata for a list of ecoinvent activity names.

    Uses an in-memory BW2 index (built lazily on first call, then cached for the
    lifetime of the process).  Falls back gracefully to an empty dict when the
    BW2 ecoinvent database has not yet been initialised.

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
    global _bw2_ref_product_index
    if _bw2_ref_product_index is None:
        _bw2_ref_product_index = _build_bw2_ref_product_index()
        if not _bw2_ref_product_index:
            # Build failed — reset to None so the next request retries rather
            # than permanently returning empty results.
            _bw2_ref_product_index = None
            return {}

    if not _bw2_ref_product_index:
        return {}

    from difflib import get_close_matches as _gcm

    index = _bw2_ref_product_index
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