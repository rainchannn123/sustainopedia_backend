# import subprocess
# import os
# import json
# import shutil
# import logging

# logging.basicConfig(
#     level=logging.INFO,                     # ← change to logging.DEBUG when troubleshooting
#     format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S',
#     handlers=[logging.StreamHandler()]
# )

# logger = logging.getLogger(__name__)

# SCRIPT_PATH = "./bw2_sources/Big Mushroom/Small Mushroom/lca_calculator.py"
# DATABASE_INITIALIZE_PATH = "./bw2_sources/Big Mushroom/Small Mushroom/init_databases.py"

# TIANGONG_DIR = "./brightway_databases/brightway2_tiangong"
# ECO_DIR = "./brightway_databases/brightway2_ecoinvent"

# def install_micromamba_linux():
#     """Install latest micromamba for linux-64 in user home directory"""
#     import urllib.request
#     import tempfile

#     install_prefix = os.path.expanduser("~/micromamba")
#     bin_dir = os.path.join(install_prefix, "bin")
#     os.makedirs(bin_dir, exist_ok=True)
#     binary_path = os.path.join(bin_dir, "micromamba")

#     if os.path.isfile(binary_path):
#         logger.info(f"micromamba already exists at {binary_path} — skipping install")
#         return True, binary_path

#     logger.info(f"Installing micromamba into {install_prefix}")

#     try:
#         url = "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64"
        
#         # Use User-Agent to avoid any GitHub blocks
#         req = urllib.request.Request(
#             url,
#             headers={'User-Agent': 'Mozilla/5.0 (compatible; MicromambaInstaller/1.0)'}
#         )

#         with tempfile.NamedTemporaryFile(delete=False) as tmp:
#             tmp_path = tmp.name
#             with urllib.request.urlopen(req) as response, open(tmp_path, "wb") as out:
#                 shutil.copyfileobj(response, out)

#         # Move the downloaded binary to final location
#         shutil.move(tmp_path, binary_path)

#         if not os.path.isfile(binary_path):
#             raise FileNotFoundError("binary not saved")

#         os.chmod(binary_path, 0o755)

#         # Add to PATH for subprocesses
#         os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"

#         logger.info("micromamba installed successfully")
#         return True, binary_path

#     except Exception as e:
#         logger.exception("Failed to install micromamba")
#         return False, str(e)
    
# def get_conda_exe():
#     logger.debug("Searching for conda / micromamba executable...")
    
#     # 1. Via environment variable
#     conda = os.environ.get("CONDA_EXE")
#     if conda and os.path.isfile(conda):
#         logger.debug(f"Found via $CONDA_EXE: {conda}")
#         return conda

#     # 2. Via PATH
#     conda = shutil.which("conda")
#     if conda and os.path.isfile(conda):
#         logger.debug(f"Found via shutil.which: {conda}")
#         return conda

#     conda = shutil.which("micromamba")
#     if conda and os.path.isfile(conda):
#         logger.debug(f"Found micromamba via shutil.which: {conda}")
#         return conda

#     # 3. Known container locations
#     possible_locations = [
#         "/usr/local/bin/conda",
#         "/usr/local/bin/micromamba",
#         "/bin/micromamba",
#         "/usr/bin/micromamba",
#         os.path.expanduser("~/micromamba/bin/micromamba"),
#     ]
#     for loc in possible_locations:
#         if os.path.isfile(loc):
#             logger.debug(f"Found in known location: {loc}")
#             return loc

#     # ─── Auto-install micromamba if nothing was found ───
#     logger.info("No conda nor micromamba found → attempting to install micromamba ...")
    
#     install_ok, install_msg = install_micromamba_linux()
#     if install_ok:
#         # After installation, try to find it again
#         new_path = shutil.which("micromamba")
#         if new_path and os.path.isfile(new_path):
#             logger.info(f"micromamba successfully installed and found at: {new_path}")
#             return new_path
#         else:
#             # Sometimes PATH isn't updated in current process → try common location
#             common_loc = os.path.expanduser("~/micromamba/bin/micromamba")
#             if os.path.isfile(common_loc):
#                 logger.info(f"Found micromamba after install in home directory: {common_loc}")
#                 return common_loc

#     logger.warning("Automatic micromamba installation did not succeed or binary not found in PATH")
#     return None


# def check_conda_available():
#     logger.info("Checking for conda/micromamba availability...")
#     conda = get_conda_exe()
#     if not conda:
#         logger.error("conda/micromamba not found")
#         return False, "conda/micromamba not found in PATH or known locations"
#     logger.info(f"conda/micromamba located at: {conda}")
#     return True, conda


# def install_packages_in_env(env_name, packages):
#     logger.info(f"Installing {len(packages)} packages into environment '{env_name}'")
#     logger.debug(f"Package list: {', '.join(packages[:10])}{' ...' if len(packages) > 10 else ''}")

#     conda_exe = get_conda_exe()
#     if not conda_exe:
#         logger.error("Cannot install packages - conda/micromamba executable missing")
#         return False, "conda/micromamba executable not found"

#     cmd = [conda_exe, "run", "-n", env_name, "python", "-m", "pip", "install"] + packages
#     logger.debug(f"Executing pip install command: {' '.join(cmd)}")

#     res = _run_cmd(cmd)

#     if res.get("returncode", -1) != 0:
#         logger.error(f"Package installation failed in env '{env_name}' (rc={res.get('returncode')})")
#         if res.get("stderr"):
#             logger.error(f"stderr excerpt: {res['stderr'].strip()[:600]}...")
#         return False, f"pip install failed: {res.get('stderr', '')[:300]}..."
    
#     logger.info(f"Successfully installed packages in '{env_name}'")
#     return True, f"Installed packages in {env_name}"


# def check_env_exists(env_name, ensure_deps=True):
#     logger.info(f"Verifying environment '{env_name}' (ensure_deps={ensure_deps})")

#     ok, conda_path_or_msg = check_conda_available()
#     if not ok:
#         logger.error(f"Conda/micromamba unavailable: {conda_path_or_msg}")
#         return False, conda_path_or_msg

#     conda_exe = conda_path_or_msg

#     exists_ok, exists_msg = env_exists(env_name)
#     if not exists_ok:
#         logger.info(f"Environment '{env_name}' does not exist → creating new environment...")
#         created_ok, created_msg = create_conda_env(env_name)
#         if not created_ok:
#             logger.error(f"Failed to create environment '{env_name}': {created_msg}")
#             return False, f"Environment creation failed: {created_msg}"

#     if ensure_deps:
#         logger.info(f"Installing/ensuring dependencies for '{env_name}'...")
#         if env_name.lower() == "lca_ecoinvent":
#             pkgs = [
#                 "brightway2==2.4.7", "bw2analyzer==0.10", "bw2calc==1.8.2",
#                 "bw2data==3.6.6", "bw2io==0.8.12", "bw2parameters==0.7",
#                 "sentence-transformers==5.0.0", "numpy==1.24.3", "pandas", "scipy",
#                 "matplotlib", "seaborn", "requests", "tqdm", "openpyxl",
#                 "azure-identity", "azure-storage-blob"
#             ]
#         elif env_name.lower() == "lca_tiangong":
#             pkgs = [
#                 "brightway2==2.4.7", "bw2analyzer>=0.10", "bw2calc>=1.8.2",
#                 "bw2data==3.6.6", "bw2io>=0.8.7", "bw2parameters>=0.6.6",
#                 "sentence-transformers>=2.2.2", "numpy", "pandas", "scipy",
#                 "matplotlib", "seaborn", "requests", "tqdm", "openpyxl",
#                 "azure-identity", "azure-storage-blob"
#             ]
#         else:
#             logger.info(f"No predefined dependency list for '{env_name}'")
#             return True, f"Environment '{env_name}' exists (no deps managed)"

#         ok_inst, msg_inst = install_packages_in_env(env_name, pkgs)
#         if not ok_inst:
#             logger.error(f"Dependency installation failed: {msg_inst}")
#             return False, msg_inst

#         logger.info(f"Dependencies successfully ensured for '{env_name}'")
#         return True, f"Environment '{env_name}' ready with dependencies"

#     logger.info(f"Environment '{env_name}' exists (dependency check skipped)")
#     return True, f"Environment '{env_name}' exists"


# def check_db_initialized(env_name, engine_arg):
#     logger.info(f"Checking database status for engine '{engine_arg}' in env '{env_name}'")

#     target = TIANGONG_DIR if engine_arg == "tiangong" else ECO_DIR if engine_arg == "ecoinvent" else None
#     if not target:
#         logger.error(f"Unknown engine type: {engine_arg}")
#         return False, f"Unknown engine: {engine_arg}"

#     if os.path.isdir(target):
#         logger.info(f"Database directory already present: {target}")
#         return True, f"{engine_arg} database already exists"

#     logger.info(f"Database directory missing ({target}) → starting initialization...")

#     conda_exe = get_conda_exe()
#     if not conda_exe:
#         logger.error("Cannot initialize database - conda/micromamba missing")
#         return False, "conda/micromamba not found"

#     cmd = [conda_exe, "run", "-n", env_name, "python", DATABASE_INITIALIZE_PATH, engine_arg]
#     logger.debug(f"Database initialization command: {' '.join(cmd)}")

#     res = _run_cmd(cmd)

#     if res.get("returncode", -1) != 0:
#         logger.error(f"Database initialization failed (return code {res.get('returncode')})")
#         if res.get("stderr"):
#             logger.error(f"stderr excerpt: {res['stderr'].strip()[:600]}...")
#         return False, f"Database initialization failed: {res.get('stderr', '')[:300]}..."

#     logger.info(f"Database '{engine_arg}' successfully initialized")
#     return True, f"Successfully initialized {engine_arg} database"


# def _run_cmd(cmd):
#     cmd_str = ' '.join(cmd) if isinstance(cmd, (list, tuple)) else cmd
#     logger.debug(f"Executing subprocess: {cmd_str}")

#     try:
#         p = subprocess.run(
#             cmd,
#             capture_output=True,
#             text=True,
#             encoding='utf-8',
#             errors='replace',
#             timeout=7200,               # 2 hours - adjust based on expected max duration
#             env=os.environ.copy()       # ensure child sees same env vars
#         )
#         logger.debug(f"Subprocess completed - returncode={p.returncode}")
#         if p.returncode != 0:
#             logger.warning(f"Non-zero return code detected: {p.returncode}")
#         return {
#             "returncode": p.returncode,
#             "stdout": p.stdout,
#             "stderr": p.stderr
#         }
#     except subprocess.TimeoutExpired:
#         logger.error(f"Subprocess timed out after 7200 seconds: {cmd_str}")
#         return {"returncode": -9, "stdout": "", "stderr": "Timeout after 120 minutes"}
#     except Exception as e:
#         logger.exception(f"Failed to run subprocess: {cmd_str}")
#         return {"returncode": -1, "stdout": "", "stderr": str(e)}


# def env_exists(env_name):
#     logger.debug(f"Checking existence of conda environment: {env_name}")
#     conda_exe = get_conda_exe()
#     if not conda_exe:
#         logger.error("Cannot check env existence - conda not found")
#         return False, "conda not found"

#     cmd = [conda_exe, "env", "list", "--json"]
#     res = _run_cmd(cmd)

#     if res.get("returncode", -1) != 0:
#         logger.warning("JSON env list failed → falling back to text parsing")
#         res2 = _run_cmd([conda_exe, "info", "--envs"])
#         if res2.get("returncode", -1) != 0:
#             logger.error(f"Fallback env list also failed: {res2.get('stderr','')}")
#             return False, "failed_to_list_envs"

#         out = res2.get("stdout", "")
#         for line in out.splitlines():
#             line = line.strip()
#             if not line or '*' in line:  # skip active marker
#                 continue
#             parts = line.split()
#             if len(parts) < 2:
#                 continue
#             path = parts[-1]
#             if os.path.basename(path) == env_name or path.endswith(os.path.join("envs", env_name)):
#                 logger.debug(f"Found environment via text parse: {path}")
#                 return True, path
#         logger.debug("Environment not found via text parse")
#         return False, "env_not_found"

#     try:
#         data = json.loads(res.get("stdout", "{}"))
#         envs = data.get("envs", [])
#         for p in envs:
#             if os.path.basename(p) == env_name or p.endswith(os.path.join("envs", env_name)):
#                 logger.debug(f"Found environment via JSON: {p}")
#                 return True, p
#         logger.debug("Environment not found in JSON output")
#         return False, "env_not_found"
#     except Exception as e:
#         logger.error(f"Failed to parse conda env list JSON: {e}")
#         return False, f"parse_error: {e}"


# def create_conda_env(env_name, python_version="3.10"):
#     logger.info(f"Creating new conda environment: {env_name} (python={python_version})")
#     conda_exe = get_conda_exe()
#     if not conda_exe:
#         logger.error("Cannot create environment - conda not found")
#         return False, "conda not found"

#     cmd = [conda_exe, "create", "-n", env_name, f"python={python_version}", "-y", "--quiet"]
#     logger.debug(f"Create environment command: {' '.join(cmd)}")

#     res = _run_cmd(cmd)
#     if res.get("returncode", -1) != 0:
#         logger.error(f"Environment creation failed (rc={res.get('returncode')})")
#         if res.get("stderr"):
#             logger.error(f"stderr excerpt: {res['stderr'].strip()[:600]}...")
#         return False, f"conda create failed: {res.get('stderr', '')[:300]}..."

#     logger.info(f"Successfully created environment '{env_name}'")
#     return True, f"Environment {env_name} created"


# def connect_and_calculate_lca(env_name, engine_arg, json_path, background=False):
#     logger.info(f"Starting LCA calculation | env={env_name} | engine={engine_arg} | input={json_path} | background={background}")

#     if not os.path.isfile(json_path):
#         logger.error(f"Input JSON file not found: {json_path}")
#         return {"error": "json_not_found", "message": f"JSON file not found: {json_path}"}

#     ok, msg = check_conda_available()
#     if not ok:
#         logger.error(f"Conda unavailable: {msg}")
#         return {"error": "conda_not_found", "message": msg}

#     ok_env, msg_env = check_env_exists(env_name)
#     if not ok_env:
#         logger.error(f"Environment check failed: {msg_env}")
#         return {"error": "env_not_found", "message": msg_env}

#     ok_db, msg_db = check_db_initialized(env_name, engine_arg)
#     if not ok_db:
#         logger.error(f"Database initialization check failed: {msg_db}")
#         return {"error": "db_not_initialized", "message": msg_db}

#     conda_exe = get_conda_exe()
#     cmd = [conda_exe, "run", "-n", env_name, "--no-capture-output", "python", SCRIPT_PATH, engine_arg, json_path]
#     logger.info("Launching LCA calculation subprocess...")
#     logger.debug(f"Full calculation command: {' '.join(cmd)}")

#     res = _run_cmd(cmd)
#     print(f"[DEBUG] connect_and_calculate_lca(): Synchronous execution completed")

#     return res


# def start_lca_calculation(choice, json_path, background=False):
#     logger.info(f"start_lca_calculation called | choice={choice} | input={json_path} | background={background}")

#     if choice == "1":
#         logger.info("Selected Tiangong calculation")
#         return connect_and_calculate_lca('lca_tiangong', 'tiangong', json_path, background)
#     elif choice == "2":
#         logger.info("Selected Ecoinvent calculation")
#         return connect_and_calculate_lca('lca_ecoinvent', 'ecoinvent', json_path, background)
#     elif choice == "3":
#         logger.info("Selected both Tiangong + Ecoinvent calculations")
#         out1 = connect_and_calculate_lca('lca_tiangong', 'tiangong', json_path, False)
#         out2 = connect_and_calculate_lca('lca_ecoinvent', 'ecoinvent', json_path, False)
#         return {"tiangong": out1, "ecoinvent": out2}
#     else:
#         logger.error(f"Invalid choice received: {choice}")
#         return {"error": "invalid_choice", "message": "choice must be '1', '2' or '3'"}


import os
import importlib
import sys
import logging
from pathlib import Path

# Set up logging at the module level
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s [%(levelname)s] %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )
# logger = logging.getLogger(__name__)

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