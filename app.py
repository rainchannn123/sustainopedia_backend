# Override system sqlite3 with pysqlite3-binary BEFORE any ChromaDB import.
# Azure App Service Linux ships SQLite < 3.35.0 which ChromaDB rejects.
import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass  # local dev — system SQLite is fine
# __import__('pysqlite3')
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import logging
import threading
import uuid
from datetime import datetime, timezone

from flask import Flask, request, jsonify
from Generation.query_transformation import generate_response_fast, generate_response_full_lca
from flask_cors import CORS

from dotenv import load_dotenv
load_dotenv(dotenv_path="./.env")

# ── JSON encoder: handle numpy scalars / arrays and datetime objects ───────────
# BW2 / numpy return float64, int64, ndarray etc. which Flask's default encoder
# cannot serialise → 500 with empty body on Azure.
from flask.json.provider import DefaultJSONProvider

class _SafeJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

app = Flask(__name__)
app.json_provider_class = _SafeJSONProvider
app.json = _SafeJSONProvider(app)
from dotenv import load_dotenv
load_dotenv('./.env') 
# explicitly allow all origins on API endpoints to satisfy frontend CORS requirements
# CORS(app)
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "").split(",")
CORS(app, resources={r"/api/*": {"origins": allowed_origins}})

# ── In-memory job store ────────────────────────────────────────────────────────
# Each job: { status, question, product, created_at, finished_at, result, error }
_jobs: dict = {}
_jobs_lock = threading.Lock()
_JOB_TTL_SECONDS = 3600  # expire jobs after 1 hour

# ── Per-job log capture ────────────────────────────────────────────────────────
# Each background job thread sets _thread_local.job_id so the handler below
# can route Python log records into that job's 'logs' list.  All original
# logging.info() / logging.error() calls continue to print to the shell via
# the StreamHandler added by basicConfig in query_transformation.py.
_thread_local = threading.local()

class _JobLogHandler(logging.Handler):
    """Routes log records into their job's log list for frontend display.

    Only records tagged with  extra={'frontend': True}  are forwarded.
    Usage in pipeline code:
        logging.info("My message", extra={"frontend": True})   # → shell + frontend
        logging.info("My message")                              # → shell only
    """
    def emit(self, record):
        if not getattr(record, 'frontend', False):
            return
        job_id = getattr(_thread_local, 'job_id', None)
        if not job_id:
            return
        level = 'error' if record.levelno >= logging.ERROR else ('warning' if record.levelno >= logging.WARNING else '')
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job['logs'].append({
                    'ts':      datetime.now(timezone.utc).isoformat(),
                    'message': record.getMessage(),
                    'level':   level
                })

_job_log_handler = _JobLogHandler(level=logging.INFO)
logging.getLogger().addHandler(_job_log_handler)


def _purge_stale_jobs():
    """Remove jobs older than _JOB_TTL_SECONDS. Call inside _jobs_lock."""
    now = datetime.now(timezone.utc)
    stale = [
        jid for jid, job in _jobs.items()
        if (now - datetime.fromisoformat(job['created_at'])).total_seconds() > _JOB_TTL_SECONDS
    ]
    for jid in stale:
        del _jobs[jid]


def _run_job_fast(job_id: str, question: str, product: str):
    """Background thread: execute LCA pipeline and store the result."""
    _thread_local.job_id = job_id
    with _jobs_lock:
        _jobs[job_id]['status'] = 'running'
    try:
        answer_pack = generate_response_fast(question, product)
        with _jobs_lock:
            _jobs[job_id].update({
                'status': 'done',
                'result': answer_pack,
                'finished_at': datetime.now(timezone.utc).isoformat()
            })
    except Exception as exc:
        app.logger.exception(f"Job {job_id} failed: {exc}")
        with _jobs_lock:
            _jobs[job_id].update({
                'status': 'error',
                'error': str(exc),
                'finished_at': datetime.now(timezone.utc).isoformat()
            })
    finally:
        _thread_local.job_id = None

def _run_job_full_lca(job_id: str, user_input_params: str, product: str):
    """Background thread: execute LCA pipeline and store the result."""
    _thread_local.job_id = job_id
    with _jobs_lock:
        _jobs[job_id]['status'] = 'running'
    try:
        answer_pack = generate_response_full_lca(user_input_params, product)
        with _jobs_lock:
            _jobs[job_id].update({
                'status': 'done',
                'result': answer_pack,
                'finished_at': datetime.now(timezone.utc).isoformat()
            })
    except Exception as exc:
        app.logger.exception(f"Job {job_id} failed: {exc}")
        with _jobs_lock:
            _jobs[job_id].update({
                'status': 'error',
                'error': str(exc),
                'finished_at': datetime.now(timezone.utc).isoformat()
            })
    finally:
        _thread_local.job_id = None
            
# ── Job endpoints ──────────────────────────────────────────────────────────────

@app.route('/api/jobs', methods=['POST'])
def create_job():
    """Submit an LCA query as a background job. Returns jobId immediately."""
    data = request.get_json(silent=True) or {}
    job_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    
    if data.get('mode', '') == 'fast':
        question = data.get('question', '')
        product  = data.get('product', 'Not Specified')
        
        with _jobs_lock:
            _purge_stale_jobs()
            _jobs[job_id] = {
                'status':     'pending',
                'question':   question,
                'product':    product,
                'created_at': created_at,
                'logs':       []
            }

        thread = threading.Thread(target=_run_job_fast, args=(job_id, question, product), daemon=True)
        thread.start()
        
    elif data.get('mode', '') == 'full_lca':
        user_input_params = data.get('question', '')
        product  = data.get('product', 'Not Specified')
        
        with _jobs_lock:
            _purge_stale_jobs()
            _jobs[job_id] = {
                'status':     'pending',
                'question':   user_input_params,
                'product':    product,
                'created_at': created_at,
                'logs':       []
            }

        thread = threading.Thread(target=_run_job_full_lca, args=(job_id, user_input_params, product), daemon=True)
        thread.start()
    
    else:
        return jsonify({'error': 'Invalid mode. Must be "fast" or "full_lca".'}), 400

    return jsonify({'jobId': job_id}), 202


@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job(job_id: str):
    """Poll job status. Returns status + answer_pack + accumulated logs."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify({'status': 'not_found'}), 404
        status = job['status']
        logs   = list(job.get('logs', []))
        result = job.get('result') if status == 'done' else None
        error  = job.get('error')  if status == 'error' else None

    if status == 'done':
        return jsonify({'status': 'done', 'answer_pack': result, 'logs': logs})
    if status == 'error':
        return jsonify({'status': 'error', 'error': error or 'Unknown error', 'logs': logs})
    return jsonify({'status': status, 'logs': logs})


# ── Legacy synchronous endpoint (kept for compatibility) ───────────────────────

# @app.route('/', methods=['GET'])
# def health():
#     # simple health check so curl on / doesn't 404
#     return jsonify({'status': 'ok'})

# @app.route('/api/query', methods=['POST'])
# def query():
#     try:
#         data = request.get_json()
#         question = data.get('question', '')
#         product = data.get('product', 'Not Specified')
#         answer_pack = generate_response(question, product)
#         return jsonify({'answer_pack': answer_pack})
#     except Exception as e:
#         app.logger.exception(f"Error in /api/query: {e}")
#         return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5052))  # Azure sets PORT automatically
    app.run(host="0.0.0.0", port=port, threaded=True)

#please note that for dev env, port should change to 5051 or any available port to avoid conflict with docker backend (app).