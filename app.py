# Override system sqlite3 with pysqlite3-binary BEFORE any ChromaDB import.
# Azure App Service Linux ships SQLite < 3.35.0 which ChromaDB rejects.
import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass  # local dev — system SQLite is fine

import os
import threading
import uuid
from datetime import datetime, timezone

from flask import Flask, request, jsonify
from Generation.query_transformation import generate_response
from flask_cors import CORS
app = Flask(__name__)
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


def _purge_stale_jobs():
    """Remove jobs older than _JOB_TTL_SECONDS. Call inside _jobs_lock."""
    now = datetime.now(timezone.utc)
    stale = [
        jid for jid, job in _jobs.items()
        if (now - datetime.fromisoformat(job['created_at'])).total_seconds() > _JOB_TTL_SECONDS
    ]
    for jid in stale:
        del _jobs[jid]


def _run_job(job_id: str, question: str, product: str, mode: str = 'thinking'):
    """Background thread: execute LCA pipeline and store the result."""
    with _jobs_lock:
        _jobs[job_id]['status'] = 'running'
    try:
        answer_pack = generate_response(question, product, mode)
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


# ── Job endpoints ──────────────────────────────────────────────────────────────

@app.route('/api/jobs', methods=['POST'])
def create_job():
    """Submit an LCA query as a background job. Returns jobId immediately."""
    data = request.get_json(silent=True) or {}
    question = data.get('question', '')
    product  = data.get('product', 'Not Specified')
    mode     = data.get('mode', 'thinking')  # 'thinking' (full LCIA) or 'fast' (RAG only)

    job_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    with _jobs_lock:
        _purge_stale_jobs()
        _jobs[job_id] = {
            'status':     'pending',
            'question':   question,
            'product':    product,
            'created_at': created_at,
        }

    thread = threading.Thread(target=_run_job, args=(job_id, question, product, mode), daemon=True)
    thread.start()

    return jsonify({'jobId': job_id}), 202


@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job(job_id: str):
    """Poll job status. Returns status + answer_pack when done."""
    with _jobs_lock:
        job = _jobs.get(job_id)

    if not job:
        return jsonify({'status': 'not_found'}), 404

    if job['status'] == 'done':
        return jsonify({'status': 'done', 'answer_pack': job['result']})

    if job['status'] == 'error':
        return jsonify({'status': 'error', 'error': job.get('error', 'Unknown error')})

    # pending or running
    return jsonify({'status': job['status']})


# ── Legacy synchronous endpoint (kept for compatibility) ───────────────────────

# @app.route('/', methods=['GET'])
# def health():
#     # simple health check so curl on / doesn't 404
#     return jsonify({'status': 'ok'})

@app.route('/api/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        question = data.get('question', '')
        product = data.get('product', 'Not Specified')
        answer_pack = generate_response(question, product)
        return jsonify({'answer_pack': answer_pack})
    except Exception as e:
        app.logger.exception(f"Error in /api/query: {e}")
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5052))  # Azure sets PORT automatically
    app.run(host="0.0.0.0", port=port, threaded=True)

#please note that for dev env, port should change to 5051 or any available port to avoid conflict with docker backend (app).