# SKILL.md — Sustainopedia Backend (LLM Reference)

> **Purpose:** Authoritative map of this codebase for LLM agents. Read this before editing any file.  
> **Runtime:** Python 3.10+, Flask, Brightway2 (BW2), LangChain, ChromaDB, Azure OpenAI.  
> **Working directory convention:** all path literals in the code are relative to `sustainopedia_backend/` (the repo root). Always resolve absolute paths from that root.

---

## 1. Repository Layout

```
sustainopedia_backend/
├── app.py                          # Flask entry point — HTTP API + async job engine
├── requirements.txt                # Python dependencies
├── docker-compose.yml              # Local containerised stack
├── .env                            # Secrets (AZURE_API_KEY, LANGCHAIN_API_KEY, MONGODB_URI, ALLOWED_ORIGINS)
│
├── Generation/
│   ├── query_transformation.py     # Pipeline orchestrator — called by app.py
│   └── functions/
│       ├── after_bw2.py            # Post-processing: BW2 results → structured data + reports
│       ├── non_RAG_methods.py      # Pre-processing: user query → BW2 input JSON (LLM-driven)
│       ├── bw2_calculation.py      # BW2 engine wrapper — runs ecoinvent LCA calculations
│       ├── vectorstore.py          # ChromaDB build/load facade (re-exports lca_knowledge_pipeline)
│       ├── lca_knowledge_pipeline.py  # Parent-child PDF ingestion + LcaKnowledgeRetriever
│       ├── RAG_methods.py          # RAG utilities: query decomposition, RRF, doc joining
│       ├── intent_classifier.py    # (commented-out) intent + parameter extraction
│       └── prebuild_bw2db.py       # (commented-out) pre-build ecoinvent DB on deploy
│
├── bw2_input_templates/            # LLM-generated BW2 input JSONs  → naming: <Product>_Ecoinvent_<YYYYMMDDHHMMSS>.json
├── bw2_results/                    # BW2 output JSONs               → naming: <same>_output.json
│
├── chroma_db_lca_knowledge/        # Vectorstore: peer-reviewed LCA papers (parent-child indexed)
├── chroma_db_eco_activity/         # Vectorstore: ecoinvent activity names
├── chroma_db_eco_methods/          # Vectorstore: ecoinvent LCIA methods
├── chroma_db_tg_methods/           # Vectorstore: Tiangong LCIA methods
│
├── brightway_databases/
│   └── brightway2_ecoinvent/       # BW2 project files (ecoinvent database)
│
├── _additionals/
│   └── methods/
│       ├── EcoInvent_methods.txt   # Source for chroma_db_eco_methods
│       ├── tiangong_methods.txt    # Source for chroma_db_tg_methods
│       └── Ecoinvent_activity.txt  # Source for chroma_db_eco_activity
├── _raw knowledge pdfs/            # Source PDFs ingested into chroma_db_lca_knowledge
└── _models/
    └── models--sentence-transformers--all-MiniLM-L6-v2/  # Local HuggingFace embedding model
```

---

## 2. Entry Point — `app.py`

### Global Configuration
| Item | Value |
|---|---|
| Framework | Flask with `_SafeJSONProvider` (handles numpy int64/float64/ndarray + datetime) |
| CORS | Allowed origins read from `ALLOWED_ORIGINS` env var (comma-separated) |
| Job TTL | `_JOB_TTL_SECONDS = 3600` — jobs auto-purged after 1 hour |
| In-memory store | `_jobs: dict` (keyed by UUID), `_cancel_events: dict` (keyed by UUID) |
| Thread-local | `_thread_local.job_id` — routes log records into the correct job's log list |

### Classes
```python
class _SafeJSONProvider(DefaultJSONProvider)
    # Overrides .default() to serialize numpy scalar/array types and datetime to JSON.

class _JobLogHandler(logging.Handler)
    # emit(record): only forwards records where getattr(record, 'frontend', False) is True.
    # Routes those records into _jobs[job_id]['logs'] for frontend display.
    # Usage in pipeline: logging.info("msg", extra={"frontend": True})
```

### Background Worker Functions
```python
def _purge_stale_jobs() -> None
    # Remove jobs older than _JOB_TTL_SECONDS from _jobs + _cancel_events.
    # Must be called inside _jobs_lock.

def _run_job_fast(job_id: str, question: str, product: str) -> None
    # Background thread. Calls generate_response_fast(question, product).
    # Sets job status: running → done | error | cancelled.

def _run_job_full_lca(job_id: str, user_input_params: str, product: str) -> None
    # Background thread. Calls generate_response_full_lca(user_input_params, product).
    # Sets job status: running → done | error | cancelled.
```

### API Endpoints
| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/api/jobs` | None | Submit job. Body: `{mode: "fast"|"full_lca", question: str, product: str}`. Returns `{jobId}` (202). |
| `GET` | `/api/jobs/<job_id>` | None | Poll job. Returns `{status, answer_pack?, error?, logs[]}`. |
| `DELETE` | `/api/jobs/<job_id>` | None | Request cancellation. Sets cancel event; worker aborts after current step. |

### `answer_pack` Shape (returned when status = `"done"`)
```json
{
  "answer": "markdown string",
  "markdown": { "<product>": "markdown report string" },
  "processed_json": { "<product>": { ...lcia_data... } },
  "lcia_table": "markdown table string",
  "intent_params": { "intent": "fast|computation|general|literature", "product": "..." }
}
```

---

## 3. Pipeline Orchestrator — `Generation/query_transformation.py`

### Module-Level Singletons (initialised on import)
```python
gpt5_client     # AzureChatOpenAI — deployment "sustainopedia-gpt-5", api_version "2024-12-01-preview"
embeddings      # AzureOpenAIEmbeddings — deployment "text-embedding-3-large", api_version "2024-02-01"
LCI_DATABASE    # str — "Ecoinvent" (default) | "Tiangong"
```
**Critical:** These are singletons. Do **not** re-instantiate them inside function bodies.

### Functions
```python
def generate_response_fast(question: str, product: str) -> dict
    # Fast path: load chroma_db_lca_knowledge only → RAG retrieval → single LLM call → answer.
    # Intent classification is rule-based (no LLM). Does NOT invoke BW2.
    # Returns answer_pack with answer str; markdown/processed_json/lcia_table are None.

def generate_response_full_lca(user_input_params: str, product: str) -> dict
    # Full LCA pipeline (see §4 for detailed flow).
    # Loads all 4 vectorstores → LCIA method selection → activity retrieval →
    # BW2 input JSON generation → BW2 calculation → post-processing → narrative.
    # Returns complete answer_pack.
```

### Full LCA Pipeline Execution Order
1. `infer_product_from_query` — if `product` is empty  
2. `input_q_to_json` — parse form text to dict  
3. Load all 4 vectorstores (`load_vectorstores` or `build_and_save_vectorstores`)  
4. `Get_LCIA_methods` — select 1–3 LCIA methods  
5. `query_decomposition_retrieval` — retrieve supplementary LCA knowledge  
6. `generate_list_of_activities` — retrieve ecoinvent activity candidates per stage  
7. `generate_response_bw2_input` — LLM generates BW2 input JSON + mass flow validation  
8. `start_lca_calculation` — BW2 runs Monte Carlo / deterministic LCA  
9. `final_mode_output_bw2_included` — post-process results → reports + narrative  

---

## 4. Pre-Processing — `Generation/functions/non_RAG_methods.py`

> Imports `_validate_mass_flow_chain` from `after_bw2.py` (circular-safe: after_bw2 does not import non_RAG).

### Functions
```python
def input_q_to_json(question: str) -> dict
    # Parse "Key: Value\nKey: Value" multiline form text into a dict.
    # Used to convert the frontend's structured LCA form submission.

def _word_to_num_en(text: str) -> float | None
    # Convert English number-word phrase to float ("two hundred fifty" → 250.0).
    # Returns None if unparseable.

def _word_to_num_zh(text: str) -> float | None
    # Convert Chinese numeral string to float ("两百五十" → 250.0).
    # Returns None if unparseable.

def extract_functional_unit(question: str) -> tuple[float, str, float]
    # Returns (display_amount, display_unit, scale_factor).
    # scale_factor = (amount_in_kg) / 1000 — multiplier for BW2 results (computed at 1-tonne reference).
    # Handles Arabic numerals, English words, Chinese characters, and "per <unit>" patterns.
    # Default: (1.0, 'tonne', 1.0) when no quantity detected.

def infer_product_from_query(question: str, gpt5_client) -> str
    # LLM call: extract primary product/material name from query.
    # Returns short noun phrase (e.g., "crude steel"). Fallback: "Unknown Product".

def conduct_lca_anaylsis(question: str, gpt5_client) -> str
    # LLM call: decides whether an LCA is needed and replies accordingly.
    # Returns "Yes" (LCA needed) or a short natural-language answer (LCA not needed).

def Get_LCIA_methods(question: str, gpt5_client, retriever_lca_methods) -> list[Document]
    # LLM call: select 1–3 LCIA methods from hardcoded list + extract retrieval keywords.
    # Then retrieves matching method docs from vectorstore.
    # Returns list of LangChain Document objects.

def join_activities_by_stage(activities_by_stage: dict) -> str
    # Format {stage: [Document, ...]} dict into a flat string for LLM prompt injection.

def generate_list_of_activities(gpt5_client, retriever_eco_activity, product, user_input_params) -> dict
    # LLM generates stage descriptions based on system_boundary in user_input_params.
    # Then retrieves ecoinvent activity docs per stage from chroma_db_eco_activity.
    # Returns {stage_name: [Document, ...]} dict.

def generate_response_bw2_input(
    question, product, context, methods, activities_by_stage,
    gpt5_client, retriever_eco_activity,
    display_amount=1.0, display_unit='tonne', scale_factor=1.0,
    n_simulations=25
) -> dict
    # Core LLM call that generates the BW2 input JSON.
    # Uses final_template_1st (ChatPromptTemplate) with embedded 3-step mass flow procedure.
    # Post-generation: calls _validate_mass_flow_chain on functional_unit.
    # Saves JSON to bw2_input_templates/<Product>_Ecoinvent_<timestamp>.json.
    # Returns the parsed dict (results_dict) with mass_flow_audit embedded.
```

### BW2 Input JSON Schema (`results_dict`)
```json
{
  "Project": "Ecoinvent",
  "method": ["IPCC 2021", "climate change", "global warming potential (GWP100)"],
  "system_boundary": "cradle-to-gate",
  "n_simulations": 25,
  "mass_flow_audit": { "status": "pass|warning|gap|skipped", "overall_yield": 0.006, "transitions": [...] },
  "functional_unit": [
    {
      "activity_name": "copper mine operation and beneficiation, sulfide ore",
      "amount": 1133.787,        // = mass_out_kg — ecoinvent reference product (OUTPUT)
      "unit": "kilogram",
      "location": "GLO",
      "stage": "raw material extraction",
      "mass_in_kg": 157470.397,  // ore INPUT — for mass balance tracking only
      "mass_out_kg": 1133.787,   // concentrate OUTPUT — equals amount for mass-bearing activities
      "recovery_rate": 0.0072    // mass_out_kg / mass_in_kg
    }
  ]
}
```
**Critical mass flow rule:** `amount` **must equal** `mass_out_kg` for mass-bearing activities (unit = kilogram/tonne/etc.) because ecoinvent scales the activity by its reference product (output). Setting `amount = mass_in_kg` causes systematic overestimation proportional to `1/recovery_rate`.

---

## 5. Post-Processing — `Generation/functions/after_bw2.py`

### Module-Level Constants
```python
_UPSTREAM_KEYWORDS   = {"raw material", "extraction", "mining", "beneficiation", ...}
_DOWNSTREAM_KEYWORDS = {"distribution", "transport", "use phase", "end-of-life", ...}
_MASS_FLOW_TOLERANCE_WARN: float = 10.0   # % discrepancy → "warning"
_MASS_FLOW_TOLERANCE_GAP:  float = 25.0   # % discrepancy → "gap"
_MASS_BEARING_UNITS: frozenset = {"kg", "kilogram", "kilograms", "g", "gram", "grams",
                                   "tonne", "ton", "metric ton", "metric tons",
                                   "lb", "pound", "pounds"}
```

### Functions
```python
def _classify_stream(stage: str) -> str
    # Returns "upstream" | "downstream" | "gate-to-gate".
    # Matches stage string against _UPSTREAM_KEYWORDS and _DOWNSTREAM_KEYWORDS sets.

def _is_mass_bearing(unit: str) -> bool
    # Returns True if unit.strip().lower() is in _MASS_BEARING_UNITS.

def _validate_mass_flow_chain(functional_unit: list) -> dict
    # ISO 14040/44 mass balance checker.
    # Iterates functional_unit; collects activities where _is_mass_bearing(unit) is True.
    # For each consecutive pair: disc_pct = abs(prev_mass_out - next_mass_in) / prev_mass_out * 100
    #   disc_pct > 25 → "gap", > 10 → "warning", else → "ok"
    # Returns:
    #   {"status": "pass|warning|gap|skipped", "overall_yield": float,
    #    "activities": [...], "transitions": [{"from", "to", "expected_kg", "actual_kg", "disc_pct", "flag"}]}
    # Returns {"status": "skipped"} when < 2 mass-bearing activities found.

def compute_mass_flow_balance(output_filepath: str) -> dict
    # Loads the paired input template for a given output file path.
    # Derives input path: output_filepath basename → replace "_output.json" → ".json" → bw2_input_templates/
    # Calls _validate_mass_flow_chain on functional_unit; returns result dict.
    # Returns {"status": "skipped"} on missing file or parse error.

def result_data_processing(
    filename, product, scale_factor=1.0, display_amount=1.0,
    display_unit='tonne', activities_by_stage_dict=None
) -> dict
    # Loads BW2 output JSON (filename = path to _output.json).
    # Calls compute_mass_flow_balance to get mass flow audit.
    # Builds _mf_act_map: {activity_name_lower: {mass_in_kg, mass_out_kg, recovery_rate}}
    # For each activity in results, looks up mass flow data and builds process dict:
    #   {name, mean, sd, p5, p95, stream, mass_in_kg, mass_out_kg, recovery_rate}
    # Embeds mass_flow_balance into lcia_data["metadata"].
    # Returns lcia_data dict (see schema below).

def final_mode_output_bw2_included(
    filename, question, gpt5_client, product,
    scale_factor=1.0, n_simulations=25,
    display_amount=1.0, display_unit='tonne',
    activities_by_stage_dict=None
) -> dict
    # Orchestrates: result_data_processing → generate_markdown → generate_json → LLM narrative.
    # Returns answer_pack dict.

def generate_markdown(product, lcia_data, total_mean, display_amount=1.0, display_unit='tonne') -> str
    # Builds full markdown report: process table + summary stats + mass flow audit section.
    # Appends "### Mass Flow Balance Audit (ISO 14040/44)" table if transitions exist.

def generate_json(product, lcia_data, total_mean) -> dict
    # Returns structured JSON report dict including "massFlowBalance" key.
```

### `lcia_data` Dict Schema
```python
lcia_data = {
    "metadata": {
        "product":        str,
        "total_mean":     float,   # kg CO2-eq (scaled)
        "total_sd":       float,
        "stream_counts":  {"upstream": int, "downstream": int, "gate-to-gate": int},
        "mass_flow_balance": { ...output of _validate_mass_flow_chain... }
    },
    "<product_name>": {
        "<activity_name>": {
            "name":          str,
            "mean":          float,
            "sd":            float,
            "p5":            float,
            "p95":           float,
            "stream":        "upstream" | "downstream" | "gate-to-gate",
            "mass_in_kg":    float | None,
            "mass_out_kg":   float | None,
            "recovery_rate": float | None,
        }
    }
}
```

---

## 6. BW2 Calculation Engine — `Generation/functions/bw2_calculation.py`

### Functions
```python
def _ensure_lca_source_path() -> None
    # Verify BW2 source data path exists; raises if missing.

def _ensure_ecoinvent_directory() -> None
    # Verify ecoinvent data directory under brightway_databases/; raises if missing.

def _refresh_engine_modules() -> None
    # Force-reload BW2 Python modules (handles module-caching issues on Azure).

def _initialize_ecoinvent_if_needed() -> None
    # Set up the ecoinvent BW2 project if not already initialised.
    # Reads from brightway_databases/brightway2_ecoinvent/.

def connect_and_calculate_lca(engine_arg: str, json_path: str, background: bool = False) -> str
    # Run the BW2 LCA calculation for a single input JSON.
    # engine_arg: BW2 engine identifier ("Ecoinvent").
    # json_path: absolute path to bw2_input_templates/*.json.
    # Returns path to the written _output.json in bw2_results/.

def start_lca_calculation(choice: str, json_path: str, background: bool = False) -> str
    # Entry point called by query_transformation.py.
    # choice: "Ecoinvent" | "Tiangong".
    # Calls _initialize_ecoinvent_if_needed then connect_and_calculate_lca.
    # Returns output file path.
```

---

## 7. Vectorstore Layer — `Generation/functions/vectorstore.py`

> Re-exports `build_lca_knowledge_store`, `load_lca_knowledge_store`, `LcaKnowledgeRetriever` from `lca_knowledge_pipeline.py`.

### Functions
```python
def build_and_save_vectorstores(
    embeddings,
    eco_methods='_additionals/methods/EcoInvent_methods.txt',
    tg_methods='_additionals/methods/tiangong_methods.txt',
    eco_activity='_additionals/methods/Ecoinvent_activity.txt',
    pdf_dir='_raw knowledge pdfs',
    persist_dir_lca_knowledge='./chroma_db_lca_knowledge',
    persist_dir_eco_methods='./chroma_db_eco_methods',
    persist_dir_tg_methods='./chroma_db_tg_methods',
    persist_dir_eco_activity='./chroma_db_eco_activity',
) -> tuple[Chroma, dict, Chroma, Chroma, Chroma]
    # Deletes existing Chroma DBs and rebuilds all four from source files.
    # Returns (vs_knowledge, parent_lookup, vs_eco_methods, vs_tg_methods, vs_eco_activity).

def load_vectorstores(embeddings, ...) -> tuple[Chroma, dict, Chroma, Chroma, Chroma]
    # Load all four pre-built Chroma DBs from disk.

def load_vectorstores_fast(embeddings, persist_dir_lca_knowledge='./chroma_db_lca_knowledge')
    -> tuple[Chroma, dict]
    # Load only chroma_db_lca_knowledge (used by generate_response_fast).
```

---

## 8. LCA Knowledge Pipeline — `Generation/functions/lca_knowledge_pipeline.py`

> **v2 — state-of-the-art PDF ingestion** (requires `pdfplumber>=0.11.0`)

### Architecture

```
PDF files
  └─ pdfplumber extraction (page-by-page)
       ├─ text (non-table regions, filtered for noise)   → _Segment(seg_type="text")
       └─ tables (structured cell data → Markdown)       → _Segment(seg_type="table")
           ↓
Section-aware segmentation
  ├─ Heading detection (Abstract / Methods / Results / Discussion / Conclusion)
  ├─ References section EXCLUDED from index
  └─ Large text blocks split at paragraph boundaries (≤1500 tokens/parent)
           ↓
Parent–Child pair building
  ├─ Text parent  → sentence-group children (≈200 tokens, 2-sentence overlap)
  └─ Table parent → row-group children (≤350 tokens, header repeated per group)
           ↓
ChromaDB (children embedded)  +  parent_lookup.json (full context on disk)
```

### Pydantic Models (v2 — backward-compatible with old parent_lookup.json)
```python
class ChildChunk(BaseModel)
    # Fields: chunk_id, parent_id, text, source, page,
    #         isic_code, geography, lifecycle_stage,
    #         chunk_type ("text"|"table"),       ← NEW
    #         section_type ("abstract"|"introduction"|"methods"|
    #                        "results"|"discussion"|"conclusion"|"other"),  ← NEW
    #         table_caption (str, populated for table chunks)   ← NEW

class ParentBlock(BaseModel)
    # Fields: parent_id, text, source, page_start,
    #         isic_code, geography, lifecycle_stage, child_ids,
    #         chunk_type, section_type, table_caption, table_index  ← NEW
```

### Internal Dataclass
```python
@dataclass
class _Segment
    # Internal unit produced by _extract_document_segments().
    # Fields: seg_type ("text"|"table"), content, source, page,
    #         section_type, caption, table_index
```

### Public Functions
```python
def standardize_text(text: str) -> str
    # Normalise vocabulary (abbreviations → canonical ecoinvent terms), collapse whitespace.

def infer_isic_code(text: str) -> str
def infer_lifecycle_stage(text: str) -> str
def infer_geography(text: str) -> str
    # Keyword/regex inference helpers; unchanged from v1.

def build_lca_knowledge_store(embeddings, pdf_dir, persist_dir, rebuild) -> (Chroma, dict)
    # Full v2 pipeline: pdfplumber extraction → section segmentation →
    # parent-child pairs → ChromaDB embed → parent_lookup.json persist.
    # Raises ImportError if pdfplumber is not installed.

def load_lca_knowledge_store(embeddings, persist_dir) -> (Chroma, dict)
    # Load Chroma DB + parent_lookup.json from disk (unchanged API).

class LcaKnowledgeRetriever
    # Hierarchical retriever: child-chunk similarity search → full parent text expansion.
    # __init__(vectorstore, parent_lookup, k_children=6)
    # .retrieve(query, k) → list[str]   (parent texts, tables returned as Markdown)
    # .as_langchain_retriever(k)         → RunnableLambda[str, list[Document]]
```

### Internal Functions (do not call directly)
```python
def _estimate_tokens(text) -> int          # ~4 chars/token estimate
def _tokenize_sentences(text) -> list[str] # regex sentence splitter, respects abbreviations
def _group_sentences(sentences, max_tokens=200, overlap=2) -> list[str]
def _clean_text(text) -> str               # remove page numbers, DOIs, copyright lines
def _table_to_markdown(table) -> str       # pdfplumber rows → GFM Markdown table
def _split_table_into_row_groups(table_md, caption, max_tokens=350) -> list[str]
def _find_table_caption(page_text, table_index) -> str
def _detect_section_heading(line) -> str | None  # returns section_type or None
def _extract_page_content(page) -> (str, list[str])  # text + table_mds per page
def _extract_document_segments(pdf_path) -> list[_Segment]
def _split_text_block(text, ...) -> list[_Segment]
def _build_text_parent_children(seg) -> (ParentBlock, list[ChildChunk])
def _build_table_parent_children(seg) -> (ParentBlock, list[ChildChunk])
def build_parent_child_pairs(segments) -> (list[ParentBlock], list[ChildChunk])
```

### Key Behavioural Notes
- **Table chunks are atomic**: a full Markdown table is always one `ParentBlock`. Children are row-groups so a query can match specific rows/values.
- **References section skipped**: once "references" / "bibliography" heading is detected, no more text segments are created (avoids citation noise).
- **Backward compatible**: old `parent_lookup.json` files load fine — new fields default to `"other"` / `""` / `-1`.
- **`chunk_type` in ChromaDB metadata** lets the retriever log whether it hit a table or text parent.

---

## 9. RAG Methods — `Generation/functions/RAG_methods.py`

```python
def query_decomposition_retrieval(query, gpt5_client, retriever_j, vectorstore, k_per_sub=3) -> list[Document]
    # Decomposes query into sub-questions (LLM), retrieves docs for each, applies RRF.
    # Returns deduplicated, re-ranked list of Documents.

def get_unique_union(documents: list[list]) -> list[Document]
    # Deduplicate across multiple retrieval lists by page_content hash.

def reciprocal_rank_fusion(results: list[list], k=60) -> list[Document]
    # RRF scoring across ranked lists. Higher combined rank → lower final rank.
    # Returns sorted list of (doc, score) pairs → caller extracts docs.

def join_docs(docs) -> str
    # "\n\n".join(doc.page_content for doc in docs)

def join_methods(methods) -> str
    # "\n".join(doc.page_content for doc in methods)
```

---

## 10. Key Conventions & Invariants

### File Naming
- Input templates: `bw2_input_templates/<Product>_Ecoinvent_<YYYYMMDDHHMMSS>.json`
- Output results: `bw2_results/<Product>_Ecoinvent_<YYYYMMDDHHMMSS>_output.json`
- Pairing: `compute_mass_flow_balance` derives input path by replacing `_output.json` → `.json` and moving to `bw2_input_templates/`.

### BW2 Reference Convention
- All BW2 calculations are normalised to **1 tonne (1000 kg)** of final product.
- `scale_factor` = `display_amount_kg / 1000` — multiplied onto results in `result_data_processing`.
- `amount` in each `functional_unit` activity = `mass_out_kg` (reference product output). Setting `amount = mass_in_kg` causes overestimation by factor `1/recovery_rate`.

### Mass Flow Fields in `functional_unit`
| Field | BW2 uses? | Meaning |
|---|---|---|
| `amount` | ✅ Yes | = `mass_out_kg` for mass-bearing activities; service quantity for transport/energy |
| `mass_out_kg` | ❌ No | Output of this process (kg) — `amount` must equal this |
| `mass_in_kg` | ❌ No | Input to this process (kg) = `mass_out_kg / recovery_rate` |
| `recovery_rate` | ❌ No | `mass_out_kg / mass_in_kg`; 1.0 for non-mass activities |

### Logging & Frontend Log Routing
- Add `extra={"frontend": True}` to any `logging.info()` / `logging.warning()` call that should appear in the frontend's live log panel.
- Plain `logging.info("msg")` goes to server console only.

### SQLite Override (Azure Compatibility)
- `app.py` and `vectorstore.py` both patch `sys.modules['sqlite3']` with `pysqlite3` at startup before any ChromaDB import. This is required for Azure App Service (ships SQLite < 3.35.0).

### Active LCI Database Switch
```python
# In query_transformation.py — change to switch database:
LCI_DATABASE = "Ecoinvent"   # default
LCI_DATABASE = "Tiangong"    # alternative
```

### LLM Error Handling Pattern
All LLM chain invocations in this codebase follow this pattern:
```python
try:
    result = chain.invoke({...})
except:
    result = chain.invoke({...})   # single retry on any exception
```

---

## 11. Environment Variables (`.env`)
| Variable | Used By | Purpose |
|---|---|---|
| `AZURE_API_KEY` | `query_transformation.py` | Azure OpenAI API key |
| `LANGCHAIN_API_KEY` | `query_transformation.py` | LangSmith tracing |
| `ALLOWED_ORIGINS` | `app.py` | Comma-separated allowed CORS origins |

---

## 12. Dependency Notes
- **`brightway2==2.4.7`** — pinned; do not upgrade without testing (BW2 API changed significantly in v2.5+).
- **`numpy==1.24.3`** — pinned to match BW2 compatibility; bw2calc is incompatible with numpy 2.x.
- **`pysqlite3-binary`** — production only; local dev uses system SQLite via the `try/except` in `app.py`.
- **`sentence-transformers==5.0.0`** — used by the local HuggingFace model in `_models/`.
