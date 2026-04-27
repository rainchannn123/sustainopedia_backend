"""
lca_knowledge_pipeline.py
─────────────────────────
High-fidelity data-preparation and ingestion pipeline for the LCA-focused RAG
system.  Implements:

  1. Vocabulary Standardization Layer   – ecoinvent 3.x/4.x nomenclature
  2. Hierarchical (Small-to-Big) Chunking – Parent (System Block) / Child units
  3. Structured Metadata Schema         – Pydantic-validated, ChromaDB-ready
  4. Verification helper                – round-trips a child → parent query

Usage (standalone):
    python -m Generation.functions.lca_knowledge_pipeline

Usage (from vectorstore / query_transformation):
    from Generation.functions.lca_knowledge_pipeline import build_lca_knowledge_store, LcaKnowledgeRetriever
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import uuid
from pathlib import Path
from typing import Optional
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  VOCABULARY STANDARDIZATION
# ─────────────────────────────────────────────────────────────────────────────

# Dictionary-based mapping: abbreviation/alias → ecoinvent canonical term
_VOCAB_MAP: dict[str, str] = {
    # Materials
    r"\bAl\b":           "Aluminium",
    r"\bFe\b":           "Iron",
    r"\bCu\b":           "Copper",
    r"\bNi\b":           "Nickel",
    r"\bMn\b":           "Manganese",
    r"\bCo\b":           "Cobalt",
    r"\bLi\b":           "Lithium",
    r"\bPb\b":           "Lead",
    r"\bZn\b":           "Zinc",
    r"\bSi\b":           "Silicon",
    r"\bAg\b":           "Silver",
    r"\bPt\b":           "Platinum",
    r"\bNd\b":           "Neodymium",
    # Technologies / systems
    r"\bPV\b":           "Photovoltaics",
    r"\bBEV\b":          "Battery Electric Vehicle",
    r"\bHEV\b":          "Hybrid Electric Vehicle",
    r"\bFCEV\b":         "Fuel Cell Electric Vehicle",
    r"\bICE\b":          "Internal Combustion Engine",
    r"\bLIB\b":          "Lithium-Ion Battery",
    r"\bLFP\b":          "Lithium Iron Phosphate Battery",
    r"\bNMC\b":          "Nickel Manganese Cobalt Battery",
    r"\bNCA\b":          "Nickel Cobalt Aluminium Battery",
    r"\bCCGT\b":         "Combined Cycle Gas Turbine",
    r"\bPCCP\b":         "Pulverised Coal Combustion Power",
    r"\bCCS\b":          "Carbon Capture and Storage",
    r"\bRES\b":          "Renewable Energy Source",
    # Impact categories
    r"\bGWP\b":          "Global Warming Potential",
    r"\bAP\b":           "Acidification Potential",
    r"\bEP\b":           "Eutrophication Potential",
    r"\bODP\b":          "Ozone Depletion Potential",
    r"\bADP\b":          "Abiotic Depletion Potential",
    r"\bCED\b":          "Cumulative Energy Demand",
    r"\bEF\b":           "Environmental Footprint",
    r"\bLCA\b":          "Life Cycle Assessment",
    r"\bLCI\b":          "Life Cycle Inventory",
    r"\bLCIA\b":         "Life Cycle Impact Assessment",
    # Units – normalise to SI / ecoinvent convention
    r"\bkg CO2[ -]?eq\.?/?\b": "kg CO2-eq",
    r"\bkWh\b":          "kWh",
    r"\bMWh\b":          "MWh",
    r"\bGJ\b":           "GJ",
    r"\bt CO2\b":        "tonne CO2",
    r"\bton\b":          "tonne",
    r"\btons\b":         "tonnes",
    r"\bliter\b":        "litre",
    r"\bliters\b":       "litres",
    r"\bm3\b":           "m³",
}

# Patterns compiled once at import time
_COMPILED_VOCAB = [(re.compile(pattern), replacement) for pattern, replacement in _VOCAB_MAP.items()]


def standardize_text(text: str) -> str:
    """Apply vocabulary standardization to a raw text string."""
    for pattern, replacement in _COMPILED_VOCAB:
        text = pattern.sub(replacement, text)
    # Collapse multiple spaces / normalize whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PYDANTIC DATA SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class ChildChunk(BaseModel):
    """A single retrievable unit (small chunk) stored in ChromaDB."""

    chunk_id:       str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id:      str
    text:           str
    source:         str
    page:           int = 0
    isic_code:      str = ""
    geography:      str = "GLO"
    lifecycle_stage: str = "unspecified"

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Child chunk text must not be empty.")
        return v


class ParentBlock(BaseModel):
    """A full System Block (large context) linked to multiple child chunks."""

    parent_id:       str = Field(default_factory=lambda: str(uuid.uuid4()))
    text:            str
    source:          str
    page_start:      int = 0
    isic_code:       str = ""
    geography:       str = "GLO"
    lifecycle_stage: str = "unspecified"
    child_ids:       list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  METADATA INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Simplified ISIC code inference from keywords in document text
_ISIC_KEYWORDS: list[tuple[str, str]] = [
    (r"steel|iron|blast furnace|coke",                       "2410"),  # Manufacture of basic iron and steel
    (r"copper|smelting|refin",                               "2420"),  # Non-ferrous metals
    (r"alumin",                                              "2420"),  # Non-ferrous metals
    (r"lithium.ion battery|battery cell|cathode|anode",      "2720"),  # Accumulators / batteries
    (r"electric vehicle|BEV|hybrid",                         "2910"),  # Motor vehicles
    (r"photovoltaic|solar panel|solar cell",                 "2790"),  # Electrical equipment
    (r"cement|concrete|clinker",                             "2394"),  # Cement
    (r"glass|silica",                                        "2310"),  # Glass
    (r"pulp|paper|cardboard",                                "1701"),  # Pulp and paper
    (r"plastic|polymer|polyethylene|PET|PP\b|HDPE",         "2013"),  # Plastics
    (r"electricity generation|power plant|grid mix",         "3510"),  # Electricity
    (r"coal mining|coal extraction",                         "0510"),  # Coal mining
    (r"crude oil|petroleum refin",                           "1920"),  # Petroleum refining
    (r"natural gas|LNG",                                     "0620"),  # Natural gas extraction
]
_COMPILED_ISIC = [(re.compile(p, re.IGNORECASE), code) for p, code in _ISIC_KEYWORDS]

_STAGE_KEYWORDS: list[tuple[str, str]] = [
    (r"mining|extraction|quarrying|ore",         "raw_material_extraction"),
    (r"manufactur|production|process|smelting|refin|assembl",  "manufacturing"),
    (r"transport|logistics|shipping|freight",    "distribution_transport"),
    (r"use phase|operation|driving|consumption", "use_phase"),
    (r"recycl|end.of.life|disposal|landfill|waste", "end_of_life"),
]
_COMPILED_STAGE = [(re.compile(p, re.IGNORECASE), stage) for p, stage in _STAGE_KEYWORDS]

_GEO_PATTERN = re.compile(
    r"\b(GLO|RER|RoW|CN|US|DE|FR|JP|AU|IN|BR|ZA|GB|IT|ES|KR|CA|MX|RU|TR|SE|NO|CH)\b"
)


def infer_isic_code(text: str) -> str:
    for pattern, code in _COMPILED_ISIC:
        if pattern.search(text):
            return code
    return ""


def infer_lifecycle_stage(text: str) -> str:
    for pattern, stage in _COMPILED_STAGE:
        if pattern.search(text):
            return stage
    return "unspecified"


def infer_geography(text: str) -> str:
    match = _GEO_PATTERN.search(text)
    return match.group(1) if match else "GLO"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PDF LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_pdfs(pdf_dir: str | Path) -> list[Document]:
    """
    Load all PDFs from *pdf_dir*, returning a list of LangChain Documents.
    Each document carries 'source' and 'page' metadata.
    Falls back gracefully if a single file fails.
    """
    pdf_dir = Path(pdf_dir)
    all_docs: list[Document] = []

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", pdf_dir)
        return all_docs

    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages  = loader.load()
            # Apply standardization to every page
            for page in pages:
                page.page_content = standardize_text(page.page_content)
                page.metadata.setdefault("source", pdf_path.name)
            all_docs.extend(pages)
            logger.info("Loaded %d pages from '%s'", len(pages), pdf_path.name)
        except Exception:
            logger.exception("Failed to load '%s' — skipping.", pdf_path.name)

    logger.info("Total pages loaded from PDF directory: %d", len(all_docs))
    return all_docs


# ─────────────────────────────────────────────────────────────────────────────
# 5.  HIERARCHICAL CHUNKING  (Small-to-Big / Parent-Document strategy)
# ─────────────────────────────────────────────────────────────────────────────

# Parent splitter: ~1000–1500 tokens → ~3500–5500 chars (≈ 3.5 chars/token)
_PARENT_SPLITTER = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1200,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " "],
)

# Child splitter: ~100–200 tokens → ~350–700 chars
_CHILD_SPLITTER = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=150,
    chunk_overlap=20,
    separators=["\n", ". ", " "],
)


def build_parent_child_pairs(docs: list[Document]) -> tuple[list[ParentBlock], list[ChildChunk]]:
    """
    Given a list of raw page Documents, produce validated ParentBlock and
    ChildChunk pairs using the Small-to-Big strategy.

    Returns:
        parents: list of ParentBlock (full context, not stored as vectors)
        children: list of ChildChunk (stored as vectors in ChromaDB)
    """
    parents:  list[ParentBlock] = []
    children: list[ChildChunk]  = []

    parent_docs = _PARENT_SPLITTER.split_documents(docs)

    for parent_doc in parent_docs:
        text   = parent_doc.page_content.strip()
        source = parent_doc.metadata.get("source", "unknown")
        page   = int(parent_doc.metadata.get("page", 0))

        if not text:
            continue

        parent = ParentBlock(
            text           = text,
            source         = source,
            page_start     = page,
            isic_code      = infer_isic_code(text),
            geography      = infer_geography(text),
            lifecycle_stage= infer_lifecycle_stage(text),
        )

        # Split parent into children
        child_docs = _CHILD_SPLITTER.split_text(text)
        for child_text in child_docs:
            child_text = child_text.strip()
            if not child_text:
                continue
            try:
                child = ChildChunk(
                    parent_id      = parent.parent_id,
                    text           = child_text,
                    source         = source,
                    page           = page,
                    isic_code      = parent.isic_code,
                    geography      = parent.geography,
                    lifecycle_stage= parent.lifecycle_stage,
                )
                parent.child_ids.append(child.chunk_id)
                children.append(child)
            except Exception:
                logger.warning("Skipping invalid child chunk from source='%s'.", source)

        parents.append(parent)

    logger.info(
        "Chunking complete: %d parent blocks → %d child chunks.",
        len(parents), len(children),
    )
    return parents, children


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CHROMADB INGESTION
# ─────────────────────────────────────────────────────────────────────────────

# Name for the knowledge-base collection (separate from existing chroma_db_journal)
KNOWLEDGE_COLLECTION = "lca_knowledge"
DEFAULT_PERSIST_DIR   = "./chroma_db_lca_knowledge"


def _parents_to_lookup(parents: list[ParentBlock]) -> dict[str, ParentBlock]:
    return {p.parent_id: p for p in parents}


def build_lca_knowledge_store(
    embeddings,
    pdf_dir:     str | Path = "./_raw knowledge pdfs",
    persist_dir: str        = DEFAULT_PERSIST_DIR,
    rebuild:     bool       = False,
) -> tuple[Chroma, dict[str, ParentBlock]]:
    """
    Full pipeline:
      1. Load PDFs from *pdf_dir*
      2. Standardize vocabulary
      3. Build Parent / Child pairs with validated metadata
      4. Embed child chunks and persist to ChromaDB

    Args:
        embeddings:   Any LangChain-compatible embedding function.
        pdf_dir:      Directory containing source PDFs.
        persist_dir:  Where to persist the ChromaDB collection.
        rebuild:      If True, wipe existing store and rebuild from scratch.

    Returns:
        vectorstore:    Chroma instance (children indexed).
        parent_lookup:  Dict mapping parent_id → ParentBlock (in-memory).
    """
    pdf_dir = Path(pdf_dir)
    persist_dir_path = Path(persist_dir)

    if rebuild and persist_dir_path.exists():
        shutil.rmtree(persist_dir_path)
        logger.info("Wiped existing knowledge store at '%s'.", persist_dir)

    # ── Load ──────────────────────────────────────────────────────────────────
    raw_docs = load_pdfs(pdf_dir)
    if not raw_docs:
        raise RuntimeError(f"No PDF documents could be loaded from '{pdf_dir}'.")

    # ── Transform ─────────────────────────────────────────────────────────────
    parents, children = build_parent_child_pairs(raw_docs)
    if not children:
        raise RuntimeError("Chunking produced zero child chunks. Aborting ingestion.")

    # ── Convert children to LangChain Documents with full metadata ────────────
    child_docs: list[Document] = []
    for c in children:
        child_docs.append(Document(
            page_content=c.text,
            metadata={
                "chunk_id":       c.chunk_id,
                "parent_id":      c.parent_id,
                "source":         c.source,
                "page":           c.page,
                "isic_code":      c.isic_code,
                "geography":      c.geography,
                "lifecycle_stage": c.lifecycle_stage,
            },
        ))

    # ── Ingest to ChromaDB ────────────────────────────────────────────────────
    vectorstore = Chroma.from_documents(
        documents        = child_docs,
        embedding        = embeddings,
        persist_directory= persist_dir,
        collection_name  = KNOWLEDGE_COLLECTION,
    )
    vectorstore.persist()
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    logger.info(
        "Knowledge store persisted: %d child chunks at '%s'.",
        len(child_docs), persist_dir,
    )

    parent_lookup = _parents_to_lookup(parents)

    # Persist parent_lookup alongside the ChromaDB so it can be reloaded later
    lookup_path = Path(persist_dir) / "parent_lookup.json"
    with open(lookup_path, "w", encoding="utf-8") as f:
        json.dump(
            {pid: block.model_dump() for pid, block in parent_lookup.items()},
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Parent lookup persisted: %d blocks at '%s'.", len(parent_lookup), lookup_path)

    return vectorstore, parent_lookup


def load_lca_knowledge_store(
    embeddings,
    persist_dir: str = DEFAULT_PERSIST_DIR,
) -> tuple[Chroma, dict[str, ParentBlock]]:
    """Load an already-built knowledge store from disk."""
    persist_dir_path = Path(persist_dir)
    if not persist_dir_path.exists():
        raise FileNotFoundError(
            f"Knowledge store not found at '{persist_dir}'. "
            "Run build_lca_knowledge_store() first."
        )
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=KNOWLEDGE_COLLECTION,
    )
    
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    # Load persisted parent_lookup
    lookup_path = persist_dir_path / "parent_lookup.json"
    if lookup_path.exists():
        with open(lookup_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        parent_lookup = {pid: ParentBlock.model_validate(data) for pid, data in raw.items()}
        logger.info("Parent lookup loaded: %d blocks from '%s'.", len(parent_lookup), lookup_path)
    else:
        logger.warning("parent_lookup.json not found in '%s'. Parent lookup will be empty.", persist_dir)
        parent_lookup = {}
    logger.info("LCA knowledge store loaded from '%s'.", persist_dir)
    return vectorstore, parent_lookup


# ─────────────────────────────────────────────────────────────────────────────
# 7.  PARENT-DOCUMENT RETRIEVER  (Small-to-Big query interface)
# ─────────────────────────────────────────────────────────────────────────────

class LcaKnowledgeRetriever:
    """
    Retrieves child chunks from ChromaDB, then returns the full Parent context.

    Usage:
        retriever = LcaKnowledgeRetriever(vectorstore, parent_lookup)
        parent_texts = retriever.retrieve("lithium-ion battery cathode material", k=4)
    """

    def __init__(
        self,
        vectorstore:    Chroma,
        parent_lookup:  dict[str, ParentBlock],
        k_children:     int = 6,
    ) -> None:
        self._vs            = vectorstore
        self._parents       = parent_lookup
        self._k_children    = k_children

    def retrieve(self, query: str, k: Optional[int] = None) -> list[str]:
        """
        Similarity-search for child chunks, then return the full parent text
        for each match (deduplicated by parent_id).

        Returns a list of parent text strings, ordered by first-match rank.
        """
        k = k or self._k_children
        child_hits = self._vs.similarity_search(query, k=k)

        seen_parents: dict[str, str] = {}  # parent_id → parent text
        for hit in child_hits:
            pid = hit.metadata.get("parent_id", "")
            if pid and pid not in seen_parents:
                parent = self._parents.get(pid)
                if parent:
                    seen_parents[pid] = parent.text

        logger.debug(
            "Query '%s': %d child hits → %d unique parents returned.",
            query[:80], len(child_hits), len(seen_parents),
        )
        return list(seen_parents.values())

    def as_langchain_retriever(self, k: Optional[int] = None):
        """
        Return a RunnableLambda compatible with LangChain LCEL pipelines.
        Supports .map() so it can be used in query-decomposition chains.
        The callable accepts a query string and returns a list of Documents
        (one per unique parent).
        """
        def _retrieve(query: str) -> list[Document]:
            texts = self.retrieve(query, k=k)
            return [Document(page_content=t) for t in texts]
        return RunnableLambda(_retrieve)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  VERIFICATION TEST  (run standalone to validate the pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def _run_verification(retriever: LcaKnowledgeRetriever, queries: list[str]) -> None:
    """
    Smoke-test: run sample queries and log the first 300 chars of each parent
    returned to confirm child → parent resolution works correctly.
    """
    logger.info("─── Verification queries ───")
    for q in queries:
        results = retriever.retrieve(q, k=4)
        if results:
            logger.info(
                "Query: '%s'\n  → %d parent(s) returned. First 300 chars:\n    %s",
                q, len(results), results[0][:300].replace("\n", " "),
            )
        else:
            logger.warning("Query: '%s' → No results.", q)
    logger.info("─── Verification complete ───")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  STANDALONE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from langchain_openai import AzureOpenAIEmbeddings

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    )

    load_dotenv("./.env")

    endpoint       = "https://seesustainopedia-gpt5-resource.cognitiveservices.azure.com/"
    tm_deployment  = "text-embedding-3-large"
    tm_api_version = "2024-02-01"
    subscription_key = os.getenv("AZURE_API_KEY")

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint   = endpoint,
        api_version      = tm_api_version,
        azure_deployment = tm_deployment,
        model            = tm_deployment,
        api_key          = subscription_key,
    )

    logger.info("Building LCA knowledge store from './_raw knowledge pdfs' …")
    vs, parent_lookup = build_lca_knowledge_store(
        embeddings  = embeddings,
        pdf_dir     = "./_raw knowledge pdfs",
        persist_dir = DEFAULT_PERSIST_DIR,
        rebuild     = True,
    )

    retriever = LcaKnowledgeRetriever(vs, parent_lookup, k_children=6)

    _run_verification(retriever, queries=[
        "lithium-ion battery cathode material Global Warming Potential",
        "crude steel production carbon emission China",
        "photovoltaics manufacturing energy consumption",
        "electric vehicle life cycle assessment",
        "copper smelting environmental impact",
    ])
