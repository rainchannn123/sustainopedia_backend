from __future__ import annotations
import json
import logging
import re
import shutil
import uuid
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional
import chromadb

try:
    import pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PDFPLUMBER_AVAILABLE = False

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
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
    # Units - normalise to SI / ecoinvent convention
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

    chunk_id:        str  = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id:       str
    text:            str
    source:          str
    page:            int  = 0
    isic_code:       str  = ""
    geography:       str  = "GLO"
    lifecycle_stage: str  = "unspecified"
    # v2 enriched fields (backward-compatible defaults)
    chunk_type:      str  = "text"       # "text" | "table"
    section_type:    str  = "other"      # "abstract" | "introduction" | "methods" |
                                         # "results" | "discussion" | "conclusion" | "other"
    table_caption:   str  = ""           # populated for table chunks

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Child chunk text must not be empty.")
        return v


class ParentBlock(BaseModel):
    """A full context block (section or table) linked to multiple child chunks."""

    parent_id:       str       = Field(default_factory=lambda: str(uuid.uuid4()))
    text:            str
    source:          str
    page_start:      int       = 0
    isic_code:       str       = ""
    geography:       str       = "GLO"
    lifecycle_stage: str       = "unspecified"
    child_ids:       list[str] = Field(default_factory=list)
    # v2 enriched fields (backward-compatible defaults)
    chunk_type:      str       = "text"
    section_type:    str       = "other"
    table_caption:   str       = ""
    table_index:     int       = -1      # -1 = not a table


# ─────────────────────────────────────────────────────────────────────────────
# 3.  METADATA INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_ISIC_KEYWORDS: list[tuple[str, str]] = [
    (r"steel|iron|blast furnace|coke",                       "2410"),
    (r"copper|smelting|refin",                               "2420"),
    (r"alumin",                                              "2420"),
    (r"lithium.ion battery|battery cell|cathode|anode",      "2720"),
    (r"electric vehicle|BEV|hybrid",                         "2910"),
    (r"photovoltaic|solar panel|solar cell",                 "2790"),
    (r"cement|concrete|clinker",                             "2394"),
    (r"glass|silica",                                        "2310"),
    (r"pulp|paper|cardboard",                                "1701"),
    (r"plastic|polymer|polyethylene|PET|PP\b|HDPE",         "2013"),
    (r"electricity generation|power plant|grid mix",         "3510"),
    (r"coal mining|coal extraction",                         "0510"),
    (r"crude oil|petroleum refin",                           "1920"),
    (r"natural gas|LNG",                                     "0620"),
]
_COMPILED_ISIC = [(re.compile(p, re.IGNORECASE), code) for p, code in _ISIC_KEYWORDS]

_STAGE_KEYWORDS: list[tuple[str, str]] = [
    (r"mining|extraction|quarrying|ore",                          "raw_material_extraction"),
    (r"manufactur|production|process|smelting|refin|assembl",     "manufacturing"),
    (r"transport|logistics|shipping|freight",                     "distribution_transport"),
    (r"use phase|operation|driving|consumption",                  "use_phase"),
    (r"recycl|end.of.life|disposal|landfill|waste",               "end_of_life"),
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
# 4.  TEXT UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token (adequate for chunking decisions)."""
    return max(1, len(text) // 4)


# Abbreviations common in academic LCA text that should NOT trigger sentence splits
_ABBREV_RE = re.compile(
    r"\b(?:Fig|Figs|Tab|Eq|Eqs|Ref|Refs|Vol|No|pp|ed|eds|et al|e\.g|i\.e|"
    r"vs|approx|cf|Dr|Mr|Mrs|Prof|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
    r"min|max|avg|std|km|kg|kWh|MWh|GJ|MJ|kJ|m3|pcs|est|incl|excl)\."
    r"|\b(?:Eq|Tab|Fig|Sec)\.",
    re.IGNORECASE,
)


def _tokenize_sentences(text: str) -> list[str]:
    """
    Split text into sentences using a regex approach that respects academic
    abbreviations (Fig., Tab., et al., e.g., i.e., unit labels, etc.).
    """
    # Temporarily mask abbreviation dots to prevent false splits
    protected = _ABBREV_RE.sub(lambda m: m.group().replace(".", "\x00"), text)
    # Split on sentence-ending punctuation followed by whitespace + capital / digit
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z({\[])", protected)
    # Restore masked dots; keep only non-trivial sentences (≥3 words)
    sentences = [s.replace("\x00", ".").strip() for s in raw]
    return [s for s in sentences if len(s.split()) >= 3]


def _group_sentences(
    sentences: list[str],
    max_tokens: int = 200,
    overlap: int = 2,
) -> list[str]:
    """
    Group sentences into overlapping chunks targeting ≤ max_tokens each.
    Each consecutive chunk starts 'overlap' sentences before the previous ended,
    providing context continuity for the embedding.
    """
    if not sentences:
        return []

    groups: list[str] = []
    start = 0
    n = len(sentences)

    while start < n:
        chunk: list[str] = []
        token_count = 0
        end = start

        while end < n:
            t = _estimate_tokens(sentences[end])
            # Always include at least one sentence per group (avoids infinite loop)
            if token_count + t > max_tokens and chunk:
                break
            chunk.append(sentences[end])
            token_count += t
            end += 1

        groups.append(" ".join(chunk))
        # Advance start by (chunk_size − overlap); guarantee forward progress
        advance = max(1, (end - start) - overlap)
        start += advance

    return [g.strip() for g in groups if g.strip()]


# Lines that are almost certainly page headers / footers / noise
_NOISE_LINE_RE = re.compile(
    r"^\s*(?:"
    r"\d+"                                    # standalone page number
    r"|Page\s+\d+(?:\s+of\s+\d+)?"           # "Page N of M"
    r"|https?://\S+"                          # raw URL
    r"|doi:\s*10\.\d+/\S+"                   # DOI line
    r"|©\s*\d{4}"                             # copyright
    r"|(?:Received|Accepted|Published online):\s*\d"  # submission dates
    r"|[ivxlcdmIVXLCDM]+\s*$"                # Roman numeral page numbers
    r")\s*$",
    re.IGNORECASE,
)


def _clean_text(text: str) -> str:
    """Remove page numbers, running headers, URLs, and similar noise lines."""
    if not text:
        return ""
    cleaned = [ln for ln in text.split("\n") if not _NOISE_LINE_RE.match(ln)]
    # Collapse 3+ consecutive blank lines to a single blank line
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned))
    return result.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  TABLE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _table_to_markdown(table: list[list]) -> str:
    """
    Convert a pdfplumber table (list of rows → list of cell values) to a
    GitHub-flavoured Markdown table string.  Handles None cells (merged cells)
    and normalises multi-line cell content.
    """
    if not table or len(table) < 2:
        return ""

    def clean_cell(c) -> str:
        if c is None:
            return ""
        return str(c).strip().replace("\n", " ")

    # Remove entirely empty rows
    cleaned_rows = [
        [clean_cell(c) for c in row]
        for row in table
        if any(c for c in row if c)
    ]
    if len(cleaned_rows) < 2:
        return ""

    # Pad all rows to the same width
    n_cols = max(len(row) for row in cleaned_rows)
    padded = [row + [""] * (n_cols - len(row)) for row in cleaned_rows]

    header = padded[0]
    data_rows = padded[1:]

    lines: list[str] = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * n_cols) + " |")
    for row in data_rows:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _split_table_into_row_groups(
    table_md: str,
    caption: str = "",
    max_tokens: int = 350,
) -> list[str]:
    """
    Split a large Markdown table into row-group child chunks.
    Each group repeats the header row so it is self-contained for retrieval.
    """
    lines = table_md.strip().split("\n")
    if len(lines) < 3:
        return [table_md]

    header_line = lines[0]      # | col1 | col2 | …
    separator   = lines[1]      # | --- | --- | …
    data_rows   = lines[2:]

    prefix = f"{caption}\n\n{header_line}\n{separator}" if caption else f"{header_line}\n{separator}"
    base_tokens = _estimate_tokens(prefix)

    groups: list[str] = []
    current_rows: list[str] = []
    current_tokens = base_tokens

    for row in data_rows:
        rt = _estimate_tokens(row)
        if current_tokens + rt > max_tokens and current_rows:
            groups.append(prefix + "\n" + "\n".join(current_rows))
            current_rows = []
            current_tokens = base_tokens
        current_rows.append(row)
        current_tokens += rt

    if current_rows:
        groups.append(prefix + "\n" + "\n".join(current_rows))

    return groups if groups else [table_md]


_TABLE_CAPTION_RE = re.compile(
    r"Table\s+(\d+)[\.:]?\s*([^\n]{5,200})",
    re.IGNORECASE,
)


def _find_table_caption(page_text: str, table_index: int) -> str:
    """
    Search page_text for a "Table N" caption matching the given table_index.
    Returns a formatted caption string or a generic fallback.
    """
    n = table_index + 1
    pattern = re.compile(
        rf"Table\s+{n}[\.:]?\s*([^\n]{{5,200}})",
        re.IGNORECASE,
    )
    match = pattern.search(page_text)
    if match:
        return f"Table {n}: {match.group(1).strip()}"
    return f"Table {n}"


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PDF EXTRACTION WITH PDFPLUMBER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Segment:
    """Internal unit produced by the PDF extraction pass."""
    seg_type:     str    # "text" | "table"
    content:      str
    source:       str
    page:         int
    section_type: str = "other"
    caption:      str = ""
    table_index:  int = -1


# Section-heading patterns for academic LCA papers.
# Each tuple: (compiled regex, section_type string).
# Patterns use $ anchors + a length guard to minimise false positives.
_HEADING_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^(?:\d+\.?\s+)?abstract\s*$", re.I),                 "abstract"),
    (re.compile(r"^(?:\d+\.?\s+)?introduction\s*$", re.I),             "introduction"),
    (re.compile(
        r"^(?:\d+[\.\d]*\s+)?(?:materials?\s+and\s+)?(?:methods?|"
        r"methodology|goal\s+and\s+scope|system\s+boundary|"
        r"functional\s+unit|life\s+cycle\s+inventor[yi]|lci|"
        r"lca\s+framework|study\s+design)\s*$", re.I),                 "methods"),
    (re.compile(
        r"^(?:\d+[\.\d]*\s+)?(?:results?(?:\s+and\s+discussion)?|"
        r"findings|lcia\s+results?|impact\s+assessment\s+results?|"
        r"sensitivity\s+analysis|uncertainty\s+analysis|"
        r"scenario\s+analysis)\s*$", re.I),                            "results"),
    (re.compile(r"^(?:\d+[\.\d]*\s+)?discussion\s*$", re.I),           "discussion"),
    (re.compile(
        r"^(?:\d+[\.\d]*\s+)?(?:conclusions?|summary(?:\s+and\s+outlook)?|"
        r"concluding\s+remarks?|final\s+remarks?)\s*$", re.I),         "conclusion"),
    (re.compile(
        r"^(?:references?|bibliography|works?\s+cited)\s*$", re.I),    "references"),
    (re.compile(r"^(?:\d+[\.\d]*\s+)?acknowledgm?ents?\s*$", re.I),   "acknowledgments"),
    (re.compile(
        r"^(?:\d+[\.\d]*\s+)?(?:background|literature\s+review|"
        r"related\s+work|state\s+of\s+(?:the\s+)?art)\s*$", re.I),    "introduction"),
]


def _detect_section_heading(line: str) -> Optional[str]:
    """
    Return the section type if *line* is a section heading, else None.
    Lines longer than 80 chars are rejected to avoid false positives on
    normal sentences that begin with a section keyword.
    """
    if len(line) > 80:
        return None
    for pattern, section_type in _HEADING_PATTERNS:
        if pattern.match(line):
            return section_type
    return None


def _extract_page_content(page) -> tuple[str, list[str]]:
    """
    Extract text and tables from a single pdfplumber page.

    Returns:
        page_text:       Text from non-table regions (cleaned).
        table_markdowns: List of Markdown-formatted table strings.
    """
    table_markdowns: list[str] = []
    table_bboxes:    list[tuple] = []

    # ── Extract structured tables ─────────────────────────────────────────────
    for tbl_obj in page.find_tables():
        rows = tbl_obj.extract()
        md = _table_to_markdown(rows)
        if md:
            table_markdowns.append(md)
            table_bboxes.append(tbl_obj.bbox)   # (x0, top, x1, bottom)

    # ── Extract text, excluding table bounding boxes ──────────────────────────
    if table_bboxes:
        # Filter out words that fall entirely within a table bbox
        words = page.extract_words(x_tolerance=3, y_tolerance=3)
        non_table_words: list[str] = []
        for word in words:
            wx0, wtop = float(word["x0"]), float(word["top"])
            wx1, wbot = float(word["x1"]), float(word["bottom"])
            inside = any(
                wx0 >= bx0 - 5 and wtop >= btop - 5
                and wx1 <= bx1 + 5 and wbot <= bbot + 5
                for (bx0, btop, bx1, bbot) in table_bboxes
            )
            if not inside:
                non_table_words.append(word["text"])
        page_text = " ".join(non_table_words)
    else:
        page_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""

    return page_text, table_markdowns


def _extract_document_segments(pdf_path: Path) -> list[_Segment]:
    """
    Parse a single PDF file into an ordered list of _Segment objects.

    Text is grouped by detected section headings.  Tables are created as
    dedicated atomic segments.  The References / Acknowledgments sections
    are excluded from the output to keep the knowledge base focused.
    """
    if not _PDFPLUMBER_AVAILABLE:
        raise ImportError(
            "pdfplumber is required for PDF extraction. "
            "Install it with: pip install pdfplumber"
        )

    source = pdf_path.name
    segments: list[_Segment] = []
    current_section = "other"
    text_buffer: list[str] = []
    text_start_page = 0
    table_counter = 0
    skip_text = False       # True once we enter References / Acknowledgments
    _MAX_BUFFER_TOKENS = 1600

    def _flush_buffer(end_page: int) -> None:
        nonlocal text_buffer, text_start_page
        joined = " ".join(text_buffer).strip()
        if len(joined.split()) >= 20:
            # If the buffer is too large for one parent, split on paragraph-like
            # double-spaces or long runs of text
            for sub in _split_text_block(joined, text_start_page, current_section, source):
                segments.append(sub)
        text_buffer = []
        text_start_page = end_page

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text, table_mds = _extract_page_content(page)
                page_text_clean = _clean_text(page_text)

                # ── Tables on this page ───────────────────────────────────────
                for tbl_md in table_mds:
                    caption = _find_table_caption(page_text_clean, table_counter)
                    full_content = f"{caption}\n\n{tbl_md}" if caption else tbl_md
                    segments.append(_Segment(
                        seg_type=     "table",
                        content=      full_content,
                        source=       source,
                        page=         page_num,
                        section_type= current_section,
                        caption=      caption,
                        table_index=  table_counter,
                    ))
                    table_counter += 1

                if skip_text:
                    continue

                # ── Text lines on this page ───────────────────────────────────
                for line in page_text_clean.split("\n"):
                    stripped = line.strip()
                    if not stripped:
                        continue

                    heading = _detect_section_heading(stripped)
                    if heading is not None:
                        # Flush accumulated text before switching section
                        _flush_buffer(page_num)
                        current_section = heading
                        text_start_page = page_num
                        # Stop indexing text once we reach references or acknowledgments
                        if heading in ("references", "acknowledgments"):
                            skip_text = True
                        continue

                    # Skip very short lines that look like artefacts
                    if len(stripped) <= 2:
                        continue

                    text_buffer.append(stripped)

                    # Flush if buffer is growing too large
                    if _estimate_tokens(" ".join(text_buffer)) > _MAX_BUFFER_TOKENS:
                        _flush_buffer(page_num)

        # Final flush
        _flush_buffer(page_num if "page_num" in dir() else 0)

    except Exception:
        logger.exception("Failed to extract segments from '%s' — skipping.", pdf_path.name)

    logger.info(
        "Extracted %d segments (%d tables) from '%s'.",
        len(segments),
        sum(1 for s in segments if s.seg_type == "table"),
        pdf_path.name,
    )
    return segments


def _split_text_block(
    text: str,
    start_page: int,
    section_type: str,
    source: str,
    max_tokens: int = 1500,
) -> list[_Segment]:
    """
    Split a large text block into multiple _Segments using paragraph
    boundaries.  Returns a single segment if the text is within budget.
    """
    if _estimate_tokens(text) <= max_tokens:
        return [_Segment("text", text, source, start_page, section_type)]

    paragraphs = [p.strip() for p in re.split(r"\n{2,}|\.\s{2,}", text) if p.strip()]
    if len(paragraphs) <= 1:
        return [_Segment("text", text, source, start_page, section_type)]

    result: list[_Segment] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        pt = _estimate_tokens(para)
        if current_tokens + pt > max_tokens and current:
            result.append(_Segment("text", "\n\n".join(current), source, start_page, section_type))
            # one-paragraph overlap for context continuity
            current = current[-1:]
            current_tokens = _estimate_tokens(current[0]) if current else 0
        current.append(para)
        current_tokens += pt

    if current:
        result.append(_Segment("text", "\n\n".join(current), source, start_page, section_type))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 7.  PARENT-CHILD PAIR BUILDING
# ─────────────────────────────────────────────────────────────────────────────

def _build_text_parent_children(
    seg: _Segment,
) -> tuple[ParentBlock, list[ChildChunk]]:
    """Build one ParentBlock + sentence-group ChildChunks for a text segment."""
    text = standardize_text(seg.content)

    parent = ParentBlock(
        text=           text,
        source=         seg.source,
        page_start=     seg.page,
        isic_code=      infer_isic_code(text),
        geography=      infer_geography(text),
        lifecycle_stage=infer_lifecycle_stage(text),
        chunk_type=     "text",
        section_type=   seg.section_type,
    )

    sentences   = _tokenize_sentences(text)
    child_texts = _group_sentences(sentences, max_tokens=200, overlap=2)

    # Fallback: if the sentence tokenizer returns nothing, use a single child
    if not child_texts:
        child_texts = [text[:800].strip()] if text else []

    # Contextual prefix prepended to every child's embedded text.
    # The parent text (returned to the LLM) is unaffected; only the embedding
    # input is enriched — this is the "Contextual Retrieval" technique that
    # significantly reduces retrieval failures for dense academic text.
    section_label = seg.section_type.replace("_", " ").title()
    source_stem   = Path(seg.source).stem
    ctx_prefix    = f"[{section_label} | {source_stem}] "

    children: list[ChildChunk] = []
    for ct in child_texts:
        ct = ct.strip()
        if len(ct.split()) < 8:   # raised from 4: short fragments add embedding noise
            continue
        try:
            child = ChildChunk(
                parent_id=      parent.parent_id,
                text=           ctx_prefix + ct,  # enriched embedding input
                source=         seg.source,
                page=           seg.page,
                isic_code=      parent.isic_code,
                geography=      parent.geography,
                lifecycle_stage=parent.lifecycle_stage,
                chunk_type=     "text",
                section_type=   seg.section_type,
            )
            parent.child_ids.append(child.chunk_id)
            children.append(child)
        except Exception:
            logger.warning("Skipping invalid text child from '%s'.", seg.source)

    return parent, children


def _build_table_parent_children(
    seg: _Segment,
) -> tuple[ParentBlock, list[ChildChunk]]:
    """
    Build one ParentBlock for the full table + ChildChunks as row-group slices.
    Small tables (≤ 350 tokens) produce a single child embedding the full table.
    Large tables are split into row-groups with the header repeated in each.
    """
    text = standardize_text(seg.content)

    parent = ParentBlock(
        text=           text,
        source=         seg.source,
        page_start=     seg.page,
        isic_code=      infer_isic_code(text),
        geography=      infer_geography(text),
        lifecycle_stage=infer_lifecycle_stage(seg.caption or text),
        chunk_type=     "table",
        section_type=   seg.section_type,
        table_caption=  seg.caption,
        table_index=    seg.table_index,
    )

    if _estimate_tokens(text) <= 350:
        child_texts = [text]
    else:
        # Isolate the raw markdown part (after any caption prefix)
        md_start = text.find("|")
        raw_md = text[md_start:] if md_start != -1 else text
        caption_part = text[:md_start].strip() if md_start > 0 else ""
        child_texts = _split_table_into_row_groups(raw_md, caption_part, max_tokens=350)

    # Contextual prefix: adds paper source to table child embeddings.
    # Tables already carry their caption in the text; this adds the file origin.
    source_stem      = Path(seg.source).stem
    tbl_ctx_prefix   = f"[{source_stem}] "

    children: list[ChildChunk] = []
    for ct in child_texts:
        ct = ct.strip()
        if not ct:
            continue
        try:
            child = ChildChunk(
                parent_id=      parent.parent_id,
                text=           tbl_ctx_prefix + ct,  # source context prepended
                source=         seg.source,
                page=           seg.page,
                isic_code=      parent.isic_code,
                geography=      parent.geography,
                lifecycle_stage=parent.lifecycle_stage,
                chunk_type=     "table",
                section_type=   seg.section_type,
                table_caption=  seg.caption,
            )
            parent.child_ids.append(child.chunk_id)
            children.append(child)
        except Exception:
            logger.warning("Skipping invalid table child from '%s'.", seg.source)

    return parent, children


def build_parent_child_pairs(
    segments: list[_Segment],
) -> tuple[list[ParentBlock], list[ChildChunk]]:
    """
    Convert a list of _Segment objects into validated ParentBlock / ChildChunk
    pairs using the Small-to-Big strategy.

    Returns:
        parents:  ParentBlock list (full context stored in parent_lookup).
        children: ChildChunk list (embedded and stored in ChromaDB).
    """
    parents:  list[ParentBlock] = []
    children: list[ChildChunk]  = []

    for seg in segments:
        if not seg.content.strip() or len(seg.content.split()) < 5:
            continue
        if seg.seg_type == "table":
            parent, seg_children = _build_table_parent_children(seg)
        else:
            parent, seg_children = _build_text_parent_children(seg)

        if seg_children:        # skip parents that produced no usable children
            parents.append(parent)
            children.extend(seg_children)

    logger.info(
        "Chunking complete: %d segments → %d parent blocks → %d child chunks.",
        len(segments), len(parents), len(children),
    )
    return parents, children


# ─────────────────────────────────────────────────────────────────────────────
# 8.  CHROMADB INGESTION
# ─────────────────────────────────────────────────────────────────────────────

KNOWLEDGE_COLLECTION = "lca_knowledge"
DEFAULT_PERSIST_DIR   = "./chroma_db_lca_knowledge"

# Max documents per embedding API call.
# text-embedding-3-large via Azure has a 300,000-token/request hard limit.
# Our child chunks are ≤350 tokens each → 200 × 350 = 70,000 tokens/batch (safe headroom).
_EMBED_BATCH_SIZE: int = 200


def _parents_to_lookup(parents: list[ParentBlock]) -> dict[str, ParentBlock]:
    return {p.parent_id: p for p in parents}


def build_lca_knowledge_store(
    embeddings,
    pdf_dir:     str | Path = "./_raw knowledge pdfs",
    persist_dir: str        = DEFAULT_PERSIST_DIR,
    rebuild:     bool       = False,
) -> tuple[Chroma, dict[str, ParentBlock]]:
    """
    Full ingestion pipeline:
      1. Extract text + tables from every PDF in *pdf_dir* via pdfplumber.
      2. Segment by detected section headings; filter out References section.
      3. Build Parent / Child pairs:
           - Text parents  → sentence-group children (200-token, 2-sentence overlap)
           - Table parents → row-group children (≤350 tokens, header repeated)
      4. Embed child chunks and persist to ChromaDB.
      5. Serialise parent_lookup.json for full-context retrieval.

    Args:
        embeddings:   Any LangChain-compatible embedding function.
        pdf_dir:      Directory containing source PDFs.
        persist_dir:  ChromaDB persistence directory.
        rebuild:      Wipe and rebuild if True.

    Returns:
        vectorstore:   Chroma instance (children indexed).
        parent_lookup: Dict mapping parent_id → ParentBlock.
    """
    if not _PDFPLUMBER_AVAILABLE:
        raise ImportError(
            "pdfplumber is required.  Install with: pip install pdfplumber"
        )

    pdf_dir = Path(pdf_dir)
    persist_dir_path = Path(persist_dir)

    if rebuild and persist_dir_path.exists():
        shutil.rmtree(persist_dir_path)
        logger.info("Wiped existing knowledge store at '%s'.", persist_dir)

    # ── Extract all PDFs into segments ────────────────────────────────────────
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"No PDF files found in '{pdf_dir}'.")

    all_segments: list[_Segment] = []
    for pdf_path in pdf_files:
        segs = _extract_document_segments(pdf_path)
        all_segments.extend(segs)
        logger.info("Loaded %d segments from '%s'.", len(segs), pdf_path.name)

    if not all_segments:
        raise RuntimeError(f"No content could be extracted from PDFs in '{pdf_dir}'.")

    # ── Build parent-child pairs ───────────────────────────────────────────────
    parents, children = build_parent_child_pairs(all_segments)
    if not children:
        raise RuntimeError("Chunking produced zero child chunks. Aborting ingestion.")

    # ── Build LangChain Document list for ChromaDB ────────────────────────────
    child_docs: list[Document] = []
    for c in children:
        child_docs.append(Document(
            page_content=c.text,
            metadata={
                "chunk_id":        c.chunk_id,
                "parent_id":       c.parent_id,
                "source":          c.source,
                "page":            c.page,
                "isic_code":       c.isic_code,
                "geography":       c.geography,
                "lifecycle_stage": c.lifecycle_stage,
                "chunk_type":      c.chunk_type,
                "section_type":    c.section_type,
                "table_caption":   c.table_caption,
            },
        ))

    # ── Ingest into ChromaDB (batched to respect the 300K token/request limit) ──
    vectorstore = Chroma(
        persist_directory= persist_dir,
        embedding_function=embeddings,
        collection_name=   KNOWLEDGE_COLLECTION,
    )
    total_batches = (len(child_docs) + _EMBED_BATCH_SIZE - 1) // _EMBED_BATCH_SIZE
    for batch_idx in range(total_batches):
        batch = child_docs[batch_idx * _EMBED_BATCH_SIZE : (batch_idx + 1) * _EMBED_BATCH_SIZE]
        vectorstore.add_documents(batch)
        logger.info(
            "Embedded batch %d/%d  (%d chunks, cumulative %d/%d)",
            batch_idx + 1, total_batches, len(batch),
            min((batch_idx + 1) * _EMBED_BATCH_SIZE, len(child_docs)), len(child_docs),
        )
    vectorstore.persist()
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    logger.info(
        "Knowledge store persisted: %d child chunks (%d tables, %d text) at '%s'.",
        len(child_docs),
        sum(1 for c in children if c.chunk_type == "table"),
        sum(1 for c in children if c.chunk_type == "text"),
        persist_dir,
    )

    # ── Persist parent_lookup ─────────────────────────────────────────────────
    parent_lookup = _parents_to_lookup(parents)
    lookup_path = persist_dir_path / "parent_lookup.json"
    persist_dir_path.mkdir(parents=True, exist_ok=True)
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
        persist_directory= persist_dir,
        embedding_function=embeddings,
        collection_name=   KNOWLEDGE_COLLECTION,
    )
    chromadb.api.client.SharedSystemClient.clear_system_cache()

    lookup_path = persist_dir_path / "parent_lookup.json"
    if lookup_path.exists():
        with open(lookup_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        parent_lookup = {pid: ParentBlock.model_validate(data) for pid, data in raw.items()}
        logger.info("Parent lookup loaded: %d blocks from '%s'.", len(parent_lookup), lookup_path)
    else:
        logger.warning(
            "parent_lookup.json not found at '%s'. Parent-context retrieval will be empty.",
            persist_dir,
        )
        parent_lookup = {}

    logger.info("LCA knowledge store loaded from '%s'.", persist_dir)
    return vectorstore, parent_lookup


# ─────────────────────────────────────────────────────────────────────────────
# 9.  PARENT-DOCUMENT RETRIEVER  (Small-to-Big query interface)
# ─────────────────────────────────────────────────────────────────────────────

class LcaKnowledgeRetriever:
    """
    Retrieves child chunks from ChromaDB, then returns the full Parent context.

    For table chunks the returned parent text is the complete Markdown table,
    enabling downstream LLMs to read exact emission factors / inventory data.

    Usage:
        retriever = LcaKnowledgeRetriever(vectorstore, parent_lookup)
        parent_texts = retriever.retrieve("copper smelting GWP emission factor", k=6)
    """

    def __init__(
        self,
        vectorstore:   Chroma,
        parent_lookup: dict[str, ParentBlock],
        k_children:    int = 6,
    ) -> None:
        self._vs       = vectorstore
        self._parents  = parent_lookup
        self._k        = k_children

    def retrieve(self, query: str, k: Optional[int] = None) -> list[str]:
        """
        Similarity-search for child chunks, de-duplicate by parent_id, and
        return the full parent text for each unique match (ordered by rank).
        """
        k = k or self._k
        child_hits = self._vs.similarity_search(query, k=k)

        seen: dict[str, str] = {}       # parent_id → parent text
        for hit in child_hits:
            pid = hit.metadata.get("parent_id", "")
            if pid and pid not in seen:
                parent = self._parents.get(pid)
                if parent:
                    seen[pid] = parent.text

        logger.debug(
            "Query '%s…': %d child hits → %d unique parents.",
            query[:60], len(child_hits), len(seen),
        )
        return list(seen.values())

    def as_langchain_retriever(self, k: Optional[int] = None):
        """
        Return a RunnableLambda compatible with LangChain LCEL pipelines.
        Supports .map() for use in query-decomposition chains.
        """
        def _retrieve(query: str) -> list[Document]:
            return [Document(page_content=t) for t in self.retrieve(query, k=k)]
        return RunnableLambda(_retrieve)


# ─────────────────────────────────────────────────────────────────────────────
# 10.  VERIFICATION TEST  (run standalone to validate the pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def _run_verification(retriever: LcaKnowledgeRetriever, queries: list[str]) -> None:
    """Smoke-test: confirm child → parent resolution for a set of sample queries."""
    logger.info("─── Verification queries ───")
    for q in queries:
        results = retriever.retrieve(q, k=4)
        if results:
            first = results[0]
            chunk_type = "TABLE" if first.startswith("|") or "Table" in first[:30] else "text"
            logger.info(
                "Query: '%s'\n  → %d parent(s) [first is %s]. Preview:\n    %s",
                q, len(results), chunk_type,
                first[:300].replace("\n", " ↵ "),
            )
        else:
            logger.warning("Query: '%s' → No results.", q)
    logger.info("─── Verification complete ───")


# ─────────────────────────────────────────────────────────────────────────────
# 11.  STANDALONE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from langchain_openai import AzureOpenAIEmbeddings

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    load_dotenv("./.env")

    endpoint         = "https://seesustainopedia-gpt5-resource.cognitiveservices.azure.com/"
    tm_deployment    = "text-embedding-3-large"
    tm_api_version   = "2024-02-01"
    subscription_key = os.getenv("AZURE_API_KEY")

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=   endpoint,
        api_version=      tm_api_version,
        azure_deployment= tm_deployment,
        model=            tm_deployment,
        api_key=          subscription_key,
    )

    logger.info("Building LCA knowledge store from './_raw knowledge pdfs' …")
    vs, parent_lookup = build_lca_knowledge_store(
        embeddings=  embeddings,
        pdf_dir=     "./_raw knowledge pdfs",
        persist_dir= DEFAULT_PERSIST_DIR,
        rebuild=     True,
    )

    retriever = LcaKnowledgeRetriever(vs, parent_lookup, k_children=6)

    _run_verification(retriever, queries=[
        "Global Warming Potential copper production emission factor",
        "lithium-ion battery cathode material life cycle assessment",
        "crude steel production carbon emission China",
        "photovoltaics manufacturing energy consumption",
        "electric vehicle cradle-to-gate environmental impact",
        "life cycle inventory data table emission factors",
    ])

