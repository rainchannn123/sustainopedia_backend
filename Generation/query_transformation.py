import os
import json
import logging
import sys
import time
import re
from datetime import datetime, timezone
from io import BytesIO
from operator import itemgetter
from xml.sax.saxutils import escape as xml_escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings, ChatOpenAI
import sqlite3
from .functions.non_RAG_methods import conduct_lca_anaylsis, Get_LCIA_methods, generate_response_bw2_input, generate_response_bw2_input_workbench_strict, extract_functional_unit, infer_product_from_query, input_q_to_json, generate_list_of_activities
from .functions.vectorstore import build_and_save_vectorstores, load_vectorstores, LcaKnowledgeRetriever, load_vectorstores_fast
from .functions.after_bw2 import final_mode_output_bw2_included
from .functions.bw2_calculation import start_lca_calculation
from .functions.RAG_methods import query_decomposition_retrieval, lca_template_knowledge_retrieval
from .functions.fast_lca_cache import cache_lookup, cache_store
# from .functions.intent_classifier import _DEFAULTS, classify_intent  # query intent classification and parameter extraction
# Brightway files are cached from Azure Blob on demand by bw2_calculation.

from dotenv import load_dotenv
load_dotenv(dotenv_path="./.env")

LCI_DATABASE = "Ecoinvent"
# LCI_DATABASE = "Tiangong"

start_time = time.time()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

section1_start = time.time()

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
_langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if _langchain_api_key:
    os.environ['LANGCHAIN_API_KEY'] = _langchain_api_key
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = 'false'

embedding_endpoint = os.getenv("AZURE_EMBEDDING_ENDPOINT")
llm_base_url = os.getenv("AZURE_OPENAI_ENDPOINT")
llm_deployment = "gpt-6-luna"
subscription_key = os.getenv("AZURE_API_KEY") 

tm_model_name = "text-embedding-3-large"
tm_deployment = "text-embedding-3-large"
tm_api_version = "2024-02-01"

llm_client = ChatOpenAI(
    base_url=llm_base_url,
    model=llm_deployment,
    api_key=subscription_key,
    use_responses_api=True,
)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=embedding_endpoint,
    api_version=tm_api_version,
    azure_deployment=tm_deployment,
    model=tm_model_name,
    api_key=subscription_key
)

_CO2_VALUE_RE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:kg\s*CO2|kgCO2|t\s*CO2|tonne\s*CO2|CO2[-\u2010]\s*eq|co2e|CO\u2082|GWP)",
    re.IGNORECASE,
)

# Pattern for an explicit mass quantity in a user query, e.g. "10 tonnes", "500 kg".
_MASS_QUERY_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:-\s*)?"
    r"(tonnes?|metric\s*tons?|tons?|kilograms?|kg|grams?\b|g\b|lbs?|pounds?|mt\b)",
    re.IGNORECASE,
)


def _extract_mass_from_query(question: str) -> tuple:
    m = _MASS_QUERY_RE.search(question)
    if m:
        amount = float(m.group(1))
        raw = m.group(2).lower().replace(" ", "").rstrip("s")  # normalise plural
        _UNIT_NORM = {
            "tonne": "tonne", "ton": "tonne", "metricton": "tonne", "mt": "tonne",
            "kg": "kg", "kilogram": "kg",
            "g": "g", "gram": "g",
            "lb": "lb", "pound": "lb",
        }
        unit = _UNIT_NORM.get(raw, raw)
        return amount, unit
    return 1.0, "tonne"


def _extract_n_simulations_from_query(question: str, default: int = 25) -> int:
    _N_SIM_PROMPT = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a parameter extractor for an LCA computation engine. "
            "Your only job is to find the number of Monte Carlo simulations "
            "explicitly requested in the user query. "
            "Reply with a single integer only (no units, no explanation). "
            "If no simulation count is mentioned, reply with exactly: None",
        ),
        ("human", "Query: {question}"),
    ])
    try:
        chain = _N_SIM_PROMPT | llm_client | StrOutputParser()
        raw = chain.invoke({"question": question}).strip()
        if raw.lower() == "none":
            return default
        n = int(raw)
        if 1 <= n <= 10_000:
            return n
    except Exception as exc:
        logging.warning("n_simulations extraction failed: %s", exc)
    return default


# ── Shared response-style rules injected into every fast-mode RAG prompt ──────
_RAG_STYLE_RULES = (
    "You are conversing directly with a user. Respond as a knowledgeable expert "
    "speaking naturally and confidently, in the same register as modern AI assistants. "
    "You have deep specialist knowledge relevant to this query — draw on it seamlessly "
    "without ever acknowledging its origin.\n\n"
    "STRICT PROHIBITION: Never mention, reference, or hint at any text, document, "
    "retrieved information, context, knowledge base, or external source in your reply. "
    "Banned phrases include (but are not limited to): 'the provided text', "
    "'based on the context', 'the retrieved information', 'the context indicates', "
    "'according to the knowledge provided', 'from the information given', "
    "'the text shows', 'based on available context', 'the data provided'. "
    "If you lack reliable data for a specific case, say so briefly and naturally "
    "(e.g. 'Reliable figures for this specific case aren\'t well-established …') "
    "without hedging via source references.\n\n"
    "FORMATTING:\n"
    "- Paragraphs are the default format. Write in clear, flowing prose.\n"
    "- Use ## markdown headings only when the answer covers two or more clearly "
    "distinct topics that benefit from separation.\n"
    "- Use a bullet list only when presenting three or more genuinely discrete, "
    "enumerable items that read awkwardly as prose. Never use bullets as a default "
    "structure or to pad a short answer.\n"
    "- Bold text very sparingly: at most one key value or conclusion per section, "
    "only when it genuinely deserves emphasis. Do not bold general terms, "
    "headings, or routine descriptive phrases.\n"
    "- Format all numeric values with their units as inline code, e.g. `2.1\u202fkg\u202fCO\u2082-eq`.\n"
    "- If the question is completely unrelated to LCA, sustainability, or environmental "
    "science, decline politely in a single sentence and do not elaborate."
)

_LCA_INTENT_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict intent classifier for a Life Cycle Assessment (LCA) platform. "
        "Answer ONLY with 'YES' or 'NO'.",
    ),
    (
        "human",
        "Does the following query ask to COMPUTE, CALCULATE, QUANTIFY, or ESTIMATE the "
        "carbon footprint, CO\u2082 emissions, GWP, or any environmental impact of a specific "
        "product, material, or process? \n"
        "Answer YES only if the intent is to obtain actual emission/impact values through "
        "LCA computation. Answer NO if the query is asking for a definition, explanation, "
        "literature review, or general concept.\n\n"
        "Query: {question}",
    ),
])


def _is_lca_computation_query(question: str) -> bool:
    """Return True when the LLM judges the query as requesting an LCA computation."""
    try:
        chain = _LCA_INTENT_CLASSIFIER_PROMPT | llm_client | StrOutputParser()
        result = chain.invoke({"question": question}).strip().upper()
        return result.startswith("YES")
    except Exception as exc:
        logging.warning("LCA intent classifier failed: %s", exc)
        return False


def _context_has_co2_values(context: str, product: str) -> bool:
    if not context:
        return False
    matches = _CO2_VALUE_RE.findall(context)
    if not matches:
        return False
    if product and product.strip().lower() not in ("not specified", "unknown product", ""):
        return product.lower() in context.lower()
    return len(matches) >= 2


def _build_fast_lca_user_params(
    question: str,
    product: str,
    amount: float = 1.0,
    unit: str = "tonne",
    n_simulations: int = 25,
) -> str:
    q = question.lower()
    if any(kw in q for kw in ("cradle to grave", "full lifecycle", "end of life", "end-of-life", "use phase")):
        system_boundary = "cradle-to-grave"
    elif "gate to gate" in q or "gate-to-gate" in q:
        system_boundary = "gate-to-gate"
    else:
        system_boundary = "cradle-to-gate"

    return (
        f"product: {product}\n"
        f"amount: {amount}\n"
        f"unit: {unit}\n"
        f"system_boundary: {system_boundary}\n"
        f"n_simulations: {n_simulations}\n"
        f"further_notes: {question}"
    )


def _rag_answer(question: str, product: str, context: str, intent: str) -> dict:
    """Generate a RAG-based answer using the retrieved context (no BW2 computation)."""
    intent_params = {"intent": intent, "product": product or "Not Specified"}

    # ── Role persona (intent-specific) ───────────────────────────────────────
    if intent == "Specific_LCA_result_inquiry":
        role_instruction = (
            "You are a Senior LCA Consultant conversing directly with a user about their "
            "Life Cycle Assessment results. The user has already run an LCA and is asking "
            "follow-up questions about the findings. Interpret and explain the results clearly, "
            "and draw on your broader LCA expertise to add insight where the data alone is "
            "insufficient. Be concise and direct."
            "If you have used any retrieved journal articles to inform your answer, you must clearly indicate the full citation of those articles at the end of your answer in a separate subsection with title \"References\""
        )
    elif intent == "literature":
        role_instruction = (
            "You are a leading LCA researcher with comprehensive knowledge of peer-reviewed "
            "literature on Life Cycle Assessment, environmental science, and sustainable systems. "
            "The user wants insight from academic research. Discuss findings, methodological "
            "nuances, and quantitative data authoritatively. Do not fabricate studies or "
            "cite specific papers unless you are certain they exist."
            "If you have used any retrieved journal articles to inform your answer, you must clearly indicate the full citation of those articles at the end of your answer in a separate subsection with title \"References\""

        )
    elif intent == "general":
        role_instruction = (
            "You are a Senior LCA Consultant with broad expertise in Life Cycle Assessment, "
            "environmental impact, and sustainable materials. The user has a conceptual or "
            "general question about LCA. Explain clearly and engagingly, using concrete "
            "examples where they genuinely add value."
            "IMPORTANT NOTE: If the question is completely unrelated to LCA, sustainability, or environmental science, decline politely in a single sentence and do not elaborate."
        )
    else:  # computation fallback
        role_instruction = (
            "You are a Senior LCA Consultant specialising in carbon footprint quantification "
            "and environmental impact assessment. The user wants LCA-based estimates for a "
            "product or material. Provide a well-grounded, quantitative answer rooted in "
            "established LCA methodology."
            "If you have used any retrieved journal articles to inform your answer, you must clearly indicate the full citation of those articles at the end of your answer in a separate subsection with title \"References\""

        )

    # ── Build the system message: role + style rules + background knowledge ──
    # The context is embedded directly — no label — so it reads as internalized
    # expertise rather than an external document being quoted.
    context_block = context.strip() if context and context.strip() else ""
    system_content = (
        role_instruction
        + "\n\n"
        + _RAG_STYLE_RULES
        + ("\n\n" + context_block if context_block else "")
    )

    # Prepend product hint to the user turn when a product is known.
    product_known = product and product.strip().lower() not in ("", "not specified")
    user_message = (
        f"[Product: {product.strip()}]\n\n{question}" if product_known else question
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_content),
        ("human", "{question}"),
    ])
    chain = prompt | llm_client | StrOutputParser()
    try:
        answer = chain.invoke({"question": user_message})
    except Exception:
        answer = ""

    return {"answer": answer, "markdown": None, "processed_json": None, "lcia_table": None, "intent_params": intent_params}


def _run_fast_lca_pipeline(
    question: str,
    product: str,
    knowledge_docs: list,
    req_amount: float = 1.0,
    req_unit: str = "tonne",
    n_simulations: int = 25,
) -> dict:
    """
    Run a streamlined BW2 LCA computation for the fast-response mode.

    Differences from the full LCA pipeline:
    - LCIA methods are resolved dynamically via Get_LCIA_methods (RAG + LLM), matching
      the same method-selection logic used by the full pipeline.
    - Supplementary context is taken directly from already-retrieved knowledge docs
      (skips the multi-sub-query lca_template_knowledge_retrieval step).
    - Monte Carlo simulations are capped at 5 for speed.
    - Only the eco-activity and LCIA-methods vectorstores are loaded (not all four).
    """
    logging.info("Fast LCA: loading vectorstores...", extra={"frontend": True})
    vectorstore_eco_activity = Chroma(
        persist_directory="./chroma_db_eco_activity",
        embedding_function=embeddings,
    )
    retriever_eco_activity = vectorstore_eco_activity.as_retriever(search_kwargs={"k": 10})

    if LCI_DATABASE == "Tiangong":
        vectorstore_methods = Chroma(
            persist_directory="./chroma_db_tg_methods",
            embedding_function=embeddings,
        )
        retriever_lca_methods = vectorstore_methods.as_retriever(search_kwargs={"k": 3})
    else:  # Ecoinvent (default)
        vectorstore_methods = Chroma(
            persist_directory="./chroma_db_eco_methods",
            embedding_function=embeddings,
        )
        retriever_lca_methods = vectorstore_methods.as_retriever(search_kwargs={"k": 5})

    # Wrap already-retrieved docs as a LangChain Runnable (avoids extra LLM calls).
    context_runnable = RunnableLambda(lambda _: knowledge_docs)

    user_params_str = _build_fast_lca_user_params(question, product, req_amount, req_unit, n_simulations)
    structured_params = input_q_to_json(user_params_str)

    logging.info("Fast LCA: selecting LCIA methods...", extra={"frontend": True})
    LCIA_method_list = Get_LCIA_methods(question, llm_client, retriever_lca_methods)

    logging.info("Fast LCA: generating activity list...", extra={"frontend": True})
    retrieved_activity_list = generate_list_of_activities(
        llm_client, retriever_eco_activity, product, user_params_str
    )

    logging.info("Fast LCA: constructing BW2 input template...", extra={"frontend": True})
    bw2_input_filepath, activities_by_stage_dict = generate_response_bw2_input(
        llm_client,
        context_runnable,
        LCIA_method_list,
        retrieved_activity_list,
        product,
        LCI_DATABASE,
        retriever_eco_activity,
        user_input_params=structured_params,
    )

    lcidb_choice = "2" if LCI_DATABASE == "Ecoinvent" else "1"
    logging.info("Fast LCA: running BW2 engine...", extra={"frontend": True})
    start_lca_calculation(lcidb_choice, bw2_input_filepath)

    bw2_output_rawpath = os.path.splitext(os.path.basename(bw2_input_filepath))[0]
    bw2_results_file = f"bw2_results/{bw2_output_rawpath}_output.json"

    display_amount, display_unit, scale_factor = extract_functional_unit(
        f"{structured_params.get('amount', '1')} {structured_params.get('unit', 'tonne')}"
    )

    logging.info("Fast LCA: generating final output...", extra={"frontend": True})
    res = final_mode_output_bw2_included(
        bw2_results_file,
        question,
        llm_client,
        product,
        scale_factor=scale_factor,
        n_simulations=structured_params.get("n_simulations", 25),
        display_amount=display_amount,
        display_unit=display_unit,
        activities_by_stage_dict=activities_by_stage_dict,
    )

    final_answer, lcia_data, markdown_output, json_output = (
        res if isinstance(res, tuple) and len(res) == 4
        else (res[0], res[1] if len(res) > 1 else {}, None, None)
    )

    intent_params = {"intent": "fast_lca_computation", "product": product}

    # Return a table only when multiple processes are present; for single-process
    # queries the LLM answer already contains the CO₂ value inline.
    n_processes = len((lcia_data or {}).get("processes", []))
    if n_processes > 1:
        return {
            "answer":        final_answer,
            "markdown":      markdown_output,
            "processed_json": json_output,
            "lcia_table":    lcia_data,
            "intent_params": intent_params,
        }
    else:
        return {
            "answer":        final_answer,
            "markdown":      None,
            "processed_json": None,
            "lcia_table":    None,
            "intent_params": intent_params,
        }


def generate_response_fast(question: str, product: str) -> dict:
    """
    Fast-path pipeline.
    Routing logic:
    1. Always retrieves context from chroma_db_lca_knowledge via RAG.
    2. Classifies intent with keyword heuristics (no LLM call).
    3. For LCA computation queries:
       a. If retrieved context already contains specific CO₂ values for the product,
          answer directly from context (skips BW2).
       b. Check MongoDB cache: if a result for a similar product (cosine similarity ≥ 0.80)
          was computed before, return it immediately.
       c. Otherwise, run the simplified BW2 pipeline (IPCC 2013 / GWP100a, n_simulations=25).
          Store the result in the MongoDB cache before returning.
    4. For all other intents (literature, general), return a RAG-based LLM answer.
    """
    # ── Step 1: RAG retrieval from lca_knowledge ──────────────────────────────
    context = ""
    knowledge_docs: list = []
    if os.path.isdir("./chroma_db_lca_knowledge"):
        try:
            vs_knowledge, parent_lookup = load_vectorstores_fast(
                embeddings,
                persist_dir_lca_knowledge="./chroma_db_lca_knowledge",
            )
            retriever = LcaKnowledgeRetriever(
                vs_knowledge, parent_lookup, k_children=4
            ).as_langchain_retriever()
            knowledge_docs = retriever.invoke(question) or []
            context = "\n\n".join(d.page_content for d in knowledge_docs)
        except Exception as exc:
            logging.warning("Fast RAG retrieval failed: %s", exc)

    # ── Step 2: Intent classification (keyword-based, no LLM) ────────────────
    q_lower = (question or "").lower()
    if any(kw in q_lower for kw in ("current lca result", "current lca context",
                                     "current lca insights", "current lca findings")):
        intent = "Specific_LCA_result_inquiry"
    elif any(kw in q_lower for kw in ("literature", "study", "studies", "review",
                                       "paper", "papers", "summarize")):
        intent = "literature"
    elif _is_lca_computation_query(question):
        intent = "computation"
    else:
        intent = "general"

    # ── Step 3: LCA computation routing ──────────────────────────────────────
    if intent == "computation":
        # 3a. Context already has specific numeric CO₂ values → answer directly.
        if _context_has_co2_values(context, product):
            logging.info("Fast LCA: context contains CO\u2082 values, answering from RAG.")
            return _rag_answer(question, product, context, intent)

        # 3b. Infer product name if not supplied.
        resolved_product = (product or "").strip()
        if not resolved_product or resolved_product.lower() == "not specified":
            resolved_product = infer_product_from_query(question, llm_client)
            
        elif resolved_product.isupper() and len(resolved_product) <= 10:
            expanded = infer_product_from_query(question, llm_client)
            if expanded and expanded.lower() not in ("unknown product", "not specified"):
                logging.info(
                    "Fast LCA: expanding acronym '%s' → '%s'",
                    resolved_product,
                    expanded,
                )
                resolved_product = expanded

        # 3c. Extract requested mass and simulation count for cache-aware lookup and scaling.
        req_amount, req_unit = _extract_mass_from_query(question)
        req_n_simulations = _extract_n_simulations_from_query(question)

        # 3d. MongoDB cache lookup (returns mass-scaled result when masses differ).
        try:
            query_embedding = embeddings.embed_query(question)
            cached = cache_lookup(resolved_product, query_embedding, req_amount, req_unit)
            if cached:
                logging.info(
                    "Fast LCA: cache hit for '%s' — skipping BW2 computation.",
                    resolved_product, extra={"frontend": True}
                )
                return cached
        except Exception as exc:
            logging.warning("Fast LCA: cache lookup error: %s", exc)

        # 3e. Run fast BW2 computation pipeline with the requested functional unit.
        try:
            result = _run_fast_lca_pipeline(
                question, resolved_product, knowledge_docs, req_amount, req_unit, req_n_simulations
            )

            # 3f. Persist result to MongoDB cache.
            try:
                query_embedding = embeddings.embed_query(question)
                cache_store(
                    resolved_product,
                    query_embedding,
                    result["answer"],
                    result.get("processed_json"),
                    result.get("lcia_table"),
                    result["intent_params"],
                    mass_amount=req_amount,
                    mass_unit=req_unit,
                )
            except Exception as exc:
                logging.warning("Fast LCA: cache store error: %s", exc)

            return result
        except Exception as exc:
            logging.warning(
                "Fast LCA: BW2 pipeline failed for '%s': %s — falling back to RAG.",
                resolved_product, exc,
            )
            return _rag_answer(question, resolved_product, context, intent)

    # ── Step 4: Literature / general intent → RAG answer ─────────────────────
    return _rag_answer(question, product, context, intent)


def generate_response_full_lca(user_input_params, product):
    section1_end = time.time()
    logging.info(f"Environment Setting took {section1_end - section1_start:.2f} seconds")
    
    # Infer product from the query if the user left the product field empty
    if not product or not product.strip():
        logging.info("Product field is empty — inferring product from user query...")
        product = infer_product_from_query(user_input_params, llm_client)
        logging.info(f"Inferred product: {product}")
        
    structured_user_input_json = input_q_to_json(user_input_params)
    logging.info(f"Retreiving relative knowledge from localized knowledge database...", extra={"frontend": True})


    ########### VECTORSTORE BUILDING AND LOADING
    if os.path.isdir('./chroma_db_lca_knowledge') and os.path.isdir('./chroma_db_eco_methods') and os.path.isdir('./chroma_db_tg_methods') and os.path.isdir('./chroma_db_eco_activity'):
        vectorstore_lca_knowledge, parent_lookup, vectorstore_eco_methods, vectorstore_tg_methods, vectorstore_eco_activity = load_vectorstores(embeddings, persist_dir_lca_knowledge='./chroma_db_lca_knowledge', persist_dir_eco_methods='./chroma_db_eco_methods', persist_dir_tg_methods='./chroma_db_tg_methods', persist_dir_eco_activity='./chroma_db_eco_activity')
    else:
        vectorstore_lca_knowledge, parent_lookup, vectorstore_eco_methods, vectorstore_tg_methods, vectorstore_eco_activity = build_and_save_vectorstores(embeddings, eco_methods='./_additionals/methods/EcoInvent_methods.txt', tg_methods='./_additionals/methods/tiangong_methods.txt', eco_activity='./_additionals/methods/Ecoinvent_activity.txt', pdf_dir='./_raw knowledge pdfs', persist_dir_lca_knowledge='./chroma_db_lca_knowledge', persist_dir_eco_methods='./chroma_db_eco_methods', persist_dir_tg_methods='./chroma_db_tg_methods', persist_dir_eco_activity='./chroma_db_eco_activity')

    retriever_lca_knowledge = LcaKnowledgeRetriever(vectorstore_lca_knowledge, parent_lookup, k_children=6).as_langchain_retriever()
    retriever_eco_methods = vectorstore_eco_methods.as_retriever(search_kwargs={"k": 5})
    retriever_tg_methods = vectorstore_tg_methods.as_retriever(search_kwargs={"k": 3})
    retriever_eco_activity = vectorstore_eco_activity.as_retriever(search_kwargs={"k": 10})

    #start rendering RAG
    logging.info(f"Computing the best LCIA methods for current analysis...", extra={"frontend": True})
    if LCI_DATABASE == "Tiangong":
        retriever_lca_methods = retriever_tg_methods
    elif LCI_DATABASE == "Ecoinvent":
        retriever_lca_methods = retriever_eco_methods
    LCIA_method_list = Get_LCIA_methods(user_input_params, llm_client, retriever_lca_methods)
    logging.info(f"Found a list of suitable LCIA methods ✔", extra={"frontend": True})
    

    #RAG retrieval
    logging.info(f"Retrieving more knowledge from localized knowledge database...", extra={"frontend": True})
    section4_start = time.time()
    query_decomposition_retrival = lca_template_knowledge_retrieval(user_input_params, llm_client, retriever_lca_knowledge, vectorstore_lca_knowledge, k_per_sub=4)
    section4_end = time.time()
    logging.info(f"retrieve alternative questions took {section4_end - section4_start:.2f} seconds")
    logging.info(f"Knowledge retrieval completed ✔", extra={"frontend": True})

    #1st results = bw2 input
    section5_start = time.time()
    logging.info(f"Constructing a detailed value chain template for LCA Engine computation...", extra={"frontend": True})
    retrieved_acitivity_list = generate_list_of_activities(llm_client, retriever_eco_activity, product, user_input_params)
    bw2_input_filepath, activities_by_stage_dict = generate_response_bw2_input(
        llm_client, query_decomposition_retrival, LCIA_method_list, retrieved_acitivity_list,
        product, LCI_DATABASE, retriever_eco_activity,
        user_input_params=structured_user_input_json
    )
    section5_end = time.time()
    logging.info(f"generate bw2 input took {section5_end - section5_start:.2f} seconds")
    logging.info(f"Template generation completed ✔", extra={"frontend": True})
    logging.info(f"Routing to the LCA Engine...", extra={"frontend": True})
    
    # # #BW2 Calculation
    if LCI_DATABASE == "Tiangong":
        lcidb_choice = "1"
    elif LCI_DATABASE == "Ecoinvent":
        lcidb_choice = "2"
    
    initialize_bw2_calculation = start_lca_calculation(lcidb_choice, bw2_input_filepath)
    bw2_output_rawpath = os.path.splitext(os.path.basename(bw2_input_filepath))[0]

    logging.info(f"BW2 calculation completed, output raw path: {bw2_output_rawpath}")
    logging.info(f"LCA computation completed. Ready to generate final output.", extra={"frontend": True})
    
    bw2_results_raw_filename = f"bw2_results/{bw2_output_rawpath}_output.json"

    _amount_str = str(structured_user_input_json.get("amount", "1"))
    _unit_str = structured_user_input_json.get("unit", "tonne")
    display_amount, display_unit, scale_factor = extract_functional_unit(f"{_amount_str} {_unit_str}")
    res = final_mode_output_bw2_included(bw2_results_raw_filename, user_input_params, llm_client, product, scale_factor=scale_factor, n_simulations=structured_user_input_json.get("n_simulations", 25), display_amount=display_amount, display_unit=display_unit, activities_by_stage_dict=activities_by_stage_dict)
    logging.info(f"Final output generation completed ✔, starting to prepare answer pack...", extra={"frontend": True})
    final_output_with_bw2 = None
    lcia_data = {}
    markdown_output = None
    json_output = None

    if isinstance(res, tuple):
        if len(res) == 2:
            final_output_with_bw2, lcia_data = res
        elif len(res) == 4:
            final_output_with_bw2, lcia_data, markdown_output, json_output = res
        else:
            final_output_with_bw2 = res[0]
            lcia_data = res[1] if len(res) > 1 else {}
    else:
        final_output_with_bw2 = res

    answer_pack = {
        "answer": final_output_with_bw2,
        "markdown": markdown_output,
        "processed_json": json_output,
        "lcia_table": lcia_data,
        "intent_params": structured_user_input_json  # stored for frontend display and MongoDB
    }

    return answer_pack



def _to_float(value, default=1.0):
    try:
        v = float(value)
        return v if v > 0 else default
    except Exception:
        return default


def _to_int(value, default=25, min_v=1, max_v=10000):
    try:
        v = int(value)
        return max(min_v, min(max_v, v))
    except Exception:
        return default


def _build_workbench_query_text(product: str, params: dict, nodes: list) -> str:
    chain_name = params.get("chain_name", "Workbench Value Chain")
    lines = [
        f"Please run a strict workbench LCA for product: {product}",
        f"Value chain: {chain_name}",
        f"System boundary: {params.get('system_boundary', 'cradle-to-gate')}",
        f"Functional unit: {params.get('amount', 1)} {params.get('unit', 'tonne')}",
        f"Monte Carlo simulations: {params.get('n_simulations', 25)}",
        "Strict process order:",
    ]
    for idx, node in enumerate(nodes, start=1):
        region = f" [{node.get('region')}]" if node.get("region") else ""
        lines.append(f"{idx}. {node.get('processName') or node.get('processId')}{region}")
    if params.get("further_notes"):
        lines.append(f"Additional notes: {params['further_notes']}")
    return "\n".join(lines)


def _escape_latex(text: str) -> str:
    value = str(text or "")
    replacements = {
        "\\": r"\\textbackslash{}",
        "&": r"\\&",
        "%": r"\\%",
        "$": r"\\$",
        "#": r"\\#",
        "_": r"\\_",
        "{": r"\\{",
        "}": r"\\}",
        "~": r"\\textasciitilde{}",
        "^": r"\\textasciicircum{}",
    }
    for k, v in replacements.items():
        value = value.replace(k, v)
    return value


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalise_scope_key(value: str) -> str:
    key = str(value or "").strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if key in {"scope1", "s1", "1"}:
        return "scope1"
    if key in {"scope2", "s2", "2"}:
        return "scope2"
    if key in {"scope3", "s3", "3"}:
        return "scope3"
    return ""


def _normalise_scope_assignments(payload: dict) -> dict:
    grouped = {"scope1": [], "scope2": [], "scope3": []}
    raw_grouped = payload.get("scope_assignments") or payload.get("scopes") or {}

    if isinstance(raw_grouped, dict):
        for raw_key, raw_activities in raw_grouped.items():
            scope_key = _normalise_scope_key(raw_key)
            if not scope_key or not isinstance(raw_activities, list):
                continue
            for idx, item in enumerate(raw_activities):
                if not isinstance(item, dict):
                    continue
                name = (
                    item.get("process")
                    or item.get("name")
                    or item.get("matched_activity")
                    or item.get("activity")
                    or f"Activity {idx + 1}"
                )
                grouped[scope_key].append(
                    {
                        "name": str(name).strip() or f"Activity {idx + 1}",
                        "mean_impact": _safe_float(item.get("mean_impact", item.get("impact", 0.0)), 0.0),
                        "amount_location": str(item.get("amount_location") or "").strip(),
                        "unit_location": str(item.get("unit_location") or "").strip(),
                        "ref_product": str(item.get("ref_product") or "").strip(),
                        "percentile": str(item.get("percentile") or "").strip(),
                    }
                )

    flat = payload.get("activities")
    if isinstance(flat, list):
        for idx, item in enumerate(flat):
            if not isinstance(item, dict):
                continue
            scope_key = _normalise_scope_key(item.get("scope"))
            if scope_key not in grouped:
                continue
            name = (
                item.get("process")
                or item.get("name")
                or item.get("matched_activity")
                or item.get("activity")
                or f"Activity {idx + 1}"
            )
            grouped[scope_key].append(
                {
                    "name": str(name).strip() or f"Activity {idx + 1}",
                    "mean_impact": _safe_float(item.get("mean_impact", item.get("impact", 0.0)), 0.0),
                    "amount_location": str(item.get("amount_location") or "").strip(),
                    "unit_location": str(item.get("unit_location") or "").strip(),
                    "ref_product": str(item.get("ref_product") or "").strip(),
                    "percentile": str(item.get("percentile") or "").strip(),
                }
            )

    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda x: x.get("mean_impact", 0.0), reverse=True)

    return grouped


def _load_scope_report_retriever():
    try:
        if os.path.isdir('./chroma_db_lca_knowledge') and os.path.isdir('./chroma_db_eco_methods') and os.path.isdir('./chroma_db_tg_methods') and os.path.isdir('./chroma_db_eco_activity'):
            vectorstore_lca_knowledge, parent_lookup, _, _, _ = load_vectorstores(
                embeddings,
                persist_dir_lca_knowledge='./chroma_db_lca_knowledge',
                persist_dir_eco_methods='./chroma_db_eco_methods',
                persist_dir_tg_methods='./chroma_db_tg_methods',
                persist_dir_eco_activity='./chroma_db_eco_activity',
            )
        else:
            vectorstore_lca_knowledge, parent_lookup, _, _, _ = build_and_save_vectorstores(
                embeddings,
                eco_methods='./_additionals/methods/EcoInvent_methods.txt',
                tg_methods='./_additionals/methods/tiangong_methods.txt',
                eco_activity='./_additionals/methods/Ecoinvent_activity.txt',
                pdf_dir='./_raw knowledge pdfs',
                persist_dir_lca_knowledge='./chroma_db_lca_knowledge',
                persist_dir_eco_methods='./chroma_db_eco_methods',
                persist_dir_tg_methods='./chroma_db_tg_methods',
                persist_dir_eco_activity='./chroma_db_eco_activity',
            )
        return LcaKnowledgeRetriever(vectorstore_lca_knowledge, parent_lookup, k_children=6).as_langchain_retriever()
    except Exception as exc:
        logging.warning("Scope report: failed to load knowledge retriever: %s", exc)
        return None


def _build_scope_report_paragraph(
    product: str,
    company_name: str,
    system_boundary: str,
    scope_label: str,
    activities: list,
    total_impact: float,
    knowledge_context: str,
) -> str:
    if not activities:
        return (
            f"No activities were assigned to {scope_label}. "
            "This section is therefore disclosed as not material for the current reporting cycle."
        )

    top_names = ", ".join(a["name"] for a in activities[:5])
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a senior ESG reporting consultant. Write concise, professional disclosure text "
            "for annual sustainability reports. Output plain text only (no markdown, no bullets). "
            "Use a formal corporate tone and include risk/management implications where relevant.",
        ),
        (
            "human",
            "Company: {company_name}\n"
            "Product / system assessed: {product}\n"
            "System boundary: {system_boundary}\n"
            "Scope section: {scope_label}\n"
            "Total emissions in this scope (kg CO2-eq): {total_impact:.3f}\n"
            "Top activities: {top_names}\n"
            "\n"
            "Activity details JSON:\n{activities_json}\n\n"
            "Knowledge context (internal):\n{knowledge_context}\n\n"
            "Write one polished paragraph (120-170 words) summarising this scope's emissions profile, "
            "main drivers, and recommended management actions."
            "Keep ALL numeric values in the text to 1 decimal place.",
        ),
    ])
    chain = prompt | llm_client | StrOutputParser()
    payload = {
        "company_name": company_name,
        "product": product,
        "system_boundary": system_boundary,
        "scope_label": scope_label,
        "total_impact": total_impact,
        "top_names": top_names,
        "activities_json": json.dumps(activities, ensure_ascii=False),
        "knowledge_context": knowledge_context,
    }
    try:
        return chain.invoke(payload).strip()
    except Exception:
        return chain.invoke(payload).strip()


def generate_scope_report_latex(report_payload: dict) -> dict:
    product = str(report_payload.get("product") or report_payload.get("product_name") or "Unknown Product").strip()
    company_name = str(report_payload.get("company_name") or "Reporting Entity").strip()

    form = report_payload.get("form") if isinstance(report_payload.get("form"), dict) else {}
    system_boundary = str(
        report_payload.get("system_boundary")
        or form.get("systemBoundary")
        or "cradle-to-gate"
    ).strip()

    scopes = _normalise_scope_assignments(report_payload)
    total_assigned = sum(len(v) for v in scopes.values())
    if total_assigned == 0:
        raise ValueError("No scoped activities were provided for report generation.")

    retriever = _load_scope_report_retriever()

    scope_meta = [
        ("scope1", "Scope 1 (Direct Emissions)"),
        ("scope2", "Scope 2 (Energy Indirect Emissions)"),
        ("scope3", "Scope 3 (Other Indirect Emissions)"),
    ]

    preview_sections = []
    latex_sections = []

    for scope_key, scope_label in scope_meta:
        activities = scopes.get(scope_key, [])
        scope_total = round(sum(_safe_float(a.get("mean_impact"), 0.0) for a in activities), 6)

        top_names = ", ".join(a.get("name", "") for a in activities[:6])
        retrieval_query = (
            f"ESG {scope_label} emissions accounting guidance for {product}. "
            f"System boundary: {system_boundary}. Activities: {top_names}"
        )
        knowledge_context = ""
        if retriever is not None:
            try:
                docs = retriever.invoke(retrieval_query) or []
                joined = []
                for d in docs[:4]:
                    joined.append(d.page_content if hasattr(d, "page_content") else str(d))
                knowledge_context = "\n\n".join(joined)
            except Exception as exc:
                logging.warning("Scope report retrieval failed for %s: %s", scope_key, exc)

        paragraph = _build_scope_report_paragraph(
            product=product,
            company_name=company_name,
            system_boundary=system_boundary,
            scope_label=scope_label,
            activities=activities,
            total_impact=scope_total,
            knowledge_context=knowledge_context,
        )

        preview_sections.append(
            {
                "scope": scope_key,
                "label": scope_label,
                "total_kg_co2eq": scope_total,
                "paragraph": paragraph,
                "activities": activities,
            }
        )

        if activities:
            rows = [
                f"{_escape_latex(a['name'])} & {a['mean_impact']:.1f} & {_escape_latex(a.get('unit_location') or '-') } \\\\"
                for a in activities
            ]
            rows_text = "\n".join(rows)
            table_block = (
                "\\begin{tabular}{p{0.57\\textwidth} r p{0.24\\textwidth}}\\n"
                "\\toprule\\n"
                "Activity & Mean Impact (kg CO$_2$-eq) & Unit / Location \\\\n"
                "\\midrule\\n"
                f"{rows_text}\\n"
                "\\bottomrule\\n"
                "\\end{tabular}"
            )
        else:
            table_block = "No activities assigned in this scope."

        latex_sections.append(
            "\n".join(
                [
                    f"\\section*{{{_escape_latex(scope_label)}}}",
                    f"\\textbf{{Reported total:}} {scope_total:.1f} kg CO$_2$-eq\\\\",
                    _escape_latex(paragraph),
                    "\\vspace{0.5em}",
                    table_block,
                ]
            )
        )

    total_reported = round(sum(s["total_kg_co2eq"] for s in preview_sections), 6)
    generated_at = datetime.now(timezone.utc).isoformat()

    latex_document = "\n".join(
        [
            "\\documentclass[a4paper,11pt]{article}",
            "\\usepackage[margin=1in]{geometry}",
            "\\usepackage[T1]{fontenc}",
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage{booktabs}",
            "\\usepackage{array}",
            "\\usepackage{setspace}",
            "\\setstretch{1.15}",
            "\\begin{document}",
            f"\\title{{ESG Scope Emissions Report: {_escape_latex(product)}}}",
            f"\\author{{{_escape_latex(company_name)}}}",
            f"\\date{{Generated: {_escape_latex(generated_at)}}}",
            "\\maketitle",
            f"\\textbf{{System boundary:}} {_escape_latex(system_boundary)}\\\\",
            f"\\textbf{{Total reported emissions:}} {total_reported:.1f} kg CO$_2$-eq",
            "\\vspace{1em}",
            *latex_sections,
            "\\end{document}",
        ]
    )

    return {
        "latex": latex_document,
        "preview": {
            "product": product,
            "company_name": company_name,
            "system_boundary": system_boundary,
            "generated_at": generated_at,
            "total_kg_co2eq": total_reported,
            "sections": preview_sections,
        },
    }

def generate_scope_report_pdf(payload: dict) -> tuple[bytes, str]:
    """Build a polished PDF ESG scope report.

    Accepts either:
    - a full report object: { report: { latex, preview } }
    - a raw report object directly: { latex, preview }
    - a scope assignment payload (same shape as /api/esg/scope-report generation input)
      in which case LaTeX+preview are generated first.

    Returns (pdf_bytes, product_name).
    """
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object.")

    report = payload.get("report") if isinstance(payload.get("report"), dict) else payload
    preview = report.get("preview") if isinstance(report, dict) else None

    if not isinstance(preview, dict):
        generated = generate_scope_report_latex(report if isinstance(report, dict) else {})
        preview = generated.get("preview") if isinstance(generated, dict) else None

    if not isinstance(preview, dict):
        raise ValueError("A valid report preview is required to generate PDF.")

    product = str(preview.get("product") or payload.get("product") or "Unknown Product").strip()
    company_name = str(preview.get("company_name") or payload.get("company_name") or "Reporting Entity").strip()
    system_boundary = str(preview.get("system_boundary") or payload.get("system_boundary") or "N/A").strip()
    def _format_generated_at(value: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

        candidates = [raw]
        if raw.endswith("Z"):
            candidates.append(raw[:-1] + "+00:00")

        for candidate in candidates:
            try:
                return datetime.fromisoformat(candidate).strftime("%Y-%m-%d %H:%M")
            except ValueError:
                continue

        fallback_match = re.search(r"(\d{4}-\d{2}-\d{2})[T\s](\d{2}:\d{2})", raw)
        if fallback_match:
            return f"{fallback_match.group(1)} {fallback_match.group(2)}"

        return raw

    generated_at = _format_generated_at(preview.get("generated_at") or datetime.now(timezone.utc).isoformat())
    total_reported = _safe_float(preview.get("total_kg_co2eq"), 0.0)
    sections = preview.get("sections") if isinstance(preview.get("sections"), list) else []


    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"ESG Scope Emissions Report - {product}",
        author=company_name,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ScopeReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        alignment=0,
        spaceAfter=8,
    )
    heading_style = ParagraphStyle(
        "ScopeSectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "ScopeBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        spaceAfter=6,
    )
    meta_style = ParagraphStyle(
        "ScopeMeta",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        spaceAfter=3,
    )

        
    def _p(text: str) -> str:
        return xml_escape(str(text or "")).replace("\n", "<br/>")

    def _clean_activity_name(value: str, fallback: str) -> str:

        raw = str(value or fallback).strip()
        cleaned = re.sub(r"^\s*Process\s*\d+\s*:\s*", "", raw, flags=re.IGNORECASE)
        return cleaned or fallback

    location_display_map = {
        "GLO": "Global (GLO)",
        "RER": "Europe (RER)",
        "WEU": "Western Europe (WEU)",
        "EEU": "Eastern Europe (EEU)",
        "MEA": "Middle East & Africa (MEA)",
        "ASI": "Asia (ASI)",
        "NAM": "North America (NAM)",
        "SAM": "South America (SAM)",
        "CN": "China (CN)",
        "US": "United States (US)",
        "CH": "Switzerland (CH)",
        "BR": "Brazil (BR)",
        "IN": "India (IN)",
        "RoW": "Rest of World (RoW)",
    }
    location_codes = sorted(location_display_map.keys(), key=len, reverse=True)

    def _extract_location_code(value: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""

        for code in location_codes:
            esc = re.escape(code)
            if re.search(rf"(?:\(\s*{esc}\s*\)|(?:[\/|,]|\s+-\s+|\s+){esc})\s*$", raw, flags=re.IGNORECASE):
                return code
            if raw.lower() == code.lower():
                return code
        return ""

    def _expand_location(value: str) -> str:
        code = _extract_location_code(value)
        if code:
            return location_display_map.get(code, code)
        return str(value or "").strip()

    def _parse_amount_and_location(activity: dict) -> tuple[str, str]:
        raw_amount_location = str(activity.get("amount_location") or "").strip()
        raw_unit_location = str(activity.get("unit_location") or "").strip()

        location_code = _extract_location_code(raw_amount_location) or _extract_location_code(raw_unit_location)

        amount = raw_amount_location
        if location_code and raw_amount_location:
            esc = re.escape(location_code)
            amount = re.sub(rf"\(\s*{esc}\s*\)\s*$", "", amount, flags=re.IGNORECASE)
            amount = re.sub(rf"(?:[\/|,]|\s+-\s+)\s*{esc}\s*$", "", amount, flags=re.IGNORECASE).strip()

        location = location_display_map.get(location_code, location_code) if location_code else (_expand_location(raw_unit_location) or "-")
        return (amount or raw_amount_location or "-", location or "-")


    table_header_style = ParagraphStyle(
        "ScopeTableHeader",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=8.6,
        leading=10.2,
        alignment=1,
    )
    table_cell_style = ParagraphStyle(
        "ScopeTableCell",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.6,
        leading=10.2,
        wordWrap="CJK",
    )
    table_cell_right_style = ParagraphStyle(
        "ScopeTableCellRight",
        parent=table_cell_style,
        alignment=2,
    )

    story = [

        Paragraph("ESG Scope Emissions Report", title_style),
        Paragraph(f"<b>Product:</b> {_p(product)}", meta_style),
        Paragraph(f"<b>Company:</b> {_p(company_name)}", meta_style),
        Paragraph(f"<b>System boundary:</b> {_p(system_boundary)}", meta_style),
        Paragraph(f"<b>Generated:</b> {_p(generated_at)}", meta_style),
        Paragraph(f"<b>Total reported emissions:</b> {total_reported:.1f} kg CO2-eq", meta_style),
        Spacer(1, 6),
    ]

    for section in sections:
        if not isinstance(section, dict):
            continue

        section_label = str(section.get("label") or section.get("scope") or "Scope Section")
        section_total = _safe_float(section.get("total_kg_co2eq"), 0.0)
        section_paragraph = str(section.get("paragraph") or "").strip()
        activities = section.get("activities") if isinstance(section.get("activities"), list) else []

        story.append(Paragraph(_p(section_label), heading_style))
        story.append(Paragraph(f"<b>Reported total:</b> {section_total:.1f} kg CO2-eq", body_style))
        if section_paragraph:
            story.append(Paragraph(_p(section_paragraph), body_style))

            table_rows = [[
            Paragraph("Activity", table_header_style),
            Paragraph("Amount", table_header_style),
            Paragraph("Location", table_header_style),
            Paragraph("Mean Impact (kg CO<sub>2</sub>-eq)", table_header_style),
        ]]

        if activities:
            for idx, a in enumerate(activities, start=1):
                if not isinstance(a, dict):
                    continue
                activity_name = _clean_activity_name(
                    a.get("name") or a.get("process"),
                    f"Activity {idx}",
                )
                amount, location = _parse_amount_and_location(a)
                mean_impact = _safe_float(a.get("mean_impact"), 0.0)
                table_rows.append([
                    Paragraph(_p(activity_name), table_cell_style),
                    Paragraph(_p(amount), table_cell_style),
                    Paragraph(_p(location), table_cell_style),
                    Paragraph(f"{mean_impact:.1f}", table_cell_right_style),
                ])
        else:
            table_rows.append([
                Paragraph("No activities assigned in this scope.", table_cell_style),
                Paragraph("-", table_cell_style),
                Paragraph("-", table_cell_style),
                Paragraph("-", table_cell_right_style),
            ])

        table = Table(table_rows, colWidths=[80 * mm, 25 * mm, 25 * mm, 40 * mm], repeatRows=1)


        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF2FF")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.8),
                        ("ALIGN", (3, 1), (3, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))

        story.append(table)
        story.append(Spacer(1, 8))

    if len(story) <= 8:
        story.append(Paragraph("No scope sections were available for rendering.", body_style))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    if not pdf_bytes:
        raise ValueError("PDF generation produced an empty file.")

    return pdf_bytes, product


def generate_response_workbench_lca(workbench_payload: dict) -> dict:

    section1_end = time.time()

    logging.info(f"Environment Setting took {section1_end - section1_start:.2f} seconds")

    nodes = workbench_payload.get("nodes") or []
    strict_nodes = [n for n in nodes if (n.get("processName") or n.get("processId"))]
    if not strict_nodes:
        raise ValueError("workbench_lca requires at least one ordered node")

    product = (
        workbench_payload.get("product")
        or workbench_payload.get("productName")
        or "Unknown Product"
    )

    params = {
        "chain_name": (workbench_payload.get("chainName") or "Workbench Value Chain").strip(),
        "product": product,
        "amount": _to_float(workbench_payload.get("functionalUnitAmount"), 1.0),
        "unit": (workbench_payload.get("functionalUnitUnit") or "tonne").strip() or "tonne",
        "system_boundary": (workbench_payload.get("systemBoundary") or "cradle-to-gate").strip() or "cradle-to-gate",
        "n_simulations": _to_int(workbench_payload.get("nSimulations"), 25),
        "further_notes": (workbench_payload.get("notes") or workbench_payload.get("additionalNotes") or "").strip(),
        "strict_nodes": [
            {
                "order": idx,
                "processId": n.get("processId", ""),
                "processName": (n.get("processName") or n.get("processId") or "").strip(),
                "region": (n.get("region") or "").strip(),
                "providerName": (n.get("providerName") or "").strip(),
            }
            for idx, n in enumerate(strict_nodes)
        ],
    }

    logging.info("Retreiving relative knowledge from localized knowledge database...", extra={"frontend": True})
    if os.path.isdir('./chroma_db_lca_knowledge') and os.path.isdir('./chroma_db_eco_methods') and os.path.isdir('./chroma_db_tg_methods') and os.path.isdir('./chroma_db_eco_activity'):
        vectorstore_lca_knowledge, parent_lookup, vectorstore_eco_methods, vectorstore_tg_methods, vectorstore_eco_activity = load_vectorstores(embeddings, persist_dir_lca_knowledge='./chroma_db_lca_knowledge', persist_dir_eco_methods='./chroma_db_eco_methods', persist_dir_tg_methods='./chroma_db_tg_methods', persist_dir_eco_activity='./chroma_db_eco_activity')
    else:
        vectorstore_lca_knowledge, parent_lookup, vectorstore_eco_methods, vectorstore_tg_methods, vectorstore_eco_activity = build_and_save_vectorstores(embeddings, eco_methods='./_additionals/methods/EcoInvent_methods.txt', tg_methods='./_additionals/methods/tiangong_methods.txt', eco_activity='./_additionals/methods/Ecoinvent_activity.txt', pdf_dir='./_raw knowledge pdfs', persist_dir_lca_knowledge='./chroma_db_lca_knowledge', persist_dir_eco_methods='./chroma_db_eco_methods', persist_dir_tg_methods='./chroma_db_tg_methods', persist_dir_eco_activity='./chroma_db_eco_activity')

    retriever_lca_knowledge = LcaKnowledgeRetriever(vectorstore_lca_knowledge, parent_lookup, k_children=6).as_langchain_retriever()
    retriever_eco_methods = vectorstore_eco_methods.as_retriever(search_kwargs={"k": 5})
    retriever_tg_methods = vectorstore_tg_methods.as_retriever(search_kwargs={"k": 3})
    retriever_eco_activity = vectorstore_eco_activity.as_retriever(search_kwargs={"k": 10})

    logging.info("Computing the best LCIA methods for current analysis...", extra={"frontend": True})
    retriever_lca_methods = retriever_tg_methods if LCI_DATABASE == "Tiangong" else retriever_eco_methods
    query_text = _build_workbench_query_text(product, params, params["strict_nodes"])
    LCIA_method_list = Get_LCIA_methods(query_text, llm_client, retriever_lca_methods)
    logging.info("Found a list of suitable LCIA methods ✔", extra={"frontend": True})

    logging.info("Retrieving more knowledge from localized knowledge database...", extra={"frontend": True})
    query_decomposition_retrival = lca_template_knowledge_retrieval(query_text, llm_client, retriever_lca_knowledge, vectorstore_lca_knowledge, k_per_sub=4)
    logging.info("Knowledge retrieval completed ✔", extra={"frontend": True})

    logging.info("Constructing a detailed value chain template for LCA Engine computation...", extra={"frontend": True})
    bw2_input_filepath, activities_by_stage_dict = generate_response_bw2_input_workbench_strict(
        llm_client,
        query_decomposition_retrival,
        LCIA_method_list,
        params["strict_nodes"],
        product,
        LCI_DATABASE,
        retriever_eco_activity,
        user_input_params=params,
    )
    logging.info("Template generation completed ✔", extra={"frontend": True})
    logging.info("Routing to the LCA Engine...", extra={"frontend": True})

    lcidb_choice = "2" if LCI_DATABASE == "Ecoinvent" else "1"
    start_lca_calculation(lcidb_choice, bw2_input_filepath)

    bw2_output_rawpath = os.path.splitext(os.path.basename(bw2_input_filepath))[0]
    bw2_results_raw_filename = f"bw2_results/{bw2_output_rawpath}_output.json"

    display_amount, display_unit, scale_factor = extract_functional_unit(f"{params['amount']} {params['unit']}")

    res = final_mode_output_bw2_included(
        bw2_results_raw_filename,
        query_text,
        llm_client,
        product,
        scale_factor=scale_factor,
        n_simulations=params.get("n_simulations", 25),
        display_amount=display_amount,
        display_unit=display_unit,
        activities_by_stage_dict=activities_by_stage_dict,
    )
    logging.info("Final output generation completed ✔, starting to prepare answer pack...", extra={"frontend": True})

    final_output_with_bw2 = None
    lcia_data = {}
    markdown_output = None
    json_output = None
    if isinstance(res, tuple):
        if len(res) == 2:
            final_output_with_bw2, lcia_data = res
        elif len(res) == 4:
            final_output_with_bw2, lcia_data, markdown_output, json_output = res
        else:
            final_output_with_bw2 = res[0]
            lcia_data = res[1] if len(res) > 1 else {}
    else:
        final_output_with_bw2 = res

    return {
        "answer": final_output_with_bw2,
        "markdown": markdown_output,
        "processed_json": json_output,
        "lcia_table": lcia_data,
        "intent_params": params,
    }

