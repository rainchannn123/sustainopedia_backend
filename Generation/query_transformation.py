import os
import json
import logging
import sys
import time
import re
from operator import itemgetter
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import sqlite3
from .functions.non_RAG_methods import conduct_lca_anaylsis, Get_LCIA_methods, generate_response_bw2_input, extract_functional_unit, infer_product_from_query, input_q_to_json, generate_list_of_activities
from .functions.vectorstore import build_and_save_vectorstores, load_vectorstores, LcaKnowledgeRetriever, load_vectorstores_fast
from .functions.after_bw2 import final_mode_output_bw2_included
from .functions.bw2_calculation import start_lca_calculation
from .functions.RAG_methods import query_decomposition_retrieval, lca_template_knowledge_retrieval
from .functions.fast_lca_cache import cache_lookup, cache_store
# from .functions.intent_classifier import _DEFAULTS, classify_intent  # query intent classification and parameter extraction
# from .functions.prebuild_bw2db import preinstall_ecoinvent_database  # pre-build BW2 databases to speed up first run

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

endpoint = "https://seesustainopedia-gpt5-resource.cognitiveservices.azure.com/"

gpt5_model_name = "gpt-5.4"
gpt5_deployment = "gpt-5.4"
subscription_key = os.getenv("AZURE_API_KEY") 
gpt5_api_version = "2024-12-01-preview"

tm_model_name = "text-embedding-3-large"
tm_deployment = "text-embedding-3-large"
tm_api_version = "2024-02-01"

gpt5_client = AzureChatOpenAI(
    azure_endpoint=endpoint,
    api_version=gpt5_api_version,
    azure_deployment=gpt5_deployment,
    api_key=subscription_key,
    # temperature=0.2
)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=endpoint,
    api_version=tm_api_version,
    azure_deployment=tm_deployment,
    model=tm_model_name,
    api_key=subscription_key
)

# preinstall_ecoinvent_database()  # pre-build BW2 databases in deployment
    
# if os.path.isdir('./chroma_db_lca_knowledge') and os.path.isdir('./chroma_db_eco_methods') and os.path.isdir('./chroma_db_tg_methods') and os.path.isdir('./chroma_db_eco_activity'):
#     pass
# else:
#     vectorstore_lca_knowledge, parent_lookup, vectorstore_eco_methods, vectorstore_tg_methods, vectorstore_eco_activity = build_and_save_vectorstores(embeddings, eco_methods='./_additionals/methods/EcoInvent_methods.txt', tg_methods='./_additionals/methods/tiangong_methods.txt', eco_activity='./_additionals/methods/Ecoinvent_activity.txt', pdf_dir='./_raw knowledge pdfs', persist_dir_lca_knowledge='./chroma_db_lca_knowledge', persist_dir_eco_methods='./chroma_db_eco_methods', persist_dir_tg_methods='./chroma_db_tg_methods', persist_dir_eco_activity='./chroma_db_eco_activity')


# ── Fast-mode LCA helpers ──────────────────────────────────────────────────────

# Pattern for specific numeric CO₂ values in retrieved context.
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
    """
    Extract (amount, unit) from the user query string.

    Returns (1.0, 'tonne') when no explicit mass is found so that the BW2
    pipeline defaults to a 1-tonne functional unit.
    """
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
    """
    Extract the number of Monte Carlo simulations requested in the user query.

    Uses a lightweight LLM call to handle the full range of natural-language
    phrasings (e.g. "run a hundred simulations", "n_simulations=200",
    "fifty Monte Carlo runs").  Falls back to *default* (25) when the LLM
    returns no valid number or the call fails.
    """
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
        chain = _N_SIM_PROMPT | gpt5_client | StrOutputParser()
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
# These are embedded in the system message so the LLM never sees the RAG
# context as 'external' — it treats it as its own internalized knowledge.
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
        chain = _LCA_INTENT_CLASSIFIER_PROMPT | gpt5_client | StrOutputParser()
        result = chain.invoke({"question": question}).strip().upper()
        return result.startswith("YES")
    except Exception as exc:
        logging.warning("LCA intent classifier failed: %s", exc)
        return False


def _context_has_co2_values(context: str, product: str) -> bool:
    """
    Return True when the retrieved context already contains specific numeric CO₂
    values relevant to the queried product, so BW2 computation can be skipped.
    """
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
    """
    Construct a minimal user_input_params string for the fast LCA BW2 pipeline.
    System boundary is inferred from the query; amount/unit use the requested
    functional unit so the BW2 result already reflects the correct mass.
    n_simulations defaults to 25 when not explicitly specified by the user.
    """
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
        )
    elif intent == "literature":
        role_instruction = (
            "You are a leading LCA researcher with comprehensive knowledge of peer-reviewed "
            "literature on Life Cycle Assessment, environmental science, and sustainable systems. "
            "The user wants insight from academic research. Discuss findings, methodological "
            "nuances, and quantitative data authoritatively. Do not fabricate studies or "
            "cite specific papers unless you are certain they exist."
        )
    elif intent == "general":
        role_instruction = (
            "You are a Senior LCA Consultant with broad expertise in Life Cycle Assessment, "
            "environmental impact, and sustainable materials. The user has a conceptual or "
            "general question about LCA. Explain clearly and engagingly, using concrete "
            "examples where they genuinely add value."
        )
    else:  # computation fallback
        role_instruction = (
            "You are a Senior LCA Consultant specialising in carbon footprint quantification "
            "and environmental impact assessment. The user wants LCA-based estimates for a "
            "product or material. Provide a well-grounded, quantitative answer rooted in "
            "established LCA methodology."
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
    chain = prompt | gpt5_client | StrOutputParser()
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
    LCIA_method_list = Get_LCIA_methods(question, gpt5_client, retriever_lca_methods)

    logging.info("Fast LCA: generating activity list...", extra={"frontend": True})
    retrieved_activity_list = generate_list_of_activities(
        gpt5_client, retriever_eco_activity, product, user_params_str
    )

    logging.info("Fast LCA: constructing BW2 input template...", extra={"frontend": True})
    bw2_input_filepath, activities_by_stage_dict = generate_response_bw2_input(
        gpt5_client,
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
        gpt5_client,
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
            resolved_product = infer_product_from_query(question, gpt5_client)
            
        elif resolved_product.isupper() and len(resolved_product) <= 10:
            # Short all-caps acronym (e.g. "PDMS", "PET", "PVC", "HDPE").
            # Expand to the full chemical name so vectorstore stage queries can
            # match ecoinvent activity names via token overlap.  Without this,
            # _rerank_eco_activities scores "polydimethylsiloxane production" as
            # 0 overlap against a "PDMS …" query → activity never retrieved →
            # functional_unit ends up empty.
            expanded = infer_product_from_query(question, gpt5_client)
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
    # answer_pack = {
    #     "answer": None,
    #     "markdown": {},
    #     "processed_json": None,
    #     "lcia_table": None
    # }
    section1_end = time.time()
    logging.info(f"Environment Setting took {section1_end - section1_start:.2f} seconds")
    
    # Infer product from the query if the user left the product field empty
    if not product or not product.strip():
        logging.info("Product field is empty — inferring product from user query...")
        product = infer_product_from_query(user_input_params, gpt5_client)
        logging.info(f"Inferred product: {product}")
        
    structured_user_input_json = input_q_to_json(user_input_params)

    # Classify intent and extract structured LCA parameters for routing and storage
    # intent_params = classify_intent(question, product, gpt5_client)
    # logging.info(
    #     f"Intent: {intent_params.intent} | Product: {intent_params.product} | "
    #     f"{intent_params.amount} {intent_params.unit} | {intent_params.system_boundary}"
    # )
    
    # # Use the classifier's product if it is more specific than the provided value
    # if intent_params.product and intent_params.product not in ('Not Specified', ''):
    #     product = intent_params.product
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
    LCIA_method_list = Get_LCIA_methods(user_input_params, gpt5_client, retriever_lca_methods)
    logging.info(f"Found a list of suitable LCIA methods ✔", extra={"frontend": True})
    

    #RAG retrieval
    logging.info(f"Retrieving more knowledge from localized knowledge database...", extra={"frontend": True})
    section4_start = time.time()
    query_decomposition_retrival = lca_template_knowledge_retrieval(user_input_params, gpt5_client, retriever_lca_knowledge, vectorstore_lca_knowledge, k_per_sub=4)
    section4_end = time.time()
    logging.info(f"retrieve alternative questions took {section4_end - section4_start:.2f} seconds")
    logging.info(f"Knowledge retrieval completed ✔", extra={"frontend": True})

    #1st results = bw2 input
    section5_start = time.time()
    logging.info(f"Constructing a detailed value chain template for LCA Engine computation...", extra={"frontend": True})
    retrieved_acitivity_list = generate_list_of_activities(gpt5_client, retriever_eco_activity, product, user_input_params)
    bw2_input_filepath, activities_by_stage_dict = generate_response_bw2_input(
        gpt5_client, query_decomposition_retrival, LCIA_method_list, retrieved_acitivity_list,
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
    # initialize_bw2_calculation = start_lca_calculation(lcidb_choice, f"bw2_input_templates/Crude Steel_Ecoinvent_20260302195453.json")
    # bw2_output_rawpath = os.path.splitext(os.path.basename(f"bw2_input_templates/Crude Steel_Ecoinvent_20260302195453.json"))[0]
    
    logging.info(f"BW2 calculation completed, output raw path: {bw2_output_rawpath}")
    logging.info(f"LCA computation completed. Ready to generate final output.", extra={"frontend": True})
    
    bw2_results_raw_filename = f"bw2_results/{bw2_output_rawpath}_output.json"
    # bw2_results_raw_filename = f"bw2_results/Crude Steel_Ecoinvent_20260421184356_output.json"
    

    _amount_str = str(structured_user_input_json.get("amount", "1"))
    _unit_str = structured_user_input_json.get("unit", "tonne")
    display_amount, display_unit, scale_factor = extract_functional_unit(f"{_amount_str} {_unit_str}")
    res = final_mode_output_bw2_included(bw2_results_raw_filename, user_input_params, gpt5_client, product, scale_factor=scale_factor, n_simulations=structured_user_input_json.get("n_simulations", 25), display_amount=display_amount, display_unit=display_unit, activities_by_stage_dict=activities_by_stage_dict)
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

    # logging.info(f"Answer pack prepared:{markdown_output}") 
    answer_pack = {
        "answer": final_output_with_bw2,
        "markdown": markdown_output,
        "processed_json": json_output,
        "lcia_table": lcia_data,
        "intent_params": structured_user_input_json  # stored for frontend display and MongoDB
    }
    
    return answer_pack
