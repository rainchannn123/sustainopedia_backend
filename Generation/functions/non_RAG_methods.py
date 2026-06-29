import os
import json
import logging
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from json import dumps, loads
import sys
import time
import re
from operator import itemgetter
from langchain_core.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from .RAG_methods import join_docs, join_methods
from .after_bw2 import _validate_mass_flow_chain, _balance_disposal_mass
from .bw2_calculation import lookup_activity_ref_products_batch
from datetime import datetime, timezone, timedelta

def input_q_to_json(question):
    lines = question.splitlines()

    data = {}
    for line in lines:
        if ": " in line:
            key, value = line.split(": ", 1)
            data[key.strip()] = value.strip()

    return data

def _word_to_num_en(text: str):
    """
    Convert an English number-word phrase (e.g. 'two hundred fifty') to float.
    Returns None if the phrase cannot be parsed as a valid positive number.
    """
    ones = {
        'zero': 0, 'a': 1, 'an': 1,
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
        'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
        'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
        'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
    }
    multipliers = {
        'hundred': 100, 'thousand': 1_000,
        'million': 1_000_000, 'billion': 1_000_000_000,
    }
    tokens = [t for t in text.strip().lower().replace('-', ' ').split()
              if t not in ('and', '')]
    if not tokens:
        return None
    result = 0.0
    current = 0.0
    for token in tokens:
        if token in ones:
            current += ones[token]
        elif token in multipliers:
            mult = multipliers[token]
            if mult == 100:
                current = (current if current > 0 else 1.0) * 100
            else:
                result += (current if current > 0 else 1.0) * mult
                current = 0.0
        else:
            return None  # unrecognised token — bail out
    result += current
    return float(result) if result > 0 else None


def _word_to_num_zh(text: str):
    """
    Convert a Chinese number string (e.g. '两百五十', '一万') to float.
    Returns None if the string cannot be parsed as a valid positive number.
    Handles bare 十X as 1X (e.g. '十五' → 15, '十' → 10).
    """
    digits = {
        '零': 0, '一': 1, '二': 2, '两': 2,
        '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
    }
    magnitudes = {'十': 10, '百': 100, '千': 1_000, '万': 10_000, '亿': 100_000_000, '萬': 10_000, '億': 100_000_000}
    if not text:
        return None
    result = 0
    current_digit = None
    section = 0
    for ch in text:
        if ch in digits:
            current_digit = digits[ch]
        elif ch in magnitudes:
            mag = magnitudes[ch]
            if mag == 10 and current_digit is None:
                current_digit = 1  # bare 十X → 1*10 + X
            if current_digit is None:
                return None
            if mag >= 10_000:
                section += current_digit
                result += section * mag
                section = 0
            else:
                section += current_digit * mag
            current_digit = None
        else:
            return None  # non-numeric character — bail out
    if current_digit is not None:
        section += current_digit
    result += section
    return float(result) if result > 0 else None


def extract_functional_unit(question: str):
    """
    Parse the requested quantity from a user LCA query.
    Handles Arabic numerals, English number words (e.g. 'five kilograms',
    'two hundred tonnes'), and Chinese number characters (e.g. '五吨', '两百千克').
    Returns (display_amount, display_unit, scale_factor) where scale_factor
    is the multiplier to apply to BW2 results computed at the 1-tonne reference.
    Defaults to (1.0, 'tonne', 1.0) when no quantity is detected.
    """
    # (regex_pattern, canonical_unit, kg_equivalent)
    # Patterns cover both English and Chinese unit keywords.
    unit_map = [
        (r'metric\s*tons?|公吨',                   'tonne',         1000.0),
        (r'tonnes?|吨',                            'tonne',         1000.0),
        (r'tons?',                                 'tonne',         1000.0),
        (r'kilograms?|千克|公斤',                  'kilogram',         1.0),
        (r'kg',                                    'kilogram',         1.0),
        (r'grams?|克(?!拉)',                        'gram',           0.001),
        (r'pounds?|lbs?|磅',                       'kilogram',    0.453592),
        (r'litres?|liters?|升',                    'litre',            1.0),
        (r'cubic\s*met(?:er|re)s?|m[³3]|立方米',   'cubic meter',      1.0),
        (r'kwh|千瓦时|度电',                        'kilowatt hour',    1.0),
        (r'mwh|兆瓦时',                             'megawatt hour', 1000.0),
        (r'mj|兆焦',                                'megajoule',        1.0),
        (r'gj|吉焦',                                'gigajoule',     1000.0),
        (r'units?|pieces?|件|个',                  'unit',             1.0),
    ]

    # ── Pass 1: Arabic numerals  "100 kg", "1.5 tonnes", "2,000吨" ────────────
    for unit_pattern, canonical, kg_equiv in unit_map:
        m = re.search(
            rf'(\d+(?:[.,]\d+)?)\s*(?:{unit_pattern})\b',
            question, re.IGNORECASE
        )
        if m:
            num = float(m.group(1).replace(',', '.'))
            return num, canonical, (num * kg_equiv) / 1000.0

    # ── Pass 2: English word numbers  "five kilograms", "two hundred tonnes" ──
    _EN_WORD = (
        r'(?:a\b|an\b|(?:'
        r'one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|'
        r'thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|'
        r'twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|'
        r'hundred|thousand|million|billion'
        r')(?:[\s\-]+(?:and[\s\-]+)?(?:'
        r'one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|'
        r'thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|'
        r'twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|'
        r'hundred|thousand|million|billion'
        r'))*)'
    )
    for unit_pattern, canonical, kg_equiv in unit_map:
        m = re.search(
            rf'\b({_EN_WORD})\s+(?:{unit_pattern})\b',
            question, re.IGNORECASE
        )
        if m:
            num = _word_to_num_en(m.group(1))
            if num is not None and num > 0:
                return num, canonical, (num * kg_equiv) / 1000.0

    # ── Pass 3: Chinese word numbers  "五吨", "两百千克", "一万吨" ──────────────
    _ZH_NUM = r'[零一二两三四五六七八九十百千万亿萬億]+'
    for unit_pattern, canonical, kg_equiv in unit_map:
        m = re.search(
            rf'({_ZH_NUM})\s*(?:{unit_pattern})',
            question, re.IGNORECASE
        )
        if m:
            num = _word_to_num_zh(m.group(1))
            if num is not None and num > 0:
                return num, canonical, (num * kg_equiv) / 1000.0

    # ── Pass 4: "per <unit>" / "每<unit>" — implicit quantity of 1 ─────────────
    for unit_pattern, canonical, kg_equiv in unit_map:
        if re.search(rf'(?:\bper\s+|每\s*)(?:{unit_pattern})\b', question, re.IGNORECASE):
            return 1.0, canonical, kg_equiv / 1000.0

    # ── Default: 1 tonne (matches BW2 reference — no scaling needed) ──────────
    return 1.0, 'tonne', 1.0


def infer_product_from_query(question: str, gpt5_client) -> str:
    template = (
        "You are a Life Cycle Assessment (LCA) expert. "
        "Extract the primary product or material being assessed in the following LCA query. "
        "Return only the product name as a short noun phrase "
        "(e.g., \"crude steel\", \"polypropylene\", \"coal power electricity\"). "
        "Do not include units, quantities, or any explanation.\n\n"
        "Query: {question}"
    )
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | gpt5_client | StrOutputParser()
    try:
        result = chain.invoke({"question": question})
    except:
        # rerun-it to prevent occasional LLM generation failure
        result = chain.invoke({"question": question})
    return result.strip() or "Unknown Product"


def conduct_lca_anaylsis(question, gpt5_client):
    print("Deciding on whether to conduct LCA analysis...")
    # template = """You are an expert in sustainability and environmental impact analysis. Your task is to determine whether a given user query requires a Life Cycle Assessment (LCA) to be properly addressed.

    #             A Life Cycle Assessment (LCA) is a method used to evaluate the environmental impacts associated with all stages of a product's life—from raw material extraction through production, use, and disposal.

    #             Instructions:
    #             - If the query involves evaluating the environmental impact of a product, material, process, or decision across its life cycle, reply with **"Yes"**.
    #             - If the query can be answered without considering life cycle impacts, reply with **"No"**.
    #             - Do not explain your answer. Only reply with **"Yes"** or **"No"**.

    #             User Query: "{question}"""
    
    template = """You are an expert in sustainability and environmental impact analysis. Your task is to determine whether a given user query requires a Life Cycle Assessment (LCA) to be properly addressed.

            A Life Cycle Assessment (LCA) is a method used to evaluate the environmental impacts associated with all stages of a product's life—from raw material extraction through production, use, and disposal.

            Instructions:
            - If the query involves evaluating the environmental impact or CO2 emissions of a product, material, process, or decision across its life cycle, reply with **"Yes"**.
            - If the query should be answered with consideration of life cycle impacts, reply with **"Yes"**.
            - If the query can be answered without considering life cycle impacts, reply normally to the user query in short (1-2 sentences).

            User Query: "{question}"""
    
                
    decision_prompt = ChatPromptTemplate.from_template(template)

    decision_chain = (
        {"question": itemgetter("question")}
        | decision_prompt 
        | gpt5_client
        | StrOutputParser()
    )

    try:
        decision_results = decision_chain.invoke({"question": question})
    except:
        # rerun-it to prevent occasional LLM generation failure
        decision_results = decision_chain.invoke({"question": question})
    print(decision_results)
    return decision_results

def Get_LCIA_methods(question, gpt5_client, retriever_lca_methods):
    print("Generating LCIA methods...")
    
    method_search_template = """You are a life-cycle assessment (LCA) expert. First, You must choose only one to three the most relevant LCIA methods from the following list based on the user question: {question}.
    LCIA method list: ['CML v4.8 2016 no LT', 'CML v4.8 2016', 'EF v3.0 EN15804', 'EF v3.0 no LT', 'EF v3.0', 'EF v3.1 EN15804', 'EF v3.1 no LT', 'EF v3.1', 'IPCC 2013 no LT', 'IPCC 2013', 'IPCC 2021 no LT', 'IPCC 2021', 'ReCiPe 2016 v1.0', 'TRACI v2.1 no LT', 'TRACI v2.1', 'USEtox no LT', 'USEtox']

    Then, your task is to choose top five relevent keywords in the below list of keywords that help to retrieve relevant LCIA methods of user question: {question}
    ['climate change no LT', 'global warming potential (GWP100) no LT', 'climate change', 'global warming potential (GWP100)', 'climate change: biogenic', 'climate change: fossil', 'climate change: land use and land use change', 'climate change: biogenic no LT', 'climate change: fossil no LT', 'climate change: land use and land use change no LT', 'global temperature change potential (GTP100) no LT', 'global temperature change potential (GTP20) no LT', 'global warming potential (GWP20) no LT', 'global temperature change potential (GTP100)', 'global temperature change potential (GTP20)', 'global warming potential (GWP20)', 'global temperature change potential (GTP50) no LT', 'global warming potential (GWP500) no LT', 'climate change: including SLCFs no LT', 'climate change: land use no LT', 'global temperature change potential (GTP50)', 'global warming potential (GWP500)', 'climate change: including SLCFs', 'climate change: land use']

    Output the methods and keywords in one line separated by commas. Do not use any other sentences or explanations.
    """
    prompt_methods = ChatPromptTemplate.from_template(method_search_template)

    # Unused full list of methods: ['acidification no LT', 'acidification (incl. fat', 'climate change no LT', 'global warming potential (GWP100) no LT', 'ecotoxicity: freshwater no LT', 'freshwater aquatic ecotoxicity (FAETP inf) no LT', 'ecotoxicity: marine no LT', 'marine aquatic ecotoxicity (MAETP inf) no LT', 'ecotoxicity: terrestrial no LT', 'terrestrial ecotoxicity (TETP inf) no LT', 'energy resources: non-renewable no LT', 'abiotic depletion potential (ADP): fossil fuels no LT', 'eutrophication no LT', 'eutrophication (fate not incl.) no LT', 'human toxicity no LT', 'human toxicity (HTP inf) no LT', 'material resources: metals/minerals no LT', 'abiotic depletion potential (ADP): elements (ultimate reserves) no LT', 'ozone depletion no LT', 'ozone layer depletion (ODP steady state) no LT', 'photochemical oxidant formation no LT', 'photochemical oxidation (high NOx) no LT', 'acidification', 'climate change', 'global warming potential (GWP100)', 'ecotoxicity: freshwater', 'freshwater aquatic ecotoxicity (FAETP inf)', 'ecotoxicity: marine', 'marine aquatic ecotoxicity (MAETP inf)', 'ecotoxicity: terrestrial', 'terrestrial ecotoxicity (TETP inf)', 'energy resources: non-renewable', 'abiotic depletion potential (ADP): fossil fuels', 'eutrophication', 'eutrophication (fate not incl.)', 'human toxicity', 'human toxicity (HTP inf)', 'material resources: metals/minerals', 'abiotic depletion potential (ADP): elements (ultimate reserves)', 'ozone depletion', 'ozone layer depletion (ODP steady state)', 'photochemical oxidant formation', 'photochemical oxidation (high NOx)', 'accumulated exceedance (AE)', 'climate change: biogenic', 'climate change: fossil', 'climate change: land use and land use change', 'comparative toxic unit for ecosystems (CTUe)', 'eutrophication: freshwater', 'fraction of nutrients reaching freshwater end compartment (P)', 'eutrophication: marine', 'fraction of nutrients reaching marine end compartment (N)', 'eutrophication: terrestrial', 'human toxicity: carcinogenic', 'comparative toxic unit for human (CTUh)', 'human toxicity: non-carcinogenic', 'ionising radiation: human health', 'human exposure efficiency relative to u235', 'land use', 'soil quality index', 'ozone depletion potential (ODP)', 'particulate matter formation', 'impact on human health', 'photochemical oxidant formation: human health', 'tropospheric ozone concentration increase', 'water use', 'user deprivation potential (deprivation-weighted water consumption)', 'accumulated exceedance (AE) no LT', 'climate change: biogenic no LT', 'climate change: fossil no LT', 'climate change: land use and land use change no LT', 'comparative toxic unit for ecosystems (CTUe) no LT', 'ecotoxicity: freshwate', 'norganics no LT', 'etals no LT', 'rganics no LT', 'eutrophication: freshwater no LT', 'fraction of nutrients reaching freshwater end compartment (P) no LT', 'eutrophication: marine no LT', 'fraction of nutrients reaching marine end compartment (N) no LT', 'eutrophication: terrestrial no LT', 'human toxicity: carcinogenic no LT', 'comparative toxic unit for human (CTUh) no LT', 'human toxicity: carcinogeni', 'human toxicity: non-carcinogenic no LT', 'human toxicity: non-carcinogeni', 'ionising radiation: human health no LT', 'human exposure efficiency relative to u235 no LT', 'land use no LT', 'soil quality index no LT', 'ozone depletion potential (ODP) no LT', 'particulate matter formation no LT', 'impact on human health no LT', 'photochemical oxidant formation: human health no LT', 'tropospheric ozone concentration increase no LT', 'water use no LT', 'user deprivation potential (deprivation-weighted water consumption) no LT', 'norganics', 'etals', 'rganics', 'global temperature change potential (GTP100) no LT', 'global temperature change potential (GTP20) no LT', 'global warming potential (GWP20) no LT', 'global temperature change potential (GTP100)', 'global temperature change potential (GTP20)', 'global warming potential (GWP20)', 'global temperature change potential (GTP50) no LT', 'global warming potential (GWP500) no LT', 'climate change: biogeni', 'ncluding SLCFs no LT', 'climate change: fossi', 'climate change: including SLCFs no LT', 'climate change: land use no LT', 'climate change: land us', 'global temperature change potential (GTP50)', 'global warming potential (GWP500)', 'ncluding SLCFs', 'climate change: including SLCFs', 'climate change: land use', 'ndpoint (E) no LT', 'ecosystem quality no LT', 'human health no LT', 'natural resources no LT', 'total: ecosystem quality no LT', 'total: human health no LT', 'total: natural resources no LT', 'ndpoint (E)', 'ecosystem quality', 'human health', 'natural resources', 'total: ecosystem quality', 'total: human health', 'total: natural resources', 'ndpoint (H) no LT', 'ndpoint (H)', 'ndpoint (I) no LT', 'ndpoint (I)', 'idpoint (E) no LT', 'acidification: terrestrial no LT', 'energy resources: non-renewabl', 'ionising radiation no LT', 'photochemical oxidant formation: terrestrial ecosystems no LT', 'idpoint (E)', 'acidification: terrestrial', 'ionising radiation', 'photochemical oxidant formation: terrestrial ecosystems', 'idpoint (H) no LT', 'idpoint (H)', 'idpoint (I) no LT', 'idpoint (I)', 'acidification potential (AP) no LT', 'eutrophication potential no LT', 'particulate matter formation potential (PMFP) no LT', 'maximum incremental reactivity (MIR) no LT', 'acidification potential (AP)', 'eutrophication potential', 'particulate matter formation potential (PMFP)', 'maximum incremental reactivity (MIR)', 'ecotoxicity no LT', 'total no LT', 'carcinogenic no LT', 'non-carcinogenic no LT', 'ecotoxicity', 'total', 'carcinogenic', 'non-carcinogenic']

    generate_methods = (
        prompt_methods
        | gpt5_client
        | StrOutputParser()
    )
    try:
        methods_output = generate_methods.invoke({"question": question})
    except:
        #rerun-it to prevent occasional LLM generation failure
        methods_output = generate_methods.invoke({"question": question})
        
    methods_list = retriever_lca_methods.invoke(methods_output)
    
    return methods_list

# -------------- To retrieve the best sets of activity that align with user's requests ----------------#
def join_activities_by_stage(activities_by_stage: dict) -> str:
    """Format stage-grouped activity docs into a string for LLM consumption."""
    parts = []
    for stage, docs in activities_by_stage.items():
        names = "; ".join(doc.page_content for doc in docs)
        parts.append(f"[{stage}]: {names}")
    return "\n".join(parts)


def _rerank_eco_activities(query: str, docs: list) -> list:
    """
    Re-rank a small list of ecoinvent activity candidate documents by keyword-
    overlap relevance against the stage retrieval query.

    Vector-similarity search alone can surface false positives because short
    technical activity names share embedding space with unrelated activities
    that happen to contain overlapping general terms (e.g. "power supply unit,
    for desktop computer" surfacing for an EV powertrain query). This function
    re-scores each candidate by counting how many meaningful tokens from the
    query appear in the activity name, then sorts descending. Ties are broken
    by preserving the original vector rank so that among equal-overlap
    candidates the best cosine match still wins.

    Note: a neural cross-encoder (e.g. ms-marco-MiniLM) is NOT used here
    because it is trained on web passages, not on 2-8 word ecoinvent technical
    labels — it would add latency and degrade ranking quality for this domain.

    Parameters
    ----------
    query : str
        The full stage retrieval query (product + stage name + stage description).
    docs : list[Document]
        Candidate documents returned by the vector retriever.

    Returns
    -------
    list[Document]
        Same documents, re-ordered by descending keyword-overlap score.
    """
    _STOP = frozenset({
        'a', 'an', 'at', 'be', 'by', 'for', 'in', 'is', 'it', 'its', 'of',
        'or', 'the', 'to', 'are', 'and', 'but', 'from', 'into', 'this', 'that',
        'with', 'used', 'which', 'have', 'been', 'also', 'such', 'these',
        'their', 'each', 'more', 'both', 'some', 'than', 'then', 'them',
        'stage', 'process', 'involves', 'include', 'includes', 'typically',
        'describes', 'undergoes', 'products', 'using', 'during', 'where',
    })

    def _tokens(text: str) -> set:
        return {t.lower() for t in re.split(r'[\s,;/\-\(\)\.]+', text)
                if len(t) > 3 and t.lower() not in _STOP}

    query_tokens = _tokens(query)

    scored: list = []
    for rank, doc in enumerate(docs):
        name_tokens = _tokens(doc.page_content)
        overlap = len(query_tokens & name_tokens)
        # Negative rank so lower original vector rank breaks ties in favour of
        # the better cosine match when keyword overlap is equal.
        scored.append((overlap, -rank, doc))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [doc for _, _, doc in scored]


def generate_list_of_activities(gpt5_client, retriever_eco_activity, product, user_input_params):
    stage_description_template = """
    You are a life-cycle assessment (LCA) expert. Based on the product and user request parameters below,
    identify the LCA stages within the requested system boundary and write a short paragraph for each stage
    describing the key industrial processes that {product} typically undergoes at that stage.

    User Request Parameters:
    {user_input_params}
    
    # Important note: "further_notes" in User Request Parameters refers to user's actual query. If there is any specific LCI activity or keywords requested for LCI activity search, you must note the full LCI activity name or keywords in your reply for precise retrieval of relevant LCI activities in the next step.
    # Following the previous line of instruction, if there is a very specific LCI activity requested by the user, you may response with a single stage with the stage name as "general" and the stage description as the specific LCI activity name or keywords requested by the user for precise retrieval of relevant LCI activities in the next step.
    # If the user request is general, like "What are the lifecycle emissions for steel production?", you can provide the response as what is indicated in following instructions.
    
    For your reference, there are only four possible system boundaries: "cradle-to-gate", "cradle-to-grave", "gate-to-gate", and "cradle-to-cradle", and you should learn the stages associated with each system boundary in the following knowledge text provided:
    ## Cradle-to-Gate
    - Raw material extraction (mining, harvesting, primary production)
    - Material processing/beneficiation (ore concentration, refining, intermediate chemicals)
    - Manufacturing/production (smelting, fabrication, assembly)
    - Gate (finished product leaving the factory)
    ## Cradle-to-Grave
    - Raw material extraction
    - Material processing
    - Manufacturing/production
    - Distribution/transport
    - Use phase (operation, maintenance, energy consumption during use)
    - End-of-life (disposal, recycling, incineration, landfill)
    ## Gate-to-Gate
    - Single process or facility boundary (e.g., only the smelting plant or only the assembly line)
    - Focuses on inputs/outputs of that specific stage without upstream or downstream
    ## Cradle-to-Cradle (Circular economy perspective)
    - Raw material extraction
    - Processing/manufacturing
    - Use phase
    - End-of-life recovery (recycling, remanufacturing, closed-loop material flows)

    Based on the "system_boundary" in the user parameters (default to "cradle-to-grave" if not specified),
    output a JSON object where:
    - Each key is a stage name (e.g., "raw material extraction", "processing", "manufacturing", "transport", "use phase", "end-of-life")
    - Each value is a short paragraph describing the specific activities and processes for {product} at that stage

    Only include stages relevant to the system boundary. Output only the JSON object. No other text.
    """
    prompt_stages = ChatPromptTemplate.from_template(stage_description_template)
    generate_stages = (
        prompt_stages
        | gpt5_client
        | StrOutputParser()
    )

    #prevent occasional LLM generation failure by rerunning it once if it fails at the first time
    try:
        stages_json_str = generate_stages.invoke({"product": product, "user_input_params": user_input_params})
    except:
        stages_json_str = generate_stages.invoke({"product": product, "user_input_params": user_input_params})

    try:
        clean = re.sub(r"```(?:json)?|```", "", stages_json_str).strip()
        stages_dict = json.loads(clean)
    except json.JSONDecodeError:
        stages_dict = {"general": f"Key lifecycle activities for {product} production"}

    activities_by_stage = {}
    for stage, description in stages_dict.items():
        query = f"{product} {stage}: {description}"
        docs = retriever_eco_activity.invoke(query)
        docs = _rerank_eco_activities(query, docs)
        activities_by_stage[stage] = docs

    return activities_by_stage


def _remove_market_activity_duplicates(functional_unit: list) -> list:
    """
    Remove 'market for X' activities that duplicate an already-present
    production/operation activity in the functional unit.

    In ecoinvent, 'market for X' is a background supply-mix aggregator that
    internally calls the same production datasets you have already explicitly
    listed.  Including both double-counts every upstream impact for material X
    and makes the mass-flow sequence appear to go backwards.

    Algorithm
    ---------
    For each 'market for [material]' entry, extract the core material tokens
    and compare them against every non-market activity already in the list.
    If >= 2 significant tokens match (or 1 token with > 6 characters that
    unambiguously identifies the material), the market activity is removed and
    a warning is logged.  The final-product market activity is only removed
    when a production activity for the same final product is also present.
    """
    _MARKET_RE = re.compile(r'^market\s+for\s+', re.IGNORECASE)
    _STOP = frozenset({'for', 'the', 'and', 'with', 'from', 'into', 'used',
                       'market', 'activity', 'treatment'})

    def _tokens(name: str) -> set:
        return {t for t in re.split(r'[\s,;/\-\(\)]+', name.lower())
                if len(t) > 3 and t not in _STOP}

    non_market = [(i, act) for i, act in enumerate(functional_unit)
                  if not _MARKET_RE.match(act.get("activity_name", ""))]

    to_remove: set = set()
    for i, act in enumerate(functional_unit):
        name = act.get("activity_name", "")
        if not _MARKET_RE.match(name):
            continue
        material_part = _MARKET_RE.sub("", name).strip()
        m_tokens = _tokens(material_part)
        if not m_tokens:
            continue
        for j, other_act in non_market:
            if i == j:
                continue
            o_tokens = _tokens(other_act.get("activity_name", ""))
            overlap = m_tokens & o_tokens
            # Remove if >= 2 tokens overlap, or a single highly-specific token (>6 chars) matches
            if len(overlap) >= 2 or (len(overlap) == 1 and any(len(t) > 6 for t in overlap)):
                to_remove.add(i)
                logging.warning(
                    "Anti-double-counting: removing 'market for' activity '%s' "
                    "because production activity '%s' already covers material '%s' "
                    "(overlapping tokens: %s). Including both would double-count "
                    "upstream impacts and break mass-flow continuity.",
                    name,
                    other_act.get("activity_name", ""),
                    material_part,
                    sorted(overlap),
                )
                break

    if to_remove:
        removed_names = [functional_unit[i]["activity_name"] for i in sorted(to_remove)]
        logging.warning(
            "Removed %d duplicate 'market for' activit%s from functional unit: %s",
            len(to_remove),
            "y" if len(to_remove) == 1 else "ies",
            removed_names,
        )

    return [act for i, act in enumerate(functional_unit) if i not in to_remove]


def _refine_activities_with_ref_product_critique(
    gpt5_client,
    functional_unit: list,
    product: str,
    activities_by_stage: dict,
    ref_product_lookup: dict,
    retriever_eco_activity,
) -> list:
    """
    Second-pass LLM quality gate: validates each activity's ecoinvent
    reference_product against its assigned LCA stage and the product being studied.

    Decision strategy
    -----------------
    KEEP      — activity is valid (real ecoinvent name, correct stage, correct technology).
    SUBSTITUTE — the stage/technology assignment is conceptually right but the activity name
                 may be slightly wrong, unverifiable, or missing from BW2 close-matches.
                 The function re-queries the vectorstore with a targeted query
                 (product + stage + original name) to find the best real ecoinvent name,
                 preserving the slot in the mass-flow chain.
    OMIT      — reserved ONLY for fundamental stage/technology mismatches that a re-retrieval
                 cannot fix: e.g., a maintenance/servicing activity in a manufacturing stage,
                 or a process for a completely different product technology.
                 Absence from the candidate sample is NOT grounds for omission.

    Returns a refined functional_unit list.  Falls back to the original list
    on any LLM or parsing failure.
    """
    # ── Step 0: drop "unit"-scale duplicates when a mass-based twin exists ──────
    # e.g. "powertrain production, for electric passenger car, 1.0 unit" is
    # dropped when the same name also appears with unit="kilogram".
    _MASS_UNITS = frozenset({
        "kilogram", "kg", "gram", "g", "tonne", "ton", "metric ton",
        "metric tonne", "pound", "lb",
    })
    _name_unit_sets: dict = {}
    for _a in functional_unit:
        _n = _a.get("activity_name", "")
        _u = (_a.get("unit") or "").lower().strip()
        _name_unit_sets.setdefault(_n, set()).add(_u)
    _has_mass_twin = {
        _n for _n, _us in _name_unit_sets.items()
        if any(_u in _MASS_UNITS for _u in _us)
    }
    _pre_filtered = []
    for _a in functional_unit:
        _n = _a.get("activity_name", "")
        _u = (_a.get("unit") or "").lower().strip()
        if _u == "unit" and _n in _has_mass_twin:
            logging.info(
                "[ref_product_critique] Pre-filtering unit-scale duplicate of '%s'"
                " (mass-based version present)", _n,
            )
            continue
        _pre_filtered.append(_a)
    functional_unit = _pre_filtered

    # Build the per-activity review payload
    review_items = []
    for act in functional_unit:
        name = act.get("activity_name", "")
        stage = act.get("stage", "")

        ref_info = ref_product_lookup.get(name, {})
        ref_product = ref_info.get("reference_product", "")
        close_matches = ref_info.get("close_matches", [])

        # Retrieval candidates are a SAMPLE, not an exhaustive list
        stage_docs = activities_by_stage.get(stage, [])
        candidates = [d.page_content for d in stage_docs]
        

        review_items.append({
            "activity_name": name,
            "stage": stage,
            "unit": act.get("unit", ""),
            "inferred_reference_product": ref_product,
            "bw2_close_matches": [
                {
                    "name": m["name"],
                    "reference_product": m["reference_product"],
                    "unit": m["unit"],
                }
                for m in close_matches[:3]
            ],
            "retrieval_candidates_for_stage": candidates[:10],
        })

    critique_template = """You are a senior LCA expert and ecoinvent database specialist reviewing an LCA activity list for the product: {product}.

CRITICAL CONTEXT: The retrieval_candidates_for_stage list is a small SAMPLE (≤10) drawn from tens of thousands of ecoinvent datasets — it is NOT a complete list of valid activities. An activity name that is absent from this sample may still be a perfectly real and correct ecoinvent dataset. Judge validity using your expert LCA knowledge, not solely by list membership.

For each activity you are given:
- Its assigned **stage** in the lifecycle
- Its **inferred_reference_product** from BW2 (what the dataset actually produces — may be empty)
- **bw2_close_matches** (closest real ecoinvent datasets by name similarity)
- **retrieval_candidates_for_stage** (a sample of valid ecoinvent names for that stage)

## Decision rules — read carefully:

### KEEP — use when the activity is valid:
- The name is a known or plausible ecoinvent dataset for this stage
- The reference_product confirm the process produces/provides what is expected at this stage, and the current activity is the most appropriate one to describe the process among the closest matches activities.
- The technology matches {product} (e.g., electric drive components for an EV LCA)
- Minor name uncertainty is NOT a reason to avoid KEEP if the concept is clearly right

### SUBSTITUTE — use when the name may be imprecise or unverifiable, but the function is right:
- The stage assignment is conceptually correct for {product}
- The underlying industrial process clearly belongs at this stage
- There is a better and more precisely accurate activity listed in the closest matches activities that should replace the current activity.
- An activity with `unit = "unit"` (discrete count) when the same activity name also appears with a mass-based unit (kilogram, tonne, etc.) elsewhere inside the "bw2_close_matches" objects in the list — you must indicate in the search hint that the correct activity should be the mass-based version, not the unit-based one, with the same activity name.
- A targeted re-retrieval from the vectorstore (done automatically after this step) will find the best real ecoinvent name
- Examples where SUBSTITUTE is correct: "blast furnace production" (stage right, name slightly off), "iron sinter production" (real process, name needs verification)
- You MUST provide a **search_hint**: a short, precise English description of what the correct ecoinvent activity should produce or do at this stage. This is used as the retrieval query — write it as a description of the DESIRED process function, NOT a repeat of the wrong name. Good examples:
  - For "blast furnace production": search_hint = "iron smelting in blast furnace using coke and iron ore"
  - For "iron sinter production": search_hint = "sintering of iron ore fines for blast furnace feed"
  - For "blast oxygen furnace converter production": search_hint = "basic oxygen steelmaking converter crude steel from pig iron"

### OMIT — use ONLY for fundamental mismatches that re-retrieval cannot fix:
- A MAINTENANCE, SERVICING, or REPAIR activity assigned to a MANUFACTURING or RAW MATERIAL stage
- A process for a COMPLETELY DIFFERENT product technology (e.g., gasoline engine components in an electric vehicle LCA)
- An END-OF-LIFE or DISPOSAL activity assigned to a PRODUCTION or RAW MATERIAL stage
- Absence from the candidate sample alone is NEVER grounds for omission
- An activity with `unit = "unit"` (discrete count) when the same activity name also appears with a mass-based unit (kilogram, tonne, etc.) elsewhere in the "activity_name" key's value in the list (not bw2_close_matches listed activities) — OMIT the unit-based variant; the mass-based version covers the same process

## Concrete examples for {product}:
- "maintenance, passenger car" in "manufacturing" → OMIT (servicing ≠ production)
- "treatment of scrap steel" in "use phase" → OMIT (disposal ≠ use)
- "desktop computer production" in "manufacturing" of an EV → OMIT (wrong technology)
- "iron sinter production" in "raw material extraction" → SUBSTITUTE (real process, name needs re-retrieval)
- "blast furnace production" in "manufacturing" → SUBSTITUTE (correct stage, name may be off)
- "steel production, converter, unalloyed" in "manufacturing" → KEEP (known valid ecoinvent dataset)
- "hot rolling, steel" in "manufacturing" → KEEP (legitimate shaping process)
- "hard coal mine operation" in "raw material extraction" → KEEP (valid)

## Activities to review:
{review_items}

Return a JSON array with one object per activity: {{"activity_name": "...", "decision": "keep"|"substitute"|"omit", "search_hint": null|"description of desired process", "reason": "one sentence"}}.
For SUBSTITUTE decisions, search_hint must be a precise description of what the correct ecoinvent activity should produce or do — describe the desired process function, not the wrong name.
For KEEP and OMIT decisions, set search_hint to null.
Return ONLY the JSON array. No prose, no markdown fences."""

    prompt = ChatPromptTemplate.from_template(critique_template)
    chain = prompt | gpt5_client | StrOutputParser()

    try:
        critique_result = chain.invoke({
            "product": product,
            "review_items": json.dumps(review_items, indent=2, ensure_ascii=False),
        })
    except Exception as exc:
        logging.warning("Reference-product critique LLM call failed — keeping original activities: %s", exc)
        return functional_unit

    # Parse decisions
    try:
        clean = re.sub(r"```(?:json)?|```", "", critique_result).strip()
        decisions = json.loads(clean)
        if not isinstance(decisions, list):
            raise ValueError("Expected a JSON array")
    except Exception as exc:
        logging.warning("Could not parse reference-product critique output — keeping original activities: %s", exc)
        return functional_unit

    decision_map = {
        d["activity_name"]: d
        for d in decisions
        if isinstance(d, dict) and "activity_name" in d
    }

    refined = []
    n_substituted = 0
    n_omitted = 0
    for act in functional_unit:
        name = act.get("activity_name", "")
        stage = act.get("stage", "")
        decision = decision_map.get(name, {})
        action = decision.get("decision", "keep")
        reason = decision.get("reason", "")

        if action == "omit":
            logging.info("[ref_product_critique] Omitting '%s': %s", name, reason)
            n_omitted += 1
            continue

        if action == "substitute":
            # Build a retrieval query from the LLM's search_hint — a semantic
            # description of the DESIRED process function written by the LLM.
            # Using the original (wrong) name as the query would simply return
            # itself as the top hit, so we use the hint instead.
            search_hint = (decision.get("search_hint") or "").strip()
            if search_hint:
                retr_query = f"{product} {stage}: {search_hint}"
            else:
                # Fallback: describe the stage generically without the wrong name
                retr_query = f"{product} {stage} process"
            new_name = None
            try:
                matches = retriever_eco_activity.invoke(retr_query)
                if isinstance(matches, list) and matches:
                    new_name = matches[0].page_content
                elif hasattr(matches, "page_content"):
                    new_name = matches.page_content
            except Exception as exc:
                logging.warning(
                    "[ref_product_critique] Re-retrieval failed for '%s': %s — keeping original name",
                    name, exc,
                )

            if new_name and new_name != name:
                logging.info(
                    "[ref_product_critique] Re-retrieved '%s' → '%s' (stage: %s): %s",
                    name, new_name, stage, reason,
                )
                act = dict(act)
                act["activity_name"] = new_name
                n_substituted += 1
            else:
                logging.info(
                    "[ref_product_critique] Re-retrieval for '%s' returned same or empty — keeping original name",
                    name,
                )

        refined.append(act)

    logging.info(
        "[ref_product_critique] Review complete: %d kept, %d re-retrieved, %d omitted (of %d original)",
        len(refined) - n_substituted, n_substituted, n_omitted, len(functional_unit),
    )

    # ── Deduplicate: a substitution may have pointed two slots to the same name
    seen_names: set = set()
    deduped: list = []
    for act in refined:
        n = act.get("activity_name", "")
        if n in seen_names:
            logging.info("[ref_product_critique] Dropping post-substitution duplicate '%s'", n)
            continue
        seen_names.add(n)
        deduped.append(act)

    # ── Re-stitch mass flow chain after deduplication ─────────────────────────
    # For consecutive mass-bearing activities, set mass_in_kg[i] = mass_out_kg[i-1]
    # and recompute mass_out_kg[i] = mass_in_kg[i] * recovery_rate[i].
    for i in range(1, len(deduped)):
        prev = deduped[i - 1]
        curr = deduped[i]
        if ((prev.get("unit") or "").lower() in _MASS_UNITS and
                (curr.get("unit") or "").lower() in _MASS_UNITS):
            prev_out = prev.get("mass_out_kg") or 0.0
            if prev_out > 0.0:
                recovery = curr.get("recovery_rate") or 1.0
                curr = dict(curr)
                curr["mass_in_kg"] = prev_out
                curr["mass_out_kg"] = round(prev_out * recovery, 6)
                deduped[i] = curr

    return deduped


def generate_response_bw2_input(
    gpt5_client,
    query_decomposition_retrival,
    LCIA_method_list,
    retrieved_acitivity_list,
    product,
    LCI_DATABASE,
    retriever_eco_activity,
    user_input_params=None,
):
    print("Generating Brightway2 input JSON...")

    params = user_input_params or {}
    resolved_n_simulations = int(params.get("n_simulations", 25))
    
    final_template_1st = """
    You are an assistant that converts lifecycle assessment (LCA) queries into structured JSON instructions for the Brightway2 LCA engine.

    ## Your Task
    - Read the section **User Requests** carefully. You should primarily rely on this section to insert the values for the required fields in the output JSON.
    - Regarding the activities inside the functional unit section for a complete value chain in your output JSON, you MUST copy the `activity_name` for each process **VERBATIM** (exact spelling, exact lower-case formatting, exact punctuation) from the section **List of available activities by stages** below. The `activity_name` field is used as a direct lookup key by the Brightway2 database engine — any modification (capitalisation, paraphrasing, rewording, translation, or adding/removing words) will cause BW2 to fail the exact match and fall back to a wrong ecoinvent dataset. From your LCA expertise, select the most relevant activities for each stage. Choose at least one activity per stage, and include as many additional activities as possible in the list if they are related and do not conflict. Ensure the selected activities together form a complete and detailed supply chain. Do not repeat any activity in your JSON output. Insert each selected activity into the functional unit section of the output JSON in order with the stages.
    - **CRITICAL — do NOT invent activity names that are not in the List of available activities by stages.** If no suitable activity is available in the retrieved list for a particular stage, omit that stage entirely rather than generating a plausible-sounding name. A missing stage produces zero impact contribution; a wrong stage match can produce physically impossible results (e.g., −76,000 kg CO₂-eq from a mismatched nuclear-fuel dataset).
    - The section **Supplementary Context** are ground-truth sources that are retrieved from peer-reviewed journals or facts. You may use them to revise your identified key activities and their associated stages, validate them, and make them be more accurate.
    - Finally, output the result as a structured JSON object.

    ## Input Sections
    ### User Requests
    {user_input_params}
    
    ** Important note for User Requests: if "further notes" exists in User Requests information, you must pay highest attention to it as it is the raw user query and may contain specific request for constructing the following bw2 input template, such as specific LCI activity, locations, and other parameters needed for constructing the bw2 input template in your response.

    ### List of available activities by stages (note: activities in each stage are separated by semicolons)
    {retrieved_acitivity_list}

    ### Supplementary Context
    {context}
    
    
    ### Typical LCA values for recovery rates if not provided in the Supplementary Context:
    a) ORE MINING & BENEFICIATION (Flotation/Concentration):
        - Base Metals (Sulfide Cu, Ni, Zn, Pb): 0.80 - 0.95
        - Base Metals (Oxide Leaching): 0.65 - 0.85
        - Precious Metals (Au, Ag, PGMs): 0.85 - 0.95
        - Iron Ore: 0.75 - 0.90
        - Critical Minerals / REEs / Lithium Spodumene: 0.60 - 0.82
        - Sand & Gravel Quarrying (Aggregate Crushing & Screening): 0.88 - 0.95
        - Limestone Quarrying & Crushing (Construction / Cement Grade): 0.90 - 0.96
        - Bauxite Open-Cast Mining & Primary Screening: 0.88 - 0.94
        - Phosphate Rock Wet-Process Flotation Beneficiation: 0.72 - 0.85
        - Kaolin / Ball Clay Primary Processing (Washing & Screening): 0.70 - 0.82
        *Note: If calculating from raw run-of-mine (ROM) ore mass, multiply recovery rate by Ore Grade (Base: 0.004-0.02; Precious: 0.000001-0.00001).
    
    b) SMELTING & HYDRO/PYROMETALLURGICAL EXTRACTION:
        - Flash/Bath Smelting (Cu, Ni): 0.95 - 0.98
        - Iron/Steel Blast Furnace: 0.96 - 0.99
        - Aluminum Smelting (Alumina to Metal): 0.98 - 0.995 (Bauxite to Alumina: 0.40 - 0.50)
        - REE Solvent Extraction Cascades: 0.85 - 0.93
        - Zinc Roast-Leach-Electrowin (RLE) Process: 0.94 - 0.97
        - Lead Blast Furnace Smelting & Softening Refinery: 0.94 - 0.97
        - Manganese Ferroalloy Smelting (Submerged Arc Furnace): 0.88 - 0.94
    
    c) REFINING & INTERMEDIATE CHEMICAL SYNTHESIS:
        - Electrolytic Refining (Cu, Pb, Zn): 0.98 - 0.995
        - Precious Metals (Aqua Regia/Wohlwill): 0.995 - 0.999
        - Electronic/PV Grade Silicon Ingots: 0.85 - 0.95
        - Battery Chemicals (Lithium Carbonate/Hydroxide): 0.80 - 0.90
        - Battery Active Material Synthesis (pCAM to CAM): 0.92 - 0.97
        - Industrial Organic Chemical Synthesis (Batch): 0.80 - 0.95
        - Solvent Distillation & On-Site Recovery: 0.75 - 0.92
        - Zinc Electrowinning (SHG-Grade, Post-Purification): 0.96 - 0.98
        - Nickel Carbonyl Refining (Mond Process): 0.97 - 0.99
    
    d) ELECTRONICS & SEMICONDUCTOR MANUFACTURING:
        - Silicon Ingot Slicing (Wafer Production): 0.45 - 0.55
        - Printed Circuit Board (PCB) Fabrication (Etching/Trimming): 0.75 - 0.88
        - SMT Component Placement / Assembly: 0.97 - 0.99
        - Cable & Wire Harness Assembly: 0.92 - 0.96
    
    e) POLYMERS & ADVANCED COMPOSITES PROCESSING:
        - Plastic Injection Molding: 0.94 - 0.98
        - Plastic Extrusion (Film/Blow Molding): 0.96 - 0.99
        - Carbon Fiber Reinforced Polymer (CFRP) Layup: 0.65 - 0.85
    
    f) METALLURGICAL FABRICATION / MANUFACTURING / SHAPING:
        - Primary Casting, Rolling, & Extrusion: 0.90 - 0.97
        - Metal Sheet Stamping / Deep Drawing: 0.55 - 0.75
        - High-Precision Machining (Subtractive): 0.40 - 0.85
        - Powder Metallurgy / Sintering: 0.95 - 0.98
        - CVD / PVD Coating Deposition: 0.15 - 0.40
    
    g) RECYCLING & SECONDARY MATERIAL PROCESSING:
        - Mechanical Sorting / Shredder Separation: 0.70 - 0.90
        - Hydrometallurgical Recycling (Battery NMC/Co): 0.92 - 0.98
        - Hydrometallurgical Recycling (Battery Lithium): 0.80 - 0.90
        - E-Waste Pyrometallurgical Smelting: 0.90 - 0.96
    
    h) GLASS, CERAMICS, & INORGANIC INSULATION:
        - Container / Hollow Glass Production: 0.85 - 0.92
        - Flat / Float Glass Production (Windows, PV Substrates): 0.78 - 0.88
        - Technical Ceramics (Firing & Machining): 0.70 - 0.85
        - Mineral Wool Fiber Insulation: 0.90 - 0.95
        
    i) TEXTILES & APPAREL FABRICATION:
        - Natural Fiber Spinning (Cotton/Wool Yarn): 0.68 - 0.82
        - Synthetic Fiber Melt Extrusion (PET, Nylon): 0.95 - 0.98
        - Fabric Weaving & Knitting: 0.90 - 0.95
        - Textile Cutting & Garment Assembly: 0.75 - 0.85
    
    j) PULP, PAPER, & BIO-BASED MATERIALS:
        - Chemical Wood Pulping (Kraft Process): 0.45 - 0.52
        - Paper & Cardboard Machine Forming: 0.88 - 0.94
        - Box Converting & Corrugated Board Cutting: 0.85 - 0.92
    
    k) INDUSTRIAL COATING & FINAL ASSEMBLY:
        - Automated Adhesive / Sealant Dispensing: 0.92 - 0.97
        - Industrial Spray Painting / Coating (Transfer Efficiency): 0.35 - 0.65
        - Interior Wall Paint Application (Roller / Brush Method): 0.92 - 0.97
        - Exterior Masonry Render & Stucco Wall Coating Application: 0.85 - 0.93
        - Architectural Powder Coating (Electrostatic Spray, Transfer Efficiency): 0.95 - 0.99
        - Final Mechanical Assembly Integration: 0.99 - 1.00
        
        
    l) CONSTRUCTION & BULK MINERALS INFRASTRUCTURE:
        - Cement Raw Material Preparation (Quarrying/Crushing): 0.35 - 0.45
        - Clinker Pyro-processing & Rotary Kiln Calcination: 0.35 - 0.40
        - Cement Finish Grinding & Blending: 0.22 - 0.28
        - Concrete Batching & Readymix Plant Mixing: 0.96 - 0.98
        - Brick & Structural Clay Kiln Firing: 0.88 - 0.94
        - Fired Clay Brick Wet/Stiff Extrusion & Tunnel Kiln Processing (Clay to Product): 0.80 - 0.88
        - Clay Roof Tile Extrusion & Kiln Firing: 0.82 - 0.90
        - Quicklime Production via Rotary/Shaft Kiln Calcination (CaO mass yield from CaCO₃): 0.54 - 0.58
        - Gypsum Rock Calcination to β-Hemihydrate Plaster / Stucco: 0.82 - 0.88
        - Dimension Stone Cutting & Quarry Slab Dressing: 0.40 - 0.60
        *Note: Quicklime recovery of 0.54-0.58 reflects stoichiometric CO₂ release during CaCO₃ calcination (not a process loss); gypsum value reflects bound water loss during calcination.
        
    m) PRECISION ADDITIVE MANUFACTURING (AM):
        - Laser Powder Bed Fusion (L-PBF) Metal Printing: 0.94 - 0.98
        - Fused Filament Fabrication (FFF/FDM) Thermoplastic Printing: 0.85 - 0.92
        - Stereolithography (SLA) Liquid Resin Photopolymerization: 0.90 - 0.96
        - Wire Arc Additive Manufacturing (WAAM): 0.88 - 0.95
        *Note: AM recovery depends heavily on post-processing support structure removal and powder/resin sieve recycling loop efficiencies.
    
    n) FOOD, AGRO-PROCESSING, & INDUSTRIAL BIOMASS CONVERSION:
        - Biomass Acid Pretreatment & Sugar Extraction (e.g., Straw to Glucose): 0.45 - 0.55
        - Industrial Starch Production (Wet Milling Corn/Wheat): 0.60 - 0.70
        - Oilseed Mechanical Pressing & Solvent Extraction (Crude Oil Yield): 0.18 - 0.42
        - High-Value Bio-Chemical Synthesis (e.g., HMF to FDCA Oxidation): 0.95 - 0.99
        - Fruit/Vegetable Commercial Juicing & Concentration: 0.45 - 0.65
        - Grain Milling & Flour Refining (Wheat to White Flour): 0.72 - 0.78
    
    o) PHARMACEUTICAL & SPECIALTY FINE CHEMICALS:
        - Active Pharmaceutical Ingredient (API) Multi-Step Organic Synthesis: 0.10 - 0.45
        - Bioactive Plant Extraction (Microwave/Supercritical Fluid): 0.01 - 0.15
        - Chromatographic Purification & Crystallization (Fine Chemicals): 0.60 - 0.85
        - Automated Liquid Formulation & Dose Filling: 0.97 - 0.99
    
    p) ADVANCED WATER TREATMENT & DESALINATION SEPARATION:
        - Seawater Reverse Osmosis (SWRO) Desalination (Permeate Yield): 0.40 - 0.50
        - Brackish Water Reverse Osmosis (BWRO) Desalination: 0.75 - 0.85
        - Multi-Effect Distillation (MED) Thermal Desalination: 0.20 - 0.40
        - Industrial Ion Exchange Demineralization: 0.90 - 0.95
    
    q) MUNICIPAL SOLID WASTE SEPARATION FACILITIES (MRFs):
        - Single-Stream MRF Automated Optical Sortation (PET/HDPE): 0.82 - 0.90
        - Mixed Waste Mechanical Processing Facility (Fiber/Paper Drop): 0.55 - 0.70
        - Glass Density & Air Classification Separation (Cullet Sort): 0.65 - 0.80
        - Eddy Current Non-Ferrous Metal Sorting from Shredded MSW: 0.85 - 0.93
    
    r) SPECIALTY METALLURGY, REFRACTORY METALS, & CO-PRODUCTS:
        - Titanium Sponge Production (Kroll Process via TiCl4): 0.78 - 0.88
        - Magnesium Extraction (Pidgeon Silicothermic Process): 0.75 - 0.85
        - Tungsten Powder Production (APT Hydrogen Reduction): 0.94 - 0.98
        - Secondary Precious Metals Slime Electrolysis (Anode Slime Loop): 0.88 - 0.94
        - Anodizing / Surface Passivation Layering (Anodic Yield): 0.82 - 0.90
        
    s) INDUSTRIAL GASES & CRYOGENIC FRACTIONATION:
        - Atmospheric Air Separation (ASU Cryogenic Distillation for O2/N2): 0.18 - 0.22
        - Steam Methane Reforming (SMR Hydrogen Yield based on Carbon Balance): 0.28 - 0.35
        - Carbon Capture & Storage (CCS Amine Scrubbing Absorption Efficiency): 0.85 - 0.92
    
    t) HEAVY MECHANICAL STRUCTURAL MANUFACTURING:
        - Open-Die & Closed-Die Heavy Metal Forging: 0.70 - 0.85
        - Large-Scale Industrial Sand Casting (Iron/Steel Foundry): 0.82 - 0.92
        - Structural Profile Laser / Plasma Cutting (Skeleton Waste): 0.65 - 0.80
        - Thermal Spray / Plasma Arc Coating Application: 0.45 - 0.65
    
    u) SILICON PHOTOVOLTAIC (PV) & GRAPHITE SUPPLY CHAINS:
        - Quartzite Reduction to Metallurgical Grade Silicon (MG-Si): 0.78 - 0.85
        - Trichlorosilane Distillation to Poly-Silicon (Siemens Process): 0.88 - 0.94
        - Synthetic Graphite Calcination & Graphitization (Petroleum Coke Base): 0.60 - 0.70
        - Natural Flake Graphite Spheroidization (Battery Anode Precursor): 0.35 - 0.50
    
    v) PRINTING, PACKAGING, & GRAPHIC MEDIA MANUFACTURING:
        - Commercial Web Offset Printing (Paper Runaround & Web Breaks): 0.88 - 0.94
        - Flexible Multi-Layer Packaging Lamination (Plastic/Foil/Paper): 0.90 - 0.96
        - Industrial Cardboard Die-Cutting & Structural Creasing: 0.78 - 0.88
        

    ## Output Requirements
    Regarding the output JSON, please strictly follows the following instructions:
    - "goal": Always "lifecycle impact assessment"  
    - "Project": Insert the project name as "{database}"  
    - "method": Select the most relevant LCIA methods from provided list: {methods}. If no specific impact category is requested, default to a climate change / GWP100 method (e.g., IPCC 2021, EF v3.1).
    - "system_boundary": reference to the user's strcutured parameter "system_boundary" if provided, otherwise default to "cradle-to-grave".
    - "functional_unit": A list of multiple activity objects, each describing one process involved in the product system:  
    Each object includes:  
    - "activity_name": The name of a process of the given product that usually involves in the product value chain and its lifecycle.
    - "unit": Use units that are commonly used in LCA databases, e.g., "kilogram" for mass-bearing steps, "ton kilometer" for transport, "kilowatt hour" for energy.
    - "location": ISO country code. Select the most appropriate ISO country or region code based on 'region' information in the User Requests section; available codes are Global:`GLO`, Europe:`RER`, Western Europe:`WEU`, Eastern Europe:`EEU`, Middle East & Africa:`MEA`, Asia:`ASI`, North America:`NAM`, South America:`SAM`, Switzerland:`CH`, China:`CN`, United States:`US`, Brazil:`BR`, India:`IN`; if no region is specified, default to `GLO`.
    - "stage": The value chain stage (e.g., "raw material extraction", "manufacturing", "transport", "use", "disposal")

    **CRITICAL — 1-tonne (1000 kg) reference basis rule**: The functional unit template MUST ALWAYS be computed on a fixed 1-tonne (1000 kg) output reference basis, regardless of the quantity the user has requested. The "amount" and "unit" fields in the User Requests section (e.g., "amount": "10", "unit": "tonne") describe the USER'S DESIRED DISPLAY QUANTITY — they are NOT the target production mass for this template. A separate downstream scale_factor (= requested_quantity_in_tonnes) is applied after BW2 computes impacts. If you scale the template to the user's requested quantity, the scale_factor will multiply a result that is already scaled, causing a systematic N-fold overcount (e.g., for "10 tonnes": template scaled to 10,000 kg × scale_factor 10 = 100-tonne result instead of 10 tonnes). Therefore:
      - The last mass-bearing activity MUST output exactly mass_out_kg = 1000.0 (1 tonne).
      - ALL upstream amounts must be derived backwards from this fixed 1000 kg endpoint.
      - NEVER use the user's requested amount as the template output mass.
      - NEVER set the final mass_out_kg to anything other than 1000.0.

    **Before filling in "amount", "mass_in_kg", "mass_out_kg", and "recovery_rate", follow this procedure in order:**
    Step 1 — Determine mass flow for each stage by working BACKWARDS from the final product:
    - The last material-producing stage must output exactly 1000 kg (mass_out_kg = 1000).
    - For each earlier stage, use recovery rates from the Supplementary Context if available; otherwise use typical LCA values for recovery rates provided above to estimate the recovery rate for that stage. Then,
    - Compute: mass_in_kg[i] = mass_out_kg[i] / recovery_rate[i]; set amount[i] = mass_out_kg[i]; set mass_out_kg[i] = mass_in_kg[i+1] to maintain chain continuity.
    - If the Supplementary Context provides specific material quantities or yields for a stage, use those exact values instead of guessing.
    Step 2 — Set "amount", "mass_in_kg", and "mass_out_kg":
    - In ecoinvent, every activity is defined per unit of its OUTPUT (reference product). Therefore **"amount" must equal "mass_out_kg"** for mass-bearing activities. This is the quantity BW2 uses to scale the environmental inventory.
    - "mass_in_kg" is the input material to this process (= mass_out_kg / recovery_rate). It is used only for the mass flow balance chain and does NOT go into the BW2 calculation directly.
    - For non-mass activities (transport, energy, services): set amount to the appropriate service quantity (e.g., tonne-kilometres for transport); set mass_in_kg = 0, mass_out_kg = 0, recovery_rate = 1.0.
    Step 3 — Set "recovery_rate" = mass_out_kg / mass_in_kg for mass-bearing activities.
    - "amount": For mass-bearing activities, this equals mass_out_kg (the reference product output of this ecoinvent process per 1 tonne final product). For non-mass activities, this is the service quantity (e.g., transport distance * cargo mass).
    - "mass_in_kg": Input mass (kg) of the primary material entering this process per 1 tonne of final product (= amount / recovery_rate). Used for mass flow chain validation only. Set to 0 for non-mass activities.
    - "mass_out_kg": Useful output mass (kg) of the primary material leaving this process. Equals amount for mass-bearing activities. The final material-producing stage must output 1000 kg. Set to 0 for non-mass activities.
    - "recovery_rate": mass_out_kg / mass_in_kg (0.0-1.0). Use 1.0 for non-mass activities.
    
    **Mass flow consistency rule (ISO 14040/44)**: mass_out_kg of stage i must equal mass_in_kg of stage i+1 (within ~10 %). Non-mass activities (transport, energy) are exempt.
    **Critical exclusion rule — capital infrastructure activities**: Do NOT include activities whose names contain words such as "construction", "building", "facility", "plant construction", or "infrastructure" (e.g., "mine construction, iron", "factory construction", "road construction") as standalone entries in the functional unit, unless the study goal is explicitly the construction of that specific facility. These are one-time capital goods activities whose unit is typically "unit" (one whole facility), and they carry enormous embedded impacts (tens to hundreds of millions of kg CO₂-eq per unit). Their impacts are already proportionally allocated in the corresponding per-mass or per-energy operation activities (e.g., per-kilogram mine operation, per-kWh power plant operation). Listing them separately as a top-level functional unit item constitutes severe double-counting. For a product value-chain LCA, only include the operational flow activities (e.g., "copper mine operation and beneficiation", "steel production, converter") and let the background system handle infrastructure allocation automatically.
    **Critical unit-check rule — "unit"-referenced activities in disposal/end-of-life stages**: In ecoinvent, activities with reference unit "unit" represent one complete physical facility or asset (a landfill cell, an incineration plant, a treatment plant). They are capital goods, NOT per-mass treatment services. NEVER use a "unit"-referenced activity for disposal, end-of-life, recycling, or waste treatment stages where you intend to model the treatment of N kg of waste. Doing so causes BW2 to scale the entire facility's impact by N, producing absurdly large results (e.g., 100,000 kg of waste * [1 landfill facility @ 1,400 tonne CO₂-eq/unit] = 1.4*10^(11) kg CO₂-eq). For disposal/end-of-life stages, ONLY select activities whose ecoinvent reference unit is "kilogram", "m³", or "tonne". Correct per-mass disposal activities include (but are not limited to): "market for inert waste, for final disposal" (kg), "treatment of inert waste, sanitary landfill" (kg), "market for waste concrete, not reinforced" (kg), "market for waste, for final disposal" (kg). If the available activity list only contains "unit"-referenced activities for a disposal stage, omit that stage entirely rather than including a facility-level activity.
    **Anti-double-counting rule — "market for" vs. production/operation activities**: In ecoinvent, every "market for [material X]" activity is a background supply-mix aggregator — it internally calls the corresponding production datasets to supply material X to a requesting foreground process. It is NOT an independent processing step. When your functional unit already contains a production or operation activity that PRODUCES or PROCESSES material X (e.g., "copper mine operation and beneficiation, sulfide ore" produces copper concentrate; "smelting of copper concentrate, sulfide ore" produces copper anode; "electrorefining of copper, anode" produces copper cathode), you MUST NOT also separately list the corresponding "market for [X]" activity (e.g., "market for copper concentrate, sulfide ore", "market for copper, anode") as a top-level entry. Including both constitutes SEVERE double-counting: the production impacts of material X are counted once via the explicit production activity entry AND again via the "market for" activity, which internally includes that same production dataset in its background supply mix. This error directly causes the mass flow sequence to appear physically impossible (the chain can appear to go "backwards" in mass — e.g., decreasing from 10,000 kg back to 10,248 kg) because the "market for" activity is assigned amounts corresponding to the input of the stage it duplicates.
    STRICT APPLICATION: For each "market for [X]" activity you are considering adding to the functional unit, apply this test:
      - If a production, operation, smelting, refining, beneficiation, or manufacturing activity that produces material X as its OUTPUT is ALREADY in your functional unit → OMIT "market for [X]" entirely.
      - If NO production activity for X exists in your functional unit AND you need to represent the average market supply of X (e.g., grid electricity, generic steel input) → the "market for [X]" activity may be retained as a supply placeholder.
    The ONLY legitimate top-level "market for" entry in a production-chain LCA is the market activity for the FINAL OUTPUT PRODUCT (e.g., "market for copper, cathode" as the last step representing market distribution to the buyer), and only when it is not duplicating a production step already listed. Even then, if the production step is explicitly modeled, prefer the production activity over the market activity.
    Worked example (copper cathode): If your functional unit lists "copper mine operation and beneficiation" + "smelting of copper concentrate" + "electrorefining of copper, anode", then "market for copper concentrate, sulfide ore" and "market for copper, anode" are FORBIDDEN duplicates and must be omitted. "Market for copper, cathode" may be kept only if the system boundary explicitly calls for a market distribution step and it is the sole representative of the final output.
    **Discrete component quantity rule — "unit"-referenced activities for countable product components**: Some legitimate ecoinvent activities carry a reference unit of "unit" because they represent a discrete, manufactured, countable item rather than a bulk material or a facility. Examples include: a lithium-ion battery pack, an electric motor, a heat exchanger, a photovoltaic module, a wind turbine nacelle, a fuel cell stack, a transformer, a pump, a compressor, or an airbag module. These are NOT infrastructure capital goods and must NOT be confused with facility-level activities. When such an activity appears in the functional unit, you MUST reason from first principles about how many of that component the studied product physically contains, then set "amount" to that integer or fractional count. Follow this decision procedure:
      1. Identify the component type and the studied product (e.g., "battery pack" for a 3-tonne electric vehicle).
      2. Apply engineering common sense to determine the typical count per unit of final product. Use the following as guidance (scale proportionally if the user specifies a non-standard product size):
         - Battery pack (EV passenger car, any size): 1 unit
         - Electric traction motor (single-motor EV): 1 unit; (dual-motor EV): 2 units
         - Photovoltaic module (rooftop residential 5 kWp system, ~400 W/module): ~12-13 units
         - Wind turbine nacelle / generator (single turbine LCA): 1 unit
         - Heat exchanger (industrial HVAC unit): 1-4 units depending on circuit count
         - Fuel cell stack (passenger car FCEV): 1 unit
         - Automotive airbag module (standard car): 6-10 units
         - Hydraulic pump (industrial machine): 1-3 units
         - Large power transformer (substation): 1 unit
         - Tire / wheel assembly (passenger car): 4 units; (truck): 6-18 units
      3. If the product size or configuration deviates from the typical case (e.g., an articulated bus vs. a passenger car), scale the count accordingly using your domain knowledge.
      3b. **CRITICAL — the 1-tonne reference basis does NOT apply to "unit" activities**: The post-processing engine applies a fixed scale factor of 1.0 to all "unit"-referenced activities, meaning the `amount` you write here is used exactly as-is in the final result — NO downstream scaling is ever applied to it. Therefore `amount` must be the true physical count of that component in ONE complete product unit, regardless of what quantity the user has requested (1 tonne, 2 tonnes, or 10 tonnes). Do NOT divide by the product mass. Do NOT multiply by any fraction derived from 1000 ÷ product_mass_kg. Examples for a 2-tonne (2000 kg) passenger car — correct amounts:
         - Electric motor (single-motor EV): 1 unit 
         - Battery pack: 1 unit
         - Powertrain: 1 unit
         - Tires: 4 units
      4. Set "amount" to the physical count from steps 2-3. Set "mass_in_kg" = 0, "mass_out_kg" = 0, "recovery_rate" = 1.0 (treat like a non-mass service activity for the mass flow chain, since the component's internal mass balance is already captured inside its own ecoinvent dataset).
      5. Never set "amount" = 1000 or any bulk-mass figure for a discrete-component "unit" activity. Never set "amount" = 0. If you genuinely cannot determine a reasonable count, default to 1.
    **Product mass descriptor vs. product quantity rule**: When a user describes the studied product with a mass qualifier (e.g., "a 2-tonne electric vehicle", "a 500 kg industrial pump", "a 12-tonne lorry", "a 250 g smartphone"), that number is the PHYSICAL MASS of ONE product unit — it is NOT the number of product units to study. The entire functional unit must represent exactly one (1) product of that described mass. Do NOT assign any activity amount equal to the product's stated mass value simply because that value appears in the product description. Component counts are governed exclusively by the Discrete component quantity rule above, and bulk material amounts are derived from the 1-tonne backward mass-flow calculation. Treating a mass descriptor as a quantity multiplier will systematically over-scale every impact in the inventory by that factor.
    **Activity retrieval coherence rule**: After an `activity_name` is resolved to an ecoinvent dataset by the retrieval system, verify that the resolved reference product is semantically coherent with the intended process before accepting it. Reject the match and omit or substitute if any of the following conditions are met:
      - A PRODUCTION or MANUFACTURING activity (e.g., "battery cell production", "motor winding") resolves to a FACILITY or FACTORY dataset (reference product name contains words such as "factory", "plant", "workshop", or an unrelated technology, e.g., "photovoltaic cell factory"). This is a capital-goods mismatch that inflates GWP by orders of magnitude. Substitute the closest per-kg or per-unit operational production dataset for the same material instead.
      - A PRODUCTION activity resolves to a MAINTENANCE, MARKET FOR, or DISPOSAL dataset. The resulting impact will be near-zero or physically meaningless. Substitute a production-stage dataset.
      - Two distinct `activity_name` entries in the functional unit resolve to the exact same ecoinvent reference product name. One is a duplicate. Keep only the more specific entry and remove the other.
      - When in doubt, prefer omitting an activity over retaining a semantically mismatched one that will produce an absurd result.
    **End-of-life recovery physical ceiling rule**: Activities representing material recovery, recycling credits, or avoided burdens — identifiable by negative GWP contributions or names containing phrases such as "treatment of used", "market for non-Fe", "hydrometallurgical processing", "pyrometallurgical processing of used", or "collection for recycling" — have amounts that are strictly bounded by the mass of the recoverable material actually present in ONE unit of the studied product. Before finalising any such amount:
      1. Estimate the mass of the relevant material sub-component within one product unit (e.g., Li-ion battery pack in a passenger car ≈ 300-600 kg; copper wiring in a household appliance ≈ 1-5 kg; aluminium body panels in a car ≈ 150-250 kg).
      2. Apply a realistic end-of-life collection and separation efficiency for that material stream (typically 0.70-0.95 for well-regulated industrial recycling).
      3. The final amount must satisfy: amount ≤ component_mass_in_product * collection_efficiency. If a mass-flow calculation yields an amount exceeding this ceiling, cap it at the ceiling value. An end-of-life recovery amount that exceeds the total mass of the studied product is always an error and must be corrected before output.
   
    After defining the functional unit, also include:
    - "project_name": Insert "{database}"  
    - "dataset": Insert "{database}", but in lowercase  

    Example of expected output JSON format:
    {{
    "goal": "lifecycle impact assessment",
    "Project": "Ecoinvent",
    "method": ["IPCC 2021", "climate change", "global warming potential (GWP100)"],
    "system_boundary": "cradle-to-grave",
    "functional_unit": [
        {{
        "activity_name": "hard coal production",
        "amount": 1000.0,
        "unit": "kilogram",
        "location": "GLO",
        "stage": "manufacturing",
        "mass_in_kg": 1050.0,
        "mass_out_kg": 1000.0,
        "recovery_rate": 0.952
        }},
        {{
        "activity_name": "transport, freight, lorry 16-32 metric ton, EURO6",
        "amount": 0.1,
        "unit": "ton kilometer",
        "location": "GLO",
        "stage": "transport",
        "mass_in_kg": 0,
        "mass_out_kg": 0,
        "recovery_rate": 1.0
        }}
    ],
    "project_name": "Ecoinvent",
    "dataset": "ecoinvent"
    }}

    **Pre-output self-verification — mandatory before returning your answer. Work through each check in order; correct any violation before finalising the JSON:**
    1. **1-tonne reference basis**: Is the final mass-bearing activity's `mass_out_kg` exactly 1000.0? If not, recompute the entire mass-flow chain backwards from 1000.0 and update all amounts.
    2. **amount = mass_out_kg parity**: For every mass-bearing activity, does `amount` equal `mass_out_kg`? If any value differs, set `amount = mass_out_kg`.
    3. **Mass-flow continuity (ISO 14040/44)**: For every consecutive pair of mass-bearing activities, does `mass_out_kg[i] ≈ mass_in_kg[i+1]` within 10 %? If not, adjust the chain to restore continuity.
    4. **No "market for" duplicates**: Does the list contain a "market for [X]" activity alongside a production or operation activity that already produces material X? If yes, remove the "market for [X]" entry.
    5. **No capital-infrastructure activities**: Does the list contain any activity named with words such as "construction", "building", or "facility" as a standalone top-level entry? If yes, remove it.
    6. **No "unit"-referenced disposal activities**: Does the list contain a "unit"-referenced activity in a disposal or end-of-life stage? If yes, remove it or substitute the nearest per-kilogram treatment alternative.
    7. **recovery_rate arithmetic**: For each mass-bearing activity, does `recovery_rate = mass_out_kg / mass_in_kg`? If not, recompute and correct the value.
    8. **unit-activity physical count**: For every "unit"-referenced activity, does `amount` equal the actual physical count of that component in ONE product unit (e.g., 1 motor per car, 1 battery per car, 4 tires per car)? The amount must be a reasonable integer.
    9. **Verbatim activity names**: For every entry in the functional_unit, is the `activity_name` copied exactly (character-for-character, lowercase) from the **List of available activities by stages**? If any name was paraphrased, reworded, capitalised differently, or invented, replace it with the exact string from the list — or remove the entry entirely if no suitable name exists in the list.
    10. **Disposal double-counting**: Does the disposal / end-of-life stage contain both a whole-product dismantling or collection activity AND separate per-material treatment activities for sub-components? If yes, remove the whole-product dismantling activity — the sub-component treatments represent the same mass stream with greater specificity, and retaining both counts the product mass twice through end-of-life. You MUST ensure the sum of mass in disposal or recycling activities is equal or acceptably close to the total product mass (1000 kg) to avoid an impossible mass flow. If the sum of sub-component treatments is significantly less than the product mass, you may retain a single whole-product disposal activity to balance the mass flow, but do NOT include both.

    Only after all ten checks pass, return the final JSON. Do not include explanations, comments, or any text outside the JSON object. Answer in English only.
    """

    final_prompt_1st = ChatPromptTemplate.from_template(final_template_1st)

    activities_by_stage_str = join_activities_by_stage(retrieved_acitivity_list)

    final_rag_chain_1st = (
        {"context": query_decomposition_retrival | RunnableLambda(join_docs),
        "methods": RunnableLambda(lambda x: LCIA_method_list) | RunnableLambda(join_methods),
        "retrieved_acitivity_list": RunnableLambda(lambda x: activities_by_stage_str),
        "product": itemgetter("product"),
        "database": itemgetter("database"),
        "user_input_params": itemgetter("user_input_params")
        }
        | final_prompt_1st
        | gpt5_client
        | StrOutputParser()
    )

    try:
        final_results_1st = final_rag_chain_1st.invoke({
            "question": json.dumps(params, ensure_ascii=False),
            "product": product,
            "database": LCI_DATABASE,
            "user_input_params": json.dumps(params, ensure_ascii=False, indent=2),
        })
    except:
        # rerun-it to prevent occasional LLM generation failure
        final_results_1st = final_rag_chain_1st.invoke({
            "question": json.dumps(params, ensure_ascii=False),
            "product": product,
            "database": LCI_DATABASE,
            "user_input_params": json.dumps(params, ensure_ascii=False, indent=2),
        })
    
    os.makedirs("bw2_input_templates", exist_ok=True)
    results_dict = json.loads(final_results_1st)

    if "functional_unit" in results_dict:
        for activity in results_dict["functional_unit"]:
            original_name = activity.get("activity_name", "")
            if original_name:
                # Use retriever_eco_activity to match and replace
                matched = None
                try:
                    # Use the retriever to get the best match (returns a list, take first if available)
                    matches = retriever_eco_activity.invoke(original_name)
                    if isinstance(matches, list) and matches:
                        matched = matches[0].page_content
                    else:
                        matched = matches.page_content  
                except Exception as e:
                    logging.warning(f"Error retrieving LCIA activity for '{original_name}': {e}")
                    logging.warning(f"Error retrieving LCIA activity for '{original_name}': {matches}")

                if matched:
                    activity["activity_name"] = str(matched)

    # ── Reference-product critique pass (second-pass quality gate) ────────────
    # Query BW2 for the actual reference_product of each activity, then ask the
    # LLM to validate semantic coherence with the assigned stage.  Falls back
    # silently to the original list if BW2 or the LLM are unavailable.
    draft_names = [
        a.get("activity_name", "")
        for a in results_dict.get("functional_unit", [])
        if a.get("activity_name")
    ]
    ref_product_lookup = {}
    try:
        ref_product_lookup = lookup_activity_ref_products_batch(draft_names)
    except Exception as exc:
        logging.warning("Reference product lookup skipped: %s", exc)

    results_dict["functional_unit"] = _refine_activities_with_ref_product_critique(
        gpt5_client,
        results_dict.get("functional_unit", []),
        product,
        retrieved_acitivity_list,
        ref_product_lookup,
        retriever_eco_activity,
    )

    # ── Remove 'market for X' duplicates to prevent double-counting ───────────
    # Must run AFTER retriever name resolution so activity names are final.
    results_dict["functional_unit"] = _remove_market_activity_duplicates(
        results_dict.get("functional_unit", [])
    )

    # # ── Disposal mass balance: remove whole-product dismantling when ──────────
    # # sub-component treatments already cover the same mass stream.
    # results_dict["functional_unit"] = _balance_disposal_mass(
    #     results_dict.get("functional_unit", [])
    # )

    # Inject n_simulations so the BW2 calculator can read it when supported
    results_dict['n_simulations'] = resolved_n_simulations

    # ── Pre-flight mass flow balance validation ────────────────────────────────
    mass_flow_audit = _validate_mass_flow_chain(results_dict.get("functional_unit", []))
    results_dict["mass_flow_audit"] = mass_flow_audit
    if mass_flow_audit["status"] == "pass":
        logging.info("Mass flow audit passed. Overall material yield: %.4f",
                     mass_flow_audit.get("overall_yield") or 0.0)
    elif mass_flow_audit["status"] in ("warning", "gap"):
        logging.warning("Mass flow audit - %s. Transitions: %s",
                        mass_flow_audit["status"],
                        mass_flow_audit.get("transitions", []))

    current_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d%H%M%S")
    output_filepath = f"bw2_input_templates/{product}_{LCI_DATABASE}_{current_time}.json"
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4, ensure_ascii=False)

    # Return filepath and activities_by_stage as dict {stage_name: [Document, ...]}
    return output_filepath, retrieved_acitivity_list



def _build_workbench_stage_activity_list(strict_nodes: list) -> dict:
    """Create deterministic stage->activity candidates from user-selected nodes."""
    activities_by_stage = {}
    for i, node in enumerate(strict_nodes):
        process_name = (node.get("processName") or node.get("processId") or "").strip()
        if not process_name:
            continue
        stage_key = f"step_{i + 1}"
        activities_by_stage[stage_key] = [Document(page_content=process_name)]
    return activities_by_stage


def _resolve_workbench_activity_name(retriever_eco_activity, activity_name: str) -> str:
    """Resolve a selected activity name to the closest ecoinvent entry, preserving fallback."""
    name = (activity_name or "").strip()
    if not name:
        return name
    try:
        matches = retriever_eco_activity.invoke(name)
        if isinstance(matches, list) and matches:
            return str(matches[0].page_content or name)
        if matches and getattr(matches, "page_content", None):
            return str(matches.page_content)
    except Exception as exc:
        logging.warning("Workbench strict retrieval failed for '%s': %s", name, exc)
    return name


def generate_response_bw2_input_workbench_strict(
    gpt5_client,
    query_decomposition_retrival,
    LCIA_method_list,
    strict_nodes,
    product,
    LCI_DATABASE,
    retriever_eco_activity,
    user_input_params=None,
):
    """
    Strict Workbench BW2 input generation:
    - Uses exactly the user-selected node order as the primary functional chain.
    - Allows fallback retrieval only to map selected names to closest ecoinvent labels.
    """
    strict_nodes = [n for n in (strict_nodes or []) if (n.get("processName") or n.get("processId"))]
    if not strict_nodes:
        raise ValueError("Workbench strict mode requires at least one selected node")

    retrieved_activity_list = _build_workbench_stage_activity_list(strict_nodes)

    output_filepath, activities_by_stage_dict = generate_response_bw2_input(
        gpt5_client,
        query_decomposition_retrival,
        LCIA_method_list,
        retrieved_activity_list,
        product,
        LCI_DATABASE,
        retriever_eco_activity,
        user_input_params=user_input_params,
    )

    with open(output_filepath, "r", encoding="utf-8") as f:
        results_dict = json.load(f)

    existing_fu = results_dict.get("functional_unit", []) or []
    strict_fu = []

    for idx, node in enumerate(strict_nodes):
        selected_name = (node.get("processName") or node.get("processId") or "").strip()
        resolved_name = _resolve_workbench_activity_name(retriever_eco_activity, selected_name)
        base = existing_fu[idx] if idx < len(existing_fu) else {}

        strict_fu.append({
            "activity_name": resolved_name,
            "amount": base.get("amount", 1000.0 if idx == len(strict_nodes) - 1 else 0.0),
            "unit": base.get("unit", "kilogram"),
            "location": base.get("location") or node.get("region") or "GLO",
            "stage": base.get("stage", f"step {idx + 1}"),
            "mass_in_kg": base.get("mass_in_kg", 0),
            "mass_out_kg": base.get("mass_out_kg", 1000.0 if idx == len(strict_nodes) - 1 else 0),
            "recovery_rate": base.get("recovery_rate", 1.0),
        })

    results_dict["functional_unit"] = strict_fu

    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4, ensure_ascii=False)

    return output_filepath, activities_by_stage_dict


'''

    You are an assistant that converts lifecycle assessment (LCA) queries into structured JSON instructions for the Brightway2 LCA engine.  
    Extract key activities along the value chain and sort by stage based on collected information of product {product} given by the user query: {question} and the database: {context}, and output a JSON with:

    - "goal": Always "lifecycle impact assessment"  
    - "Project": Insert the project name as "{database}"  
    - "method": Select the LCIA method most relevant to the user's request from the provided list: {methods}. If no specific impact category is requested, default to a climate change / GWP100 method (e.g., IPCC 2021, EF v3.1).
    - "system_boundary": Use "cradle-to-gate" or "cradle-to-grave"  
    - "functional_unit": A list of multiple activity objects, each describing one process involved in the product system:  
    Each object includes:  
    - "activity_name": The exact process name as used in Brightway2-compatible databases  
    - "amount": A float  
    - "unit": e.g., "kilogram"  
    - "location": ISO country code  
    - "stage": The value chain stage (e.g., "raw material extraction", "manufacturing", "transport", "use", "disposal")  
    - "project_name": Insert "{database}"  
    - "dataset": Insert "{database}", but in lowercase  

    Example of expected output JSON format:
    {{
    "goal": "lifecycle impact assessment",
    "Project": "Ecoinvent",
    "method": ["IPCC 2013", "climate change", "GWP 100a"],
    "system_boundary": "cradle-to-grave",
    "functional_unit": [
        {{
        "activity_name": "hard coal coke production",
        "amount": 1.0,
        "unit": "kilogram",
        "location": "CN",
        "stage": "manufacturing"
        }},
        {{
        "activity_name": "transport, freight, lorry 16-32 metric ton, EURO6",
        "amount": 0.1,
        "unit": "ton kilometer",
        "location": "CN",
        "stage": "transport"
        }}
    ],
    "project_name": "Ecoinvent",
    "dataset": "ecoinvent"
    }}
    '''
    
    
'''
You are an assistant that converts lifecycle assessment (LCA) queries into structured JSON instructions for the Brightway2 LCA engine.  
Extract key activities along the value chain and sort by stage based on collected information of product {product} given by the user query: {question} and the database: {context}, and output a JSON with:

- "goal": Always "lifecycle impact assessment"  
- "method": Select the most relevant LCIA methods from provided list: {methods}. Insert the first element of your chosen LCIA method, and insert the second and third element as "climate change" and "GWP 100a" respectively, as the "method" value in the template. Also, Please note that if "{database}" is "Tiangong", you must replace the first element in the method list with "Tiangong". 
- "system_boundary": Use "cradle-to-gate" or "cradle-to-grave"  
- "functional_unit": A list of multiple activity objects, each describing one process involved in the product system:  
Each object includes:  
- "activity_name": The exact process name as used in Brightway2-compatible databases  
- "amount": A float  
- "unit": e.g., "kilogram"  
- "location": ISO country code  
- "stage": The value chain stage (e.g., "raw material extraction", "manufacturing", "transport", "use", "disposal")

Example of expected output JSON format:
{{
"goal": "lifecycle impact assessment",
"method": ["IPCC 2013", "climate change", "GWP 100a"],
"system_boundary": "cradle-to-gate",
"functional_unit": [
    {{
    "activity_name": "limestone quarry operation",
    "amount": 1.0,
    "unit": "kilogram",
    "location": "CN",
    "stage": "raw material extraction"
    }},
    {{
    "activity_name": "brick production, at plant",
    "amount": 1.0,
    "unit": "kilogram",
    "location": "CN",
    "stage": "manufacturing"
    }}
]
}}

Only return the JSON object in text format. Do not explain or comment. Answer in English and do not use <think> mode if you have.
'''