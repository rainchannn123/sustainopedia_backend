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
from datetime import datetime, timezone, timedelta


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
    ['acidification no LT', 'acidification (incl. fat', 'climate change no LT', 'global warming potential (GWP100) no LT', 'ecotoxicity: freshwater no LT', 'freshwater aquatic ecotoxicity (FAETP inf) no LT', 'ecotoxicity: marine no LT', 'marine aquatic ecotoxicity (MAETP inf) no LT', 'ecotoxicity: terrestrial no LT', 'terrestrial ecotoxicity (TETP inf) no LT', 'energy resources: non-renewable no LT', 'abiotic depletion potential (ADP): fossil fuels no LT', 'eutrophication no LT', 'eutrophication (fate not incl.) no LT', 'human toxicity no LT', 'human toxicity (HTP inf) no LT', 'material resources: metals/minerals no LT', 'abiotic depletion potential (ADP): elements (ultimate reserves) no LT', 'ozone depletion no LT', 'ozone layer depletion (ODP steady state) no LT', 'photochemical oxidant formation no LT', 'photochemical oxidation (high NOx) no LT', 'acidification', 'climate change', 'global warming potential (GWP100)', 'ecotoxicity: freshwater', 'freshwater aquatic ecotoxicity (FAETP inf)', 'ecotoxicity: marine', 'marine aquatic ecotoxicity (MAETP inf)', 'ecotoxicity: terrestrial', 'terrestrial ecotoxicity (TETP inf)', 'energy resources: non-renewable', 'abiotic depletion potential (ADP): fossil fuels', 'eutrophication', 'eutrophication (fate not incl.)', 'human toxicity', 'human toxicity (HTP inf)', 'material resources: metals/minerals', 'abiotic depletion potential (ADP): elements (ultimate reserves)', 'ozone depletion', 'ozone layer depletion (ODP steady state)', 'photochemical oxidant formation', 'photochemical oxidation (high NOx)', 'accumulated exceedance (AE)', 'climate change: biogenic', 'climate change: fossil', 'climate change: land use and land use change', 'comparative toxic unit for ecosystems (CTUe)', 'eutrophication: freshwater', 'fraction of nutrients reaching freshwater end compartment (P)', 'eutrophication: marine', 'fraction of nutrients reaching marine end compartment (N)', 'eutrophication: terrestrial', 'human toxicity: carcinogenic', 'comparative toxic unit for human (CTUh)', 'human toxicity: non-carcinogenic', 'ionising radiation: human health', 'human exposure efficiency relative to u235', 'land use', 'soil quality index', 'ozone depletion potential (ODP)', 'particulate matter formation', 'impact on human health', 'photochemical oxidant formation: human health', 'tropospheric ozone concentration increase', 'water use', 'user deprivation potential (deprivation-weighted water consumption)', 'accumulated exceedance (AE) no LT', 'climate change: biogenic no LT', 'climate change: fossil no LT', 'climate change: land use and land use change no LT', 'comparative toxic unit for ecosystems (CTUe) no LT', 'ecotoxicity: freshwate', 'norganics no LT', 'etals no LT', 'rganics no LT', 'eutrophication: freshwater no LT', 'fraction of nutrients reaching freshwater end compartment (P) no LT', 'eutrophication: marine no LT', 'fraction of nutrients reaching marine end compartment (N) no LT', 'eutrophication: terrestrial no LT', 'human toxicity: carcinogenic no LT', 'comparative toxic unit for human (CTUh) no LT', 'human toxicity: carcinogeni', 'human toxicity: non-carcinogenic no LT', 'human toxicity: non-carcinogeni', 'ionising radiation: human health no LT', 'human exposure efficiency relative to u235 no LT', 'land use no LT', 'soil quality index no LT', 'ozone depletion potential (ODP) no LT', 'particulate matter formation no LT', 'impact on human health no LT', 'photochemical oxidant formation: human health no LT', 'tropospheric ozone concentration increase no LT', 'water use no LT', 'user deprivation potential (deprivation-weighted water consumption) no LT', 'norganics', 'etals', 'rganics', 'global temperature change potential (GTP100) no LT', 'global temperature change potential (GTP20) no LT', 'global warming potential (GWP20) no LT', 'global temperature change potential (GTP100)', 'global temperature change potential (GTP20)', 'global warming potential (GWP20)', 'global temperature change potential (GTP50) no LT', 'global warming potential (GWP500) no LT', 'climate change: biogeni', 'ncluding SLCFs no LT', 'climate change: fossi', 'climate change: including SLCFs no LT', 'climate change: land use no LT', 'climate change: land us', 'global temperature change potential (GTP50)', 'global warming potential (GWP500)', 'ncluding SLCFs', 'climate change: including SLCFs', 'climate change: land use', 'ndpoint (E) no LT', 'ecosystem quality no LT', 'human health no LT', 'natural resources no LT', 'total: ecosystem quality no LT', 'total: human health no LT', 'total: natural resources no LT', 'ndpoint (E)', 'ecosystem quality', 'human health', 'natural resources', 'total: ecosystem quality', 'total: human health', 'total: natural resources', 'ndpoint (H) no LT', 'ndpoint (H)', 'ndpoint (I) no LT', 'ndpoint (I)', 'idpoint (E) no LT', 'acidification: terrestrial no LT', 'energy resources: non-renewabl', 'ionising radiation no LT', 'photochemical oxidant formation: terrestrial ecosystems no LT', 'idpoint (E)', 'acidification: terrestrial', 'ionising radiation', 'photochemical oxidant formation: terrestrial ecosystems', 'idpoint (H) no LT', 'idpoint (H)', 'idpoint (I) no LT', 'idpoint (I)', 'acidification potential (AP) no LT', 'eutrophication potential no LT', 'particulate matter formation potential (PMFP) no LT', 'maximum incremental reactivity (MIR) no LT', 'acidification potential (AP)', 'eutrophication potential', 'particulate matter formation potential (PMFP)', 'maximum incremental reactivity (MIR)', 'ecotoxicity no LT', 'total no LT', 'carcinogenic no LT', 'non-carcinogenic no LT', 'ecotoxicity', 'total', 'carcinogenic', 'non-carcinogenic']

    Output the methods and keywords in one line separated by commas. Do not use any other sentences or explanations.
    """
    prompt_methods = ChatPromptTemplate.from_template(method_search_template)

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

# def Get_LCIA_recommended_activity(question, gpt5_client, retriever_eco_activity, product):
#     print("Recommending LCIA activities...")
    
#     activity_search_template = """You are a life-cycle assessment (LCA) expert. First, You must choose only one to three the most relevant LCIA methods from the following list based on the user question: {question}.
#     LCIA method list: ['CML v4.8 2016 no LT', 'CML v4.8 2016', 'EF v3.0 EN15804', 'EF v3.0 no LT', 'EF v3.0', 'EF v3.1 EN15804', 'EF v3.1 no LT', 'EF v3.1', 'IPCC 2013 no LT', 'IPCC 2013', 'IPCC 2021 no LT', 'IPCC 2021', 'ReCiPe 2016 v1.0', 'TRACI v2.1 no LT', 'TRACI v2.1', 'USEtox no LT', 'USEtox']

#     Then, your task is to choose top five relevent keywords in the below list of keywords that help to retrieve relevant LCIA methods of user question: {question}
#     ['acidification no LT', 'acidification (incl. fat', 'climate change no LT', 'global warming potential (GWP100) no LT', 'ecotoxicity: freshwater no LT', 'freshwater aquatic ecotoxicity (FAETP inf) no LT', 'ecotoxicity: marine no LT', 'marine aquatic ecotoxicity (MAETP inf) no LT', 'ecotoxicity: terrestrial no LT', 'terrestrial ecotoxicity (TETP inf) no LT', 'energy resources: non-renewable no LT', 'abiotic depletion potential (ADP): fossil fuels no LT', 'eutrophication no LT', 'eutrophication (fate not incl.) no LT', 'human toxicity no LT', 'human toxicity (HTP inf) no LT', 'material resources: metals/minerals no LT', 'abiotic depletion potential (ADP): elements (ultimate reserves) no LT', 'ozone depletion no LT', 'ozone layer depletion (ODP steady state) no LT', 'photochemical oxidant formation no LT', 'photochemical oxidation (high NOx) no LT', 'acidification', 'climate change', 'global warming potential (GWP100)', 'ecotoxicity: freshwater', 'freshwater aquatic ecotoxicity (FAETP inf)', 'ecotoxicity: marine', 'marine aquatic ecotoxicity (MAETP inf)', 'ecotoxicity: terrestrial', 'terrestrial ecotoxicity (TETP inf)', 'energy resources: non-renewable', 'abiotic depletion potential (ADP): fossil fuels', 'eutrophication', 'eutrophication (fate not incl.)', 'human toxicity', 'human toxicity (HTP inf)', 'material resources: metals/minerals', 'abiotic depletion potential (ADP): elements (ultimate reserves)', 'ozone depletion', 'ozone layer depletion (ODP steady state)', 'photochemical oxidant formation', 'photochemical oxidation (high NOx)', 'accumulated exceedance (AE)', 'climate change: biogenic', 'climate change: fossil', 'climate change: land use and land use change', 'comparative toxic unit for ecosystems (CTUe)', 'eutrophication: freshwater', 'fraction of nutrients reaching freshwater end compartment (P)', 'eutrophication: marine', 'fraction of nutrients reaching marine end compartment (N)', 'eutrophication: terrestrial', 'human toxicity: carcinogenic', 'comparative toxic unit for human (CTUh)', 'human toxicity: non-carcinogenic', 'ionising radiation: human health', 'human exposure efficiency relative to u235', 'land use', 'soil quality index', 'ozone depletion potential (ODP)', 'particulate matter formation', 'impact on human health', 'photochemical oxidant formation: human health', 'tropospheric ozone concentration increase', 'water use', 'user deprivation potential (deprivation-weighted water consumption)', 'accumulated exceedance (AE) no LT', 'climate change: biogenic no LT', 'climate change: fossil no LT', 'climate change: land use and land use change no LT', 'comparative toxic unit for ecosystems (CTUe) no LT', 'ecotoxicity: freshwate', 'norganics no LT', 'etals no LT', 'rganics no LT', 'eutrophication: freshwater no LT', 'fraction of nutrients reaching freshwater end compartment (P) no LT', 'eutrophication: marine no LT', 'fraction of nutrients reaching marine end compartment (N) no LT', 'eutrophication: terrestrial no LT', 'human toxicity: carcinogenic no LT', 'comparative toxic unit for human (CTUh) no LT', 'human toxicity: carcinogeni', 'human toxicity: non-carcinogenic no LT', 'human toxicity: non-carcinogeni', 'ionising radiation: human health no LT', 'human exposure efficiency relative to u235 no LT', 'land use no LT', 'soil quality index no LT', 'ozone depletion potential (ODP) no LT', 'particulate matter formation no LT', 'impact on human health no LT', 'photochemical oxidant formation: human health no LT', 'tropospheric ozone concentration increase no LT', 'water use no LT', 'user deprivation potential (deprivation-weighted water consumption) no LT', 'norganics', 'etals', 'rganics', 'global temperature change potential (GTP100) no LT', 'global temperature change potential (GTP20) no LT', 'global warming potential (GWP20) no LT', 'global temperature change potential (GTP100)', 'global temperature change potential (GTP20)', 'global warming potential (GWP20)', 'global temperature change potential (GTP50) no LT', 'global warming potential (GWP500) no LT', 'climate change: biogeni', 'ncluding SLCFs no LT', 'climate change: fossi', 'climate change: including SLCFs no LT', 'climate change: land use no LT', 'climate change: land us', 'global temperature change potential (GTP50)', 'global warming potential (GWP500)', 'ncluding SLCFs', 'climate change: including SLCFs', 'climate change: land use', 'ndpoint (E) no LT', 'ecosystem quality no LT', 'human health no LT', 'natural resources no LT', 'total: ecosystem quality no LT', 'total: human health no LT', 'total: natural resources no LT', 'ndpoint (E)', 'ecosystem quality', 'human health', 'natural resources', 'total: ecosystem quality', 'total: human health', 'total: natural resources', 'ndpoint (H) no LT', 'ndpoint (H)', 'ndpoint (I) no LT', 'ndpoint (I)', 'idpoint (E) no LT', 'acidification: terrestrial no LT', 'energy resources: non-renewabl', 'ionising radiation no LT', 'photochemical oxidant formation: terrestrial ecosystems no LT', 'idpoint (E)', 'acidification: terrestrial', 'ionising radiation', 'photochemical oxidant formation: terrestrial ecosystems', 'idpoint (H) no LT', 'idpoint (H)', 'idpoint (I) no LT', 'idpoint (I)', 'acidification potential (AP) no LT', 'eutrophication potential no LT', 'particulate matter formation potential (PMFP) no LT', 'maximum incremental reactivity (MIR) no LT', 'acidification potential (AP)', 'eutrophication potential', 'particulate matter formation potential (PMFP)', 'maximum incremental reactivity (MIR)', 'ecotoxicity no LT', 'total no LT', 'carcinogenic no LT', 'non-carcinogenic no LT', 'ecotoxicity', 'total', 'carcinogenic', 'non-carcinogenic']

#     Output the methods and keywords in one line separated by commas. Do not use any other sentences or explanations.
#     """
#     prompt_activity = ChatPromptTemplate.from_template(activity_search_template)

#     generate_activity = (
#         prompt_activity
#         | gpt5_client
#         | StrOutputParser()
#     )
#     activity_output = generate_activity.invoke({"question": question})
#     activity_list = retriever_eco_activity.invoke(activity_output)
    
#     return activity_list
    
def generate_response_bw2_input(question, gpt5_client, query_decomposition_retrival, LCIA_method_list, product, LCI_DATABASE, retriever_eco_activity, n_simulations=25):
    print("Generating Brightway2 input JSON...")
    
    final_template_1st = """
    You are an assistant that converts lifecycle assessment (LCA) queries into structured JSON instructions for the Brightway2 LCA engine.  
    Extract key activities along the value chain and sort by stage based on collected information of product {product} given by the user query: {question} and some useful related reliable paragraphs in the database: {context}, and output a JSON with:
    #Please strictly note that the maximum number of activities in the functional unit should not exceed 10, and the amount of each activity should be normalized to the functional unit of 1 ton of product production. If there are more than 10 activities, please select the most relevant ones based on the user query and the database information.

    - "goal": Always "lifecycle impact assessment"  
    - "Project": Insert the project name as "{database}"  
    - "method": Select the top 1 most relevant LCIA methods from provided list: {methods}.
    - "system_boundary": Use "cradle-to-gate" or "cradle-to-grave"  
    - "functional_unit": A list of multiple activity objects, each describing one process involved in the product system:  
    Each object includes:  
    - "activity_name": The name of a process of the given product that usually involves in the product value chain and its lifecycle.
    - "amount": A float value representing the quantity of the given product involved in each subprocess, normalized to the functional unit of 1 ton of product production. This ensures consistency across all stages of the product's lifecycle value chain.
    - "unit": e.g., "kilogram"  
    - "location": ISO country code. If "{database}" == "Ecoinvent" or "ecoinvent", use "GLO", otherwise "CN"
    - "stage": The value chain stage (e.g., "raw material extraction", "manufacturing", "transport", "use", "disposal")
    After defining the functional unit, also include:
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
        "activity_name": "hard coal production",
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

    Only return the JSON object in text format. Do not explain or comment. Answer in English and do not use <think> mode if you have.
    """

    final_prompt_1st = ChatPromptTemplate.from_template(final_template_1st)

    final_rag_chain_1st = (
        {"context": query_decomposition_retrival | RunnableLambda(join_docs),
        "methods": RunnableLambda(lambda x: LCIA_method_list) | RunnableLambda(join_methods),  # Adjusted to use methods_list directly
        "question": itemgetter("question"),
        "product": itemgetter("product"),
        "database": itemgetter("database")}
        | final_prompt_1st
        | gpt5_client
        | StrOutputParser()
    )

    try:
        final_results_1st = final_rag_chain_1st.invoke({"question": question, "product":product, "database":LCI_DATABASE})
    except:
        # rerun-it to prevent occasional LLM generation failure
        final_results_1st = final_rag_chain_1st.invoke({"question": question, "product":product, "database":LCI_DATABASE})
    
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
    
    # Inject n_simulations so the BW2 calculator can read it when supported
    results_dict['n_simulations'] = n_simulations
    current_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d%H%M%S")
    output_filepath = f"bw2_input_templates/{product}_{LCI_DATABASE}_{current_time}.json"
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4, ensure_ascii=False)

    return output_filepath





'''
    You are an assistant that converts lifecycle assessment (LCA) queries into structured JSON instructions for the Brightway2 LCA engine.  
    Extract key activities along the value chain and sort by stage based on collected information of product {product} given by the user query: {question} and the database: {context}, and output a JSON with:

    - "goal": Always "lifecycle impact assessment"  
    - "Project": Insert the project name as "{database}"  
    - "method": Select the most relevant LCIA methods from provided list: {methods}. Insert the first element of your chosen LCIA method, and insert the second and third element as "climate change" and "GWP 100a" respectively, as the "method" value in the template. Also, Please note that if "{database}" is "Tiangong", you must replace the first element in the method list with "Tiangong". 
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