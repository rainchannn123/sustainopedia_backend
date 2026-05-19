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
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import sqlite3
from .functions.non_RAG_methods import conduct_lca_anaylsis, Get_LCIA_methods, generate_response_bw2_input, extract_functional_unit, infer_product_from_query, input_q_to_json, generate_list_of_activities
from .functions.vectorstore import build_and_save_vectorstores, load_vectorstores, LcaKnowledgeRetriever, load_vectorstores_fast
from .functions.after_bw2 import final_mode_output_bw2_included
from .functions.bw2_calculation import start_lca_calculation
from.functions.RAG_methods import query_decomposition_retrieval
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


def generate_response_fast(question: str, product: str) -> dict:
    """Fast-path: RAG retrieval from chroma_db_lca_knowledge + single LLM call, no BW2 pipeline."""
    context = ""
    # Retrieve relevant LCA knowledge if vectorstores are available
    if all(os.path.isdir(d) for d in ['./chroma_db_lca_knowledge', './chroma_db_eco_methods',
                                       './chroma_db_tg_methods', './chroma_db_eco_activity']):
        vs_knowledge, parent_lookup = load_vectorstores_fast(
            embeddings,
            persist_dir_lca_knowledge='./chroma_db_lca_knowledge',
        )
        retriever = LcaKnowledgeRetriever(vs_knowledge, parent_lookup, k_children=4).as_langchain_retriever()
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs) if docs else ""

    # Simple intent classification (no LLM call)
    q_lower = (question or "").lower()
    if any(word in q_lower for word in ["current lca result", "current lca context", "current lca insights", "current lca findings"]):
        intent = "Specific_LCA_result_inquiry"
    if any(word in q_lower for word in ["literature", "study", "studies", "review", "paper", "papers", "summarize"]):
        intent = "literature"
    elif any(word in q_lower for word in ["calculate", "estimate", "compute", "carbon footprint", "climate impact", "lca result", "lca analysis", "carbon emission", "co2 emission", "how many", "how much"]):
        intent = "computation"
    else:
        intent = "general"

    # Only include necessary fields for intent_params (frontend only expects intent and product for fast mode)
    intent_params = {
        "intent": intent,
        "product": product or "Not Specified"
    }

    if intent == 'Specific_LCA_result_inquiry':
        role_instruction = (
            "You are a Senior LCA Consultant. The user is asking about the current LCA results, which is shown in 'Current LCA Results Context for reference'. "
            "Answer the question based on the current LCA results, and based on the external LCA knowledge provided if the information in the current LCA results is insufficient for answering user's question."
        )
    elif intent == 'literature':
        role_instruction = (
            "You are an expert LCA researcher. Summarize relevant peer-reviewed studies, "
            "methodologies, and findings from the knowledge provided. "
            "Highlight key data points and cite specific values where available. "
            "Structure your response with ## headings for major themes. "
            "You must not make up studies or data not present in the context. Focus on summarizing the existing literature."
        )
    elif intent == 'general':
        role_instruction = (
            "You are a Senior LCA Consultant. Explain the concept clearly and concisely "
            "using the knowledge provided. Use practical examples where helpful."
        )
    else:  # computation or fast mode
        role_instruction = (
            "You are a Senior LCA Consultant. Provide a concise, evidence-based answer "
            "to this LCA query using the retrieved context. "
            "Include key numeric estimates where available."
        )

    # Build template with role_instruction embedded directly (not as a LangChain template variable)
    template = (
        role_instruction + "\n\n"
        "Product: {product_hint}\n"
        "User Query: {question}\n\n"
        "External LCA Knowledge:\n{context}\n\n"
        "Output valid Markdown. Use ## for section headings and - for bullet points. "
        "Use inline code for numeric values (e.g., `12.5 kg CO\u2082-eq`). "
        "Do not fabricate data not present in the context. "
        "If Product is not specified, try to guess it if you are confident, otherwise do not make assumptions and focus on general insights from the context."
        "If User's query is completely unrelated to LCA, energy and environmental impact, or sustainability fields, politely decline to answer and explain in one sentence that you can only assist with LCA-related queries."
    )

    chain = ChatPromptTemplate.from_template(template) | gpt5_client | StrOutputParser()
    try:
        answer = chain.invoke({"question": question, "product_hint": product, "context": context})
    except:
        # rerun-it to prevent occasional LLM generation failure
        answer = chain.invoke({"question": question, "product_hint": product, "context": context})

    return {"answer": answer, "markdown": None, "processed_json": None, "lcia_table": None, "intent_params": intent_params}


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
    retriever_eco_activity = vectorstore_eco_activity.as_retriever(search_kwargs={"k": 8})

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
    query_decomposition_retrival = query_decomposition_retrieval(user_input_params, gpt5_client, retriever_lca_knowledge, vectorstore_lca_knowledge, k_per_sub=3)
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
