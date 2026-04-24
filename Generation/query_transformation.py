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
from .functions.non_RAG_methods import conduct_lca_anaylsis, Get_LCIA_methods, generate_response_bw2_input, extract_functional_unit, infer_product_from_query
from .functions.vectorstore import build_and_save_vectorstores, load_vectorstores, LcaKnowledgeRetriever
from .functions.after_bw2 import final_mode_output_bw2_included
from .functions.bw2_calculation import start_lca_calculation
from.functions.RAG_methods import query_decomposition_retrieval
from .functions.intent_classifier import classify_intent  # query intent classification and parameter extraction
from .functions.prebuild_bw2db import preinstall_ecoinvent_database  # pre-build BW2 databases to speed up first run
# from dotenv import load_dotenv
# load_dotenv('./.env')  # loads from .env file

# import psutil, threading

# # Configure logging to stdout or a file
# logging.basicConfig(level=logging.INFO)

# def log_memory_usage(interval=60):
#     process = psutil.Process(os.getpid())
#     while True:
#         mem_mb = process.memory_info().rss / (1024 * 1024)
#         logging.info(f"Memory usage: {mem_mb:.2f} MB")
#         time.sleep(interval)  # wait before logging again

# # Start background thread
# thread = threading.Thread(target=log_memory_usage, args=(60,), daemon=True)
# thread.start()

LCI_DATABASE = "Ecoinvent"
# LCI_DATABASE = "Tiangong"

def _fast_response(question: str, product: str, gpt5_client, embeddings, intent_params=None) -> dict:
    """Fast-path: RAG retrieval from chroma_db_lca_knowledge + single LLM call, no BW2 pipeline."""
    context = ""
    # Retrieve relevant LCA knowledge if vectorstores are available
    if all(os.path.isdir(d) for d in ['./chroma_db_lca_knowledge', './chroma_db_eco_methods',
                                       './chroma_db_tg_methods', './chroma_db_eco_activity']):
        vs_knowledge, parent_lookup, _, _, _ = load_vectorstores(
            embeddings,
            persist_dir_lca_knowledge='./chroma_db_lca_knowledge',
            persist_dir_eco_methods='./chroma_db_eco_methods',
            persist_dir_tg_methods='./chroma_db_tg_methods',
            persist_dir_eco_activity='./chroma_db_eco_activity'
        )
        retriever = LcaKnowledgeRetriever(vs_knowledge, parent_lookup, k_children=4).as_langchain_retriever()
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs) if docs else ""

    product_hint = f"Product: {product}\n" if product and product.strip() and product != 'Not Specified' else ""

    # Select role instruction based on classified intent
    intent = (intent_params.intent if intent_params else 'computation')
    if intent == 'literature':
        role_instruction = (
            "You are an expert LCA researcher. Summarize relevant peer-reviewed studies, "
            "methodologies, and findings from the knowledge provided. "
            "Highlight key data points and cite specific values where available. "
            "Structure your response with ## headings for major themes."
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
        "{product_hint}User Query: {question}\n\n"
        "Relevant LCA Knowledge:\n{context}\n\n"
        "Output valid Markdown. Use ## for section headings and - for bullet points. "
        "Use inline code for numeric values (e.g., `12.5 kg CO\u2082-eq`). "
        "Do not fabricate data not present in the context."
    )

    chain = ChatPromptTemplate.from_template(template) | gpt5_client | StrOutputParser()
    answer = chain.invoke({"question": question, "product_hint": product_hint, "context": context})

    return {"answer": answer, "markdown": None, "processed_json": None, "lcia_table": None}


def generate_response(question, product, mode='thinking'):
    # answer_pack = {
    #     "answer": None,
    #     "markdown": {},
    #     "processed_json": None,
    #     "lcia_table": None
    # }
    
    # return answer_pack
    preinstall_ecoinvent_database()  # pre-build BW2 databases to speed up first run
    start_time = time.time()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    section1_start = time.time()

    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = 'false'

    endpoint = "https://seesustainopedia-gpt5-resource.cognitiveservices.azure.com/"

    gpt5_model_name = "gpt-5"
    gpt5_deployment = "sustainopedia-gpt-5"
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
    section1_end = time.time()
    logging.info(f"Environment Setting took {section1_end - section1_start:.2f} seconds")

    # Infer product from the query if the user left the product field empty
    if not product or not product.strip():
        logging.info("Product field is empty — inferring product from user query...")
        product = infer_product_from_query(question, gpt5_client)
        logging.info(f"Inferred product: {product}")

    # Classify intent and extract structured LCA parameters for routing and storage
    intent_params = classify_intent(question, product, gpt5_client)
    logging.info(
        f"Intent: {intent_params.intent} | Product: {intent_params.product} | "
        f"{intent_params.amount} {intent_params.unit} | {intent_params.system_boundary}"
    )

    # Use the classifier's product if it is more specific than the provided value
    if intent_params.product and intent_params.product not in ('Not Specified', ''):
        product = intent_params.product

    # Fast mode: skip BW2 pipeline and return a direct LLM+RAG response
    if mode == 'fast':
        result = _fast_response(question, product, gpt5_client, embeddings, intent_params)
        result['intent_params'] = intent_params.to_dict()
        return result

    # Thinking mode: non-computation intents bypass the BW2 pipeline
    if intent_params.intent in ('literature', 'general'):
        logging.info(f"Routing '{intent_params.intent}' query to fast response")
        result = _fast_response(question, product, gpt5_client, embeddings, intent_params)
        result['intent_params'] = intent_params.to_dict()
        return result


    ###########1st stage -- Decision making: whether to do LCA analysis######
    # decision = conduct_lca_anaylsis(question, gpt5_client)
    # if decision.strip().lower() != "yes":
    #     answer_pack = {
    #     "answer": decision,
    #     "markdown": None,
    #     "processed_json": None,
    #     "lcia_table": None
    # }
    #     return answer_pack
    #2nd stage -- RAG to generate structured LCA JSON######



    ########### VECTORSTORE BUILDING AND LOADING
    if os.path.isdir('./chroma_db_lca_knowledge') and os.path.isdir('./chroma_db_eco_methods') and os.path.isdir('./chroma_db_tg_methods') and os.path.isdir('./chroma_db_eco_activity'):
        vectorstore_lca_knowledge, parent_lookup, vectorstore_eco_methods, vectorstore_tg_methods, vectorstore_eco_activity = load_vectorstores(embeddings, persist_dir_lca_knowledge='./chroma_db_lca_knowledge', persist_dir_eco_methods='./chroma_db_eco_methods', persist_dir_tg_methods='./chroma_db_tg_methods', persist_dir_eco_activity='./chroma_db_eco_activity')
    else:
        vectorstore_lca_knowledge, parent_lookup, vectorstore_eco_methods, vectorstore_tg_methods, vectorstore_eco_activity = build_and_save_vectorstores(embeddings, eco_methods='./_additionals/methods/EcoInvent_methods.txt', tg_methods='./_additionals/methods/tiangong_methods.txt', eco_activity='./_additionals/methods/Ecoinvent_activity.txt', pdf_dir='./_raw knowledge pdfs', persist_dir_lca_knowledge='./chroma_db_lca_knowledge', persist_dir_eco_methods='./chroma_db_eco_methods', persist_dir_tg_methods='./chroma_db_tg_methods', persist_dir_eco_activity='./chroma_db_eco_activity')

    retriever_lca_knowledge = LcaKnowledgeRetriever(vectorstore_lca_knowledge, parent_lookup, k_children=6).as_langchain_retriever()
    retriever_eco_methods = vectorstore_eco_methods.as_retriever(search_kwargs={"k": 3})
    retriever_tg_methods = vectorstore_tg_methods.as_retriever(search_kwargs={"k": 3})
    retriever_eco_activity = vectorstore_eco_activity.as_retriever(search_kwargs={"k": 1})

    #start rendering RAG
    if LCI_DATABASE == "Tiangong":
        retriever_lca_methods = retriever_tg_methods
    elif LCI_DATABASE == "Ecoinvent":
        retriever_lca_methods = retriever_eco_methods
    LCIA_method_list = Get_LCIA_methods(question, gpt5_client, retriever_lca_methods)

    #RAG retrieval
    section4_start = time.time()
    query_decomposition_retrival = query_decomposition_retrieval(question, gpt5_client, retriever_lca_knowledge, vectorstore_lca_knowledge, k_per_sub=3)
    section4_end = time.time()
    logging.info(f"retrieve alternative questions took {section4_end - section4_start:.2f} seconds")

    #1st results = bw2 input
    section5_start = time.time()
    bw2_input_filepath = generate_response_bw2_input(
        question, gpt5_client, query_decomposition_retrival, LCIA_method_list,
        product, LCI_DATABASE, retriever_eco_activity,
        n_simulations=intent_params.n_simulations
    )
    section5_end = time.time()
    logging.info(f"generate bw2 input took {section5_end - section5_start:.2f} seconds")
    
    
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
    
    bw2_results_raw_filename = f"bw2_results/{bw2_output_rawpath}_output.json"
    # bw2_results_raw_filename = f"bw2_results/Crude Steel_Ecoinvent_20260421184356_output.json"
    
    _, _, scale_factor = extract_functional_unit(question)
    res = final_mode_output_bw2_included(bw2_results_raw_filename, question, gpt5_client, product, scale_factor=scale_factor)
    logging.info(f"Final output generation completed, starting to prepare answer pack")
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
        "intent_params": intent_params.to_dict(),  # stored for frontend display and MongoDB
    }
    
    return answer_pack
