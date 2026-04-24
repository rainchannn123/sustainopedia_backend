import os
import json
import logging
import re
import shutil
import sys

# Override system sqlite3 with pysqlite3-binary BEFORE chromadb is imported.
# Azure App Service Linux ships SQLite < 3.35.0 which ChromaDB rejects.
# try:
#     import pysqlite3
#     sys.modules['sqlite3'] = pysqlite3
# except ImportError:
    # pass  # local dev — system SQLite is fine
    
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Re-export the new pipeline helpers so callers only need to import from here
from .lca_knowledge_pipeline import (
    build_lca_knowledge_store,
    load_lca_knowledge_store,
    LcaKnowledgeRetriever,
)


def build_and_save_vectorstores(
    embeddings,
    eco_methods='./_additionals/methods/EcoInvent_methods.txt',
    tg_methods='./_additionals/methods/tiangong_methods.txt',
    eco_activity='./_additionals/methods/Ecoinvent_activity.txt',
    pdf_dir='./_raw knowledge pdfs',
    persist_dir_lca_knowledge='./chroma_db_lca_knowledge',
    persist_dir_eco_methods='./chroma_db_eco_methods',
    persist_dir_tg_methods='./chroma_db_tg_methods',
    persist_dir_eco_activity='./chroma_db_eco_activity',
):
    # Delete existing Chroma databases if they exist
    for persist_dir in [persist_dir_lca_knowledge, persist_dir_eco_methods, persist_dir_tg_methods, persist_dir_eco_activity]:
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
            logging.info(f"Deleted existing Chroma database at {persist_dir}")

    # Read EcoInvent methods
    with open(eco_methods, 'r') as file:
        lines = file.readlines()
    ei_methods = []
    for line in lines:
        text = re.sub(r'^\d+\.\s*', '', line).strip()
        if text.startswith('(') and text.endswith(')'):
            text = '[' + text[1:-1] + ']'
        ei_methods.append(text)

    # Read Tiangong methods
    with open(tg_methods, 'r') as file:
        lines_tg = file.readlines()
    tg_methods_list = []
    for line in lines_tg:
        text_tg = re.sub(r'^\d+\.\s*', '', line).strip()
        if text_tg.startswith('(') and text_tg.endswith(')'):
            text_tg = '[' + text_tg[1:-1] + ']'
        tg_methods_list.append(text_tg)

    with open(eco_activity, 'r') as file:
        lines_ecoa = file.readlines()
    ecoa_activity = [line.strip() for line in lines_ecoa if line.strip()]

    # Build LCA knowledge store from PDFs (replaces the old JSON-based journal store)
    vectorstore_lca_knowledge, parent_lookup = build_lca_knowledge_store(
        embeddings=embeddings,
        pdf_dir=pdf_dir,
        persist_dir=persist_dir_lca_knowledge,
        rebuild=True,
    )

    # Create LCA methods documents
    splits_eco_methods = [Document(page_content=method, metadata={"source": "ecoinvent_methods", "method_name": method})
                          for method in ei_methods]
    splits_tg_methods = [Document(page_content=method, metadata={"source": "tiangong_methods", "method_name": method})
                         for method in tg_methods_list]
    splits_eco_activity = [Document(page_content=act, metadata={"source": "ecoinvent_activity", "method_name": act})
                           for act in ecoa_activity]

    vectorstore_eco_methods = Chroma.from_documents(
        documents=splits_eco_methods,
        embedding=embeddings,
        persist_directory=persist_dir_eco_methods,
    )
    vectorstore_tg_methods = Chroma.from_documents(
        documents=splits_tg_methods,
        embedding=embeddings,
        persist_directory=persist_dir_tg_methods,
    )
    vectorstore_eco_activity = Chroma.from_documents(
        documents=splits_eco_activity,
        embedding=embeddings,
        persist_directory=persist_dir_eco_activity,
    )

    vectorstore_eco_methods.persist()
    vectorstore_tg_methods.persist()
    vectorstore_eco_activity.persist()
    logging.info("Vectorstores built and saved successfully.")
    return vectorstore_lca_knowledge, parent_lookup, vectorstore_eco_methods, vectorstore_tg_methods, vectorstore_eco_activity

def load_vectorstores(
    embeddings,
    persist_dir_lca_knowledge='./chroma_db_lca_knowledge',
    persist_dir_eco_methods='./chroma_db_eco_methods',
    persist_dir_tg_methods='./chroma_db_tg_methods',
    persist_dir_eco_activity='./chroma_db_eco_activity',
):
    # Check if vectorstore directories exist
    for d in [persist_dir_lca_knowledge, persist_dir_eco_methods, persist_dir_tg_methods, persist_dir_eco_activity]:
        if not os.path.exists(d):
            raise FileNotFoundError(f"Vectorstore directory not found: '{d}'. Run build_and_save_vectorstores first.")
    # Load existing vectorstores
    vectorstore_lca_knowledge, parent_lookup = load_lca_knowledge_store(embeddings, persist_dir=persist_dir_lca_knowledge)
    vectorstore_eco_methods   = Chroma(persist_directory=persist_dir_eco_methods,   embedding_function=embeddings)
    vectorstore_tg_methods    = Chroma(persist_directory=persist_dir_tg_methods,    embedding_function=embeddings)
    vectorstore_eco_activity  = Chroma(persist_directory=persist_dir_eco_activity,  embedding_function=embeddings)

    logging.info("Vectorstores loaded from disk successfully.")
    return vectorstore_lca_knowledge, parent_lookup, vectorstore_eco_methods, vectorstore_tg_methods, vectorstore_eco_activity
