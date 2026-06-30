import os
import json
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from json import dumps, loads
from langchain_huggingface import HuggingFacePipeline
import sys
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter
import time
import re
from langchain_community.retrievers import BM25Retriever

# # 4. Query Decomposition
def query_decomposition_retrieval(query, gpt5_client, retriever_j, vectorstore, k_per_sub=3):
    template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide only these alternative questions separated by newlines without any other sentences other than these five questions. The answer should be in five rows with each row contains an alternative question. Original question: {question}"""
    prompt_perspectives = ChatPromptTemplate.from_template(template)
    generate_queries = (
        prompt_perspectives
        | gpt5_client
        | StrOutputParser()
        | RunnableLambda(lambda x: [q for q in re.split(r'\n+', x) if q.strip()])
    )
    try:
        fused_docs = generate_queries | retriever_j.map() | get_unique_union
    except:
        #rerun-it to prevent occasional LLM generation failure
        fused_docs = generate_queries | retriever_j.map() | get_unique_union
    return fused_docs


# # 4b. LCA Template-Targeted Knowledge Retrieval
def lca_template_knowledge_retrieval(query, gpt5_client, retriever_j, vectorstore, k_per_sub=4):
    """
    Targeted multi-objective RAG retrieval pipeline for BW2 input template generation.

    Whereas generic query decomposition generates perspective reformulations of the
    original question, this function generates sub-queries structured around the three
    knowledge dimensions that directly determine BW2 input template accuracy:

      Objective A — Recovery rates & process yields (3 sub-queries)
        Numeric yield percentages, material recovery fractions, conversion efficiencies,
        or process-loss data at each production stage of the product.

      Objective B — Life cycle process chain & activity sequence (3 sub-queries)
        Ordered unit processes, manufacturing stages, and supply chain activities
        involved in producing the product from raw material extraction to end-of-life.

      Objective C — Mass flows & material quantities per stage (3 sub-queries)
        Quantitative input/output data: kilograms of raw material consumed or converted
        at each stage per kilogram of product output (the basis for mass_in_kg /
        mass_out_kg fields in the BW2 template).

    The original query is prepended to the 9 sub-queries before retrieval so that
    baseline document coverage is always guaranteed regardless of sub-query quality.
    All 10 retrieval result sets are merged and deduplicated via get_unique_union (RRF).

    Returns a LangChain Runnable that accepts {"question": <str>} and produces
    list[Document], fully compatible with the context-injection chain in
    generate_response_bw2_input.
    """
    _targeted_template = """\
You are an LCA knowledge retrieval specialist preparing queries for a scientific \
literature vector database. From the user query below, generate exactly 9 targeted \
sub-questions covering three retrieval objectives. Output ONLY the 9 questions, \
one per line, no labels, no numbering, no extra text.

Objective A — Recovery rates & process yields (questions 1-3):
  Each question must target a numeric yield percentage, recovery rate, conversion \
efficiency, or material-loss fraction at a specific production stage of the product. \
Example form: "What is the recovery rate of [material] during [process stage]?"

Objective B — Life cycle process chain & activity sequence (questions 4-6):
  Each question must target the ordered set of unit processes, manufacturing stages, \
or supply chain activities for producing the product from cradle to grave. \
Example form: "What are the main life cycle stages and unit processes for [product]?"

Objective C — Mass flows & material quantities per stage (questions 7-9):
  Each question must target quantitative material input/output data: how many kilograms \
of raw material are consumed or transformed at each stage to yield one kilogram of \
product output. \
Example form: "How many kg of [raw material] are required per kg of [product] output at [stage]?"

User query: {question}"""

    prompt = ChatPromptTemplate.from_template(_targeted_template)

    # Parse LLM output into a clean list of at most 9 sub-questions
    _parse_sub_queries = RunnableLambda(
        lambda x: [q.strip() for q in re.split(r'\n+', x) if q.strip()][:9]
    )

    # Build a chain that keeps the original question alongside the generated sub-queries
    # so the retriever always has baseline coverage from the unmodified input.
    _sub_query_chain = prompt | gpt5_client | StrOutputParser() | _parse_sub_queries

    generate_queries = (
        RunnablePassthrough.assign(sub_questions=_sub_query_chain)
        | RunnableLambda(lambda x: [x["question"]] + x["sub_questions"])
    )

    try:
        fused_docs = generate_queries | retriever_j.map() | get_unique_union
    except:
        # Rerun to guard against occasional LLM generation failures
        fused_docs = generate_queries | retriever_j.map() | get_unique_union

    return fused_docs

def get_unique_union(documents: list[list]):
    def document_to_dict(doc):
        return {"page_content": doc.page_content, "metadata": doc.metadata}
    
    #documents.append(hyde_retrieved_docs)
    flattened = [dumps(document_to_dict(doc)) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened))
    return [Document(**loads(doc)) for doc in unique_docs]

# # Simplified reciprocal rank fusion
def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            print('reciprocal')
            doc_str = dumps({"page_content": doc.page_content, "metadata": doc.metadata})
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return [doc for doc, _ in reranked_results[:5]]  # Limit to top 5 results

def join_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def join_methods(methods):
    print(", ".join(method.page_content for method in methods))
    return ", ".join(method.page_content for method in methods) 