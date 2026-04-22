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
from langchain_core.runnables import RunnableLambda
from operator import itemgetter
import time
import re
from langchain_community.retrievers import BM25Retriever
# from langchain.chains.llm import LLMChain

# # 1. Hybrid Search (vectors + keywords)
# def hybrid_search(query, vectorstore, all_docs, k=5):
#     # Vector search
#     vector_docs = vectorstore.similarity_search(query, k=k)
    
#     # Keyword search using BM25
#     bm25_retriever = BM25Retriever.from_documents(all_docs)
#     bm25_retriever.k = k
#     keyword_docs = bm25_retriever.get_relevant_documents(query)
    
#     # Fuse results using reciprocal rank fusion
#     fused_docs = reciprocal_rank_fusion([vector_docs, keyword_docs])
#     return fused_docs

# # 2. HyDE (Hypothetical Document Embeddings)
# def hyde_retrieval(query, llm, embeddings, vectorstore, k=5):
#     # Prompt to generate hypothetical document
#     hyde_prompt = ChatPromptTemplate.from_template(
#         "Please write a detailed passage that could answer the following query: {query}"
#     )
#     hyde_chain = hyde_prompt | llm | StrOutputParser()
    
#     # Generate hypothetical document
#     hyp_doc = hyde_chain.invoke({"query": query})
    
#     # Embed the hypothetical document
#     hyp_embedding = embeddings.embed_query(hyp_doc)
    
#     # Retrieve using the hypothetical embedding
#     docs = vectorstore.similarity_search_by_vector(hyp_embedding, k=k)
#     return docs

# # 3. Hierarchical Indexing
# # Note: This assumes a simple hierarchical setup where summaries are stored in a separate vectorstore,
# # and full documents are linked via metadata. Setup would involve creating summary_docs and summary_vectorstore.
# # For retrieval: First retrieve summaries, then fetch corresponding full docs.
# def hierarchical_retrieval(query, summary_vectorstore, full_docs_dict, k_summaries=3, k_children=5):
#     # Retrieve top summaries
#     summary_docs = summary_vectorstore.similarity_search(query, k=k_summaries)
    
#     # Fetch child documents based on metadata (assuming metadata has 'children_ids')
#     child_ids = []
#     for summary in summary_docs:
#         child_ids.extend(summary.metadata.get('children_ids', []))
    
#     # Get unique full docs from dict (assuming full_docs_dict maps id to Document)
#     retrieved_docs = [full_docs_dict[cid] for cid in set(child_ids) if cid in full_docs_dict]
    
#     # Optional: Rerank or limit
#     if len(retrieved_docs) > k_children:
#         retrieved_docs = retrieved_docs[:k_children]
    
#     return retrieved_docs

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
    fused_docs = generate_queries | retriever_j.map() | get_unique_union
    return fused_docs

# # 5. Reranking
# def rerank_documents(query, docs, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
#     # Load cross-encoder for reranking
#     reranker = CrossEncoder(model_name)
    
#     # Prepare pairs
#     pairs = [[query, doc.page_content] for doc in docs]
    
#     # Get scores
#     scores = reranker.predict(pairs)
    
#     # Sort docs by score descending
#     sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
#     reranked_docs = [docs[i] for i in sorted_indices]
    
#     return reranked_docs[:5]  # Limit to top 5, matching pattern

# # 6. Self-RAG
# # Simplified: Retrieve, judge relevance with LLM, reflect and retrieve again if needed
# def self_rag_retrieval(query, llm, vectorstore, initial_docs, max_iterations=2):
#     current_docs = initial_docs
#     for _ in range(max_iterations):
#         # Judge relevance
#         judge_prompt = ChatPromptTemplate.from_template(
#             "Assess if the following documents are relevant to the query: {query}\nDocuments: {docs}\n"
#             "Output 'relevant' or 'needs_refinement'."
#         )
#         judge_chain = judge_prompt | llm | StrOutputParser()
#         docs_str = "\n".join([doc.page_content[:200] for doc in current_docs])  # Truncate for prompt
#         judgment = judge_chain.invoke({"query": query, "docs": docs_str}).strip().lower()
        
#         if 'relevant' in judgment:
#             break
        
#         # Reflect and generate better query
#         reflect_prompt = ChatPromptTemplate.from_template(
#             "The retrieval for query '{query}' needs refinement. Suggest an improved query based on documents: {docs}"
#         )
#         reflect_chain = reflect_prompt | llm | StrOutputParser()
#         improved_query = reflect_chain.invoke({"query": query, "docs": docs_str})
        
#         # Retrieve again
#         current_docs = vectorstore.similarity_search(improved_query, k=5)
    
#     return current_docs

# # 7. Corrective RAG (CRAG)
# # Simplified: Evaluate retrieval quality, if low, refine knowledge (here, assume refine by decomposing or skipping bad docs)
# # def corrective_rag(query, llm, vectorstore, initial_docs):
# #     # Evaluate each doc's relevance
# #     eval_prompt = PromptTemplate.from_template(
# #         "Score the relevance of this document to the query '{query}' on a scale of 1-5:\nDocument: {doc}\nScore:"
# #     )
# #     eval_chain = LLMChain(llm=llm, prompt=eval_prompt, output_key="score")
    
# #     corrected_docs = []
# #     for doc in initial_docs:
# #         result = eval_chain({"query": query, "doc": doc.page_content[:500]})
# #         score = int(re.search(r'\d+', result["score"]).group()) if re.search(r'\d+', result["score"]) else 1
# #         if score >= 3:
# #             corrected_docs.append(doc)
    
# #     # If too few good docs, fallback to query decomposition
# #     if len(corrected_docs) < 3:
# #         corrected_docs = query_decomposition_retrieval(query, llm, vectorstore)
    
# #     return corrected_docs

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