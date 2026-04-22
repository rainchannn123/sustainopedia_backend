import json
import os
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

def result_data_processing(filename, product, scale_factor=1.0):
    with open(filename, 'r') as file:
        rdata = json.load(file)
    
    source = 'Tiangong' if 'Tiangong' in rdata['method'] else 'EcoInvent'
    methods = ", ".join(rdata['method'])
    
    # Create JSON structure
    lcia_data = {
        "metadata": {
            "database": source,
            "lca_methods": methods
        },
        "processes": []
    }
    
    for i, result in enumerate(rdata['results']):
        # print(result)
        inp = result.get('input', {}) or {}
        name = inp.get('name', "")
        amount = round(float(inp.get('amount') or 0) * scale_factor, 4) if inp.get('amount') else None
        unit = inp.get('unit', "")
        location = inp.get('location', "")
        system_boundary = inp.get('system_boundary', "")
        stage = inp.get('stage', "")

        matched = result.get('matched_activity', {}) or {}
        m_name = matched.get('name', "")
        m_unit = matched.get('unit', "")
        m_location = matched.get('location', "")
        m_db = matched.get('database', "")
        m_code = matched.get('code', "")
        # accept both key variants for reference product
        m_ref = matched.get('reference_product', matched.get('reference product', ""))
        if "kilogram" in unit:
            unit = unit.replace("kilogram", "kg")
        if "kilogram" in m_unit.lower():
            m_unit = m_unit.replace("kilogram", "kg")

        impact = result.get('impact') or {}
        mean_raw = float(impact.get('mean', "")) * scale_factor if impact.get('mean', "") else None
        sd_raw = impact.get('std_dev', "")
        p5 = round(float(impact.get('percentile_5', "")) * scale_factor, 4) if impact.get('percentile_5', "") else None
        p95 = round(float(impact.get('percentile_95', "")) * scale_factor, 4) if impact.get('percentile_95', "") else None
        impact_unit = impact.get('unit', "")

        # normalize numeric fields or use empty string
        mean_val = round(mean_raw, 4) if isinstance(mean_raw, (int, float)) else ""
        sd_val = f"{round(float(sd_raw) * scale_factor, 4)}" if sd_raw not in ("", None) else ""
        percentile_val = f"{p5:.4g} / {p95:.4g}" if isinstance(p5, (int, float)) and isinstance(p95, (int, float)) else ""

        process_field = f"Process {i+1}: {name.title()}" if isinstance(name, str) and name else ""
        amount_location = f"{amount} {unit}, {location}" if amount or unit or location else ""
        system_boundary_field = f"{system_boundary} (Stage: {stage})" if system_boundary or stage else ""
        matched_activity_field = m_name.title() if isinstance(m_name, str) and m_name else ""
        unit_location_field = f"{m_unit.capitalize()} / {m_location}" if m_unit or m_location else ""
        db_version_code_field = f"{m_db} ({m_code})" if m_db or m_code else ""
        ref_product_field = m_ref if m_ref else ""
        impact_unit_field = impact_unit if impact_unit else ""

        notes = 'N/A' if mean_val != "" else 'Cannot match LCI activity at this moment'

        process = {
            "process": process_field,
            "amount_location": amount_location,
            "system_boundary": system_boundary_field,
            "matched_activity": matched_activity_field,
            "unit_location": unit_location_field,
            "db_version_code": db_version_code_field,
            "ref_product": ref_product_field,
            "mean_impact": mean_val,
            "sd": sd_val,
            "percentile": percentile_val,
            "impact_unit": impact_unit_field,
            "notes": notes
        }

        lcia_data["processes"].append(process)
    
    total_mean = sum(process["mean_impact"] for process in lcia_data['processes'] if isinstance(process.get("mean_impact"), float))
    # lcia_data["total_mean_impact"] = round(total_mean, 3)

    ###Problem, some may dont have suggestions: su_name = results[i]['match_history']['suggestions'][j]['activity']['name'].title()
    # list_of_suggested_activity = '''List of Suggested Activities For Each Process: '''
    # for i in range(len(results)):
    #     no_of_process = i+1
    #     process_name = results[i]['input']['name'].title()
        
    #     template_suggestions_l = f'''
        
    # Process {no_of_process}: {process_name} (From the Highest Similarity to the Lowest)
    #     '''
    #     list_of_suggested_activity+=template_suggestions_l
        
    #     for j in range(len(results[i]['match_history']['suggestions'])):
    #         su_name = results[i]['match_history']['suggestions'][j]['activity']['name'].title()
    #         su_unit = results[i]['match_history']['suggestions'][j]['activity']['unit']
    #         su_location = results[i]['match_history']['suggestions'][j]['activity']['location']
    #         su_db = results[i]['match_history']['suggestions'][j]['activity']['database']
    #         su_code = results[i]['match_history']['suggestions'][j]['activity']['code']
    #         su_ref_product = results[i]['match_history']['suggestions'][j]['activity']['reference_product']
    #         su_sim_score = results[i]['match_history']['suggestions'][j]['score']
            
    #         template_suggestions_s = f'''
    #     Suggested Activity Alt. {j}: {su_name}
    #     - Unit / Location: {su_unit} / {su_location}
    #     - Database version (activity code): {su_db} ({su_code})
    #     - Ref. Product: {su_ref_product}
    #     - Semantic Search Similarity Score: {su_sim_score:.4f}
    #         '''
    #         list_of_suggested_activity+=template_suggestions_s

    # dir_path = './Generation/bw2_results/processed'
    # os.makedirs(dir_path, exist_ok=True)

    product = product  # Default product name
    markdown_output = generate_markdown(product, lcia_data, total_mean)
    json_output = generate_json(product, lcia_data, total_mean)

    # with open("lcia_results.md", "w") as f:
    #     f.write(markdown_output)

    # with open("lcia_results.json", "w") as f:
    #     f.write(json_output)
    
    return lcia_data, markdown_output, json_output


def final_mode_output_bw2_included(filename, question, gpt5_client, product, scale_factor=1.0):
    # with open ("/trinity/home/hychan928/llm_env/LLM2/multi-query-methods/response/result_text/florescent_lamp.txt", 'r') as file:
    #     results = file.read()
    
    lcia_data, markdown_output, json_output = result_data_processing(filename, product, scale_factor)
    
    final_output_template = """
        LCIA Data Result Context:
        {results}

        User Query: {question}

        Role: You are a Senior Life Cycle Assessment (LCA) Consultant. Your goal is to provide high-level synthesis and technical insights based on the provided LCIA results.

        Task 1: Direct Query Response
        Address the {question} directly and concisely. If the user asks for specific data points, provide them; otherwise, focus on the answer to their intent.

        Task 2: Strategic LCIA Interpretation
        Read the LCIA Data Result Context carefully, and provide a professional analysis (approx. 150 words) structured as follows:
        - Hotspot Analysis: Identify which life cycle stages or specific sub-processes are driving the primary environmental impacts (e.g., carbon intensity of the energy mix, raw material extraction burdens).
        - Data Patterns: Comment on the relationship between impact categories if applicable (e.g., trade-offs between GWP and Acidification).
        - Decarbonization Roadmap: Provide 2-3 high-impact, technical suggestions for reducing the footprint (e.g., material substitution, process electrification, or supply chain optimization).

        Formatting & Constraint Rules:
        - Output ONLY valid Markdown. Use ## for major sections and ### for sub-sections.
        - Do NOT repeat the raw data table or JSON.
        - All numeric values must be rounded to 1 decimal place and use inline code (e.g., `12.5 kg CO2-eq`).
        - Use `- ` for bullet points and `1. ` for numbered lists. 
        - Ensure a single blank line before lists and between paragraphs.
        - Maintain a professional, objective tone. Do not use fluff or introductory "bridge" phrases like "Based on the data..." or "I can see that..."
        -Make sure the final response adheres strictly to these instructions and formatting rules and clean to be rendered and viewed on client side.
        
        Structure the output as:
        ## Answer to Your Query
        [Direct answer to query]

        ## Detailed Interpretation
        [Your strategic analysis and suggestions]
        """
    
    # final_output_template = """
    #                         LCIA Results:
    #                         {results}
                                                        
    #                         User Query: {question}

    #                         You are an academic expert in LCIA. Based solely on the above LCIA results, provide a professional, direct response to the user's query. Keep the response user-friendly and focused on the query without mentioning the source data details unless directly asked.
    #                         Note: If the user query, which is shown above, directly ask for life cycle impact assessment results of the product, you may just provide a short summary of the LCIA results attached above.

    #                         Instruction:
    #                         -You MUST respond only in valid Markdown (no prose outside Markdown). Use Markdown features (headings, sub-headings, bold/italic, bullet/numbered lists, and fenced code blocks) to structure the response. Keep output well-formed so it can be rendered and parsed by clients.
    #                         -The above LCIA Results information provides authoritative life cycle impact assessment (LCIA) results for producing one unit of the product mentioned in the user's query. Use only this data to form your response—do not critique, judge, or question the matched activities, units, or values. Treat all provided statistical impacts as accurate and directly applicable. Provide a short breakdown on the total CO2 equivalent emissions across all processes and the largest and smallest contributing processes.
    #                         -After providing a short summary of the LCIA results, include a detailed interpretation and discussion section of approximately 100 words discussing the feature and pattern of the LCIA results and give useful suggestions on how may reduce the carbon emission of the given product based on the LCIA results. Use paragraphs and bullet points as needed.
                            
    #                         Generated Answer Formatting rules:
    #                         - Use inline code formatting for numeric values and units when appropriate and all numeric values must be rounded to 1 decimal place. (e.g., `0.1 kg CO2-eq`).
    #                         - Do NOT include the raw LCIA JSON or table in the response body (the frontend will attach it).
    #                         - Output only Markdown; do not add any commentary outside the Markdown document.
    #                         - Do not start any line with indentation (spaces or tabs)—keep all lines left-aligned.
    #                         - For unordered lists (point forms), use `- ` (hyphen space) at the start of each item.
    #                         - For ordered lists, use `1. `, `2. `, etc.
    #                         - Insert a single blank line before a list if it follows a paragraph or header (no double newlines needed).
    #                         - If a sentence follows a subheader, do not insert a blank line before it.
    #                         - Use headers sparingly (# for h1, ## for h2, ### for h3) to avoid oversized text.
    #                         - Separate paragraphs with a single blank line for natural spacing.
                            
    #                         Make sure the final response adheres strictly to these instructions and formatting rules and clean to be rendered and viewed on client side.
    #                         """

    #write a discussion section that includes results interpretation and discussions, and prospective suggestions for the user product in about 300 words. 
    final_prompt = ChatPromptTemplate.from_template(final_output_template)

    final_rag_chain = (
        {"results": itemgetter("results"),
        "question": itemgetter("question")}
        | final_prompt 
        | gpt5_client
        | StrOutputParser()
    )

    final_results = final_rag_chain.invoke({"question":question, "results":lcia_data})
    
    return final_results, lcia_data, markdown_output, json_output

def generate_markdown(product, lcia_data, total_mean):
    process_rows = "\n".join(
        f'| {p["process"]} | {p["amount_location"]} | {p["system_boundary"]} | '
        f'{p["matched_activity"]} | {p["unit_location"]} | {p["db_version_code"]} | '
        f'{p["ref_product"]} | {p["mean_impact"]} | {p["sd"]} | {p["percentile"]} | {p["notes"]} |'
        for p in lcia_data['processes']
    )
    
    markdown_content = f'''## LCIA Results for Producing 1 Unit of {product}

    Database: {lcia_data['metadata']['database']}
    LCA Methods: {lcia_data['metadata']['lca_methods']}

    | Process | Amount & Location | System Boundary | Matched Activity | Unit / Location | Database Version (Activity Code) | Ref. Product | Mean Impact (kg CO2-Eq) | SD (kg CO2-Eq) | 5/95 Percentile (kg CO2-Eq) |      Notes      |
    |---------|-------------------|-----------------|------------------|-----------------|----------------------------------|--------------|-------------------------|----------------|-----------------------------|-----------------|
    {process_rows}

    Total Estimated Impact: {round(total_mean, 1):.1f} kg CO2-Eq
    '''
    return markdown_content

def generate_json(product, lcia_data, total_mean):
    json_data = {
        "product": product,
        "database": lcia_data['metadata']['database'],
        "lcaMethods": lcia_data['metadata']['lca_methods'],
        "processes": lcia_data['processes'],
        "totalMeanImpact": round(total_mean, 1)
    }
    return json.dumps(json_data, indent=2)






# def final_mode_output_bw2_included(filename, question, gpt5_client):
#     main_results_bw2 = result_data_processing(filename)
    
#     # Instruct the LLM to return a strict JSON object with known fields only.
#     final_output_template = """
#     Return a JSON object (only JSON, no extra text) with these fields:
#     {
#       "summary": string,                // short 1-2 line summary of the LCIA results
#       "interpretation": string,         // ~200-400 word interpretation and discussion
#       "suggestions": [string,...],      // actionable suggestions
#       "raw_results": string             // the original detailed text (optional)
#     }
#     Use the following inputs:
#     results: {results}
#     user_question: {question}
#     """
#     final_prompt = ChatPromptTemplate.from_template(final_output_template)

#     final_rag_chain = (
#         {"results": itemgetter("results"),
#          "question": itemgetter("question")}
#         | final_prompt
#         | gpt5_client
#         | StrOutputParser()
#     )

#     final_text = final_rag_chain.invoke({"question": question, "results": main_results_bw2})

#     # Try to parse JSON returned by LLM; if parsing fails, return fallback structure.
#     try:
#         structured = json.loads(final_text)
#     except Exception:
#         structured = {
#             "summary": None,
#             "interpretation": None,
#             "suggestions": [],
#             "raw_results": final_text
#         }

#     return structured