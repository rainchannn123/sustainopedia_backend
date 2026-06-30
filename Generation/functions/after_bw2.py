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

_UPSTREAM_KEYWORDS = {
    "raw material", "extraction", "mining", "beneficiation",
    "processing", "manufacturing", "production", "smelting",
    "fabrication", "assembly", "refining"
}
_DOWNSTREAM_KEYWORDS = {
    "distribution", "transport", "use phase", "use",
    "end-of-life", "end of life", "disposal", "recycling",
    "landfill", "incineration", "recovery", "remanufacturing"
}

def _classify_stream(stage: str) -> str:
    """Classify a stage name as upstream, downstream, or gate-to-gate."""
    s = (stage or "").lower()
    if any(kw in s for kw in _UPSTREAM_KEYWORDS):
        return "upstream"
    if any(kw in s for kw in _DOWNSTREAM_KEYWORDS):
        return "downstream"
    return "gate-to-gate"


# ─── Mass flow balance constants and functions ──────────────────────────────────
#: ISO 14040/44 mass balance tolerance thresholds.
#: 10 %/25 % are practical bounds for LLM-estimated flows;
#: stricter 1-2 % would apply to measured plant data.
_MASS_FLOW_TOLERANCE_WARN: float = 10.0   # % discrepancy → warning
_MASS_FLOW_TOLERANCE_GAP: float  = 25.0   # % discrepancy → gap (hard miss)

#: Physical mass units eligible for flow tracking.
_MASS_BEARING_UNITS: frozenset = frozenset({
    "kg", "kilogram", "kilograms", "g", "gram", "grams",
    "tonne", "ton", "metric ton", "metric tons",
    "lb", "pound", "pounds",
})


def _is_mass_bearing(unit: str) -> bool:
    """Return True when *unit* represents a physical mass quantity."""
    return unit.strip().lower() in _MASS_BEARING_UNITS



# following function is temporaryly closed
_EOL_STAGE_KEYWORDS: frozenset = frozenset({
    "disposal", "end-of-life", "end of life", "recycling", "waste",
})

_DISMANTLING_KEYWORDS: frozenset = frozenset({
    "dismantling", "Treatment", "dismantle", "collection for recycling", "collection for final",
})


def _balance_disposal_mass(functional_unit: list) -> list:
    """
    Remove whole-product dismantling/collection activities from the
    end-of-life stage when explicit sub-component treatment activities
    already cover the same mass stream, preventing ~2x double-counting.

    In ecoinvent, a "manual dismantling of used X" dataset and the
    individual per-material treatment datasets (powertrain, glider,
    electronics, etc.) both represent the same physical mass moving
    through disposal.  Including both a whole-product dismantling
    activity *and* its per-material downstream treatments in the
    functional unit counts every kg of the product twice through
    end-of-life, inflating (or deflating for credit streams) disposal
    impacts by approximately 2x.

    Rule: when the disposal stage contains mass-bearing
    dismantling/collection activities *and* mass-bearing sub-component
    treatment activities, and the sub-treatment total is ≥ 50 % of the
    dismantling total, the dismantling activities are removed — the
    sub-component breakdown is the more detailed representation.
    """
    def _is_eol(stage: str) -> bool:
        s = (stage or "").lower()
        return any(kw in s for kw in _EOL_STAGE_KEYWORDS)

    eol_pairs = [
        (i, a) for i, a in enumerate(functional_unit)
        if _is_eol(a.get("stage", "")) and _is_mass_bearing(a.get("unit", ""))
    ]
    if len(eol_pairs) < 2:
        return functional_unit

    dismantling_idxs = []
    sub_treatment_idxs = []
    for i, a in eol_pairs:
        name = (a.get("activity_name") or "").lower()
        if any(kw in name for kw in _DISMANTLING_KEYWORDS):
            dismantling_idxs.append(i)
        else:
            sub_treatment_idxs.append(i)

    if not dismantling_idxs or not sub_treatment_idxs:
        return functional_unit

    d_mass = sum((functional_unit[i].get("amount") or 0) for i in dismantling_idxs)
    s_mass = sum((functional_unit[i].get("amount") or 0) for i in sub_treatment_idxs)

    if d_mass <= 0 or s_mass < 0.5 * d_mass:
        return functional_unit

    to_remove = set(dismantling_idxs)
    removed_names = [functional_unit[i]["activity_name"] for i in sorted(to_remove)]
    logging.warning(
        "[disposal_mass_balance] Removing %d whole-product dismantling/collection "
        "activit%s %s (%.1f kg) — sub-component treatments (%.1f kg) already cover "
        "the same mass stream; retaining both would double-count end-of-life impacts.",
        len(removed_names),
        "y" if len(removed_names) == 1 else "ies",
        removed_names,
        d_mass,
        s_mass,
    )
    return [a for i, a in enumerate(functional_unit) if i not in to_remove]


def _validate_mass_flow_chain(functional_unit: list) -> dict:
    """
    Validate material mass flow consistency across unit processes.

    Implements ISO 14040/44 per-process mass balance checking: for each
    consecutive pair of mass-bearing activities carrying explicit
    ``mass_in_kg`` / ``mass_out_kg`` annotations, verifies that

        mass_out_kg[i]  ≈  mass_in_kg[i+1]

    within *_MASS_FLOW_TOLERANCE_WARN* / *_MASS_FLOW_TOLERANCE_GAP* thresholds.
    Activities that lack mass flow annotations or have non-mass units
    (transport, energy, services) are silently skipped.

    Parameters
    ----------
    functional_unit : list
        Activity dicts from a bw2 input template.  Each dict may optionally
        carry ``mass_in_kg``, ``mass_out_kg``, and ``recovery_rate`` fields.

    Returns
    -------
    dict
        ``status``        - "pass" | "warning" | "gap" | "skipped"
        ``overall_yield`` - last_mass_out / first_mass_in  (None if skipped)
        ``activities``    - per-activity mass summary list
        ``transitions``   - per-stage transition audit list
    """
    mass_activities = []
    for act in functional_unit:
        m_in  = act.get("mass_in_kg")
        m_out = act.get("mass_out_kg")
        if m_in is None or m_out is None:
            continue
        if not _is_mass_bearing(act.get("unit", "")):
            continue
        m_in_f  = float(m_in)
        m_out_f = float(m_out)
        r = act.get("recovery_rate")
        if r is not None:
            recovery = float(r)
        elif m_in_f > 0:
            recovery = round(m_out_f / m_in_f, 6)
        else:
            recovery = 1.0
        mass_activities.append({
            "stage":         act.get("stage", "unspecified"),
            "activity_name": act.get("activity_name", ""),
            "mass_in_kg":    round(m_in_f,  4),
            "mass_out_kg":   round(m_out_f, 4),
            "recovery_rate": round(recovery, 6),
        })

    if len(mass_activities) == 0:
        return {
            "status":        "skipped",
            "reason":        "no mass-annotated activities found",
            "overall_yield": None,
            "activities":    mass_activities,
            "transitions":   [],
        }

    if len(mass_activities) == 1:
        # Single-activity system: no inter-stage transitions to check.
        # Accept the user-supplied mass as-is and report pass so that
        # downstream LCIA computation is not altered or blocked.
        act = mass_activities[0]
        overall_yield = (
            round(act["mass_out_kg"] / act["mass_in_kg"], 6)
            if act["mass_in_kg"] > 0 else 1.0
        )
        return {
            "status":        "pass",
            "reason":        "single-activity system — no transitions to validate",
            "overall_yield": overall_yield,
            "activities":    mass_activities,
            "transitions":   [],
        }

    transitions = []
    for i in range(len(mass_activities) - 1):
        cur      = mass_activities[i]
        nxt      = mass_activities[i + 1]
        expected = cur["mass_out_kg"]
        actual   = nxt["mass_in_kg"]
        disc_pct = abs(expected - actual) / expected * 100.0 if expected > 0 else 0.0
        if disc_pct > _MASS_FLOW_TOLERANCE_GAP:
            t_status = "gap"
        elif disc_pct > _MASS_FLOW_TOLERANCE_WARN:
            t_status = "warning"
        else:
            t_status = "ok"
        transitions.append({
            "from_stage":          cur["stage"],
            "to_stage":            nxt["stage"],
            "expected_mass_in_kg": round(expected, 4),
            "actual_mass_in_kg":   round(actual,   4),
            "discrepancy_pct":     round(disc_pct, 2),
            "status":              t_status,
        })

    first_in      = mass_activities[0]["mass_in_kg"]
    last_out      = mass_activities[-1]["mass_out_kg"]
    overall_yield = round(last_out / first_in, 6) if first_in > 0 else None

    t_statuses = {t["status"] for t in transitions}
    agg_status = (
        "gap"     if "gap"     in t_statuses else
        "warning" if "warning" in t_statuses else
        "pass"
    )
    return {
        "status":        agg_status,
        "overall_yield": overall_yield,
        "activities":    mass_activities,
        "transitions":   transitions,
    }


def compute_mass_flow_balance(output_filepath: str) -> dict:
    """
    Load the corresponding bw2 input template and run mass flow chain
    validation on its ``functional_unit`` activities.

    The input template path is derived by replacing the ``_output.json``
    suffix with ``.json`` and the ``bw2_results/`` directory with
    ``bw2_input_templates/``.

    Returns the structured audit dict from :func:`_validate_mass_flow_chain`,
    or a ``"skipped"`` dict when the template cannot be located or parsed.
    """
    basename   = os.path.basename(output_filepath)
    input_name = basename.replace("_output.json", ".json")
    input_path = os.path.join("bw2_input_templates", input_name)

    if not os.path.isfile(input_path):
        logging.warning("Mass flow balance: input template not found at %s", input_path)
        return {
            "status":        "skipped",
            "reason":        f"input template not found: {input_path}",
            "overall_yield": None,
            "activities":    [],
            "transitions":   [],
        }

    try:
        with open(input_path, "r", encoding="utf-8") as fh:
            input_data = json.load(fh)
        return _validate_mass_flow_chain(input_data.get("functional_unit", []))
    except Exception as exc:
        logging.warning("Mass flow balance: error reading input template: %s", exc)
        return {
            "status":        "skipped",
            "reason":        str(exc),
            "overall_yield": None,
            "activities":    [],
            "transitions":   [],
        }


def result_data_processing(filename, product, scale_factor=1.0, display_amount=1.0, display_unit='tonne', activities_by_stage_dict=None):
    with open(filename, 'r') as file:
        rdata = json.load(file)

    source = 'Tiangong' if 'Tiangong' in rdata['method'] else 'EcoInvent'
    methods = ", ".join(rdata['method'])

    # ── Mass flow balance audit ────────────────────────────────────────────────
    mass_flow_balance = compute_mass_flow_balance(filename)
    # Per-activity mass annotation lookup: activity_name (lower) → mass fields
    _mf_act_map: dict = {}
    _input_path = os.path.join(
        "bw2_input_templates",
        os.path.basename(filename).replace("_output.json", ".json")
    )
    if os.path.isfile(_input_path):
        try:
            with open(_input_path, "r", encoding="utf-8") as _fh:
                _tpl = json.load(_fh)
            for _act in _tpl.get("functional_unit", []):
                _key = _act.get("activity_name", "").lower()
                if _key:
                    _mf_act_map[_key] = {
                        "mass_in_kg":   _act.get("mass_in_kg"),
                        "mass_out_kg":  _act.get("mass_out_kg"),
                        "recovery_rate": _act.get("recovery_rate"),
                    }
        except Exception:
            pass

    # Build a stage→stream lookup from the activities_by_stage_dict keys (if provided)
    stage_stream_map = {}
    if activities_by_stage_dict:
        for stage_key in activities_by_stage_dict.keys():
            stage_stream_map[stage_key.lower()] = _classify_stream(stage_key)

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
        unit = inp.get('unit', "")
        location = inp.get('location', "")
        system_boundary = inp.get('system_boundary', "")
        stage = inp.get('stage', "")

        # Discrete countable components (unit = "unit") must not be scaled by
        # scale_factor: a 2-tonne product still contains 1 motor, 1 battery, etc.
        _eff_scale = 1.0 if unit.strip().lower() == "unit" else scale_factor

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
        mean_raw = float(impact.get('mean', "")) * _eff_scale if impact.get('mean', "") else None
        sd_raw = impact.get('std_dev', "")
        p5 = round(float(impact.get('percentile_5', "")) * _eff_scale, 4) if impact.get('percentile_5', "") else None
        p95 = round(float(impact.get('percentile_95', "")) * _eff_scale, 4) if impact.get('percentile_95', "") else None
        impact_unit = impact.get('unit', "")

        # normalize numeric fields or use empty string
        mean_val = round(mean_raw, 4) if isinstance(mean_raw, (int, float)) else ""
        sd_val = f"{round(float(sd_raw) * _eff_scale, 4)}" if sd_raw not in ("", None) else ""
        percentile_val = f"{p5:.4g} / {p95:.4g}" if isinstance(p5, (int, float)) and isinstance(p95, (int, float)) else ""

        process_field = f"Process {i+1}: {name.title()}" if isinstance(name, str) and name else ""
        amount_location = f"{round(float(inp.get('amount') or 0) * _eff_scale, 4)} {unit}, {location}" if inp.get('amount') else (f"{unit}, {location}" if unit or location else "")
        system_boundary_field = f"{system_boundary} (Stage: {stage})" if system_boundary or stage else ""
        matched_activity_field = m_name.title() if isinstance(m_name, str) and m_name else ""
        unit_location_field = f"{m_unit.capitalize()} / {m_location}" if m_unit or m_location else ""
        db_version_code_field = f"{m_db} ({m_code})" if m_db or m_code else ""
        ref_product_field = m_ref if m_ref else ""
        impact_unit_field = impact_unit if impact_unit else ""

        notes = 'N/A' if mean_val != "" else 'Cannot match LCI activity at this moment'

        # Classify stream: prefer lookup from activities_by_stage_dict, fall back to keywords
        stream = stage_stream_map.get(stage.lower(), _classify_stream(stage))

        # Per-activity mass flow annotation (from input template lookup)
        _mf = _mf_act_map.get(name.lower()) or _mf_act_map.get(m_name.lower())

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
            "notes": notes,
            "stream": stream,
            "mass_in_kg":    _mf["mass_in_kg"]    if _mf else None,
            "mass_out_kg":   _mf["mass_out_kg"]   if _mf else None,
            "recovery_rate": _mf["recovery_rate"] if _mf else None,
        }

        lcia_data["processes"].append(process)
    
    total_mean = sum(process["mean_impact"] for process in lcia_data['processes'] if isinstance(process.get("mean_impact"), float))

    # Tally stream counts for the pie chart
    stream_counts = {"upstream": 0, "downstream": 0, "gate_to_gate": 0}
    for p in lcia_data["processes"]:
        key = p.get("stream", "gate-to-gate").replace("-", "_")
        if key in stream_counts:
            stream_counts[key] += 1
    lcia_data["metadata"]["stream_summary"] = stream_counts

    # ── Embed mass flow balance audit in metadata ─────────────────────────────
    lcia_data["metadata"]["mass_flow_balance"] = mass_flow_balance

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
    markdown_output = generate_markdown(product, lcia_data, total_mean, display_amount=display_amount, display_unit=display_unit)
    json_output = generate_json(product, lcia_data, total_mean)

    # with open("lcia_results.md", "w") as f:
    #     f.write(markdown_output)

    # with open("lcia_results.json", "w") as f:
    #     f.write(json_output)
    
    return lcia_data, markdown_output, json_output


def final_mode_output_bw2_included(filename, question, gpt5_client, product, scale_factor=1.0, n_simulations=25, display_amount=1.0, display_unit='tonne', activities_by_stage_dict=None):
    # with open ("/trinity/home/hychan928/llm_env/LLM2/multi-query-methods/response/result_text/florescent_lamp.txt", 'r') as file:
    #     results = file.read()
    
    lcia_data, markdown_output, json_output = result_data_processing(filename, product, scale_factor, display_amount=display_amount, display_unit=display_unit, activities_by_stage_dict=activities_by_stage_dict)
    
    final_output_template = """
        LCIA Data Result Context:
        {results}

        Functional Unit: {functional_unit}
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
        - Make sure the final response adheres strictly to these instructions and formatting rules and is clean to be rendered and viewed on the client side.
        - The number of simulations to run Monte Carlo analysis is {n_simulations}. You can add this information in your answer shortly and clearly. No need for further explanation on what Monte Carlo analysis is or why this number of simulations unless the user explicitly asks for it.     
        - All reported emissions or impacts must be expressed exclusively in kg CO₂-eq. Do not use other equivalence units (e.g., Sb-eq, SO₂-eq, PO₄³⁻-eq). Always normalize results to CO₂-eq.
        - You must label the LCA methods used for the LCIA results in short for providing this information to the user.
        
        Structure the output as:
        ## Summary of LCIA Results
        [Direct answer to query]

        ## Detailed Interpretation
        [Your strategic analysis and suggestions]
        """
        
    final_prompt = ChatPromptTemplate.from_template(final_output_template)

    final_rag_chain = (
        {"results": itemgetter("results"),
        "question": itemgetter("question"),
        "n_simulations": itemgetter("n_simulations"),
        "functional_unit": itemgetter("functional_unit"),
        }
        | final_prompt 
        | gpt5_client
        | StrOutputParser()
    )

    functional_unit_str = f"{display_amount} {display_unit} of {product}"
    try:
        final_results = final_rag_chain.invoke({"question":question, "results":lcia_data, "n_simulations": n_simulations, "functional_unit": functional_unit_str})
    except:
        # Rerun once to mitigate occasional LLM generation failures
        final_results = final_rag_chain.invoke({"question":question, "results":lcia_data, "n_simulations": n_simulations, "functional_unit": functional_unit_str})
    
    return final_results, lcia_data, markdown_output, json_output

def generate_markdown(product, lcia_data, total_mean, display_amount=1.0, display_unit='tonne'):
    process_rows = "\n".join(
        f'| {p["process"]} | {p["amount_location"]} | {p["system_boundary"]} | '
        f'{p["matched_activity"]} | {p["unit_location"]} | {p["db_version_code"]} | '
        f'{p["ref_product"]} | {p["mean_impact"]} | '
        f'{"{:.1f}".format(p["contribution_pct"]) if isinstance(p.get("contribution_pct"), float) else "N/A"} | '
        f'{p["sd"]} | {p["percentile"]} | {p["notes"]} |'
        for p in lcia_data['processes']
    )

    _mfb             = lcia_data['metadata'].get('mass_flow_balance', {})
    _mfb_status      = _mfb.get('status', 'skipped')
    _mfb_transitions = _mfb.get('transitions', [])
    _mfb_yield       = _mfb.get('overall_yield')
    if _mfb_status not in ('skipped',) and _mfb_transitions:
        _mf_rows = "\n    ".join(
            f'- {t["from_stage"]} → {t["to_stage"]}: '
            f'expected `{t["expected_mass_in_kg"]:.1f} kg` / '
            f'actual `{t["actual_mass_in_kg"]:.1f} kg` '
            f'(discrepancy `{t["discrepancy_pct"]:.1f} %`) [{t["status"].upper()}]'
            for t in _mfb_transitions
        )
        _yield_str   = f"`{_mfb_yield:.4f}`" if _mfb_yield is not None else "N/A"
        _mfb_section = (
            f"\n    ### Mass Flow Balance Audit (ISO 14040/44)\n"
            f"    Status: **{_mfb_status.upper()}** | "
            f"Overall Material Yield: {_yield_str}\n"
            f"    {_mf_rows}\n"
        )
    else:
        _mfb_section = ''

    markdown_content = f'''## LCIA Results for Producing {display_amount} {display_unit} of {product}

    Database: {lcia_data['metadata']['database']}
    LCA Methods: {lcia_data['metadata']['lca_methods']}

    | Process | Amount & Location | System Boundary | Matched Activity | Unit / Location | Database Version (Activity Code) | Ref. Product | Mean Impact (kg CO2-Eq) | Contribution (%) | SD (kg CO2-Eq) | 5/95 Percentile (kg CO2-Eq) |      Notes      |
    |---------|-------------------|-----------------|------------------|-----------------|----------------------------------|--------------|-------------------------|------------------|----------------|-----------------------------|-----------------|
    {process_rows}

    Total Estimated Impact: {round(total_mean, 1):.1f} kg CO2-Eq{_mfb_section}
    '''
    return markdown_content

def generate_json(product, lcia_data, total_mean):
    json_data = {
        "product": product,
        "database": lcia_data['metadata']['database'],
        "lcaMethods": lcia_data['metadata']['lca_methods'],
        "processes": lcia_data['processes'],
        "totalMeanImpact": round(total_mean, 1),
        "streamSummary": lcia_data['metadata'].get('stream_summary', {}),
        "stageImpacts": lcia_data['metadata'].get('stage_impacts', {}),
        "massFlowBalance": lcia_data['metadata'].get('mass_flow_balance', {}),
    }
    return json.dumps(json_data, indent=2)