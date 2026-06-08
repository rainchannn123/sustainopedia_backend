"""
analyze_activity.py
───────────────────
Inspect a specific ecoinvent activity in detail:

  1. List every direct exchange  (production output / technosphere inputs / biosphere emissions)
  2. Show the full LCIA score breakdown:
       - direct emissions and their characterisation contribution
       - top upstream processes ranked by contribution
       - cumulative % coverage
  3. Export results to two CSV files next to this script

Run from the project root (where brightway_databases/ lives):
    python "bw2_sources/Big Mushroom/Small Mushroom/analyze_activity.py"

──────────────────────────────────────────────────────────────────────────────
CONFIGURE HERE
──────────────────────────────────────────────────────────────────────────────
"""
ACTIVITY_NAME     = "market for fibre, viscose"
ACTIVITY_LOCATION = "GLO"          # set to None to match any location
AMOUNT            = 1.0            # functional unit quantity
LCIA_METHOD       = ('IPCC 2021', 'climate change', 'global warming potential (GWP100)')
                                   # set to None to skip LCIA and only show exchanges
TOP_N             = 20             # number of top flows / processes to display

# ── Specification per process (SimaPro equivalent) ────────────────────────────
# Set SPEC_FLOW_NAME to an elementary flow name to see which upstream processes
# emit/consume it and their individual contributions to the LCIA score.
# Equivalent to SimaPro → Network/Tree → "Specification per process" for a substance.
SPEC_FLOW_NAME        = "Carbon dioxide, fossil"  # set to None to skip
SPEC_FLOW_COMPARTMENT = None  # None = aggregate ALL compartments; or e.g. "air" to filter

# Set True to print all biosphere flows matching SPEC_FLOW_NAME (with exact
# compartments) WITHOUT running LCIA — use this to find the right values above.
SPEC_FLOW_DISCOVER    = False

# ─────────────────────────────────────────────────────────────────────────────

import os, sys, csv
import numpy as np

# Must be set before ANY bw2data import so brightway2 points at the right dir
os.environ["BRIGHTWAY2_DIR"] = "./brightway_databases/brightway2_ecoinvent"

import bw2data as _bw2data
from bw2data import projects, databases, Database, methods as bw_methods
import bw2calc

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── brightway2 setup ──────────────────────────────────────────────────────────

def setup():
    projects.set_current("ecoinvent")
    print(f"Project   : {projects.current}")
    print(f"Databases : {list(databases)}")

# ── activity lookup ───────────────────────────────────────────────────────────

def find_activity(name, location=None):
    db = Database("ecoinvent")
    all_named = [a for a in db if a["name"].lower() == name.lower()]

    if not all_named:
        raise ValueError(f"Activity not found with name: '{name}'")

    # Try exact location first
    if location:
        exact = [a for a in all_named if a.get("location") == location]
        if exact:
            if len(exact) > 1:
                print(f"\n[!] {len(exact)} exact matches — using first.")
            return exact[0]

        # NOTE: ecoinvent production activities (e.g. "X production, Y") are
        # region-specific and never have a GLO entry. GLO is reserved for
        # market activities that aggregate weighted regional contributions.
        # RoW (Rest of World) is the closest proxy for a global average when
        # no GLO variant exists.
        print(f"\n[!] Location '{location}' not found for '{name}'.")
        print(f"    Available locations: "
              f"{sorted(set(a.get('location','?') for a in all_named))}")

        # Fallback priority: RoW → first alphabetically
        row = [a for a in all_named if a.get("location") == "RoW"]
        if row:
            print(f"    → Falling back to 'RoW' (Rest of World — global proxy).")
            return row[0]

        fallback = sorted(all_named, key=lambda a: a.get("location", ""))[0]
        print(f"    → Falling back to '{fallback.get('location')}' "
              f"(first available location).")
        return fallback

    # No location filter — if multiple, show options and use first
    if len(all_named) > 1:
        print(f"\n[!] {len(all_named)} matches found (no location filter) — using first.")
        print(f"    All locations: "
              f"{sorted(set(a.get('location','?') for a in all_named))}")
        print(f"    Set ACTIVITY_LOCATION to narrow down.")
    return all_named[0]

# ── exchange display ──────────────────────────────────────────────────────────

def print_exchanges(activity):
    sep = "═" * 80
    print(f"\n{sep}")
    print(f"  ACTIVITY  :  {activity['name']}")
    print(f"  Ref. prod :  {activity.get('reference product', 'N/A')}")
    print(f"  Location  :  {activity.get('location', 'N/A')}")
    print(f"  Unit      :  {activity.get('unit', 'N/A')}")
    print(f"  Code      :  {activity.get('code', 'N/A')}")
    print(sep)

    groups = {"production": [], "technosphere": [], "biosphere": [], "other": []}
    for exc in activity.exchanges():
        t = exc.get("type", "other")
        groups.setdefault(t, []).append(exc)

    # Production
    print("\n── PRODUCTION OUTPUT " + "─" * 59)
    for exc in groups["production"]:
        inp = exc.input
        print(f"  + {exc['amount']:>14.6g}  {exc.get('unit',''):12}  {inp['name']}")

    # Technosphere inputs
    print("\n── TECHNOSPHERE INPUTS  (upstream supply chain) " + "─" * 32)
    tech = sorted(groups["technosphere"], key=lambda e: abs(e["amount"]), reverse=True)
    if tech:
        print(f"  {'Amount':>16}  {'Unit':12}  {'Input activity'}  [location]")
        print(f"  {'─'*16}  {'─'*12}  {'─'*45}")
        for exc in tech:
            inp = exc.input
            print(f"  {exc['amount']:>16.6g}  {exc.get('unit',''):12}  "
                  f"{inp['name']}  [{inp.get('location', '')}]")
    else:
        print("  (none)")

    # Biosphere emissions
    print("\n── DIRECT BIOSPHERE EMISSIONS / RESOURCE USE " + "─" * 35)
    bio = sorted(groups["biosphere"], key=lambda e: abs(e["amount"]), reverse=True)
    if bio:
        print(f"  {'Amount':>16}  {'Unit':12}  {'Elementary flow'}  (compartment)")
        print(f"  {'─'*16}  {'─'*12}  {'─'*45}")
        for exc in bio:
            inp = exc.input
            cats = " > ".join(inp.get("categories", []))
            print(f"  {exc['amount']:>16.6g}  {exc.get('unit',''):12}  "
                  f"{inp['name']}  ({cats})")
    else:
        print("  (none — market/linking activity: no direct emissions)")

    print()
    return groups

# ── Contribution analysis helpers (bypass bw2analyzer scipy int-index bug) ────

def _top_emissions(lca, limit=20):
    """Top elementary flows by characterised impact score."""
    scores = np.array(lca.characterized_inventory.sum(axis=1)).flatten()
    rev_bio = {v: k for k, v in lca.biosphere_dict.items()}
    top_idx = np.argsort(np.abs(scores))[::-1][:limit]
    results = []
    for idx in top_idx:
        i = int(idx)
        score = float(scores[i])
        key = rev_bio.get(i)
        if key and score != 0:
            flow = _bw2data.get_activity(key)
            results.append((score, float(lca.inventory[i, :].sum()), flow))
    return results


def _top_processes(lca, limit=20):
    """Top upstream processes by characterised impact score."""
    scores = np.array(lca.characterized_inventory.sum(axis=0)).flatten()
    rev_act = {v: k for k, v in lca.activity_dict.items()}
    top_idx = np.argsort(np.abs(scores))[::-1][:limit]
    results = []
    for idx in top_idx:
        i = int(idx)
        score = float(scores[i])
        key = rev_act.get(i)
        if key and score != 0:
            act = _bw2data.get_activity(key)
            results.append((score, float(lca.supply_array[i]), act))
    return results


# ── LCIA analysis ─────────────────────────────────────────────────────────────

def run_lcia(activity, method, amount=1.0):
    sep = "═" * 80
    print(f"\n{sep}")
    print(f"  LCIA SCORE BREAKDOWN")
    print(f"  Method         :  {method}")
    print(f"  Functional unit:  {amount} {activity.get('unit','?')} of "
          f"'{activity['name']}' [{activity.get('location','')}]")
    print(sep)

    fu  = {activity: amount}
    lca = bw2calc.LCA(fu, method)
    lca.lci()
    lca.lcia()
    print(f"Characterized inventory: {lca.characterized_inventory}")

    print(f"\n  ► Total LCIA score : {lca.score:.6g}  (unit of method's characterisation)")

    # ── Direct emission contributions ─────────────────────────────────────────
    # For the specific activity column in the B-matrix (before supply chain expansion)
    print(f"\n── HOW THE SCORE IS BUILT  (score = Q · B · A⁻¹ · f) " + "─" * 27)
    print("""
  Step-by-step calculation logic:
    f      → functional unit demand vector  (1 unit of this activity, 0 for all others)
    A⁻¹·f  → scaling vector: how many times EACH upstream process must run
    B·s    → inventory vector: total elementary flows summed over all processes
    Q·inv  → characterised impacts: each flow × its LCIA characterisation factor
    score  → sum of all characterised impacts
""")

    # ── Top contributing elementary flows ─────────────────────────────────────
    print(f"── TOP {TOP_N} CONTRIBUTING ELEMENTARY FLOWS " + "─" * 39)
    print(f"  {'Rank':>4}  {'Contribution':>14}  {'% of total':>10}  "
          f"{'Elementary flow'}  (compartment)")
    print(f"  {'─'*4}  {'─'*14}  {'─'*10}  {'─'*50}")
    top_flows = _top_emissions(lca, limit=TOP_N)
    cumulative = 0.0
    for rank, (score, _, flow) in enumerate(top_flows, 1):
        pct = 100.0 * score / lca.score if lca.score else 0.0
        cumulative += pct
        cats = " > ".join(flow.get("categories", []))
        print(f"  {rank:>4}  {score:>14.6g}  {pct:>9.2f}%  "
              f"{flow['name']}  ({cats})")
    print(f"\n  Cumulative coverage of top {TOP_N} flows: {cumulative:.1f}%")

    # ── Top contributing upstream processes ───────────────────────────────────
    print(f"\n── TOP {TOP_N} CONTRIBUTING UPSTREAM PROCESSES " + "─" * 37)
    print(f"  {'Rank':>4}  {'Contribution':>14}  {'% of total':>10}  "
          f"{'Process'}  [location]")
    print(f"  {'─'*4}  {'─'*14}  {'─'*10}  {'─'*50}")
    top_procs = _top_processes(lca, limit=TOP_N)
    cumulative = 0.0
    for rank, (score, _, act) in enumerate(top_procs, 1):
        pct = 100.0 * score / lca.score if lca.score else 0.0
        cumulative += pct
        print(f"  {rank:>4}  {score:>14.6g}  {pct:>9.2f}%  "
              f"{act['name']}  [{act.get('location','')}]")
    print(f"\n  Cumulative coverage of top {TOP_N} processes: {cumulative:.1f}%")

    print()
    return lca

# ── CSV export ────────────────────────────────────────────────────────────────

def export_csvs(activity, groups, lca=None, method=None):
    safe_name = (activity["name"].replace("/", "_")
                                 .replace(",", "")
                                 .replace(" ", "_"))[:60]

    # exchanges CSV
    exc_path = os.path.join(OUT_DIR, f"{safe_name}_exchanges_detail.csv")
    with open(exc_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["exchange_type", "amount", "unit",
                    "input_name", "input_location", "input_database",
                    "input_code", "categories"])
        for exc_type, excs in groups.items():
            for exc in excs:
                inp = exc.input
                cats = " > ".join(inp.get("categories", []))
                w.writerow([exc_type, exc["amount"], exc.get("unit", ""),
                             inp["name"], inp.get("location", ""),
                             inp.get("database", ""), inp.get("code", ""), cats])
    print(f"  [saved] {exc_path}")

    # LCIA contribution CSV
    if lca and method:
        contrib_path = os.path.join(OUT_DIR, f"{safe_name}_lcia_contributions.csv")
        with open(contrib_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rank", "category", "contribution",
                         "pct_of_total", "name", "location_or_categories", "unit"])
            for i, (score, _, item) in enumerate(_top_emissions(lca, limit=50), 1):
                pct = round(100.0 * score / lca.score, 4) if lca.score else 0
                cats = " > ".join(item.get("categories", []))
                w.writerow([i, "elementary_flow", score, pct,
                             item["name"], cats, item.get("unit", "")])
            for i, (score, _, item) in enumerate(_top_processes(lca, limit=50), 1):
                pct = round(100.0 * score / lca.score, 4) if lca.score else 0
                w.writerow([i, "upstream_process", score, pct,
                             item["name"], item.get("location", ""),
                             item.get("unit", "")])
        print(f"  [saved] {contrib_path}")

# ── Markdown report ───────────────────────────────────────────────────────────

def export_markdown(activity, groups, lca=None, method=None):
    import datetime
    safe_name = (activity["name"].replace("/", "_")
                                 .replace(",", "")
                                 .replace(" ", "_"))[:60]
    md_path = os.path.join(OUT_DIR, f"{safe_name}_report.md")

    lines = []
    def ln(s=""): lines.append(s)

    ln(f"# LCA Activity Analysis Report")
    ln()
    ln(f"**Generated:** {datetime.date.today()}  ")
    ln(f"**Activity:** {activity['name']}  ")
    ln(f"**Location:** {activity.get('location', 'N/A')}  ")
    ln(f"**Reference product:** {activity.get('reference product', 'N/A')}  ")
    ln(f"**Unit:** {activity.get('unit', 'N/A')}  ")
    ln(f"**Code:** {activity.get('code', 'N/A')}  ")
    ln()
    ln("---")
    ln()

    # Exchanges
    ln("## Exchanges")
    ln()

    # Production
    ln("### Production Output")
    ln()
    ln("| Amount | Unit | Product |")
    ln("|---|---|---|")
    for exc in groups.get("production", []):
        ln(f"| {exc['amount']:.6g} | {exc.get('unit','')} | {exc.input['name']} |")
    ln()

    # Technosphere
    ln("### Technosphere Inputs (Upstream Supply Chain)")
    ln()
    tech = sorted(groups.get("technosphere", []), key=lambda e: abs(e["amount"]), reverse=True)
    if tech:
        ln("| Amount | Unit | Input Activity | Location |")
        ln("|---|---|---|---|")
        for exc in tech:
            inp = exc.input
            ln(f"| {exc['amount']:.6g} | {exc.get('unit','')} | {inp['name']} | {inp.get('location','')} |")
    else:
        ln("*(none — market/linking activity)*")
    ln()

    # Biosphere
    ln("### Direct Biosphere Emissions / Resource Use")
    ln()
    bio = sorted(groups.get("biosphere", []), key=lambda e: abs(e["amount"]), reverse=True)
    if bio:
        ln("| Amount | Unit | Elementary Flow | Compartment |")
        ln("|---|---|---|---|")
        for exc in bio:
            inp = exc.input
            cats = " > ".join(inp.get("categories", []))
            ln(f"| {exc['amount']:.6g} | {exc.get('unit','')} | {inp['name']} | {cats} |")
    else:
        ln("*(none — market/linking activity: no direct emissions)*")
    ln()

    # LCIA
    if lca and method:
        ln("---")
        ln()
        ln("## LCIA Score Breakdown")
        ln()
        ln(f"**Method:** {method}  ")
        ln(f"**Functional unit:** {AMOUNT} {activity.get('unit','?')} of "
           f"*{activity['name']}* [{activity.get('location','')}]  ")
        ln(f"**Total LCIA score:** `{lca.score:.6g}`  ")
        ln()
        ln("> **Score equation:** score = Q · B · A⁻¹ · f  ")
        ln("> - **f** = functional unit demand vector  ")
        ln("> - **A⁻¹·f** = supply chain scaling vector (how many times each upstream process runs)  ")
        ln("> - **B·s** = total elementary flows across all processes  ")
        ln("> - **Q·inv** = each flow × its characterisation factor → impact score  ")
        ln()

        # Top flows
        ln(f"### Top {TOP_N} Contributing Elementary Flows")
        ln()
        ln("| Rank | Contribution | % of Total | Elementary Flow | Compartment |")
        ln("|---|---|---|---|---|")
        top_flows = _top_emissions(lca, limit=TOP_N)
        cumulative = 0.0
        for rank, (score, _, flow) in enumerate(top_flows, 1):
            pct = 100.0 * score / lca.score if lca.score else 0.0
            cumulative += pct
            cats = " > ".join(flow.get("categories", []))
            ln(f"| {rank} | {score:.6g} | {pct:.2f}% | {flow['name']} | {cats} |")
        ln()
        ln(f"*Cumulative coverage: {cumulative:.1f}%*")
        ln()

        # Top processes
        ln(f"### Top {TOP_N} Contributing Upstream Processes")
        ln()
        ln("| Rank | Contribution | % of Total | Process | Location |")
        ln("|---|---|---|---|---|")
        top_procs = _top_processes(lca, limit=TOP_N)
        cumulative = 0.0
        for rank, (score, _, act) in enumerate(top_procs, 1):
            pct = 100.0 * score / lca.score if lca.score else 0.0
            cumulative += pct
            ln(f"| {rank} | {score:.6g} | {pct:.2f}% | {act['name']} | {act.get('location','')} |")
        ln()
        ln(f"*Cumulative coverage: {cumulative:.1f}%*")
        ln()

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [saved] {md_path}")

# ── Specification per process ────────────────────────────────────────────────

def specification_per_process(lca, flow_name, compartment=None, limit=200):
    """
    For a specific elementary flow, list every upstream process and its individual
    contribution to that flow's total characterized impact — the brightway2
    equivalent of SimaPro's 'Specification per process' view.

    Methodology
    ───────────
    lca.characterized_inventory  is a (biosphere flows × activities) sparse matrix.
    Each element [i, j] = characterisation_factor[i] × inventory[i, j]
                        = the characterized impact of flow i emitted by process j.
    Summing row i across all columns gives the total characterized impact of flow i.
    Each column j in that row is the per-process contribution shown here.
    """
    sep = "═" * 80

    # ── find matching biosphere flow row(s) ──────────────────────────────────
    rev_bio = {v: k for k, v in lca.biosphere_dict.items()}
    matches = []
    for idx, key in rev_bio.items():
        try:
            flow = _bw2data.get_activity(key)
        except Exception:
            continue
        if flow["name"].lower() == flow_name.lower():
            if compartment is None:
                matches.append((idx, flow))
            else:
                cats = " > ".join(flow.get("categories", [])).lower()
                if compartment.lower() in cats:
                    matches.append((idx, flow))

    if not matches:
        # Separate diagnosis: did the name match but the compartment filter excluded it?
        name_only = []
        for idx, key in rev_bio.items():
            try:
                flow = _bw2data.get_activity(key)
            except Exception:
                continue
            if flow["name"].lower() == flow_name.lower():
                name_only.append((idx, flow))

        if name_only:
            print(f"\n[!] Flow '{flow_name}' found in the biosphere matrix but "
                  f"compartment filter '{compartment}' matched nothing.")
            print(f"    Flows with that name and their EXACT compartments:")
            print(f"    {'idx':>6}  {'Compartment (categories)'}")
            print(f"    {'─'*6}  {'─'*55}")
            for idx, flow in sorted(name_only, key=lambda x: x[0]):
                cats = " > ".join(flow.get("categories", []))
                print(f"    {idx:>6}  {cats!r}")
            print(f"\n    → Set SPEC_FLOW_COMPARTMENT to a substring of one of the "
                  f"compartments above, or None to accept all.")
        else:
            # Name itself not found — search broadly
            similar = []
            for k in rev_bio.values():
                try:
                    n = _bw2data.get_activity(k).get("name", "")
                except Exception:
                    continue
                if flow_name.lower() in n.lower():
                    similar.append(n)
            similar = sorted(set(similar))[:15]
            print(f"\n[!] Flow '{flow_name}' not found in the biosphere matrix.")
            if similar:
                print(f"    Flows with similar names in the matrix:")
                for n in similar:
                    print(f"      {n!r}")
            else:
                print(f"    No similar names found — run with SPEC_FLOW_DISCOVER=True "
                      f"to search the full biosphere database.")
            print(f"    Update SPEC_FLOW_NAME / SPEC_FLOW_COMPARTMENT and re-run.")
        return None

    flow_idx, matched_flow = matches[0]
    cats = " > ".join(matched_flow.get("categories", []))

    if len(matches) > 1:
        print(f"\n  [{len(matches)} compartment variants found — aggregating all into one total]")
        for idx, flow in matches:
            c = " > ".join(flow.get("categories", [])) or "(no compartment stored)"
            print(f"    row {idx:>4}:  {flow['name']}  |  {c}")
        print()

    # ── extract per-process contributions for this flow ───────────────────────
    # If multiple compartments matched, sum their rows so we get the total
    # contribution of this substance across ALL compartments (e.g. CO2 fossil
    # in air, air > unspecified, etc.) — matching SimaPro's aggregated view.
    if len(matches) > 1:
        row = np.zeros(lca.characterized_inventory.shape[1], dtype=np.float64)
        for fidx, _ in matches:
            row += np.asarray(lca.characterized_inventory[fidx, :].todense(),
                              dtype=np.float64).flatten()
        cats = f"{len(matches)} compartments aggregated"
    else:
        row = np.asarray(lca.characterized_inventory[flow_idx, :].todense(),
                         dtype=np.float64).flatten()
    total = float(row.sum())
    compartment_note = cats  # carry forward for export functions

    print(f"\n{sep}")
    print(f"  SPECIFICATION PER PROCESS  —  '{matched_flow['name']}'")
    print(f"  Compartment              :  {cats}")
    print(f"  Total contribution       :  {total:.6g}  (method units)")
    if lca.score:
        print(f"  % of total LCIA score    :  {100.0*total/lca.score:.2f}%")
    print(sep)
    print()
    print(f"  Calculation:  characterized_inventory[flow_row, :]  =  CF × B × s")
    print(f"  where CF = characterisation factor, B = biosphere matrix,")
    print(f"        s  = supply vector (A⁻¹ · f), summed over all upstream processes")
    print()

    rev_act = {v: k for k, v in lca.activity_dict.items()}
    top_idx = np.argsort(np.abs(row))[::-1]

    print(f"  {'Rank':>4}  {'Contribution':>14}  {'% of flow':>10}  "
          f"{'% of score':>10}  Process  [location]")
    print(f"  {'─'*4}  {'─'*14}  {'─'*10}  {'─'*10}  {'─'*55}")

    results = []
    cumulative_flow = 0.0
    for rank, i in enumerate(top_idx, 1):
        score = float(row[int(i)])
        if score == 0:
            break
        if rank > limit:
            break
        key = rev_act.get(int(i))
        if not key:
            continue
        try:
            act = _bw2data.get_activity(key)
        except Exception:
            continue
        pct_flow  = 100.0 * score / total if total else 0.0
        pct_score = 100.0 * score / lca.score if lca.score else 0.0
        cumulative_flow += pct_flow
        print(f"  {rank:>4}  {score:>14.6g}  {pct_flow:>9.2f}%  "
              f"{pct_score:>9.2f}%  {act['name']}  [{act.get('location','')}]")
        results.append((rank, score, pct_flow, pct_score,
                        act["name"], act.get("location", "")))

    print(f"\n  Cumulative % of flow total for {len(results)} processes: "
          f"{cumulative_flow:.1f}%")
    print()
    return results, matched_flow, total, compartment_note


# ── CSV / Markdown export for specification per process ───────────────────────

def export_spec_csv(activity, spec_results, matched_flow, total_score, lca,
                    compartment_note=""):
    safe_name = (activity["name"].replace("/", "_")
                                  .replace(",", "")
                                  .replace(" ", "_"))[:60]
    flow_safe = (matched_flow["name"].replace("/", "_")
                                      .replace(",", "")
                                      .replace(" ", "_"))[:40]
    path = os.path.join(OUT_DIR, f"{safe_name}_spec_per_process_{flow_safe}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Metadata header rows (SimaPro-style)
        w.writerow(["Activity", activity["name"], activity.get("location", "")])
        w.writerow(["Elementary flow", matched_flow["name"], compartment_note])
        w.writerow(["Flow total contribution", total_score,
                    f"{100.0*total_score/lca.score:.4f}% of LCIA score" if lca.score else ""])
        w.writerow([])
        w.writerow(["rank", "contribution", "pct_of_flow", "pct_of_score",
                    "process_name", "location"])
        for row in spec_results:
            w.writerow(row)
    print(f"  [saved] {path}")


def export_spec_markdown(activity, spec_results, matched_flow, total_score,
                         lca, method, compartment_note=""):
    import datetime
    safe_name = (activity["name"].replace("/", "_")
                                  .replace(",", "")
                                  .replace(" ", "_"))[:60]
    flow_safe = (matched_flow["name"].replace("/", "_")
                                      .replace(",", "")
                                      .replace(" ", "_"))[:40]
    path = os.path.join(OUT_DIR,
           f"{safe_name}_spec_per_process_{flow_safe}_report.md")
    cats = compartment_note or " > ".join(matched_flow.get("categories", []))
    lines = []
    def ln(s=""): lines.append(s)
    ln("# Specification per Process")
    ln()
    ln(f"**Generated:** {datetime.date.today()}  ")
    ln(f"**Activity:** {activity['name']} [{activity.get('location','')}]  ")
    ln(f"**Elementary flow:** {matched_flow['name']}  ")
    ln(f"**Compartment:** {cats}  ")
    ln(f"**Method:** {method}  ")
    ln(f"**Flow total contribution:** `{total_score:.6g}`  ")
    if lca.score:
        ln(f"**% of total LCIA score:** `{100.0*total_score/lca.score:.2f}%`  ")
    ln()
    ln("> **Methodology:** `characterized_inventory[flow_row, :]` = CF × B × s  ")
    ln("> Each column is the characterized contribution of one upstream process  ")
    ln("> to this specific elementary flow.")
    ln()
    ln("| Rank | Contribution | % of Flow | % of Score | Process | Location |")
    ln("|---|---|---|---|---|---|")
    for rank, score, pct_flow, pct_score, name, loc in spec_results:
        ln(f"| {rank} | {score:.6g} | {pct_flow:.2f}% | {pct_score:.2f}% "
           f"| {name} | {loc} |")
    ln()
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [saved] {path}")


# ── Biosphere flow discovery ─────────────────────────────────────────────────

def discover_biosphere_flows(name_pattern, lca=None):
    """
    Search the full biosphere3 database for flows whose name contains
    `name_pattern` (case-insensitive).  Also marks which flows are present
    in the current LCA biosphere matrix (and therefore can be used in
    specification_per_process).

    Use this to find the exact SPEC_FLOW_NAME and SPEC_FLOW_COMPARTMENT values.
    """
    sep = "═" * 80
    print(f"\n{sep}")
    print(f"  BIOSPHERE FLOW DISCOVERY  —  pattern: '{name_pattern}'")
    print(sep)

    bio_db = _bw2data.Database("biosphere3")
    in_matrix = set()
    if lca:
        for key in lca.biosphere_dict.keys():
            in_matrix.add(key)

    found = []
    for flow in bio_db:
        if name_pattern.lower() in flow["name"].lower():
            key = (flow["database"], flow["code"])
            found.append((flow["name"],
                          " > ".join(flow.get("categories", [])),
                          flow.get("unit", ""),
                          key in in_matrix,
                          flow["code"]))

    if not found:
        print(f"  No flows found matching '{name_pattern}' in biosphere3.")
        print(sep)
        return

    found.sort(key=lambda x: (x[0].lower(), x[1]))
    in_matrix_count = sum(1 for f in found if f[3])
    print(f"  Found {len(found)} flows  ({in_matrix_count} present in current LCA matrix)")
    print()
    print(f"  {'In matrix':^9}  {'Name':<40}  {'Compartment (categories)'}")
    print(f"  {'─'*9}  {'─'*40}  {'─'*45}")
    for name, cats, unit, in_mat, code in found:
        flag = "  ✓ YES  " if in_mat else "   no   "
        print(f"  {flag}  {name:<40}  {cats!r}")
    print()
    print(f"  → Copy the exact name and a substring of the compartment into:")
    print(f"      SPEC_FLOW_NAME        = \"<exact name>\"")
    print(f"      SPEC_FLOW_COMPARTMENT = \"<compartment substring>\"  # or None")
    print(sep)
    print()

    # ── CSV export ────────────────────────────────────────────────────────────
    pattern_safe = (name_pattern.replace("/", "_")
                                .replace(",", "")
                                .replace(" ", "_"))[:50]
    csv_path = os.path.join(OUT_DIR, f"discover_flows_{pattern_safe}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["in_lca_matrix", "name", "compartment", "unit", "code"])
        for name, cats, unit, in_mat, code in found:
            w.writerow(["YES" if in_mat else "no", name, cats, unit, code])
    print(f"  [saved] {csv_path}")


# ── method helper ─────────────────────────────────────────────────────────────

def resolve_method(method_tuple):
    """Find the exact method key; print candidates if not found exactly."""
    if method_tuple in bw_methods:
        return method_tuple
    # fuzzy: match on first element (method family name)
    candidates = [m for m in bw_methods if m[0] == method_tuple[0]]
    if candidates:
        print(f"\n[!] Method {method_tuple} not found exactly.")
        print(f"    Available methods in family '{method_tuple[0]}':")
        for m in sorted(candidates):
            print(f"      {m}")
        print(f"\n    Update LCIA_METHOD in this script to one of the above.")
    else:
        print(f"\n[!] No methods found for family '{method_tuple[0]}'.")
        print(f"    All available method families:")
        families = sorted(set(m[0] for m in bw_methods))
        for f in families:
            print(f"      {f}")
    return None

# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup()

    act = find_activity(ACTIVITY_NAME, ACTIVITY_LOCATION)
    groups = print_exchanges(act)

    lca = None
    if LCIA_METHOD:
        resolved = resolve_method(LCIA_METHOD)
        if resolved:
            lca = run_lcia(act, resolved, AMOUNT)
        else:
            print("\n[!] Skipping LCIA — fix LCIA_METHOD above and re-run.")

    # ── flow discovery mode ───────────────────────────────────────────────────
    if SPEC_FLOW_NAME and SPEC_FLOW_DISCOVER:
        discover_biosphere_flows(SPEC_FLOW_NAME, lca=lca)

    spec = None
    if lca and SPEC_FLOW_NAME and not SPEC_FLOW_DISCOVER:
        spec = specification_per_process(lca, SPEC_FLOW_NAME,
                                         SPEC_FLOW_COMPARTMENT, limit=200)

    print("── EXPORT " + "─" * 70)
    export_csvs(act, groups, lca, LCIA_METHOD)
    export_markdown(act, groups, lca, LCIA_METHOD)
    if spec is not None:
        spec_results, matched_flow, total_score, compartment_note = spec
        export_spec_csv(act, spec_results, matched_flow, total_score, lca,
                        compartment_note)
        export_spec_markdown(act, spec_results, matched_flow, total_score,
                             lca, LCIA_METHOD, compartment_note)
    print("\nDone.")

