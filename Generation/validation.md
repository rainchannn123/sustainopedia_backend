# Mass Flow Balance — What Was Changed and Why

## The Big Idea

Imagine you are making copper wire. You start with raw copper ore from a mine, then melt and refine it, then draw it into wire. At every step, some material is lost (as waste or slag), but the total mass of material flowing through the process should always make sense — you cannot end up with *more* copper wire than the copper you started with.

This feature makes the system **check that the mass of material entering each step is consistent with the mass that left the previous step**, following the rules of a standard called ISO 14040/44 (an international rule book for doing Life Cycle Assessment properly). If there is a big mismatch, the system raises a warning so the user knows the numbers might be wrong.

---

## File 1 — `Generation/functions/after_bw2.py` (5 changes)

This file is responsible for **reading the LCA result** (after the calculation engine finishes) and turning it into a readable report.

### Change 1 — Added warning thresholds (2 new number constants)

```
_MASS_FLOW_TOLERANCE_WARN = 10.0   ← warn if mass mismatch is more than 10%
_MASS_FLOW_TOLERANCE_GAP  = 25.0   ← flag as a hard error if more than 25%
```

Think of these like exam score boundaries. A 10% mismatch gets a yellow flag ("something looks a bit off"), and a 25%+ mismatch gets a red flag ("this is clearly wrong").

---

### Change 2 — Added a list of "mass units" and a helper check

```
_MASS_BEARING_UNITS = { "kg", "kilogram", "tonne", "ton", "gram", ... }

def _is_mass_bearing(unit):
    → returns True if the unit is a physical weight (kg, tonne, etc.)
    → returns False for things like "ton kilometer" (transport) or "kWh" (energy)
```

Not every step in the value chain moves physical material. A transport step moves goods but the *activity itself* is measured in distance×weight, not mass. An energy step is measured in kWh. These non-mass steps are **excluded** from the mass balance check. This function is the gatekeeper that decides whether a step should be checked or skipped.

---

### Change 3 — Added the main checker function `_validate_mass_flow_chain`

This is the core logic. It:

1. **Goes through every step** in the value chain (e.g., mining → smelting → refining → fabrication).
2. **Skips** any step that is not a physical mass step (transport, energy, services).
3. For every **pair of consecutive mass steps**, checks:
   > "Does the output mass of step A equal the input mass of step B?"
4. If the mismatch is under 10% → marks that transition as **OK**.
5. If 10–25% mismatch → marks it as a **WARNING**.
6. If over 25% mismatch → marks it as a **GAP** (hard problem).
7. At the end, calculates an **overall material yield** = (final output mass) ÷ (starting input mass). For example, if you put in 25,000 kg of copper ore and get out 1,000 kg of refined copper, the yield is 0.04 (4%).

The function returns a summary dictionary with the status and a list of every transition checked.

---

### Change 4 — Added a loader function `compute_mass_flow_balance`

The checker above needs the value chain data as input. This function:

1. **Finds the matching input file** for the LCA result (e.g., if the result file is `Copper_output.json`, it looks for `Copper.json` in the input templates folder).
2. Loads that file and passes the activity list to the checker.
3. If the file is missing or broken, it safely returns a `"skipped"` result instead of crashing.

This function is called automatically every time an LCA result is processed.

---

### Change 5 (three sub-parts) — Attach the results to the report

**5a — Store mass flow data per process:**
Each individual process in the report now also records `mass_in_kg`, `mass_out_kg`, and `recovery_rate` (looked up from the input template). This lets the frontend display, for example, "this smelting step takes in 1,200 kg and outputs 1,000 kg, recovery = 83.3%".

**5b — Store the overall audit in the metadata:**
After all processes are loaded, the full audit result is stored in `lcia_data["metadata"]["mass_flow_balance"]` so it travels with the data everywhere.

**5c — Show it in the Markdown and JSON reports:**

- In the **Markdown report** (the human-readable text output), a new section appears at the bottom:
  ```
  ### Mass Flow Balance Audit (ISO 14040/44)
  Status: PASS | Overall Material Yield: 0.0400
  - mining → smelting: expected 25000.0 kg / actual 25000.0 kg (discrepancy 0.0%) [OK]
  - smelting → refining: expected 2000.0 kg / actual 1950.0 kg (discrepancy 2.5%) [OK]
  ```
  This section is hidden automatically if there was not enough data to run the check.

- In the **JSON output** (the machine-readable result), a new key `"massFlowBalance"` is added, containing the full audit object. This lets the web frontend read and display the data programmatically.

---

## File 2 — `Generation/functions/non_RAG_methods.py` (4 changes)

This file is responsible for **asking the AI (LLM) to generate the value chain** — basically, it writes the prompt that tells the AI what kind of JSON to produce.

### Change 1 — Import the checker

```python
from .after_bw2 import _validate_mass_flow_chain
```

This line simply makes the checker function available in this file so it can be used later.

---

### Change 2 — Tell the AI about mass flow fields

The AI is now instructed to include three new fields for every activity it generates:

| Field | What it means |
|---|---|
| `mass_in_kg` | How many kg of material enters this step (per 1 tonne of final product) |
| `mass_out_kg` | How many kg of useful material leaves this step |
| `recovery_rate` | The fraction that makes it through (e.g., 0.95 means 95% of material is kept) |

The AI is also given **typical recovery rate ranges** to guide it:
- Ore mining: 0.003–0.05 (only 0.3–5% of ore becomes useful material — most is waste rock)
- Smelting / flotation: 0.80–0.92
- Refining: 0.95–0.99
- Fabrication / assembly: 0.93–0.99

Non-mass steps (transport, energy) are told to use `mass_in_kg = 0`, `mass_out_kg = 0`, `recovery_rate = 1.0`.

The AI is also given the **ISO 14040/44 consistency rule** in plain language: "the output mass of one step must equal the input mass of the next step (within ~10%)."

---

### Change 3 — Updated the example JSON in the prompt

The example the AI learns from now includes the three new fields, so the AI understands the exact format expected:

```json
{
  "activity_name": "hard coal production",
  "unit": "kilogram",
  "stage": "manufacturing",
  "mass_in_kg": 1050.0,
  "mass_out_kg": 1000.0,
  "recovery_rate": 0.952
},
{
  "activity_name": "transport, freight, lorry ...",
  "unit": "ton kilometer",
  "stage": "transport",
  "mass_in_kg": 0,
  "mass_out_kg": 0,
  "recovery_rate": 1.0
}
```

---

### Change 4 — Run the check immediately after the AI generates the value chain

Right after the AI produces the JSON and before saving it to a file, the system now:

1. Runs `_validate_mass_flow_chain` on the AI's output.
2. Saves the audit result inside the JSON file under `"mass_flow_audit"`.
3. Logs the result:
   - If everything is fine → logs "Mass flow audit passed. Overall yield: 0.04"
   - If there is a mismatch → logs a warning with details of which transitions are problematic

This means problems are caught **at generation time** (before the LCA engine even runs), not just after the fact.

---

## Summary in One Sentence

We added a system that tracks how much physical material flows into and out of each step in a product's value chain, checks that those numbers are physically consistent, and reports any mismatches — both before the LCA engine runs and after it finishes.


2026-05-21 01:27:05,039 - INFO - bw2data projects._base_data_dir was 'C:\Users\user\Documents\_sustainopedia\sustainopedia_backend\brightway_databases\brightway2_ecoinvent'; overriding to 'C:\Users\user\Documents\_sustainopedia\sustainopedia_backend\brightway_databases\brightway2_ecoinvent'
2026-05-21 01:27:05,056 - INFO - BW2 reference-product index built: 0 activities
2026-05-21 01:27:05,056 - INFO - {'activity_name': 'rare earth element mine operation and beneficiation, ion adsorption clays', 'stage': 'raw material extraction', 'unit': 'kilogram', 'inferred_reference_product': '', 'bw2_close_matches': [], 'retrieval_candidates_for_stage': ['rare earth element mine operation and beneficiation, ion adsorption clays', 'mine construction, bauxite', 'mine construction, iron', 'processing of nickel-rich materials', 'electric motor production, vehicle (electric powertrain)', 'electric motor production, vehicle', 'market for non-Fe-Co-metals, from Li-ion battery, hydrometallurgical processing', 'market for non-Fe-Co-metals, from used Li-ion battery, pyrometallurgical processing', 'mine construction, open cast, ilmenite from hard-rock ore', 'treatment of used powertrain for electric passenger car, manual dismantling']}
2026-05-21 01:27:05,056 - INFO - {'activity_name': 'processing of nickel-rich materials', 'stage': 'material processing', 'unit': 'kilogram', 'inferred_reference_product': '', 'bw2_close_matches': [], 'retrieval_candidates_for_stage': ['electric motor production, vehicle (electric powertrain)', 'processing of nickel-rich materials', 'electric motor production, vehicle', 'market for non-Fe-Co-metals, from used Li-ion battery, pyrometallurgical processing', 'market for non-Fe-Co-metals, from Li-ion battery, hydrometallurgical processing', 'powertrain production, for electric passenger car', 'treatment of non-Fe-Co-metals, from used Li-ion battery, hydrometallurgical processing', 'treatment of non-Fe-Co-metals, from used Li-ion battery, pyrometallurgical processing', 'inverter production, for electric passenger car', 'treatment of used powertrain for electric passenger car, manual dismantling']}
2026-05-21 01:27:05,056 - INFO - {'activity_name': 'inverter production, for electric passenger car', 'stage': 'manufacturing/production', 'unit': 'unit', 'inferred_reference_product': '', 'bw2_close_matches': [], 'retrieval_candidates_for_stage': ['electric motor production, vehicle (electric powertrain)', 'electric motor production, vehicle', 'inverter production, for electric passenger car', 'passenger car production, electric, without battery', 'powertrain production, for electric passenger car', 'power distribution unit production, for electric passenger car', 'powertrain production, for electric scooter', 'electric bicycle production', 'converter production, for electric passenger car', 'road vehicle factory construction']}
2026-05-21 01:27:05,056 - INFO - {'activity_name': 'power distribution unit production, for electric passenger car', 'stage': 'manufacturing/production', 'unit': 'unit', 'inferred_reference_product': '', 'bw2_close_matches': [], 'retrieval_candidates_for_stage': ['electric motor production, vehicle (electric powertrain)', 'electric motor production, vehicle', 'inverter production, for electric passenger car', 'passenger car production, electric, without battery', 'powertrain production, for electric passenger car', 'power distribution unit production, for electric passenger car', 'powertrain production, for electric scooter', 'electric bicycle production', 'converter production, for electric passenger car', 'road vehicle factory construction']}
2026-05-21 01:27:05,056 - INFO - {'activity_name': 'converter production, for electric passenger car', 'stage': 'manufacturing/production', 'unit': 'unit', 'inferred_reference_product': '', 'bw2_close_matches': [], 'retrieval_candidates_for_stage': ['electric motor production, vehicle (electric powertrain)', 'electric motor production, vehicle', 'inverter production, for electric passenger car', 'passenger car production, electric, without battery', 'powertrain production, for electric passenger car', 'power distribution unit production, for electric passenger car', 'powertrain production, for electric scooter', 'electric bicycle production', 'converter production, for electric passenger car', 'road vehicle factory construction']}
2026-05-21 01:27:05,056 - INFO - {'activity_name': 'electric motor production, vehicle (electric powertrain)', 'stage': 'manufacturing/production', 'unit': 'unit', 'inferred_reference_product': '', 'bw2_close_matches': [], 'retrieval_candidates_for_stage': ['electric motor production, vehicle (electric powertrain)', 'electric motor production, vehicle', 'inverter production, for electric passenger car', 'passenger car production, electric, without battery', 'powertrain production, for electric passenger car', 'power distribution unit production, for electric passenger car', 'powertrain production, for electric scooter', 'electric bicycle production', 'converter production, for electric passenger car', 'road vehicle factory construction']}
2026-05-21 01:27:05,056 - INFO - {'activity_name': 'powertrain production, for electric passenger car', 'stage': 'manufacturing/production', 'unit': 'unit', 'inferred_reference_product': '', 'bw2_close_matches': [], 'retrieval_candidates_for_stage': ['electric motor production, vehicle (electric powertrain)', 'electric motor production, vehicle', 'inverter production, for electric passenger car', 'passenger car production, electric, without battery', 'powertrain production, for electric passenger car', 'power distribution unit production, for electric passenger car', 'powertrain production, for electric scooter', 'electric bicycle production', 'converter production, for electric passenger car', 'road vehicle factory construction']}
2026-05-21 01:27:05,056 - INFO - {'activity_name': 'passenger car production, electric, without battery', 'stage': 'manufacturing/production', 'unit': 'kilogram', 'inferred_reference_product': '', 'bw2_close_matches': [], 'retrieval_candidates_for_stage': ['electric motor production, vehicle (electric powertrain)', 'electric motor production, vehicle', 'inverter production, for electric passenger car', 'passenger car production, electric, without battery', 'powertrain production, for electric passenger car', 'power distribution unit production, for electric passenger car', 'powertrain production, for electric scooter', 'electric bicycle production', 'converter production, for electric passenger car', 'road vehicle factory construction']}
2026-05-21 01:27:05,057 - INFO - {'activity_name': 'transport, freight, light commercial vehicle', 'stage': 'distribution/transport', 'unit': 'ton kilometer', 'inferred_reference_product': '', 'bw2_close_matches': [], 'retrieval_candidates_for_stage': ['power distribution unit production, for electric passenger car', 'transport, freight, light commercial vehicle', 'market for transport, passenger car, electric', 'transport, passenger car, electric', 'transport, passenger, electric bicycle, electricity from renewable energy products', 'transport, passenger, electric bicycle', 'powertrain production, for electric passenger car', 'market for transport, passenger, electric bicycle', 'transport, passenger, electric scooter', 'market for transport, passenger, electric bicycle, electricity from renewable energy products']}
2026-05-21 01:27:05,057 - INFO - {'activity_name': 'market for transport, passenger car, electric', 'stage': 'distribution/transport', 'unit': 'unit', 'inferred_reference_product': '', 'bw2_close_matches': [], 'retrieval_candidates_for_stage': ['power distribution unit production, for electric passenger car', 'transport, freight, light commercial vehicle', 'market for transport, passenger car, electric', 'transport, passenger car, electric', 'transport, passenger, electric bicycle, electricity from renewable energy products', 'transport, passenger, electric bicycle', 'powertrain production, for electric passenger car', 'market for transport, passenger, electric bicycle', 'transport, passenger, electric scooter', 'market for transport, passenger, electric bicycle, electricity from renewable energy products']}
2026-05-21 01:27:05,057 - INFO - {'activity_name': 'charger production, for electric passenger car', 'stage': 'use phase', 'unit': 'unit', 'inferred_reference_product': '', 'bw2_close_matches': [], 'retrieval_candidates_for_stage': ['market for maintenance, passenger car, electric, without battery', 'maintenance, passenger car, electric, without battery', 'energy use and operation emissions, electric bicycle, electricity from renewable energy products', 'electric motor production, vehicle (electric powertrain)', 'electric motor production, vehicle', 'energy use and operation emissions, electric bicycle', 'converter production, for electric passenger car', 'powertrain production, for electric passenger car', 'charger production, for electric passenger car', 'treatment of used powertrain for electric passenger car, manual dismantling']}
2026-05-21 01:27:05,057 - INFO - {'activity_name': 'maintenance, passenger car, electric, without battery', 'stage': 'use phase', 'unit': 'unit', 'inferred_reference_product': '', 'bw2_close_matches': [], 'retrieval_candidates_for_stage': ['market for maintenance, passenger car, electric, without battery', 'maintenance, passenger car, electric, without battery', 'energy use and operation emissions, electric bicycle, electricity from renewable energy products', 'electric motor production, vehicle (electric powertrain)', 'electric motor production, vehicle', 'energy use and operation emissions, electric bicycle', 'converter production, for electric passenger car', 'powertrain production, for electric passenger car', 'charger production, for electric passenger car', 'treatment of used powertrain for electric passenger car, manual dismantling']}
2026-05-21 01:27:05,057 - INFO - {'activity_name': 'manual dismantling of used electric passenger car', 'stage': 'end-of-life', 'unit': 'unit', 'inferred_reference_product': '', 'bw2_close_matches': [], 'retrieval_candidates_for_stage': ['treatment of waste electric wiring, collection for final disposal', 'treatment of waste electric and electronic equipment, shredding', 'treatment of used industrial electronic device, manual dismantling', 'treatment of used glider, electric scooter, shredding', 'treatment of used powertrain for electric passenger car, manual dismantling', 'treatment of used powertrain, electric scooter, manual dismantling', 'treatment of used glider, passenger car, shredding', 'manual dismantling of used electric passenger car', 'market for manual dismantling of used electric passenger car', 'market for used powertrain from electric passenger car, manual dismantling']}
2026-05-21 01:27:05,057 - INFO - {'activity_name': 'treatment of used glider, passenger car, shredding', 'stage': 'end-of-life', 'unit': 'kilogram', 'inferred_reference_product': '', 'bw2_close_matches': [], 'retrieval_candidates_for_stage': ['treatment of waste electric wiring, collection for final disposal', 'treatment of waste electric and electronic equipment, shredding', 'treatment of used industrial electronic device, manual dismantling', 'treatment of used glider, electric scooter, shredding', 'treatment of used powertrain for electric passenger car, manual dismantling', 'treatment of used powertrain, electric scooter, manual dismantling', 'treatment of used glider, passenger car, shredding', 'manual dismantling of used electric passenger car', 'market for manual dismantling of used electric passenger car', 'market for used powertrain from electric passenger car, manual dismantling']}
2026-05-21 01:27:05,057 - INFO - {'activity_name': 'treatment of used powertrain for electric passenger car, manual dismantling', 'stage': 'end-of-life', 'unit': 'kilogram', 'inferred_reference_product': '', 'bw2_close_matches': [], 'retrieval_candidates_for_stage': ['treatment of waste electric wiring, collection for final disposal', 'treatment of waste electric and electronic equipment, shredding', 'treatment of used industrial electronic device, manual dismantling', 'treatment of used glider, electric scooter, shredding', 'treatment of used powertrain for electric passenger car, manual dismantling', 'treatment of used powertrain, electric scooter, manual dismantling', 'treatment of used glider, passenger car, shredding', 'manual dismantling of used electric passenger car', 'market for manual dismantling of used electric passenger car', 'market for used powertrain from electric passenger car, manual dismantling']}
2026-05-21 01:27:05,057 - INFO - {'activity_name': 'treatment of waste electric wiring, collection for final disposal', 'stage': 'end-of-life', 'unit': 'kilogram', 'inferred_reference_product': '', 'bw2_close_matches': [], 'retrieval_candidates_for_stage': ['treatment of waste electric wiring, collection for final disposal', 'treatment of waste electric and electronic equipment, shredding', 'treatment of used industrial electronic device, manual dismantling', 'treatment of used glider, electric scooter, shredding', 'treatment of used powertrain for electric passenger car, manual dismantling', 'treatment of used powertrain, electric scooter, manual dismantling', 'treatment of used glider, passenger car, shredding', 'manual dismantling of used electric passenger car', 'market for manual dismantling of used electric passenger car', 'market for used powertrain from electric passenger car, manual dismantling']}
2026-05-21 01:27:05,057 - INFO - {'activity_name': 'treatment of waste electric and electronic equipment, shredding', 'stage': 'end-of-life', 'unit': 'kilogram', 'inferred_reference_product': '', 'bw2_close_matches': [], 'retrieval_candidates_for_stage': ['treatment of waste electric wiring, collection for final disposal', 'treatment of waste electric and electronic equipment, shredding', 'treatment of used industrial electronic device, manual dismantling', 'treatment of used glider, electric scooter, shredding', 'treatment of used powertrain for electric passenger car, manual dismantling', 'treatment of used powertrain, electric scooter, manual dismantling', 'treatment of used glider, passenger car, shredding', 'manual dismantling of used electric passenger car', 'market for manual dismantling of used electric passenger car', 'market for used powertrain from electric passenger car, manual dismantling']}