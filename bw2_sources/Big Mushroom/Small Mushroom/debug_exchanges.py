"""
Export script: dump all ecoinvent activities + their technosphere exchanges to CSV.
Run from the project root (where brightway_databases/ lives):
    python bw2_sources/Big\ Mushroom/Small\ Mushroom/debug_exchanges.py

Outputs two files in the same directory as this script:
    ecoinvent_activities.csv   — one row per activity (metadata only)
    ecoinvent_exchanges.csv    — one row per technosphere exchange (with parent activity info)
"""
import csv
import os
import sys

# --- point brightway2 at the correct data dir ---
os.environ["BRIGHTWAY2_DIR"] = "./brightway_databases/brightway2_ecoinvent"

from bw2data import projects, databases, Database

# --- activate the ecoinvent project ---
projects.set_current("ecoinvent")
print("Current project :", projects.current)
print("Available DBs   :", list(databases))

# --- load the ecoinvent database ---
DB_NAME = "ecoinvent"
if DB_NAME not in databases:
    print(f"ERROR: '{DB_NAME}' not found. Available: {list(databases)}")
    sys.exit(1)

db = Database(DB_NAME)
print(f"Database '{DB_NAME}' has {len(db)} activities — exporting…\n")

# --- output paths ---
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
ACTIVITIES_CSV = os.path.join(OUT_DIR, "ecoinvent_activities.csv")
EXCHANGES_CSV  = os.path.join(OUT_DIR, "ecoinvent_exchanges.csv")

# --- write activities ---
ACT_FIELDS = ["name", "reference product", "location", "unit", "database", "code"]

with open(ACTIVITIES_CSV, "w", newline="", encoding="utf-8") as act_f, \
     open(EXCHANGES_CSV,  "w", newline="", encoding="utf-8") as exc_f:

    act_writer = csv.DictWriter(act_f, fieldnames=ACT_FIELDS)
    act_writer.writeheader()

    exc_fields = [
        "parent_name", "parent_location", "parent_code",
        "exchange_amount", "exchange_name", "exchange_location",
        "exchange_unit", "exchange_database", "exchange_code",
    ]
    exc_writer = csv.DictWriter(exc_f, fieldnames=exc_fields)
    exc_writer.writeheader()

    for i, act in enumerate(db, start=1):
        # activity row
        act_writer.writerow({
            "name":               act.get("name", ""),
            "reference product":  act.get("reference product", ""),
            "location":           act.get("location", ""),
            "unit":               act.get("unit", ""),
            "database":           act.get("database", ""),
            "code":               act.get("code", ""),
        })

        # exchange rows
        for exc in act.exchanges():
            if exc["type"] != "technosphere":
                continue
            src = exc.input
            exc_writer.writerow({
                "parent_name":       act.get("name", ""),
                "parent_location":   act.get("location", ""),
                "parent_code":       act.get("code", ""),
                "exchange_amount":   exc.get("amount", ""),
                "exchange_name":     src.get("name", ""),
                "exchange_location": src.get("location", ""),
                "exchange_unit":     src.get("unit", ""),
                "exchange_database": src.get("database", ""),
                "exchange_code":     src.get("code", ""),
            })

        if i % 1000 == 0:
            print(f"  {i:,} / {len(db):,} activities processed…")

print(f"\nDone.")
print(f"  Activities : {ACTIVITIES_CSV}")
print(f"  Exchanges  : {EXCHANGES_CSV}")
