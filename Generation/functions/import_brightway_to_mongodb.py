"""Import Brightway activity metadata into MongoDB (not the database files).

Run from the backend root after configuring MONGODB_URI and either Blob access or --source:
    python -m Generation.functions.import_brightway_to_mongodb
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from pymongo import ASCENDING, ReplaceOne

from Generation.functions.azure_brightway_store import ensure_ecoinvent_database
from Generation.functions.lca_activity_store import (
    COLLECTION_NAME,
    get_lca_activities_collection,
    get_mongodb_database_name,
)

logger = logging.getLogger("brightway_metadata_import")
REPO_ROOT = Path(__file__).resolve().parents[2]


def import_activity_metadata(source: Path | None, project: str, database_name: str):
    source = Path(source).resolve() if source else ensure_ecoinvent_database()
    marker = source / ".azure_blob_snapshot.json"
    source_info = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {}
    blob_version = source_info.get("version_id") or source_info.get("etag")
    source_version = (
        f"{source_info.get('container', '')}/{source_info.get('blob', '')}@{blob_version}"
        if blob_version else None
    )
    snapshot_id = hashlib.sha256(source_version.encode()).hexdigest() if source_version else uuid.uuid4().hex

    os.environ["BRIGHTWAY2_DIR"] = str(source)
    from bw2data import Database, projects

    projects.set_current(project)
    database = Database(database_name)
    collection = get_lca_activities_collection()
    collection.create_index(
        [("snapshot_id", ASCENDING), ("name_lower", ASCENDING), ("location", ASCENDING), ("code", ASCENDING)],
        partialFilterExpression={"kind": "activity"},
    )

    operations = []
    count = 0
    for activity in database:
        name = str(activity.get("name") or "").strip()
        code = str(activity.get("code") or "").strip()
        if not name or not code:
            raise ValueError("Encountered a Brightway activity without a name or stable code")

        database_id = str(activity.get("database") or database_name)
        identity = hashlib.sha256(f"{database_id}\0{code}".encode("utf-8")).hexdigest()
        document = {
            "_id": f"{snapshot_id}:activity:{identity}",
            "snapshot_id": snapshot_id,
            "kind": "activity",
            "database": database_id,
            "code": code,
            "name": name,
            "name_lower": name.lower(),
            "reference_product": str(activity.get("reference product") or ""),
            "unit": str(activity.get("unit") or ""),
            "location": str(activity.get("location") or ""),
        }
        operations.append(ReplaceOne({"_id": document["_id"]}, document, upsert=True))
        if len(operations) >= 1_000:
            collection.bulk_write(operations, ordered=False)
            count += len(operations)
            operations.clear()
            logger.info("Imported activity metadata: %d", count)

    if operations:
        collection.bulk_write(operations, ordered=False)
        count += len(operations)
    if count == 0:
        raise ValueError(f"Brightway database {database_name!r} contains no activities")

    manifest = {
        "_id": f"{snapshot_id}:manifest",
        "snapshot_id": snapshot_id,
        "kind": "activity_snapshot",
        "database": database_name,
        "project": project,
        "source_version": source_version,
        "activity_count": count,
        "created_at": datetime.now(timezone.utc),
    }
    collection.replace_one({"_id": manifest["_id"]}, manifest, upsert=True)
    collection.replace_one(
        {"_id": "active_activity_snapshot"},
        {
            "_id": "active_activity_snapshot",
            "kind": "activity_active",
            "snapshot_id": snapshot_id,
            "activity_count": count,
            "source_version": source_version,
        },
        upsert=True,
    )
    logger.info("Activated %d activity records in %s.%s", count, get_mongodb_database_name(), COLLECTION_NAME)
    return snapshot_id


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Existing local Brightway directory; defaults to Blob cache")
    parser.add_argument("--project", default="ecoinvent", help="Brightway project name")
    parser.add_argument("--database", default="ecoinvent", help="Brightway database name")
    args = parser.parse_args()
    if not os.getenv("MONGODB_URI"):
        parser.error("Set MONGODB_URI before importing activity metadata")
    import_activity_metadata(args.source, args.project, args.database)


if __name__ == "__main__":
    main()
