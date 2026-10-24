"""Import a local Brightway snapshot and activity index into MongoDB.

Run from the backend root after installing requirements and setting MONGODB_URI:
    python -m Generation.functions.import_brightway_to_mongodb
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from pymongo import ASCENDING

from Generation.functions.lca_activity_store import (
    COLLECTION_NAME,
    get_mongodb_database_name,
    get_lca_activities_collection,
)

logger = logging.getLogger("brightway_mongodb_import")
REPO_ROOT = Path(__file__).resolve().parents[2]
CHUNK_SIZE = 4 * 1024 * 1024
MAX_BATCH_BYTES = 12 * 1024 * 1024


def _archive_directory(source: Path, archive_path: Path) -> int:
    files = []
    for root, dirs, names in os.walk(source, followlinks=False):
        dirs.sort()
        names.sort()
        root_path = Path(root)
        for name in dirs:
            if (root_path / name).is_symlink():
                raise ValueError(f"Brightway snapshot contains an unsupported symlink: {root_path / name}")
        for name in names:
            file_path = root_path / name
            if file_path.is_symlink():
                raise ValueError(f"Brightway snapshot contains an unsupported symlink: {file_path}")
            if not file_path.is_file():
                raise ValueError(f"Brightway snapshot contains a non-regular file: {file_path}")
            files.append(file_path)

    if not files:
        raise ValueError(f"No files found in Brightway snapshot: {source}")

    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
        allowZip64=True,
    ) as archive:
        for file_path in files:
            archive.write(file_path, file_path.relative_to(source).as_posix())
    return len(files)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _store_archive(collection, snapshot_id: str, archive_path: Path):
    batch = []
    batch_bytes = 0
    chunk_count = 0

    def flush():
        nonlocal batch, batch_bytes
        if batch:
            collection.insert_many(batch, ordered=True)
            batch = []
            batch_bytes = 0

    with archive_path.open("rb") as stream:
        for data in iter(lambda: stream.read(CHUNK_SIZE), b""):
            batch.append({
                "_id": f"{snapshot_id}:chunk:{chunk_count:08d}",
                "snapshot_id": snapshot_id,
                "kind": "archive_chunk",
                "chunk_index": chunk_count,
                "data": data,
            })
            batch_bytes += len(data)
            chunk_count += 1
            if batch_bytes >= MAX_BATCH_BYTES:
                flush()
    flush()
    return chunk_count


def _store_activity_metadata(collection, snapshot_id: str, source: Path, project: str, database_name: str):
    # Set the project data path before importing bw2data.
    os.environ["BRIGHTWAY2_DIR"] = str(source)
    from bw2data import Database, projects

    projects.set_current(project)
    database = Database(database_name)
    batch = []
    count = 0

    for activity in database:
        name = str(activity.get("name") or "").strip()
        code = str(activity.get("code") or "").strip()
        if not name or not code:
            raise ValueError("Encountered a Brightway activity without a name or stable code")

        database_id = str(activity.get("database") or database_name)
        identity = hashlib.sha256(f"{database_id}\0{code}".encode("utf-8")).hexdigest()
        batch.append({
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
        })
        if len(batch) >= 1_000:
            collection.insert_many(batch, ordered=True)
            count += len(batch)
            batch.clear()
            logger.info("Imported activity metadata: %d", count)

    if batch:
        collection.insert_many(batch, ordered=True)
        count += len(batch)
    if count == 0:
        raise ValueError(f"Brightway database {database_name!r} contains no activities")
    return count


def import_snapshot(source: Path, project: str, database_name: str):
    source = source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Brightway database directory not found: {source}")

    collection = get_lca_activities_collection()
    collection.create_index(
        [("snapshot_id", ASCENDING), ("name_lower", ASCENDING)],
        partialFilterExpression={"kind": "activity"},
    )
    collection.create_index(
        [("snapshot_id", ASCENDING), ("chunk_index", ASCENDING)],
        partialFilterExpression={"kind": "archive_chunk"},
    )
    old_active = collection.find_one({"_id": "active_snapshot", "kind": "active"})

    archive_path = None
    snapshot_id = None
    activated = False
    try:
        with tempfile.NamedTemporaryFile(prefix="brightway-export-", suffix=".zip", delete=False) as temp:
            archive_path = Path(temp.name)
        file_count = _archive_directory(source, archive_path)
        archive_sha256 = _sha256_file(archive_path)
        snapshot_id = archive_sha256

        if old_active and old_active.get("snapshot_id") == snapshot_id:
            logger.info("This Brightway snapshot is already active; import skipped")
            return snapshot_id

        # A prior interrupted run with the same archive may have left staged records.
        collection.delete_many({"snapshot_id": snapshot_id})
        archive_chunks = _store_archive(collection, snapshot_id, archive_path)
        activity_count = _store_activity_metadata(collection, snapshot_id, source, project, database_name)
        manifest = {
            "_id": f"{snapshot_id}:manifest",
            "snapshot_id": snapshot_id,
            "kind": "snapshot",
            "source": str(source),
            "archive_sha256": archive_sha256,
            "archive_size": archive_path.stat().st_size,
            "archive_chunks": archive_chunks,
            "file_count": file_count,
            "activity_count": activity_count,
            "database": database_name,
            "project": project,
            "created_at": datetime.now(timezone.utc),
        }
        collection.insert_one(manifest)
        collection.replace_one(
            {"_id": "active_snapshot"},
            {"_id": "active_snapshot", "kind": "active", "snapshot_id": snapshot_id},
            upsert=True,
        )
        activated = True
        logger.info(
            "Activated snapshot %s: %d files, %d activities, %d archive chunks",
            snapshot_id, file_count, activity_count, archive_chunks,
        )

        if old_active and old_active.get("snapshot_id") != snapshot_id:
            logger.info("Previous snapshot retained so in-flight restores can finish")
        return snapshot_id
    except Exception:
        if snapshot_id and not activated:
            try:
                collection.delete_many({"snapshot_id": snapshot_id})
            except Exception:
                logger.exception("Could not remove incomplete Brightway snapshot %s", snapshot_id)
        raise
    finally:
        if archive_path and archive_path.exists():
            archive_path.unlink()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=REPO_ROOT / "brightway_databases" / "brightway2_ecoinvent",
        help="Existing, initialized Brightway data directory",
    )
    parser.add_argument("--project", default="ecoinvent", help="Brightway project name")
    parser.add_argument("--database", default="ecoinvent", help="Brightway database name")
    args = parser.parse_args()

    if not os.getenv("MONGODB_URI"):
        parser.error("Set MONGODB_URI before importing the Brightway snapshot")
    logger.info("Writing to MongoDB %s.%s", get_mongodb_database_name(), COLLECTION_NAME)
    import_snapshot(args.source, args.project, args.database)


if __name__ == "__main__":
    main()
