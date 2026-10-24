"""MongoDB storage and restore helpers for the Brightway ecoinvent snapshot."""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import shutil
import stat
import tempfile
import threading
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from pymongo import MongoClient
from pymongo.uri_parser import parse_uri

logger = logging.getLogger(__name__)

COLLECTION_NAME = "lca_activities"
_client = None
_database_name = None
_client_lock = threading.Lock()
_restore_lock = threading.Lock()


def get_mongodb_database_name(uri=None):
    uri = uri or os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI is required for Brightway activity storage")
    database_name = parse_uri(uri).get("database")
    if not database_name:
        raise RuntimeError("MONGODB_URI must include the target database name in its path")
    return database_name


def get_lca_activities_collection():
    """Return the configured collection, connecting lazily and failing clearly."""
    global _client, _database_name
    if _client is None:
        with _client_lock:
            if _client is None:
                uri = os.getenv("MONGODB_URI")
                database_name = get_mongodb_database_name(uri)
                client = MongoClient(uri, serverSelectionTimeoutMS=10_000)
                try:
                    client.admin.command("ping")
                except Exception:
                    client.close()
                    raise
                _client = client
                _database_name = database_name
    return _client[_database_name][COLLECTION_NAME]


def get_active_snapshot(collection=None):
    if collection is None:
        collection = get_lca_activities_collection()
    active = collection.find_one({"_id": "active_snapshot", "kind": "active"})
    if not active or not active.get("snapshot_id"):
        raise RuntimeError("No Brightway snapshot is active in sustainopedia.lca_activities")
    manifest = collection.find_one({
        "_id": f"{active['snapshot_id']}:manifest",
        "kind": "snapshot",
    })
    if not manifest:
        raise RuntimeError("The active Brightway snapshot manifest is missing")
    return manifest


def iter_activity_metadata():
    """Yield metadata records from the active snapshot in stable name order."""
    collection = get_lca_activities_collection()
    manifest = get_active_snapshot(collection)
    projection = {
        "_id": 0,
        "name": 1,
        "name_lower": 1,
        "reference_product": 1,
        "unit": 1,
        "location": 1,
        "database": 1,
        "code": 1,
    }
    yield from collection.find(
        {"snapshot_id": manifest["snapshot_id"], "kind": "activity"}, projection
    ).sort([("name_lower", 1), ("location", 1), ("code", 1)]).batch_size(1_000)


def _remove_path(path: Path):
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


@contextmanager
def _cross_process_lock(path: Path, timeout: float = 1_800):
    """Serialize restores across Gunicorn worker processes on the same host."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock_file:
        if os.name == "nt" and path.stat().st_size == 0:
            lock_file.write(b"\0")
            lock_file.flush()

        deadline = time.monotonic() + timeout
        while True:
            try:
                lock_file.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for Brightway restore lock: {path}")
                time.sleep(0.25)

        try:
            yield
        finally:
            lock_file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _unpack_archive(archive_path: Path, target: Path, expected_file_count: int):
    target_resolved = target.resolve()
    extracted_files = 0
    seen_paths = set()

    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts or "\\" in info.filename:
                raise ValueError(f"Unsafe path in Brightway archive: {info.filename!r}")
            if not member.parts:
                continue

            output_path = (target / Path(*member.parts)).resolve()
            if not _is_within(output_path, target_resolved):
                raise ValueError(f"Archive path escapes restore directory: {info.filename!r}")
            if info.is_dir():
                output_path.mkdir(parents=True, exist_ok=True)
                continue

            target_key = os.path.normcase(str(output_path))
            if target_key in seen_paths:
                raise ValueError(f"Duplicate path in Brightway archive: {info.filename!r}")
            seen_paths.add(target_key)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"Symbolic links are not supported in Brightway archive: {info.filename!r}")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, output_path.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            extracted_files += 1

    if extracted_files != expected_file_count:
        raise ValueError(
            f"Brightway archive has {extracted_files} files; expected {expected_file_count}"
        )
    if extracted_files == 0:
        raise ValueError("Brightway archive contains no files")


def restore_brightway_database(destination: Path) -> str:
    """Restore the active MongoDB snapshot to Brightway's required local directory."""
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    lock_path = destination.with_name(f"{destination.name}.restore.lock")
    with _restore_lock, _cross_process_lock(lock_path):
        collection = get_lca_activities_collection()
        manifest = get_active_snapshot(collection)
        snapshot_id = manifest["snapshot_id"]
        marker = destination / ".mongodb_snapshot.json"
        try:
            if (
                marker.is_file()
                and (destination / "projects.db").is_file()
                and json.loads(marker.read_text(encoding="utf-8")).get("snapshot_id") == snapshot_id
            ):
                backup = destination.with_name(f"{destination.name}.restore-backup")
                _remove_path(backup)
                return snapshot_id
        except (OSError, ValueError, TypeError):
            pass

        stage = destination.with_name(f"{destination.name}.restore-tmp")
        backup = destination.with_name(f"{destination.name}.restore-backup")
        archive_path = None
        _remove_path(stage)
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        elif backup.exists():
            _remove_path(backup)
        stage.mkdir(parents=True)

        try:
            with tempfile.NamedTemporaryFile(prefix="brightway-", suffix=".zip", delete=False) as archive_file:
                archive_path = Path(archive_file.name)
                digest = hashlib.sha256()
                bytes_written = 0
                expected_index = 0
                cursor = collection.find(
                    {"snapshot_id": snapshot_id, "kind": "archive_chunk"},
                    {"_id": 0, "chunk_index": 1, "data": 1},
                ).sort("chunk_index", 1).batch_size(4)
                for chunk in cursor:
                    if chunk["chunk_index"] != expected_index:
                        raise RuntimeError(f"Brightway archive is missing chunk {expected_index}")
                    data = bytes(chunk["data"])
                    archive_file.write(data)
                    digest.update(data)
                    bytes_written += len(data)
                    expected_index += 1

            if expected_index != manifest["archive_chunks"]:
                raise RuntimeError(
                    f"Brightway archive has {expected_index} chunks; expected {manifest['archive_chunks']}"
                )
            if bytes_written != manifest["archive_size"] or digest.hexdigest() != manifest["archive_sha256"]:
                raise RuntimeError("Brightway archive size or SHA-256 checksum does not match its manifest")

            _unpack_archive(archive_path, stage, manifest["file_count"])
            if not (stage / "projects.db").is_file():
                raise RuntimeError("Restored Brightway snapshot is missing projects.db")
            (stage / ".mongodb_snapshot.json").write_text(
                json.dumps({"snapshot_id": snapshot_id}), encoding="utf-8"
            )

            replaced_existing = False
            if destination.exists():
                os.replace(destination, backup)
                replaced_existing = True
            try:
                os.replace(stage, destination)
            except Exception:
                if replaced_existing and backup.exists():
                    os.replace(backup, destination)
                raise
            if backup.exists():
                _remove_path(backup)
            logger.info("Restored Brightway snapshot %s to %s", snapshot_id, destination)
            return snapshot_id
        finally:
            if archive_path and archive_path.exists():
                archive_path.unlink()
            _remove_path(stage)
