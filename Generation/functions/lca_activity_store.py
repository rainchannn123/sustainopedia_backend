"""MongoDB access to the searchable Brightway activity metadata index."""

from __future__ import annotations

import os
import threading

from pymongo import MongoClient
from pymongo.uri_parser import parse_uri

COLLECTION_NAME = "lca_activities"
_client = None
_database_name = None
_client_lock = threading.Lock()


def get_mongodb_database_name(uri=None):
    uri = uri or os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI is required for Brightway activity metadata")
    database_name = parse_uri(uri).get("database")
    if not database_name:
        raise RuntimeError("MONGODB_URI must include the target database name in its path")
    return database_name


def get_lca_activities_collection():
    """Connect lazily to the database selected in MONGODB_URI."""
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


def get_active_activity_snapshot(collection=None):
    collection = collection if collection is not None else get_lca_activities_collection()
    active = collection.find_one({"_id": "active_activity_snapshot", "kind": "activity_active"})
    if not active or not active.get("snapshot_id"):
        raise RuntimeError("No activity metadata snapshot is active in lca_activities")
    return active


def iter_activity_metadata(snapshot_id=None):
    """Yield the active snapshot's activity metadata in stable name order."""
    collection = get_lca_activities_collection()
    snapshot_id = snapshot_id or get_active_activity_snapshot(collection)["snapshot_id"]
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
        {"snapshot_id": snapshot_id, "kind": "activity"}, projection
    ).sort([("name_lower", 1), ("location", 1), ("code", 1)]).batch_size(1_000)
