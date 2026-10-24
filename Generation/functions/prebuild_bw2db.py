"""Restore the Brightway database snapshot from MongoDB when calculation needs it."""

import logging
from pathlib import Path

from .lca_activity_store import restore_brightway_database

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
ECOINVENT_DIR = REPO_ROOT / "brightway_databases" / "brightway2_ecoinvent"


def preinstall_ecoinvent_database():
    """Restore the active MongoDB snapshot into Brightway's local data directory."""
    try:
        restore_brightway_database(ECOINVENT_DIR)
        return True
    except Exception:
        logger.exception("Could not restore the Brightway snapshot from MongoDB")
        return False
