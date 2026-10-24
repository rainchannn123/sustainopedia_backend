"""Ensure a version-checked local Brightway cache exists for LCA calculations."""

import logging

from .azure_brightway_store import ensure_ecoinvent_database

logger = logging.getLogger(__name__)


def preinstall_ecoinvent_database():
    """Return the local data directory, restoring it from Blob only if needed."""
    try:
        return ensure_ecoinvent_database()
    except Exception:
        logger.exception("Could not prepare the Brightway database cache")
        return None
