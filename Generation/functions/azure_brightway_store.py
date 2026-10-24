"""Version-aware Azure Blob cache for the Brightway database files."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import stat
import tempfile
import threading
import zipfile
from pathlib import Path, PurePosixPath

from .file_lock import acquire_file_lock, exclusive_file_lock

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_ECO_DIR = REPO_ROOT / "brightway_databases" / "brightway2_ecoinvent"
_blob_service = None
_blob_lock = threading.Lock()
_restore_lock = threading.RLock()
_process_data_dir = None
_process_lease = None
_process_lease_identity = None


def _get_blob_client():
    global _blob_service
    from azure.storage.blob import BlobServiceClient

    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
    container = os.getenv("AZURE_CONTAINER_NAME")
    blob_name = os.getenv("ECOINVENT_ZIP_BLOB")
    if not container or not blob_name:
        raise RuntimeError("AZURE_CONTAINER_NAME and ECOINVENT_ZIP_BLOB must be configured")

    if _blob_service is None:
        with _blob_lock:
            if _blob_service is None:
                if connection_string:
                    _blob_service = BlobServiceClient.from_connection_string(connection_string)
                elif account_url:
                    from azure.identity import DefaultAzureCredential

                    _blob_service = BlobServiceClient(account_url, credential=DefaultAzureCredential())
                else:
                    raise RuntimeError(
                        "Set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_URL "
                        "with a managed identity that can read blobs"
                    )
    return _blob_service.get_blob_client(container=container, blob=blob_name), container, blob_name


def _cache_root() -> Path:
    configured = os.getenv("BRIGHTWAY_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    if os.getenv("WEBSITE_SITE_NAME"):
        if os.getenv("WEBSITE_ENABLE_APP_SERVICE_STORAGE", "true").lower() == "false":
            logger.warning("App Service storage is disabled; Brightway cache may not persist across deployments")
        return (Path(os.getenv("HOME", "/home")) / "data" / "brightway").resolve()
    return (REPO_ROOT / "brightway_databases" / ".blob-cache").resolve()


def _valid_brightway_dir(path: Path) -> bool:
    return (path / "projects.db").is_file()


def _configured_limit(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def _extract_archive(
    archive_path: Path,
    target: Path,
    max_files: int | None = None,
    max_expanded_bytes: int | None = None,
) -> int:
    max_files = max_files or _configured_limit("BRIGHTWAY_MAX_FILES", 250_000)
    max_expanded_bytes = max_expanded_bytes or _configured_limit(
        "BRIGHTWAY_MAX_EXPANDED_BYTES", 50 * 1024**3
    )
    max_entries = _configured_limit("BRIGHTWAY_MAX_ZIP_ENTRIES", 500_000)
    target_resolved = target.resolve()
    extracted = 0
    seen = set()
    expanded_bytes = 0

    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if len(infos) > max_entries:
            raise ValueError(f"Brightway zip has too many entries ({len(infos)} > {max_entries})")
        files = [info for info in infos if not info.is_dir()]
        registries = [info for info in files if PurePosixPath(info.filename).name == "projects.db"]
        if not registries:
            raise ValueError("Brightway zip does not contain projects.db")
        registry = min(registries, key=lambda info: len(PurePosixPath(info.filename).parts))
        registry_path = PurePosixPath(registry.filename).as_posix()
        prefix = registry_path.rsplit("/", 1)[0] + "/" if "/" in registry_path else ""

        selected = []
        for info in infos:
            name = PurePosixPath(info.filename).as_posix()
            if prefix:
                if not name.startswith(prefix):
                    continue
                name = name[len(prefix):]
            if name:
                selected.append((info, name))
        selected_files = [(info, name) for info, name in selected if not info.is_dir()]
        if len(selected_files) > max_files:
            raise ValueError(f"Brightway zip has too many files ({len(selected_files)} > {max_files})")
        declared_bytes = sum(info.file_size for info, _ in selected_files)
        if declared_bytes > max_expanded_bytes:
            raise ValueError(
                f"Brightway zip expands to {declared_bytes} bytes; limit is {max_expanded_bytes}"
            )

        for info, name in selected:
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts or "\\" in name:
                raise ValueError(f"Unsafe path in Brightway zip: {info.filename!r}")
            output = (target / Path(*member.parts)).resolve()
            try:
                output.relative_to(target_resolved)
            except ValueError as exc:
                raise ValueError(f"Zip path escapes cache directory: {info.filename!r}") from exc

            if info.is_dir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            key = os.path.normcase(str(output))
            if key in seen:
                raise ValueError(f"Duplicate Brightway zip path: {info.filename!r}")
            seen.add(key)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Symlinks are not supported in Brightway zip: {info.filename!r}")

            output.parent.mkdir(parents=True, exist_ok=True)
            member_bytes = 0
            with archive.open(info) as source, output.open("wb") as destination:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    member_bytes += len(block)
                    expanded_bytes += len(block)
                    if expanded_bytes > max_expanded_bytes:
                        raise ValueError("Brightway zip exceeded its expanded-size limit")
                    destination.write(block)
            if member_bytes != info.file_size:
                raise ValueError(f"Brightway zip entry size mismatch: {info.filename!r}")
            extracted += 1

    if extracted == 0 or not (target / "projects.db").is_file():
        raise ValueError("Brightway zip did not restore a valid projects.db")
    return extracted


def _hold_process_lease(lease, identity: str):
    global _process_lease, _process_lease_identity
    if _process_lease_identity not in (None, identity):
        raise RuntimeError("This worker is already pinned to a different Brightway snapshot")
    _process_lease = lease
    _process_lease_identity = identity


def _prune_unused_versions(root: Path, active_identity: str):
    if os.name == "nt":
        return
    for cached in root.glob("brightway2_ecoinvent-*"):
        identity = cached.name.removeprefix("brightway2_ecoinvent-")
        if identity == active_identity or cached.is_symlink() or not cached.is_dir():
            continue
        lease = acquire_file_lock(root / f".{identity}.use.lock", timeout=0)
        if lease is None:
            continue
        try:
            if cached.exists() and not cached.is_symlink():
                shutil.rmtree(cached)
                logger.info("Pruned unused Brightway cache version %s", identity)
        except OSError:
            logger.warning("Could not prune unused Brightway cache %s", cached, exc_info=True)
        finally:
            lease.release()


def _restore_blob(blob_client, container: str, blob_name: str, properties) -> Path:
    etag = str(properties.etag)
    version_id = getattr(properties, "version_id", None)
    identity = hashlib.sha256(
        f"{blob_client.url}\0{version_id or etag}\0{etag}".encode()
    ).hexdigest()[:20]
    max_archive_bytes = _configured_limit("BRIGHTWAY_MAX_ARCHIVE_BYTES", 8 * 1024**3)
    if properties.size > max_archive_bytes:
        raise ValueError(
            f"Brightway Blob is {properties.size} bytes; archive limit is {max_archive_bytes}"
        )
    root = _cache_root()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"brightway2_ecoinvent-{identity}"
    marker_path = destination / ".azure_blob_snapshot.json"
    expected_metadata_hash = (getattr(properties, "metadata", None) or {}).get("sha256")
    expected_hash = (os.getenv("ECOINVENT_BLOB_SHA256") or expected_metadata_hash or "").strip().lower()

    lock_path = root / f".{identity}.restore.lock"
    lease_path = root / f".{identity}.use.lock"
    with _restore_lock, exclusive_file_lock(lock_path):
        version_lease = None
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            same_blob_version = marker.get("etag") == etag and marker.get("version_id") == version_id
        except (OSError, ValueError, TypeError):
            marker = {}
            same_blob_version = False

        if same_blob_version and _valid_brightway_dir(destination):
            if expected_hash and marker.get("archive_sha256") != expected_hash:
                raise ValueError("Cached Brightway snapshot does not match ECOINVENT_BLOB_SHA256")
            version_lease = acquire_file_lock(lease_path, shared=True)
            if os.name == "nt" or version_lease is not None:
                if _valid_brightway_dir(destination):
                    _hold_process_lease(version_lease, identity)
                    backup = root / f".{identity}.restore-backup"
                    if backup.exists():
                        shutil.rmtree(backup) if backup.is_dir() else backup.unlink()
                    _prune_unused_versions(root, identity)
                    return destination
            if version_lease is not None:
                version_lease.release()
                version_lease = None

        stage = root / f".{identity}.restore-tmp"
        backup = root / f".{identity}.restore-backup"
        archive_path = None
        if destination.exists() and version_lease is None and os.name != "nt":
            version_lease = acquire_file_lock(lease_path, timeout=1_800)
        elif not destination.exists() and os.name != "nt":
            version_lease = acquire_file_lock(lease_path, shared=True)

        if stage.exists():
            shutil.rmtree(stage) if stage.is_dir() else stage.unlink()
        if backup.exists():
            if not destination.exists():
                try:
                    backup_marker = json.loads((backup / ".azure_blob_snapshot.json").read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    backup_marker = {}
                if (
                    backup_marker.get("etag") == etag
                    and backup_marker.get("version_id") == version_id
                    and (not expected_hash or backup_marker.get("archive_sha256") == expected_hash)
                    and _valid_brightway_dir(backup)
                ):
                    os.replace(backup, destination)
                    _hold_process_lease(version_lease, identity)
                    version_lease = None
                    _prune_unused_versions(root, identity)
                    return destination
                if version_lease is not None:
                    version_lease.release()
                    version_lease = None
                if os.name != "nt":
                    version_lease = acquire_file_lock(lease_path, timeout=1_800)
            shutil.rmtree(backup) if backup.is_dir() else backup.unlink()
        stage.mkdir(parents=True)

        try:
            for stale_archive in root.glob(f"brightway-{identity}-*.zip"):
                stale_archive.unlink(missing_ok=True)
            from azure.core import MatchConditions

            with tempfile.NamedTemporaryFile(
                prefix=f"brightway-{identity}-", suffix=".zip", dir=root, delete=False
            ) as archive_file:
                archive_path = Path(archive_file.name)
                digest = hashlib.sha256()
                byte_count = 0
                downloader = blob_client.download_blob(
                    etag=properties.etag,
                    match_condition=MatchConditions.IfNotModified,
                    max_concurrency=4,
                )
                for block in downloader.chunks():
                    if byte_count + len(block) > max_archive_bytes:
                        raise ValueError("Brightway Blob exceeded BRIGHTWAY_MAX_ARCHIVE_BYTES")
                    archive_file.write(block)
                    digest.update(block)
                    byte_count += len(block)

            if byte_count != properties.size:
                raise RuntimeError(f"Downloaded {byte_count} bytes; Blob reports {properties.size}")
            archive_sha256 = digest.hexdigest()
            if expected_hash and archive_sha256 != expected_hash:
                raise RuntimeError("Brightway Blob SHA-256 does not match the configured checksum")

            file_count = _extract_archive(archive_path, stage)
            (stage / ".azure_blob_snapshot.json").write_text(
                json.dumps({
                    "container": container,
                    "blob": blob_name,
                    "etag": etag,
                    "version_id": version_id,
                    "archive_sha256": archive_sha256,
                    "archive_size": byte_count,
                    "file_count": file_count,
                }, sort_keys=True),
                encoding="utf-8",
            )

            if destination.exists():
                os.replace(destination, backup)
            try:
                os.replace(stage, destination)
            except Exception:
                if backup.exists() and not destination.exists():
                    os.replace(backup, destination)
                raise
            if backup.exists():
                shutil.rmtree(backup) if backup.is_dir() else backup.unlink()
            if version_lease is not None and not version_lease.shared:
                version_lease.downgrade()
            _hold_process_lease(version_lease, identity)
            version_lease = None
            logger.info(
                "Cached Brightway Blob %s/%s (ETag %s, SHA-256 %s) at %s",
                container, blob_name, etag, archive_sha256, destination,
            )
            _prune_unused_versions(root, identity)
            return destination
        finally:
            if version_lease is not None:
                version_lease.release()
            if archive_path and archive_path.exists():
                archive_path.unlink()
            if stage.exists():
                shutil.rmtree(stage) if stage.is_dir() else stage.unlink()


def ensure_ecoinvent_database() -> Path:
    """Return this process's pinned local snapshot; restart workers to adopt updates."""
    global _process_data_dir
    if _process_data_dir is not None:
        return _process_data_dir

    with _restore_lock:
        if _process_data_dir is not None:
            return _process_data_dir

        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
        container = os.getenv("AZURE_CONTAINER_NAME")
        blob_name = os.getenv("ECOINVENT_ZIP_BLOB")
        configured = any((connection_string, account_url, container, blob_name))
        if not configured:
            local_path = Path(os.getenv("BRIGHTWAY_DATA_DIR") or _LOCAL_ECO_DIR).expanduser().resolve()
            if _valid_brightway_dir(local_path) and not os.getenv("WEBSITE_SITE_NAME"):
                logger.info("Using existing local Brightway database at %s", local_path)
                _process_data_dir = local_path
                return local_path
            raise RuntimeError(
                "Configure Azure Blob access (AZURE_STORAGE_CONNECTION_STRING or "
                "AZURE_STORAGE_ACCOUNT_URL, AZURE_CONTAINER_NAME, ECOINVENT_ZIP_BLOB)"
            )

        blob_client, container, blob_name = _get_blob_client()
        properties = blob_client.get_blob_properties()
        _process_data_dir = _restore_blob(blob_client, container, blob_name, properties)
        return _process_data_dir
