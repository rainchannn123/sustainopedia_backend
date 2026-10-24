"""Small cross-process file locks for Windows and Unix deployments."""

from __future__ import annotations

import errno
import os
import time
from contextlib import contextmanager
from pathlib import Path


class FileLock:
    def __init__(self, handle, shared=False):
        self.handle = handle
        self.shared = shared

    def downgrade(self):
        if self.shared:
            return
        if os.name == "nt":
            raise OSError("Windows byte-range locks cannot be downgraded to shared locks")
        import fcntl

        fcntl.flock(self.handle.fileno(), fcntl.LOCK_SH)
        self.shared = True

    def release(self):
        if self.handle.closed:
            return
        self.handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def acquire_file_lock(path: Path, *, shared=False, timeout: float = 1_800):
    """Acquire a process lock; return None when a shared lock is unsupported."""
    path = Path(path)
    if shared and os.name == "nt":
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    if os.name == "nt" and path.stat().st_size == 0:
        handle.write(b"\0")
        handle.flush()

    deadline = time.monotonic() + timeout
    while True:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
                fcntl.flock(handle.fileno(), operation | fcntl.LOCK_NB)
            return FileLock(handle, shared=shared)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                handle.close()
                raise
            if timeout <= 0:
                handle.close()
                return None
            if time.monotonic() >= deadline:
                handle.close()
                raise TimeoutError(f"Timed out waiting for file lock: {path}") from exc
            time.sleep(0.25)


@contextmanager
def exclusive_file_lock(path: Path, timeout: float = 1_800):
    lock = acquire_file_lock(path, timeout=timeout)
    try:
        yield
    finally:
        lock.release()
