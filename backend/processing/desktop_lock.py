"""Share the existing single-model slot with the Rust desktop preparation queue."""

from __future__ import annotations

import asyncio
import errno
import os
from contextlib import asynccontextmanager
from pathlib import Path


def _try_lock(file) -> bool:
    if os.name == "nt":
        import msvcrt

        try:
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise
            return False
    else:
        import fcntl

        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
    return True


@asynccontextmanager
async def processing_slot(outputs: str, semaphore: asyncio.Semaphore):
    async with semaphore:
        with (Path(outputs) / "processing.lock").open("a+b") as lock:
            lock.seek(0)
            while not _try_lock(lock):
                await asyncio.sleep(0.2)
            yield
