"""Keep native runs, validation scripts and tests from owning separate worlds."""
import fcntl
import hashlib
import os
from pathlib import Path
import tempfile
import threading

_guard = threading.Lock()
_owner_file = None
_owner_pid = None


def acquire_genesis_owner(path=None):
    global _owner_file, _owner_pid
    with _guard:
        if _owner_file is not None and _owner_pid==os.getpid():
            return
        if _owner_file is not None:
            _owner_file.close()
        root = str(Path(__file__).resolve().parents[3])
        digest = hashlib.sha256(root.encode()).hexdigest()[:12]
        path = Path(path) if path is not None else Path(tempfile.gettempdir())/f'sim-agent-genesis-{digest}.lock'
        stream = path.open('a+',encoding='utf-8')
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            stream.seek(0)
            owner = stream.read(80).strip()
            stream.close()
            raise RuntimeError(f'Another Sim-Agent Genesis process owns the world (PID {owner}). Finish that run before starting a viewer or validation.') from None
        stream.seek(0);stream.truncate();stream.write(str(os.getpid()));stream.flush()
        _owner_file, _owner_pid = stream, os.getpid()
