import os
import signal
import subprocess
import time
from typing import Any


def _group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    return True


def terminate_process_group(
    process: subprocess.Popen[Any],
    term_timeout: float = 20,
    kill_timeout: float = 10,
) -> None:
    """Terminate every process in a start_new_session=True child group."""
    process_group_id = process.pid
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        process.poll()
        return
    except OSError:
        if process.poll() is None:
            process.terminate()

    deadline = time.monotonic() + term_timeout
    while time.monotonic() < deadline:
        process.poll()
        if not _group_exists(process_group_id):
            return
        time.sleep(0.1)

    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        if process.poll() is None:
            process.kill()

    deadline = time.monotonic() + kill_timeout
    while time.monotonic() < deadline:
        process.poll()
        if not _group_exists(process_group_id):
            return
        time.sleep(0.1)
    process.poll()
