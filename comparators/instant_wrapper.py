from __future__ import annotations

import subprocess
from pathlib import Path


def run_instant(command: list[str], cwd: str | Path | None = None) -> int:
    """Run an external INSTANT command.

    The INSTANT implementation is not vendored here. Install the released
    INSTANT code separately and pass its command line to this wrapper.
    """
    return subprocess.call(command, cwd=None if cwd is None else str(cwd))
