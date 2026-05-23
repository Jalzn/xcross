"""Streaming I/O for PFF tracking files."""

from __future__ import annotations

import bz2
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def iter_frames(path: Path) -> Iterator[dict[str, Any]]:
    """Yield one parsed frame dict per line of a `<match>.jsonl.bz2` file."""
    with bz2.open(path, mode="rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
