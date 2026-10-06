"""
Console encoding safety for Windows.

TorchXRayVision prints a UTF-8 block-character progress bar ("█") while downloading model
weights. Windows terminals default to the cp1252 code page, which cannot encode that glyph,
so the write raises UnicodeEncodeError mid-download — which aborts the download and leaves a
*corrupted, truncated* weights file. Forcing stdout/stderr to UTF-8 (with errors="replace")
makes the progress bar harmless and lets the download complete.
"""
from __future__ import annotations

import sys


def ensure_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            # Stream may be redirected/replaced (e.g. in some test runners); ignore.
            pass
