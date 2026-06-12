"""Lazy loader for bundled package config (the IMEPLAN column rename map).

This is *code config*, not data: ``rename_imeplan`` needs it to run, so it ships inside the
package (not via the data mirror) and the package works offline.
"""
from __future__ import annotations

import functools
import json
from importlib import resources


@functools.cache
def imeplan_rename_map() -> dict:
    """Column rename map + ordinal mappings for the three IMEPLAN tables."""
    text = (resources.files("eodgdl") / "imeplan_rename_map.json").read_text(encoding="utf-8")
    return json.loads(text)
