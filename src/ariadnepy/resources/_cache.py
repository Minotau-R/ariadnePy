from __future__ import annotations

import os
import re
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

from ariadnepy.exceptions import AriadneCacheError, AriadneError

try:
    from platformdirs import user_cache_dir as _user_cache_dir
except ImportError:
    try:
        from appdirs import user_cache_dir as _user_cache_dir
    except ImportError:
        _user_cache_dir = None

try:
    import requests as _requests
except ImportError:
    _requests = None


# ── Cache directory ───────────────────────────────────────────────────────────

def init_cache() -> Path:
    """Return (and create) the ariadnepy resource cache directory."""
    if _user_cache_dir is not None:
        base = Path(_user_cache_dir("ariadnepy"))
    elif os.name == "nt":
        base = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "ariadnepy"
    else:
        base = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "ariadnepy"
    cache = base / "resource_cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def add_to_cache(df: pd.DataFrame, cache_path: Path) -> Path:
    """Write a DataFrame to parquet at cache_path, creating parent dirs."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return cache_path


# ── Download helpers ──────────────────────────────────────────────────────────

def _download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _requests is not None:
        resp = _requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(8192):
                if chunk:
                    fh.write(chunk)
        return dest
    with urllib.request.urlopen(url, timeout=120) as resp, dest.open("wb") as fh:
        shutil.copyfileobj(resp, fh)
    return dest


def _localize(url: str) -> Path:
    """Return a local Path for url — downloading to a temp file if needed."""
    path = Path(url)
    if path.exists():
        return path
    if str(url).startswith(("http://", "https://")):
        suffix = Path(url).suffix or ""
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.close()
        return _download_file(url, Path(tmp.name))
    raise FileNotFoundError(f"Resource not found: {url}")


# ── Per-resource preprocessing ────────────────────────────────────────────────

def _preprocess(url: str, res_name: str, from_col: str, to_col: str) -> pd.DataFrame:
    """Download and parse one resource into a normalised 2-column DataFrame."""
    from ariadnepy.resources._parsers import (
        process_one2one,
        process_one2many,
        process_complex_modules,
    )
    from ariadnepy.resources._rds import process_rdslist

    if res_name == "ChocoPhlAn":
        src = _localize(url)
        return process_one2many(
            src, from_col, to_col,
            key_fn=lambda k: k.replace("GO:", ""),
        )

    if res_name == "WoL":
        src = _localize(url)
        key_fn = (lambda k: "UniRef90_" + k) if from_col == "uniref90" else None
        return process_one2many(
            src, from_col, to_col,
            key_fn=key_fn,
            val_fn=lambda v: v.replace("EC-", ""),
        )

    if res_name == "BugSigDB":
        src = _localize(url)

        def _bug_key(k: str) -> str:
            k = re.sub(r"^bsdb:", "", k)
            return re.sub(r"_.*$", "", k)

        return process_one2many(
            src, from_col, to_col,
            key_fn=_bug_key,
            skiprows=1,
            val_cols=slice(2, None),
        )

    if res_name in {"TIGRFAMs", "GO"}:
        src = _localize(url)
        return process_one2one(src, from_col, to_col)

    if res_name == "GM":
        src = _localize(url)
        return process_complex_modules(src, from_col, to_col)

    if res_name == "MSigDB":
        return process_rdslist(url)

    # Generic fallback
    src = _localize(url)
    return process_one2many(src, from_col, to_col)


# ── Public API ────────────────────────────────────────────────────────────────

def cache_resource(
    url: str,
    res_name: str,
    from_col: str,
    to_col: str,
    force: bool = False,
) -> Path:
    """Return the path to a cached parquet linkmap, downloading if needed.

    Parameters
    ----------
    url:
        Remote or local URL of the raw resource file.
    res_name:
        Resource identifier (e.g. ``"GO"``, ``"KEGG"``, ``"WoL"``).
    from_col:
        Name of the source column in the resulting linkmap.
    to_col:
        Name of the target column in the resulting linkmap.
    force:
        If True, re-download and re-process even if a cached file exists.

    Returns
    -------
    Path
        Local path to the ``.parquet`` linkmap file.

    Examples
    --------
    >>> from ariadnepy.resources import cache_resource
    >>> path = cache_resource(url, "GO", "go_id", "go_name")
    """
    cache_dir = init_cache()
    basename = Path(url).name or res_name
    safe_name = re.sub(r"[^A-Za-z0-9_.\-]+", "_", basename)
    cache_path = cache_dir / f"{res_name}_{safe_name}.parquet"

    if cache_path.exists() and not force:
        return cache_path

    try:
        df = _preprocess(url, res_name, from_col, to_col)
    except Exception as exc:
        raise AriadneCacheError(
            f"Failed to preprocess resource {res_name!r} from {url!r}: {exc}"
        ) from exc

    return add_to_cache(df, cache_path)
