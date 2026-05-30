from __future__ import annotations

import json
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ariadnepy.exceptions import AriadneError, AriadneVersionError

try:
    import requests as _requests
except ImportError:
    _requests = None

ZENODO_BASE_API = "https://zenodo.org/api/records"
GML_URL_TEMPLATE = "https://zenodo.org/records/{record_id}/files/{source}.gml"
DEFAULT_GRAPH_RECORD = 19397292

# Canonical sysdata.rda from Giulio's R package — single source of truth.
_RDA_URL = "https://raw.githubusercontent.com/Minotau-R/ariadne/main/R/sysdata.rda"

# URL base for each source — used to build human-readable URLs in list_resource_versions()
_SOURCE_BASE_URLS: Dict[str, str] = {
    "BugSigDB":   "https://zenodo.org/records/",
    "ChocoPhlAn": "https://zenodo.org/records/",
    "GM":         "https://github.com/",
    "GO":         "https://release.geneontology.org/",
    "KEGG":       "https://www.genome.jp/kegg",
    "MSigDB":     "https://zenodo.org/records/",
    "OTT":        "https://opentreeoflife.github.io",
    "Rhea":       "https://www.rhea-db.org",
    "TIGRFAMs":   "https://ftp.ncbi.nlm.nih.gov/hmm/TIGRFAMs/",
    "UniProt":    "https://www.uniprot.org",
    "WoL":        "https://ftp.microbio.me/pub/",
}

# Bundled fallback — used when offline or GitHub is unreachable
_BUNDLED_METADATA = Path(__file__).with_name("versions.json")


def fetch_json(url: str) -> Dict[str, Any]:
    """GET a URL and return parsed JSON, using requests if available."""
    if _requests is not None:
        resp = _requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise AriadneDownloadError(f"Request failed: {exc}") from exc
    except urllib.error.URLError as exc:
        raise AriadneDownloadError(f"Request failed: {exc}") from exc


def _load_from_rda() -> Optional[pd.DataFrame]:
    """Fetch sysdata.rda from Giulio's R package on GitHub and extract versionMetadata.

    This is the single source of truth. Returns None if the network is
    unreachable, so the caller can fall back to the bundled versions.json.
    """
    import pyreadr

    tmp_path: Optional[str] = None
    try:
        # Download binary .rda to a temp file
        with tempfile.NamedTemporaryFile(suffix=".rda", delete=False) as tmp:
            tmp_path = tmp.name
            if _requests is not None:
                resp = _requests.get(_RDA_URL, timeout=30)
                resp.raise_for_status()
                tmp.write(resp.content)
            else:
                urllib.request.urlretrieve(_RDA_URL, tmp_path)

        result = pyreadr.read_r(tmp_path)
        raw: Optional[pd.DataFrame] = result.get("versionMetadata")
        if raw is None or raw.empty:
            return None

        df: pd.DataFrame = raw.copy()
        df["graph"] = df["graph"].fillna(DEFAULT_GRAPH_RECORD).astype(int)
        df["default"] = df["default"].astype(bool)
        df["key"] = df["key"].fillna("").astype(str)
        return df

    except Exception:
        return None  # network unavailable — caller falls back to bundled JSON

    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _load_from_bundled_json() -> pd.DataFrame:
    """Load the bundled versions.json fallback."""
    with open(_BUNDLED_METADATA, encoding="utf-8") as fh:
        cfg = json.load(fh)
    rows: List[Dict[str, Any]] = cfg.get("static", [])
    return pd.DataFrame(rows)


def load_version_metadata() -> pd.DataFrame:
    """Return a DataFrame of all registered resource versions.

    Priority:
    1. ariadne/R/sysdata.rda — Giulio's source of truth (requires pyreadr)
    2. bundled versions.json — ships with ariadnePy as offline fallback
    """
    metadata = _load_from_rda()
    if metadata is None:
        metadata = _load_from_bundled_json()

    if metadata.empty:
        raise AriadneError("Failed to build ariadne version metadata.")

    metadata.attrs["urls"] = _SOURCE_BASE_URLS
    return metadata


def resolve_versions(versions: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Resolve user-requested versions against available metadata.

    Fills in defaults for any source not explicitly requested and validates
    that each requested version actually exists.
    """
    metadata = load_version_metadata()
    default_rows = metadata[metadata["default"]].set_index("source")
    defaults = default_rows["version"].to_dict()

    resolved = dict(defaults)
    if versions:
        resolved.update(versions)

    for source, version in resolved.items():
        available = metadata[metadata["source"] == source]["version"].tolist()
        if not available:
            raise AriadneVersionError(f"Unknown source: {source!r}")
        if version not in available:
            raise AriadneVersionError(
                f"Version {version!r} not available for {source}. "
                f"Available: {available}"
            )
    return resolved


def list_resource_versions(default: bool = False) -> pd.DataFrame:
    """Return available resource versions as a tidy DataFrame.

    Parameters
    ----------
    default:
        If True, return only the default version for each resource.

    Returns
    -------
    pd.DataFrame
        Columns: resource, version, url.

    Examples
    --------
    >>> from ariadnepy import list_resource_versions
    >>> list_resource_versions()
    >>> list_resource_versions(default=True)
    """
    metadata = load_version_metadata()
    if default:
        metadata = metadata[metadata["default"]].copy()
    else:
        metadata = metadata.copy()
    urls = metadata.attrs.get("urls", _SOURCE_BASE_URLS)
    metadata["url"] = metadata["source"].map(urls).fillna("") + metadata["key"].astype(str) + "/"
    metadata = metadata.rename(columns={"source": "resource"})
    return metadata[["resource", "version", "url"]].reset_index(drop=True)


# Avoid circular import — imported inside fetch_json only if needed
from ariadnepy.exceptions import AriadneDownloadError  # noqa: E402
