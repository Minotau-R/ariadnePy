from __future__ import annotations

import json
import urllib.error
import urllib.request
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

# Sources whose versions are fetched dynamically from Zenodo
_DYNAMIC_ZENODO_SOURCES: Dict[str, int] = {
    "BugSigDB": 5606166,
    "ChocoPhlAn": 17100034,
    "MSigDB": 15377497,
}

# Sources with pinned version metadata
_STATIC_VERSION_METADATA: List[Dict[str, Any]] = [
    {"source": "GO", "version": "2026-03-25", "key": "2026-03-25", "graph": DEFAULT_GRAPH_RECORD, "default": True},
    {"source": "GO", "version": "2026-01-23", "key": "2026-01-23", "graph": DEFAULT_GRAPH_RECORD, "default": False},
    {"source": "GO", "version": "2025-10-10", "key": "2025-10-10", "graph": DEFAULT_GRAPH_RECORD, "default": False},
    {"source": "GM", "version": "v1", "key": "omixer/omixer-rpmR/raw/refs/heads/main/inst/extdata", "graph": DEFAULT_GRAPH_RECORD, "default": True},
    {"source": "TIGRFAMs", "version": "v15", "key": "release_15.0", "graph": DEFAULT_GRAPH_RECORD, "default": True},
    {"source": "WoL", "version": "v2", "key": "wol2", "graph": DEFAULT_GRAPH_RECORD, "default": True},
    {"source": "WoL", "version": "v20April2021", "key": "wol-20April2021", "graph": 18788726, "default": False},
    {"source": "KEGG", "version": "latest", "key": "", "graph": DEFAULT_GRAPH_RECORD, "default": True},
    {"source": "OTT", "version": "latest", "key": "", "graph": DEFAULT_GRAPH_RECORD, "default": True},
    {"source": "Rhea", "version": "latest", "key": "", "graph": DEFAULT_GRAPH_RECORD, "default": True},
    {"source": "UniProt", "version": "latest", "key": "", "graph": DEFAULT_GRAPH_RECORD, "default": True},
]

_SOURCE_BASE_URLS: Dict[str, str] = {
    "BugSigDB": "https://zenodo.org/records/",
    "ChocoPhlAn": "https://zenodo.org/records/",
    "GM": "https://github.com/",
    "GO": "https://release.geneontology.org/",
    "KEGG": "https://www.genome.jp/kegg",
    "MSigDB": "https://zenodo.org/records/",
    "OTT": "https://opentreeoflife.github.io",
    "Rhea": "https://www.rhea-db.org",
    "TIGRFAMs": "https://ftp.ncbi.nlm.nih.gov/hmm/TIGRFAMs/",
    "UniProt": "https://www.uniprot.org",
    "WoL": "https://ftp.microbio.me/pub/",
}


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


def _load_dynamic_versions(source: str) -> List[Dict[str, Any]]:
    record_id = _DYNAMIC_ZENODO_SOURCES[source]
    data = fetch_json(f"{ZENODO_BASE_API}/{record_id}/versions")
    hits = data.get("hits", {}).get("hits", [])
    rows = []
    for idx, hit in enumerate(hits):
        version_name = hit.get("metadata", {}).get("version")
        if not version_name:
            continue
        rows.append({
            "source": source,
            "version": version_name,
            "key": str(hit.get("id", "")),
            "graph": DEFAULT_GRAPH_RECORD,
            "default": idx == 0,
        })
    if not rows:
        raise AriadneError(f"No valid version entries for dynamic source {source}")
    return rows


def load_version_metadata() -> pd.DataFrame:
    """Return a DataFrame of all registered resource versions."""
    rows = list(_STATIC_VERSION_METADATA)
    for source in sorted(_DYNAMIC_ZENODO_SOURCES):
        rows.extend(_load_dynamic_versions(source))
    metadata = pd.DataFrame(rows)
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
    metadata["url"] = metadata["source"].map(urls) + metadata["key"].astype(str) + "/"
    metadata = metadata.rename(columns={"source": "resource"})
    return metadata[["resource", "version", "url"]].reset_index(drop=True)


# Avoid circular import — imported inside fetch_json only if needed
from ariadnepy.exceptions import AriadneDownloadError  # noqa: E402
