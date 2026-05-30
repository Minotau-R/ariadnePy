from __future__ import annotations

import json
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

# URL of the canonical versions.json — single source of truth shared with R.
# Point this at the R repo (raw GitHub) or a Zenodo record once Giulio decides.
# ariadnePy falls back to the bundled versions.json when the URL is unreachable.
REMOTE_METADATA_URL: Optional[str] = None  # e.g. "https://raw.githubusercontent.com/Minotau-R/ariadne/main/inst/extdata/versions.json"

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


def _load_metadata_json() -> Dict[str, Any]:
    """Load versions.json from the remote URL, falling back to the bundled copy.

    Priority:
    1. REMOTE_METADATA_URL (when set) — keeps Python in sync with R automatically
    2. Bundled versions.json shipped with the package — works offline
    """
    if REMOTE_METADATA_URL:
        try:
            return fetch_json(REMOTE_METADATA_URL)
        except Exception:
            pass  # fall through to bundled copy
    with open(_BUNDLED_METADATA, encoding="utf-8") as fh:
        return json.load(fh)


def _load_dynamic_versions(source: str, record_id: int) -> List[Dict[str, Any]]:
    default_graph_record = 19397292
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
            "graph": default_graph_record,
            "default": idx == 0,
        })
    if not rows:
        raise AriadneError(f"No valid version entries for dynamic source {source}")
    return rows


def load_version_metadata() -> pd.DataFrame:
    """Return a DataFrame of all registered resource versions.

    Static and dynamic-Zenodo entries are read from versions.json (remote or
    bundled). Dynamic sources have their version list fetched live from Zenodo.
    """
    cfg = _load_metadata_json()

    rows: List[Dict[str, Any]] = list(cfg.get("static", []))

    for source, record_id in sorted(cfg.get("dynamic_zenodo", {}).items()):
        rows.extend(_load_dynamic_versions(source, record_id))

    metadata = pd.DataFrame(rows)
    if metadata.empty:
        raise AriadneError("Failed to build ariadne version metadata.")
    metadata.attrs["urls"] = cfg.get("source_urls", {})
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
    urls = metadata.attrs.get("urls", {})
    metadata["url"] = metadata["source"].map(urls).fillna("") + metadata["key"].astype(str) + "/"
    metadata = metadata.rename(columns={"source": "resource"})
    return metadata[["resource", "version", "url"]].reset_index(drop=True)


# Avoid circular import — imported inside fetch_json only if needed
from ariadnepy.exceptions import AriadneDownloadError  # noqa: E402
