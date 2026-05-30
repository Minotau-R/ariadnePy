from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import igraph as ig

from ariadnepy.exceptions import AriadneDownloadError, AriadneParseError

try:
    import requests as _requests
except ImportError:
    _requests = None


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_gml(url: str, cache_dir: Path) -> Path:
    """Download a GML file to cache_dir, skipping if already present."""
    ensure_directory(cache_dir)
    filename = Path(url).name
    dest = cache_dir / filename

    if dest.exists() and dest.stat().st_size > 0:
        return dest

    if _requests is not None:
        with _requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
        return dest

    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            with open(dest, "wb") as fh:
                fh.write(resp.read())
        return dest
    except urllib.error.HTTPError as exc:
        raise AriadneDownloadError(f"Failed to download GML: {exc}") from exc
    except urllib.error.URLError as exc:
        raise AriadneDownloadError(f"Failed to download GML: {exc}") from exc


def read_gml(path: Path) -> ig.Graph:
    """Parse a GML file into a directed igraph Graph."""
    if not path.exists():
        raise AriadneDownloadError(f"GML file not found: {path}")
    try:
        g = ig.Graph.Read_GML(str(path))
        if not g.is_directed():
            g = g.as_directed()
        return g
    except (AriadneDownloadError, AriadneParseError):
        raise
    except Exception as exc:
        raise AriadneParseError(f"Cannot parse GML file {path}: {exc}") from exc


def insert_version(graph: ig.Graph, key: str) -> None:
    """Replace '{version}' placeholders in all vertex and edge attributes in-place."""
    for v in graph.vs:
        for attr in graph.vertex_attributes():
            val = v[attr]
            if isinstance(val, str) and "{version}" in val:
                v[attr] = val.replace("{version}", key)

    for e in graph.es:
        for attr in graph.edge_attributes():
            val = e[attr]
            if isinstance(val, str) and "{version}" in val:
                e[attr] = val.replace("{version}", key)
