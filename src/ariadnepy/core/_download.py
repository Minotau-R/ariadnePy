from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import networkx as nx

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


def read_gml(path: Path) -> nx.MultiDiGraph:
    """Parse a GML file into a NetworkX MultiDiGraph."""
    if not path.exists():
        raise AriadneDownloadError(f"GML file not found: {path}")
    try:
        return nx.read_gml(str(path), label="name")
    except Exception as exc:
        raise AriadneParseError(f"Cannot parse GML file {path}: {exc}") from exc


def insert_version(graph: nx.MultiDiGraph, key: str) -> None:
    """Replace '{version}' placeholders in all node and edge attributes in-place."""
    for node, attrs in graph.nodes(data=True):
        for name, value in list(attrs.items()):
            if isinstance(value, str) and "{version}" in value:
                graph.nodes[node][name] = value.replace("{version}", key)

    for u, v, attrs in graph.edges(data=True):
        for name, value in list(attrs.items()):
            if isinstance(value, str) and "{version}" in value:
                attrs[name] = value.replace("{version}", key)
