from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

ZENODO_BASE_API = "https://zenodo.org/api/records"
GML_URL_TEMPLATE = "https://zenodo.org/records/{record_id}/files/{source}.gml"
DEFAULT_GRAPH_RECORD = 19397292
_DYNAMIC_ZENODO_SOURCES = {
    "BugSigDB": 5606166,
    "ChocoPhlAn": 17100034,
    "MSigDB": 15377497,
}

_STATIC_VERSION_METADATA = [
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

_SOURCE_BASE_URLS = {
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


class AriadneError(Exception):
    pass


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_dynamic_versions(source: str) -> pd.DataFrame:
    record_id = _DYNAMIC_ZENODO_SOURCES[source]
    response = _fetch_json(f"{ZENODO_BASE_API}/{record_id}/versions")
    hits = response.get("hits", {}).get("hits", [])
    if not hits:
        raise AriadneError(f"No versions found in Zenodo for {source}")

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

    return pd.DataFrame(rows)


def _fetch_json(url: str) -> Dict[str, Any]:
    if requests is not None:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        raise AriadneError(f"Zenodo metadata request failed: {exc}") from exc
    except urllib.error.URLError as exc:
        raise AriadneError(f"Zenodo metadata request failed: {exc}") from exc


def _load_version_metadata() -> pd.DataFrame:
    rows = list(_STATIC_VERSION_METADATA)
    for source in sorted(_DYNAMIC_ZENODO_SOURCES):
        rows.extend(_load_dynamic_versions(source).to_dict(orient="records"))

    metadata = pd.DataFrame(rows)
    if metadata.empty:
        raise AriadneError("Failed to build ariadne version metadata")

    metadata.attrs["urls"] = _SOURCE_BASE_URLS
    return metadata


def _resolve_requested_versions(versions: Optional[Dict[str, str]]) -> Dict[str, str]:
    metadata = _load_version_metadata()
    default_rows = metadata[metadata["default"]].set_index("source")
    default_versions = default_rows["version"].to_dict()

    versions = {} if versions is None else dict(versions)
    for source, version in default_versions.items():
        versions.setdefault(source, version)

    requested = []
    for source, version in versions.items():
        available = metadata[metadata["source"] == source]["version"].tolist()
        if version not in available:
            raise AriadneError(
                f"Invalid version {version!r} for source {source}. "
                f"Available versions are: {available}"
            )
        requested.append((source, version))

    return versions


def _build_graph_url(source: str, graph_record: int) -> str:
    return GML_URL_TEMPLATE.format(record_id=graph_record, source=source)


def _download_gml_file(url: str, cache_dir: Path) -> Path:
    cache_dir = _ensure_directory(cache_dir)
    filename = Path(url).name
    destination = cache_dir / filename
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    if requests is not None:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(destination, "wb") as file_handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file_handle.write(chunk)
        return destination

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            with open(destination, "wb") as file_handle:
                file_handle.write(response.read())
        return destination
    except urllib.error.HTTPError as exc:
        raise AriadneError(f"Failed to download GML file: {exc}") from exc
    except urllib.error.URLError as exc:
        raise AriadneError(f"Failed to download GML file: {exc}") from exc


def _insert_version_into_graph(graph: nx.Graph, key: str) -> None:
    for node, attrs in graph.nodes(data=True):
        for name, value in list(attrs.items()):
            if isinstance(value, str) and "{version}" in value:
                graph.nodes[node][name] = value.replace("{version}", key)

    if graph.is_multigraph():
        for u, v, _, attrs in graph.edges(keys=True, data=True):
            for name, value in list(attrs.items()):
                if isinstance(value, str) and "{version}" in value:
                    attrs[name] = value.replace("{version}", key)
    else:
        for u, v, attrs in graph.edges(data=True):
            for name, value in list(attrs.items()):
                if isinstance(value, str) and "{version}" in value:
                    attrs[name] = value.replace("{version}", key)


def _read_graph_file(path: Path) -> nx.MultiDiGraph:
    if not path.exists():
        raise AriadneError(f"GML file not found: {path}")
    try:
        return nx.read_gml(str(path), label="name")
    except Exception as exc:
        raise AriadneError(f"Unable to parse GML file {path}: {exc}") from exc


def _combine_graphs(graphs: List[Tuple[str, nx.Graph]]) -> nx.MultiDiGraph:
    combined = nx.MultiDiGraph()
    for source, graph in graphs:
        for node, attrs in graph.nodes(data=True):
            if not combined.has_node(node):
                combined.add_node(node, **attrs)
            else:
                combined.nodes[node].update(attrs)

        for u, v, attrs in graph.edges(data=True):
            decorated_attrs = dict(attrs)
            decorated_attrs["source"] = source
            combined.add_edge(u, v, **decorated_attrs)

    return combined


def ariadne(
    versions: Optional[Dict[str, str]] = None,
    cache_dir: Optional[str] = None,
) -> nx.MultiDiGraph:
    """Build the ariadne resource graph from version metadata and GML sources."""
    requested_versions = _resolve_requested_versions(versions)
    metadata = _load_version_metadata()
    selected = metadata[metadata["source"].isin(requested_versions)].copy()
    selected["requested_version"] = selected["source"].map(requested_versions)
    selected = selected[selected["version"] == selected["requested_version"]]

    if selected.empty:
        raise AriadneError("No resources were selected for graph construction.")

    cache_path = Path(cache_dir) if cache_dir is not None else Path(os.getcwd()) / ".ariadne_cache"
    graphs: List[Tuple[str, nx.Graph]] = []

    def fetch_one(row: pd.Series) -> Tuple[str, nx.Graph]:
        source = row["source"]
        graph_record = int(row["graph"])
        version_key = str(row["key"])
        url = _build_graph_url(source, graph_record)
        local_path = _download_gml_file(url, cache_path)
        source_graph = _read_graph_file(local_path)
        _insert_version_into_graph(source_graph, version_key)
        return source, source_graph

    with ThreadPoolExecutor(max_workers=min(6, len(selected))) as executor:
        results = list(executor.map(fetch_one, [row for _, row in selected.iterrows()]))

    combined_graph = _combine_graphs(results)
    combined_graph.graph["versions"] = requested_versions
    return combined_graph


def list_resource_versions(default: bool = False) -> pd.DataFrame:
    """Return available resource versions."""
    metadata = _load_version_metadata()
    if default:
        metadata = metadata[metadata["default"]]
    metadata = metadata.copy()
    metadata["url"] = metadata["source"].map(metadata.attrs["urls"]) + metadata["key"].astype(str) + "/"
    metadata = metadata.rename(columns={"source": "resource"})
    return metadata[["resource", "version", "url"]]
