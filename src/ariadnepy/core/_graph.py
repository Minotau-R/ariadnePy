from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Tuple

import networkx as nx
import pandas as pd

from ariadnepy.exceptions import AriadneError
from ariadnepy.core._versions import (
    GML_URL_TEMPLATE,
    load_version_metadata,
    resolve_versions,
)
from ariadnepy.core._download import download_gml, read_gml, insert_version


def _combine_graphs(graphs: list[Tuple[str, nx.Graph]]) -> nx.MultiDiGraph:
    """Merge a list of (source_name, graph) pairs into one MultiDiGraph."""
    combined = nx.MultiDiGraph()
    for source, graph in graphs:
        for node, attrs in graph.nodes(data=True):
            if not combined.has_node(node):
                combined.add_node(node, **attrs)
            else:
                combined.nodes[node].update(attrs)
        for u, v, attrs in graph.edges(data=True):
            combined.add_edge(u, v, source=source, **attrs)
    return combined


def ariadne(
    versions: Optional[Dict[str, str]] = None,
    cache_dir: Optional[str] = None,
) -> nx.MultiDiGraph:
    """Build the ariadne resource graph.

    Downloads GML files from Zenodo (or uses the local cache) and merges them
    into a single NetworkX MultiDiGraph where nodes are biological feature
    types and edges represent mappings between them.

    Parameters
    ----------
    versions:
        Dict mapping resource names to specific versions, e.g.
        ``{"GO": "2026-01-23"}``. Defaults are used for omitted resources.
    cache_dir:
        Directory for caching downloaded GML files.
        Defaults to ``.ariadne_cache`` in the current working directory.

    Returns
    -------
    nx.MultiDiGraph
        The assembled knowledge graph. ``graph.graph["versions"]`` holds the
        resolved version dict.

    Examples
    --------
    >>> from ariadnepy import ariadne
    >>> graph = ariadne()
    >>> graph = ariadne(versions={"GO": "2026-01-23"})
    """
    requested = resolve_versions(versions)
    metadata = load_version_metadata()

    selected = metadata[metadata["source"].isin(requested)].copy()
    selected["requested_version"] = selected["source"].map(requested)
    selected = selected[selected["version"] == selected["requested_version"]]

    if selected.empty:
        raise AriadneError("No resources were selected for graph construction.")

    cache_path = (
        Path(cache_dir) if cache_dir else Path(os.getcwd()) / ".ariadne_cache"
    )

    def _fetch_one(row: pd.Series) -> Tuple[str, nx.Graph]:
        source = row["source"]
        record_id = int(row["graph"])
        version_key = str(row["key"])
        url = GML_URL_TEMPLATE.format(record_id=record_id, source=source)
        local = download_gml(url, cache_path)
        graph = read_gml(local)
        insert_version(graph, version_key)
        return source, graph

    with ThreadPoolExecutor(max_workers=min(6, len(selected))) as pool:
        results = list(pool.map(_fetch_one, [row for _, row in selected.iterrows()]))

    combined = _combine_graphs(results)
    combined.graph["versions"] = requested
    return combined
