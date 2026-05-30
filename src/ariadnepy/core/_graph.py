from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import igraph as ig
import pandas as pd

from ariadnepy.exceptions import AriadneError
from ariadnepy.core._versions import (
    GML_URL_TEMPLATE,
    load_version_metadata,
    resolve_versions,
)
from ariadnepy.core._download import download_gml, read_gml, insert_version


def _combine_graphs(graphs: List[Tuple[str, ig.Graph]]) -> ig.Graph:
    """Merge a list of (source_name, graph) pairs into one directed igraph Graph."""
    combined = ig.Graph(directed=True)

    # Track vertex names already added
    name_to_idx: Dict[str, int] = {}

    for source, graph in graphs:
        # Add vertices (nodes)
        for v in graph.vs:
            name = v["name"] if "name" in graph.vertex_attributes() else str(v.index)
            if name not in name_to_idx:
                attrs = {k: v[k] for k in graph.vertex_attributes()}
                combined.add_vertex(name=name, **{k: v for k, v in attrs.items() if k != "name"})
                name_to_idx[name] = combined.vcount() - 1
            else:
                # Update existing vertex attributes
                idx = name_to_idx[name]
                for attr in graph.vertex_attributes():
                    if attr != "name":
                        val = v[attr]
                        if val is not None:
                            combined.vs[idx][attr] = val

        # Add edges with source label
        for e in graph.es:
            src_name = graph.vs[e.source]["name"]
            tgt_name = graph.vs[e.target]["name"]
            src_idx = name_to_idx[src_name]
            tgt_idx = name_to_idx[tgt_name]
            combined.add_edge(src_idx, tgt_idx)
            edge_attrs = {k: e[k] for k in graph.edge_attributes()}
            edge_attrs["source"] = source
            for k, v in edge_attrs.items():
                combined.es[-1][k] = v

    return combined


def ariadne(
    versions: Optional[Dict[str, str]] = None,
    cache_dir: Optional[str] = None,
) -> ig.Graph:
    """Build the ariadne resource graph.

    Downloads GML files from Zenodo (or uses the local cache) and merges them
    into a single directed igraph Graph where vertices are biological feature
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
    ig.Graph
        The assembled knowledge graph. ``graph["versions"]`` holds the
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

    def _fetch_one(row: pd.Series) -> Tuple[str, ig.Graph]:
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
    combined["versions"] = requested
    return combined
