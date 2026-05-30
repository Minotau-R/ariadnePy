"""Tests for core/_graph.py — the ariadne() graph builder.

All tests mock download_gml / read_gml so no network is needed.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import igraph as ig
import pandas as pd
import pytest

from ariadnepy.exceptions import AriadneError


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_graph(nodes=("ko", "ec")) -> ig.Graph:
    """Build a minimal synthetic ig.Graph like ariadne() returns."""
    g = ig.Graph(directed=True)
    for n in nodes:
        g.add_vertex(name=n)
    # Don't include 'source' in attrs — _combine_graphs adds it separately
    g.add_edge(
        g.vs.find(name=nodes[0]).index,
        g.vs.find(name=nodes[1]).index,
        url="",
        from_=nodes[0],
        to=nodes[1],
    )
    return g


def _mock_metadata() -> pd.DataFrame:
    return pd.DataFrame([
        {"source": "KEGG", "version": "latest", "key": "", "graph": 19397292, "default": True},
    ])


# ── ariadne() return type ─────────────────────────────────────────────────────

def test_ariadne_returns_multidigraph():
    with patch("ariadnepy.core._graph.load_version_metadata", return_value=_mock_metadata()), \
         patch("ariadnepy.core._graph.resolve_versions", return_value={"KEGG": "latest"}), \
         patch("ariadnepy.core._graph.download_gml", return_value=Path("fake.gml")), \
         patch("ariadnepy.core._graph.read_gml", return_value=_make_graph()), \
         patch("ariadnepy.core._graph.insert_version"):
        from ariadnepy.core._graph import ariadne
        g = ariadne()
    assert isinstance(g, ig.Graph)


def test_ariadne_graph_has_nodes():
    with patch("ariadnepy.core._graph.load_version_metadata", return_value=_mock_metadata()), \
         patch("ariadnepy.core._graph.resolve_versions", return_value={"KEGG": "latest"}), \
         patch("ariadnepy.core._graph.download_gml", return_value=Path("fake.gml")), \
         patch("ariadnepy.core._graph.read_gml", return_value=_make_graph()), \
         patch("ariadnepy.core._graph.insert_version"):
        from ariadnepy.core._graph import ariadne
        g = ariadne()
    assert g.vcount() > 0


def test_ariadne_graph_has_edges():
    with patch("ariadnepy.core._graph.load_version_metadata", return_value=_mock_metadata()), \
         patch("ariadnepy.core._graph.resolve_versions", return_value={"KEGG": "latest"}), \
         patch("ariadnepy.core._graph.download_gml", return_value=Path("fake.gml")), \
         patch("ariadnepy.core._graph.read_gml", return_value=_make_graph()), \
         patch("ariadnepy.core._graph.insert_version"):
        from ariadnepy.core._graph import ariadne
        g = ariadne()
    assert g.ecount() > 0


def test_ariadne_stores_versions():
    with patch("ariadnepy.core._graph.load_version_metadata", return_value=_mock_metadata()), \
         patch("ariadnepy.core._graph.resolve_versions", return_value={"KEGG": "latest"}), \
         patch("ariadnepy.core._graph.download_gml", return_value=Path("fake.gml")), \
         patch("ariadnepy.core._graph.read_gml", return_value=_make_graph()), \
         patch("ariadnepy.core._graph.insert_version"):
        from ariadnepy.core._graph import ariadne
        g = ariadne()
    assert "versions" in g.attributes()
    assert g["versions"]["KEGG"] == "latest"


def test_ariadne_edges_have_source_attribute():
    with patch("ariadnepy.core._graph.load_version_metadata", return_value=_mock_metadata()), \
         patch("ariadnepy.core._graph.resolve_versions", return_value={"KEGG": "latest"}), \
         patch("ariadnepy.core._graph.download_gml", return_value=Path("fake.gml")), \
         patch("ariadnepy.core._graph.read_gml", return_value=_make_graph()), \
         patch("ariadnepy.core._graph.insert_version"):
        from ariadnepy.core._graph import ariadne
        g = ariadne()
    assert "source" in g.edge_attributes()


def test_ariadne_merges_multiple_sources():
    """Two sources → both contribute nodes/edges to the combined graph."""
    meta = pd.DataFrame([
        {"source": "KEGG",    "version": "latest", "key": "", "graph": 19397292, "default": True},
        {"source": "UniProt", "version": "latest", "key": "", "graph": 19397292, "default": True},
    ])
    graphs = {
        "KEGG":    _make_graph(("ko", "ec")),
        "UniProt": _make_graph(("uniref90", "ec")),
    }
    call_count = [0]
    def _mock_read(*_):
        src = list(graphs.keys())[call_count[0] % 2]
        call_count[0] += 1
        return graphs[src]

    with patch("ariadnepy.core._graph.load_version_metadata", return_value=meta), \
         patch("ariadnepy.core._graph.resolve_versions", return_value={"KEGG": "latest", "UniProt": "latest"}), \
         patch("ariadnepy.core._graph.download_gml", return_value=Path("fake.gml")), \
         patch("ariadnepy.core._graph.read_gml", side_effect=_mock_read), \
         patch("ariadnepy.core._graph.insert_version"):
        from ariadnepy.core._graph import ariadne
        g = ariadne()
    assert g.vcount() >= 3  # ko, ec, uniref90


def test_ariadne_no_resources_raises():
    """Empty selected DataFrame should raise AriadneError."""
    with patch("ariadnepy.core._graph.load_version_metadata", return_value=_mock_metadata()), \
         patch("ariadnepy.core._graph.resolve_versions", return_value={}):
        from ariadnepy.core._graph import ariadne
        with pytest.raises(AriadneError):
            ariadne()


# ── _combine_graphs ───────────────────────────────────────────────────────────

def test_combine_graphs_merges_nodes():
    from ariadnepy.core._graph import _combine_graphs
    g1 = _make_graph(("ko", "ec"))
    g2 = _make_graph(("ec", "pathway"))
    combined = _combine_graphs([("KEGG", g1), ("KEGG", g2)])
    node_names = combined.vs["name"]
    assert "ko" in node_names
    assert "pathway" in node_names


def test_combine_graphs_shared_node_attrs_updated():
    from ariadnepy.core._graph import _combine_graphs
    g1 = ig.Graph(directed=True)
    g1.add_vertex(name="ko", extra="first")
    g2 = ig.Graph(directed=True)
    g2.add_vertex(name="ko", extra="second")
    combined = _combine_graphs([("A", g1), ("B", g2)])
    ko_v = combined.vs.find(name="ko")
    assert ko_v["extra"] == "second"
