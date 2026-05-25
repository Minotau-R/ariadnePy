"""Tests for ariadnepy._core (offline / unit-level)."""

from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from ariadnepy._core import (
    AriadneError,
    _combine_graphs,
    _insert_version_into_graph,
    _read_graph_file,
)


def _simple_graph() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    g.add_node("A", label="node_{version}")
    g.add_edge("A", "B", weight=1.0)
    return g


class TestInsertVersion:
    def test_replaces_version_placeholder_in_node(self):
        g = _simple_graph()
        _insert_version_into_graph(g, "v42")
        assert g.nodes["A"]["label"] == "node_v42"

    def test_no_placeholder_unchanged(self):
        g = nx.MultiDiGraph()
        g.add_node("X", label="static")
        _insert_version_into_graph(g, "v1")
        assert g.nodes["X"]["label"] == "static"


class TestCombineGraphs:
    def test_merges_nodes_and_edges(self):
        g1 = nx.MultiDiGraph()
        g1.add_node("A", color="red")
        g1.add_edge("A", "B")

        g2 = nx.MultiDiGraph()
        g2.add_node("C", color="blue")
        g2.add_edge("C", "A")

        combined = _combine_graphs([("src1", g1), ("src2", g2)])
        assert set(combined.nodes) == {"A", "B", "C"}
        assert combined.number_of_edges() == 2

    def test_edge_source_attribute_set(self):
        g = nx.MultiDiGraph()
        g.add_edge("X", "Y")
        combined = _combine_graphs([("mySource", g)])
        edge_data = list(combined.edges(data=True))[0][2]
        assert edge_data["source"] == "mySource"


class TestReadGraphFile:
    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(AriadneError, match="GML file not found"):
            _read_graph_file(tmp_path / "nonexistent.gml")

    def test_raises_on_corrupt_gml(self, tmp_path):
        bad = tmp_path / "bad.gml"
        bad.write_text("not valid gml content @@@@")
        with pytest.raises(AriadneError, match="Unable to parse"):
            _read_graph_file(bad)
