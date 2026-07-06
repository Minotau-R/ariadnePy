"""Tests for plot/_custom.py::add_resource.

add_resource connects CSV *column names* as graph nodes (not row values).
The fixture CSV has columns 'known_node' and 'new_node', so those two strings
become the candidate vertex names.  The base graph below contains 'known_node'
as a pre-existing vertex so that the "at least one feature must already be in
the graph" invariant is satisfied for all positive-path tests.

All tests use a synthetic offline undirected graph — no network calls.
"""
from __future__ import annotations

import igraph as ig
import pandas as pd
import pytest

from ariadnepy.exceptions import AriadneError
from ariadnepy.plot._custom import add_resource


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def base_graph() -> ig.Graph:
    """Minimal undirected graph that already contains 'known_node'.

    Having 'known_node' pre-existing means a CSV whose first column is named
    'known_node' will pass the "at least one feature in graph" guard.
    """
    g = ig.Graph(directed=False)
    g.add_vertex(name="known_node")
    return g


@pytest.fixture
def tmp_linkmap(tmp_path) -> str:
    """CSV with columns 'known_node' and 'new_node' — two rows of sample data."""
    df = pd.DataFrame({"known_node": ["x", "y"], "new_node": ["p", "q"]})
    p = tmp_path / "map.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def both_new_linkmap(tmp_path) -> str:
    """CSV whose column names ('alpha', 'beta') are both absent from the base graph.

    Used to verify that add_resource raises AriadneError when neither column
    name is already present as a graph vertex.
    """
    df = pd.DataFrame({"alpha": ["a", "b"], "beta": ["c", "d"]})
    p = tmp_path / "both_new.csv"
    df.to_csv(p, index=False)
    return str(p)


# ── Return-type test ──────────────────────────────────────────────────────────


def test_add_resource_returns_igraph(base_graph, tmp_linkmap):
    """add_resource must return an igraph.Graph instance."""
    result = add_resource(base_graph, tmp_linkmap, res_name="TestDB")
    assert isinstance(result, ig.Graph)


# ── Edge addition tests ───────────────────────────────────────────────────────


def test_add_resource_adds_new_edge(base_graph, tmp_linkmap):
    """A new edge must be present in the returned graph."""
    before = base_graph.ecount()
    result = add_resource(base_graph, tmp_linkmap, res_name="TestDB")
    assert result.ecount() == before + 1


def test_add_resource_edge_has_source_attribute(base_graph, tmp_linkmap):
    """The newly added edge must carry the 'source' attribute set to res_name."""
    result = add_resource(base_graph, tmp_linkmap, res_name="TestDB")
    assert result.es[-1]["source"] == "TestDB"


# ── Node addition test ────────────────────────────────────────────────────────


def test_add_resource_adds_new_node(base_graph, tmp_linkmap):
    """The column name 'new_node' (absent from base_graph) must appear as a vertex."""
    before_names = set(base_graph.vs["name"])
    result = add_resource(base_graph, tmp_linkmap, res_name="TestDB")
    after_names = set(result.vs["name"])
    assert "new_node" in after_names - before_names


# ── Validation-gap test — both columns new ───────────────────────────────────


def test_add_resource_raises_when_both_columns_new(base_graph, both_new_linkmap):
    """add_resource must raise AriadneError when neither CSV column name is in
    the graph.

    This guards against silently grafting a disconnected subgraph — mirroring
    R's addResource check that at least one feature type must already exist.
    """
    with pytest.raises(AriadneError, match="At least one feature"):
        add_resource(base_graph, both_new_linkmap, res_name="Disconnected")


# ── One-column-known test ─────────────────────────────────────────────────────


def test_add_resource_succeeds_when_one_column_known(base_graph, tmp_linkmap):
    """add_resource must succeed (not raise) when one column name is already in
    the graph as a vertex — 'known_node' is pre-existing in base_graph."""
    result = add_resource(base_graph, tmp_linkmap, res_name="TestDB")
    assert "known_node" in result.vs["name"]


# ── Duplicate-edge guard (force=False) ───────────────────────────────────────


def test_add_resource_force_false_raises_on_duplicate(base_graph, tmp_linkmap):
    """Calling add_resource twice with force=False must raise AriadneError on the
    second call to prevent silently creating duplicate edges for the same resource.

    This test documents the intended validation behaviour; the implementation
    must enforce it.
    """
    result = add_resource(base_graph, tmp_linkmap, res_name="TestDB", force=False)
    with pytest.raises(AriadneError):
        add_resource(result, tmp_linkmap, res_name="TestDB", force=False)
