"""Tests for plot/_draw.py::plot_path.

All tests use a synthetic offline undirected graph — no network calls.
Matplotlib is forced into the non-interactive Agg backend so the suite
runs headlessly in CI.
"""
from __future__ import annotations

import igraph as ig
import matplotlib
import pytest

matplotlib.use("Agg")  # headless — must be set before importing pyplot

from matplotlib.figure import Figure

from ariadnepy.plot._draw import plot_path


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def simple_graph() -> ig.Graph:
    """Minimal undirected graph: A -- B -- C with a single resource label."""
    g = ig.Graph(directed=False)
    for n in ("A", "B", "C"):
        g.add_vertex(name=n)
    g.add_edge(0, 1)
    g.es[0]["source"] = "DB1"
    g.add_edge(1, 2)
    g.es[1]["source"] = "DB1"
    return g


# ── Return-type tests ─────────────────────────────────────────────────────────


def test_plot_path_no_by_returns_figure(simple_graph):
    """plot_path(graph) with no 'by' formula must return a matplotlib Figure."""
    fig = plot_path(simple_graph)
    assert isinstance(fig, Figure)


def test_plot_path_by_k1_returns_figure(simple_graph):
    """plot_path(graph, 'A ~ C', k=1) must return a matplotlib Figure."""
    fig = plot_path(simple_graph, "A ~ C", k=1)
    assert isinstance(fig, Figure)


def test_plot_path_prune_returns_figure(simple_graph):
    """plot_path with prune=True must return a matplotlib Figure."""
    fig = plot_path(simple_graph, "A ~ C", k=1, prune=True)
    assert isinstance(fig, Figure)


def test_plot_path_focus_returns_figure(simple_graph):
    """plot_path with focus=True must return a matplotlib Figure."""
    fig = plot_path(simple_graph, "A ~ C", k=1, focus=True)
    assert isinstance(fig, Figure)


# ── Title tests ───────────────────────────────────────────────────────────────


def test_plot_path_no_by_title_is_resource_graph(simple_graph):
    """When no path formula is given the figure title must be 'Resource Graph'."""
    fig = plot_path(simple_graph)
    ax = fig.axes[0]
    assert ax.get_title() == "Resource Graph"


def test_plot_path_by_k2_title_contains_path_2(simple_graph):
    """With k=2 the figure title must contain 'Path 2' (mirrors R plotPath behaviour)."""
    # Build a branching graph so two distinct shortest paths exist.
    g = ig.Graph(directed=False)
    for n in ("A", "M1", "M2", "C"):
        g.add_vertex(name=n)
    g.add_edge(g.vs.find(name="A").index, g.vs.find(name="M1").index)
    g.es[-1]["source"] = "DB1"
    g.add_edge(g.vs.find(name="M1").index, g.vs.find(name="C").index)
    g.es[-1]["source"] = "DB1"
    g.add_edge(g.vs.find(name="A").index, g.vs.find(name="M2").index)
    g.es[-1]["source"] = "DB1"
    g.add_edge(g.vs.find(name="M2").index, g.vs.find(name="C").index)
    g.es[-1]["source"] = "DB1"

    fig = plot_path(g, "A ~ C", k=2)
    ax = fig.axes[0]
    assert "Path 2" in ax.get_title()


# ── Validation tests ──────────────────────────────────────────────────────────


def test_plot_path_prune_without_by_raises(simple_graph):
    """prune=True without a 'by' formula must raise ValueError."""
    with pytest.raises(ValueError, match="prune"):
        plot_path(simple_graph, prune=True)


# ── Axes structure tests ──────────────────────────────────────────────────────


def test_plot_path_returns_exactly_one_axes(simple_graph):
    """The returned Figure must contain exactly one Axes object."""
    fig = plot_path(simple_graph)
    assert len(fig.axes) == 1


# ── R parity: res_name without 'by' (plotPath(graph, res.name = c(...))) ───────


@pytest.fixture
def two_resource_graph() -> ig.Graph:
    """A -[DB1]- B -[DB2]- C: two edges from different resources."""
    g = ig.Graph(directed=False)
    for n in ("A", "B", "C"):
        g.add_vertex(name=n)
    g.add_edge(0, 1)
    g.es[0]["source"] = "DB1"
    g.add_edge(1, 2)
    g.es[1]["source"] = "DB2"
    return g


def test_plot_path_res_name_without_by_fades_other_resources(two_resource_graph):
    """R: plotPath(graph, res.name = "DB1") fades the DB2 edge, not just DB1's."""
    fig = plot_path(two_resource_graph, res_name=["DB1"])
    ax = fig.axes[0]
    alphas = sorted(line.get_alpha() for line in ax.lines)
    assert alphas == pytest.approx([0.15, 1.0])


def test_plot_path_res_name_without_by_focus_crops_to_resource(two_resource_graph):
    """focus=True with only res_name (no 'by') must crop the graph structure
    itself down to just the DB1 edge — only 1 line, only A/B remain."""
    fig = plot_path(two_resource_graph, res_name=["DB1"], focus=True)
    ax = fig.axes[0]
    assert len(ax.lines) == 1
    assert len(ax.collections[0].get_offsets()) == 2  # only A, B


def test_plot_path_res_name_without_by_no_focus_keeps_edge_structure(two_resource_graph):
    """Without focus, both edges stay on the canvas (DB2 merely faded), but a
    node marker/label is only drawn for nodes that participate in DB1 — same
    as R's geom_node_point(aes(filter = alpha)), which removes non-alpha
    nodes from rendering regardless of 'focus'."""
    fig = plot_path(two_resource_graph, res_name=["DB1"])
    ax = fig.axes[0]
    assert len(ax.lines) == 2  # DB1 and DB2 edges both still drawn
    assert len(ax.collections[0].get_offsets()) == 2  # only A, B (not C)


# ── R parity: prune (fade) vs focus (crop) are independent axes ───────────────


@pytest.fixture
def branching_graph() -> ig.Graph:
    """A -[DB1]-> M1 -[DB1]-> C  and  A -[DB2]-> M2 -[DB2]-> C: two equal paths."""
    g = ig.Graph(directed=False)
    for n in ("A", "M1", "M2", "C"):
        g.add_vertex(name=n)
    g.add_edge(g.vs.find(name="A").index, g.vs.find(name="M1").index)
    g.es[-1]["source"] = "DB1"
    g.add_edge(g.vs.find(name="M1").index, g.vs.find(name="C").index)
    g.es[-1]["source"] = "DB1"
    g.add_edge(g.vs.find(name="A").index, g.vs.find(name="M2").index)
    g.es[-1]["source"] = "DB2"
    g.add_edge(g.vs.find(name="M2").index, g.vs.find(name="C").index)
    g.es[-1]["source"] = "DB2"
    return g


def test_plot_path_prune_fades_but_keeps_edges_without_focus(branching_graph):
    """R: prune=TRUE alone fades non-path edges but does not remove them.
    Node markers are still filtered to path-connected nodes only (A, M1, C) —
    M2 has no alpha=TRUE edge, so its marker/label is dropped even though its
    faded edges remain drawn, matching geom_node_point's filter aesthetic."""
    fig = plot_path(branching_graph, "A ~ C", k=1, prune=True)
    ax = fig.axes[0]
    assert len(ax.lines) == 4  # all 4 edges still drawn
    assert len(ax.collections[0].get_offsets()) == 3  # A, M1, C (not M2)
    alphas = sorted(line.get_alpha() for line in ax.lines)
    assert alphas == pytest.approx([0.15, 0.15, 1.0, 1.0])


def test_plot_path_focus_alone_is_a_noop_without_prune_or_res_name(branching_graph):
    """R quirk: focus=TRUE with prune=FALSE and no res_name fades nothing, so
    there is nothing to crop — the full graph is still shown."""
    fig = plot_path(branching_graph, "A ~ C", k=1, focus=True)
    ax = fig.axes[0]
    assert len(ax.lines) == 4
    assert len(ax.collections[0].get_offsets()) == 4


def test_plot_path_prune_and_focus_together_crop_to_path_only(branching_graph):
    """focus=True structurally drops the 2 non-path edges, leaving the 2-edge
    path A-M1-C — 3 nodes, not 2, since a 2-edge path has 3 stops."""
    fig = plot_path(branching_graph, "A ~ C", k=1, prune=True, focus=True)
    ax = fig.axes[0]
    assert len(ax.lines) == 2
    assert len(ax.collections[0].get_offsets()) == 3  # A, M1, C
