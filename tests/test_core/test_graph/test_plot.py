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
