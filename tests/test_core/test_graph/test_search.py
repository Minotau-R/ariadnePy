"""Tests for graph/_weave.py::search_path (and draw_path public API).

Uses synthetic offline graphs — no network calls.
"""
from __future__ import annotations

import igraph as ig
import pandas as pd
import pytest

from ariadnepy.exceptions import AriadneError


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _file_graph() -> ig.Graph:
    """Minimal linear graph: ko -KEGG-> ec -KEGG-> pathway (no back-edges)."""
    g = ig.Graph(directed=True)
    for n in ("ko", "ec", "pathway"):
        g.add_vertex(name=n)
    g.add_edge(g.vs.find(name="ko").index, g.vs.find(name="ec").index)
    g.es[0]["source"] = "KEGG"
    g.es[0]["url"] = ""
    g.es[0]["from_"] = "ko"
    g.es[0]["to"] = "ec"
    g.add_edge(g.vs.find(name="ec").index, g.vs.find(name="pathway").index)
    g.es[1]["source"] = "KEGG"
    g.es[1]["url"] = ""
    g.es[1]["from_"] = "ec"
    g.es[1]["to"] = "pathway"
    return g


# ── search_path ───────────────────────────────────────────────────────────────

def test_search_path_returns_none():
    """search_path prints to console and returns None (like R's invisible(NULL))."""
    from ariadnepy.graph._weave import search_path
    g = _file_graph()
    result = search_path(g, "ko ~ ec", k=1, )
    assert result is None


def test_search_path_prints_output(capsys):
    from ariadnepy.graph._weave import search_path
    g = _file_graph()
    search_path(g, "ko ~ ec", k=1, )
    captured = capsys.readouterr()
    assert "Path 1" in captured.out


def test_search_path_shows_path_nodes(capsys):
    from ariadnepy.graph._weave import search_path
    g = _file_graph()
    search_path(g, "ko ~ ec", k=1, )
    out = capsys.readouterr().out
    assert "ko" in out
    assert "ec" in out


def test_search_path_k2_prints_two_paths(capsys):
    """With k=2 and a branching graph, two path blocks should be printed."""
    g = ig.Graph(directed=True)
    for n in ("ko", "mid1", "mid2", "ec"):
        g.add_vertex(name=n)
    g.add_edge(g.vs.find(name="ko").index,   g.vs.find(name="mid1").index)
    g.add_edge(g.vs.find(name="ko").index,   g.vs.find(name="mid2").index)
    g.add_edge(g.vs.find(name="mid1").index, g.vs.find(name="ec").index)
    g.add_edge(g.vs.find(name="mid2").index, g.vs.find(name="ec").index)
    for e in g.es:
        e["source"] = "KEGG"
        e["url"] = ""

    from ariadnepy.graph._weave import search_path
    search_path(g, "ko ~ ec", k=2, )
    out = capsys.readouterr().out
    assert "Path 1" in out
    assert "Path 2" in out


def test_search_path_invalid_formula_raises():
    from ariadnepy.graph._weave import search_path
    g = _file_graph()
    with pytest.raises(AriadneError):
        search_path(g, "ko", k=1)


def test_search_path_unknown_node_prints_error(capsys):
    """search_path prints an error message (like R's message()) instead of raising."""
    from ariadnepy.graph._weave import search_path
    g = _file_graph()
    search_path(g, "ko ~ unknown", k=1)
    out = capsys.readouterr().out
    assert "unknown" in out.lower() or "not found" in out.lower() or "path 1" in out.lower()


# ── draw_path (public API) ────────────────────────────────────────────────────

def test_draw_path_returns_dataframe():
    from ariadnepy.graph._weave import draw_path
    g = _file_graph()
    df = draw_path(g, "ko ~ ec", k=1)
    assert isinstance(df, pd.DataFrame)


def test_draw_path_has_required_columns():
    from ariadnepy.graph._weave import draw_path
    g = _file_graph()
    df = draw_path(g, "ko ~ ec", k=1)
    for col in ("from", "to", "source"):
        assert col in df.columns


def test_draw_path_starts_at_from_node():
    from ariadnepy.graph._weave import draw_path
    g = _file_graph()
    df = draw_path(g, "ko ~ ec", k=1)
    assert df.iloc[0]["from"] == "ko"


def test_draw_path_ends_at_to_node():
    from ariadnepy.graph._weave import draw_path
    g = _file_graph()
    df = draw_path(g, "ko ~ ec", k=1)
    assert df.iloc[-1]["to"] == "ec"


def test_draw_path_multi_hop():
    """ko → ec → pathway should produce 2 rows."""
    from ariadnepy.graph._weave import draw_path
    g = _file_graph()
    df = draw_path(g, "ko ~ pathway", k=1)
    assert len(df) == 2


def test_draw_path_version_column_present():
    """draw_path adds version and url columns from graph metadata."""
    from ariadnepy.graph._weave import draw_path
    g = _file_graph()
    g["versions"] = {"KEGG": "latest"}
    df = draw_path(g, "ko ~ ec", k=1)
    assert "version" in df.columns
