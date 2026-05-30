"""Tests for graph/_names.py::link_names.

Network calls are mocked — all tests run offline.
"""
from __future__ import annotations

import warnings
from unittest.mock import patch

import igraph as ig
import pandas as pd
import pytest

from ariadnepy.exceptions import AriadneError


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _graph_with_node(name: str, **attrs) -> ig.Graph:
    g = ig.Graph(directed=True)
    g.add_vertex(name=name, **attrs)
    return g


_MOCK_NAMES = pd.DataFrame({
    "ids":   ["K00001", "K00002", "K00003"],
    "names": ["alcohol dehydrogenase", "catalase", "glucokinase"],
})


# ── Argument validation ───────────────────────────────────────────────────────

def test_link_names_both_ids_and_names_raises():
    from ariadnepy.graph._names import link_names
    g = _graph_with_node("ko")
    with pytest.raises(AriadneError, match="only 'ids' or 'names'"):
        link_names(g, "ko", ids=["K00001"], names=["alcohol dehydrogenase"])


def test_link_names_unknown_node_raises():
    from ariadnepy.graph._names import link_names
    g = _graph_with_node("ko")
    with pytest.raises(AriadneError, match="not found in graph"):
        link_names(g, "ec")


# ── Return value structure ────────────────────────────────────────────────────

def test_link_names_returns_dataframe():
    from ariadnepy.graph._names import link_names
    g = _graph_with_node("ko")
    with patch("ariadnepy.graph._names._fetch_node_names", side_effect=lambda *_: _MOCK_NAMES.copy()):
        result = link_names(g, "ko")
    assert isinstance(result, pd.DataFrame)


def test_link_names_columns_named_after_node():
    from ariadnepy.graph._names import link_names
    g = _graph_with_node("ko")
    with patch("ariadnepy.graph._names._fetch_node_names", side_effect=lambda *_: _MOCK_NAMES.copy()):
        result = link_names(g, "ko")
    assert result is not None
    assert list(result.columns) == ["ko", "ko.name"]


def test_link_names_no_filter_returns_all():
    from ariadnepy.graph._names import link_names
    g = _graph_with_node("ko")
    with patch("ariadnepy.graph._names._fetch_node_names", side_effect=lambda *_: _MOCK_NAMES.copy()):
        result = link_names(g, "ko")
    assert result is not None
    assert len(result) == 3


# ── ids filtering ─────────────────────────────────────────────────────────────

def test_link_names_ids_filters_rows():
    from ariadnepy.graph._names import link_names
    g = _graph_with_node("ko")
    with patch("ariadnepy.graph._names._fetch_node_names", side_effect=lambda *_: _MOCK_NAMES.copy()):
        result = link_names(g, "ko", ids=["K00001"])
    assert result is not None
    assert len(result) == 1
    assert result.iloc[0]["ko"] == "K00001"
    assert result.iloc[0]["ko.name"] == "alcohol dehydrogenase"


def test_link_names_ids_missing_entry_is_none():
    from ariadnepy.graph._names import link_names
    g = _graph_with_node("ko")
    with patch("ariadnepy.graph._names._fetch_node_names", side_effect=lambda *_: _MOCK_NAMES.copy()):
        result = link_names(g, "ko", ids=["K00001", "K99999"], verbose=False)
    assert result is not None
    assert result.iloc[1]["ko.name"] is None


def test_link_names_ids_missing_warns_when_verbose():
    from ariadnepy.graph._names import link_names
    g = _graph_with_node("ko")
    with patch("ariadnepy.graph._names._fetch_node_names", side_effect=lambda *_: _MOCK_NAMES.copy()):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            link_names(g, "ko", ids=["K99999"], verbose=True)
    assert any("not found" in str(warning.message).lower() for warning in w)


# ── names filtering ───────────────────────────────────────────────────────────

def test_link_names_names_filters_rows():
    from ariadnepy.graph._names import link_names
    g = _graph_with_node("ko")
    with patch("ariadnepy.graph._names._fetch_node_names", side_effect=lambda *_: _MOCK_NAMES.copy()):
        result = link_names(g, "ko", names=["catalase"])
    assert result is not None
    assert len(result) == 1
    assert result.iloc[0]["ko"] == "K00002"


def test_link_names_names_missing_entry_is_none():
    from ariadnepy.graph._names import link_names
    g = _graph_with_node("ko")
    with patch("ariadnepy.graph._names._fetch_node_names", side_effect=lambda *_: _MOCK_NAMES.copy()):
        result = link_names(g, "ko", names=["nonexistent enzyme"], verbose=False)
    assert result is not None
    assert result.iloc[0]["ko"] is None


# ── None backend ──────────────────────────────────────────────────────────────

def test_link_names_returns_none_when_no_backend():
    """Nodes with no name backend (e.g. taxid) should return None."""
    from ariadnepy.graph._names import link_names
    g = _graph_with_node("taxid")
    with patch("ariadnepy.graph._names._fetch_node_names", return_value=None):
        result = link_names(g, "taxid")
    assert result is None


# ── _match_key2val internals ──────────────────────────────────────────────────

def test_match_key2val_no_init_returns_unchanged():
    from ariadnepy.graph._names import _match_key2val
    lm = _MOCK_NAMES.copy()
    result = _match_key2val(lm, "ko", 1, None, verbose=False)
    pd.testing.assert_frame_equal(result, lm)


def test_match_key2val_filters_by_id():
    from ariadnepy.graph._names import _match_key2val
    result = _match_key2val(_MOCK_NAMES.copy(), "ko", 1, ["K00001"], verbose=False)
    assert result.iloc[0]["ids"] == "K00001"
    assert result.iloc[0]["names"] == "alcohol dehydrogenase"


def test_match_key2val_filters_by_name():
    from ariadnepy.graph._names import _match_key2val
    result = _match_key2val(_MOCK_NAMES.copy(), "ko", 2, ["catalase"], verbose=False)
    assert result.iloc[0]["ids"] == "K00002"
