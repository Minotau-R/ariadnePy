"""Tests for graph/_weave.py — mirrors R's test-weave.R."""
from __future__ import annotations

import importlib.util
from unittest.mock import patch

import igraph as ig
import pandas as pd
import pytest

from ariadnepy.exceptions import AriadneError
from ariadnepy.graph._weave import (
    _draw_path,
    _map_complex_modules,
    _parse_by,
    _process_complex_modules,
    draw_path,
    weave_complex,
    weave_path,
)

needs_scipy = pytest.mark.skipif(
    importlib.util.find_spec("scipy") is None,
    reason="scipy not installed",
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _ig_build(nodes, edges):
    """Build a directed igraph. edges = [(src, tgt, {attr: val, ...})]."""
    g = ig.Graph(directed=True)
    for n in nodes:
        g.add_vertex(name=n)
    for src, tgt, attrs in edges:
        g.add_edge(g.vs.find(name=src).index, g.vs.find(name=tgt).index)
        for k, v in attrs.items():
            g.es[-1][k] = v
    return g


@pytest.fixture
def linear_graph():
    """A -[DB1]-> B -[DB1]-> C -[DB1]-> D  (exactly one path)."""
    return _ig_build(
        ["A", "B", "C", "D"],
        [("A", "B", {"source": "DB1", "url": None}),
         ("B", "C", {"source": "DB1", "url": None}),
         ("C", "D", {"source": "DB1", "url": None})],
    )


@pytest.fixture
def branching_graph():
    """
    A -[DB1]-> B -[DB1]-> D
    A -[DB2]-> C -[DB2]-> D
    Two equal-length paths — both igraph (Python) and R use the same C-level Yen's.
    """
    return _ig_build(
        ["A", "B", "C", "D"],
        [("A", "B", {"source": "DB1", "url": None}),
         ("B", "D", {"source": "DB1", "url": None}),
         ("A", "C", {"source": "DB2", "url": None}),
         ("C", "D", {"source": "DB2", "url": None})],
    )


@pytest.fixture
def include_exclude_graph():
    """
    A -[DB1]-> B -[DB1]-> D
    A -[DB1]-> C -[DB1]-> D
    Mirrors R test: .draw_path(graph, bugsig~ko, 1, 'uniref90', 'taxid', NULL)
    """
    return _ig_build(
        ["A", "B", "C", "D"],
        [("A", "B", {"source": "DB1", "url": None}),
         ("B", "D", {"source": "DB1", "url": None}),
         ("A", "C", {"source": "DB1", "url": None}),
         ("C", "D", {"source": "DB1", "url": None})],
    )


def _file_graph():
    """Single-step ko → ec graph with a file-backend edge."""
    return _ig_build(
        ["ko", "ec"],
        [("ko", "ec", {"source": "FileDB", "url": "dummy_url"})],
    )


@pytest.fixture
def gmm_graph(tmp_path):
    """Graph for weave_complex tests with a real GMM file."""
    gmm_file = tmp_path / "test.gmm"
    gmm_file.write_text(
        "M001\tModule 1\nK00001,K00002\tK00003\n///\nM002\tModule 2\nK00004\n///\n",
        encoding="utf-8",
    )
    return _ig_build(
        ["ec", "gmm", "ko"],
        [("ec",  "ko", {"source": "FileDB", "url": "dummy_url"}),
         ("gmm", "ko", {"source": "GMM",    "url": str(gmm_file)})],
    )


# ── _parse_by ─────────────────────────────────────────────────────────────────


def test_parse_by_valid():
    assert _parse_by("ko ~ ec") == ("ko", "ec")
    assert _parse_by("taxname ~ bugsig") == ("taxname", "bugsig")
    assert _parse_by(" ko  ~  ec ") == ("ko", "ec")


def test_parse_by_missing_tilde():
    with pytest.raises(AriadneError):
        _parse_by("ko")


def test_parse_by_three_parts():
    with pytest.raises(AriadneError):
        _parse_by("ko ~ ec ~ gmm")


def test_parse_by_empty_side():
    with pytest.raises(AriadneError):
        _parse_by("~ ec")


# ── igraph graph properties ───────────────────────────────────────────────────

def test_graph_is_igraph(linear_graph):
    assert isinstance(linear_graph, ig.Graph)


def test_graph_vertex_count(linear_graph):
    assert linear_graph.vcount() == 4


def test_graph_vertex_names(linear_graph):
    assert set(linear_graph.vs["name"]) == {"A", "B", "C", "D"}


def test_graph_edge_count(linear_graph):
    assert linear_graph.ecount() == 3


def test_graph_is_directed(linear_graph):
    assert linear_graph.is_directed()


# ── _draw_path — basic path traversal ────────────────────────────────────────


def test_draw_path_k1_linear_nodes(linear_graph):
    df = _draw_path(linear_graph, "A", "D", k=1, include=[], exclude=[], res_name=None)
    assert list(df["from"]) == ["A", "B", "C"]
    assert list(df["to"]) == ["B", "C", "D"]


def test_draw_path_k1_returns_dataframe(branching_graph):
    df = _draw_path(branching_graph, "A", "D", k=1, include=[], exclude=[], res_name=None)
    assert isinstance(df, pd.DataFrame)
    assert {"from", "to", "source"} <= set(df.columns)


def test_draw_path_k1_starts_and_ends_correctly(branching_graph):
    df = _draw_path(branching_graph, "A", "D", k=1, include=[], exclude=[], res_name=None)
    assert df.iloc[0]["from"] == "A"
    assert df.iloc[-1]["to"] == "D"


def test_draw_path_k2_differs_from_k1(branching_graph):
    """Two equal-length paths exist — k=1 and k=2 must return different ones."""
    df1 = _draw_path(branching_graph, "A", "D", k=1, include=[], exclude=[], res_name=None)
    df2 = _draw_path(branching_graph, "A", "D", k=2, include=[], exclude=[], res_name=None)
    assert list(df1["to"]) != list(df2["to"])


def test_draw_path_matches_igraph_directly(branching_graph):
    """_draw_path k=1 must agree with igraph.get_k_shortest_paths directly.
    Graph is already igraph — same C-level algorithm as R's igraph.
    """
    node_to_idx = {v["name"]: v.index for v in branching_graph.vs}
    raw = branching_graph.get_k_shortest_paths(
        node_to_idx["A"], to=node_to_idx["D"], k=1, mode="all", output="vpath"
    )
    expected = [branching_graph.vs[i]["name"] for i in raw[0]]

    df = _draw_path(branching_graph, "A", "D", k=1, include=[], exclude=[], res_name=None)
    actual = [df.iloc[0]["from"]] + list(df["to"])
    assert actual == expected


# ── _draw_path — include / exclude (mirrors R test-weave.R) ──────────────────


def test_draw_path_include_forces_node_in_path(include_exclude_graph):
    """R: .draw_path(graph, bugsig~ko, 1, include='uniref90', exclude='taxid')
       expect_true('uniref90' %in% path_df$from || 'uniref90' %in% path_df$to)
    """
    df = _draw_path(
        include_exclude_graph, "A", "D", k=1,
        include=["B"], exclude=["C"], res_name=None,
    )
    all_nodes = set(df["from"]) | set(df["to"])
    assert "B" in all_nodes


def test_draw_path_exclude_removes_node_from_path(include_exclude_graph):
    """R: expect_false('taxid' %in% path_df$from || 'taxid' %in% path_df$to)"""
    df = _draw_path(
        include_exclude_graph, "A", "D", k=1,
        include=["B"], exclude=["C"], res_name=None,
    )
    all_nodes = set(df["from"]) | set(df["to"])
    assert "C" not in all_nodes


def test_draw_path_overlapping_include_exclude_raises(include_exclude_graph):
    """R: expect_error(.draw_path(..., 'uniref90', 'uniref90'), 'cannot overlap')"""
    with pytest.raises(AriadneError, match="cannot overlap"):
        _draw_path(
            include_exclude_graph, "A", "D", k=1,
            include=["B"], exclude=["B"], res_name=None,
        )


def test_draw_path_exclude_only(include_exclude_graph):
    df = _draw_path(
        include_exclude_graph, "A", "D", k=1,
        include=[], exclude=["C"], res_name=None,
    )
    assert "C" not in (set(df["from"]) | set(df["to"]))


# ── _draw_path — error handling ───────────────────────────────────────────────


def test_draw_path_k_zero_raises(linear_graph):
    with pytest.raises(AriadneError, match="positive integer"):
        _draw_path(linear_graph, "A", "D", k=0, include=[], exclude=[], res_name=None)


def test_draw_path_k_negative_raises(linear_graph):
    with pytest.raises(AriadneError, match="positive integer"):
        _draw_path(linear_graph, "A", "D", k=-1, include=[], exclude=[], res_name=None)


def test_draw_path_missing_source_raises(linear_graph):
    with pytest.raises(AriadneError, match="not found in graph"):
        _draw_path(linear_graph, "Z", "D", k=1, include=[], exclude=[], res_name=None)


def test_draw_path_missing_target_raises(linear_graph):
    with pytest.raises(AriadneError, match="not found in graph"):
        _draw_path(linear_graph, "A", "Z", k=1, include=[], exclude=[], res_name=None)


def test_draw_path_k_exceeds_paths_raises(branching_graph):
    """Only 2 paths exist between A and D; k=3 must raise."""
    with pytest.raises(AriadneError, match="greater than the number"):
        _draw_path(branching_graph, "A", "D", k=3, include=[], exclude=[], res_name=None)


def test_draw_path_include_both_branches_raises(include_exclude_graph):
    """Requiring B and C simultaneously is impossible — both are on different branches."""
    with pytest.raises(AriadneError):
        _draw_path(
            include_exclude_graph, "A", "D", k=1,
            include=["B", "C"], exclude=[], res_name=None,
        )


# ── draw_path (public reproducibility table) ─────────────────────────────────


def test_draw_path_public_columns(linear_graph):
    """draw_path returns the reproducibility table with required columns."""
    df = draw_path(linear_graph, "A ~ D")
    assert {"from", "to", "source", "version", "url"} <= set(df.columns)


def test_draw_path_public_k2_differs(branching_graph):
    """draw_path k=2 returns a different path than k=1."""
    df1 = draw_path(branching_graph, "A ~ D", k=1)
    df2 = draw_path(branching_graph, "A ~ D", k=2)
    assert list(df1["to"]) != list(df2["to"])


# ── weave_path ────────────────────────────────────────────────────────────────

_MOCK_LM_KO_EC = pd.DataFrame({
    "ko": pd.Categorical(["K00001", "K00002"]),
    "ec": pd.Categorical(["1.1.1.1", "1.1.1.2"]),
})


def test_weave_path_returns_two_columns():
    """R: ko2gbm <- weavePath(graph, ko~gbm, use.names=FALSE); expect_identical(ncol(ko2gbm), 2L)"""
    g = _file_graph()
    with patch("ariadnepy.graph._weave._fetch_edge", return_value=_MOCK_LM_KO_EC):
        result = weave_path(g, "ko ~ ec", use_names=False, verbose=False)
    assert result.shape[1] == 2


def test_weave_path_column_names_match_formula():
    g = _file_graph()
    with patch("ariadnepy.graph._weave._fetch_edge", return_value=_MOCK_LM_KO_EC):
        result = weave_path(g, "ko ~ ec", use_names=False, verbose=False)
    assert list(result.columns) == ["ko", "ec"]


def test_weave_path_returns_dataframe():
    g = _file_graph()
    with patch("ariadnepy.graph._weave._fetch_edge", return_value=_MOCK_LM_KO_EC):
        result = weave_path(g, "ko ~ ec", use_names=False, verbose=False)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_weave_path_invalid_by_formula_raises():
    g = _file_graph()
    with pytest.raises(AriadneError):
        weave_path(g, "ko", verbose=False)


def test_weave_path_missing_node_raises():
    g = _file_graph()
    with pytest.raises(AriadneError):
        weave_path(g, "ko ~ unknown_node", verbose=False)


def test_weave_path_init_filters_rows():
    """init is forwarded to _fetch_edge, which filters rows to those seed IDs."""
    g = _file_graph()
    def _mock_fetch(_step, init, *_, **__):
        df = _MOCK_LM_KO_EC.copy()
        if init is not None:
            mask = df["ko"].isin(set(init))
            df = df[mask].reset_index(drop=True)
        return df
    with patch("ariadnepy.graph._weave._fetch_edge", side_effect=_mock_fetch):
        result = weave_path(g, "ko ~ ec", init=["K00001"], use_names=False, verbose=False)
    assert set(result["ko"]) == {"K00001"}


# ── weave_complex ─────────────────────────────────────────────────────────────

_MOCK_LM_EC_KO = pd.DataFrame({
    "ec": pd.Categorical(["1.1.1.1", "1.1.1.1", "1.1.1.2"]),
    "ko": pd.Categorical(["K00001",  "K00002",  "K00004"]),
})


def test_weave_complex_threshold_above_one_raises(gmm_graph):
    """Threshold validation runs before scipy — no scipy needed."""
    with pytest.raises(AriadneError, match="threshold"):
        weave_complex(gmm_graph, "ec ~ gmm", threshold=1.5, verbose=False)


def test_weave_complex_threshold_zero_raises(gmm_graph):
    with pytest.raises(AriadneError, match="threshold"):
        weave_complex(gmm_graph, "ec ~ gmm", threshold=0.0, verbose=False)


@needs_scipy
def test_weave_complex_has_required_columns(gmm_graph):
    """R: expect_identical(colnames(ec2gmm), c('ec','gmm','cov','gmm.name'))
    Core three columns must always be present; gmm.name appears when link_names succeeds.
    """
    with patch("ariadnepy.graph._weave._fetch_edge", return_value=_MOCK_LM_EC_KO):
        result = weave_complex(gmm_graph, "ec ~ gmm", use_names=False, verbose=False)
    assert {"ec", "gmm", "cov"} <= set(result.columns)


@needs_scipy
def test_weave_complex_coverage_bounded(gmm_graph):
    """Coverage must be in (0, 1]."""
    with patch("ariadnepy.graph._weave._fetch_edge", return_value=_MOCK_LM_EC_KO):
        result = weave_complex(gmm_graph, "ec ~ gmm", use_names=False, verbose=False)
    assert (result["cov"] > 0).all()
    assert (result["cov"] <= 1).all()


@needs_scipy
def test_weave_complex_known_coverage_values(gmm_graph):
    """
    1.1.1.1 has K00001+K00002 → covers M001 complex {K00001,K00002} → cov=1.0
    1.1.1.2 has K00004        → covers M002 complex {K00004}         → cov=1.0
    """
    with patch("ariadnepy.graph._weave._fetch_edge", return_value=_MOCK_LM_EC_KO):
        result = weave_complex(gmm_graph, "ec ~ gmm", use_names=False, verbose=False)
    row_111 = result[result["ec"] == "1.1.1.1"]
    row_112 = result[result["ec"] == "1.1.1.2"]
    assert not row_111.empty
    assert not row_112.empty
    assert float(row_111["cov"].iloc[0]) == pytest.approx(1.0)
    assert float(row_112["cov"].iloc[0]) == pytest.approx(1.0)


@needs_scipy
def test_weave_complex_threshold_filters_rows(gmm_graph):
    """Only rows with cov >= threshold must be returned."""
    with patch("ariadnepy.graph._weave._fetch_edge", return_value=_MOCK_LM_EC_KO):
        result = weave_complex(
            gmm_graph, "ec ~ gmm", threshold=0.9, use_names=False, verbose=False
        )
    assert (result["cov"] >= 0.9).all()


# ── _process_complex_modules ──────────────────────────────────────────────────


def test_process_complex_modules_keys(tmp_path):
    f = tmp_path / "m.gmm"
    f.write_text("M001\tModule 1\nK00001,K00002\n///\n", encoding="utf-8")
    result = _process_complex_modules(str(f))
    assert set(result.keys()) == {"module2component", "component2complex", "complex2feature"}


def test_process_complex_modules_module_present(tmp_path):
    f = tmp_path / "m.gmm"
    f.write_text("M001\tModule 1\nK00001,K00002\n///\n", encoding="utf-8")
    result = _process_complex_modules(str(f))
    assert "M001" in result["module2component"]["module"].values


def test_process_complex_modules_features(tmp_path):
    f = tmp_path / "m.gmm"
    f.write_text("M001\tModule 1\nK00001,K00002\n///\n", encoding="utf-8")
    result = _process_complex_modules(str(f))
    assert {"K00001", "K00002"} <= set(result["complex2feature"]["feature"])


def test_process_complex_modules_multiple_modules(tmp_path):
    f = tmp_path / "m.gmm"
    f.write_text("M001\tA\nK00001\n///\nM002\tB\nK00002\n///\n", encoding="utf-8")
    result = _process_complex_modules(str(f))
    assert set(result["module2component"]["module"]) == {"M001", "M002"}


# ── _map_complex_modules ──────────────────────────────────────────────────────


@pytest.fixture
def complex_linkmaps():
    """
    Module M001, one component, one complex {K00001, K00002}.
    org1 has both features → full coverage (1.0).
    org2 has only K00001   → complex not covered (absent from output).
    """
    return {
        "origin2feature": pd.DataFrame({
            "origin":  ["org1", "org1", "org2"],
            "feature": ["K00001", "K00002", "K00001"],
        }),
        "module2component": pd.DataFrame({
            "module":    ["M001"],
            "component": ["M001_part_1"],
        }),
        "component2complex": pd.DataFrame({
            "component": ["M001_part_1"],
            "complex":   ["K00001,K00002"],
        }),
        "complex2feature": pd.DataFrame({
            "complex": ["K00001,K00002", "K00001,K00002"],
            "feature": ["K00001", "K00002"],
        }),
    }


@needs_scipy
def test_map_complex_full_coverage(complex_linkmaps):
    result = _map_complex_modules(complex_linkmaps, "origin", "feature")
    org1 = result[result["origin"] == "org1"]
    assert not org1.empty
    assert float(org1["cov"].iloc[0]) == pytest.approx(1.0)


@needs_scipy
def test_map_complex_partial_coverage_absent(complex_linkmaps):
    """org2 only has K00001 — complex not fully covered — not in output."""
    result = _map_complex_modules(complex_linkmaps, "origin", "feature")
    org2 = result[result["origin"] == "org2"]
    assert org2.empty or float(org2["cov"].iloc[0]) == pytest.approx(0.0)


@needs_scipy
def test_map_complex_output_columns(complex_linkmaps):
    result = _map_complex_modules(complex_linkmaps, "origin", "feature")
    assert {"origin", "module", "cov"} <= set(result.columns)


def test_map_complex_missing_table_raises():
    """Missing required linkmap tables raises before scipy is touched."""
    with pytest.raises(AriadneError, match="Missing required"):
        _map_complex_modules({"origin2feature": pd.DataFrame()}, "origin", "feature")
