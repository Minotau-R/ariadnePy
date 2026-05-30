"""Tests for core/_download.py — GML downloading, parsing, version insertion."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import igraph as ig
import pytest

from ariadnepy.exceptions import AriadneDownloadError, AriadneParseError


# ── ensure_directory ──────────────────────────────────────────────────────────

def test_ensure_directory_creates_dir(tmp_path):
    from ariadnepy.core._download import ensure_directory
    target = tmp_path / "new" / "nested"
    ensure_directory(target)
    assert target.exists()


def test_ensure_directory_returns_path(tmp_path):
    from ariadnepy.core._download import ensure_directory
    result = ensure_directory(tmp_path / "sub")
    assert isinstance(result, Path)


def test_ensure_directory_idempotent(tmp_path):
    from ariadnepy.core._download import ensure_directory
    ensure_directory(tmp_path)
    ensure_directory(tmp_path)  # calling twice should not raise


# ── download_gml ──────────────────────────────────────────────────────────────

def test_download_gml_skips_if_cached(tmp_path):
    """If the file already exists and is non-empty, no download should happen."""
    cached = tmp_path / "KEGG.gml"
    cached.write_text("graph []")  # non-empty existing file
    with patch("ariadnepy.core._download._requests") as mock_req:
        from ariadnepy.core._download import download_gml
        result = download_gml("https://example.com/KEGG.gml", tmp_path)
    mock_req.get.assert_not_called()
    assert result == cached


def test_download_gml_downloads_when_missing(tmp_path):
    mock_resp = MagicMock()
    mock_resp.iter_content.return_value = [b"graph []"]
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    with patch("ariadnepy.core._download._requests") as mock_req:
        mock_req.get.return_value = mock_resp
        from ariadnepy.core._download import download_gml
        result = download_gml("https://example.com/KEGG.gml", tmp_path)
    assert result == tmp_path / "KEGG.gml"


def test_download_gml_raises_on_http_error(tmp_path):
    import urllib.error
    with patch("ariadnepy.core._download._requests", None), \
         patch("ariadnepy.core._download.urllib.request.urlopen",
               side_effect=urllib.error.URLError("connection refused")):
        from ariadnepy.core._download import download_gml
        with pytest.raises(AriadneDownloadError):
            download_gml("https://example.com/missing.gml", tmp_path)


# ── read_gml ──────────────────────────────────────────────────────────────────

def test_read_gml_parses_valid_file(tmp_path):
    gml = tmp_path / "test.gml"
    gml.write_text('graph [\n  directed 1\n  node [ id 0 name "ko" ]\n]\n')
    from ariadnepy.core._download import read_gml
    g = read_gml(gml)
    assert isinstance(g, ig.Graph)


def test_read_gml_missing_file_raises(tmp_path):
    from ariadnepy.core._download import read_gml
    with pytest.raises(AriadneDownloadError, match="not found"):
        read_gml(tmp_path / "nonexistent.gml")


def test_read_gml_invalid_content_raises(tmp_path):
    bad = tmp_path / "bad.gml"
    bad.write_text("this is not valid GML content !!!")
    from ariadnepy.core._download import read_gml
    with pytest.raises(AriadneParseError):
        read_gml(bad)


# ── insert_version ────────────────────────────────────────────────────────────

def test_insert_version_replaces_node_attr():
    g = ig.Graph(directed=True)
    g.add_vertex(name="ko", url="https://example.com/{version}/ko.parquet")
    from ariadnepy.core._download import insert_version
    insert_version(g, "v2")
    assert g.vs.find(name="ko")["url"] == "https://example.com/v2/ko.parquet"


def test_insert_version_replaces_edge_attr():
    g = ig.Graph(directed=True)
    g.add_vertex(name="ko")
    g.add_vertex(name="ec")
    g.add_edge(0, 1)
    g.es[0]["url"] = "https://example.com/{version}/ko2ec.parquet"
    from ariadnepy.core._download import insert_version
    insert_version(g, "v2")
    assert g.es[0]["url"] == "https://example.com/v2/ko2ec.parquet"


def test_insert_version_leaves_non_template_attrs_unchanged():
    g = ig.Graph(directed=True)
    g.add_vertex(name="ko", url="https://example.com/static.parquet")
    from ariadnepy.core._download import insert_version
    insert_version(g, "v2")
    assert g.vs.find(name="ko")["url"] == "https://example.com/static.parquet"
    assert g.vs.find(name="ko")["name"] == "ko"
