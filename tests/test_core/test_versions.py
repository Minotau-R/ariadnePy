"""Tests for core/_versions.py — version metadata loading and resolution."""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from ariadnepy.exceptions import AriadneVersionError


# ── _load_from_bundled_json ───────────────────────────────────────────────────

def test_bundled_json_returns_dataframe():
    from ariadnepy.core._versions import _load_from_bundled_json
    df = _load_from_bundled_json()
    assert isinstance(df, pd.DataFrame)


def test_bundled_json_has_required_columns():
    from ariadnepy.core._versions import _load_from_bundled_json
    df = _load_from_bundled_json()
    for col in ("source", "version", "key", "graph", "default"):
        assert col in df.columns


def test_bundled_json_has_rows():
    from ariadnepy.core._versions import _load_from_bundled_json
    df = _load_from_bundled_json()
    assert len(df) > 0


def test_bundled_json_has_default_versions():
    from ariadnepy.core._versions import _load_from_bundled_json
    df = _load_from_bundled_json()
    assert df["default"].any()


def test_bundled_json_has_multiple_sources():
    """At least several distinct sources must be present — not tied to specific names."""
    from ariadnepy.core._versions import _load_from_bundled_json
    df = _load_from_bundled_json()
    assert df["source"].nunique() >= 5


def test_bundled_json_sources_match_sysdata():
    """Bundled JSON sources should match what's in ariadne's sysdata.rda."""
    from ariadnepy.core._versions import _load_from_bundled_json, _load_from_rda
    json_sources = set(_load_from_bundled_json()["source"])
    rda = _load_from_rda()
    if rda is not None:
        rda_sources = set(rda["source"])
        assert json_sources == rda_sources, (
            f"versions.json and sysdata.rda have different sources.\n"
            f"JSON only: {json_sources - rda_sources}\n"
            f"RDA only:  {rda_sources - json_sources}"
        )


# ── load_version_metadata ─────────────────────────────────────────────────────

def test_load_version_metadata_returns_dataframe():
    """When RDA fetch fails, falls back to bundled JSON."""
    with patch("ariadnepy.core._versions._load_from_rda", return_value=None):
        from ariadnepy.core._versions import load_version_metadata
        df = load_version_metadata()
    assert isinstance(df, pd.DataFrame)


def test_load_version_metadata_prefers_rda_over_json():
    """If RDA is available it should be used instead of bundled JSON."""
    mock_df = pd.DataFrame([
        {"source": "KEGG", "version": "latest", "key": "", "graph": 19397292, "default": True}
    ])
    with patch("ariadnepy.core._versions._load_from_rda", return_value=mock_df):
        from ariadnepy.core._versions import load_version_metadata
        df = load_version_metadata()
    assert list(df["source"]) == ["KEGG"]


def test_load_version_metadata_has_urls_attr():
    with patch("ariadnepy.core._versions._load_from_rda", return_value=None):
        from ariadnepy.core._versions import load_version_metadata
        df = load_version_metadata()
    assert "urls" in df.attrs


# ── resolve_versions ──────────────────────────────────────────────────────────

def _mock_meta():
    return pd.DataFrame([
        {"source": "KEGG",    "version": "latest",     "key": "", "graph": 1, "default": True},
        {"source": "GO",      "version": "2026-03-25", "key": "", "graph": 1, "default": True},
        {"source": "GO",      "version": "2026-01-23", "key": "", "graph": 1, "default": False},
        {"source": "UniProt", "version": "latest",     "key": "", "graph": 1, "default": True},
    ])


def test_resolve_versions_returns_dict():
    with patch("ariadnepy.core._versions.load_version_metadata", return_value=_mock_meta()):
        from ariadnepy.core._versions import resolve_versions
        result = resolve_versions(None)
    assert isinstance(result, dict)


def test_resolve_versions_fills_defaults():
    with patch("ariadnepy.core._versions.load_version_metadata", return_value=_mock_meta()):
        from ariadnepy.core._versions import resolve_versions
        result = resolve_versions(None)
    assert "KEGG" in result
    assert result["KEGG"] == "latest"


def test_resolve_versions_override_accepted():
    with patch("ariadnepy.core._versions.load_version_metadata", return_value=_mock_meta()):
        from ariadnepy.core._versions import resolve_versions
        result = resolve_versions({"GO": "2026-01-23"})
    assert result["GO"] == "2026-01-23"


def test_resolve_versions_invalid_version_raises():
    with patch("ariadnepy.core._versions.load_version_metadata", return_value=_mock_meta()):
        from ariadnepy.core._versions import resolve_versions
        with pytest.raises(AriadneVersionError, match="not available"):
            resolve_versions({"GO": "1999-01-01"})


def test_resolve_versions_unknown_source_raises():
    with patch("ariadnepy.core._versions.load_version_metadata", return_value=_mock_meta()):
        from ariadnepy.core._versions import resolve_versions
        with pytest.raises(AriadneVersionError, match="Unknown source"):
            resolve_versions({"FakeDB": "v1"})


# ── list_resource_versions ────────────────────────────────────────────────────

def test_list_resource_versions_returns_dataframe():
    with patch("ariadnepy.core._versions.load_version_metadata", return_value=_mock_meta()):
        from ariadnepy.core._versions import list_resource_versions
        df = list_resource_versions()
    assert isinstance(df, pd.DataFrame)


def test_list_resource_versions_has_correct_columns():
    with patch("ariadnepy.core._versions.load_version_metadata", return_value=_mock_meta()):
        from ariadnepy.core._versions import list_resource_versions
        df = list_resource_versions()
    assert list(df.columns) == ["resource", "version", "url"]


def test_list_resource_versions_default_filters():
    with patch("ariadnepy.core._versions.load_version_metadata", return_value=_mock_meta()):
        from ariadnepy.core._versions import list_resource_versions
        all_versions = list_resource_versions(default=False)
        defaults_only = list_resource_versions(default=True)
    assert len(defaults_only) < len(all_versions)
    assert len(defaults_only) == _mock_meta()["default"].sum()
