"""Tests for bundled data and list_resource_versions utility."""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest


# ── load_butyrate ─────────────────────────────────────────────────────────────

def test_load_butyrate_returns_dataframe():
    from ariadnepy.resources._data import load_butyrate
    df = load_butyrate()
    assert isinstance(df, pd.DataFrame)


def test_load_butyrate_has_16_rows():
    from ariadnepy.resources._data import load_butyrate
    df = load_butyrate()
    assert len(df) == 16


def test_load_butyrate_has_required_columns():
    from ariadnepy.resources._data import load_butyrate
    df = load_butyrate()
    assert "taxname" in df.columns


def test_load_butyrate_taxname_nonempty():
    from ariadnepy.resources._data import load_butyrate
    df = load_butyrate()
    assert df["taxname"].notna().all()
    assert (df["taxname"].str.len() > 0).all()


def test_load_butyrate_missing_file_raises(monkeypatch):
    """If the CSV is somehow missing, FileNotFoundError should be raised."""
    from pathlib import Path
    import ariadnepy.resources._data as _mod
    monkeypatch.setattr(Path, "exists", lambda *_: False)
    with pytest.raises(FileNotFoundError):
        _mod.load_butyrate()


# ── list_resource_versions ────────────────────────────────────────────────────

def _mock_meta():
    return pd.DataFrame([
        {"source": "KEGG",    "version": "latest",     "key": "",          "graph": 1, "default": True},
        {"source": "GO",      "version": "2026-03-25", "key": "2026-03-25","graph": 1, "default": True},
        {"source": "GO",      "version": "2026-01-23", "key": "2026-01-23","graph": 1, "default": False},
    ])


def test_list_resource_versions_all_returns_all_rows():
    with patch("ariadnepy.core._versions.load_version_metadata", return_value=_mock_meta()):
        from ariadnepy.core._versions import list_resource_versions
        df = list_resource_versions(default=False)
    assert len(df) == 3


def test_list_resource_versions_default_true_filters():
    with patch("ariadnepy.core._versions.load_version_metadata", return_value=_mock_meta()):
        from ariadnepy.core._versions import list_resource_versions
        df = list_resource_versions(default=True)
    assert len(df) == 2
    assert "GO" in df["resource"].values
    assert "KEGG" in df["resource"].values


def test_list_resource_versions_url_column_is_string():
    with patch("ariadnepy.core._versions.load_version_metadata", return_value=_mock_meta()):
        from ariadnepy.core._versions import list_resource_versions
        df = list_resource_versions()
    assert df["url"].dtype == object
    assert df["url"].str.len().gt(0).all()
