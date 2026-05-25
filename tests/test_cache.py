"""Tests for ariadnepy._cache."""

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from ariadnepy._cache import (
    add_to_cache,
    init_cache,
    process_complex_modules,
    process_one2many,
    process_one2one,
)


def test_init_cache_returns_existing_directory():
    cache_dir = init_cache()
    assert cache_dir.exists()
    assert cache_dir.is_dir()


def test_add_to_cache_writes_parquet(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    dest = tmp_path / "test.parquet"
    result = add_to_cache(df, dest)
    assert Path(result).exists()
    reloaded = pd.read_parquet(result)
    pd.testing.assert_frame_equal(df, reloaded)


def test_process_one2one(tmp_path):
    tsv = tmp_path / "data.tsv"
    tsv.write_text("ns:A\tGO:001\nns:B\tGO:002\n")
    df = process_one2one(tsv, "from", "to")
    assert list(df["from"]) == ["A", "B"]
    assert list(df["to"]) == ["001", "002"]


def test_process_one2one_missing_column(tmp_path):
    tsv = tmp_path / "data.tsv"
    tsv.write_text("only_one_column\n")
    with pytest.raises(ValueError, match="two columns"):
        process_one2one(tsv, "from", "to")


def test_process_one2many(tmp_path):
    tsv = tmp_path / "data.tsv"
    tsv.write_text("key1\tval1\tval2\nkey2\tval3\n")
    df = process_one2many(tsv, "from", "to")
    assert len(df) == 3
    assert set(df["from"]) == {"key1", "key2"}


def test_process_complex_modules(tmp_path):
    tsv = tmp_path / "data.tsv"
    tsv.write_text("mod1\tA;B\tC\nmod2\tD,E\n")
    df = process_complex_modules(tsv, "mod", "gene")
    assert set(df["gene"]) == {"A", "B", "C", "D", "E"}
