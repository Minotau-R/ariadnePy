"""Tests for anndata/_modules.py and anndata/_humann.py.

All tests use synthetic AnnData objects — no network calls, no real data.
anndata-dependent tests are skipped when anndata is not installed.
"""
from __future__ import annotations

import importlib.util
import warnings

import numpy as np
import pandas as pd
import pytest

from ariadnepy.exceptions import AriadneError

needs_anndata = pytest.mark.skipif(
    importlib.util.find_spec("anndata") is None,
    reason="anndata not installed",
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_adata():
    """AnnData with 5 features, 3 samples. var has a 'Genus' column."""
    import anndata as ad
    var = pd.DataFrame(
        {"Genus": ["Bacteroides", "Faecalibacterium", "Blautia", "Roseburia", "Clostridium"]},
        index=["feat1", "feat2", "feat3", "feat4", "feat5"],
    )
    obs = pd.DataFrame(index=["s1", "s2", "s3"])
    X = np.zeros((3, 5))
    return ad.AnnData(X=X, var=var, obs=obs)


@pytest.fixture
def simple_linkmap():
    """2-col linkmap: feature ID → module ID."""
    return pd.DataFrame({
        "feature": ["feat1", "feat2", "feat3", "feat4"],
        "module":  ["M1",    "M1",    "M2",    "M2"],
    })


@pytest.fixture
def coverage_linkmap():
    """Linkmap with coverage values (as returned by weave_complex)."""
    return pd.DataFrame({
        "feature": ["feat1", "feat2", "feat3"],
        "module":  ["M1",    "M1",    "M2"],
        "cov":     [0.8,     0.6,     1.0],
    })


@pytest.fixture
def named_linkmap():
    """Linkmap with an extra names column."""
    return pd.DataFrame({
        "feature":     ["feat1", "feat2", "feat3"],
        "module":      ["M1",    "M1",    "M2"],
        "module.name": ["Butyrate producers", "Butyrate producers", "Acetate producers"],
    })


@pytest.fixture
def genus_linkmap():
    """Linkmap keyed by genus name instead of feature ID."""
    return pd.DataFrame({
        "Genus":  ["Bacteroides", "Faecalibacterium", "Blautia"],
        "module": ["M1",          "M2",               "M2"],
    })


@pytest.fixture
def humann_adata():
    """AnnData with HUMAnN-style feature names."""
    import anndata as ad
    var_names = [
        "UniRef90_A0A010|g__Bacteroides.s__fragilis",
        "UniRef90_B0B020|g__Faecalibacterium.s__prausnitzii",
        "UniRef90_C0C030|g__Blautia.s__obeum",
        "UniRef90_UNMAPPED",           # no | → should be filtered out
        "UniRef90_D0D040|unclassified", # unclassified → should be filtered out
    ]
    obs = pd.DataFrame(index=["s1", "s2"])
    X = np.zeros((2, 5))
    return ad.AnnData(X=X, var=pd.DataFrame(index=var_names), obs=obs)


# ── get_modules ───────────────────────────────────────────────────────────────

@needs_anndata
def test_get_modules_returns_dataframe(simple_adata, simple_linkmap):
    from ariadnepy.anndata._modules import get_modules
    result = get_modules(simple_adata, simple_linkmap)
    assert isinstance(result, pd.DataFrame)


@needs_anndata
def test_get_modules_columns_are_module_ids(simple_adata, simple_linkmap):
    from ariadnepy.anndata._modules import get_modules
    result = get_modules(simple_adata, simple_linkmap)
    assert set(result.columns) == {"M1", "M2"}


@needs_anndata
def test_get_modules_index_matches_adata_var(simple_adata, simple_linkmap):
    from ariadnepy.anndata._modules import get_modules
    result = get_modules(simple_adata, simple_linkmap)
    assert list(result.index) == list(simple_adata.var_names)


@needs_anndata
def test_get_modules_membership_values_binary(simple_adata, simple_linkmap):
    """feat1 and feat2 → M1; feat3 and feat4 → M2; feat5 → neither."""
    from ariadnepy.anndata._modules import get_modules
    result = get_modules(simple_adata, simple_linkmap)
    assert result.loc["feat1", "M1"] > 0
    assert result.loc["feat2", "M1"] > 0
    assert result.loc["feat3", "M2"] > 0
    assert result.loc["feat5", "M1"] == 0
    assert result.loc["feat5", "M2"] == 0


@needs_anndata
def test_get_modules_missing_features_filled_with_zero(simple_adata, simple_linkmap):
    from ariadnepy.anndata._modules import get_modules
    result = get_modules(simple_adata, simple_linkmap)
    # feat5 not in linkmap — both module columns should be 0
    assert result.loc["feat5", "M1"] == 0
    assert result.loc["feat5", "M2"] == 0


@needs_anndata
def test_get_modules_coverage_values_preserved(simple_adata, coverage_linkmap):
    from ariadnepy.anndata._modules import get_modules
    result = get_modules(simple_adata, coverage_linkmap)
    assert result.loc["feat1", "M1"] == pytest.approx(0.8)
    assert result.loc["feat3", "M2"] == pytest.approx(1.0)


@needs_anndata
def test_get_modules_use_names(simple_adata, named_linkmap):
    """use='names' should use the last column as module identifier."""
    from ariadnepy.anndata._modules import get_modules
    result = get_modules(simple_adata, named_linkmap, use="names")
    assert "Butyrate producers" in result.columns
    assert "Acetate producers" in result.columns


@needs_anndata
def test_get_modules_key_genus_column(simple_adata, genus_linkmap):
    """key='Genus' matches by the Genus column in adata.var."""
    from ariadnepy.anndata._modules import get_modules
    result = get_modules(simple_adata, genus_linkmap, key="Genus")
    # Bacteroides → M1
    assert result.loc["feat1", "M1"] > 0
    # Faecalibacterium → M2
    assert result.loc["feat2", "M2"] > 0


@needs_anndata
def test_get_modules_by_obs(simple_adata, simple_linkmap):
    """by='obs' applies to adata.obs instead of adata.var."""
    import anndata as ad
    from ariadnepy.anndata._modules import get_modules
    obs_linkmap = pd.DataFrame({
        "sample": ["s1", "s2"],
        "group":  ["G1", "G2"],
    })
    result = get_modules(simple_adata, obs_linkmap, by="obs")
    assert list(result.index) == list(simple_adata.obs_names)


@needs_anndata
def test_get_modules_rows_alias(simple_adata, simple_linkmap):
    """by='rows' is accepted as an alias for by='var'."""
    from ariadnepy.anndata._modules import get_modules
    result_var  = get_modules(simple_adata, simple_linkmap, by="var")
    result_rows = get_modules(simple_adata, simple_linkmap, by="rows")
    pd.testing.assert_frame_equal(result_var, result_rows)


@needs_anndata
def test_get_modules_invalid_by_raises(simple_adata, simple_linkmap):
    from ariadnepy.anndata._modules import get_modules
    with pytest.raises(AriadneError, match="'by' must be"):
        get_modules(simple_adata, simple_linkmap, by="invalid")


@needs_anndata
def test_get_modules_invalid_use_raises(simple_adata, simple_linkmap):
    from ariadnepy.anndata._modules import get_modules
    with pytest.raises(AriadneError, match="'use' must be"):
        get_modules(simple_adata, simple_linkmap, use="invalid")


@needs_anndata
def test_get_modules_missing_key_column_raises(simple_adata, simple_linkmap):
    from ariadnepy.anndata._modules import get_modules
    with pytest.raises(AriadneError, match="not found in side information"):
        get_modules(simple_adata, simple_linkmap, key="NonExistentCol")


# ── add_modules ───────────────────────────────────────────────────────────────

@needs_anndata
def test_add_modules_returns_anndata(simple_adata, simple_linkmap):
    import anndata as ad
    from ariadnepy.anndata._modules import add_modules
    result = add_modules(simple_adata, simple_linkmap)
    assert isinstance(result, ad.AnnData)


@needs_anndata
def test_add_modules_columns_added_to_var(simple_adata, simple_linkmap):
    from ariadnepy.anndata._modules import add_modules
    result = add_modules(simple_adata, simple_linkmap)
    assert "M1" in result.var.columns
    assert "M2" in result.var.columns


@needs_anndata
def test_add_modules_original_var_preserved(simple_adata, simple_linkmap):
    """Existing var columns (e.g. Genus) should still be present."""
    from ariadnepy.anndata._modules import add_modules
    result = add_modules(simple_adata, simple_linkmap)
    assert "Genus" in result.var.columns


@needs_anndata
def test_add_modules_does_not_mutate_input(simple_adata, simple_linkmap):
    from ariadnepy.anndata._modules import add_modules
    original_cols = list(simple_adata.var.columns)
    add_modules(simple_adata, simple_linkmap)
    assert list(simple_adata.var.columns) == original_cols


@needs_anndata
def test_add_modules_duplicate_columns_warns(simple_adata, simple_linkmap):
    """If a module column already exists in var, a warning should be raised."""
    from ariadnepy.anndata._modules import add_modules
    # Pre-add M1 column
    simple_adata.var["M1"] = 99
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        add_modules(simple_adata, simple_linkmap)
    assert any("replaced" in str(warning.message).lower() for warning in w)


@needs_anndata
def test_add_modules_shape_unchanged(simple_adata, simple_linkmap):
    """Adding modules should not change the number of features or samples."""
    from ariadnepy.anndata._modules import add_modules
    result = add_modules(simple_adata, simple_linkmap)
    assert result.shape == simple_adata.shape


# ── process_gene_families ─────────────────────────────────────────────────────

@needs_anndata
def test_process_gene_families_returns_anndata(humann_adata):
    import anndata as ad
    from ariadnepy.anndata._humann import process_gene_families
    result = process_gene_families(humann_adata)
    assert isinstance(result, ad.AnnData)


@needs_anndata
def test_process_gene_families_adds_four_columns(humann_adata):
    from ariadnepy.anndata._humann import process_gene_families
    result = process_gene_families(humann_adata)
    for col in ("uniref90", "taxname", "genus", "species"):
        assert col in result.var.columns


@needs_anndata
def test_process_gene_families_filters_no_pipe(humann_adata):
    """Features without | should be removed."""
    from ariadnepy.anndata._humann import process_gene_families
    result = process_gene_families(humann_adata)
    assert all("|" in name for name in result.var_names)


@needs_anndata
def test_process_gene_families_filters_unclassified(humann_adata):
    """Features containing 'unclassified' should be removed."""
    from ariadnepy.anndata._humann import process_gene_families
    result = process_gene_families(humann_adata)
    assert all("unclassified" not in name.lower() for name in result.var_names)


@needs_anndata
def test_process_gene_families_correct_row_count(humann_adata):
    """Only 3 of 5 features are valid HUMAnN entries."""
    from ariadnepy.anndata._humann import process_gene_families
    result = process_gene_families(humann_adata)
    assert result.n_vars == 3


@needs_anndata
def test_process_gene_families_parses_uniref90(humann_adata):
    from ariadnepy.anndata._humann import process_gene_families
    result = process_gene_families(humann_adata)
    assert result.var.loc[
        "UniRef90_A0A010|g__Bacteroides.s__fragilis", "uniref90"
    ] == "UniRef90_A0A010"


@needs_anndata
def test_process_gene_families_parses_taxname(humann_adata):
    from ariadnepy.anndata._humann import process_gene_families
    result = process_gene_families(humann_adata)
    assert result.var.loc[
        "UniRef90_A0A010|g__Bacteroides.s__fragilis", "taxname"
    ] == "g__Bacteroides.s__fragilis"


@needs_anndata
def test_process_gene_families_parses_genus(humann_adata):
    from ariadnepy.anndata._humann import process_gene_families
    result = process_gene_families(humann_adata)
    assert result.var.loc[
        "UniRef90_A0A010|g__Bacteroides.s__fragilis", "genus"
    ] == "g__Bacteroides"


@needs_anndata
def test_process_gene_families_parses_species(humann_adata):
    from ariadnepy.anndata._humann import process_gene_families
    result = process_gene_families(humann_adata)
    assert result.var.loc[
        "UniRef90_A0A010|g__Bacteroides.s__fragilis", "species"
    ] == "s__fragilis"


@needs_anndata
def test_process_gene_families_no_valid_features_raises():
    """AnnData with no valid HUMAnN features should raise AriadneError."""
    import anndata as ad
    from ariadnepy.anndata._humann import process_gene_families
    bad = ad.AnnData(
        X=np.zeros((2, 2)),
        var=pd.DataFrame(index=["UNMAPPED", "UNGROUPED"]),
        obs=pd.DataFrame(index=["s1", "s2"]),
    )
    with pytest.raises(AriadneError, match="No valid HUMAnN"):
        process_gene_families(bad)
