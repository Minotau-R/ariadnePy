"""process_gene_families — mirrors R's processGeneFamilies in ariadne.

Prepares an AnnData object containing HUMAnN gene families by parsing
the feature names into uniref90, taxname, genus, and species columns
and adding them to adata.var.

R equivalent:
    processGeneFamilies(genes)

HUMAnN feature name format:
    UniRef90_XXXXXX|g__Genus.s__species
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from ariadnepy.exceptions import AriadneError

if TYPE_CHECKING:
    from anndata import AnnData


def _require_anndata() -> None:
    try:
        import anndata  # noqa: F401
    except ImportError:
        raise AriadneError(
            "'anndata' is required for process_gene_families. "
            "Install with: pip install anndata"
        )


def process_gene_families(adata: "AnnData") -> "AnnData":
    """Parse HUMAnN gene family feature names and add metadata to adata.var.

    Filters to features that contain a ``|`` separator and do not contain
    ``"unclassified"``, then splits the name into four columns:
    ``uniref90``, ``taxname``, ``genus``, and ``species``.

    Equivalent to R's ``processGeneFamilies(genes)``.

    Parameters
    ----------
    adata:
        AnnData object whose ``var_names`` follow the HUMAnN format
        ``UniRef90_XXXXXX|g__Genus.s__species``.

    Returns
    -------
    AnnData
        Filtered AnnData (unclassified and non-stratified rows removed)
        with four new columns in ``adata.var``:
        ``uniref90``, ``taxname``, ``genus``, ``species``.

    Examples
    --------
    >>> from ariadnepy import process_gene_families
    >>> genes = process_gene_families(genes)
    >>> genes.var.head()
    """
    _require_anndata()

    names = pd.Series(adata.var_names, index=adata.var_names)

    # Keep only stratified (contains |) and classified features
    mask = names.str.contains("|", regex=False) & ~names.str.contains(
        "unclassified", case=False, regex=False
    )

    if not mask.any():
        raise AriadneError(
            "No valid HUMAnN gene family features found. "
            "Expected feature names like 'UniRef90_XXXXX|g__Genus.s__species'."
        )

    adata = adata[:, mask].copy()
    names = pd.Series(adata.var_names, index=adata.var_names)

    # Split on | to get uniref90 and taxname
    split_gene = names.str.split("|", n=1, expand=True)
    split_gene.columns = ["uniref90", "taxname"]
    split_gene.index = adata.var_names

    # Split taxname on . to get genus and species
    split_tax = split_gene["taxname"].str.split(".", n=1, expand=True)
    split_tax.columns = ["genus", "species"]
    split_tax.index = adata.var_names

    # Append to existing var
    for col in ["uniref90", "taxname", "genus", "species"]:
        if col in adata.var.columns:
            import warnings
            warnings.warn(f"Column '{col}' in adata.var was replaced.", UserWarning, stacklevel=2)

    adata.var = pd.concat(
        [adata.var, split_gene[["uniref90", "taxname"]], split_tax],
        axis=1,
    )

    return adata
