"""add_modules / get_modules — mirrors R's addModules / getModules in ariadne.

Attaches or retrieves a linkmap as wide-format module membership columns
in the var (feature) or obs (sample) metadata of an AnnData object.

R equivalent:
    addModules(tse, butyrate, key = "Genus", as = "names")
    getModules(tse, butyrate, key = "Genus")
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Union

import pandas as pd

from ariadnepy.exceptions import AriadneError

if TYPE_CHECKING:
    from anndata import AnnData


def _require_anndata() -> None:
    try:
        import anndata  # noqa: F401
    except ImportError:
        raise AriadneError(
            "'anndata' is required for add_modules / get_modules. "
            "Install with: pip install anndata"
        )


def _build_wide_modules(
    modules: pd.DataFrame,
    use: str,
) -> pd.DataFrame:
    """Pivot a linkmap to wide format: origin × module → coverage value.

    Mirrors R's dcast(modules, origin ~ target, value.var="cov", fill=0).
    """
    if modules.shape[1] < 2:
        raise AriadneError("'modules' must have at least 2 columns.")
    if use not in ("ids", "names"):
        raise AriadneError("'use' must be either 'ids' or 'names'.")

    origin_col = modules.columns[0]
    # "ids" uses column 2, "names" uses the last column (name column)
    target_col = modules.columns[1] if use == "ids" else modules.columns[-1]

    df = modules[[origin_col, target_col]].copy()

    # Add coverage column if not present (plain linkmap → binary membership)
    if "cov" in modules.columns:
        df = modules[[origin_col, target_col, "cov"]].copy()
        value_col = "cov"
    else:
        df = df.copy()
        df["cov"] = 1.0
        value_col = "cov"

    wide = df.pivot_table(
        index=origin_col,
        columns=target_col,
        values=value_col,
        aggfunc="max",
        fill_value=0,
    )
    wide.columns.name = None
    return wide


def _match_index(
    side_df: pd.DataFrame,
    key: Union[str, List[str]],
    wide: pd.DataFrame,
) -> pd.Index:
    """Return the wide-table index value to use for each row of side_df.

    Mirrors R's multi-key priority matching:
      - key == "index": match by side_df.index
      - single key: match by side_df[key]
      - multiple keys: try each in order, take first match per row
    """
    origin_set = set(wide.index)

    if key == "index" or key == ["index"]:
        return side_df.index

    keys = [key] if isinstance(key, str) else list(key)
    if not all(k in side_df.columns for k in keys):
        missing = [k for k in keys if k not in side_df.columns]
        raise AriadneError(
            f"'key' columns not found in side information: {missing}. "
            "Use key='index' to match by feature/sample names."
        )

    # For each row, try keys in priority order (first key wins)
    result = pd.Series([None] * len(side_df), index=side_df.index, dtype=object)
    for k in reversed(keys):
        vals = side_df[k].astype(str)
        mask = vals.isin(origin_set)
        result[mask] = vals[mask]

    return result


def get_modules(
    adata: "AnnData",
    modules: pd.DataFrame,
    by: str = "var",
    key: Union[str, List[str]] = "index",
    use: str = "ids",
) -> pd.DataFrame:
    """Retrieve module membership for features or samples in an AnnData object.

    Equivalent to R's ``getModules(tse, modules, by="rows", key="row.names")``.

    Parameters
    ----------
    adata:
        AnnData object (equivalent to SummarizedExperiment).
    modules:
        Linkmap DataFrame as returned by ``weave_path`` or ``weave_complex``.
        First column = origin IDs, second column = module IDs,
        optional last column = module names, optional ``cov`` column = coverage.
    by:
        ``"var"`` to match against feature metadata (rowData equivalent),
        ``"obs"`` to match against sample metadata (colData equivalent).
        Also accepts ``"rows"`` / ``"cols"`` as R-style aliases.
    key:
        Column name(s) in ``adata.var`` / ``adata.obs`` to match against the
        first column of ``modules``. Use ``"index"`` (default) to match by
        the AnnData index (feature/sample names).
        Pass a list for priority matching — first match wins per row.
    use:
        ``"ids"`` to use module IDs (column 2), ``"names"`` to use module
        names (last column). Requires a name column in ``modules``.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame: rows = features/samples, columns = modules,
        values = coverage (0–1) or binary membership (0/1).

    Examples
    --------
    >>> from ariadnepy import weave_path, get_modules
    >>> lm = weave_path(graph, "taxname ~ bugsig")
    >>> membership = get_modules(adata, lm, key="index")
    """
    _require_anndata()

    by = _normalise_by(by)
    side_df: pd.DataFrame = adata.var if by == "var" else adata.obs

    wide = _build_wide_modules(modules, use)
    lookup = _match_index(side_df, key, wide)

    # Reindex wide table to align with side_df rows
    aligned = wide.reindex(lookup.values).set_index(side_df.index)
    aligned = aligned.fillna(0)

    return aligned


def add_modules(
    adata: "AnnData",
    modules: pd.DataFrame,
    by: str = "var",
    key: Union[str, List[str]] = "index",
    use: str = "ids",
) -> "AnnData":
    """Append module membership columns to an AnnData object's var or obs.

    Equivalent to R's ``addModules(tse, modules, by="rows", key="row.names")``.

    Parameters
    ----------
    adata:
        AnnData object. Modified in-place (a copy is returned).
    modules:
        Linkmap DataFrame as returned by ``weave_path`` or ``weave_complex``.
    by:
        ``"var"`` (default) to append to feature metadata,
        ``"obs"`` to append to sample metadata.
        Also accepts ``"rows"`` / ``"cols"`` as R-style aliases.
    key:
        Column(s) to match on. Use ``"index"`` to match by feature/sample names.
    use:
        ``"ids"`` (default) or ``"names"`` — which module column to use.

    Returns
    -------
    AnnData
        Updated AnnData object with new module columns added to var/obs.
        Existing columns with the same name are replaced with a warning.

    Examples
    --------
    >>> from ariadnepy import weave_path, add_modules
    >>> lm = weave_path(graph, "taxname ~ bugsig")
    >>> adata = add_modules(adata, lm, key="index")
    """
    _require_anndata()
    import anndata as ad

    by = _normalise_by(by)
    membership = get_modules(adata, modules, by=by, key=key, use=use)

    adata = adata.copy()
    side_df: pd.DataFrame = adata.var.copy() if by == "var" else adata.obs.copy()

    # Warn and drop any columns that would be overwritten
    duplicates = [c for c in membership.columns if c in side_df.columns]
    if duplicates:
        import warnings
        warnings.warn(
            f"The following columns were replaced: {duplicates}", UserWarning, stacklevel=2
        )
        side_df = side_df.drop(columns=duplicates)

    updated = pd.concat([side_df, membership], axis=1)

    if by == "var":
        adata.var = updated
    else:
        adata.obs = updated

    return adata


def _normalise_by(by: str) -> str:
    """Accept 'rows'/'cols' as R-style aliases for 'var'/'obs'."""
    mapping = {"rows": "var", "cols": "obs", "var": "var", "obs": "obs"}
    if by not in mapping:
        raise AriadneError("'by' must be 'var', 'obs', 'rows', or 'cols'.")
    return mapping[by]
