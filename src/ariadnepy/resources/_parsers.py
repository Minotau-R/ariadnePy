from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from ariadnepy.exceptions import AriadneParseError


def process_one2one(
    path: str | Path,
    from_col: str,
    to_col: str,
    select: Sequence[int] = (0, 1),
    header: bool = False,
) -> pd.DataFrame:
    """Parse a two-column TSV where each row is a 1-to-1 mapping.

    Strips common prefixes like 'GO:' from both columns.
    Used for: GO, TIGRFAMs.
    """
    path = Path(path)
    df = pd.read_csv(path, sep="\t", header=0 if header else None, dtype=str)
    if df.shape[1] < max(select) + 1:
        raise AriadneParseError(
            f"Expected at least {max(select) + 1} columns, got {df.shape[1]}"
        )
    col_a = df.iloc[:, select[0]].astype(str).str.replace(r"^.*?:", "", regex=True)
    col_b = df.iloc[:, select[1]].astype(str).str.replace(r"^GO:", "", regex=True)
    return pd.DataFrame({from_col: col_a, to_col: col_b})


def process_one2many(
    path: str | Path,
    from_col: str,
    to_col: str,
    key_col: int = 0,
    key_fn: Callable[[str], str] | None = None,
    val_fn: Callable[[str], str] | None = None,
    skiprows: int = 0,
    val_cols: slice | list[int] | None = None,
) -> pd.DataFrame:
    """Parse a TSV where one column maps to multiple values in the rest.

    Used for: ChocoPhlAn, WoL, BugSigDB.

    Parameters
    ----------
    key_col:
        Index of the column holding the "one" side of the mapping.
        (R equivalent: ``key.col``.)
    val_cols:
        Indices/slice of columns holding the "many" side. Defaults to every
        column except ``key_col``. (R equivalent: ``val.cols``.)
    """
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    if skiprows:
        lines = lines[skiprows:]

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2 or key_col >= len(parts):
            continue
        key = parts[key_col]
        if val_cols is None:
            selection = [p for i, p in enumerate(parts) if i != key_col]
        elif isinstance(val_cols, slice):
            selection = parts[val_cols]
        else:
            selection = [parts[i] for i in val_cols if 0 <= i < len(parts)]

        for val in selection:
            if not val:
                continue
            k = key_fn(key) if key_fn else key
            v = val_fn(val) if val_fn else val
            rows.append({from_col: k, to_col: v})

    return pd.DataFrame(rows, columns=[from_col, to_col])


def process_complex_modules(
    path: str | Path,
    from_col: str,
    to_col: str,
) -> pd.DataFrame:
    """Parse a KEGG-style module flat file into a 2-column linkmap.

    Each entry block uses tab-separated complexes and comma-separated features.
    Used for: GM (gut metabolic modules), GBM.
    """
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            key = parts[0]
            for value in parts[1:]:
                for token in re.split(r"[;,]", value):
                    token = token.strip()
                    if token:
                        rows.append({from_col: key, to_col: token})
    return pd.DataFrame(rows, columns=[from_col, to_col])
