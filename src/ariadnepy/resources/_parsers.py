from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import pandas as pd

from ariadnepy.exceptions import AriadneParseError


def process_one2one(
    path: Union[str, Path],
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
    path: Union[str, Path],
    from_col: str,
    to_col: str,
    key_fn: Optional[Callable[[str], str]] = None,
    val_fn: Optional[Callable[[str], str]] = None,
    skiprows: int = 0,
    val_cols: Optional[Union[slice, List[int]]] = None,
) -> pd.DataFrame:
    """Parse a TSV where the first column maps to multiple values in the rest.

    Used for: ChocoPhlAn, WoL, BugSigDB.
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
        if len(parts) < 2:
            continue
        key = parts[0]
        if val_cols is None:
            selection = parts[1:]
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
    path: Union[str, Path],
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
