from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_butyrate() -> pd.DataFrame:
    """Load the bundled butyrate-producing microbe dataset.

    16 butyrate-producing microbial features from:
    Kullberg et al., The Lancet Microbe 5.9 (2024).

    Returns
    -------
    pd.DataFrame
        Feature table with microbial identifiers.

    Examples
    --------
    >>> from ariadnepy.resources import load_butyrate
    >>> df = load_butyrate()
    """
    data_file = Path(__file__).resolve().parent.parent.parent.parent / "data" / "butyrate.csv"
    if not data_file.exists():
        raise FileNotFoundError(
            f"Bundled dataset not found at {data_file}. "
            "Ensure the package was installed correctly."
        )
    return pd.read_csv(data_file)
