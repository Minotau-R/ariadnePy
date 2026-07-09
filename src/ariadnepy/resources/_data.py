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
    data_file = Path(__file__).resolve().parent / "data" / "butyrate.csv"
    if not data_file.exists():
        raise FileNotFoundError(
            f"Bundled dataset not found at {data_file}. "
            "Ensure the package was installed correctly."
        )
    return pd.read_csv(data_file)


def load_pathmeta() -> pd.DataFrame:
    """Load the bundled example pathway from chebi to gmm.

    A minimal path DataFrame describing one route through the ariadne graph
    (chebi -> rhea -> ec -> ko -> gmm), the kind of table normally produced by
    ``draw_path()`` and consumed by ``weave_path()``/``weave_complex()``.

    Returns
    -------
    pd.DataFrame
        Path steps with columns: from, to, source, version, url.

    Examples
    --------
    >>> from ariadnepy.resources import load_pathmeta
    >>> pathmeta = load_pathmeta()
    >>> from ariadnepy.graph import weave_path
    >>> chebi2gmm = weave_path(pathmeta, init=[15377, 30616, 4167])
    """
    data_file = Path(__file__).resolve().parent / "data" / "pathmeta.csv"
    if not data_file.exists():
        raise FileNotFoundError(
            f"Bundled dataset not found at {data_file}. "
            "Ensure the package was installed correctly."
        )
    return pd.read_csv(data_file)
