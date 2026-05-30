from __future__ import annotations

from pathlib import Path
import igraph as ig
import pandas as pd

from ariadnepy.exceptions import AriadneError
from ariadnepy.resources._cache import add_to_cache, init_cache

try:
    import requests as _requests
except ImportError:
    _requests = None


def _read_linkmap(file: str, **kwargs) -> pd.DataFrame:
    """Read a local or remote file into a 2-column linkmap DataFrame."""
    path = Path(file)
    if path.exists():
        df = pd.read_csv(path, **kwargs)
    elif str(file).startswith(("http://", "https://")):
        if _requests is None:
            raise AriadneError(
                "'requests' is required to fetch remote resources. "
                "Install with: pip install requests"
            )
        import io
        resp = _requests.get(file, timeout=120)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), **kwargs)
    else:
        raise FileNotFoundError(f"Resource file not found: {file}")

    if df.shape[1] < 2:
        raise AriadneError(
            f"Resource file must have at least 2 columns, got {df.shape[1]}."
        )
    return df


def add_resource(
    graph: ig.Graph,
    file: str,
    res_name: str = "Custom",
    force: bool = False,
    **kwargs,
) -> ig.Graph:
    """Add a user-defined resource to the ariadne graph.

    The file must be a two-column table where column names are the feature
    types to connect (e.g. ``taxid``, ``aro``). Each row is one mapping.

    Equivalent to R's ``addResource(graph, file, res.name="AMR")``.

    Parameters
    ----------
    graph:
        NetworkX MultiDiGraph returned by ``ariadne()``.
    file:
        Path or URL to a CSV/TSV with exactly two named columns.
    res_name:
        Label for the new resource (appears as edge ``source`` attribute).
    force:
        Re-download and re-cache even if a cached version exists.
    **kwargs:
        Additional arguments forwarded to ``pd.read_csv`` (e.g. ``sep``,
        ``usecols``, ``names``).

    Returns
    -------
    ig.Graph
        Updated graph with new nodes and edges for the custom resource.

    Examples
    --------
    >>> from ariadnepy import ariadne
    >>> from ariadnepy.plot import add_resource
    >>> graph = ariadne()
    >>> graph = add_resource(graph, "my_mapping.csv", res_name="MyDB")
    >>> from ariadnepy.graph import search_path
    >>> search_path(graph, "taxid ~ aro")
    """
    df = _read_linkmap(file, **kwargs)
    if df.shape[1] != 2:
        df = df.iloc[:, :2]
    from_col, to_col = df.columns[0], df.columns[1]

    # Cache the linkmap as parquet
    cache_dir = init_cache()
    safe = "".join(c if c.isalnum() else "_" for c in str(file))[:80]
    cache_path = cache_dir / f"{res_name}_{safe}.parquet"
    if not cache_path.exists() or force:
        add_to_cache(df, cache_path)

    # Build edge attributes matching the graph schema
    url = file if str(file).startswith(("http://", "https://")) else str(Path(file).resolve())
    edge_attrs = {
        "source": res_name,
        "url": url,
        "from": from_col,
        "to": to_col,
    }

    graph = graph.copy()

    # Add vertices if missing
    existing = set(graph.vs["name"]) if graph.vcount() > 0 else set()
    for node in (from_col, to_col):
        if node not in existing:
            graph.add_vertex(name=node)
            existing.add(node)

    # Add edge
    src_idx = graph.vs.find(name=from_col).index
    tgt_idx = graph.vs.find(name=to_col).index
    graph.add_edge(src_idx, tgt_idx)
    for k, v in edge_attrs.items():
        graph.es[-1][k] = v
    return graph
