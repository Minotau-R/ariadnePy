from __future__ import annotations

import warnings
from typing import List, Optional, Sequence

import pandas as pd

from ariadnepy.exceptions import AriadneError

try:
    import requests as _requests
except ImportError:
    _requests = None

_KEGG_REST = "https://rest.kegg.jp"
_BUGSIG_GMT = "https://zenodo.org/records/15272273/files/bugsigdb_signatures_mixed_ncbi.gmt"


# ── Node lookup backends ──────────────────────────────────────────────────────

def _fetch_kegg_names(node_name: str, ids: Optional[List[str]]) -> Optional[pd.DataFrame]:
    """Fetch id→name pairs from the KEGG REST API."""
    if _requests is None:
        return None
    kegg_targets = {"ko", "pathway", "enzyme", "ec", "network", "reaction",
                    "compound", "glycan", "drug", "dgroup", "disease"}
    target = node_name if node_name in kegg_targets else None
    if target is None:
        return None
    try:
        if ids and len(ids) <= 50:
            query = "+".join(str(i) for i in ids)
            resp = _requests.get(f"{_KEGG_REST}/list/{query}", timeout=30)
        else:
            resp = _requests.get(f"{_KEGG_REST}/list/{target}", timeout=30)
        resp.raise_for_status()
        rows = []
        for line in resp.text.strip().splitlines():
            parts = line.split("\t", 1)
            if len(parts) == 2:
                id_ = parts[0].strip()
                name = parts[1].strip()
                # Keep only first name for ko (strip semicolon-delimited extras)
                if target == "ko":
                    name = name.split(";")[0].strip()
                else:
                    name = name.split(";")[-1].strip()
                rows.append((id_, name))
        return pd.DataFrame(rows, columns=["ids", "names"]) if rows else None
    except Exception:
        return None


def _fetch_bugsig_names(ids: Optional[List[str]]) -> Optional[pd.DataFrame]:
    """Fetch BugSigDB signature id→name pairs from Zenodo GMT file."""
    if _requests is None:
        return None
    try:
        resp = _requests.get(_BUGSIG_GMT, timeout=60)
        resp.raise_for_status()
        rows = []
        for line in resp.text.splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            header = parts[0]
            pieces = header.split("_", 1)
            sig_id = pieces[0].replace("bsdb:", "")
            sig_name = pieces[1].split(":", 1)[-1] if len(pieces) > 1 else header
            rows.append((sig_id, sig_name))
        return pd.DataFrame(rows, columns=["ids", "names"]) if rows else None
    except Exception:
        return None


def _fetch_gmm_gbm_names(url: str) -> Optional[pd.DataFrame]:
    """Fetch module id→name from a tab-separated URL (GMM/GBM)."""
    if _requests is None or not url:
        return None
    try:
        resp = _requests.get(url, timeout=60)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text), sep=None, engine="python", header=None)
        if df.shape[1] >= 2:
            return pd.DataFrame({"ids": df.iloc[:, 0].astype(str),
                                  "names": df.iloc[:, 1].astype(str)})
    except Exception:
        pass
    return None


def _fetch_file_names(
    node_row: pd.Series,
    ids: Optional[List[str]],
) -> Optional[pd.DataFrame]:
    """Read id→name from a cached parquet file for file-backed resources."""
    url = node_row.get("url")
    source = node_row.get("source", "")
    spec = node_row.get("spec", node_row.get("name", ""))
    if not url or pd.isna(url):
        return None
    try:
        from ariadnepy.resources._cache import cache_resource
        name_col = f"{spec}_name"
        if node_row.get("name") == "msig":
            name_col = name_col.replace("_id", "")
        cached = cache_resource(url, source, spec, name_col)
        df = pd.read_parquet(cached, columns=[spec, name_col])
        if ids:
            df = df[df[spec].isin(set(ids))]
        return df.rename(columns={spec: "ids", name_col: "names"}).reset_index(drop=True)
    except Exception:
        return None


def _fetch_node_names(
    node_row: pd.Series,
    ids: Optional[List[str]],
) -> Optional[pd.DataFrame]:
    """Route to the correct name-fetching backend for a graph node."""
    name = node_row.get("name", "")

    if name == "bugsig":
        return _fetch_bugsig_names(ids)

    if name in ("gmm", "gbm"):
        return _fetch_gmm_gbm_names(node_row.get("url", ""))

    url = node_row.get("url")
    if url and not pd.isna(url):
        return _fetch_file_names(node_row, ids)

    return _fetch_kegg_names(name, ids)


# ── Key matching ──────────────────────────────────────────────────────────────

def _match_key2val(
    linkmap: pd.DataFrame,
    x: str,
    what: int,
    init: Optional[Sequence[str]],
    verbose: bool,
) -> pd.DataFrame:
    """Filter/reorder linkmap rows to match init values.

    what=1 → input are ids, return (ids, names)
    what=2 → input are names, return (names, ids)
    """
    if init is None:
        return linkmap
    key_col = "ids" if what == 1 else "names"
    val_col = "names" if what == 1 else "ids"
    lm = linkmap.set_index(key_col)
    found = []
    missing = 0
    for v in init:
        try:
            val = lm.at[v, val_col]
        except KeyError:
            val = None
        if val is None or (isinstance(val, float) and pd.isna(val)):
            missing += 1
            found.append(None)
        else:
            found.append(val)
    if verbose and missing:
        warnings.warn(f"{missing} {x} {key_col} not found.", stacklevel=3)
    if what == 1:
        return pd.DataFrame({"ids": list(init), "names": found})
    return pd.DataFrame({"ids": found, "names": list(init)})


# ── Public API ────────────────────────────────────────────────────────────────

def link_names(
    graph,
    x: str,
    ids: Optional[Sequence[str]] = None,
    names: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> Optional[pd.DataFrame]:
    """Retrieve names for IDs (or IDs for names) of a resource node.

    Equivalent to R's ``linkNames(graph, "gmm")`` or
    ``linkNames(graph, "ko", ids=c("K00001","K00844"))``.

    Parameters
    ----------
    graph:
        NetworkX MultiDiGraph returned by ``ariadne()``.
    x:
        Node name in the graph (e.g. ``"ko"``, ``"gmm"``, ``"bugsig"``).
    ids:
        IDs to look up → returns matching names.
    names:
        Names to look up → returns matching IDs.
    verbose:
        Warn when IDs/names have no match.

    Returns
    -------
    pd.DataFrame or None
        Two columns: ``(x, x.name)`` with one row per matched pair,
        or None if no name data is available for the node.

    Examples
    --------
    >>> from ariadnepy import ariadne
    >>> from ariadnepy.graph import link_names
    >>> graph = ariadne()
    >>> link_names(graph, "gmm")
    >>> link_names(graph, "ko", ids=["K00001", "K00844"])
    >>> link_names(graph, "ec", names=["alcohol dehydrogenase"])
    """
    if ids is not None and names is not None:
        raise AriadneError("Specify only 'ids' or 'names', not both.")

    import igraph as ig
    if isinstance(graph, ig.Graph):
        node_names = graph.vs["name"]
        if x not in node_names:
            raise AriadneError(f"Node {x!r} not found in graph.")
        v = graph.vs.find(name=x)
        node_row = pd.Series({"name": x, **{a: v[a] for a in graph.vertex_attributes() if a != "name"}})
    else:
        node_attrs = dict(graph.nodes(data=True))
        if x not in node_attrs:
            raise AriadneError(f"Node {x!r} not found in graph.")
        node_row = pd.Series({"name": x, **node_attrs[x]})

    name_links = _fetch_node_names(node_row, list(ids) if ids is not None else None)
    if name_links is None:
        return None

    name_links = _match_key2val(name_links, x, 1, ids, verbose)
    name_links = _match_key2val(name_links, x, 2, names, verbose)
    name_links.columns = [x, f"{x}.name"]
    return name_links
