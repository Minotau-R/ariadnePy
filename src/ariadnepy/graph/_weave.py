from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx
import pandas as pd

from ariadnepy.exceptions import AriadneError

try:
    import requests as _requests
except ImportError:
    _requests = None

try:
    import scipy.sparse as sp
    import numpy as np
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ── Formula parsing ───────────────────────────────────────────────────────────

def _parse_by(by: str) -> Tuple[str, str]:
    """Parse 'taxname ~ bugsig' into ('taxname', 'bugsig')."""
    parts = [p.strip() for p in by.split("~")]
    if len(parts) != 2 or not all(parts):
        raise AriadneError(
            f"'by' must be a formula string like 'taxname ~ bugsig', got: {by!r}"
        )
    return parts[0], parts[1]


# ── Graph introspection helpers ───────────────────────────────────────────────

def _get_graph_dataframes(
    graph: nx.MultiDiGraph,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (node_df, edge_df) extracted from the NetworkX graph."""
    node_df = pd.DataFrame(
        [{"name": n, **attrs} for n, attrs in graph.nodes(data=True)]
    )
    if node_df.empty:
        node_df = pd.DataFrame(columns=["name"])

    edge_rows = []
    for u, v, attrs in graph.edges(data=True):
        edge_rows.append({"from": u, "to": v, **attrs})
    edge_df = pd.DataFrame(edge_rows)
    if edge_df.empty:
        edge_df = pd.DataFrame(columns=["from", "to"])

    return node_df, edge_df


def _get_edge_key(from_: str, to: str) -> str:
    return f"{from_}--{to}"


def _generic2specific(
    path_df: pd.DataFrame, node_df: pd.DataFrame, col: str
) -> List[str]:
    """Map a generic node name (e.g. 'ko') to its specific query attribute."""
    result = []
    for name in path_df[col]:
        matched = False
        if not node_df.empty and "spec" in node_df.columns:
            row = node_df[node_df["name"] == name]
            if not row.empty and pd.notna(row.iloc[0].get("spec")):
                result.append(str(row.iloc[0]["spec"]))
                matched = True
        if not matched:
            for cand in (f"{name}_id", f"{name}_name", "id", "ids", "name", "names"):
                if not node_df.empty and cand in node_df.columns:
                    result.append(cand)
                    matched = True
                    break
        if not matched:
            result.append(name)
    return result


# ── Path finding ──────────────────────────────────────────────────────────────

def _draw_path(
    graph: nx.MultiDiGraph,
    from_: str,
    to: str,
    k: int,
    include: Optional[List[str]],
    exclude: Optional[List[str]],
    res_name: Optional[List[str]],
    buffer_factor: int = 2,
    max_attempts: int = 5,
) -> pd.DataFrame:
    """Find the k-th shortest path in the graph, return as a step DataFrame."""
    if k < 1:
        raise AriadneError("'k' must be a positive integer.")

    include = include or []
    exclude = exclude or []

    if set(include) & set(exclude):
        raise AriadneError("'include' and 'exclude' cannot overlap.")
    if from_ not in graph:
        raise AriadneError(f"Source node {from_!r} not found in graph.")
    if to not in graph:
        raise AriadneError(f"Target node {to!r} not found in graph.")

    work_graph = graph
    if res_name is not None:
        kept_edges = [
            (u, v, ek)
            for u, v, ek, d in graph.edges(keys=True, data=True)
            if d.get("source") in res_name
        ]
        work_graph = graph.edge_subgraph(kept_edges).copy()

    # Use undirected traversal (equivalent to R's mode = "all")
    undirected = work_graph.to_undirected()

    candidate = k
    attempt = 0
    kept_paths: List[List[str]] = []

    while attempt < max_attempts and len(kept_paths) < k:
        try:
            gen = nx.shortest_simple_paths(undirected, from_, to)
            all_paths: List[List[str]] = []
            for _ in range(candidate):
                try:
                    all_paths.append(next(gen))
                except StopIteration:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            raise AriadneError(f"No path found between {from_!r} and {to!r}.")

        kept_paths = [
            p for p in all_paths
            if all(n in p for n in include) and not any(n in p for n in exclude)
        ]
        candidate *= buffer_factor
        attempt += 1

    if not kept_paths:
        raise AriadneError("No paths meet 'include' and 'exclude' criteria.")
    if k > len(kept_paths):
        raise AriadneError("'k' is greater than the number of possible paths.")

    path_nodes = kept_paths[k - 1]

    rows = []
    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i + 1]
        edge_data: dict = {}
        if work_graph.has_edge(u, v):
            edge_data = next(iter(work_graph[u][v].values()))
        elif work_graph.has_edge(v, u):
            edge_data = next(iter(work_graph[v][u].values()))
        rows.append({"from": u, "to": v, "source": edge_data.get("source", "")})

    return pd.DataFrame(rows)


def _add_edge_metadata(
    path_df: pd.DataFrame, graph: nx.MultiDiGraph, internal: bool
) -> pd.DataFrame:
    """Enrich path_df with url, version, and (if internal) spec column names."""
    path_df = path_df.copy()
    node_df, edge_df = _get_graph_dataframes(graph)

    versions = graph.graph.get("versions", {})
    path_df["version"] = path_df["source"].map(versions)

    graph_keys = [_get_edge_key(r["from"], r["to"]) for _, r in edge_df.iterrows()]
    path_keys = [_get_edge_key(r["from"], r["to"]) for _, r in path_df.iterrows()]

    url_map = dict(zip(graph_keys, edge_df.get("url", pd.Series(dtype=str))))
    path_df["url"] = [url_map.get(k) for k in path_keys]

    if internal:
        path_df["initFrom"] = path_df["from"]
        path_df["initTo"] = path_df["to"]

        from_map = dict(zip(graph_keys, edge_df["from"]))
        to_map = dict(zip(graph_keys, edge_df["to"]))
        for i, (idx, row) in enumerate(path_df.iterrows()):
            if pd.notna(row.get("url")):
                cf = from_map.get(path_keys[i])
                ct = to_map.get(path_keys[i])
                if cf:
                    path_df.at[idx, "from"] = cf
                if ct:
                    path_df.at[idx, "to"] = ct

        path_df["specFrom"] = _generic2specific(path_df, node_df, "from")
        path_df["specTo"] = _generic2specific(path_df, node_df, "to")
        path_df["specInitFrom"] = _generic2specific(path_df, node_df, "initFrom")

    return path_df


# ── IRI prefix helpers (SPARQL) ───────────────────────────────────────────────

def _add_iri(ids: Sequence[str], source: str, from_: str) -> List[str]:
    """Attach IRI prefixes required by UniProt/Rhea SPARQL endpoints."""
    ids = list(ids)
    if from_ in ("ecocyc", "metacyc"):
        prefix = (
            ("MetaCyc" if from_ == "metacyc" else "EcoCyc")
            if source == "UniProt"
            else from_.upper()
        )
        return [f"{prefix}:{x}" for x in ids]
    if from_ in ("chebi", "go"):
        return [f"{from_.upper()}_{x}" for x in ids]
    return ids


def _strip_iri(ids: Sequence[str], name: str) -> List[str]:
    """Remove IRI prefixes from SPARQL result values."""
    out = []
    for x in ids:
        y = re.sub(r"http.+/", "", str(x))
        if not re.search(r"^genes$|^uniref", name):
            y = re.sub(rf"^{re.escape(name)}[:_]", "", y, flags=re.IGNORECASE)
        out.append(y)
    return out


# ── Backend: KEGG REST ────────────────────────────────────────────────────────

_KEGG_REST = "https://rest.kegg.jp"
_KEGG_EXT_DBS = {"chebi", "geneid", "proteinid", "pubchem", "uniprotkb"}


def _fetch_kegg_edge(
    step: pd.Series, init: Optional[List[str]]
) -> pd.DataFrame:
    """Fetch one KEGG linkmap step via the KEGG REST API."""
    if _requests is None:
        raise AriadneError("'requests' package is required for KEGG queries.")

    from_ = step["specFrom"]
    to = step["specTo"]
    init_from = step["initFrom"]
    init_to = step["initTo"]
    is_init = init is not None

    use_conv = any(x in _KEGG_EXT_DBS for x in (from_, to))
    endpoint = "conv" if use_conv else "link"

    if "kegg_genes" in (init_from, init_to) and is_init:
        query_targets = init
    else:
        query_targets = [from_ if endpoint == "link" else to]

    if is_init and from_ in _KEGG_EXT_DBS:
        query_targets = [f"{from_}:{x}" for x in init]

    rows = []
    for chunk_start in range(0, max(1, len(query_targets)), 100):
        chunk = query_targets[chunk_start : chunk_start + 100]
        query = "+".join(chunk) if len(chunk) > 1 else chunk[0]
        url = f"{_KEGG_REST}/{endpoint}/{to}/{query}"
        resp = _requests.get(url, timeout=30)
        resp.raise_for_status()
        for line in resp.text.strip().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                xv = parts[0].strip()
                yv = parts[1].strip()
                if init_from != "kegg_genes":
                    xv = re.sub(r"^[^:]*:", "", xv)
                if init_to != "kegg_genes":
                    yv = re.sub(r"^[^:]*:", "", yv)
                rows.append((xv, yv))

    df = pd.DataFrame(rows, columns=[init_from, init_to])
    if is_init:
        df = df[df[init_from].isin(set(init))].reset_index(drop=True)
    return df


# ── Backend: SPARQL ───────────────────────────────────────────────────────────

_SPARQL_ENDPOINTS = {
    "UniProt": "https://sparql.uniprot.org/sparql",
    "Rhea": "https://sparql.rhea-db.org/sparql",
}


def _fetch_sparql_edge(
    step: pd.Series, init: Optional[List[str]], timeout: float
) -> pd.DataFrame:
    """Fetch one SPARQL edge, delegating to io._sparql when available."""
    try:
        from ariadnepy.io._sparql import query_sparql
        return query_sparql(
            from_=step["specFrom"],
            to=step["specTo"],
            endpoint=step["source"],
            init=init,
            timeout=timeout,
        )
    except (ImportError, AttributeError):
        pass

    if _requests is None:
        raise AriadneError("'requests' package is required for SPARQL queries.")

    endpoint_url = _SPARQL_ENDPOINTS.get(step["source"])
    if not endpoint_url:
        raise AriadneError(f"Unknown SPARQL source: {step['source']!r}")

    from_ = step["specFrom"]
    to = step["specTo"]
    iri_ids = _add_iri(init or [], step["source"], from_)
    values_clause = " ".join(f'"{v}"' for v in iri_ids[:500])

    query = (
        f"SELECT DISTINCT ?{from_} ?{to} WHERE {{"
        f"  VALUES ?{from_} {{ {values_clause} }}"
        f"}} LIMIT 10000"
    )
    resp = _requests.post(
        endpoint_url,
        data={"query": query, "format": "json"},
        timeout=timeout,
        headers={"Accept": "application/sparql-results+json"},
    )
    resp.raise_for_status()
    bindings = resp.json().get("results", {}).get("bindings", [])
    rows = [
        (b.get(from_, {}).get("value", ""), b.get(to, {}).get("value", ""))
        for b in bindings
    ]
    df = pd.DataFrame(rows, columns=[step["initFrom"], step["initTo"]])
    df[step["initFrom"]] = _strip_iri(df[step["initFrom"]], from_)
    df[step["initTo"]] = _strip_iri(df[step["initTo"]], to)
    return df


# ── Backend: Open Tree of Life ────────────────────────────────────────────────

_OTT_TNRS = "https://api.opentreeoflife.org/v3/tnrs/match_names"
_OTT_TAXON_INFO = "https://api.opentreeoflife.org/v3/taxonomy/taxon_info"


def _fetch_ott_edge(
    step: pd.Series, init: Optional[List[str]], timeout: float
) -> pd.DataFrame:
    """Fetch OTT taxonomy mappings, delegating to io._ott when available."""
    try:
        from ariadnepy.io._ott import query_ott
        return query_ott(
            from_=step["specFrom"],
            to=step["specTo"],
            init=init,
            timeout=timeout,
        )
    except (ImportError, AttributeError):
        pass

    if init is None:
        raise AriadneError("'init' must be provided for OTT queries.")
    if _requests is None:
        raise AriadneError("'requests' package is required for OTT queries.")

    from_ = step["specFrom"]
    names = [re.sub(r"^[a-z]__", "", x) for x in init]

    if from_ == "taxname":
        resp = _requests.post(
            _OTT_TNRS,
            json={"names": names, "do_approximate_matching": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        rows = []
        for entry in resp.json().get("results", []):
            query_name = entry.get("name", "")
            for match in entry.get("matches", []):
                ott_id = match.get("taxon", {}).get("ott_id")
                if ott_id:
                    rows.append((query_name, str(ott_id)))
                    break
        return pd.DataFrame(rows, columns=[step["initFrom"], step["initTo"]])

    if from_ in ("ncbi", "gbif", "worms", "if", "irmng"):
        rows = []
        for raw_id in init:
            source_id = f"{from_}:{raw_id}"
            try:
                resp = _requests.post(
                    _OTT_TAXON_INFO,
                    json={"source_id": source_id},
                    timeout=timeout,
                )
                resp.raise_for_status()
                ott_id = resp.json().get("ott_id")
                if ott_id:
                    rows.append((raw_id, str(ott_id)))
            except Exception:
                continue
        return pd.DataFrame(rows, columns=[step["initFrom"], step["initTo"]])

    raise AriadneError(f"Unsupported OTT from_ value: {from_!r}")


# ── Backend: file-based (parquet cache) ───────────────────────────────────────

def _fetch_file_edge(
    step: pd.Series, init: Optional[List[str]]
) -> pd.DataFrame:
    """Read a cached parquet linkmap for one graph edge."""
    try:
        from ariadnepy.resources._cache import cache_resource
        cached_path = cache_resource(
            step["url"], step["source"], step["specFrom"], step["specTo"]
        )
    except (ImportError, AttributeError):
        raise AriadneError(
            "resources._cache.cache_resource is not yet implemented. "
            "Cannot fetch file-backed edge."
        )

    cols = [step["specFrom"], step["specTo"]]
    df = pd.read_parquet(cached_path, columns=cols)

    if init is not None:
        filter_col = step.get("specInitFrom", cols[0])
        df = df[df[filter_col].isin(set(init))].reset_index(drop=True)

    # Swap column order if the traversal direction differs from file storage
    if step["from"] != step["initFrom"]:
        df = df[[step["specTo"], step["specFrom"]]].copy()

    df.columns = [step["initFrom"], step["initTo"]]
    return df


# ── Edge dispatcher ───────────────────────────────────────────────────────────

def _fetch_edge(
    step: pd.Series, init: Optional[List[str]], timeout: float
) -> pd.DataFrame:
    """Route one path step to the correct backend and return a linkmap."""
    source = step["source"]
    is_init = init is not None

    if source == "KEGG":
        if not is_init and step.get("initFrom") == "kegg_genes":
            raise AriadneError("'init' must be provided for kegg_genes queries.")
        df = _fetch_kegg_edge(step, init)

    elif source == "OTT":
        if not is_init:
            raise AriadneError("'init' must be provided for OTT queries.")
        df = _fetch_ott_edge(step, init, timeout)

    elif source in ("Rhea", "UniProt"):
        df = _fetch_sparql_edge(step, init, timeout)
        if step.get("specTo") == "BioCyc":
            key = re.sub(r"_.+$", "", step["initTo"])
            df = df[
                df.iloc[:, 1].str.contains(key, case=False, na=False)
            ].reset_index(drop=True)

    else:
        df = _fetch_file_edge(step, init)

    if df.empty:
        raise AriadneError(
            f"No mappings found for step "
            f"{step['initFrom']} → {step['initTo']} via {source}."
        )

    df.columns = [step["initFrom"], step["initTo"]]
    df[step["initFrom"]] = pd.Categorical(df[step["initFrom"]])
    df[step["initTo"]] = pd.Categorical(df[step["initTo"]])
    return df


# ── Linkmap chaining (replaces R's MultiFactor + weave) ──────────────────────

def _weave_linkmaps(
    linkmaps: Dict[str, pd.DataFrame], from_: str, to: str
) -> pd.DataFrame:
    """Chain a dict of 2-col DataFrames into a single from → to linkmap.

    Dict insertion order is preserved (Python 3.7+). For stratified input the
    ``"init"`` linkmap is stored first and acts as the first merge step, so it
    must be included — not skipped.
    """
    frames = list(linkmaps.values())
    if not frames:
        raise AriadneError("No linkmaps to weave.")

    result = frames[0].copy()
    for frame in frames[1:]:
        shared = list(set(result.columns) & set(frame.columns))
        if not shared:
            raise AriadneError("Cannot chain linkmaps: no shared column found.")
        result = result.merge(frame, on=shared[0], how="inner")

    if from_ not in result.columns or to not in result.columns:
        raise AriadneError(
            f"Expected columns {from_!r} and {to!r} in weaved result."
        )

    return result[[from_, to]].drop_duplicates().reset_index(drop=True)


# ── Complex module parsing and coverage math ──────────────────────────────────

def _process_complex_modules(url: str) -> Dict[str, pd.DataFrame]:
    """Parse a GMM/GBM flat file into module → component → complex → feature tables.

    Each block in the file is separated by '///'. Within a block, the first
    line is the module key and subsequent tab-separated lines are complexes
    (comma-separated features within each complex).
    """
    if str(url).startswith(("http://", "https://")):
        if _requests is None:
            raise AriadneError("'requests' is required to fetch complex modules.")
        resp = _requests.get(url, timeout=60)
        resp.raise_for_status()
        lines = resp.text.splitlines()
    else:
        with open(url, encoding="utf-8") as fh:
            lines = fh.read().splitlines()

    BREAK = "///"
    blocks: Dict[str, List[str]] = {}
    current_key: Optional[str] = None
    current_block: List[str] = []

    for line in lines:
        if line == BREAK:
            if current_key is not None:
                blocks[current_key] = current_block
            current_key = None
            current_block = []
        elif current_key is None:
            current_key = re.sub(r"\t.*", "", line)
        else:
            current_block.append(line)

    if current_key is not None:
        blocks[current_key] = current_block

    module_rows: List[Tuple] = []
    component_rows: List[Tuple] = []
    complex_rows: List[Tuple] = []

    for module, block_lines in blocks.items():
        for idx, bline in enumerate(block_lines):
            component = f"{module}_part_{idx + 1}"
            module_rows.append((module, component))
            for feature_complex in bline.split("\t"):
                feature_complex = feature_complex.strip()
                if not feature_complex:
                    continue
                component_rows.append((component, feature_complex))
                for feature in feature_complex.split(","):
                    feature = feature.strip()
                    if feature:
                        complex_rows.append((feature_complex, feature))

    return {
        "module2component": pd.DataFrame(module_rows, columns=["module", "component"]),
        "component2complex": pd.DataFrame(component_rows, columns=["component", "complex"]),
        "complex2feature": pd.DataFrame(complex_rows, columns=["complex", "feature"]),
    }


def _map_complex_modules(
    linkmaps: Dict[str, pd.DataFrame],
    origin_col: str,
    feature_col: str,
) -> pd.DataFrame:
    """Compute module coverage via sparse matrix math (mirrors R's .map_complex_modules).

    Coverage of module m for origin o = (components of m covered by o) / (total components in m).
    A component is covered if o provides ALL features in at least one of its complexes.
    """
    if not _HAS_SCIPY:
        raise AriadneError(
            "scipy is required for weave_complex coverage computation. "
            "Install with: pip install scipy"
        )

    def _binary_matrix(
        df: pd.DataFrame, row_col: str, col_col: str
    ) -> Tuple[sp.csr_matrix, List, List]:
        row_cat = pd.Categorical(df[row_col])
        col_cat = pd.Categorical(df[col_col])
        mat = sp.csr_matrix(
            (
                np.ones(len(df), dtype=np.float64),
                (row_cat.codes, col_cat.codes),
            ),
            shape=(len(row_cat.categories), len(col_cat.categories)),
        )
        return mat, list(row_cat.categories), list(col_cat.categories)

    o2f = linkmaps.get("origin2feature")
    m2comp = linkmaps.get("module2component")
    comp2c = linkmaps.get("component2complex")
    c2f = linkmaps.get("complex2feature")

    if any(x is None for x in (o2f, m2comp, comp2c, c2f)):
        raise AriadneError("Missing required linkmap tables for complex module coverage.")

    # Align all matrices over a shared feature vocabulary
    all_features = sorted(set(o2f[feature_col]) | set(c2f["feature"]))
    feat_idx = {f: i for i, f in enumerate(all_features)}
    n_feat = len(all_features)

    # origin × feature  (origins as rows)
    origin_cat = pd.Categorical(o2f[origin_col])
    o2f_mat = sp.csr_matrix(
        (
            np.ones(len(o2f), dtype=np.float64),
            (origin_cat.codes, [feat_idx.get(f, 0) for f in o2f[feature_col]]),
        ),
        shape=(len(origin_cat.categories), n_feat),
    )

    # complex × feature
    complex_cat = pd.Categorical(c2f["complex"])
    c2f_mat = sp.csr_matrix(
        (
            np.ones(len(c2f), dtype=np.float64),
            (complex_cat.codes, [feat_idx.get(f, 0) for f in c2f["feature"]]),
        ),
        shape=(len(complex_cat.categories), n_feat),
    )

    # complex2origin: scores = how many features each complex-origin pair shares
    # shape: (complexes × origins)
    complex2origin_scores = c2f_mat @ o2f_mat.T
    complex_sizes = np.asarray(c2f_mat.sum(axis=1)).flatten()
    # Boolean: origin covers complex only when it has ALL features
    complex2origin_bool = (complex2origin_scores >= complex_sizes[:, None]).astype(np.float64)

    # component × complex
    comp_cat = pd.Categorical(comp2c["component"])
    cx_cat = pd.Categorical(comp2c["complex"])
    # Reindex complex axis to match complex_cat.categories
    cx_reindex = {c: i for i, c in enumerate(complex_cat.categories)}
    cx_codes = [cx_reindex.get(c, 0) for c in comp2c["complex"]]
    comp2c_mat = sp.csr_matrix(
        (
            np.ones(len(comp2c), dtype=np.float64),
            (comp_cat.codes, cx_codes),
        ),
        shape=(len(comp_cat.categories), len(complex_cat.categories)),
    )

    # component2origin: component is covered if ANY of its complexes is covered
    component2origin = (comp2c_mat @ complex2origin_bool).astype(bool).astype(np.float64)

    # module × component
    mod_cat = pd.Categorical(m2comp["module"])
    comp_cat2 = pd.Categorical(m2comp["component"])
    comp_reindex = {c: i for i, c in enumerate(comp_cat.categories)}
    comp_codes2 = [comp_reindex.get(c, 0) for c in m2comp["component"]]
    m2comp_mat = sp.csr_matrix(
        (
            np.ones(len(m2comp), dtype=np.float64),
            (mod_cat.codes, comp_codes2),
        ),
        shape=(len(mod_cat.categories), len(comp_cat.categories)),
    )

    # module2origin coverage = sum(covered components) / total components per module
    module2origin_counts = m2comp_mat @ component2origin
    module_sizes = np.asarray(m2comp_mat.sum(axis=1)).flatten()
    coverage_mat = module2origin_counts / module_sizes[:, None]

    # Convert sparse coverage matrix to long DataFrame
    cx_out = sp.coo_matrix(coverage_mat)
    module_names = list(mod_cat.categories)
    origin_names = list(origin_cat.categories)

    return pd.DataFrame(
        {
            origin_col: [origin_names[j] for j in cx_out.col],
            "module": [module_names[i] for i in cx_out.row],
            "cov": cx_out.data,
        }
    )


# ── Internal orchestrator ─────────────────────────────────────────────────────

def _build_path_mf(
    graph: nx.MultiDiGraph,
    from_: str,
    to: str,
    k: int,
    include: Optional[List[str]],
    exclude: Optional[List[str]],
    res_name: Optional[List[str]],
    init: Optional[Union[List[str], pd.DataFrame]],
    prune: bool,
    prune_last: bool,
    verbose: bool,
    timeout: float,
) -> Dict[str, pd.DataFrame]:
    """Build the ordered chain of linkmaps for the path from_ → to."""
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise AriadneError("'timeout' must be a positive number.")
    if not isinstance(prune, bool):
        raise AriadneError("'prune' must be True or False.")
    if not isinstance(verbose, bool):
        raise AriadneError("'verbose' must be True or False.")

    linkmaps: Dict[str, pd.DataFrame] = {}

    # Stratified init: a 2-col DataFrame means the first column stratifies the second
    if isinstance(init, pd.DataFrame) and init.shape[1] == 2:
        init_vars = list(init.columns)
        if verbose:
            print(f"  {init_vars[0]} stratified by {init_vars[1]}")
        linkmaps["init"] = init
        from_ = init_vars[1]
        init_list: Optional[List[str]] = list(init.iloc[:, 1].unique())
    elif init is not None:
        init_list = list(dict.fromkeys(init))  # deduplicate, preserve order
    else:
        init_list = None

    path_df = _draw_path(graph, from_, to, k, include, exclude, res_name)
    path_df = _add_edge_metadata(path_df, graph, internal=True)

    n = len(path_df)
    # prune_vec[i]: whether to carry forward values from step i to step i+1
    prune_vec = [prune] * max(0, n - 1) + [prune_last] + [False]

    for i, (_, step) in enumerate(path_df.iterrows()):
        if verbose:
            print(f"  {step['initFrom']} -({step['source']})-> {step['initTo']}")
        linkmap = _fetch_edge(step, init_list, timeout)
        key = f"{step['from']}2{step['to']}"
        linkmaps[key] = linkmap
        if prune_vec[i]:
            init_list = list(linkmap[step["initTo"]].cat.categories)
        else:
            init_list = None

    return linkmaps


# ── Public API ────────────────────────────────────────────────────────────────

def weave_path(
    graph: nx.MultiDiGraph,
    by: str,
    k: int = 1,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    res_name: Optional[List[str]] = None,
    init: Optional[Union[List[str], pd.DataFrame]] = None,
    prune: bool = True,
    use_names: bool = True,
    verbose: bool = True,
    timeout: float = 1e6,
) -> pd.DataFrame:
    """Build a linkmap by traversing the resource graph from origin to target.

    Equivalent to R's ``weavePath(graph, taxname ~ bugsig, init = tax_labs)``.

    Parameters
    ----------
    graph:
        NetworkX MultiDiGraph returned by ``ariadne()``.
    by:
        Path formula string, e.g. ``"taxname ~ bugsig"`` or ``"ko ~ ec"``.
    k:
        Use the k-th shortest path (1 = shortest).
    include:
        Node names that the chosen path must pass through.
    exclude:
        Node names the chosen path must avoid.
    res_name:
        Restrict graph edges to these resource names only.
    init:
        Seed IDs for the first step (list), or a 2-column DataFrame for
        stratified input (first column = strata, second = IDs).
    prune:
        If True, carry matched values forward to prune each successive step.
    use_names:
        Append a ``<target>.name`` column using ``link_names()``.
    verbose:
        Print step-by-step progress messages.
    timeout:
        HTTP request timeout in seconds.

    Returns
    -------
    pd.DataFrame
        Two-column linkmap ``(origin, target)`` with optional ``target.name``.

    Examples
    --------
    >>> graph = ariadne()
    >>> tax2bugsig = weave_path(graph, "taxname ~ bugsig", init=tax_labs)
    >>> dis2gmm = weave_path(graph, "kegg_disease ~ gmm")
    >>> tax2bugsig_via_taxid = weave_path(
    ...     graph, "taxname ~ bugsig", include=["taxid"], init=tax_labs
    ... )
    """
    from_, to = _parse_by(by)

    linkmaps = _build_path_mf(
        graph, from_, to, k, include, exclude, res_name,
        init, prune, prune, verbose, timeout,
    )
    result = _weave_linkmaps(linkmaps, from_, to)

    if use_names:
        try:
            from ariadnepy.graph._names import link_names
            name_map = link_names(graph, to, list(result[to]))
            if name_map is not None and not name_map.empty:
                result[f"{to}.name"] = result[to].map(
                    dict(zip(name_map.iloc[:, 0], name_map.iloc[:, 1]))
                )
        except (ImportError, AttributeError, Exception):
            pass

    return result


def weave_complex(
    graph: nx.MultiDiGraph,
    by: str,
    k: int = 1,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    res_name: Optional[List[str]] = None,
    init: Optional[Union[List[str], pd.DataFrame]] = None,
    prune: bool = True,
    use_names: bool = True,
    threshold: Optional[float] = None,
    verbose: bool = True,
    timeout: float = 1e6,
) -> pd.DataFrame:
    """Like ``weave_path`` but returns module coverage scores for complex modules.

    For GMM/GBM targets, a feature must fulfill ALL complexes in a module
    component for that component to count as covered. Coverage is the fraction
    of covered components per module per origin.

    Equivalent to R's ``weaveComplex(graph, kegg_disease ~ gmm, threshold=0.8)``.

    Parameters
    ----------
    threshold:
        Only return rows where ``cov >= threshold``. Must be in (0, 1].

    Returns
    -------
    pd.DataFrame
        Three columns: ``(origin, target, cov)`` with optional ``target.name``.

    Examples
    --------
    >>> graph = ariadne()
    >>> dis2gmm = weave_complex(graph, "kegg_disease ~ gmm")
    >>> dis2gmm_filtered = weave_complex(graph, "kegg_disease ~ gmm", threshold=0.8)
    """
    if threshold is not None and not (0 < threshold <= 1):
        raise AriadneError("'threshold' must be a number between 0 and 1.")

    from_, to = _parse_by(by)
    complex_modules = {"gbm", "gmm"}

    _, edge_df = _get_graph_dataframes(graph)

    if to in complex_modules:
        inter_rows = edge_df[edge_df["from"] == to]
        if inter_rows.empty:
            raise AriadneError(
                f"No outgoing edges found from complex module node {to!r}."
            )
        inter_name = inter_rows.iloc[0]["to"]
        module_url = inter_rows.iloc[0].get("url", "")
        module_linkmaps = _process_complex_modules(module_url)
        inner_by = f"{from_} ~ {inter_name}"
    else:
        inner_by = by
        module_linkmaps = {}

    inner_from, inner_to = _parse_by(inner_by)
    linkmaps = _build_path_mf(
        graph, inner_from, inner_to, k, include, exclude, res_name,
        init, prune, True, verbose, timeout,
    )

    if to in complex_modules:
        if verbose:
            print(f"  {inter_name} -(GM)-> {to}")
        inner_result = _weave_linkmaps(linkmaps, from_, inter_name)
        inner_result.columns = ["origin", "feature"]
        module_linkmaps["origin2feature"] = inner_result

        coverage_df = _map_complex_modules(module_linkmaps, "origin", "feature")
        coverage_df = coverage_df.rename(columns={"origin": from_, "module": to})
    else:
        # Non-complex: straightforward coverage ratio per module
        all_cols = list(list(linkmaps.values())[-1].columns)
        feature_col, module_col = all_cols[0], all_cols[-1]

        full_map = _weave_linkmaps(linkmaps, from_, module_col)
        last_lm = list(linkmaps.values())[-1].copy()
        last_lm.columns = ["feature", "module"]

        module_sizes = last_lm.groupby("module")["feature"].nunique()
        merged = full_map.merge(last_lm, left_on=module_col, right_on="feature", how="inner")
        covered = (
            merged.groupby([from_, "module"]).size().reset_index(name="covered")
        )
        covered["total"] = covered["module"].map(module_sizes)
        covered["cov"] = covered["covered"] / covered["total"]
        coverage_df = covered[[from_, "module", "cov"]].rename(columns={"module": to})

    if threshold is not None:
        coverage_df = coverage_df[coverage_df["cov"] >= threshold].reset_index(drop=True)

    coverage_df[from_] = pd.Categorical(coverage_df[from_])
    coverage_df[to] = pd.Categorical(coverage_df[to])

    if use_names:
        try:
            from ariadnepy.graph._names import link_names
            name_map = link_names(graph, to, list(coverage_df[to]))
            if name_map is not None and not name_map.empty:
                coverage_df[f"{to}.name"] = coverage_df[to].map(
                    dict(zip(name_map.iloc[:, 0], name_map.iloc[:, 1]))
                )
        except (ImportError, AttributeError, Exception):
            pass

    return coverage_df


def draw_path(
    graph: nx.MultiDiGraph,
    by: str,
    k: int = 1,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    res_name: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Return a reproducibility table describing each step in the chosen path.

    Shows which resource, version, and URL is used at every hop — useful for
    citing data sources alongside analysis results.

    Equivalent to R's ``drawPath(graph, ko ~ ec, k=4)``.

    Parameters
    ----------
    graph:
        NetworkX MultiDiGraph returned by ``ariadne()``.
    by:
        Path formula string, e.g. ``"ko ~ ec"``.
    k:
        Which of the k-th shortest paths to describe.

    Returns
    -------
    pd.DataFrame
        Columns: from, to, source, version, url.

    Examples
    --------
    >>> graph = ariadne()
    >>> df = draw_path(graph, "ko ~ ec", k=4)
    >>> print(df)
    """
    from_, to = _parse_by(by)
    path_df = _draw_path(graph, from_, to, k, include, exclude, res_name)
    return _add_edge_metadata(path_df, graph, internal=False)


def search_path(
    graph: nx.MultiDiGraph,
    by: str,
    k: int = 1,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    res_name: Optional[List[str]] = None,
) -> None:
    """Print the first k shortest paths between two resources.

    Use this to explore which routes exist before committing to ``weave_path``.

    Equivalent to R's ``searchPath(graph, taxname ~ ko, k=5)``.

    Parameters
    ----------
    graph:
        NetworkX MultiDiGraph returned by ``ariadne()``.
    by:
        Path formula string, e.g. ``"taxname ~ ko"``.
    k:
        Number of paths to display.

    Examples
    --------
    >>> graph = ariadne()
    >>> search_path(graph, "ko ~ ec", k=5)
    >>> search_path(graph, "taxname ~ ko", include=["uniref90"])
    """
    from_, to = _parse_by(by)
    for j in range(1, k + 1):
        try:
            path_df = _draw_path(graph, from_, to, j, include, exclude, res_name)
        except AriadneError as exc:
            print(f"Path {j}: {exc}")
            break
        steps = "".join(
            f" -({row['source']})-> {row['to']}"
            for _, row in path_df.iterrows()
        )
        print(f"Path {j}:\n  {path_df.iloc[0]['from']}{steps}\n")
