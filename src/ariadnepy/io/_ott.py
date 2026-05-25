from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import pandas as pd

from ariadnepy.exceptions import AriadneError, AriadneDownloadError

try:
    import requests as _requests
except ImportError:
    _requests = None

_TNRS_URL      = "https://api.opentreeoflife.org/v3/tnrs/match_names"
_TAXON_INFO_URL = "https://api.opentreeoflife.org/v3/taxonomy/taxon_info"

_EXTERNAL_SOURCES = {"ncbi", "gbif", "worms", "if", "irmng"}


def _strip_rank_prefix(names: List[str]) -> List[str]:
    return [re.sub(r"^[a-z]__", "", n) for n in names]


def _query_tnrs(
    names: List[str],
    to: str,
    timeout: float,
    batch_size: int = 1000,
    workers: int = 4,
) -> List[Optional[str]]:
    """Match taxonomy names via TNRS API, return OTT id (or other target) per name."""
    if _requests is None:
        raise AriadneDownloadError("'requests' is required for OTT queries.")

    batches = [names[i : i + batch_size] for i in range(0, len(names), batch_size)]

    def _run(batch: List[str]) -> List[Optional[str]]:
        resp = _requests.post(
            _TNRS_URL,
            json={"names": batch, "do_approximate_matching": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        out: List[Optional[str]] = []
        for entry in resp.json().get("results", []):
            matches = entry.get("matches", [])
            if not matches:
                out.append(None)
                continue
            taxon = matches[0].get("taxon", {})
            if to == "ott":
                val = str(taxon.get("ott_id", "")) or None
            elif to == "taxname":
                rank = taxon.get("rank", "")[:1]
                name = taxon.get("unique_name", "")
                val = f"{rank}__{name}" if rank and name else None
            else:
                sources = taxon.get("tax_sources", [])
                matched = [s for s in sources if s.startswith(to + ":")]
                val = matched[0].split(":", 1)[1] if matched else None
            out.append(val)
        return out

    with ThreadPoolExecutor(max_workers=min(workers, len(batches))) as pool:
        nested = list(pool.map(_run, batches))
    return [item for batch in nested for item in batch]


def _query_taxon_info(
    source_id: str,
    to: str,
    prefix: str,
    timeout: float,
) -> Optional[str]:
    """Query the OTT taxon_info endpoint for a single ID."""
    if _requests is None:
        raise AriadneDownloadError("'requests' is required for OTT queries.")
    try:
        resp = _requests.post(
            _TAXON_INFO_URL,
            json={prefix: source_id},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    if to == "ott":
        return str(data.get("ott_id", "")) or None
    if to == "taxname":
        rank = (data.get("rank") or "")[:1]
        name = data.get("unique_name", "")
        return f"{rank}__{name}" if rank and name else None
    sources = data.get("tax_sources", [])
    matched = [s for s in sources if s.startswith(to + ":")]
    if matched:
        return matched[0].split(":", 1)[1]
    return None


def query_ott(
    from_: str,
    to: str,
    init: Optional[List[str]],
    timeout: float = 1e6,
    batch_size: int = 1000,
    workers: int = 4,
) -> pd.DataFrame:
    """Fetch taxonomy mappings from the Open Tree of Life API.

    Supports ``taxname``, ``ott``, and external IDs (``ncbi``, ``gbif``,
    ``worms``, ``if``, ``irmng``) as the source (``from_``).

    Parameters
    ----------
    from_:
        Source type: ``"taxname"``, ``"ott"``, or an external DB name.
    to:
        Target type: ``"ott"``, ``"taxname"``, or an external DB name.
    init:
        List of source IDs/names to query.
    timeout:
        Per-request HTTP timeout in seconds.
    batch_size:
        Number of names per TNRS batch (for taxname queries).
    workers:
        Parallel workers for batched requests.

    Returns
    -------
    pd.DataFrame
        Two-column DataFrame ``(from_, to)`` with matched pairs.

    Examples
    --------
    >>> from ariadnepy.io import query_ott
    >>> df = query_ott("taxname", "ott", init=["Bacteroides fragilis"])
    """
    if init is None:
        raise AriadneError("'init' must be provided for OTT queries.")

    clean = _strip_rank_prefix(init)

    if from_ == "taxname":
        values = _query_tnrs(clean, to, timeout, batch_size, workers)
        rows = [
            (orig, val)
            for orig, val in zip(init, values)
            if val is not None
        ]
        df = pd.DataFrame(rows, columns=[from_, to])

    elif from_ == "ott":
        def _fetch(raw_id: str) -> Optional[str]:
            return _query_taxon_info(raw_id, to, "ott_id", timeout)

        with ThreadPoolExecutor(max_workers=min(workers, len(clean))) as pool:
            values = list(pool.map(_fetch, clean))
        rows = [(orig, val) for orig, val in zip(init, values) if val is not None]
        df = pd.DataFrame(rows, columns=[from_, to])

    elif from_ in _EXTERNAL_SOURCES:
        def _fetch_ext(raw_id: str) -> Optional[str]:
            return _query_taxon_info(f"{from_}:{raw_id}", to, "source_id", timeout)

        with ThreadPoolExecutor(max_workers=min(workers, len(clean))) as pool:
            values = list(pool.map(_fetch_ext, clean))
        rows = [(orig, val) for orig, val in zip(init, values) if val is not None]
        df = pd.DataFrame(rows, columns=[from_, to])

    else:
        raise AriadneError(
            f"Unsupported OTT source: {from_!r}. "
            f"Must be 'taxname', 'ott', or one of {sorted(_EXTERNAL_SOURCES)}."
        )

    # Strip any remaining source prefixes from result column
    df[to] = df[to].str.replace(r"^.+:", "", regex=True)
    return df.dropna().reset_index(drop=True)
