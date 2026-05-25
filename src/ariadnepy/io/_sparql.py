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

_ENDPOINTS = {
    "UniProt": "https://sparql.uniprot.org/sparql",
    "Rhea":    "https://sparql.rhea-db.org/sparql",
}

# SPARQL PREFIX block shared by all queries
_PREFIXES = """
PREFIX chebislash: <http://purl.obolibrary.org/obo/chebi/>
PREFIX enzyme:     <http://purl.uniprot.org/enzyme/>
PREFIX obo:        <http://purl.obolibrary.org/obo/>
PREFIX oboInOwl:   <http://www.geneontology.org/formats/oboInOwl#>
PREFIX protein:    <http://purl.uniprot.org/uniprot/>
PREFIX rdfs:       <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rh:         <http://rdf.rhea-db.org/>
PREFIX taxon:      <http://purl.uniprot.org/taxonomy/>
PREFIX uniref:     <http://purl.uniprot.org/uniref/>
PREFIX up:         <http://purl.uniprot.org/core/>
"""

# Maps (from, to) sorted alphabetically → SPARQL WHERE clause body
_TRIPLE_MAP = {
    "taxid2uniref": """
        ?uniref up:member ?member.
        ?member up:organism ?taxid.
    """,
    "taxname2taxid": """
        ?taxid up:scientificName ?sciName; up:rank ?rank.
        BIND(lcase(substr(strafter(str(?rank),'Rank_'),1,1)) as ?prefix)
        BIND(concat(?prefix,'__',?sciName) as ?taxname)
    """,
    "taxname2uniref": """
        ?uniref up:member ?member.
        ?member up:organism ?taxid.
        ?taxid up:scientificName ?sciName; up:rank ?rank.
        BIND(lcase(substr(strafter(str(?rank),'Rank_'),1,1)) as ?prefix)
        BIND(concat(?prefix,'__',?sciName) as ?taxname)
    """,
    "uniprotkb2uniref": """
        ?uniprotkb up:representativeFor ?uniref.
    """,
    "enzyme2uniprotkb": """
        ?uniprotkb (up:enzyme|up:domain/up:enzyme|up:component/up:enzyme) ?enzyme.
    """,
    "enzyme2uniref": """
        ?uniprotkb up:representativeFor ?uniref.
        ?uniprotkb (up:enzyme|up:domain/up:enzyme|up:component/up:enzyme) ?enzyme.
    """,
    "rhea2uniprotkb": """
        ?uniprotkb up:annotation/up:catalyticActivity/up:catalyzedReaction ?rhea.
    """,
    "rhea2uniref": """
        ?uniprotkb up:representativeFor ?uniref.
        ?uniprotkb up:annotation/up:catalyticActivity/up:catalyzedReaction ?rhea.
    """,
    "chebi2rhea": """
        ?rhea rdfs:subClassOf rh:Reaction.
        ?rhea rh:side/rh:contains/rh:compound ?compound.
        ?compound rh:chebi|rh:reactivePart/rh:chebi|rh:underlyingChebi ?chebi.
    """,
    "enzyme2rhea": """
        ?rhea rdfs:subClassOf rh:Reaction.
        ?rhea rh:ec ?enzyme.
    """,
    "chebi2inchi": """
        ?chebi rdfs:subClassOf rh:Reaction.
        ?chebi rh:side/rh:contains/rh:compound ?compound.
        ?compound rh:chebi|rh:reactivePart/rh:chebi|rh:underlyingChebi ?chebi.
        ?chebi chebislash:inchi ?inchi.
    """,
    "chebi2inchikey": """
        ?chebi chebislash:inchikey ?inchikey.
    """,
    "chebi2smiles": """
        ?chebi chebislash:smiles ?smiles.
    """,
}

# IRI prefix used when injecting VALUES clauses
_IRI_PREFIX = {
    "chebi":     "obo:CHEBI_",
    "uniprotkb": "protein:",
    "uniref":    "uniref:",
    "taxid":     "taxon:",
    "enzyme":    "enzyme:",
    "rhea":      "rh:",
}


def _iri_wrap(ids: List[str], from_: str) -> List[str]:
    """Wrap raw IDs with their SPARQL IRI prefix."""
    prefix = _IRI_PREFIX.get(from_)
    if prefix is None:
        return [f'"{x}"' for x in ids]
    return [f"{prefix}{x}" for x in ids]


def _triple_clause(from_: str, to: str) -> str:
    key = "2".join(sorted([from_, to]))
    clause = _TRIPLE_MAP.get(key)
    if clause is None:
        raise AriadneError(
            f"No SPARQL triple defined for {from_!r} → {to!r}. "
            "Add a custom clause to _TRIPLE_MAP or use a file-backed resource."
        )
    return clause


def _send_sparql(query: str, endpoint: str, timeout: float) -> pd.DataFrame:
    """POST a SPARQL query and return results as a DataFrame."""
    if _requests is None:
        raise AriadneDownloadError(
            "'requests' package is required for SPARQL queries. "
            "Install with: pip install requests"
        )
    url = _ENDPOINTS.get(endpoint)
    if url is None:
        raise AriadneError(f"Unknown SPARQL endpoint: {endpoint!r}")
    resp = _requests.post(
        url,
        data={"query": query},
        headers={"Accept": "text/csv"},
        timeout=timeout,
    )
    resp.raise_for_status()
    from io import StringIO
    return pd.read_csv(StringIO(resp.text))


def _build_query(
    from_: str,
    to: str,
    clause: str,
    batch: Optional[List[str]],
    uniref_identity: Optional[float],
) -> str:
    values_block = ""
    if batch:
        spec_from = "sciName" if from_ == "taxname" else from_
        wrapped = " ".join(_iri_wrap(batch, from_))
        values_block = f"VALUES ?{spec_from} {{ {wrapped} }}\n"

    identity_block = ""
    if uniref_identity is not None and "uniref" in (from_, to):
        identity_block = f"?uniref up:identity {uniref_identity}.\n"

    return (
        f"{_PREFIXES}\n"
        f"SELECT DISTINCT ?{from_} ?{to}\n"
        f"WHERE {{\n"
        f"{values_block}"
        f"{identity_block}"
        f"{clause}\n"
        f"}}"
    )


def query_sparql(
    from_: str,
    to: str,
    endpoint: str,
    init: Optional[List[str]] = None,
    timeout: float = 1e6,
    batch_size: int = 500,
    workers: int = 4,
) -> pd.DataFrame:
    """Run a SPARQL query against UniProt or Rhea and return a linkmap DataFrame.

    Parameters
    ----------
    from_:
        Source variable name (e.g. ``"uniref"``, ``"taxname"``).
    to:
        Target variable name (e.g. ``"enzyme"``, ``"taxid"``).
    endpoint:
        ``"UniProt"`` or ``"Rhea"``.
    init:
        Seed IDs to inject as a VALUES clause (batched automatically).
    timeout:
        Per-request timeout in seconds.
    batch_size:
        Maximum number of IDs per SPARQL request.
    workers:
        Number of parallel workers for batched requests.

    Returns
    -------
    pd.DataFrame
        Two-column DataFrame ``(from_, to)``.

    Examples
    --------
    >>> from ariadnepy.io import query_sparql
    >>> df = query_sparql("uniref", "enzyme", "UniProt", init=["UniRef90_A0A000"])
    """
    uniref_identity: Optional[float] = None
    if to == "uniref50":
        uniref_identity = 0.5
    elif to == "uniref90":
        uniref_identity = 0.9

    clause = _triple_clause(from_, to)

    if not init:
        query = _build_query(from_, to, clause, None, uniref_identity)
        df = _send_sparql(query, endpoint, timeout)
        df.columns = [from_, to]
        return df

    # Split init into batches
    batches = [init[i : i + batch_size] for i in range(0, len(init), batch_size)]

    def _run_batch(batch: List[str]) -> pd.DataFrame:
        query = _build_query(from_, to, clause, batch, uniref_identity)
        return _send_sparql(query, endpoint, timeout)

    with ThreadPoolExecutor(max_workers=min(workers, len(batches))) as pool:
        results = list(pool.map(_run_batch, batches))

    df = pd.concat(results, ignore_index=True)
    if not df.empty:
        df.columns = [from_, to]
    return df
