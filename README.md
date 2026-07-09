# ariadnePy: a Python interface to the ariadne database network <img src="https://raw.githubusercontent.com/Minotau-R/ariadne/devel/inst/assets/ariadne_logo.png" align="right" width="120" />

[![issues](https://img.shields.io/github/issues/Minotau-R/ariadnePy)](https://github.com/Minotau-R/ariadnePy/issues)
[![pulls](https://img.shields.io/github/issues-pr/Minotau-R/ariadnePy)](https://github.com/Minotau-R/ariadnePy/pulls)
[![tests](https://github.com/Minotau-R/ariadnePy/actions/workflows/test.yml/badge.svg)](https://github.com/Minotau-R/ariadnePy/actions/workflows/test.yml)

ariadnePy brings the biological database integration and graph-theory tools of the R package [ariadne](https://github.com/Minotau-R/ariadne) to Python users. It downloads biological resources (Gene Ontology, KEGG, UniProt, BugSigDB, ChocoPhlAn, and more) from [Zenodo](https://zenodo.org) and assembles them into a single [igraph](https://python.igraph.org) `Graph` that can be queried, filtered, and visualised directly in Python.

---

## Installation

```bash
pip install ariadnepy
```

RDS-backed resources (e.g. MSigDB) are supported out of the box — `pyreadr` is installed as a core dependency.

To also use the AnnData integration helpers (`add_modules`, `get_modules`, `process_gene_families`):

```bash
pip install "ariadnepy[anndata]"
```

For development:

```bash
git clone https://github.com/Minotau-R/ariadnePy
cd ariadnePy
pip install -e ".[dev]"
```

---

## Quick start

```python
import ariadnepy

# Build the knowledge graph using default resource versions
# (downloads GML files from Zenodo on first run; cached locally afterwards)
graph = ariadnepy.ariadne()

print(graph)
# IGRAPH U--- <N vertices> <M edges> --

# List all available resource versions
df = ariadnepy.list_resource_versions()
print(df.head())

# Select specific versions
graph = ariadnepy.ariadne(versions={"GO": "2026-01-23", "KEGG": "latest"})

# Explore a few candidate paths between two resources
ariadnepy.search_path(graph, "ko ~ ec", k=3)

# Weave a linkmap along a chosen path
linkmap = ariadnepy.weave_path(graph, "taxname ~ bugsig", init=["s__Bacteroides_fragilis"])

# weave_path / weave_complex also accept an existing pathway DataFrame
# (e.g. from draw_path(), or the bundled pathMeta example) instead of the
# full graph, skipping path search entirely and re-running just those steps:
pathmeta = ariadnepy.load_pathmeta()
chebi2gmm = ariadnepy.weave_path(pathmeta, init=[15377, 30616, 4167])

# Visualise a path on the graph
fig = ariadnepy.plot_path(graph, "ko ~ ec", k=1)
fig.savefig("path.png")
```

---

## Supported resources

| Resource | Description |
|---|---|
| GO | Gene Ontology |
| KEGG | KEGG Pathways & Modules |
| UniProt | UniProt protein database |
| OTT | Open Tree of Life Taxonomy |
| Rhea | Rhea biochemical reactions |
| WoL | Web of Life phylogenetic tree |
| TIGRFAMs | TIGRFAM protein families |
| GM | Gut Metabolic Modules |
| BugSigDB | Bug Signatures Database |
| ChocoPhlAn | MetaPhlAn gene families |
| MSigDB | Molecular Signatures Database |

Run `ariadnepy.list_resource_versions()` for the exact versions available at any time.

---

## Project structure

```
ariadnePy/
├── src/
│   └── ariadnepy/
│       ├── __init__.py        # public API
│       ├── core/               # ariadne() graph builder, GML download/cache, resource versions
│       ├── graph/               # weave_path, weave_complex, draw_path, search_path, link_names
│       ├── io/                  # SPARQL (UniProt/Rhea) and Open Tree of Life query backends
│       ├── plot/                 # plot_path, add_resource
│       ├── resources/             # resource file caching/parsing, bundled example datasets
│       ├── anndata/                # AnnData integration (add_modules, get_modules, process_gene_families)
│       └── exceptions.py
├── tests/                          # mirrors the src/ package layout
├── pyproject.toml
└── README.md
```

---

## Running tests

```bash
pytest
```

---

## License

Artistic License 2.0, matching the [ariadne](https://github.com/Minotau-R/ariadne) R package.
