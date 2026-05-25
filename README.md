# ariadnePy

Python interface to the [ariadne](https://github.com/Minotau-R/ariadne) multi-omic knowledge graph.

ariadnePy brings the biological database integration and graph-theory tools of the R package **ariadne** to Python users. It downloads biological resources (Gene Ontology, KEGG, UniProt, BugSigDB, ChocoPhlAn, and more) from [Zenodo](https://zenodo.org) and assembles them into a single [NetworkX](https://networkx.org) `MultiDiGraph` that can be queried, filtered, and visualised directly in Python.

---

## Installation

```bash
pip install ariadnepy
```

To also read RDS files (required for MSigDB):

```bash
pip install "ariadnepy[rds]"
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
g = ariadnepy.ariadne()

print(g)
# MultiDiGraph with N nodes and M edges

# List all available resource versions
df = ariadnepy.list_resource_versions()
print(df.head())

# Select specific versions
g = ariadnepy.ariadne(versions={"GO": "2026-01-23", "KEGG": "latest"})
```

---

<!-- ## Supported resources

| Resource | Description |
|---|---|
| GO | Gene Ontology |
| KEGG | KEGG Pathways & Modules |
| UniProt | UniProt protein database |
| OTT | Open Tree of Life Taxonomy |
| Rhea | Rhea biochemical reactions |
| WoL | Web of Life phylogenetic tree |
| TIGRFAMs | TIGRFAM protein families |
| GM | Gut Metabolome modules |
| BugSigDB | Bug Signatures Database |
| ChocoPhlAn | MetaPhlAn gene families |
| MSigDB | Molecular Signatures Database |

--- -->

## Project structure

```
ariadnePy/
├── src/
│   └── ariadnepy/
│       ├── __init__.py     # public API
│       ├── _core.py        # ariadne() graph builder
│       ├── _cache.py       # resource downloading & caching
│       ├── _utils.py       # utility functions
│       └── _custom.py      # custom resource support
├── tests/
│   ├── test_core.py
│   ├── test_cache.py
│   └── test_utils.py
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

