import ariadnepy

# Build the knowledge graph using default resource versions
# (downloads GML files from Zenodo on first run; cached locally afterwards)
g = ariadnepy.ariadne()

print(g)
# MultiDiGraph with N nodes and M edges

# List all available resource versions
# df = ariadnepy.list_resource_versions()
# print(df.head())

# # Select specific versions
# g = ariadnepy.ariadne(versions={"GO": "2026-01-23", "KEGG": "latest"})

# Example code in tests
