"""AnnData integration — Python equivalent of R's SummarizedExperiment helpers."""

from ariadnepy.anndata._modules import add_modules, get_modules
from ariadnepy.anndata._humann import process_gene_families

__all__ = ["add_modules", "get_modules", "process_gene_families"]
