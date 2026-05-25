"""ariadnepy — Python extension of the ariadne multi-omic graph package."""

from ariadnepy._version import __version__
from ariadnepy.exceptions import (
    AriadneError,
    AriadneDownloadError,
    AriadneVersionError,
    AriadneParseError,
    AriadnePathError,
    AriadneCacheError,
)
from ariadnepy.core._graph import ariadne
from ariadnepy.core._versions import list_resource_versions
from ariadnepy.graph._weave import weave_path, weave_complex, draw_path, search_path
from ariadnepy.graph._names import link_names
from ariadnepy.plot._draw import plot_path
from ariadnepy.plot._custom import add_resource
from ariadnepy.resources._data import load_butyrate

__all__ = [
    "__version__",
    # exceptions
    "AriadneError",
    "AriadneDownloadError",
    "AriadneVersionError",
    "AriadneParseError",
    "AriadnePathError",
    "AriadneCacheError",
    # core
    "ariadne",
    "list_resource_versions",
    # graph traversal
    "weave_path",
    "weave_complex",
    "draw_path",
    "search_path",
    "link_names",
    # plot
    "plot_path",
    "add_resource",
    # data
    "load_butyrate",
]
