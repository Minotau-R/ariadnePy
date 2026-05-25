from __future__ import annotations

from typing import List, Optional

import pandas as pd


def path_to_edge_list(path_df: pd.DataFrame) -> List[tuple]:
    """Convert a draw_path DataFrame into a list of (from, to, source) tuples."""
    return [
        (row["from"], row["to"], row.get("source", ""))
        for _, row in path_df.iterrows()
    ]


def node_colors(
    all_nodes: List[str],
    path_nodes: Optional[List[str]] = None,
    color_default: str = "#AED6F1",
    color_highlight: str = "#E74C3C",
) -> dict:
    """Return a node→color dict, highlighting nodes that appear in path_nodes."""
    path_set = set(path_nodes or [])
    return {
        n: color_highlight if n in path_set else color_default
        for n in all_nodes
    }


def edge_colors(
    all_edges: List[tuple],
    path_edges: Optional[List[tuple]] = None,
    color_default: str = "#BDC3C7",
    color_highlight: str = "#E74C3C",
) -> List[str]:
    """Return a color per edge, highlighting edges in path_edges."""
    path_set = {(u, v) for u, v, *_ in (path_edges or [])}
    return [
        color_highlight if (u, v) in path_set else color_default
        for u, v, *_ in all_edges
    ]
