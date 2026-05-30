from __future__ import annotations

from typing import List, Optional

import networkx as nx

from ariadnepy.graph._weave import _draw_path, _parse_by
from ariadnepy.plot._utils import node_colors, edge_colors

import matplotlib.pyplot as plt


def plot_path(
    graph: nx.MultiDiGraph,
    by: Optional[str] = None,
    k: int = 1,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    res_name: Optional[List[str]] = None,
    focus: bool = False,
    figsize: tuple = (12, 8),
) -> "plt.Figure":
    """Visualise the resource graph with an optional highlighted path.

    Equivalent to R's ``plotPath(graph, ko ~ ec, k=5)``.

    Parameters
    ----------
    graph:
        NetworkX MultiDiGraph returned by ``ariadne()``.
    by:
        Path formula string, e.g. ``"ko ~ ec"``. If None, the full graph
        is drawn without any path highlighted.
    k:
        Which of the k-th shortest paths to highlight.
    include:
        Nodes the highlighted path must pass through.
    exclude:
        Nodes the highlighted path must avoid.
    res_name:
        Restrict highlighted path edges to these resource names.
    focus:
        If True, only draw the nodes and edges that belong to the path.
    figsize:
        Matplotlib figure size ``(width, height)`` in inches.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> from ariadnepy import ariadne
    >>> from ariadnepy.plot import plot_path
    >>> graph = ariadne()
    >>> fig = plot_path(graph, "ko ~ ec", k=5)
    >>> fig.savefig("path.png")
    """
    path_nodes: List[str] = []
    path_edges: List[tuple] = []

    if by is not None:
        from_, to = _parse_by(by)
        path_df = _draw_path(graph, from_, to, k, include, exclude, res_name)
        path_nodes = (
            [path_df.iloc[0]["from"]]
            + list(path_df["to"])
        )
        path_edges = [
            (row["from"], row["to"]) for _, row in path_df.iterrows()
        ]

    draw_graph = graph
    if focus and path_nodes:
        draw_graph = graph.subgraph(path_nodes).copy()

    # Build layout
    pos = nx.spring_layout(draw_graph, seed=42)

    all_nodes = list(draw_graph.nodes())
    all_edges = list(draw_graph.edges(data=True))

    n_colors = [
        node_colors(all_nodes, path_nodes)[n] for n in all_nodes
    ]
    e_colors = edge_colors(
        [(u, v) for u, v, _ in all_edges],
        path_edges,
    )

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx(
        draw_graph,
        pos=pos,
        ax=ax,
        node_color=n_colors,
        edge_color=e_colors,
        node_size=800,
        font_size=8,
        arrows=True,
        arrowsize=15,
        width=1.5,
    )

    # Edge labels: resource source names
    edge_labels = {
        (u, v): d.get("source", "")
        for u, v, d in all_edges
    }
    nx.draw_networkx_edge_labels(
        draw_graph, pos=pos, edge_labels=edge_labels,
        font_size=6, ax=ax,
    )

    ax.set_title(
        f"Path {k}: {by}" if by else "Resource Graph",
        fontsize=12,
    )
    ax.axis("off")
    plt.tight_layout()
    return fig
