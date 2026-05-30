from __future__ import annotations

from typing import List, Optional

import igraph as ig
import matplotlib.pyplot as plt

from ariadnepy.graph._weave import _draw_path, _parse_by
from ariadnepy.plot._utils import node_colors, edge_colors


def plot_path(
    graph: ig.Graph,
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
        igraph Graph returned by ``ariadne()``.
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
        path_nodes = [path_df.iloc[0]["from"]] + list(path_df["to"])
        path_edges = [(row["from"], row["to"]) for _, row in path_df.iterrows()]

    # Optionally restrict to path nodes/edges only
    draw_graph = graph
    if focus and path_nodes:
        path_idx = [graph.vs.find(name=n).index for n in path_nodes if n in graph.vs["name"]]
        draw_graph = graph.induced_subgraph(path_idx)

    # igraph Fruchterman-Reingold layout → list of (x, y) coords
    layout = draw_graph.layout("fr", seed=42)
    all_names = draw_graph.vs["name"]
    pos = {name: layout[i] for i, name in enumerate(all_names)}

    all_edges_raw = [
        (draw_graph.vs[e.source]["name"], draw_graph.vs[e.target]["name"],
         e["source"] if "source" in draw_graph.edge_attributes() else "")
        for e in draw_graph.es
    ]

    n_colors_map = node_colors(all_names, path_nodes)
    n_colors = [n_colors_map[n] for n in all_names]
    e_colors = edge_colors([(u, v) for u, v, _ in all_edges_raw], path_edges)

    fig, ax = plt.subplots(figsize=figsize)

    # Draw edges
    for (u, v, src), color in zip(all_edges_raw, e_colors):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5),
        )
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my, src, fontsize=5, ha="center", va="center", color=color)

    # Draw nodes
    xs = [pos[n][0] for n in all_names]
    ys = [pos[n][1] for n in all_names]
    ax.scatter(xs, ys, c=n_colors, s=800, zorder=3)
    for name, x, y in zip(all_names, xs, ys):
        ax.text(x, y - 0.07, name, fontsize=8, ha="center", va="top")

    ax.set_title(f"Path {k}: {by}" if by else "Resource Graph", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    return fig
