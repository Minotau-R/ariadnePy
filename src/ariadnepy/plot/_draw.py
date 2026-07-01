from __future__ import annotations

import igraph as ig
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ariadnepy.graph._weave import _draw_path, _parse_by

_C_NODE      = "#FF8C00"   # darkorange — matches R ariadne
_C_EDGE_PATH = "#E74C3C"   # red
_C_EDGE_BG   = "#CCCCCC"   # grey80


def plot_path(
    graph: ig.Graph,
    by: str | None = None,
    k: int = 1,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    res_name: list[str] | None = None,
    prune: bool = False,
    focus: bool = False,
    figsize: tuple = (12, 8),
) -> Figure:
    """Visualise the resource graph with an optional highlighted path.

    Equivalent to R's ``plotPath(graph, "ko ~ ec", k=3)``.

    Parameters
    ----------
    graph : igraph.Graph
        Graph returned by ``ariadne()``.
    by : str, optional
        Path formula, e.g. ``"ko ~ ec"``. If None, the full graph is drawn.
    k : int
        Which shortest path to highlight (1 = shortest).
    include : list of str, optional
        Nodes the path must pass through.
    exclude : list of str, optional
        Nodes the path must avoid.
    res_name : list of str, optional
        Restrict path edges to these resource names.
    prune : bool
        If True, show only path nodes and edges.
        Equivalent to R's ``prune=TRUE``.
    focus : bool
        If True, compute layout on path subgraph only (removes non-path nodes
        from the canvas entirely). Equivalent to R's ``focus=TRUE``.
    figsize : tuple
        Matplotlib figure size ``(width, height)`` in inches.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> fig = plot_path(graph, "ec ~ ko", k=3)             # full graph, path highlighted
    >>> fig = plot_path(graph, "ec ~ ko", k=3, prune=True) # only path shown
    >>> fig = plot_path(graph)                              # full resource graph
    >>> fig.savefig("path.png")
    """
    if prune and by is None:
        raise ValueError("'prune' requires 'by' to be specified.")

    path_nodes: list[str] = []
    path_edges: list[tuple] = []
    path_edge_labels: dict = {}

    if by is not None:
        from_, to = _parse_by(by)
        path_df = _draw_path(graph, from_, to, k, include, exclude, res_name)
        path_nodes = [path_df.iloc[0]["from"]] + list(path_df["to"])
        for _, row in path_df.iterrows():
            path_edges.append((row["from"], row["to"]))
            path_edge_labels[(row["from"], row["to"])] = row.get("source", "")

    # Use frozensets so edge lookup is direction-agnostic, matching R's sort(c(from,to)) approach.
    path_edge_set = {frozenset(e) for e in path_edges}

    # prune/focus both restrict layout to the path subgraph.
    if path_nodes and (focus or prune):
        path_idx = [
            graph.vs.find(name=n).index
            for n in path_nodes
            if n in graph.vs["name"]
        ]
        draw_graph = graph.induced_subgraph(path_idx)
    else:
        draw_graph = graph

    # Kamada-Kawai layout — closest available equivalent to R ariadne's "stress" layout
    layout = draw_graph.layout("kk")
    all_names = draw_graph.vs["name"]
    pos = {name: tuple(layout[i]) for i, name in enumerate(all_names)}

    all_edges_raw = [
        (
            draw_graph.vs[e.source]["name"],
            draw_graph.vs[e.target]["name"],
            e["source"] if "source" in draw_graph.edge_attributes() else "",
        )
        for e in draw_graph.es
    ]

    # plt.close(fig) before returning removes it from pyplot's display queue,
    # preventing Jupyter from rendering it twice while still showing it as the
    # cell's return value via Jupyter's own repr mechanism.
    fig, ax = plt.subplots(figsize=figsize)

    # ── Edges ─────────────────────────────────────────────────────────────────
    for u, v, src in all_edges_raw:
        if u not in pos or v not in pos:
            continue
        is_path_edge = frozenset((u, v)) in path_edge_set
        color = _C_EDGE_PATH if is_path_edge else _C_EDGE_BG
        lw    = 2.0 if is_path_edge else 1.0
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot(
            [x0, x1], [y0, y1],
            color=color, lw=lw, zorder=1, solid_capstyle="round",
        )
        if is_path_edge and src:
            ax.text(
                (x0 + x1) / 2, (y0 + y1) / 2, src,
                fontsize=8, ha="center", va="center",
                fontweight="bold", color=_C_EDGE_PATH, zorder=4,
                bbox={
                    "boxstyle": "round,pad=0.2", "fc": "white",
                    "ec": _C_EDGE_PATH, "linewidth": 0.8, "alpha": 0.9,
                },
            )

    # ── Nodes ─────────────────────────────────────────────────────────────────
    draw_names = list(all_names)
    if draw_names:
        ax.scatter(
            [pos[n][0] for n in draw_names],
            [pos[n][1] for n in draw_names],
            c=_C_NODE, s=150, zorder=3, edgecolors="none",
        )

    # ── Labels ────────────────────────────────────────────────────────────────
    ys_drawn = [pos[n][1] for n in draw_names] if draw_names else [0.0]
    y_span = (max(ys_drawn) - min(ys_drawn)) if len(ys_drawn) > 1 else 1.0
    offset = y_span * 0.06

    for name in draw_names:
        x, y = pos[name]
        ax.text(
            x, y - offset, name,
            fontsize=9, ha="center", va="top", color="#333333",
            zorder=5,
            bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.7},
        )

    title = f"Path {k}: {by}" if by else "Resource Graph"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.axis("off")
    fig.tight_layout()
    plt.close(fig)
    return fig
