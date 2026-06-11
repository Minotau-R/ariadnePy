from __future__ import annotations

import math
from typing import List, Optional, Tuple

import igraph as ig
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ariadnepy.graph._weave import _draw_path, _parse_by


# Colours matching R ariadne's plotPath (ggraph + darkorange/red scheme)
_C_NODE_DEFAULT = "#FF8C00"   # darkorange
_C_NODE_PATH    = "#E74C3C"   # red
_C_EDGE_DEFAULT = "#C8C8C8"   # grey80
_C_EDGE_PATH    = "#E74C3C"   # red
_ALPHA_BG       = 0.12        # opacity for non-path elements when path shown


def _arrow(ax, p0: Tuple, p1: Tuple, color: str, lw: float, alpha: float,
           shrink_pts: float = 13.0, zorder: int = 1) -> None:
    """Draw one directed edge as an annotate arrow, shrunk away from node centres."""
    ax.annotate(
        "", xy=p1, xytext=p0,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            connectionstyle="arc3,rad=0.05",
            shrinkA=shrink_pts,
            shrinkB=shrink_pts,
        ),
        alpha=alpha,
        zorder=zorder,
    )


def plot_path(
    graph: ig.Graph,
    by: Optional[str] = None,
    k: int = 1,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    res_name: Optional[List[str]] = None,
    focus: Optional[bool] = None,
    figsize: tuple = (12, 8),
) -> Figure:
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
        If True, draw only the path nodes/edges (clean, readable).
        If False, draw the full graph with path highlighted.
        Defaults to True when ``by`` is given, False when ``by`` is None.
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
    >>> fig = plot_path(graph, "ko ~ ec")           # focused path (default)
    >>> fig = plot_path(graph, "ko ~ ec", focus=False)  # full graph
    >>> fig = plot_path(graph)                      # full resource graph
    >>> fig.savefig("path.png")
    """
    path_nodes: List[str] = []
    path_edges: List[tuple] = []

    if by is not None:
        from_, to = _parse_by(by)
        path_df = _draw_path(graph, from_, to, k, include, exclude, res_name)
        path_nodes = [path_df.iloc[0]["from"]] + list(path_df["to"])
        path_edges = [(row["from"], row["to"]) for _, row in path_df.iterrows()]

    # Default: focus=True when a path is requested, False for full-graph view
    if focus is None:
        focus = bool(path_nodes)

    draw_graph = graph
    if focus and path_nodes:
        path_idx = [graph.vs.find(name=n).index for n in path_nodes if n in graph.vs["name"]]
        draw_graph = graph.induced_subgraph(path_idx)

    # Kamada-Kawai layout: deterministic, minimises edge crossing — mirrors
    # R ariadne's "stress" layout from graphlayouts in readability.
    layout = draw_graph.layout("kk")
    all_names = draw_graph.vs["name"]
    pos = {name: tuple(layout[i]) for i, name in enumerate(all_names)}

    path_set = set(path_nodes)
    path_edge_set = {(u, v) for u, v in path_edges}

    all_edges_raw = [
        (draw_graph.vs[e.source]["name"],
         draw_graph.vs[e.target]["name"],
         e["source"] if "source" in draw_graph.edge_attributes() else "")
        for e in draw_graph.es
    ]

    has_path = bool(path_nodes)
    # Path nodes are drawn larger so they dominate the composition
    node_size_bg   = 400
    node_size_path = 700
    # Shrink background arrows to stop at the disk edge (radius ≈ sqrt(s / π))
    shrink_bg = math.sqrt(node_size_bg / math.pi)

    fig, ax = plt.subplots(figsize=figsize)

    # ── Edges ──────────────────────────────────────────────────────────────────
    # Draw background edges first (zorder 1), then path edges on top (zorder 3+)
    # so the highlighted path is never buried under the grey graph.
    bg_alpha = _ALPHA_BG if has_path else 1.0

    for u, v, _ in all_edges_raw:
        if (u, v) not in path_edge_set:
            _arrow(ax, pos[u], pos[v], color=_C_EDGE_DEFAULT, lw=0.8,
                   alpha=bg_alpha, shrink_pts=shrink_bg, zorder=1)

    for u, v, src in all_edges_raw:
        if (u, v) not in path_edge_set:
            continue
        # Small, fixed shrink — oversized shrink makes the arrow invisible when
        # path nodes sit close together in the layout.
        s = 6.0
        # Glow layer: wide semi-transparent halo so the edge reads clearly
        _arrow(ax, pos[u], pos[v], color=_C_EDGE_PATH, lw=8.0,
               alpha=0.25, shrink_pts=s, zorder=3)
        # Solid arrow on top
        _arrow(ax, pos[u], pos[v], color=_C_EDGE_PATH, lw=2.5,
               alpha=1.0, shrink_pts=s, zorder=4)

        if src:
            mx = (pos[u][0] + pos[v][0]) / 2
            my = (pos[u][1] + pos[v][1]) / 2
            ax.text(
                mx, my, src,
                fontsize=9, ha="center", va="center",
                fontweight="bold", color=_C_EDGE_PATH, zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=_C_EDGE_PATH,
                          linewidth=1.0, alpha=0.95),
            )

    # ── Nodes ──────────────────────────────────────────────────────────────────

    # Background (non-path) nodes drawn first so path nodes sit on top
    bg_names = [n for n in all_names if n not in path_set]
    if bg_names:
        bg_alpha = _ALPHA_BG if has_path else 1.0
        ax.scatter(
            [pos[n][0] for n in bg_names], [pos[n][1] for n in bg_names],
            c=_C_NODE_DEFAULT, s=node_size_bg, zorder=4, alpha=bg_alpha,
            edgecolors="none",
        )

    if path_set:
        pn = [n for n in path_nodes if n in pos]
        ax.scatter(
            [pos[n][0] for n in pn], [pos[n][1] for n in pn],
            c=_C_NODE_PATH, s=node_size_path, zorder=5, alpha=1.0,
            edgecolors="white", linewidths=2.5,   # white border makes them pop
        )

    # ── Labels ─────────────────────────────────────────────────────────────────
    ys_all = [pos[n][1] for n in all_names]
    y_span = (max(ys_all) - min(ys_all)) if len(ys_all) > 1 else 1.0
    offset = y_span * 0.07

    for name in all_names:
        x, y = pos[name]
        is_path = name in path_set
        label_alpha = 1.0 if (is_path or not has_path) else _ALPHA_BG + 0.15
        ax.text(
            x, y - offset, name,
            fontsize=9 if is_path else 7,
            ha="center", va="top",
            fontweight="bold" if is_path else "normal",
            color="#111111" if is_path else "#555555",
            alpha=label_alpha,
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7),
        )

    title = f"Path {k}: {by}" if by else "Resource Graph"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    return fig
