from __future__ import annotations

import igraph as ig
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ariadnepy.graph._weave import _draw_path, _get_sorted_edge_key, _parse_by

_C_NODE      = "#FF8C00"   # darkorange — matches R ariadne
_C_EDGE_PATH = "#E74C3C"   # red
_C_EDGE_BG   = "#CCCCCC"   # grey80
_FADE_ALPHA  = 0.15        # opacity for edges/nodes faded out by prune/res_name


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
        Resource names to highlight. With ``by`` set, restricts path-finding
        to these resources; with ``by=None``, fades every edge/node not
        belonging to these resources across the whole graph (R's
        ``plotPath(graph, res_name=["KEGG", "WoL"])`` usage).
    prune : bool
        If True, fade every edge/node not on the chosen path (requires
        ``by``). Equivalent to R's ``prune=TRUE``.
    focus : bool
        If True, drop every faded (non-highlighted) edge/node from the plot
        entirely, instead of just fading them. Equivalent to R's
        ``focus=TRUE``. Has no effect unless ``prune`` or ``res_name`` is
        also active, since otherwise nothing is faded to begin with.
    figsize : tuple
        Matplotlib figure size ``(width, height)`` in inches.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> fig = plot_path(graph, "ec ~ ko", k=3)               # full graph, path highlighted
    >>> fig = plot_path(graph, "ec ~ ko", k=3, prune=True)   # non-path faded
    >>> fig = plot_path(graph, "ec ~ ko", k=3, focus=True)   # only path shown
    >>> fig = plot_path(graph, res_name=["KEGG", "WoL"])     # resource highlighted
    >>> fig = plot_path(graph)                                # full resource graph
    >>> fig.savefig("path.png")
    """
    if not isinstance(prune, bool):
        raise ValueError("'prune' must be True or False.")
    if not isinstance(focus, bool):
        raise ValueError("'focus' must be True or False.")
    if prune and by is None:
        raise ValueError("'prune' requires 'by' to be specified.")

    has_source_attr = "source" in graph.edge_attributes()
    edges_info = []
    for e in graph.es:
        u = graph.vs[e.source]["name"]
        v = graph.vs[e.target]["name"]
        src = e["source"] if has_source_attr else ""
        edges_info.append({"u": u, "v": v, "source": src})

    path_edge_keys: set[str] = set()
    if by is not None:
        from_, to = _parse_by(by)
        path_df = _draw_path(graph, from_, to, k, include, exclude, res_name)
        for _, row in path_df.iterrows():
            path_edge_keys.add(
                _get_sorted_edge_key(row["from"], row["to"], row.get("source", ""))
            )

    for info in edges_info:
        info["mark"] = _get_sorted_edge_key(info["u"], info["v"], info["source"]) in path_edge_keys

    # alpha marks what's highlighted vs faded; mark (above) marks what's coloured red.
    # These are independent axes, same as R's plotPath.
    if prune:
        for info in edges_info:
            info["alpha"] = info["mark"]
    elif res_name is not None:
        for info in edges_info:
            info["alpha"] = info["source"] in res_name
    else:
        for info in edges_info:
            info["alpha"] = True

    connected_alpha_nodes = {
        n for info in edges_info if info["alpha"] for n in (info["u"], info["v"])
    }
    node_alpha = {
        name: (name in connected_alpha_nodes) if (prune or res_name is not None) else True
        for name in graph.vs["name"]
    }

    if focus:
        keep_edge_idx = [e.index for e, info in zip(graph.es, edges_info, strict=False) if info["alpha"]]
        draw_graph = graph.subgraph_edges(keep_edge_idx, delete_vertices=True)
        draw_edges_info = [info for info in edges_info if info["alpha"]]
    else:
        draw_graph = graph
        draw_edges_info = edges_info

    # Kamada-Kawai layout — closest available equivalent to R ariadne's "stress" layout
    layout = draw_graph.layout("kk")
    all_names = draw_graph.vs["name"]
    pos = {name: tuple(layout[i]) for i, name in enumerate(all_names)}

    # plt.close(fig) before returning removes it from pyplot's display queue,
    # preventing Jupyter from rendering it twice while still showing it as the
    # cell's return value via Jupyter's own repr mechanism.
    fig, ax = plt.subplots(figsize=figsize)

    # ── Edges ─────────────────────────────────────────────────────────────────
    for info in draw_edges_info:
        u, v, src = info["u"], info["v"], info["source"]
        if u not in pos or v not in pos:
            continue
        color = _C_EDGE_PATH if info["mark"] else _C_EDGE_BG
        lw = 2.0 if info["mark"] else 1.0
        line_alpha = 1.0 if info["alpha"] else _FADE_ALPHA
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot(
            [x0, x1], [y0, y1],
            color=color, lw=lw, alpha=line_alpha, zorder=1, solid_capstyle="round",
        )
        if info["mark"] and src:
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
    draw_names = [n for n in all_names if node_alpha.get(n, True)]
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
